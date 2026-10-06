# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Intent classifier agent for classifying meta vs research queries."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.common import extract_json
from deep_researcher_agent.common import load_prompt
from deep_researcher_agent.common import render_prompt_template
from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.relay import ainvoke_with_relay

from ..models import RESEARCH_WORKFLOW_FAILURE_ERROR
from ..models import ChatResearcherState
from ..models import DepthDecision
from ..models import IntentResult
from ..models import WorkflowFailure
from ..preclassification import get_preclassified_depth

logger = logging.getLogger(__name__)


_LLM_UNAVAILABLE_MESSAGE = (
    "I'm unable to reach the model service right now. "
    "Please check your LLM API key and that the configured model is available for your account."
)
_LLM_TIMEOUT_MESSAGE = "The model service took too long to respond and the request timed out. "
_LLM_ERROR_MESSAGE = "We couldn't process your request due to a temporary error. Please try again."
_REPAIR_TIMEOUT_SECONDS = 15
_ROUTE_REPORT_ASK = "report_ask"
_ROUTE_REPORT_COSMETIC_EDIT = "report_cosmetic_edit"
_ROUTE_REPORT_DELTA_RESEARCH = "report_delta_research"
_ROUTE_STANDALONE_RESEARCH = "standalone_research"
_ROUTE_META = "meta"


def _failure_update(message: str) -> dict[str, Any]:
    """Return the existing terminal meta route with an explicit failed outcome."""
    return {
        "user_intent": IntentResult(intent="meta", target="meta", raw=None),
        "messages": [AIMessage(content=message)],
        "workflow_outcome": WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR),
    }


def _is_llm_api_unavailable(err: BaseException) -> bool:
    """True if the error is from the LLM API being unreachable (e.g. 404, function not found)."""
    msg = str(err).strip()
    return (
        "[404]" in msg
        or "not found for account" in msg.lower()
        or (msg.lower().startswith("not found") and "account" in msg.lower())
    )


def _is_timeout_error(err: BaseException) -> bool:
    """True if the error is from a timeout (asyncio.wait_for or gateway 504)."""
    if isinstance(err, TimeoutError | asyncio.TimeoutError):
        return True
    msg = str(err).strip().lower()
    return "504" in msg or "gateway time-out" in msg or "gateway timeout" in msg


class IntentClassifier:
    def __init__(
        self,
        llm: BaseChatModel,
        tools_info: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        max_history: int = 20,
        llm_timeout: float = 90,
    ) -> None:
        self.llm = llm
        self.tools_info = tools_info or []
        self.prompt = prompt or self._load_default_prompt()
        self.callbacks = callbacks or []
        self.max_history = max_history
        self.llm_timeout = llm_timeout

    def _load_default_prompt(self) -> str:
        try:
            return load_prompt(Path(__file__).parent.parent / "prompts", "intent_classification.j2")
        except Exception:
            return (
                "/no_think\n\n"
                "You are an Orchestrator. Classify intent as 'meta' or 'research'.\n"
                "If meta, provide 'meta_response'. If research, provide 'research_depth'.\n"
                "Respond ONLY with JSON."
            )

    async def run(self, state: ChatResearcherState) -> dict[str, Any]:
        """Run the intent classifier node."""
        messages = state.messages
        if not messages:
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="No query"),
            }

        # Reuse a caller-supplied depth when present (e.g. the MCP JobManager, which
        # already classified this query once to persist depth and pick a poll cadence).
        # This keeps the persisted decision and the executed route identical and skips a
        # redundant intent-LLM call. Only research depth is threaded; meta short-circuits
        # before the caller ever reaches this path, so target defaults to "new_research".
        preset_depth = get_preclassified_depth()
        if preset_depth is not None:
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision=preset_depth, raw_reasoning="preclassified by caller"),
            }

        user_info = state.user_info or {}
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_content = messages[-1].content
        query = last_content if isinstance(last_content, str) else str(last_content or "")

        system_content = render_prompt_template(
            self.prompt,
            query=query,
            current_datetime=current_datetime,
            user_info=user_info,
            tools=self.tools_info,
            active_report_available=bool(state.active_report_job_id or state.last_report_markdown),
        )
        # Keep the router isolated from prior assistant report bodies. The prompt already contains
        # the latest query and report availability, which is the bounded context needed here.
        messages: list[BaseMessage] = [SystemMessage(content=system_content)]

        try:
            # This model call crosses the NAT function boundary, so it does not
            # inherit the parent graph's RunnableConfig automatically.
            response = await asyncio.wait_for(
                ainvoke_with_relay(self.llm, messages, callbacks=self.callbacks),
                timeout=self.llm_timeout,
            )

            response_text = (response.content or "").strip()
            parsed = extract_json(response_text)
            if not parsed or not isinstance(parsed, dict):
                parsed = await self._repair_json_response(
                    system_content=system_content,
                    invalid_response=response_text,
                    callbacks=self.callbacks,
                )

            if not parsed or not isinstance(parsed, dict):
                return _failure_update(_LLM_ERROR_MESSAGE)

            raw_intent = (parsed.get("intent") or "research").strip().lower()
            route = _normalize_route(parsed.get("route"))
            if route == _ROUTE_META:
                intent = "meta"
            elif route is not None:
                intent = "research"
            else:
                intent = raw_intent if raw_intent in ("meta", "research") else "research"
            meta_response = parsed.get("meta_response")
            research_depth = (parsed.get("research_depth") or "shallow").strip().lower()
            depth_reasoning = parsed.get("route_reasoning") or parsed.get("depth_reasoning") or ""
            active_report = bool(state.active_report_job_id or state.last_report_markdown)

            if intent == "meta":
                target = "meta"
                report_action = None
                use_parent_report_context = False
            elif route is not None:
                target, report_action, use_parent_report_context, research_depth, depth_reasoning = _route_to_fields(
                    route=route,
                    active_report=active_report,
                    research_depth=research_depth,
                    depth_reasoning=str(depth_reasoning),
                )
            else:
                target = "new_research"
                report_action = None
                use_parent_report_context = False

            update: dict[str, Any] = {
                "user_intent": IntentResult(
                    intent=intent,
                    target=target,
                    report_action=report_action,
                    use_parent_report_context=use_parent_report_context,
                    raw=parsed,
                ),
            }

            if intent == "meta":
                meta_text = (
                    meta_response if isinstance(meta_response, str) and meta_response.strip() else "I'm here to help."
                )
                update["messages"] = [AIMessage(content=meta_text)]
            elif target != "report":
                update["depth_decision"] = DepthDecision(
                    decision=research_depth if research_depth in ("shallow", "deep") else "shallow",
                    raw_reasoning=str(depth_reasoning),
                )

            return update

        except TimeoutError:
            logger.warning(
                "LLM call timed out after %s seconds.",
                self.llm_timeout,
            )
            return _failure_update(_LLM_TIMEOUT_MESSAGE)
        except Exception as e:
            if _is_llm_api_unavailable(e):
                logger.error(
                    "LLM API unreachable (e.g. 404 model/function not found) (error_type=%s detail_%s)",
                    type(e).__name__,
                    log_content_metadata(e),
                )
                return _failure_update(_LLM_UNAVAILABLE_MESSAGE)
            if _is_timeout_error(e):
                logger.error(
                    "LLM call failed with timeout (e.g. 504 Gateway Time-out) (error_type=%s detail_%s)",
                    type(e).__name__,
                    log_content_metadata(e),
                )
                return _failure_update(_LLM_TIMEOUT_MESSAGE)
            logger.error(
                "Error in orchestration (error_type=%s detail_%s)",
                type(e).__name__,
                log_content_metadata(e),
            )
            return _failure_update(_LLM_ERROR_MESSAGE)

    async def _repair_json_response(
        self,
        *,
        system_content: str,
        invalid_response: str,
        callbacks: list[Any],
    ) -> dict[str, Any] | None:
        repair_prompt = (
            f"{system_content}\n\n"
            "The previous classifier response was invalid because it was not one valid JSON object.\n"
            "Return only one valid JSON object matching the schema above. Do not include markdown, prose, "
            "analysis, code fences, or a rewritten report.\n\n"
            f"Invalid response:\n{invalid_response[:4000]}"
        )
        try:
            response = await asyncio.wait_for(
                ainvoke_with_relay(
                    self.llm,
                    [SystemMessage(content=repair_prompt)],
                    callbacks=callbacks,
                ),
                timeout=min(self.llm_timeout, _REPAIR_TIMEOUT_SECONDS),
            )
        except TimeoutError:
            logger.warning("Intent classifier JSON repair timed out.")
            return None
        except Exception as e:
            logger.warning(
                "Intent classifier JSON repair failed (error_type=%s detail_%s)",
                type(e).__name__,
                log_content_metadata(e),
            )
            return None

        repaired = extract_json((response.content or "").strip())
        return repaired if isinstance(repaired, dict) else None


def _normalize_route(raw_route: Any) -> str | None:
    route = raw_route.strip().lower() if isinstance(raw_route, str) else None
    if route in (
        _ROUTE_REPORT_ASK,
        _ROUTE_REPORT_COSMETIC_EDIT,
        _ROUTE_REPORT_DELTA_RESEARCH,
        _ROUTE_STANDALONE_RESEARCH,
        _ROUTE_META,
    ):
        return route
    return None


def _route_to_fields(
    *,
    route: str,
    active_report: bool,
    research_depth: str,
    depth_reasoning: str,
) -> tuple[str, str | None, bool, str, str]:
    """Map the LLM-owned semantic route onto the existing workflow fields."""
    if route == _ROUTE_REPORT_ASK:
        if active_report:
            return "report", "ask", False, research_depth, depth_reasoning
        return "new_research", None, False, research_depth, depth_reasoning

    if route == _ROUTE_REPORT_COSMETIC_EDIT:
        if active_report:
            return "report", "edit", False, research_depth, depth_reasoning
        return "new_research", None, False, research_depth, depth_reasoning

    if route == _ROUTE_REPORT_DELTA_RESEARCH:
        reasoning = depth_reasoning or "Requires fresh evidence against the active report."
        return "new_research", None, active_report, "deep", reasoning

    if route == _ROUTE_STANDALONE_RESEARCH:
        return "new_research", None, False, research_depth, depth_reasoning

    raise ValueError(f"Unsupported research route: {route}")
