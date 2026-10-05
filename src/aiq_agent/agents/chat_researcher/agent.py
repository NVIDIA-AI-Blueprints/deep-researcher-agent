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

"""
Chat Researcher Agent - Orchestrates intent classification, depth routing, and research.

This is the main orchestrator agent that coordinates the full research workflow:
1. Intent classification (meta vs research)
2. Depth routing (shallow vs deep)
3. Research execution
4. Optional escalation from shallow to deep
"""

import json
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from aiq_agent.agents.clarifier.models import ClarifierAgentState
from aiq_agent.agents.clarifier.models import ClarifierResult
from aiq_agent.agents.deep_researcher.models import DeepResearchAgentState
from aiq_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from aiq_agent.common import get_latest_user_query
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.logging_utils import log_content_metadata
from aiq_agent.common.logging_utils import log_identifier_ref
from aiq_agent.relay import run_agent

try:
    from aiq_api.auth.errors import AuthError as _AuthError
except ImportError:
    _AuthError = None  # type: ignore[assignment,misc]

try:
    from aiq_api.jobs.admission import JobAdmissionError as _JobAdmissionError
except ImportError:
    _JobAdmissionError = None  # type: ignore[assignment,misc]

from .models import RESEARCH_WORKFLOW_FAILURE_ERROR
from .models import ChatResearcherState
from .models import IntentResult
from .models import ShallowResult
from .models import WorkflowFailure
from .utils import trim_message_history

logger = logging.getLogger(__name__)


def _report_ref(job_id: str | None) -> str:
    """Redacted correlation ref for a report job UUID (an opaque bearer capability).

    The raw UUID must never enter logs; ``log_identifier_ref`` yields a stable
    ``sha256:<12hex>`` reference instead. ``active_report_job_id`` is optional, so
    fall back to ``"none"`` rather than hashing ``None``.
    """
    return log_identifier_ref(job_id) if job_id else "none"


# Async-job escalation signal. The chat WebSocket frame carries only a string ``content`` field,
# so the signal the UI consumes to start SSE streaming for a child job is encoded as a compact
# JSON object rather than a prose sentence -- this keeps detection robust to wording/punctuation
# changes. Mirrored by the UI parser in frontends/ui/src/features/chat/hooks/use-websocket-chat.ts.
_ESCALATION_KIND_DEEP_RESEARCH = "deep_research"
_ESCALATION_KIND_REPORT_EDIT = "report_edit"
_ESCALATION_KIND_DATA_SCIENCE = "data_science"


def _research_workflow_failure() -> WorkflowFailure:
    return WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)


def _job_escalation_message(kind: str, job_id: str) -> str:
    """Serialize an async-job escalation signal as a structured JSON payload."""
    return json.dumps({"type": "job_escalation", "kind": kind, "job_id": job_id})


class ChatResearcherAgent:
    """
    Orchestrates the full chat research workflow.

    Coordinates intent classification, depth routing, and research agents
    to produce research results based on user queries.

    The workflow:
    1. Classify intent (meta vs research)
    2. If meta → respond with meta chatter
    3. If research → route to shallow or deep based on complexity
    4. Optionally escalate from shallow to deep if results insufficient
    """

    def __init__(
        self,
        intent_classifier_fn: Callable[[str], Awaitable[str]],
        shallow_research_fn: Callable[[str], Awaitable[str]],
        deep_research_fn: Callable[[str], Awaitable[str]],
        clarifier_fn: Callable[
            [ClarifierAgentState | list[BaseMessage]],
            Awaitable[ClarifierResult],
        ]
        | None,
        *,
        enable_clarifier: bool = True,
        enable_escalation: bool = True,
        callbacks: list[BaseCallbackHandler] | None = None,
        max_history: int = 5,
        deep_research_job_submitter: Callable[[Any], Awaitable[str]] | None = None,
        report_ask_fn: Callable[[ChatResearcherState], Awaitable[str]] | None = None,
        hybrid_research_job_submitter: Callable[[ChatResearcherState], Awaitable[str]] | None = None,
        report_edit_job_submitter: Callable[[ChatResearcherState], Awaitable[str]] | None = None,
        report_edit_fn: Callable[[ChatResearcherState], Awaitable[str]] | None = None,
        report_seed_files_fn: Callable[[ChatResearcherState], Awaitable[dict[str, Any] | None]] | None = None,
        hybrid_research_fn: Callable[[ChatResearcherState], Awaitable[dict[str, Any]]] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        validate_deep_research_tools_fn: Callable[[list[str] | None], tuple[bool, str]] | None = None,
    ) -> None:
        """
        Initialize the chat researcher agent.

        Args:
            intent_classifier_fn: Combined orchestration (intent + meta response + depth in one node)
            shallow_research_fn: Function for shallow research
            deep_research_fn: Function for deep research
            clarifier_fn: Function for clarification
            enable_clarifier: Whether to enable clarification
            enable_escalation: Whether to escalate shallow to deep on low confidence
            callbacks: Optional list of callback handlers
            max_history: Maximum number of messages to keep in history
            deep_research_job_submitter: Optional function to submit deep research as async job
            report_ask_fn: Optional function to answer questions against the active parent report
            hybrid_research_job_submitter: Optional function to submit hybrid (data science)
                research as an async job. When omitted, hybrid runs inline in-request.
            report_edit_job_submitter: Optional function to submit report edit child jobs
            checkpointer: Optional checkpointer for persistent state (defaults to MemorySaver)
        """
        self.intent_classifier_fn = intent_classifier_fn
        self.shallow_research_fn = shallow_research_fn
        self.deep_research_fn = deep_research_fn
        self.clarifier_fn = clarifier_fn
        self.enable_clarifier = enable_clarifier
        self.enable_escalation = enable_escalation
        self.callbacks = callbacks or []
        self.max_history = max_history
        self.deep_research_job_submitter = deep_research_job_submitter
        self.report_ask_fn = report_ask_fn
        self.hybrid_research_job_submitter = hybrid_research_job_submitter
        self.report_edit_job_submitter = report_edit_job_submitter
        self.report_edit_fn = report_edit_fn
        self.report_seed_files_fn = report_seed_files_fn
        self.hybrid_research_fn = hybrid_research_fn
        self.checkpointer = checkpointer
        self.validate_deep_research_tools_fn = validate_deep_research_tools_fn

        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""

        async def intent_classifier_node(state: ChatResearcherState) -> dict[str, Any]:
            try:
                return await run_agent(
                    "intent_classifier",
                    lambda: self.intent_classifier_fn(state),
                    input_value={
                        "message_count": len(state.messages),
                        "data_source_count": len(state.data_sources or []),
                        "has_active_report": bool(state.active_report_job_id),
                    },
                )
            except Exception as error:
                logger.warning("Intent routing failed (error_type=%s)", type(error).__name__)
                return {
                    "user_intent": IntentResult(intent="meta", target="meta"),
                    "messages": [
                        AIMessage(
                            content=(
                                "There was an error routing your request. Please try again. "
                                "If the problem persists, please contact support."
                            )
                        )
                    ],
                    "workflow_outcome": _research_workflow_failure(),
                }

        async def hybrid_research_node(state: ChatResearcherState) -> dict[str, Any]:
            # A data-science run takes minutes, so submit it as a durable async job when a
            # submitter is wired. Running it inline would tie the answer to the open
            # connection and lose it on a refresh. Mirrors deep_research_node.
            if self.hybrid_research_job_submitter is not None:
                try:
                    job_id = await self.hybrid_research_job_submitter(state)
                except Exception as e:
                    if _JobAdmissionError and isinstance(e, _JobAdmissionError):
                        logger.warning(
                            "Hybrid research submission rejected (error_type=%s)",
                            type(e).__name__,
                        )
                        return {
                            "messages": [AIMessage(content=e.public_message)],
                            "workflow_outcome": _research_workflow_failure(),
                        }
                    if _AuthError and isinstance(e, _AuthError):
                        logger.warning(
                            "Auth required before hybrid research submit (error_type=%s detail_%s)",
                            type(e).__name__,
                            log_content_metadata(e),
                        )
                        return {
                            "messages": [AIMessage(content=str(e))],
                            "workflow_outcome": _research_workflow_failure(),
                        }
                    raise
                escalation = _job_escalation_message(_ESCALATION_KIND_DATA_SCIENCE, job_id)
                return {"messages": [AIMessage(content=escalation)]}

            try:
                if self.hybrid_research_fn is None:
                    raise RuntimeError("Hybrid research is not configured")
                return await self.hybrid_research_fn(state)
            except Exception as error:
                logger.error("Hybrid research failed (error_type=%s)", type(error).__name__)
                return {
                    "messages": [
                        AIMessage(content="I ran into an error while researching your question. Please try again.")
                    ],
                    "workflow_outcome": _research_workflow_failure(),
                }

        async def clarifier_node(state: ChatResearcherState) -> dict[str, Any]:
            original_query = get_latest_user_query(state.messages)

            # Validate deep research tools before proceeding to clarifier
            if self.validate_deep_research_tools_fn:
                is_valid, error_msg = self.validate_deep_research_tools_fn(state.data_sources)
                if not is_valid:
                    logger.error("Deep research tools validation failed (detail_%s)", log_content_metadata(error_msg))
                    return Command(
                        goto=END,
                        update={
                            "messages": [AIMessage(content=error_msg)],
                            "original_query": original_query,
                            "workflow_outcome": _research_workflow_failure(),
                        },
                    )

            uses_parent_report = bool(state.user_intent and state.user_intent.use_parent_report_context)
            if self.enable_clarifier and not state.skip_clarifier and not uses_parent_report:
                if self.clarifier_fn is None:
                    raise ValueError(
                        "enable_clarifier is True but clarifier_agent is not defined in config. "
                        "Either add clarifier_agent to functions or set enable_clarifier: false."
                    )
                trimmed_messages: list[BaseMessage] = trim_message_history(state.messages, self.max_history)
                available_docs = [doc.model_dump() for doc in (state.available_documents or [])]
                clarifier_state = ClarifierAgentState(
                    messages=trimmed_messages,
                    data_sources=state.data_sources,
                    available_documents=available_docs if available_docs else None,
                )
                result = await self.clarifier_fn(clarifier_state)

                return Command(
                    goto="deep_research",
                    update={
                        "clarifier_result": result.clarifier_log,
                        "original_query": original_query,
                    },
                )
            return Command(goto="deep_research", update={"original_query": original_query})

        async def shallow_research_node(state: ChatResearcherState) -> dict[str, Any]:
            trimmed_messages: list[BaseMessage] = trim_message_history(state.messages, self.max_history)

            logger.debug(
                "shallow_research_node: available_document_count=%d",
                len(state.available_documents or []),
            )

            try:
                shallow_state = ShallowResearchAgentState(
                    messages=trimmed_messages,
                    data_sources=state.data_sources,
                    available_documents=state.available_documents,
                )
                result = await self.shallow_research_fn(shallow_state)
            except EmptySourceRegistryError as exc:
                logger.warning("Shallow research produced no verifiable sources")
                err_msg = exc.public_response
                # confidence="high" reflects certainty that an error occurred and that the error
                # message is the correct response — not uncertainty about the answer quality.
                # escalate_to_deep=False because retrying deep research will not resolve a
                # source registry or transient failure; the user should rephrase and retry.
                return {
                    "messages": [AIMessage(content=err_msg)],
                    "workflow_outcome": _research_workflow_failure(),
                    "shallow_result": ShallowResult(
                        answer=err_msg,
                        confidence="high",
                        escalate_to_deep=False,
                    ),
                }
            except Exception as e:
                if _AuthError and isinstance(e, _AuthError):
                    logger.warning(
                        "Auth error in shallow research (error_type=%s detail_%s)",
                        type(e).__name__,
                        log_content_metadata(e),
                    )
                    err_msg = str(e)
                    return {
                        "messages": [AIMessage(content=err_msg)],
                        "workflow_outcome": _research_workflow_failure(),
                        "shallow_result": ShallowResult(
                            answer=err_msg,
                            confidence="high",
                            escalate_to_deep=False,
                        ),
                    }
                logger.error(
                    "Error in shallow research (error_type=%s detail_%s)",
                    type(e).__name__,
                    log_content_metadata(e),
                )
                err_msg = "An error occurred while researching your question. Please try again."
                # Same rationale as EmptySourceRegistryError: the system is certain an error
                # occurred; escalating to deep research will not resolve an unexpected exception.
                return {
                    "messages": [AIMessage(content=err_msg)],
                    "workflow_outcome": _research_workflow_failure(),
                    "shallow_result": ShallowResult(
                        answer=err_msg,
                        confidence="high",
                        escalate_to_deep=False,
                    ),
                }

            if not result.messages:
                logger.error("Shallow research agent returned no messages")
                return {
                    "shallow_result": ShallowResult(
                        answer="An error occurred during shallow research.",
                        confidence="low",
                        escalate_to_deep=True,
                        escalation_reason="Shallow research encountered an error",
                    )
                }
            new_messages = result.messages[len(trimmed_messages) :]
            final_ai_message = next(
                (m for m in reversed(new_messages) if isinstance(m, AIMessage) and not m.tool_calls),
                None,
            )
            if final_ai_message:
                return {"messages": [final_ai_message], "shallow_result": None}
            if new_messages:
                return {"messages": [new_messages[-1]], "shallow_result": None}
            return {"messages": [], "shallow_result": None}

        async def deep_research_node(state: ChatResearcherState) -> dict[str, Any]:
            if self.deep_research_job_submitter is not None:
                try:
                    job_id = await self.deep_research_job_submitter(state)
                except Exception as e:
                    if _JobAdmissionError and isinstance(e, _JobAdmissionError):
                        logger.warning(
                            "Deep research submission rejected (error_type=%s)",
                            type(e).__name__,
                        )
                        return {
                            "messages": [AIMessage(content=e.public_message)],
                            "workflow_outcome": _research_workflow_failure(),
                        }
                    # Surface auth failures to the user verbatim instead of failing the
                    # turn — submit_agent_job raises McpAuthRequiredError (an AuthError)
                    # before enqueue when a selected source isn't connected.
                    if _AuthError and isinstance(e, _AuthError):
                        logger.warning(
                            "Auth required before deep research submit (error_type=%s detail_%s)",
                            type(e).__name__,
                            log_content_metadata(e),
                        )
                        return {
                            "messages": [AIMessage(content=str(e))],
                            "workflow_outcome": _research_workflow_failure(),
                        }
                    raise
                escalation = _job_escalation_message(_ESCALATION_KIND_DEEP_RESEARCH, job_id)
                return {"messages": [AIMessage(content=escalation)]}

            research_query = state.original_query or get_latest_user_query(state.messages)
            seed_files: dict[str, Any] = {}
            if (
                self.report_seed_files_fn is not None
                and state.user_intent is not None
                and getattr(state.user_intent, "use_parent_report_context", False)
                and (state.active_report_job_id or state.last_report_markdown)
            ):
                # Delta research: seed the parent report into the deep agent's virtual filesystem so it
                # reuses it and researches only the requested delta. Best-effort -- a seed failure falls
                # back to fresh research rather than aborting the turn.
                try:
                    seed_files = await self.report_seed_files_fn(state) or {}
                except Exception as e:
                    logger.warning(
                        "Parent report seed unavailable for delta (error_type=%s); running fresh research",
                        type(e).__name__,
                    )
                    seed_files = {}
            logger.info(
                "Inline deep research: use_parent_report_context=%s has_report=%s seeded_files=%d",
                bool(getattr(state.user_intent, "use_parent_report_context", False)) if state.user_intent else False,
                bool(state.active_report_job_id or state.last_report_markdown),
                len(seed_files),
            )
            # Mirror the async job: feed the deep researcher a clean query (plus the approved plan and any
            # seeded parent-report files), NOT the prior chat history. Forwarding accumulated report
            # messages bloats the writer's context and can make it fail to emit a final report.
            deep_state = DeepResearchAgentState(
                messages=[HumanMessage(content=research_query)],
                data_sources=state.data_sources,
                clarifier_result=state.clarifier_result,
                available_documents=state.available_documents,
                user_info=state.user_info,
                files=seed_files,
            )
            try:
                result = await self.deep_research_fn(deep_state)
            except EmptySourceRegistryError as exc:
                logger.warning("Deep research produced no verifiable sources")
                err_msg = exc.public_response
                return {
                    "messages": [AIMessage(content=err_msg)],
                    "workflow_outcome": _research_workflow_failure(),
                }
            except Exception as e:
                if _AuthError and isinstance(e, _AuthError):
                    logger.warning(
                        "Auth error in deep research (error_type=%s detail_%s)",
                        type(e).__name__,
                        log_content_metadata(e),
                    )
                    return {
                        "messages": [AIMessage(content=str(e))],
                        "workflow_outcome": _research_workflow_failure(),
                    }
                # Inline (synchronous CLI) path: a raised exception would crash the whole CLI turn.
                # Degrade to a chat message instead (the error is logged for debugging).
                logger.error(
                    "Inline deep research failed (error_type=%s detail_%s)",
                    type(e).__name__,
                    log_content_metadata(e),
                )
                return {
                    "messages": [
                        AIMessage(content="I ran into an error while producing that report. Please try again.")
                    ],
                    "workflow_outcome": _research_workflow_failure(),
                }
            if not result.messages:
                error_message = "An error occurred during deep research."
                logger.error(error_message)
                final_message = AIMessage(content=error_message)
                return {
                    "messages": [final_message],
                    "workflow_outcome": _research_workflow_failure(),
                }
            else:
                report_message = result.messages[-1]
                report_md = report_message.content
                # Capture the inline report so follow-up turns in this (synchronous) session can
                # reference it without an async job. Checkpointed via the keep-if-set reducer.
                return {
                    "messages": [report_message],
                    "last_report_markdown": report_md if isinstance(report_md, str) else str(report_md),
                }

        async def report_ask_node(state: ChatResearcherState) -> dict[str, Any]:
            if self.report_ask_fn is None:
                return {
                    "messages": [AIMessage(content="Report follow-up is not available in this workflow.")],
                    "workflow_outcome": _research_workflow_failure(),
                }
            try:
                answer = await self.report_ask_fn(state)
            except TimeoutError:
                logger.warning("Report ask timed out (report_ref=%s)", _report_ref(state.active_report_job_id))
                return {
                    "messages": [AIMessage(content="The report service took too long to respond. Please try again.")],
                    "workflow_outcome": _research_workflow_failure(),
                }
            except Exception as e:
                # The node has no HTTP scope, so a raised exception would surface as an
                # opaque workflow error / empty completion. Degrade to a chat message.
                logger.warning(
                    "Report ask failed (report_ref=%s, error_type=%s)",
                    _report_ref(state.active_report_job_id),
                    type(e).__name__,
                )
                return {
                    "messages": [
                        AIMessage(content="I couldn't access that report to answer your question. Please try again.")
                    ],
                    "workflow_outcome": _research_workflow_failure(),
                }
            return {"messages": [AIMessage(content=answer)]}

        async def report_edit_node(state: ChatResearcherState) -> dict[str, Any]:
            # Async path (server): submit a report_rewriter child job; the UI streams its result.
            if self.report_edit_job_submitter is not None:
                try:
                    job_id = await self.report_edit_job_submitter(state)
                except Exception as e:
                    logger.warning(
                        "Report edit submission failed (report_ref=%s, error_type=%s)",
                        _report_ref(state.active_report_job_id),
                        type(e).__name__,
                    )
                    return {
                        "messages": [AIMessage(content="I couldn't start the report edit. Please try again.")],
                        "workflow_outcome": _research_workflow_failure(),
                    }
                escalation = _job_escalation_message(_ESCALATION_KIND_REPORT_EDIT, job_id)
                return {"messages": [AIMessage(content=escalation)]}
            # Inline path (synchronous CLI): rewrite the in-session report directly -- no job or
            # scheduler. The revised report becomes the reply and replaces last_report_markdown so
            # subsequent follow-ups operate on the edited copy.
            if self.report_edit_fn is not None:
                try:
                    revised = await self.report_edit_fn(state)
                except Exception as e:
                    logger.warning("Inline report edit failed (error_type=%s)", type(e).__name__)
                    return {
                        "messages": [AIMessage(content="I couldn't edit the report. Please try again.")],
                        "workflow_outcome": _research_workflow_failure(),
                    }
                return {"messages": [AIMessage(content=revised)], "last_report_markdown": revised}
            return {
                "messages": [AIMessage(content="Report edit is not available in this workflow.")],
                "workflow_outcome": _research_workflow_failure(),
            }

        def route_after_orchestration(state: ChatResearcherState) -> str:
            """From combined orchestration: meta -> END (response already in messages), else by depth."""
            if state.user_intent and state.user_intent.intent == "meta":
                return "END"
            # A report is available either as an async job (UI/server) or produced inline in this
            # session (synchronous CLI). Either lets the router serve report follow-up.
            has_report = bool(state.active_report_job_id or state.last_report_markdown)
            if has_report and state.user_intent and state.user_intent.target == "report":
                if state.user_intent.report_action == "edit":
                    return "report_edit"
                return "report_ask"
            if state.user_intent and state.user_intent.target == "hybrid_research":
                return "hybrid_research"
            if state.depth_decision and state.depth_decision.decision == "deep":
                return "clarifier"
            return "shallow_research"

        def should_escalate(state: ChatResearcherState) -> str:
            if not self.enable_escalation:
                return END

            # Respect explicit escalation decision from shallow research.
            # Successful shallow paths set shallow_result=None so this guard
            # only fires when shallow explicitly set escalate_to_deep.
            if state.shallow_result is not None:
                if state.shallow_result.escalate_to_deep:
                    return "deep_research"
                return END

            messages = state.messages
            if not messages:
                return END

            last_ai_content = None
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    last_ai_content = m.content if hasattr(m, "content") else str(m)
                    break
            if not last_ai_content:
                return END

            last_content = last_ai_content if isinstance(last_ai_content, str) else str(last_ai_content)
            if not last_content.strip():
                return "deep_research"

            tail = last_content[-800:].lower() if len(last_content) > 800 else last_content.lower()
            escalation_keywords = ["i don't have enough information", "unable to find", "need more research"]
            if any(kw in tail for kw in escalation_keywords):
                return "deep_research"

            return END

        graph = StateGraph(ChatResearcherState)

        graph.add_node("intent_classifier", intent_classifier_node)
        graph.add_node("shallow_research", shallow_research_node)
        graph.add_node("clarifier", clarifier_node)
        graph.add_node("deep_research", deep_research_node)
        graph.add_node("report_ask", report_ask_node)
        graph.add_node("report_edit", report_edit_node)
        graph.add_node("hybrid_research", hybrid_research_node)

        graph.set_entry_point("intent_classifier")

        graph.add_conditional_edges(
            "intent_classifier",
            route_after_orchestration,
            {
                "END": END,
                "clarifier": "clarifier",
                "shallow_research": "shallow_research",
                "report_ask": "report_ask",
                "report_edit": "report_edit",
                "hybrid_research": "hybrid_research",
            },
        )

        graph.add_conditional_edges(
            "shallow_research",
            should_escalate,
            {
                "deep_research": "clarifier",
                END: END,
            },
        )

        graph.add_edge("deep_research", END)
        graph.add_edge("report_ask", END)
        graph.add_edge("report_edit", END)
        graph.add_edge("hybrid_research", END)

        return graph.compile(checkpointer=self.checkpointer)

    async def run(
        self, state: ChatResearcherState | dict[str, Any], thread_id: str | None = None
    ) -> ChatResearcherState:
        """
        Execute the chat researcher workflow.

        Args:
            state: ChatResearcherState or dict with new messages to add.
            thread_id: Thread ID for the conversation (used for checkpointing).
        Returns:
            Updated state with response in messages.
        """
        graph_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        logger.info("ChatResearcherAgent: Starting workflow")

        if isinstance(state, dict):
            input_state = {
                **state,
                "database_name": state.get("database_name"),
                "catalog_context": None,
                "catalog_request_id": None,
            }
            messages = state.get("messages", [])
        else:
            input_state = {
                "messages": state.messages,
                "user_info": state.user_info,
                "data_sources": state.data_sources,
                "database_name": state.database_name,
                "available_documents": state.available_documents,
                "shallow_result": None,  # reset at turn boundary to avoid stale checkpoint state
                "workflow_outcome": None,
                "skip_clarifier": state.skip_clarifier,
                "active_report_job_id": state.active_report_job_id,
                "catalog_context": None,
                "catalog_request_id": None,
                # Pass through; the keep-if-set reducer preserves a prior in-session report when
                # this turn supplies None, so report follow-up works across turns without a job.
                "last_report_markdown": state.last_report_markdown,
            }
            messages = state.messages

        if messages:
            query = messages[-1].content
            logger.info("Query: %s", log_content_metadata(query or ""))

        async def _invoke_graph() -> dict[str, Any]:
            effective_config = dict(graph_config or {})
            return await self._graph.ainvoke(input_state, config=effective_config)

        input_data_sources = (
            input_state.get("data_sources") if isinstance(input_state, dict) else input_state.data_sources
        )
        relay_input_metadata = {
            "message_count": len(messages),
            "data_source_count": len(input_data_sources or []),
            "has_active_report": bool(
                input_state.get("active_report_job_id")
                if isinstance(input_state, dict)
                else input_state.active_report_job_id
            ),
        }
        result = await run_agent(
            "chat_deepresearcher_agent",
            _invoke_graph,
            session_id=thread_id,
            input_value=relay_input_metadata,
        )

        logger.info("ChatResearcherAgent: Workflow complete")

        return result

    @property
    def graph(self) -> CompiledStateGraph:
        """Get the compiled LangGraph for direct access."""
        return self._graph
