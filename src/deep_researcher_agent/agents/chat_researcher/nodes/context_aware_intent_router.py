# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-aware entry classification with code-owned catalog routing."""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any
from typing import Literal
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware import wrap_model_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import model_validator

from deep_researcher_agent.common import get_latest_user_query
from deep_researcher_agent.common import render_prompt_template

from ..models import CatalogRoutingResponse
from ..models import ChatResearcherState
from ..models import DepthDecision
from ..models import IntentResult
from ..preclassification import get_preclassified_depth
from .intent_classifier import _route_to_fields

Interaction = Literal[
    "meta",
    "report_ask",
    "report_cosmetic_edit",
    "report_delta_research",
    "new_research",
]

logger = logging.getLogger(__name__)

_RETRYABLE_CLASSIFIER_STATUS_CODES = (429, 500, 502, 503, 504)
_DEFAULT_CLASSIFIER_MAX_ATTEMPTS = 2
_DEFAULT_CLASSIFIER_RETRY_DELAY_SECONDS = 0.5


class RoutingProtocolError(RuntimeError):
    """The typed classification or catalog response violated the routing contract."""


class EntryClassification(BaseModel):
    """Single model-owned classification produced before catalog execution."""

    model_config = ConfigDict(extra="forbid")

    interaction: Interaction
    catalog_action: Literal["skip", "search"] | None
    catalog_question: str | None
    research_depth: Literal["shallow", "deep"] | None
    meta_response: str | None
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interaction_fields(self) -> "EntryClassification":
        if self.interaction == "new_research":
            if self.catalog_action is None:
                raise ValueError("New research requires a catalog action")
            if self.research_depth is None:
                raise ValueError("New research requires a fallback depth")
            if self.meta_response is not None:
                raise ValueError("New research cannot return a meta response")
            if self.catalog_action == "skip" and self.catalog_question is not None:
                raise ValueError("Catalog questions are only valid for catalog search")
            return self

        if self.catalog_action is not None or self.catalog_question is not None:
            raise ValueError("Meta and report interactions cannot request catalog search")
        if self.interaction == "meta":
            if not (self.meta_response and self.meta_response.strip()):
                raise ValueError("Meta interactions require a response")
            if self.research_depth is not None:
                raise ValueError("Meta interactions do not use research depth")
            return self

        if self.meta_response is not None:
            raise ValueError("Report interactions cannot return a meta response")
        if self.interaction == "report_delta_research":
            if self.research_depth != "deep":
                raise ValueError("Report delta research requires deep depth")
        elif self.research_depth is not None:
            raise ValueError("Report ask and cosmetic edit do not use research depth")
        return self


class RouterRunContext(TypedDict):
    active_report_available: bool
    catalog_enabled: bool
    current_datetime: str
    database_scope_provided: bool
    query: str
    user_info: dict[str, str]


def _prompt_middleware(prompt: str):
    @wrap_model_call
    async def render_router_prompt(
        request: ModelRequest[RouterRunContext],
        handler: Callable[[ModelRequest[RouterRunContext]], Any],
    ) -> ModelResponse:
        system_prompt = render_prompt_template(prompt, **request.runtime.context)
        return await handler(request.override(system_message=SystemMessage(content=system_prompt)))

    return render_router_prompt


class ContextAwareIntentRouter:
    """Classify once, call catalog in Python when selected, then route deterministically."""

    def __init__(
        self,
        llm: BaseChatModel,
        catalog_tool: BaseTool,
        prompt: str,
        *,
        catalog_source_id: str = "gsf",
        max_catalog_results: int = 10,
        catalog_confidence_threshold: float = 0.6,
        catalog_max_distance: float = 0.75,
        callbacks: list[BaseCallbackHandler] | None = None,
        llm_timeout: float = 90,
        classifier_max_attempts: int = _DEFAULT_CLASSIFIER_MAX_ATTEMPTS,
        classifier_retry_delay_seconds: float = _DEFAULT_CLASSIFIER_RETRY_DELAY_SECONDS,
    ) -> None:
        self.catalog_tool = catalog_tool
        self.catalog_input_schema = catalog_tool.get_input_schema()
        self.catalog_source_id = catalog_source_id
        self.max_catalog_results = max_catalog_results
        self.catalog_confidence_threshold = catalog_confidence_threshold
        self.catalog_max_distance = catalog_max_distance
        self.callbacks = callbacks or []
        self.llm_timeout = llm_timeout
        self.classifier_max_attempts = classifier_max_attempts
        self.classifier_retry_delay_seconds = classifier_retry_delay_seconds
        self.classifier = create_agent(
            model=llm,
            tools=[],
            response_format=ToolStrategy(EntryClassification),
            context_schema=RouterRunContext,
            middleware=[_prompt_middleware(prompt)],
        )

    async def run(self, state: ChatResearcherState) -> dict[str, Any]:
        if not state.messages:
            raise RoutingProtocolError("Routing requires a user message")

        query = get_latest_user_query(state.messages)
        catalog_enabled = state.data_sources is None or self.catalog_source_id in state.data_sources
        result = await self._classify(query=query, state=state, catalog_enabled=catalog_enabled)
        classification = EntryClassification.model_validate(result.get("structured_response"))
        classification = _normalize_report_classification(classification, state)

        if classification.interaction == "meta":
            return {
                "user_intent": IntentResult(intent="meta", target="meta"),
                "messages": [AIMessage(content=classification.meta_response or "I'm here to help.")],
            }
        if classification.interaction != "new_research":
            return _report_state_update(classification, state)

        depth = get_preclassified_depth() or classification.research_depth or "shallow"
        database_scoped = state.database_name is not None
        if database_scoped and not catalog_enabled:
            raise RoutingProtocolError("Database-scoped research requires the catalog source, but it is disabled")
        if not database_scoped and (classification.catalog_action == "skip" or not catalog_enabled):
            return _standalone_state_update(depth, classification.reasoning)

        catalog_question = _validated_catalog_question(query, classification.catalog_question)
        catalog = await self._search_catalog(catalog_question, database_name=state.database_name)
        if len(catalog.candidates) > self.max_catalog_results:
            raise RoutingProtocolError("Catalog tool exceeded the configured result limit")
        if database_scoped:
            return _hybrid_state_update(catalog)
        supports_hybrid = catalog.coverage >= self.catalog_confidence_threshold
        if supports_hybrid and not catalog.candidates:
            raise RoutingProtocolError("Catalog confidence met the hybrid threshold without candidates")
        if not supports_hybrid:
            return _standalone_state_update(depth, classification.reasoning)
        return _hybrid_state_update(catalog)

    async def _classify(
        self,
        *,
        query: str,
        state: ChatResearcherState,
        catalog_enabled: bool,
    ) -> dict[str, Any]:
        """Invoke the classifier, retrying one transient provider failure."""
        context: RouterRunContext = {
            "active_report_available": bool(state.active_report_job_id or state.last_report_markdown),
            "catalog_enabled": catalog_enabled,
            "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "database_scope_provided": state.database_name is not None,
            "query": query,
            "user_info": _router_user_info(state.user_info),
        }
        config = {"callbacks": self.callbacks} if self.callbacks else None
        async with asyncio.timeout(self.llm_timeout):
            for attempt in range(1, self.classifier_max_attempts + 1):
                try:
                    return await self.classifier.ainvoke(
                        {"messages": [HumanMessage(content=query)]},
                        config=config,
                        context=context,
                    )
                except Exception as error:
                    if attempt >= self.classifier_max_attempts or not _is_retryable_classifier_error(error):
                        raise
                    logger.warning(
                        "Entry classifier provider call failed transiently; retrying (attempt=%d/%d, error_type=%s)",
                        attempt,
                        self.classifier_max_attempts,
                        type(error).__name__,
                    )
                    await asyncio.sleep(self.classifier_retry_delay_seconds)
        raise AssertionError("Classifier retry loop exited without a result")

    async def _search_catalog(self, question: str, *, database_name: str | None) -> CatalogRoutingResponse:
        try:
            request = self.catalog_input_schema.model_validate(
                {
                    "question": question,
                    "database_name": database_name,
                    "max_results": self.max_catalog_results,
                    "max_distance": self.catalog_max_distance,
                }
            )
        except ValidationError as error:
            raise RoutingProtocolError("Code-owned catalog arguments failed validation") from error
        payload = request.model_dump(exclude_none=True)
        config = {"callbacks": self.callbacks} if self.callbacks else None
        raw = await self.catalog_tool.ainvoke(payload, config=config)
        return _catalog_response(raw)


def _router_user_info(user_info: dict[str, Any] | None) -> dict[str, str]:
    if not user_info or not isinstance(user_info.get("name"), str):
        return {}
    return {"name": user_info["name"]}


def _normalize_report_classification(
    classification: EntryClassification,
    state: ChatResearcherState,
) -> EntryClassification:
    if state.active_report_job_id or state.last_report_markdown or not classification.interaction.startswith("report_"):
        return classification
    depth = "deep" if classification.interaction == "report_delta_research" else "shallow"
    return EntryClassification(
        interaction="new_research",
        catalog_action="skip",
        catalog_question=None,
        research_depth=depth,
        meta_response=None,
        reasoning=classification.reasoning,
    )


def _validated_catalog_question(query: str, proposed: str | None) -> str:
    """Use only a safe contiguous verbatim span; otherwise fall back to the full query."""
    if proposed is None:
        return query
    candidate = proposed.strip()
    if not candidate or candidate not in query:
        return query
    return candidate


def _is_retryable_classifier_error(error: Exception) -> bool:
    """Recognize sanitized transient HTTP failures without inspecting prompts."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in _RETRYABLE_CLASSIFIER_STATUS_CODES:
        return True

    message = str(error)
    for code in _RETRYABLE_CLASSIFIER_STATUS_CODES:
        markers = (
            f"[{code}]",
            f"'status': {code}",
            f'"status": {code}',
            f"'code': {code}",
            f'"code": {code}',
        )
        if any(marker in message for marker in markers):
            return True
    return False


def _catalog_response(raw: Any) -> CatalogRoutingResponse:
    if isinstance(raw, ToolMessage):
        if raw.status == "error":
            raise RoutingProtocolError("Catalog tool failed")
        raw = raw.artifact if raw.artifact is not None else raw.content
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RoutingProtocolError("Catalog tool returned invalid JSON") from error
    if isinstance(raw, dict) and raw.get("status") == "error":
        code = raw.get("code")
        suffix = f" ({code})" if isinstance(code, str) and code else ""
        raise RoutingProtocolError(f"Catalog tool failed{suffix}")
    try:
        return CatalogRoutingResponse.model_validate(raw)
    except ValidationError as error:
        raise RoutingProtocolError("Catalog tool returned an invalid response") from error


def _standalone_state_update(depth: Literal["shallow", "deep"], reasoning: str) -> dict[str, Any]:
    return {
        "user_intent": IntentResult(intent="research", target="new_research"),
        "depth_decision": DepthDecision(decision=depth, raw_reasoning=reasoning),
    }


def _hybrid_state_update(catalog: CatalogRoutingResponse) -> dict[str, Any]:
    return {
        "user_intent": IntentResult(intent="research", target="hybrid_research"),
        "catalog_context": catalog,
        "catalog_request_id": catalog.request_id,
    }


def _report_state_update(classification: EntryClassification, state: ChatResearcherState) -> dict[str, Any]:
    target, report_action, use_parent, depth, reasoning = _route_to_fields(
        route=classification.interaction,
        active_report=bool(state.active_report_job_id or state.last_report_markdown),
        research_depth=classification.research_depth or "shallow",
        depth_reasoning=classification.reasoning,
    )
    update: dict[str, Any] = {
        "user_intent": IntentResult(
            intent="research",
            target=target,
            report_action=report_action,
            use_parent_report_context=use_parent,
        )
    }
    if target != "report":
        update["depth_decision"] = DepthDecision(decision=depth, raw_reasoning=reasoning)
    return update


__all__ = ["ContextAwareIntentRouter", "EntryClassification", "RoutingProtocolError"]
