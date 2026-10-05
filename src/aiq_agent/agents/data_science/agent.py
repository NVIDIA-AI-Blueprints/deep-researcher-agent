# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One autonomous data-science agent."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from aiq_agent.common import SourceRegistry
from aiq_agent.common import get_session_registry
from aiq_agent.common import load_prompt
from aiq_agent.common import reset_session_registry
from aiq_agent.common import sanitize_report
from aiq_agent.common import set_session_registry
from aiq_agent.common import validate_research_source_configuration
from aiq_agent.common.citation_verification import verify_citations

from .messages import is_clarification_request
from .messages import message_text
from .models import DataScienceAgentContext
from .models import DataScienceAgentState
from .models import InteractionMode
from .models import ResponseMode
from .utils.analysis_runtime import begin_analysis_run
from .utils.analysis_runtime import end_analysis_run
from .utils.analysis_runtime import get_analysis_run
from .utils.finalization import FinalizationReserveMiddleware
from .utils.prompt import build_prompt_middleware
from .utils.reporting import capture_data_sources
from .utils.reporting import citation_repair_instruction
from .utils.reporting import finalize_data_science_messages
from .utils.reporting import has_citation_integrity
from .utils.structured_data_guardrails import StructuredDataCallGuardMiddleware

AGENT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)
_HEADLESS_RETRY_MESSAGE_NAME = "aiq_headless_synthesis_retry"
_HEADLESS_RETRY_INSTRUCTION = (
    "No user interaction is available. Return the best supported answer to the original request now. "
    "Use the semantic and query evidence already gathered; make and disclose only defensible assumptions. "
    "If the request still cannot be completed safely, give a terminal explanation without asking a question."
)
_HEADLESS_TERMINAL_RESPONSE = (
    "I could not complete the request non-interactively because a material ambiguity remained after semantic "
    "discovery and one bounded synthesis retry. The available evidence did not support a safe assumption."
)
_CHOICE_REPAIR_MESSAGE_NAME = "aiq_choice_format_repair"
_CITATION_REPAIR_MESSAGE_NAME = "aiq_citation_integrity_repair"
_EMPTY_RESPONSE_RETRY_MESSAGE_NAME = "aiq_empty_response_synthesis_retry"
_EMPTY_RESPONSE_RETRY_INSTRUCTION = (
    "Your previous final response contained no visible answer. Return the best supported final answer to the "
    "original request now, using only evidence already present in the conversation. Do not call tools, ask a "
    "question, or return an empty response. Follow the required answer-first and citation contracts."
)
_EMPTY_RESPONSE_TERMINAL = (
    "I could not produce a supported answer because the final synthesis model returned no visible content after "
    "one bounded retry."
)
_PRELOADED_CATALOG_TOOL = "aiq__preloaded_catalog_context"
_PRELOADED_CATALOG_MESSAGE_NAME = "aiq_preloaded_catalog_context"
_MAX_CATALOG_ITEMS = 50
_MAX_CATALOG_TEXT_CHARS = 512
_MULTIPLE_CHOICE_PATTERNS = (
    r"\bmultiple[- ]choice\b",
    r"\bmulti(?:ple)?[- ]select\b",
    r"\b(?:select|choose|check|mark)\s+all\b",
    r"\ball\s+(?:correct|applicable)\s+(?:answers?|choices?|options?|responses?|statements?)\b",
    r"\bone\s+or\s+more\s+(?:answers?|choices?|options?|responses?|statements?)\b",
    r"\bmultiple\s+(?:answers?|choices?|options?|responses?|statements?)\b",
    r"\bmore\s+than\s+one\s+(?:answer|choice|option|response|statement)\b",
)
_SINGLE_CHOICE_PATTERNS = (
    r"\bsingle[- ]choice\b",
    r"\b(?:select|choose)\s+exactly\s+one\b",
    r"\b(?:select|choose)\s+one\s+(?:answer|choice|option|response|statement)\b",
    r"\b(?:only|exactly)\s+one\s+(?:answer|choice|option|response|statement)\b",
    r"\bone\s+correct\s+(?:answer|choice|option|response|statement)\b",
)


def _is_terminal_status_response(message: Any) -> bool:
    """Return whether an uncited response is a fixed, non-evidentiary terminal status."""

    return message_text(message).strip() in {_HEADLESS_TERMINAL_RESPONSE, _EMPTY_RESPONSE_TERMINAL}


def _bounded_catalog_text(value: str) -> str:
    return value[:_MAX_CATALOG_TEXT_CHARS]


def _preloaded_catalog_messages(context: DataScienceAgentContext) -> list[Any]:
    """Represent validated router context as untrusted tool data, never instructions."""

    catalog = context.catalog_context
    if catalog is None:
        return []
    call_id = f"preloaded-catalog-{uuid4()}"
    payload = {
        "type": "preloaded_catalog_context",
        "database_name": context.database_name,
        "coverage": catalog.coverage,
        "truncated": catalog.truncated,
        "uncovered_entities": [
            _bounded_catalog_text(value) for value in (catalog.uncovered_entities or [])[:_MAX_CATALOG_ITEMS]
        ],
        "candidates": [
            {
                "term": _bounded_catalog_text(candidate.term),
                "attribute": _bounded_catalog_text(candidate.attribute),
                "label": _bounded_catalog_text(candidate.label),
                "id": _bounded_catalog_text(candidate.id),
            }
            for candidate in catalog.candidates[:_MAX_CATALOG_ITEMS]
        ],
    }
    return [
        AIMessage(
            content="",
            id=f"{call_id}-request",
            name=_PRELOADED_CATALOG_MESSAGE_NAME,
            tool_calls=[{"name": _PRELOADED_CATALOG_TOOL, "args": {}, "id": call_id, "type": "tool_call"}],
        ),
        ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            id=f"{call_id}-result",
            name=_PRELOADED_CATALOG_TOOL,
            tool_call_id=call_id,
        ),
    ]


def _without_preloaded_catalog_messages(messages: Sequence[Any]) -> list[Any]:
    return [
        message
        for message in messages
        if getattr(message, "name", None) not in {_PRELOADED_CATALOG_MESSAGE_NAME, _PRELOADED_CATALOG_TOOL}
    ]


def _choice_contract(messages: Sequence[Any]) -> tuple[list[str], bool] | None:
    latest = next((message_text(message) for message in reversed(messages) if isinstance(message, HumanMessage)), "")
    labels = list(dict.fromkeys(match.upper() for match in re.findall(r"(?im)^\s*([A-Z])\s*(?:[.)]|:)\s+\S", latest)))
    lowered = latest.lower()
    multiple = any(re.search(pattern, lowered) for pattern in _MULTIPLE_CHOICE_PATTERNS)
    single = any(re.search(pattern, lowered) for pattern in _SINGLE_CHOICE_PATTERNS)
    if len(labels) < 2 or not (multiple or single):
        return None
    return labels, multiple


def _has_valid_choice_line(content: str, labels: Sequence[str], *, multiple: bool) -> bool:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    match = re.fullmatch(r"Answer\s*[:：]\s*([^\r\n]+)", first_line, flags=re.IGNORECASE)
    if match is None:
        return False
    values = [value.strip().upper() for value in match.group(1).split(",")]
    return bool(values) and all(value in labels for value in values) and (multiple or len(values) == 1)


def _visible_report_text(message: Any) -> str:
    """Return displayable report text after the public sanitizer runs."""
    content = message_text(message).strip()
    if not content:
        return ""
    return sanitize_report(content).sanitized_report.strip()


class DataScienceAgent:
    """Run discovery, adaptive tool calls, analysis, and writing in one history."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        recursion_limit: int = 64,
        callbacks: Sequence[Any] = (),
        middleware: Sequence[AgentMiddleware] = (),
        interaction_mode: InteractionMode = "interactive",
        response_mode: ResponseMode = "standard",
        structured_guard: StructuredDataCallGuardMiddleware | None = None,
        python_call_limit: int | None = None,
        finalization_model_call_limit: int | None = None,
    ) -> None:
        if recursion_limit < 4:
            raise ValueError("recursion_limit must be at least four")

        tool_name_counts = Counter(tool.name for tool in tools)
        duplicates = sorted(name for name, count in tool_name_counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"data-science agent received duplicate tool names: {', '.join(duplicates)}")
        if not tool_name_counts:
            raise ValueError("data-science agent has no available data tools")
        if interaction_mode not in {"interactive", "headless"}:
            raise ValueError(f"unsupported data-science interaction mode: {interaction_mode}")
        if response_mode not in {"standard", "fdabench_choice"}:
            raise ValueError(f"unsupported data-science response mode: {response_mode}")

        agent_tools = list(tools)
        prompt_middleware = build_prompt_middleware(
            load_prompt(AGENT_DIR / "prompts", "agent"),
            agent_tools,
            interaction_mode=interaction_mode,
            response_mode=response_mode,
            structured_catalog_call_limit=structured_guard.budget.catalog_calls if structured_guard else None,
            structured_text_to_sql_call_limit=structured_guard.budget.text_to_sql_calls if structured_guard else None,
            python_call_limit=python_call_limit,
        )
        agent_middleware = [prompt_middleware]
        if structured_guard is not None:
            agent_middleware.append(structured_guard)
        if python_call_limit is not None and "python" in tool_name_counts:
            agent_middleware.append(
                ToolCallLimitMiddleware(
                    tool_name="python",
                    run_limit=python_call_limit,
                    exit_behavior="continue",
                )
            )
        effective_finalization_limit = finalization_model_call_limit or max(2, (recursion_limit - 8) // 2)
        agent_middleware.append(FinalizationReserveMiddleware(effective_finalization_limit))
        agent_middleware.extend(middleware)
        self.graph: CompiledStateGraph = create_agent(
            model=llm,
            tools=agent_tools,
            middleware=agent_middleware,
            context_schema=DataScienceAgentContext,
            name="data_science_agent",
        )
        self.tools = agent_tools
        self.recursion_limit = recursion_limit
        self.source_tool_names = frozenset(tool_name_counts)
        self.callbacks = tuple(callbacks)
        self.interaction_mode = interaction_mode
        self.response_mode = response_mode
        self.structured_guard = structured_guard
        self.python_call_limit = python_call_limit
        self.finalization_model_call_limit = effective_finalization_limit

    @staticmethod
    def _validate_question(state: DataScienceAgentState) -> None:
        if not state.messages:
            raise ValueError("data-science agent requires at least one message")
        latest = next((message for message in reversed(state.messages) if isinstance(message, HumanMessage)), None)
        if latest is None or not message_text(latest).strip():
            raise ValueError("data-science agent received an empty question")

    async def run(self, state: DataScienceAgentState) -> DataScienceAgentState:
        """Execute one request while preserving any caller-owned source registry."""
        # Both the NAT registrar and the async job runner call this interface, so source
        # selection and availability must be enforced here rather than by either adapter.
        validate_research_source_configuration(state.data_sources, "data science", self.tools)
        self._validate_question(state)
        registry_token = None
        analysis_run_token = begin_analysis_run()
        structured_run_token = self.structured_guard.begin_run() if self.structured_guard is not None else None
        registry = get_session_registry()
        if registry is None:
            registry = SourceRegistry()
            registry_token = set_session_registry(registry)
        try:
            invoke_config: dict[str, Any] = {"recursion_limit": self.recursion_limit}
            if self.callbacks:
                invoke_config["callbacks"] = list(self.callbacks)
            runtime_context = DataScienceAgentContext(
                user_info=state.user_info,
                database_name=state.database_name,
                catalog_context=state.catalog_context,
                catalog_request_id=state.catalog_request_id,
            )
            input_messages = [*state.messages, *_preloaded_catalog_messages(runtime_context)]
            result = await self.graph.ainvoke(
                {"messages": input_messages},
                config=invoke_config,
                context=runtime_context,
            )
            result_messages = list(result["messages"])
            if (
                self.interaction_mode == "headless"
                and result_messages
                and is_clarification_request(result_messages[-1])
            ):
                retry_id = str(uuid4())
                retry_input = [
                    *result_messages[:-1],
                    HumanMessage(
                        content=_HEADLESS_RETRY_INSTRUCTION,
                        id=retry_id,
                        name=_HEADLESS_RETRY_MESSAGE_NAME,
                    ),
                ]
                retry_result = await self.graph.ainvoke(
                    {"messages": retry_input},
                    config=invoke_config,
                    context=runtime_context,
                )
                result_messages = [
                    message
                    for message in retry_result["messages"]
                    if getattr(message, "id", None) != retry_id
                    and getattr(message, "name", None) != _HEADLESS_RETRY_MESSAGE_NAME
                ]
                if result_messages and is_clarification_request(result_messages[-1]):
                    result_messages[-1] = result_messages[-1].model_copy(
                        update={"content": _HEADLESS_TERMINAL_RESPONSE}
                    )
            if not result_messages or not _visible_report_text(result_messages[-1]):
                run_state = get_analysis_run()
                if run_state is not None:
                    run_state.force_finalization = True
                    run_state.finalization_instruction = _EMPTY_RESPONSE_RETRY_INSTRUCTION
                retry_id = str(uuid4())
                retry_history = result_messages
                if retry_history and isinstance(retry_history[-1], AIMessage):
                    # Drop blank output and leaked tool-call markup. Keeping a max-token
                    # malformed answer in context can cause the repair call to repeat it.
                    retry_history = retry_history[:-1]
                retry_input = [
                    *retry_history,
                    HumanMessage(
                        content=_EMPTY_RESPONSE_RETRY_INSTRUCTION,
                        id=retry_id,
                        name=_EMPTY_RESPONSE_RETRY_MESSAGE_NAME,
                    ),
                ]
                retry_result = await self.graph.ainvoke(
                    {"messages": retry_input},
                    config=invoke_config,
                    context=runtime_context,
                )
                result_messages = [
                    message
                    for message in retry_result["messages"]
                    if getattr(message, "id", None) != retry_id
                    and getattr(message, "name", None) != _EMPTY_RESPONSE_RETRY_MESSAGE_NAME
                ]
                if not result_messages:
                    result_messages = [AIMessage(content=_EMPTY_RESPONSE_TERMINAL)]
                elif not _visible_report_text(result_messages[-1]):
                    result_messages[-1] = result_messages[-1].model_copy(update={"content": _EMPTY_RESPONSE_TERMINAL})
            choice_contract = _choice_contract(state.messages) if self.response_mode == "fdabench_choice" else None
            if choice_contract and result_messages:
                labels, multiple = choice_contract
                if not _has_valid_choice_line(message_text(result_messages[-1]), labels, multiple=multiple):
                    run_state = get_analysis_run()
                    if run_state is not None:
                        run_state.force_finalization = True
                        run_state.finalization_instruction = (
                            "FORMAT REPAIR ONLY: Return exactly one plain-text `Answer:` line containing "
                            "the conclusion already reached. Do not use tools, redo the analysis, add "
                            "rationale, citations, Markdown, or a Sources section."
                        )
                    retry_id = str(uuid4())
                    selection_rule = (
                        "Select every supported label, comma-separated with no spaces."
                        if multiple
                        else "Select exactly one label."
                    )
                    retry_input = [
                        *result_messages,
                        HumanMessage(
                            content=(
                                "Format repair only. Using the conclusion already reached, return exactly one line and "
                                "nothing else: `Answer: <labels>`. Valid labels are "
                                f"{', '.join(labels)}. {selection_rule}"
                            ),
                            id=retry_id,
                            name=_CHOICE_REPAIR_MESSAGE_NAME,
                        ),
                    ]
                    retry_result = await self.graph.ainvoke(
                        {"messages": retry_input},
                        config=invoke_config,
                        context=runtime_context,
                    )
                    result_messages = [
                        message
                        for message in retry_result["messages"]
                        if getattr(message, "id", None) != retry_id
                        and getattr(message, "name", None) != _CHOICE_REPAIR_MESSAGE_NAME
                    ]
            result_messages = _without_preloaded_catalog_messages(result_messages)
            capture_data_sources(
                result_messages,
                registry=registry,
                eligible_tool_names=self.source_tool_names,
            )
            if choice_contract:
                # Preserve the exact leading Answer line required by the benchmark.
                # The model still supplies rationale and sources after the blank line.
                messages = result_messages
            elif result_messages and _is_terminal_status_response(result_messages[-1]):
                messages = result_messages
            else:
                sources = registry.all_sources()
                if sources and result_messages:
                    content = message_text(result_messages[-1])
                    verification = verify_citations(content, registry, reference_sources=sources)
                    if not has_citation_integrity(verification.verified_report, verification):
                        repair_instruction = citation_repair_instruction(sources)
                        run_state = get_analysis_run()
                        if run_state is not None:
                            run_state.force_finalization = True
                            run_state.finalization_instruction = repair_instruction
                        repair_id = str(uuid4())
                        repair_result = await self.graph.ainvoke(
                            {
                                "messages": [
                                    *result_messages,
                                    HumanMessage(
                                        content=repair_instruction,
                                        id=repair_id,
                                        name=_CITATION_REPAIR_MESSAGE_NAME,
                                    ),
                                ]
                            },
                            config=invoke_config,
                            context=runtime_context,
                        )
                        result_messages = [
                            message
                            for message in repair_result["messages"]
                            if getattr(message, "id", None) != repair_id
                            and getattr(message, "name", None) != _CITATION_REPAIR_MESSAGE_NAME
                        ]
                messages = finalize_data_science_messages(
                    result_messages,
                    registry=registry,
                    callbacks=self.callbacks,
                    data_sources=state.data_sources,
                    available_tools=list(self.source_tool_names),
                )
        finally:
            try:
                if structured_run_token is not None and self.structured_guard is not None:
                    summary = self.structured_guard.summarize_run()
                    if summary and (summary["catalog_calls"] or summary["text_to_sql_calls"] or summary["cache_hits"]):
                        logger.info("Data-science structured-data call summary: %s", summary)
                    self.structured_guard.end_run(structured_run_token)
                await end_analysis_run(analysis_run_token)
            finally:
                if registry_token is not None:
                    reset_session_registry(registry_token)

        return state.model_copy(update={"messages": messages})
