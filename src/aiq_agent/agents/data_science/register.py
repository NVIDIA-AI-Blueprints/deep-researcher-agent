# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT registration and composition for the data-science agent."""

import logging
from typing import Any
from typing import Literal

from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from pydantic import ConfigDict
from pydantic import Field

from aiq_agent.agents.chat_researcher.models import ChatResearcherState
from aiq_agent.agents.chat_researcher.utils import _extract_query_context
from aiq_agent.common import VerboseTraceCallback
from aiq_agent.common import _create_chat_response
from aiq_agent.common import all_mapped_tools_filtered_out
from aiq_agent.common import filter_tools_by_sources
from aiq_agent.common import get_all_tool_refs
from aiq_agent.common import get_source_id_for_tool
from aiq_agent.common import validate_research_source_configuration
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.logging_utils import log_content_metadata
from aiq_agent.ontology import OntologyProviderConfig
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import DataScienceAgent
from .models import DataScienceAgentState
from .sandboxed_python import SandboxedPythonConfig
from .utils.structured_data_guardrails import StructuredDataCallBudget
from .utils.structured_data_guardrails import StructuredDataCallGuardMiddleware

logger = logging.getLogger(__name__)

__all__ = [
    "DataScienceAgentConfig",
    "DataScienceHybridAdapterConfig",
    "DataScienceWorkflowConfig",
    "SandboxedPythonConfig",
    "data_science_agent",
    "data_science_hybrid_adapter",
    "data_science_workflow",
]


class DataScienceAgentConfig(FunctionBaseConfig, name="data_science_agent"):
    """Configuration for one adaptive, tool-using data-science controller."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMRef
    tools: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Explicit tools. An empty list inherits all tools from data_source_registry.",
    )
    exclude_tools: list[str] = Field(
        default_factory=list,
        description="Exact runtime tool names removed after tool references are resolved.",
    )
    recursion_limit: int = Field(
        default=64,
        ge=4,
        description="Hard LangGraph step bound for one autonomous agent run.",
    )
    interaction_mode: Literal["interactive", "headless"] = Field(
        default="interactive",
        description="Whether the agent may request user clarification or must complete without interaction.",
    )
    response_mode: Literal["standard", "fdabench_choice"] = Field(
        default="standard",
        description="Optional response contract; FDABench choice mode preserves labels when choices are present.",
    )
    ontology_provider: OntologyProviderConfig | None = Field(
        default=None,
        description=(
            "Provider-neutral assignment of catalog, analytical, and predictive roles. "
            "Referenced tools must also be loaded through tools or data_source_registry."
        ),
    )
    structured_catalog_call_limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional request-local hard limit for ontology-provider catalog calls.",
    )
    structured_text_to_sql_call_limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional request-local hard limit for ontology-provider text-to-SQL calls.",
    )
    structured_cache_repeated_calls: bool = Field(
        default=True,
        description="Reuse exact repeated catalog and text-to-SQL calls within one request.",
    )
    python_call_limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional request-local hard limit for sandboxed Python analysis calls.",
    )
    finalization_model_call_limit: int | None = Field(
        default=None,
        ge=1,
        description="Model-call count at which tools are disabled and a final synthesis turn is forced.",
    )
    verbose: bool = False


class DataScienceWorkflowConfig(FunctionBaseConfig, name="data_science_workflow"):
    """String-input workflow wrapper for running the DS Agent directly."""

    model_config = ConfigDict(extra="forbid")


class DataScienceHybridAdapterConfig(FunctionBaseConfig, name="data_science_hybrid_adapter"):
    """Adapt Chat Researcher hybrid state to the autonomous DS Agent contract."""

    model_config = ConfigDict(extra="forbid")

    agent: FunctionRef = Field(description="Configured data_science_agent function to invoke.")


def _active_ontology_provider(
    provider: OntologyProviderConfig | None,
    tools: list[Any],
) -> OntologyProviderConfig | None:
    """Resolve configured provider roles against one request's filtered tools."""

    available = {tool.name for tool in tools}
    if provider is None:
        return None

    assigned = provider.tool_names & available
    if not assigned:
        return None
    missing = sorted(provider.tool_names - available)
    if missing:
        raise ValueError(f"ontology provider references unavailable tools: {', '.join(missing)}")
    return provider


def _validate_ontology_provider_source_mapping(provider: OntologyProviderConfig | None) -> None:
    """Require every configured ontology tool role to share one registry source."""

    if provider is None:
        return

    source_by_tool = {tool_name: get_source_id_for_tool(tool_name) for tool_name in provider.tool_names}
    unmapped = sorted(tool_name for tool_name, source_id in source_by_tool.items() if source_id is None)
    if unmapped:
        raise ValueError(
            f"ontology provider tools must be mapped in data_source_registry; unmapped tools: {', '.join(unmapped)}"
        )

    source_ids = {source_id for source_id in source_by_tool.values() if source_id is not None}
    if len(source_ids) != 1:
        mappings = ", ".join(f"{tool_name} -> {source_by_tool[tool_name]}" for tool_name in sorted(source_by_tool))
        raise ValueError(f"ontology provider tools must map to the same data source; mappings: {mappings}")


@register_function(config_type=DataScienceAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def data_science_agent(config: DataScienceAgentConfig, builder: Builder):
    """Resolve configured AI-Q tools and compose one contiguous ReAct loop."""
    tool_refs = config.tools or get_all_tool_refs()
    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if config.exclude_tools:
        excluded = set(config.exclude_tools)
        tools = [tool for tool in tools if tool.name not in excluded]

    _validate_ontology_provider_source_mapping(config.ontology_provider)
    validate_research_source_configuration(None, "data science", tools)

    llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    callbacks = (VerboseTraceCallback(),) if config.verbose else ()

    async def _run(state: DataScienceAgentState) -> DataScienceAgentState:
        from contextlib import AsyncExitStack

        validate_research_source_configuration(state.data_sources, "data science")
        selected_tools = filter_tools_by_sources(tools, state.data_sources)
        if all_mapped_tools_filtered_out(tools, selected_tools, state.data_sources):
            logger.warning("Data-science request selected data sources with no matching tools")

        async with AsyncExitStack() as mcp_stack:
            try:
                from aiq_api.mcp_auth.runtime_tools import PerUserMcpSourceUnavailableError
            except ImportError:
                PerUserMcpSourceUnavailableError = None

            try:
                from aiq_api.jobs.access import require_verified_principal
                from aiq_api.mcp_auth.provider import principal_user_id
                from aiq_api.mcp_auth.runtime_tools import open_per_user_mcp_tools
                from nat.builder.context import ContextState

                ContextState.get().user_id.set(principal_user_id(require_verified_principal()))
                mcp_tools = await open_per_user_mcp_tools(
                    builder=builder,
                    data_sources=state.data_sources,
                    exit_stack=mcp_stack,
                )
                if mcp_tools:
                    selected_tools = [*selected_tools, *mcp_tools]
            except ImportError:
                logger.debug("aiq_api unavailable; skipping per-user MCP tools for data science")
            except Exception as exc:
                if isinstance(exc, HTTPException) and exc.status_code == 403:
                    raise
                if PerUserMcpSourceUnavailableError is not None and isinstance(
                    exc,
                    PerUserMcpSourceUnavailableError,
                ):
                    return state.model_copy(update={"messages": [*state.messages, AIMessage(content=str(exc))]})
                logger.error(
                    "Failed to resolve per-user MCP tools for data science; continuing (error_type=%s detail_%s)",
                    type(exc).__name__,
                    log_content_metadata(exc),
                )

            validate_research_source_configuration(state.data_sources, "data science", selected_tools)
            active_provider = _active_ontology_provider(config.ontology_provider, selected_tools)
            structured_guard = None
            if active_provider is not None:
                structured_guard = StructuredDataCallGuardMiddleware(
                    provider=active_provider.provider,
                    catalog_tools=active_provider.catalog_tool_names,
                    text_to_sql_tools=active_provider.analytical_tool_names,
                    budget=StructuredDataCallBudget(
                        catalog_calls=config.structured_catalog_call_limit,
                        text_to_sql_calls=config.structured_text_to_sql_call_limit,
                        cache_repeated_calls=config.structured_cache_repeated_calls,
                    ),
                )
            active_agent = DataScienceAgent(
                llm=llm,
                tools=selected_tools,
                recursion_limit=config.recursion_limit,
                callbacks=callbacks,
                interaction_mode=config.interaction_mode,
                response_mode=config.response_mode,
                structured_guard=structured_guard,
                python_call_limit=config.python_call_limit,
                finalization_model_call_limit=config.finalization_model_call_limit,
            )
            return await active_agent.run(state)

    yield FunctionInfo.from_fn(
        _run,
        description="Adaptive data-science agent for structured data, document retrieval, web evidence, and synthesis.",
    )


@register_function(config_type=DataScienceHybridAdapterConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def data_science_hybrid_adapter(config: DataScienceHybridAdapterConfig, builder: Builder):
    """Expose the DS Agent through Chat Researcher's optional Hybrid boundary."""
    agent_fn = await builder.get_function(config.agent)

    async def _run(state: ChatResearcherState) -> dict[str, Any]:
        agent_state = DataScienceAgentState(
            messages=state.messages,
            data_sources=state.data_sources,
            user_info=state.user_info,
            database_name=state.database_name,
            catalog_context=state.catalog_context,
            catalog_request_id=state.catalog_request_id,
        )
        result = await agent_fn.ainvoke(agent_state)
        new_messages = result.messages[len(agent_state.messages) :]
        final_message = next(
            (
                message
                for message in reversed(new_messages)
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
            ),
            None,
        )
        if final_message is None:
            raise RuntimeError("Data Science Agent returned no final response")
        return {"messages": [final_message]}

    yield FunctionInfo.from_fn(
        _run,
        description="Chat Researcher Hybrid adapter for the autonomous Data Science Agent.",
    )


@register_function(config_type=DataScienceWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def data_science_workflow(config: DataScienceWorkflowConfig, builder: Builder):
    """Expose the DS Agent as a standard string-to-ChatResponse workflow."""
    agent_fn = await builder.get_function("data_science_agent")

    async def _run(query: object) -> ChatResponse:
        request_context = _extract_query_context(query)
        try:
            result = await agent_fn.ainvoke(
                DataScienceAgentState(
                    messages=[HumanMessage(content=request_context.query_text)],
                    data_sources=request_context.data_sources,
                    database_name=request_context.database_name,
                )
            )
            content = str(result.messages[-1].content)
        except EmptySourceRegistryError as exc:
            content = exc.public_response
        return _create_chat_response(
            content,
            response_id="data_science_response",
            model=config.type,
        )

    yield FunctionInfo.from_fn(_run, description="Direct data-science workflow for local development.")
