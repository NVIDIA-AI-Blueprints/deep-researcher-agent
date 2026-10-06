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

"""NAT register function for shallow research agent."""

import logging

from langchain_core.messages import HumanMessage
from pydantic import Field

from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import _create_chat_response
from deep_researcher_agent.common import all_mapped_tools_filtered_out
from deep_researcher_agent.common import filter_tools_by_sources
from deep_researcher_agent.common import validate_research_source_configuration
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryError
from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.relay.bootstrap import ensure_started as _ensure_relay_started
from deep_researcher_agent.relay.config import RelayConfig
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import ShallowResearcherAgent
from .models import ShallowResearchAgentState

logger = logging.getLogger(__name__)


class ShallowResearchAgentConfig(FunctionBaseConfig, name="shallow_research_agent"):
    """Configuration for the shallow research agent."""

    llm: LLMRef = Field(..., description="LLM to use")
    tools: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Explicit tool list. Empty = inherit all from data_source_registry.",
    )
    exclude_tools: list[str] = Field(
        default_factory=list,
        description="Tool names to exclude when inheriting from registry.",
    )
    max_llm_turns: int = Field(default=10, description="Maximum number of LLM turns")
    max_tool_iterations: int = Field(default=5, description="Maximum tool-calling iterations before forcing synthesis")
    enforce_citations: bool = Field(
        default=False,
        description="Fail instead of returning a generated answer when citation integrity cannot be preserved.",
    )
    verbose: bool = Field(default=False, description="Whether to enable verbose logging")


@register_function(config_type=ShallowResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def shallow_research_agent(config: ShallowResearchAgentConfig, builder: Builder):
    """Shallow research agent with tool-calling capabilities."""
    llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if config.tools:
        tool_refs = config.tools
    else:
        from deep_researcher_agent.common import get_all_tool_refs

        tool_refs = get_all_tool_refs()

    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if config.exclude_tools:
        excluded = set(config.exclude_tools)
        tools = [t for t in tools if getattr(t, "name", "") not in excluded]

    from deep_researcher_agent.common import validate_tool_availability

    is_valid, available_count, unavailable = validate_tool_availability(
        tools,
        research_type="shallow research",
    )
    if not is_valid:
        logger.warning(
            "Startup check: no tools available for shallow research. "
            "All queries will fail until at least one tool is properly configured.",
        )

    provider = LLMProvider()
    provider.set_default(llm)

    callbacks: list = []

    # No shared agent is built here: it is (re)built per request inside _run, since
    # the active tool set depends on the request's data_sources and the per-user MCP
    # tools resolved at run time.

    async def _run(state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        from contextlib import AsyncExitStack

        try:
            data_sources = state.data_sources
            validate_research_source_configuration(data_sources, "shallow research")

            selected_tools = filter_tools_by_sources(tools, data_sources)

            if all_mapped_tools_filtered_out(tools, selected_tools, data_sources):
                logger.warning("Shallow research received data_sources with no matching tools")

            # Per-user MCP tools (e.g. a connected Google Drive) are resolved at RUN
            # time: this runs per request with Context.user_id set, so we can build the
            # user's MCP client and add its tools to this turn. Build-time inheritance
            # can't (the agent would be a shared, user-less instance) — see
            # deep_researcher_api.mcp_auth.runtime_tools. The client stays open via mcp_stack for the run.
            async with AsyncExitStack() as mcp_stack:
                # Per-user MCP tools require ``deep_researcher_api`` (the API/auth layer under
                # frontends/deep_researcher_api), which the standalone public MCP image intentionally does
                # not bundle — mcp/Dockerfile copies only deep_researcher_agent, select sources,
                # and deep_researcher_mcp.
                # That profile runs anonymous with no per-user OAuth, so per-user sources never
                # apply there. Resolve the reconnect exception type up front (guarded) so the
                # handler below can never reference an unbound name when the ``deep_researcher_api`` import
                # fails; a missing ``deep_researcher_api`` is an expected skip, not a failure.
                try:
                    from deep_researcher_api.mcp_auth.runtime_tools import PerUserMcpSourceUnavailableError
                except ImportError:
                    PerUserMcpSourceUnavailableError = None

                try:
                    from deep_researcher_api.jobs.access import require_verified_principal
                    from deep_researcher_api.mcp_auth.provider import principal_user_id
                    from deep_researcher_api.mcp_auth.runtime_tools import open_per_user_mcp_tools
                    from nat.builder.context import ContextState

                    # Align the MCP token-lookup key with where connect stored it. The
                    # interactive session sets Context.user_id to NAT's user id — a *different*
                    # derivation than principal_user_id (used by connect/status). Without this,
                    # per_user_mcp_client looks under the wrong key, finds no token, and triggers
                    # interactive re-auth even though the source shows connected. Set with no
                    # reset: the client reads Context.user_id at build and on each tool call, so
                    # it must stay set for the whole turn.
                    ContextState.get().user_id.set(principal_user_id(require_verified_principal()))

                    mcp_tools = await open_per_user_mcp_tools(
                        builder=builder, data_sources=data_sources, exit_stack=mcp_stack
                    )
                    if mcp_tools:
                        selected_tools = [*selected_tools, *mcp_tools]
                except ImportError:
                    # Standalone MCP profile without deep_researcher_api: expected — continue with base tools.
                    logger.debug("deep_researcher_api unavailable; skipping per-user MCP tools for shallow research")
                except Exception as exc:
                    if PerUserMcpSourceUnavailableError is not None and isinstance(
                        exc, PerUserMcpSourceUnavailableError
                    ):
                        # The user explicitly selected a protected source we can't resolve
                        # (e.g. token expired). Surface a reconnect message instead of
                        # silently answering without it.
                        from langchain_core.messages import AIMessage

                        return ShallowResearchAgentState(messages=state.messages + [AIMessage(content=str(exc))])
                    logger.error(
                        "Failed to resolve per-user MCP tools for shallow research; continuing "
                        "(error_type=%s detail_%s)",
                        type(exc).__name__,
                        log_content_metadata(exc),
                    )

                # Build the agent with this turn's tool set.
                active_agent = ShallowResearcherAgent(
                    llm_provider=provider,
                    tools=selected_tools,
                    max_llm_turns=config.max_llm_turns,
                    max_tool_iterations=config.max_tool_iterations,
                    enforce_citations=config.enforce_citations,
                    callbacks=callbacks,
                )

                validate_research_source_configuration(data_sources, "shallow research", selected_tools)

                return await active_agent.run(state)
        except Exception as exc:
            logger.error(
                "Error in shallow research execution (error_type=%s detail_%s)",
                type(exc).__name__,
                log_content_metadata(exc),
            )
            raise

    yield FunctionInfo.from_fn(_run, description="Shallow research agent for fast, bounded research.")


########################################################
# Shallow Research Workflow (Wrapper for Evaluation)
########################################################
class ShallowResearchWorkflowConfig(FunctionBaseConfig, name="shallow_research_workflow"):
    """Configuration for the shallow research workflow wrapper.

    This wrapper accepts a string query and converts it to messages
    for the shallow_research_agent. Use this as the workflow for evaluation.
    """

    relay: RelayConfig = Field(default_factory=RelayConfig, description="NeMo Relay plugins and export destinations")


@register_function(config_type=ShallowResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def shallow_research_workflow(config: ShallowResearchWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries for evaluation."""
    await _ensure_relay_started(config.relay)
    shallow_research_agent_fn = await builder.get_function("shallow_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run shallow research on a query string."""
        try:
            result = await shallow_research_agent_fn.ainvoke(
                ShallowResearchAgentState(messages=[HumanMessage(content=query)])
            )
            response_content = result.messages[-1].content
        except EmptySourceRegistryError as exc:
            response_content = exc.public_response
        return _create_chat_response(response_content, response_id="research_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="Shallow research workflow for evaluation (accepts string query).")
