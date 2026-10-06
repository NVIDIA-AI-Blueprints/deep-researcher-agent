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

"""NAT register function for deep research agent."""

import asyncio
import logging
from typing import TypeVar

from langchain_core.messages import HumanMessage
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole
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

from .agent import DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS
from .agent import DEFAULT_MAX_RESEARCH_CONCURRENCY
from .agent import DEFAULT_MAX_RESEARCHER_MODEL_CALLS
from .agent import DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE
from .agent import DeepResearcherAgent
from .deepagents_runtime import DeepResearchSandboxConfig
from .deepagents_runtime import DeepResearchSkillsConfig
from .models import DeepResearchAgentState
from .resource_limits import DeepResearchExecutionTimeout
from .resource_limits import DeepResearchResourceLimits

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")


class DeepResearchAgentConfig(FunctionBaseConfig, name="deep_research_agent"):
    """Configuration for the deep research agent."""

    model_config = ConfigDict(extra="forbid")

    orchestrator_llm: LLMRef = Field(..., description="LLM for orchestrator")
    source_router_llm: LLMRef | None = Field(default=None, description="LLM for source-router subagent")
    researcher_llm: LLMRef | None = Field(default=None, description="LLM for researcher")
    planner_llm: LLMRef | None = Field(default=None, description="LLM for planner")
    writer_llm: LLMRef | None = Field(default=None, description="LLM for final writer/synthesis subagent")
    tools: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Explicit tool list. Empty = inherit all from data_source_registry.",
    )
    exclude_tools: list[str] = Field(
        default_factory=list,
        description="Tool names to exclude when inheriting from registry.",
    )
    domain_catalog_path: str | None = Field(
        default=None,
        description="Optional YAML/JSON domain catalog path for source-router-agent.",
    )
    enable_source_router: bool = Field(
        default=True,
        description="Enable the advisory source-router-agent before planning.",
    )
    enable_citation_verification: bool = Field(
        default=True,
        description="Verify generated citations against sources captured from configured tools.",
    )
    skills: DeepResearchSkillsConfig | FunctionRef | None = Field(
        default=None,
        description="Optional inline skills config or function ref to a deep_research_skills config.",
    )
    sandbox: DeepResearchSandboxConfig | FunctionRef | None = Field(
        default=None,
        description="Optional inline sandbox config or function ref to a deep_research_sandbox config.",
    )
    max_research_concurrency: int = Field(
        default=DEFAULT_MAX_RESEARCH_CONCURRENCY,
        ge=1,
        description="Maximum ResearchQuery items accepted and run concurrently per run_research_batch call.",
    )
    max_researcher_model_calls: int = Field(
        default=DEFAULT_MAX_RESEARCHER_MODEL_CALLS,
        ge=1,
        description="Maximum normal model turns per researcher worker before one reserved finalization turn.",
    )
    max_concurrent_source_tool_calls: int = Field(
        default=DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
        ge=1,
        description="Shared maximum concurrent source-tool calls across researcher workers.",
    )
    max_source_tool_batch_size: int = Field(
        default=DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
        ge=1,
        description="Maximum concrete inputs accepted by batch-capable source tool wrappers.",
    )
    resource_limits: DeepResearchResourceLimits = Field(
        default_factory=DeepResearchResourceLimits,
        description="Hard per-job limits for request, plan, notes, source calls, and execution time.",
    )

    @field_validator("skills", mode="before")
    @classmethod
    def _parse_inline_skills(cls, value):
        if isinstance(value, dict):
            return DeepResearchSkillsConfig.model_validate(value)
        return value

    @field_validator("sandbox", mode="before")
    @classmethod
    def _parse_inline_sandbox(cls, value):
        if isinstance(value, dict):
            return DeepResearchSandboxConfig.model_validate(value)
        return value

    @model_validator(mode="after")
    def _research_concurrency_fits_job_budget(self):
        if self.max_research_concurrency > self.resource_limits.max_research_queries:
            raise ValueError("max_research_concurrency cannot exceed resource_limits.max_research_queries")
        return self


@register_function(config_type=DeepResearchSkillsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_research_skills(config: DeepResearchSkillsConfig, builder: Builder):
    """Config-only function for deep research skill collection assignments."""

    async def _noop(query: str) -> str:
        """Deep research skills config placeholder."""
        return "This is a config-only function."

    yield FunctionInfo.from_fn(_noop, description="Deep research skills config-only function.")


@register_function(config_type=DeepResearchSandboxConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_research_sandbox(config: DeepResearchSandboxConfig, builder: Builder):
    """Config-only function for deep research sandbox settings."""

    async def _noop(query: str) -> str:
        """Deep research sandbox config placeholder."""
        return "This is a config-only function."

    yield FunctionInfo.from_fn(_noop, description="Deep research sandbox config-only function.")


def _resolve_config_ref(
    builder: Builder, value: ConfigT | FunctionRef | None, expected_type: type[ConfigT]
) -> ConfigT | None:
    if value is None or isinstance(value, expected_type):
        return value

    resolved = builder.get_function_config(value)
    if not isinstance(resolved, expected_type):
        raise TypeError(f"{value!r} must reference {expected_type.__name__}, got {type(resolved).__name__}")
    return resolved


def resolve_deep_research_runtime_config(
    config: DeepResearchAgentConfig,
    builder: Builder,
) -> tuple[DeepResearchSkillsConfig | None, DeepResearchSandboxConfig | None]:
    """Resolve optional Deep Research runtime config refs into concrete config objects."""
    skills = _resolve_config_ref(builder, config.skills, DeepResearchSkillsConfig)
    sandbox = _resolve_config_ref(builder, config.sandbox, DeepResearchSandboxConfig)
    return skills, sandbox


@register_function(config_type=DeepResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_research_agent(config: DeepResearchAgentConfig, builder: Builder):
    """Deep research agent using multi-phase workflow."""
    skills_config, sandbox_config = resolve_deep_research_runtime_config(config, builder)

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
        research_type="deep research",
    )
    if not is_valid:
        logger.warning(
            "Startup check: no tools available for deep research. "
            "All queries will fail until at least one tool is properly configured.",
        )

    llm = await builder.get_llm(config.orchestrator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    provider = LLMProvider()
    provider.set_default(llm)

    provider.configure(LLMRole.ORCHESTRATOR, llm)
    if config.source_router_llm:
        source_router_llm = await builder.get_llm(config.source_router_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.ROUTER, source_router_llm)
    if config.researcher_llm:
        researcher_llm = await builder.get_llm(config.researcher_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.RESEARCHER, researcher_llm)
    if config.planner_llm:
        planner_llm = await builder.get_llm(config.planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.PLANNER, planner_llm)
    if config.writer_llm:
        writer_llm = await builder.get_llm(config.writer_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.REPORT_WRITER, writer_llm)

    callbacks: list = []

    agent = DeepResearcherAgent(
        llm_provider=provider,
        tools=tools,
        callbacks=callbacks,
        domain_catalog_path=config.domain_catalog_path,
        enable_source_router=config.enable_source_router,
        enable_citation_verification=config.enable_citation_verification,
        skills=skills_config,
        sandbox=sandbox_config,
        max_research_concurrency=config.max_research_concurrency,
        max_researcher_model_calls=config.max_researcher_model_calls,
        max_concurrent_source_tool_calls=config.max_concurrent_source_tool_calls,
        max_source_tool_batch_size=config.max_source_tool_batch_size,
        resource_limits=config.resource_limits,
    )

    async def _run(state: DeepResearchAgentState) -> DeepResearchAgentState:
        """Run deep research with a list of messages or payload."""
        active_agent = agent
        owns_active_agent = False
        interrupted = False
        try:
            data_sources = state.data_sources
            validate_research_source_configuration(data_sources, "deep research")

            selected_tools = filter_tools_by_sources(tools, data_sources)
            if sandbox_config is not None or (data_sources is not None and selected_tools != tools):
                # Scope the Modal sandbox to the async job_id when one is in
                # NAT context (set by deep_researcher_api/jobs/runner.py). Falls back to a
                # per-request uuid in DeepAgentsRuntime when None.
                job_id: str | None = None
                try:
                    from nat.builder.context import Context

                    job_id = Context.get().workflow_run_id
                except Exception:  # noqa: BLE001 - Context may be unavailable in sync/eval paths
                    job_id = None
                active_agent = DeepResearcherAgent(
                    llm_provider=provider,
                    tools=selected_tools,
                    callbacks=callbacks,
                    domain_catalog_path=config.domain_catalog_path,
                    enable_source_router=config.enable_source_router,
                    enable_citation_verification=config.enable_citation_verification,
                    skills=skills_config,
                    sandbox=sandbox_config,
                    job_id=job_id,
                    max_research_concurrency=config.max_research_concurrency,
                    max_researcher_model_calls=config.max_researcher_model_calls,
                    max_concurrent_source_tool_calls=config.max_concurrent_source_tool_calls,
                    max_source_tool_batch_size=config.max_source_tool_batch_size,
                    resource_limits=config.resource_limits,
                )
                owns_active_agent = True

            if all_mapped_tools_filtered_out(tools, selected_tools, data_sources):
                logger.warning("Deep research received data_sources with no matching tools")

            result = await active_agent.run(state)
            return result
        except (asyncio.CancelledError, DeepResearchExecutionTimeout):
            interrupted = True
            raise
        except Exception as exc:
            logger.error(
                "Error in deep research execution (error_type=%s detail_%s)",
                type(exc).__name__,
                log_content_metadata(exc),
            )
            raise
        finally:
            if owns_active_agent:
                await asyncio.to_thread(active_agent.finalize, interrupted=interrupted)

    yield FunctionInfo.from_fn(_run, description="Deep research agent for comprehensive multi-phase research.")


########################################################
# Deep Research Workflow (Wrapper for Evaluation)
########################################################
class DeepResearchWorkflowConfig(FunctionBaseConfig, name="deep_research_workflow"):
    """Configuration for the deep research workflow wrapper.

    This wrapper accepts a string query and converts it to messages
    for the deep_research_agent. Use this as the workflow for evaluation.
    """

    use_async_deep_research: bool = Field(
        default=False,
        description="Submit deep research as an async job instead of running inline",
    )
    relay: RelayConfig = Field(default_factory=RelayConfig, description="NeMo Relay plugins and export destinations")


@register_function(config_type=DeepResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_research_workflow(config: DeepResearchWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries for evaluation."""
    await _ensure_relay_started(config.relay)
    deep_research_agent_fn = await builder.get_function("deep_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run deep research on a query string."""
        state = DeepResearchAgentState(messages=[HumanMessage(content=query)])
        try:
            result = await deep_research_agent_fn.ainvoke(state)
            response_content = result.messages[-1].content
        except EmptySourceRegistryError as exc:
            response_content = exc.public_response
        return _create_chat_response(response_content, response_id="research_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="Deep research workflow for evaluation (accepts string query).")
