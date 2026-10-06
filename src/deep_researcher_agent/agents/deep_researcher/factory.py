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

"""Graph and middleware factory for the deep researcher agent and its subagents."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore

from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole
from deep_researcher_agent.common import render_prompt_template
from deep_researcher_agent.relay import deepagents_kwargs
from deep_researcher_agent.relay import merge_langchain_middleware

from .custom_middleware import RESEARCHER_FINALIZATION_MODEL_CALLS
from .custom_middleware import ArtifactHarvestMiddleware
from .custom_middleware import EmptyContentFixMiddleware
from .custom_middleware import ExecuteTimeoutClampMiddleware
from .custom_middleware import FilesystemToolCallGuardMiddleware
from .custom_middleware import FinalReportCommitMiddleware
from .custom_middleware import FinalReportCommitTracker
from .custom_middleware import FinalReportOwnershipGuardMiddleware
from .custom_middleware import PlanPersistenceMiddleware
from .custom_middleware import RequiredOutputFileMiddleware
from .custom_middleware import RequiredWriterDelegationMiddleware
from .custom_middleware import ResearcherFinalizationMiddleware
from .custom_middleware import SourceRegistryMiddleware
from .custom_middleware import SourceRoutingGuardMiddleware
from .custom_middleware import SourceRoutingPersistenceMiddleware
from .custom_middleware import StateMutationGuardMiddleware
from .custom_middleware import StructuredResponseTextFallbackMiddleware
from .custom_middleware import TodoQuotaMiddleware
from .custom_middleware import TodoSuppressionMiddleware
from .custom_middleware import ToolNameSanitizationMiddleware
from .custom_middleware import ToolResultPruningMiddleware
from .custom_middleware import ToolVisibilityMiddleware
from .deepagents_runtime import BUILTIN_SKILL_SOURCE
from .deepagents_runtime import DeepAgentsRuntime
from .models import DeepResearchAgentState
from .models import ResearchNotes
from .models import ResearchPlan
from .models import SourceRoutingPlan
from .resource_limits import DeepResearchResourceLimits
from .resource_limits import StateBudgetLedger
from .tools.research import build_research_batch_tool
from .tools.source_registry import build_get_verified_sources_tool
from .tools.source_routing import build_lookup_source_catalog_tool
from .tools.source_tool_batching import adapt_source_tools_for_research

logger = logging.getLogger(__name__)

FILESYSTEM_TOOL_NAMES = {
    "edit_file",
    "execute",
    "grep",
    "glob",
    "ls",
    "read_file",
    "write_file",
}
ORCHESTRATOR_AGENT = "orchestrator"
PLANNER_AGENT = "planner-agent"
RESEARCHER_AGENT = "researcher-agent"
SOURCE_ROUTER_AGENT = "source-router-agent"
WRITER_AGENT = "writer-agent"
PARENT_REPORT_CONTEXT_FILES = frozenset(
    {
        "/shared/original_report.md",
        "/shared/parent_report_context.json",
    }
)


@tool
def think(thought: str) -> str:
    """Use this tool to reason through decisions, verify constraints, or plan next steps."""
    return "Thought recorded."


@dataclass(frozen=True)
class DeepResearchToolSet:
    """Tool groupings used by the deep researcher graph and subagents."""

    source_tool_names: set[str]
    tools_info: list[dict[str, str]]
    helper_tools: list[BaseTool]
    all_tools: list[BaseTool]
    research_source_tools: list[BaseTool]
    researcher_tools: list[BaseTool]
    writer_tools: list[BaseTool]


@dataclass(frozen=True)
class DeepResearchMiddlewareSet:
    """Middleware stacks used by the deep researcher graph and subagents."""

    researcher: list[Any]
    planner: list[Any]
    writer: list[Any]
    orchestrator: list[Any]


@dataclass(frozen=True)
class DeepResearchGraphContext:
    """Shared graph-build inputs used by the orchestrator and subagent specs."""

    llm_provider: LLMProvider
    state: DeepResearchAgentState
    prompts: dict[str, str]
    tools: Sequence[BaseTool]
    runtime: DeepAgentsRuntime
    tool_set: DeepResearchToolSet
    middleware_set: DeepResearchMiddlewareSet
    domain_catalog_path: str | None
    current_datetime: str
    max_research_concurrency: int
    max_researcher_model_calls: int
    resource_limits: DeepResearchResourceLimits
    enable_source_router: bool
    backend: Any
    visibility_middleware: list[Any]
    final_report_tracker: FinalReportCommitTracker
    state_budget: StateBudgetLedger

    @property
    def available_documents(self) -> list[dict[str, Any]]:
        """Return the user-uploaded documents for this run as serialized dicts."""
        return [doc.model_dump() for doc in (self.state.available_documents or [])]

    @property
    def parent_report_context_available(self) -> bool:
        return any(path in self.state.files for path in PARENT_REPORT_CONTEXT_FILES)

    def render_prompt(self, prompt_name: str, **values: Any) -> str:
        """Render a named prompt template with shared context plus any overrides."""
        prompt_values = {
            "current_datetime": self.current_datetime,
            "user_info": self.state.user_info,
            "available_documents": self.available_documents,
            "execution_enabled": self.runtime.execution_enabled,
            "skills_enabled": self.runtime.skills_enabled,
            "sandbox_workdir": self.runtime.workdir,
            "sandbox_artifact_dir": self.runtime.artifact_dir,
            **values,
        }
        return render_prompt_template(
            self.prompts[prompt_name],
            **prompt_values,
        )

    def middleware(self, base: Sequence[Any]) -> list[Any]:
        """Return the base middleware stack extended with tool-visibility middleware."""
        return [*base, *self.visibility_middleware]

    def permissions(self, agent_name: str) -> list[FilesystemPermission]:
        """Return the skill-derived filesystem permissions for an agent."""
        permissions = runtime_skill_filesystem_permissions(self.runtime, agent_name)
        if agent_name != WRITER_AGENT:
            permissions.append(
                FilesystemPermission(
                    operations=["write"],
                    paths=["/shared/**"] if self.runtime.execution_enabled else ["/**"],
                    mode="deny",
                )
            )
        return permissions

    def skill_sources(self, agent_name: str) -> list[str] | None:
        """Return the resolved skill source paths for an agent, or None."""
        return self.runtime.skill_sources_for(agent_name)


def build_deep_research_tool_set(
    tools: Sequence[BaseTool],
    *,
    source_registry_middleware: SourceRegistryMiddleware,
    max_concurrent_source_tool_calls: int,
    max_source_tool_batch_size: int,
) -> DeepResearchToolSet:
    """Build helper, researcher, writer, and source tool groupings."""
    source_tool_names = {tool.name for tool in tools}
    helper_tools = [think, build_get_verified_sources_tool(source_registry_middleware)]
    research_source_tools = adapt_source_tools_for_research(
        list(tools),
        source_tool_names=source_tool_names,
        max_concurrent_source_tool_calls=max_concurrent_source_tool_calls,
        max_batch_size=max_source_tool_batch_size,
    )
    return DeepResearchToolSet(
        source_tool_names=source_tool_names,
        tools_info=[{"name": tool.name, "description": tool.description} for tool in tools],
        helper_tools=helper_tools,
        all_tools=[*helper_tools, *tools],
        research_source_tools=research_source_tools,
        researcher_tools=[*helper_tools, *research_source_tools],
        writer_tools=list(helper_tools),
    )


def build_common_middleware(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    artifact_manager: object | None = None,
    extra_valid_tool_names: Sequence[str] = (),
) -> list[Any]:
    """Build the shared middleware stack with agent-specific valid tool names."""
    valid_tool_names = {tool.name for tool in [*tool_set.all_tools, *tool_set.researcher_tools]}
    valid_tool_names.update(FILESYSTEM_TOOL_NAMES)
    valid_tool_names.update(extra_valid_tool_names)
    middleware: list[Any] = [
        EmptyContentFixMiddleware(),
        ToolNameSanitizationMiddleware(valid_tool_names=sorted(valid_tool_names)),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        source_registry_middleware,
        ToolResultPruningMiddleware(keep_last_n=10, max_chars=2000),
        ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
    ]
    if artifact_manager is not None:
        middleware.append(ArtifactHarvestMiddleware(artifact_manager))
    return middleware


def build_orchestrator_middleware(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    enable_source_router: bool,
    research_batch_tool_name: str = "run_research_batch",
) -> list[Any]:
    """Middleware for the orchestrator.

    Unlike the shared stack, the tool-name sanitizer allowlist here is restricted
    to the tools the orchestrator is actually bound to — helper tools,
    ``run_research_batch``, and the filesystem tools — deliberately excluding the
    source tools. The orchestrator routes all source access through
    ``run_research_batch`` (which runs the researcher, where the source tools
    live), so a source-tool name emitted by the orchestrator must not be treated
    as a valid direct call.
    """
    valid_tool_names = {tool.name for tool in tool_set.helper_tools}
    valid_tool_names.add(research_batch_tool_name)
    valid_tool_names.update(FILESYSTEM_TOOL_NAMES)
    return [
        EmptyContentFixMiddleware(),
        SourceRoutingGuardMiddleware(enabled=enable_source_router, required_subagent=SOURCE_ROUTER_AGENT),
        ToolNameSanitizationMiddleware(valid_tool_names=sorted(valid_tool_names)),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        source_registry_middleware,
        ToolResultPruningMiddleware(keep_last_n=10, max_chars=2000),
        ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
    ]


def build_source_router_middleware(*, extra_valid_tool_names: Sequence[str] = ()) -> list[Any]:
    """Build minimal middleware for the source-router-agent."""
    return [
        EmptyContentFixMiddleware(),
        ToolNameSanitizationMiddleware(valid_tool_names=sorted(extra_valid_tool_names)),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
    ]


def build_deep_research_middleware_set(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    enable_source_router: bool = True,
    artifact_manager: object | None = None,
) -> DeepResearchMiddlewareSet:
    """Build researcher, writer, and orchestrator middleware stacks."""

    def common(extra_valid_tool_names: Sequence[str] = ()) -> list[Any]:
        """Build the shared middleware stack, allowing extra valid tool names."""
        return build_common_middleware(
            tool_set=tool_set,
            source_registry_middleware=source_registry_middleware,
            artifact_manager=artifact_manager,
            extra_valid_tool_names=extra_valid_tool_names,
        )

    return DeepResearchMiddlewareSet(
        researcher=common(),
        planner=common(),
        writer=common(),
        orchestrator=build_orchestrator_middleware(
            tool_set=tool_set,
            source_registry_middleware=source_registry_middleware,
            enable_source_router=enable_source_router,
        ),
    )


def runtime_visibility_middleware(runtime: DeepAgentsRuntime) -> list[Any]:
    """Hide execution tools unless a sandbox backend is configured."""
    if runtime.execution_enabled:
        return []
    return [ToolVisibilityMiddleware(hidden_tool_names={"execute"})]


def skill_filesystem_permissions(skill_sources: Sequence[str] | None) -> list[FilesystemPermission]:
    """Build permissions that expose only assigned built-in skill collections as read-only."""
    allowed_source_paths = [source.rstrip("/") for source in skill_sources or ()]
    rules = [
        FilesystemPermission(
            operations=["write"],
            paths=[f"{BUILTIN_SKILL_SOURCE}**"],
            mode="deny",
        )
    ]
    if allowed_source_paths:
        rules.append(
            FilesystemPermission(
                operations=["read"],
                paths=[BUILTIN_SKILL_SOURCE],
                mode="allow",
            )
        )
    rules.extend(
        FilesystemPermission(
            operations=["read"],
            paths=[f"{source_path}{{,/**}}"],
            mode="allow",
        )
        for source_path in allowed_source_paths
    )
    rules.append(
        FilesystemPermission(
            operations=["read"],
            paths=[f"{BUILTIN_SKILL_SOURCE}**"],
            mode="deny",
        )
    )
    return rules


def runtime_skill_filesystem_permissions(runtime: DeepAgentsRuntime, agent_name: str) -> list[FilesystemPermission]:
    """Return filesystem-tool permissions for an agent's configured skill sources."""
    if not runtime.skills_enabled:
        return []
    return skill_filesystem_permissions(runtime.skill_sources_for(agent_name))


def build_researcher_runnable(
    *,
    researcher_model: BaseChatModel,
    researcher_tools: list[BaseTool],
    researcher_middleware: list[Any],
    max_researcher_model_calls: int,
    system_prompt: str,
    skill_sources: list[str] | None = None,
    backend: Any = None,
    visibility_middleware: list[Any] | None = None,
    filesystem_permissions: list[FilesystemPermission] | None = None,
) -> Any:
    """Build the reusable single-query researcher runnable."""
    middleware: list[Any] = []
    if skill_sources:
        middleware.append(SkillsMiddleware(backend=backend, sources=skill_sources))
    middleware.extend(
        [
            FilesystemMiddleware(backend=backend, _permissions=filesystem_permissions),
            create_summarization_middleware(researcher_model, backend),
            PatchToolCallsMiddleware(),
            ModelCallLimitMiddleware(
                run_limit=max_researcher_model_calls + RESEARCHER_FINALIZATION_MODEL_CALLS,
                exit_behavior="error",
            ),
            ResearcherFinalizationMiddleware(max_model_calls=max_researcher_model_calls),
            StructuredResponseTextFallbackMiddleware(ResearchNotes),
            *researcher_middleware,
            *(visibility_middleware or []),
        ]
    )
    middleware = merge_langchain_middleware(middleware)
    return create_agent(
        model=researcher_model,
        tools=researcher_tools,
        system_prompt=system_prompt,
        middleware=middleware,
        response_format=ResearchNotes,
    )


def _subagent_spec(
    context: DeepResearchGraphContext,
    *,
    name: str,
    description: str,
    prompt_name: str,
    role: LLMRole,
    tools: Sequence[BaseTool],
    middleware: Sequence[Any],
    prompt_values: dict[str, Any] | None = None,
    response_format: Any = None,
    skills: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a deepagents subagent spec (prompt, model, tools, permissions, middleware)."""
    spec: dict[str, Any] = {
        "name": name,
        "description": description,
        "system_prompt": context.render_prompt(prompt_name, **(prompt_values or {})),
        "tools": list(tools),
        "model": context.llm_provider.get(role),
        "permissions": context.permissions(name),
        "middleware": context.middleware(middleware),
    }
    if response_format is not None:
        spec["response_format"] = response_format
    if skills is not None:
        spec["skills"] = skills
    return spec


def build_deep_research_subagents(context: DeepResearchGraphContext) -> list[dict[str, Any]]:
    """Build all DeepAgents subagent specs."""
    subagents: list[dict[str, Any]] = []
    if context.enable_source_router:
        source_catalog_tool = build_lookup_source_catalog_tool(
            context.tools,
            allowed_source_ids=context.state.data_sources,
            domain_catalog_path=context.domain_catalog_path,
        )
        subagents.append(
            _subagent_spec(
                context,
                name=SOURCE_ROUTER_AGENT,
                description=(
                    "Source router - chooses an advisory domain route and configured source set before detailed "
                    "planning"
                ),
                prompt_name="source_router",
                role=LLMRole.ROUTER,
                tools=[source_catalog_tool],
                middleware=[
                    *build_source_router_middleware(extra_valid_tool_names=[source_catalog_tool.name]),
                    FinalReportOwnershipGuardMiddleware(),
                    StateMutationGuardMiddleware(
                        writer=False,
                        sandbox_enabled=context.runtime.execution_enabled,
                    ),
                    TodoSuppressionMiddleware(),
                    StructuredResponseTextFallbackMiddleware(SourceRoutingPlan),
                    SourceRoutingPersistenceMiddleware(
                        backend=context.backend,
                        state_budget=context.state_budget,
                        resource_limits=context.resource_limits,
                    ),
                ],
                prompt_values={"clarifier_result": context.state.clarifier_result},
                response_format=SourceRoutingPlan,
            )
        )

    subagents.append(
        _subagent_spec(
            context,
            name=PLANNER_AGENT,
            description=(
                "Content-driven research planning - iteratively builds evidence-grounded answer strategies through "
                "interleaved search and planning"
            ),
            prompt_name="planner",
            role=LLMRole.PLANNER,
            tools=context.tool_set.researcher_tools,
            middleware=[
                *context.middleware_set.planner,
                FinalReportOwnershipGuardMiddleware(),
                StateMutationGuardMiddleware(
                    writer=False,
                    sandbox_enabled=context.runtime.execution_enabled,
                ),
                TodoSuppressionMiddleware(),
                StructuredResponseTextFallbackMiddleware(ResearchPlan),
                PlanPersistenceMiddleware(
                    backend=context.backend,
                    state_budget=context.state_budget,
                    resource_limits=context.resource_limits,
                ),
            ],
            prompt_values={
                "tools": context.tool_set.tools_info,
                "enable_source_router": context.enable_source_router,
                "max_research_concurrency": context.max_research_concurrency,
            },
            response_format=ResearchPlan,
        )
    )
    subagents.append(
        _subagent_spec(
            context,
            name=WRITER_AGENT,
            description=(
                "Final synthesis writer - reads the plan and research notes, then returns a cited Markdown answer "
                "in the requested output shape"
            ),
            prompt_name="writer",
            role=LLMRole.REPORT_WRITER,
            tools=context.tool_set.writer_tools,
            middleware=[
                *context.middleware_set.writer,
                StateMutationGuardMiddleware(
                    writer=True,
                    sandbox_enabled=context.runtime.execution_enabled,
                ),
                FinalReportCommitMiddleware(
                    backend=context.backend,
                    tracker=context.final_report_tracker,
                    state_budget=context.state_budget,
                    resource_limits=context.resource_limits,
                ),
                TodoSuppressionMiddleware(),
                RequiredOutputFileMiddleware(tracker=context.final_report_tracker),
            ],
            prompt_values={
                "parent_report_context_available": context.parent_report_context_available,
                "chart_skill_enabled": context.runtime.agent_has_chart_skill(WRITER_AGENT),
            },
            skills=context.skill_sources(WRITER_AGENT),
        ),
    )
    return subagents


def build_deep_research_graph(
    *,
    llm_provider: LLMProvider,
    state: DeepResearchAgentState,
    prompts: dict[str, str],
    tools: Sequence[BaseTool],
    runtime: DeepAgentsRuntime,
    tool_set: DeepResearchToolSet,
    middleware_set: DeepResearchMiddlewareSet,
    source_registry_middleware: SourceRegistryMiddleware,
    callbacks: list[Any],
    domain_catalog_path: str | None,
    max_research_concurrency: int,
    max_researcher_model_calls: int,
    final_report_tracker: FinalReportCommitTracker,
    state_budget: StateBudgetLedger | None = None,
    resource_limits: DeepResearchResourceLimits | None = None,
    enable_source_router: bool = True,
) -> Any:
    """Build the full DeepAgents graph for one deep research run."""
    # Cross-cutting middleware applied to every agent (researcher, subagents, orchestrator).
    # Agent-supplied execute timeouts are unreliable (LLMs pass milliseconds or arbitrarily
    # large values); clamp them to the configured sandbox lifetime so a single execute never
    # exceeds the provider's hard cap and silently fails every code run.
    cross_cutting_middleware = [
        FilesystemToolCallGuardMiddleware(),
        *runtime_visibility_middleware(runtime),
    ]
    execute_ceiling = runtime.execute_timeout_seconds
    if execute_ceiling:
        cross_cutting_middleware = [
            ExecuteTimeoutClampMiddleware(max_timeout_seconds=execute_ceiling),
            *cross_cutting_middleware,
        ]
    limits = resource_limits or DeepResearchResourceLimits()
    context = DeepResearchGraphContext(
        llm_provider=llm_provider,
        state=state,
        prompts=prompts,
        tools=tools,
        runtime=runtime,
        tool_set=tool_set,
        middleware_set=middleware_set,
        domain_catalog_path=domain_catalog_path,
        current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        max_research_concurrency=max_research_concurrency,
        max_researcher_model_calls=max_researcher_model_calls,
        resource_limits=limits,
        enable_source_router=enable_source_router,
        backend=runtime.backend,
        visibility_middleware=cross_cutting_middleware,
        final_report_tracker=final_report_tracker,
        state_budget=state_budget
        or StateBudgetLedger(
            limits=limits,
            files=state.files,
            sandbox_enabled=runtime.execution_enabled,
        ),
    )
    researcher_model = context.llm_provider.get(LLMRole.RESEARCHER)
    researcher_skill_sources = context.skill_sources(RESEARCHER_AGENT)
    researcher_runnable = build_researcher_runnable(
        researcher_model=researcher_model,
        researcher_tools=context.tool_set.researcher_tools,
        system_prompt=context.render_prompt(
            "researcher",
            tools=context.tool_set.tools_info,
            execution_enabled=context.runtime.execution_enabled,
        ),
        researcher_middleware=[
            *context.middleware_set.researcher,
            FinalReportOwnershipGuardMiddleware(),
            StateMutationGuardMiddleware(
                writer=False,
                sandbox_enabled=context.runtime.execution_enabled,
            ),
        ],
        max_researcher_model_calls=context.max_researcher_model_calls,
        skill_sources=researcher_skill_sources,
        backend=context.backend,
        visibility_middleware=context.visibility_middleware,
        filesystem_permissions=context.permissions(RESEARCHER_AGENT),
    )
    research_batch_tool = build_research_batch_tool(
        researcher_runnable=researcher_runnable,
        backend=context.backend,
        callbacks=callbacks,
        max_research_concurrency=max_research_concurrency,
        resource_limits=context.resource_limits,
        state_budget=context.state_budget,
        source_registry_middleware=source_registry_middleware,
    )

    orchestrator_tools = [*context.tool_set.helper_tools, research_batch_tool]
    agent = create_deep_agent(
        **deepagents_kwargs(
            dict(
                model=context.llm_provider.get(LLMRole.ORCHESTRATOR),
                name="deep_research_agent",
                tools=orchestrator_tools,
                system_prompt=context.render_prompt(
                    "orchestrator",
                    clarifier_result=context.state.clarifier_result,
                    # Advertise only the tools the orchestrator can actually call. Source
                    # tools (incl. per-user MCP tools like Google Drive) are NOT directly
                    # callable here — the orchestrator delegates all source access through
                    # run_research_batch to the researcher, which holds those tools. Listing
                    # them under "Available Tools" made the orchestrator call them directly
                    # (e.g. per_user_mcp_client__google_drive_read_file), which the runtime
                    # rejects since they aren't bound to this agent.
                    tools=[{"name": t.name, "description": t.description} for t in orchestrator_tools],
                    enable_source_router=context.enable_source_router,
                    max_research_concurrency=context.max_research_concurrency,
                    execution_enabled=context.runtime.execution_enabled,
                    parent_report_context_available=context.parent_report_context_available,
                ),
                subagents=build_deep_research_subagents(context),
                store=InMemoryStore(),
                middleware=context.middleware(
                    [
                        *context.middleware_set.orchestrator,
                        FinalReportOwnershipGuardMiddleware(),
                        StateMutationGuardMiddleware(
                            writer=False,
                            sandbox_enabled=context.runtime.execution_enabled,
                        ),
                        TodoQuotaMiddleware(resource_limits=context.resource_limits),
                        RequiredWriterDelegationMiddleware(tracker=context.final_report_tracker),
                    ]
                ),
                permissions=context.permissions(ORCHESTRATOR_AGENT),
                backend=context.backend,
            )
        )
    )
    return agent.with_config({"recursion_limit": 2000})
