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

"""Tests for deep researcher graph and middleware factory helpers."""

from unittest.mock import MagicMock
from unittest.mock import NonCallableMagicMock
from unittest.mock import patch

from deepagents.middleware.filesystem import _apply_permissions_to_ls_results
from deepagents.middleware.filesystem import _check_fs_permission
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from deep_researcher_agent.agents.deep_researcher.custom_middleware import ArtifactHarvestMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import FilesystemToolCallGuardMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import FinalReportCommitMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import FinalReportCommitTracker
from deep_researcher_agent.agents.deep_researcher.custom_middleware import FinalReportOwnershipGuardMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import RequiredOutputFileMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import RequiredWriterDelegationMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import SourceRoutingGuardMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import SourceRoutingPersistenceMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import StateMutationGuardMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import StructuredResponseTextFallbackMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import TodoQuotaMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import TodoSuppressionMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import ToolNameSanitizationMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import ToolVisibilityMiddleware
from deep_researcher_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
from deep_researcher_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSkillsConfig
from deep_researcher_agent.agents.deep_researcher.factory import DeepResearchGraphContext
from deep_researcher_agent.agents.deep_researcher.factory import build_deep_research_graph
from deep_researcher_agent.agents.deep_researcher.factory import build_deep_research_middleware_set
from deep_researcher_agent.agents.deep_researcher.factory import build_deep_research_subagents
from deep_researcher_agent.agents.deep_researcher.factory import build_deep_research_tool_set
from deep_researcher_agent.agents.deep_researcher.factory import build_researcher_runnable
from deep_researcher_agent.agents.deep_researcher.factory import runtime_visibility_middleware
from deep_researcher_agent.agents.deep_researcher.factory import skill_filesystem_permissions
from deep_researcher_agent.agents.deep_researcher.models import DeepResearchAgentState
from deep_researcher_agent.agents.deep_researcher.models import ResearchNotes
from deep_researcher_agent.agents.deep_researcher.models import ResearchPlan
from deep_researcher_agent.agents.deep_researcher.models import SourceRoutingPlan
from deep_researcher_agent.agents.deep_researcher.resource_limits import DeepResearchResourceLimits
from deep_researcher_agent.agents.deep_researcher.resource_limits import StateBudgetLedger
from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


def _llm_provider() -> LLMProvider:
    llm = MagicMock()
    provider = LLMProvider()
    provider.set_default(llm)
    provider.configure(LLMRole.ROUTER, llm)
    provider.configure(LLMRole.PLANNER, llm)
    provider.configure(LLMRole.RESEARCHER, llm)
    provider.configure(LLMRole.REPORT_WRITER, llm)
    provider.configure(LLMRole.ORCHESTRATOR, llm)
    return provider


def _prompts() -> dict[str, str]:
    return {
        "source_router": "router {{ current_datetime }}",
        "planner": "planner {% for tool in tools %}{{ tool.name }} {% endfor %}",
        "researcher": "researcher",
        "orchestrator": "orchestrator",
        "writer": "writer",
    }


def _tool_set_and_middleware() -> tuple[SourceRegistryMiddleware, object, object]:
    registry = SourceRegistryMiddleware(source_tool_names={web_search_tool.name})
    tool_set = build_deep_research_tool_set(
        [web_search_tool],
        source_registry_middleware=registry,
        max_concurrent_source_tool_calls=2,
        max_source_tool_batch_size=3,
    )
    middleware_set = build_deep_research_middleware_set(
        tool_set=tool_set,
        source_registry_middleware=registry,
    )
    return registry, tool_set, middleware_set


def _tool_names(tools) -> list[str]:
    return [tool.name for tool in tools]


def _sanitizer(middleware: list[object]) -> ToolNameSanitizationMiddleware:
    return next(item for item in middleware if isinstance(item, ToolNameSanitizationMiddleware))


def _routing_guard(middleware: list[object]) -> SourceRoutingGuardMiddleware:
    return next(item for item in middleware if isinstance(item, SourceRoutingGuardMiddleware))


def _graph_context(
    *,
    runtime: DeepAgentsRuntime | None = None,
    state: DeepResearchAgentState | None = None,
    provider: LLMProvider | None = None,
    enable_source_router: bool = True,
    final_report_tracker: FinalReportCommitTracker | None = None,
) -> DeepResearchGraphContext:
    _, tool_set, middleware_set = _tool_set_and_middleware()
    runtime = runtime or DeepAgentsRuntime()
    state = state or DeepResearchAgentState(messages=[])
    limits = DeepResearchResourceLimits()
    return DeepResearchGraphContext(
        llm_provider=provider or _llm_provider(),
        state=state,
        prompts=_prompts(),
        tools=[web_search_tool],
        runtime=runtime,
        tool_set=tool_set,
        middleware_set=middleware_set,
        domain_catalog_path=None,
        current_datetime="2026-06-03 12:00:00",
        max_research_concurrency=6,
        max_researcher_model_calls=100,
        resource_limits=limits,
        enable_source_router=enable_source_router,
        backend=runtime.backend,
        visibility_middleware=runtime_visibility_middleware(runtime),
        final_report_tracker=final_report_tracker or FinalReportCommitTracker(),
        state_budget=StateBudgetLedger(
            limits=limits,
            files=state.files,
            sandbox_enabled=runtime.execution_enabled,
        ),
    )


def test_tool_set_keeps_helper_researcher_and_writer_tools_separate():
    """Factory tool grouping keeps source tools away from writer-only helpers."""
    _, tool_set, _ = _tool_set_and_middleware()

    assert tool_set.source_tool_names == {"web_search_tool"}
    assert _tool_names(tool_set.helper_tools) == ["think", "get_verified_sources"]
    assert _tool_names(tool_set.writer_tools) == ["think", "get_verified_sources"]
    assert "web_search_tool" in _tool_names(tool_set.researcher_tools)
    assert "web_search_tool" not in _tool_names(tool_set.writer_tools)


def test_middleware_set_adds_orchestrator_batch_tool_name():
    """The orchestrator sanitizer accepts run_research_batch while shared stacks accept source tools."""
    registry, tool_set, middleware_set = _tool_set_and_middleware()

    researcher_sanitizer = _sanitizer(middleware_set.researcher)
    writer_sanitizer = _sanitizer(middleware_set.writer)
    orchestrator_sanitizer = _sanitizer(middleware_set.orchestrator)
    assert "web_search_tool" in researcher_sanitizer.valid_tool_names
    assert "edit_file" in writer_sanitizer.valid_tool_names
    assert "grep" in researcher_sanitizer.valid_tool_names
    assert "read_file" in researcher_sanitizer.valid_tool_names
    assert "write_file" in researcher_sanitizer.valid_tool_names
    assert "run_research_batch" not in researcher_sanitizer.valid_tool_names
    assert "run_research_batch" in orchestrator_sanitizer.valid_tool_names
    # The orchestrator allowlist matches the tools it is actually bound to: helper
    # tools, run_research_batch, and filesystem tools — but NOT source tools, which
    # it must reach only via run_research_batch (not call directly).
    assert "web_search_tool" not in orchestrator_sanitizer.valid_tool_names
    assert "think" in orchestrator_sanitizer.valid_tool_names
    assert "get_verified_sources" in orchestrator_sanitizer.valid_tool_names
    assert "read_file" in orchestrator_sanitizer.valid_tool_names
    assert registry in middleware_set.researcher
    assert registry in middleware_set.writer
    assert registry in middleware_set.orchestrator
    assert tool_set.writer_tools != tool_set.researcher_tools


def test_middleware_set_configures_orchestrator_source_routing_guard():
    """Only the orchestrator stack enforces the configured source-router transition."""
    registry, tool_set, _ = _tool_set_and_middleware()

    enabled = build_deep_research_middleware_set(
        tool_set=tool_set,
        source_registry_middleware=registry,
        enable_source_router=True,
    )
    disabled = build_deep_research_middleware_set(
        tool_set=tool_set,
        source_registry_middleware=registry,
        enable_source_router=False,
    )

    assert _routing_guard(enabled.orchestrator).enabled is True
    assert _routing_guard(disabled.orchestrator).enabled is False
    assert not any(isinstance(item, SourceRoutingGuardMiddleware) for item in enabled.researcher)
    assert not any(isinstance(item, SourceRoutingGuardMiddleware) for item in enabled.planner)
    assert not any(isinstance(item, SourceRoutingGuardMiddleware) for item in enabled.writer)


def test_middleware_set_wires_artifact_checkpoint_when_manager_present():
    """Artifact harvesting is attached to every execute-capable middleware stack."""
    registry = SourceRegistryMiddleware(source_tool_names={web_search_tool.name})
    tool_set = build_deep_research_tool_set(
        [web_search_tool],
        source_registry_middleware=registry,
        max_concurrent_source_tool_calls=2,
        max_source_tool_batch_size=3,
    )
    manager = MagicMock()

    middleware_set = build_deep_research_middleware_set(
        tool_set=tool_set,
        source_registry_middleware=registry,
        artifact_manager=manager,
    )

    for middleware in (middleware_set.researcher, middleware_set.planner, middleware_set.writer):
        checkpoint = next(item for item in middleware if isinstance(item, ArtifactHarvestMiddleware))
        assert checkpoint.artifact_manager is manager


def test_subagents_route_tools_and_writer_skills():
    """Source-router excludes source tools, planner receives source tools, and writer receives configured skills."""
    runtime = DeepAgentsRuntime(
        skills=DeepResearchSkillsConfig(
            agents={"writer-agent": ("synthesis",)},
        )
    )

    subagents = build_deep_research_subagents(_graph_context(runtime=runtime))

    by_name = {subagent["name"]: subagent for subagent in subagents}
    assert set(by_name) == {"source-router-agent", "planner-agent", "writer-agent"}
    assert by_name["source-router-agent"]["response_format"] is SourceRoutingPlan
    assert _tool_names(by_name["source-router-agent"]["tools"]) == ["lookup_source_catalog"]
    assert "web_search_tool" not in _tool_names(by_name["source-router-agent"]["tools"])
    assert by_name["planner-agent"]["response_format"] is ResearchPlan
    assert "web_search_tool" in _tool_names(by_name["planner-agent"]["tools"])
    assert _tool_names(by_name["writer-agent"]["tools"]) == ["think", "get_verified_sources"]
    assert by_name["writer-agent"]["skills"] == ["/skills/synthesis/"]
    writer_permissions = by_name["writer-agent"]["permissions"]
    assert (
        _check_fs_permission(writer_permissions, "read", "/skills/synthesis/long-form-report-writer/SKILL.md")
        == "allow"
    )
    assert (
        _check_fs_permission(writer_permissions, "write", "/skills/synthesis/long-form-report-writer/SKILL.md")
        == "deny"
    )
    assert _check_fs_permission(writer_permissions, "read", "/skills/research/data-table-analysis/SKILL.md") == "deny"
    assert _check_fs_permission(by_name["planner-agent"]["permissions"], "read", "/skills/synthesis/") == "deny"
    assert any(isinstance(item, ToolVisibilityMiddleware) for item in by_name["planner-agent"]["middleware"])
    assert any(isinstance(item, ToolVisibilityMiddleware) for item in by_name["writer-agent"]["middleware"])
    assert any(isinstance(item, TodoSuppressionMiddleware) for item in by_name["source-router-agent"]["middleware"])
    assert any(
        isinstance(item, StructuredResponseTextFallbackMiddleware)
        for item in by_name["source-router-agent"]["middleware"]
    )
    assert any(
        isinstance(item, SourceRoutingPersistenceMiddleware) for item in by_name["source-router-agent"]["middleware"]
    )
    assert any(isinstance(item, TodoSuppressionMiddleware) for item in by_name["planner-agent"]["middleware"])
    assert any(
        isinstance(item, StructuredResponseTextFallbackMiddleware) for item in by_name["planner-agent"]["middleware"]
    )
    assert any(isinstance(item, TodoSuppressionMiddleware) for item in by_name["writer-agent"]["middleware"])
    assert any(isinstance(item, RequiredOutputFileMiddleware) for item in by_name["writer-agent"]["middleware"])


def test_subagents_share_one_run_local_tracker_and_reserve_output_for_writer():
    """All non-writers are guarded while writer commit and completion share one tracker."""
    tracker = FinalReportCommitTracker()
    subagents = build_deep_research_subagents(_graph_context(final_report_tracker=tracker))
    by_name = {subagent["name"]: subagent for subagent in subagents}

    for name in ("source-router-agent", "planner-agent"):
        assert any(isinstance(item, FinalReportOwnershipGuardMiddleware) for item in by_name[name]["middleware"])
        assert any(isinstance(item, StateMutationGuardMiddleware) for item in by_name[name]["middleware"])
        assert not any(isinstance(item, FinalReportCommitMiddleware) for item in by_name[name]["middleware"])
        assert not any(isinstance(item, RequiredOutputFileMiddleware) for item in by_name[name]["middleware"])

    writer_commit = next(
        item for item in by_name["writer-agent"]["middleware"] if isinstance(item, FinalReportCommitMiddleware)
    )
    writer_completion = next(
        item for item in by_name["writer-agent"]["middleware"] if isinstance(item, RequiredOutputFileMiddleware)
    )
    assert writer_commit.tracker is tracker
    assert writer_completion.tracker is tracker
    assert not any(
        isinstance(item, FinalReportOwnershipGuardMiddleware) for item in by_name["writer-agent"]["middleware"]
    )


def test_skill_filesystem_permissions_filter_unassigned_skill_collections():
    """Filesystem tools only expose skill collections assigned to the current agent."""
    permissions = skill_filesystem_permissions(["/skills/synthesis/"])
    entries = [
        {"path": "/skills/research/", "is_dir": True, "size": 0, "modified_at": ""},
        {"path": "/skills/synthesis/", "is_dir": True, "size": 0, "modified_at": ""},
    ]

    assert _check_fs_permission(permissions, "read", "/skills/") == "allow"
    assert _check_fs_permission(permissions, "read", "/skills/synthesis/prediction-report-writer/SKILL.md") == "allow"
    assert _check_fs_permission(permissions, "write", "/skills/synthesis/prediction-report-writer/SKILL.md") == "deny"
    assert _check_fs_permission(permissions, "read", "/skills/research/data-table-analysis/SKILL.md") == "deny"
    assert _apply_permissions_to_ls_results(permissions, entries) == ["/skills/synthesis/"]


def test_non_writer_permissions_make_default_state_backend_read_only():
    """Without a sandbox, every virtual path is state-backed and read-only to model tools."""
    permissions = _graph_context().permissions("researcher-agent")

    assert _check_fs_permission(permissions, "read", "/shared/plan.json") == "allow"
    assert _check_fs_permission(permissions, "write", "/shared/research_note.json") == "deny"
    assert _check_fs_permission(permissions, "write", "/workspace/analysis.py") == "deny"


def test_non_writer_permissions_allow_sandbox_workspace_but_not_shared_state():
    """With a sandbox, non-writers may use workspace files but cannot mutate StateBackend."""
    runtime = MagicMock(spec=DeepAgentsRuntime)
    runtime.execution_enabled = True
    runtime.skills_enabled = False
    permissions = _graph_context(runtime=runtime).permissions("planner-agent")

    assert _check_fs_permission(permissions, "write", "/shared/plan.json") == "deny"
    assert _check_fs_permission(permissions, "write", "/workspace/analysis.py") == "allow"


def test_graph_uses_researcher_config_key_for_researcher_skills():
    """The researcher runnable uses the public `researcher` skill config key."""
    registry, tool_set, middleware_set = _tool_set_and_middleware()
    runtime = DeepAgentsRuntime(skills=DeepResearchSkillsConfig(agents={"researcher-agent": ("research",)}))
    fake_graph = MagicMock()
    fake_graph.with_config.return_value = fake_graph

    with (
        patch("deep_researcher_agent.agents.deep_researcher.factory.create_deep_agent", return_value=fake_graph),
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_agent", return_value=MagicMock()
        ) as create_researcher,
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=MagicMock(),
        ),
    ):
        build_deep_research_graph(
            llm_provider=_llm_provider(),
            state=DeepResearchAgentState(messages=[]),
            prompts=_prompts(),
            tools=[web_search_tool],
            runtime=runtime,
            tool_set=tool_set,
            middleware_set=middleware_set,
            source_registry_middleware=registry,
            callbacks=[],
            domain_catalog_path=None,
            max_research_concurrency=6,
            max_researcher_model_calls=100,
            final_report_tracker=FinalReportCommitTracker(),
        )

    researcher_middleware = create_researcher.call_args.kwargs["middleware"]
    skills_middleware = [item for item in researcher_middleware if item.__class__.__name__ == "SkillsMiddleware"]
    assert skills_middleware[0].sources == ["/skills/research/"]


def test_graph_wires_filesystem_tool_call_guard_cross_cutting():
    """Every agent receives the filesystem path guard, even when execution is hidden."""
    registry, tool_set, middleware_set = _tool_set_and_middleware()
    runtime = DeepAgentsRuntime()
    fake_graph = MagicMock()
    fake_graph.with_config.return_value = fake_graph

    with (
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_deep_agent", return_value=fake_graph
        ) as create_graph,
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_agent",
            return_value=MagicMock(),
        ) as create_researcher,
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=MagicMock(),
        ),
    ):
        build_deep_research_graph(
            llm_provider=_llm_provider(),
            state=DeepResearchAgentState(messages=[]),
            prompts=_prompts(),
            tools=[web_search_tool],
            runtime=runtime,
            tool_set=tool_set,
            middleware_set=middleware_set,
            source_registry_middleware=registry,
            callbacks=[],
            domain_catalog_path=None,
            max_research_concurrency=6,
            max_researcher_model_calls=100,
            final_report_tracker=FinalReportCommitTracker(),
        )

    assert any(
        isinstance(middleware, FilesystemToolCallGuardMiddleware)
        for middleware in create_graph.call_args.kwargs["middleware"]
    )
    assert any(
        isinstance(middleware, FinalReportOwnershipGuardMiddleware)
        for middleware in create_graph.call_args.kwargs["middleware"]
    )
    assert any(
        isinstance(middleware, TodoQuotaMiddleware) for middleware in create_graph.call_args.kwargs["middleware"]
    )
    assert any(
        isinstance(middleware, RequiredWriterDelegationMiddleware)
        for middleware in create_graph.call_args.kwargs["middleware"]
    )
    assert any(
        isinstance(middleware, FinalReportOwnershipGuardMiddleware)
        for middleware in create_researcher.call_args.kwargs["middleware"]
    )
    for middleware_stack in (
        create_graph.call_args.kwargs["middleware"],
        create_researcher.call_args.kwargs["middleware"],
    ):
        assert not any(isinstance(middleware, FinalReportCommitMiddleware) for middleware in middleware_stack)
        assert not any(isinstance(middleware, RequiredOutputFileMiddleware) for middleware in middleware_stack)


def test_graph_default_limits_are_shared_with_state_budget_ledger():
    """The graph and its ledger must enforce the exact same immutable limits instance."""
    registry, tool_set, middleware_set = _tool_set_and_middleware()
    runtime = DeepAgentsRuntime()
    fake_graph = MagicMock()
    fake_graph.with_config.return_value = fake_graph
    fake_batch_tool = MagicMock()
    fake_batch_tool.name = "run_research_batch"
    fake_batch_tool.description = "Run research."

    with (
        patch("deep_researcher_agent.agents.deep_researcher.factory.create_deep_agent", return_value=fake_graph),
        patch("deep_researcher_agent.agents.deep_researcher.factory.create_agent", return_value=MagicMock()),
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=MagicMock(),
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.build_research_batch_tool",
            return_value=fake_batch_tool,
        ) as build_batch_tool,
    ):
        build_deep_research_graph(
            llm_provider=_llm_provider(),
            state=DeepResearchAgentState(messages=[]),
            prompts=_prompts(),
            tools=[web_search_tool],
            runtime=runtime,
            tool_set=tool_set,
            middleware_set=middleware_set,
            source_registry_middleware=registry,
            callbacks=[],
            domain_catalog_path=None,
            max_research_concurrency=6,
            max_researcher_model_calls=100,
            final_report_tracker=FinalReportCommitTracker(),
        )

    limits = build_batch_tool.call_args.kwargs["resource_limits"]
    state_budget = build_batch_tool.call_args.kwargs["state_budget"]
    assert state_budget._limits is limits


def test_subagents_can_disable_source_router():
    """The source-router subagent can be omitted without changing the rest of the workflow."""
    provider = _llm_provider()
    provider.get = MagicMock(wraps=provider.get)

    subagents = build_deep_research_subagents(
        _graph_context(provider=provider, enable_source_router=False),
    )

    by_name = {subagent["name"]: subagent for subagent in subagents}
    assert set(by_name) == {"planner-agent", "writer-agent"}
    assert by_name["planner-agent"]["response_format"] is ResearchPlan
    assert "web_search_tool" in _tool_names(by_name["planner-agent"]["tools"])
    assert _tool_names(by_name["writer-agent"]["tools"]) == ["think", "get_verified_sources"]
    requested_roles = [args[0] for args, _kwargs in provider.get.call_args_list]
    assert LLMRole.ROUTER not in requested_roles
    assert LLMRole.EVIDENCE_JUDGE not in requested_roles


def test_researcher_runnable_uses_rendered_prompt_and_runtime_middleware():
    """Researcher runnable construction stays behavior-compatible but has a smaller interface."""

    class FakeSummarizationMiddleware(AgentMiddleware):
        pass

    researcher_agent = MagicMock()
    researcher_model = MagicMock()
    shared_middleware = [MagicMock(name="shared_middleware")]
    backend = NonCallableMagicMock()

    with (
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=FakeSummarizationMiddleware(),
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.factory.create_agent",
            return_value=researcher_agent,
        ) as create,
    ):
        result = build_researcher_runnable(
            researcher_model=researcher_model,
            researcher_tools=[web_search_tool],
            system_prompt="rendered researcher prompt",
            researcher_middleware=shared_middleware,
            max_researcher_model_calls=100,
            skill_sources=["/skills/research/"],
            backend=backend,
            visibility_middleware=[ToolVisibilityMiddleware(hidden_tool_names={"execute"})],
        )

    kwargs = create.call_args.kwargs
    middleware_names = [item.__class__.__name__ for item in kwargs["middleware"]]
    assert result is researcher_agent
    assert kwargs["model"] is researcher_model
    assert kwargs["tools"] == [web_search_tool]
    assert kwargs["response_format"] is ResearchNotes
    assert "TodoListMiddleware" not in middleware_names
    assert "SkillsMiddleware" in middleware_names
    assert "FilesystemMiddleware" in middleware_names
    assert "FakeSummarizationMiddleware" in middleware_names
    assert "PatchToolCallsMiddleware" in middleware_names
    assert "StructuredResponseTextFallbackMiddleware" in middleware_names
    assert "ToolVisibilityMiddleware" in middleware_names
    assert kwargs["middleware"][-2] is shared_middleware[0]
    assert middleware_names[0] == "NemoRelayMiddleware"
