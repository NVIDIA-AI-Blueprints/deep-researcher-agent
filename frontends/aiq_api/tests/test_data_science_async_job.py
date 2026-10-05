# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Async-job support for the data-science agent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _RecordingDataScienceAgent:
    """Stand-in that captures the kwargs the job runner constructs it with."""

    def __init__(
        self,
        *,
        llm,
        tools,
        recursion_limit: int = 64,
        callbacks=(),
        interaction_mode: str = "interactive",
        response_mode: str = "standard",
        gsf_catalog_call_limit: int | None = None,
        gsf_text_to_sql_call_limit: int | None = None,
        gsf_cache_repeated_calls: bool = True,
        python_call_limit: int | None = None,
        finalization_model_call_limit: int | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.recursion_limit = recursion_limit
        self.callbacks = callbacks
        self.interaction_mode = interaction_mode
        self.response_mode = response_mode
        self.gsf_catalog_call_limit = gsf_catalog_call_limit
        self.gsf_text_to_sql_call_limit = gsf_text_to_sql_call_limit
        self.gsf_cache_repeated_calls = gsf_cache_repeated_calls
        self.python_call_limit = python_call_limit
        self.finalization_model_call_limit = finalization_model_call_limit


def _ds_config(**overrides):
    from aiq_agent.agents.data_science.register import DataScienceAgentConfig

    return DataScienceAgentConfig(llm="data_science_llm", **overrides)


def _create(fn_config, agent_cls=_RecordingDataScienceAgent):
    from aiq_api.jobs.runner import _create_agent_instance

    return _create_agent_instance(
        agent_cls=agent_cls,
        llm_provider=SimpleNamespace(),
        llm="resolved-llm",
        tools=["tool-a"],
        fn_config=fn_config,
        callbacks=["callback"],
        job_id="job-1",
    )


class TestRegistry:
    def test_data_science_is_registered_and_public(self) -> None:
        from aiq_api.registry import AGENT_REGISTRY

        entry = AGENT_REGISTRY["data_science"]

        assert entry.class_path == "aiq_agent.agents.data_science.agent.DataScienceAgent"
        assert entry.config_name == "data_science_agent"
        assert entry.public is True

    def test_registered_class_path_resolves(self) -> None:
        from aiq_agent.agents.data_science.agent import DataScienceAgent
        from aiq_api.jobs.runner import _load_agent_class
        from aiq_api.registry import AGENT_REGISTRY

        assert _load_agent_class(AGENT_REGISTRY["data_science"].class_path) is DataScienceAgent

    def test_runner_can_discover_the_state_class(self) -> None:
        """``_run_agent`` builds worker state from this lookup; a miss silently
        degrades the job to an untyped dict state."""
        from aiq_agent.agents.data_science.agent import DataScienceAgent
        from aiq_agent.agents.data_science.models import DataScienceAgentState
        from aiq_api.jobs.runner import _get_agent_state_class

        agent = DataScienceAgent.__new__(DataScienceAgent)

        state_cls = _get_agent_state_class(agent)

        assert state_cls is DataScienceAgentState
        assert "data_sources" in state_cls.model_fields


class TestAgentConstruction:
    def test_config_fields_reach_the_agent(self) -> None:
        agent = _create(
            _ds_config(
                recursion_limit=32,
                response_mode="fdabench_choice",
                gsf_catalog_call_limit=2,
                gsf_text_to_sql_call_limit=6,
                gsf_cache_repeated_calls=False,
                python_call_limit=8,
                finalization_model_call_limit=18,
            )
        )

        assert agent.llm == "resolved-llm"
        assert agent.tools == ["tool-a"]
        assert agent.callbacks == ["callback"]
        assert agent.recursion_limit == 32
        assert agent.response_mode == "fdabench_choice"
        assert agent.gsf_catalog_call_limit == 2
        assert agent.gsf_text_to_sql_call_limit == 6
        assert agent.gsf_cache_repeated_calls is False
        assert agent.python_call_limit == 8
        assert agent.finalization_model_call_limit == 18

    @pytest.mark.parametrize("configured_mode", ["interactive", "headless"])
    def test_async_jobs_always_run_headless(self, configured_mode: str) -> None:
        """A Dask worker has no channel back to the user, so a clarification
        request would strand the job in a terminal question."""
        agent = _create(_ds_config(interaction_mode=configured_mode))

        assert agent.interaction_mode == "headless"

    def test_real_agent_signature_matches_the_runner_branch(self) -> None:
        """Guards against the branch silently falling through to the generic
        ``llm + tools`` fallback, which would drop every configured limit."""
        from aiq_agent.agents.data_science.agent import DataScienceAgent
        from aiq_api.jobs.runner import _DATA_SCIENCE_AGENT_KWARGS
        from aiq_api.jobs.runner import _constructor_accepts_explicit_kwargs

        assert _constructor_accepts_explicit_kwargs(DataScienceAgent, _DATA_SCIENCE_AGENT_KWARGS)

    def test_other_agent_configs_are_unaffected(self) -> None:
        from aiq_agent.agents.shallow_researcher.register import ShallowResearchAgentConfig

        captured = {}

        class _Shallow:
            def __init__(self, *, llm_provider, tools, max_tool_iterations, enforce_citations, callbacks):
                captured["max_tool_iterations"] = max_tool_iterations

        _create(ShallowResearchAgentConfig(llm="x", max_tool_iterations=3), agent_cls=_Shallow)

        assert captured["max_tool_iterations"] == 3


class TestSandboxDetection:
    """The submit-path sandbox cap must see sandboxes owned by a tool.

    ``data_science_agent`` has no ``sandbox`` field: the ``stateful_python``
    tool holds the sandbox ``FunctionRef``.
    """

    @staticmethod
    def _builder(configs: dict):
        def _get_function_config(name: str):
            if name not in configs:
                raise ValueError(f"Function `{name}` not found")
            return configs[name]

        return SimpleNamespace(get_function_config=_get_function_config)

    def test_detects_sandbox_behind_a_tool_function_ref(self) -> None:
        from aiq_api.routes import jobs as jobs_module

        builder = self._builder(
            {
                "data_science_agent": SimpleNamespace(tools=["gsf", "python"], exclude_tools=[]),
                "python": SimpleNamespace(sandbox="ds_python_sandbox"),
                "ds_python_sandbox": SimpleNamespace(provider="openshell"),
                "gsf": SimpleNamespace(),
            }
        )

        assert jobs_module._agent_uses_sandbox(builder, "data_science_agent") is True

    def test_excluded_tool_does_not_enable_the_cap(self) -> None:
        from aiq_api.routes import jobs as jobs_module

        builder = self._builder(
            {
                "data_science_agent": SimpleNamespace(tools=["gsf", "python"], exclude_tools=["python"]),
                "python": SimpleNamespace(sandbox="ds_python_sandbox"),
                "ds_python_sandbox": SimpleNamespace(provider="openshell"),
                "gsf": SimpleNamespace(),
            }
        )

        assert jobs_module._agent_uses_sandbox(builder, "data_science_agent") is False

    def test_no_sandbox_tool_is_still_false(self) -> None:
        from aiq_api.routes import jobs as jobs_module

        builder = self._builder(
            {
                "data_science_agent": SimpleNamespace(tools=["gsf"], exclude_tools=[]),
                "gsf": SimpleNamespace(),
            }
        )

        assert jobs_module._agent_uses_sandbox(builder, "data_science_agent") is False

    def test_unresolvable_sandbox_ref_is_false(self) -> None:
        from aiq_api.routes import jobs as jobs_module

        builder = self._builder(
            {
                "data_science_agent": SimpleNamespace(tools=["python"], exclude_tools=[]),
                "python": SimpleNamespace(sandbox="missing_sandbox"),
            }
        )

        assert jobs_module._agent_uses_sandbox(builder, "data_science_agent") is False


@pytest.mark.asyncio
async def test_list_agents_exposes_data_science_when_configured() -> None:
    import aiq_api.routes.jobs as jobs_routes

    def _get_function_config(name: str):
        if name != "data_science_agent":
            raise ValueError(f"Function `{name}` not found")
        return SimpleNamespace()

    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        builder=SimpleNamespace(get_function_config=_get_function_config),
        worker=SimpleNamespace(_dask_available=False, _job_store=None),
    )

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/agents")

    assert response.status_code == 200
    assert [agent["agent_type"] for agent in response.json()["agents"]] == ["data_science"]


class TestDatabaseScopePropagation:
    """A router-resolved database scope must survive into the worker.

    A request carrying ``database_name`` always takes the Hybrid path, so this is
    exactly the path that must not lose its scope: rebuilding agent state without
    it silently queries the agent's configured default instead.
    """

    class _ScopedAgent:
        """Minimal state-based agent, matching the DS agent's run signature."""

        def __init__(self):
            self.seen = None

        async def run(self, state):
            self.seen = state
            return state

    @staticmethod
    def _state_cls():
        from aiq_agent.agents.data_science.models import DataScienceAgentState

        return DataScienceAgentState

    @pytest.mark.asyncio
    async def test_database_name_reaches_agent_state(self, monkeypatch) -> None:
        from aiq_api.jobs import runner as runner_module

        agent = self._ScopedAgent()
        monkeypatch.setattr(runner_module, "_get_agent_state_class", lambda _a: self._state_cls())
        monkeypatch.setattr(
            runner_module,
            "run_with_cancellation",
            lambda coro, _monitor, event_store=None: coro,
        )

        result = await runner_module._run_agent(
            agent=agent,
            input_text="Which state has the most orders?",
            monitor=SimpleNamespace(),
            database_name="northwind",
        )

        assert result.database_name == "northwind"
        assert agent.seen.database_name == "northwind"

    @pytest.mark.asyncio
    async def test_unscoped_request_leaves_database_name_unset(self, monkeypatch) -> None:
        from aiq_api.jobs import runner as runner_module

        agent = self._ScopedAgent()
        monkeypatch.setattr(runner_module, "_get_agent_state_class", lambda _a: self._state_cls())
        monkeypatch.setattr(
            runner_module,
            "run_with_cancellation",
            lambda coro, _monitor, event_store=None: coro,
        )

        result = await runner_module._run_agent(
            agent=agent,
            input_text="Which state has the most orders?",
            monitor=SimpleNamespace(),
        )

        assert result.database_name is None

    def test_submit_agent_job_accepts_a_database_scope(self) -> None:
        """Guards the submission boundary: a missing parameter would be a silent drop."""
        import inspect

        from aiq_api.jobs.submit import submit_agent_job

        assert "database_name" in inspect.signature(submit_agent_job).parameters
