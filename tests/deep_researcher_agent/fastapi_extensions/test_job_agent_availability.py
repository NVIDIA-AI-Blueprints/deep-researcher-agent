# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_researcher_agent.agents.deep_researcher.register import DeepResearchAgentConfig
from deep_researcher_agent.auth import Principal
from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import reset_registry
from deep_researcher_api.registry import AgentConfig


@pytest.fixture(autouse=True)
def data_source_registry():
    reset_registry()
    populate_from_config(
        [
            {
                "id": "web_search",
                "name": "Web Search",
                "description": "Search the web.",
                "tools": ["web_search_tool", "advanced_web_search_tool"],
            }
        ]
    )
    yield
    reset_registry()


async def _build_submit_app(monkeypatch, builder, agent_config):
    import deep_researcher_agent.auth
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs import access
    from deep_researcher_api.jobs import event_store
    from deep_researcher_api.jobs import submit

    submitted_job = AsyncMock(return_value="job-1")
    require_principal = MagicMock(return_value=Principal(type="jwt", sub="user-1", email="user@example.com"))
    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_validate_artifact_store", MagicMock())
    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", AsyncMock())
    monkeypatch.setattr(jobs_routes, "_is_readable_regular_file", MagicMock(return_value=True))
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", AsyncMock())
    monkeypatch.setattr(jobs_routes, "_probe_async_job_readiness", AsyncMock(return_value=None))
    monkeypatch.setattr(jobs_routes, "get_agent_config", lambda _agent_type: agent_config)
    monkeypatch.setattr(jobs_routes, "require_verified_principal", require_principal)
    monkeypatch.setattr(deep_researcher_agent.auth, "get_auth_token", lambda: "token-1")
    monkeypatch.setattr(access, "ensure_job_access_table", MagicMock())
    monkeypatch.setattr(event_store.EventStore, "_ensure_table_exists", MagicMock())
    monkeypatch.setattr(submit, "submit_agent_job", submitted_job)

    worker = SimpleNamespace(
        _dask_available=True,
        _job_store=MagicMock(),
        _scheduler_address="tcp://localhost:8786",
        _db_url="sqlite:///./test.db",
        _config_file_path="config.yml",
        _log_level=20,
        _use_dask_threads=False,
        _front_end_config=SimpleNamespace(expiry_seconds=86400),
    )
    app = FastAPI()
    await jobs_routes.register_job_routes(app, builder, worker)
    return app, submitted_job, require_principal


@pytest.mark.asyncio
async def test_submit_accepts_inherited_source_when_partial_exclusion_leaves_tool(monkeypatch):
    builder = MagicMock()
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool"],
    )
    tools_by_name = {
        "web_search_tool": SimpleNamespace(name="web_search_tool"),
        "advanced_web_search_tool": SimpleNamespace(name="advanced_web_search_tool"),
    }

    async def _get_tools(*, tool_names, wrapper_type):  # noqa: ARG001 - mirrors the builder API
        return [tools_by_name[name] for name in tool_names]

    builder.get_tools = AsyncMock(side_effect=_get_tools)
    agent_config = AgentConfig(
        class_path="deep_researcher_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        config_name="deep_research_agent",
        description="Deep",
    )
    app, submitted_job, _require_principal = await _build_submit_app(monkeypatch, builder, agent_config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["data_sources"] == ["web_search"]


@pytest.mark.asyncio
async def test_submit_rejects_registered_but_unconfigured_agent_before_route_auth(monkeypatch):
    builder = MagicMock()
    builder.get_function_config.side_effect = ValueError("Function `shallow_research_agent` not found")
    builder.get_tools = AsyncMock()
    agent_config = AgentConfig(
        class_path="deep_researcher_agent.agents.shallow_researcher.agent.ShallowResearcherAgent",
        config_name="shallow_research_agent",
        description="Shallow",
    )
    app, submitted_job, require_principal = await _build_submit_app(monkeypatch, builder, agent_config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "shallow_researcher", "input": "query"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "agent_not_configured",
        "message": "Agent 'shallow_researcher' is not configured in the active workflow",
        "agent_type": "shallow_researcher",
        "config_name": "shallow_research_agent",
    }
    require_principal.assert_not_called()
    builder.get_tools.assert_not_awaited()
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_agents_excludes_public_agent_missing_from_active_workflow(monkeypatch):
    import deep_researcher_api.routes.jobs as jobs_routes

    monkeypatch.setattr(
        jobs_routes,
        "AGENT_REGISTRY",
        {
            "deep_researcher": AgentConfig(
                class_path="deep_researcher_agent.agents.deep_researcher.agent.DeepResearcherAgent",
                config_name="deep_research_agent",
                description="Deep",
            ),
            "shallow_researcher": AgentConfig(
                class_path="deep_researcher_agent.agents.shallow_researcher.agent.ShallowResearcherAgent",
                config_name="shallow_research_agent",
                description="Shallow",
            ),
        },
    )

    def _get_function_config(config_name):
        if config_name == "shallow_research_agent":
            raise ValueError("Function `shallow_research_agent` not found")
        return SimpleNamespace()

    builder = SimpleNamespace(get_function_config=_get_function_config)
    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        builder=builder,
        worker=SimpleNamespace(_dask_available=False, _job_store=None),
    )

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/agents")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [
            {
                "agent_type": "deep_researcher",
                "description": "Deep",
            }
        ]
    }
