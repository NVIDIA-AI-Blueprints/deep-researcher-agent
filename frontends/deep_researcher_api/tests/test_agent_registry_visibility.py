# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _builder_with_configs(*config_names: str) -> SimpleNamespace:
    configured = set(config_names)

    def _get_function_config(config_name: str):
        if config_name not in configured:
            raise ValueError(f"Function `{config_name}` not found")
        return SimpleNamespace()

    return SimpleNamespace(get_function_config=_get_function_config)


def test_agent_config_defaults_to_public():
    from deep_researcher_api.registry import AgentConfig

    config = AgentConfig(
        class_path="deep_researcher_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        config_name="deep_research_agent",
    )

    assert config.public is True


def test_register_agent_can_mark_internal(monkeypatch):
    from deep_researcher_api import registry

    monkeypatch.setattr(registry, "AGENT_REGISTRY", {})

    registry.register_agent(
        agent_type="report_rewriter",
        class_path="deep_researcher_agent.agents.report_rewriter.agent.ReportRewriterAgent",
        config_name="deep_research_agent",
        public=False,
    )

    assert registry.AGENT_REGISTRY["report_rewriter"].public is False


def test_default_registry_contains_internal_report_rewriter():
    from deep_researcher_api.registry import AGENT_REGISTRY

    assert AGENT_REGISTRY["report_rewriter"].class_path == (
        "deep_researcher_agent.agents.report_rewriter.agent.ReportRewriterAgent"
    )
    assert AGENT_REGISTRY["report_rewriter"].config_name == "deep_research_agent"
    assert AGENT_REGISTRY["report_rewriter"].public is False


@pytest.mark.asyncio
async def test_list_agents_includes_only_configured_public_agents(monkeypatch):
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.registry import AgentConfig

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
            "report_rewriter": AgentConfig(
                class_path="deep_researcher_agent.agents.report_rewriter.agent.ReportRewriterAgent",
                config_name="deep_research_agent",
                description="Internal",
                public=False,
            ),
        },
    )

    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        builder=_builder_with_configs("deep_research_agent"),
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


@pytest.mark.asyncio
async def test_list_agents_includes_configured_shallow_researcher(monkeypatch):
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.registry import AgentConfig

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
            "report_rewriter": AgentConfig(
                class_path="deep_researcher_agent.agents.report_rewriter.agent.ReportRewriterAgent",
                config_name="deep_research_agent",
                description="Internal",
                public=False,
            ),
        },
    )

    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        builder=_builder_with_configs("deep_research_agent", "shallow_research_agent"),
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
            },
            {
                "agent_type": "shallow_researcher",
                "description": "Shallow",
            },
        ]
    }
