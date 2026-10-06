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

"""Tests for async job submit data source targeting."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deep_researcher_agent.agents.deep_researcher.register import DeepResearchAgentConfig
from deep_researcher_agent.agents.shallow_researcher.register import ShallowResearchAgentConfig
from deep_researcher_agent.auth import Principal
from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import reset_registry
from deep_researcher_api.registry import AgentConfig


@pytest.fixture(autouse=True)
def data_source_registry():
    """Provide deterministic data sources for submit route validation."""
    reset_registry()
    populate_from_config(
        [
            {
                "id": "web_search",
                "name": "Web Search",
                "description": "Search the web.",
                "tools": ["web_search_tool", "advanced_web_search_tool"],
            },
            {
                "id": "knowledge_layer",
                "name": "Knowledge Base",
                "description": "Search uploaded documents.",
                "tools": ["knowledge_search_tool"],
            },
        ]
    )
    yield
    reset_registry()


@pytest.fixture
async def submit_app(monkeypatch, tmp_path):
    """Build a minimal app with async submit routes and patched side effects."""
    import deep_researcher_agent.auth
    import deep_researcher_api.routes.jobs as jobs_routes

    submitted_job = AsyncMock(return_value="job-1")
    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_validate_artifact_store", MagicMock())

    agent_config = AgentConfig(
        class_path="deep_researcher_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        config_name="deep_research_agent",
        description="Test deep researcher",
    )
    monkeypatch.setattr(jobs_routes, "get_agent_config", lambda _agent_type: agent_config)

    async def _no_op_reaper(*_args, **_kwargs):
        return None

    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", _no_op_reaper)
    monkeypatch.setattr(deep_researcher_agent.auth, "get_auth_token", lambda: "token-1")

    from deep_researcher_api.jobs import access
    from deep_researcher_api.jobs import admission
    from deep_researcher_api.jobs import event_store
    from deep_researcher_api.jobs import submit

    monkeypatch.setattr(access, "ensure_job_access_table", MagicMock())
    monkeypatch.setattr(access, "validate_job_access_table", MagicMock())
    monkeypatch.setattr(admission, "ensure_deep_research_admission_table", MagicMock())
    monkeypatch.setattr(admission, "validate_deep_research_admission_table", MagicMock())
    monkeypatch.setattr(
        jobs_routes,
        "require_verified_principal",
        lambda: Principal(type="jwt", sub="user-1", email="user@example.com"),
    )
    monkeypatch.setattr(event_store.EventStore, "_ensure_table_exists", MagicMock())
    monkeypatch.setattr(submit, "submit_agent_job", submitted_job)

    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    job_store = MagicMock()
    job_store.get_job = AsyncMock(return_value=None)
    job_store.dask_client.scheduler_info.return_value = {"workers": {}}
    monkeypatch.setattr(
        jobs_routes,
        "_table_names",
        AsyncMock(return_value=set(jobs_routes._REQUIRED_ASYNC_JOB_TABLES)),
    )

    worker = SimpleNamespace(
        _dask_available=True,
        _job_store=job_store,
        _scheduler_address="tcp://localhost:8786",
        _db_url=db_url,
        _config_file_path=str(config_path),
        _log_level=20,
        _use_dask_threads=False,
        _front_end_config=SimpleNamespace(expiry_seconds=86400),
    )

    web_tool = SimpleNamespace(name="web_search_tool")
    advanced_web_tool = SimpleNamespace(name="advanced_web_search_tool")
    knowledge_tool = SimpleNamespace(name="knowledge_search_tool")
    # Map tool names to their LangChain-wrapper stand-ins. The mock get_tools
    # below filters by tool_names so passing [] reproduces the typed-config bug
    # while inherited registry refs resolve to their real wrapper names.
    tools_by_name: dict[str, SimpleNamespace] = {
        "web_search_tool": web_tool,
        "advanced_web_search_tool": advanced_web_tool,
        "knowledge_search_tool": knowledge_tool,
    }

    async def _filtered_get_tools(*, tool_names, wrapper_type):  # noqa: ARG001 - mirror real signature
        return [tools_by_name[name] for name in tool_names if name in tools_by_name]

    builder = MagicMock()
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool"],
    )
    builder.get_tools = AsyncMock(side_effect=_filtered_get_tools)

    app = FastAPI()
    await jobs_routes.register_job_routes(app, builder, worker)
    return app, submitted_job, builder


@pytest.mark.asyncio
async def test_route_registration_validates_artifact_store(submit_app, tmp_path):
    import deep_researcher_api.routes.jobs as jobs_routes

    _app, _submitted_job, _builder = submit_app

    jobs_routes._validate_artifact_store.assert_called_once_with(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}")


@pytest.mark.asyncio
async def test_route_registration_initializes_admission_schema(submit_app, tmp_path):
    from deep_researcher_api.jobs import admission

    _app, _submitted_job, _builder = submit_app

    admission.ensure_deep_research_admission_table.assert_called_once_with(
        f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    )


def test_artifact_store_validation_propagates_failure(monkeypatch):
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_agent.agents.deep_researcher.sandbox import artifacts

    store = MagicMock()
    store.validate.side_effect = RuntimeError("storage unavailable")
    monkeypatch.setattr(artifacts, "build_artifact_store", MagicMock(return_value=store))

    with pytest.raises(RuntimeError, match="storage unavailable"):
        jobs_routes._validate_artifact_store("sqlite:///./test.db")


@pytest.mark.asyncio
async def test_submit_job_accepts_source_when_empty_tools_inherit_and_partial_exclusion_leaves_tool(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    submitted_job.assert_awaited_once()
    assert submitted_job.await_args.kwargs["data_sources"] == ["web_search"]
    builder.get_tools.assert_awaited_once()
    _, kwargs = builder.get_tools.await_args
    assert sorted(kwargs["tool_names"]) == [
        "advanced_web_search_tool",
        "knowledge_search_tool",
        "web_search_tool",
    ]


@pytest.mark.asyncio
async def test_submit_job_forwards_conversation_id_header(submit_app):
    """The custom submit route must explicitly propagate NAT's routing header."""
    app, submitted_job, _builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query"},
            headers={"conversation-id": "customer-collection"},
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["conversation_id"] == "customer-collection"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail", "expected_retry_after"),
    [
        pytest.param(
            "rate",
            429,
            "Deep research submission rate limit reached. Please try again shortly.",
            "60",
            id="rate-limit",
        ),
        pytest.param(
            "principal",
            429,
            "Active deep research job limit reached. Wait for a running job to finish.",
            "30",
            id="principal-capacity",
        ),
        pytest.param(
            "global",
            503,
            "The server is at deep research capacity. Please try again shortly.",
            "30",
            id="global-capacity",
        ),
        pytest.param(
            "input",
            413,
            "Deep research input exceeds the 123-character limit.",
            None,
            id="input-limit",
        ),
    ],
)
def test_submit_job_maps_admission_failures(
    submit_app,
    error,
    expected_status,
    expected_detail,
    expected_retry_after,
):
    from deep_researcher_api.jobs.admission import JobGlobalCapacityExceededError
    from deep_researcher_api.jobs.admission import JobInputTooLargeError
    from deep_researcher_api.jobs.admission import JobPrincipalCapacityExceededError
    from deep_researcher_api.jobs.admission import JobSubmissionRateExceededError

    errors = {
        "rate": JobSubmissionRateExceededError(),
        "principal": JobPrincipalCapacityExceededError(),
        "global": JobGlobalCapacityExceededError(),
        "input": JobInputTooLargeError(123),
    }
    app, submitted_job, _builder = submit_app
    submitted_job.side_effect = errors[error]

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query"},
        )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert response.headers.get("retry-after") == expected_retry_after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential_source", "expected_token"),
    [
        pytest.param(
            "bearer-header",
            "bearer-token",
            id="bearer-header",
        ),
        pytest.param(
            "id-token-cookie",
            "cookie-token",
            id="id-token-cookie",
        ),
    ],
)
async def test_submit_job_forwards_authenticated_request_token(
    submit_app,
    monkeypatch,
    credential_source,
    expected_token,
):
    """REST submit captures the validated request token without requiring NAT Context."""
    import deep_researcher_agent.auth
    from deep_researcher_agent.auth.utils import get_auth_token as real_get_auth_token
    from deep_researcher_api.auth.middleware import AuthMiddleware

    class AcceptTokenValidator:
        def can_handle(self, token: str) -> bool:  # noqa: ARG002 - accepts both test credentials
            return True

        async def validate(self, token: str):
            return (
                {
                    "type": "jwt",
                    "sub": "user-1",
                    "email": "user@example.com",
                    "name": "Test User",
                    "token": token,
                    "skip_clarifier": False,
                },
                None,
            )

    app, submitted_job, _builder = submit_app
    monkeypatch.setattr(deep_researcher_agent.auth, "get_auth_token", real_get_auth_token)
    app.add_middleware(
        AuthMiddleware,
        validators=[AcceptTokenValidator()],
        require_auth=True,
        external_hostnames={"testserver"},
    )

    with TestClient(app) as client:
        headers = None
        if credential_source == "bearer-header":
            headers = {"Authorization": f"Bearer {expected_token}"}
        else:
            client.cookies.set("idToken", expected_token)
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query"},
            headers=headers,
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["auth_token"] == expected_token


@pytest.mark.asyncio
async def test_submit_job_rejects_internal_agent(submit_app, monkeypatch):
    app, submitted_job, _builder = submit_app
    import deep_researcher_api.routes.jobs as jobs_routes

    monkeypatch.setattr(
        jobs_routes,
        "get_agent_config",
        lambda _agent_type: AgentConfig(
            class_path="deep_researcher_agent.agents.report_rewriter.agent.ReportRewriterAgent",
            config_name="report_rewriter_agent",
            description="Internal report rewriter",
            public=False,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "report_rewriter", "input": "revise"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Agent type is internal-only and cannot be submitted directly: report_rewriter"
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_payload",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"data_sources": None}, id="null"),
        pytest.param({"data_sources": []}, id="empty"),
        pytest.param({"data_sources": ["web_search"]}, id="selected"),
    ],
)
async def test_submit_job_rejects_registered_but_unconfigured_agent(submit_app, monkeypatch, source_payload):
    app, submitted_job, builder = submit_app
    import deep_researcher_api.routes.jobs as jobs_routes

    shallow_config = AgentConfig(
        class_path="deep_researcher_agent.agents.shallow_researcher.agent.ShallowResearcherAgent",
        config_name="shallow_research_agent",
        description="Test shallow researcher",
    )
    monkeypatch.setattr(jobs_routes, "get_agent_config", lambda _agent_type: shallow_config)
    require_principal = MagicMock(return_value=Principal(type="jwt", sub="user-1", email="user@example.com"))
    monkeypatch.setattr(jobs_routes, "require_verified_principal", require_principal)
    builder.get_function_config.side_effect = ValueError("Function `shallow_research_agent` not found")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "shallow_researcher", "input": "query", **source_payload},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "agent_not_configured",
        "message": "Agent 'shallow_researcher' is not configured in the active workflow",
        "agent_type": "shallow_researcher",
        "config_name": "shallow_research_agent",
    }
    builder.get_function_config.assert_called_with("shallow_research_agent")
    builder.get_tools.assert_not_awaited()
    require_principal.assert_not_called()
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_does_not_classify_unexpected_function_config_value_error(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_function_config.side_effect = ValueError("invalid function configuration")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query"},
        )

    assert response.status_code == 500
    builder.get_tools.assert_not_awaited()
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_omitted_data_sources_keeps_all_sources(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query"},
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["data_sources"] is None


@pytest.mark.asyncio
async def test_submit_job_explicit_null_data_sources_keeps_all_sources(submit_app):
    """Explicit `null` in the JSON body must behave identically to field omission."""
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": None},
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["data_sources"] is None


@pytest.mark.asyncio
async def test_submit_job_forwards_empty_data_sources(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": []},
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["data_sources"] == []


@pytest.mark.asyncio
async def test_submit_job_rejects_unknown_data_sources(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["does_not_exist", "also_missing"],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "Unknown data source(s): does_not_exist, also_missing",
        "invalid_ids": ["does_not_exist", "also_missing"],
        "unavailable_for_agent": [],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_rejects_source_unavailable_for_agent(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool", "advanced_web_search_tool"],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "Data source(s) are not available for agent 'deep_researcher': web_search",
        "invalid_ids": [],
        "unavailable_for_agent": ["web_search"],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_rejects_mixed_available_and_agent_unavailable_sources(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool", "advanced_web_search_tool"],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["web_search", "knowledge_layer"],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "Data source(s) are not available for agent 'deep_researcher': web_search",
        "invalid_ids": [],
        "unavailable_for_agent": ["web_search"],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_rejects_combined_unknown_and_agent_unavailable_sources(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool", "advanced_web_search_tool"],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["does_not_exist", "web_search"],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": (
            "Unknown data source(s): does_not_exist. "
            "Data source(s) are not available for agent 'deep_researcher': web_search"
        ),
        "invalid_ids": ["does_not_exist"],
        "unavailable_for_agent": ["web_search"],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_dedupes_problematic_ids_preserving_first_seen_order_and_casing(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool", "advanced_web_search_tool"],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["Missing", "web_search", "missing", "WEB_SEARCH"],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": (
            "Unknown data source(s): Missing. Data source(s) are not available for agent 'deep_researcher': web_search"
        ),
        "invalid_ids": ["Missing"],
        "unavailable_for_agent": ["web_search"],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_rejects_known_source_when_agent_has_no_available_sources(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        exclude_tools=["web_search_tool", "advanced_web_search_tool", "knowledge_search_tool"],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "Data source(s) are not available for agent 'deep_researcher': web_search",
        "invalid_ids": [],
        "unavailable_for_agent": ["web_search"],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_forwards_omitted_data_sources_without_resolving_tools(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query"},
        )

    assert response.status_code == 200
    builder.get_function_config.assert_called_once_with("deep_research_agent")
    builder.get_tools.assert_not_awaited()
    submitted_job.assert_awaited_once()
    _, kwargs = submitted_job.await_args
    assert kwargs["data_sources"] is None


@pytest.mark.asyncio
async def test_submit_job_forwards_null_data_sources_without_resolving_tools(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": None},
        )

    assert response.status_code == 200
    builder.get_function_config.assert_called_once_with("deep_research_agent")
    builder.get_tools.assert_not_awaited()
    submitted_job.assert_awaited_once()
    _, kwargs = submitted_job.await_args
    assert kwargs["data_sources"] is None


@pytest.mark.asyncio
async def test_submit_job_forwards_valid_data_sources_exactly_as_provided(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["WEB_SEARCH", "web_search", "knowledge_layer"],
            },
        )

    assert response.status_code == 200
    submitted_job.assert_awaited_once()
    _, kwargs = submitted_job.await_args
    assert kwargs["data_sources"] == ["WEB_SEARCH", "web_search", "knowledge_layer"]


@pytest.mark.asyncio
async def test_submit_job_accepts_inherited_source_for_configured_shallow_researcher(submit_app, monkeypatch):
    app, submitted_job, builder = submit_app
    import deep_researcher_api.routes.jobs as jobs_routes

    shallow_config = AgentConfig(
        class_path="deep_researcher_agent.agents.shallow_researcher.agent.ShallowResearcherAgent",
        config_name="shallow_research_agent",
        description="Test shallow researcher",
    )
    monkeypatch.setattr(jobs_routes, "get_agent_config", lambda _agent_type: shallow_config)
    builder.get_function_config.return_value = ShallowResearchAgentConfig(
        llm="llm",
        exclude_tools=["advanced_web_search_tool"],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "shallow_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["data_sources"] == ["web_search"]
    builder.get_function_config.assert_called_with("shallow_research_agent")


@pytest.mark.asyncio
async def test_submit_job_forwards_multiple_data_sources(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["web_search", "knowledge_layer"],
            },
        )

    assert response.status_code == 200
    assert submitted_job.await_args.kwargs["data_sources"] == ["web_search", "knowledge_layer"]


@pytest.mark.asyncio
async def test_submit_job_mixed_valid_and_unknown_rejects_naming_only_unknown(submit_app):
    app, submitted_job, builder = submit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={
                "agent_type": "deep_researcher",
                "input": "query",
                "data_sources": ["web_search", "does_not_exist"],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "Unknown data source(s): does_not_exist",
        "invalid_ids": ["does_not_exist"],
        "unavailable_for_agent": [],
        "known_ids": ["knowledge_layer", "web_search"],
    }
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_auth_runs_before_data_source_validation(submit_app, monkeypatch):
    app, submitted_job, builder = submit_app
    import deep_researcher_api.routes.jobs as jobs_routes

    def _deny():
        raise HTTPException(status_code=403, detail="auth required")

    monkeypatch.setattr(jobs_routes, "require_verified_principal", _deny)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["does_not_exist"]},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "auth required"
    builder.get_tools.assert_not_awaited()
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_agent_type_validation_runs_before_auth(submit_app, monkeypatch):
    app, submitted_job, builder = submit_app
    import deep_researcher_api.routes.jobs as jobs_routes

    def _deny():
        raise HTTPException(status_code=403, detail="auth required")

    monkeypatch.setattr(jobs_routes, "require_verified_principal", _deny)
    monkeypatch.setattr(
        jobs_routes,
        "get_agent_config",
        lambda _agent_type: (_ for _ in ()).throw(KeyError("unknown agent")),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "nonexistent_agent", "input": "query"},
        )

    assert response.status_code == 400
    builder.get_tools.assert_not_awaited()
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_returns_500_when_validation_tool_resolution_fails_for_known_source(submit_app, caplog):
    app, submitted_job, builder = submit_app
    builder.get_tools.side_effect = RuntimeError("tool resolution failed")
    caplog.set_level(logging.ERROR, logger="deep_researcher_api.routes.jobs")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to validate data sources for selected agent"
    assert "Failed to validate data sources for agent deep_researcher using config deep_research_agent" in caplog.text
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_job_returns_500_when_validation_tool_resolution_fails_for_unknown_source(submit_app):
    app, submitted_job, builder = submit_app
    builder.get_tools.side_effect = RuntimeError("tool resolution failed")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["does_not_exist"]},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to validate data sources for selected agent"
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_validation_calls_get_all_tool_refs_when_fn_config_tools_is_empty(submit_app, monkeypatch):
    """Regression test: ``tools=[]`` must inherit refs via ``get_all_tool_refs``.

    The fixture's typed default ``tools=[]`` exercises this branch, but no other
    test asserts the exact call. This pins the contract so future refactors
    that swap the inheritance mechanism (e.g. caching, alternate registries)
    surface here instead of silently changing agent capability resolution.
    """
    app, submitted_job, builder = submit_app
    # Spy on the name as bound
    # in the routes module (Python testing idiom: patch where used, not where
    # defined).
    import deep_researcher_api.routes.jobs as jobs_routes

    spy = MagicMock(wraps=jobs_routes.get_all_tool_refs)
    monkeypatch.setattr(jobs_routes, "get_all_tool_refs", spy)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["web_search"]},
        )

    assert response.status_code == 200
    spy.assert_called_once()
    # Sanity check: the spy returned the registry-populated tool refs and the
    # builder was asked to resolve those exact refs via LangChain wrappers.
    builder.get_tools.assert_awaited_once()
    _, kwargs = builder.get_tools.await_args
    assert sorted(kwargs["tool_names"]) == [
        "advanced_web_search_tool",
        "knowledge_search_tool",
        "web_search_tool",
    ]
    submitted_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_validation_does_not_call_get_all_tool_refs_when_fn_config_tools_is_explicit(submit_app, monkeypatch):
    """Regression test: explicit ``tools=[...]`` must override registry inheritance.

    Mirror of the test above for the opposite branch: when the agent declares
    an explicit tool list, the validator must use it verbatim and skip the
    inheritance call entirely.
    """
    app, submitted_job, builder = submit_app
    builder.get_function_config.return_value = DeepResearchAgentConfig(
        orchestrator_llm="llm",
        tools=["knowledge_search_tool"],
    )

    import deep_researcher_api.routes.jobs as jobs_routes

    spy = MagicMock(wraps=jobs_routes.get_all_tool_refs)
    monkeypatch.setattr(jobs_routes, "get_all_tool_refs", spy)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "query", "data_sources": ["knowledge_layer"]},
        )

    assert response.status_code == 200
    spy.assert_not_called()
    builder.get_tools.assert_awaited_once()
    _, kwargs = builder.get_tools.await_args
    assert kwargs["tool_names"] == ["knowledge_search_tool"]
    submitted_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_data_sources_exposes_default_enabled(submit_app):
    """GET /v1/data_sources must surface the registry's default_enabled (not hardcode True)."""
    app, _submitted_job, _builder = submit_app
    # Re-populate at request time: list_data_sources() reads the registry per request.
    reset_registry()
    populate_from_config(
        [
            {"id": "web_search", "name": "Web Search", "description": "x"},  # default_enabled -> True
            {"id": "off_by_default", "name": "Off", "description": "y", "default_enabled": False},
        ]
    )

    with TestClient(app) as client:
        response = client.get("/v1/data_sources")

    assert response.status_code == 200
    by_id = {s["id"]: s for s in response.json()}
    assert by_id["web_search"]["default_enabled"] is True
    assert by_id["off_by_default"]["default_enabled"] is False
