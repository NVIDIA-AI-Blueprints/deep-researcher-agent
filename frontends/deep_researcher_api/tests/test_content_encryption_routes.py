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

from __future__ import annotations

import base64
import json
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from deep_researcher_agent.auth import Principal
from deep_researcher_api.jobs import crypto
from deep_researcher_api.registry import AgentConfig


def _static_key() -> str:
    return base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")


def _other_static_key() -> str:
    return base64.urlsafe_b64encode(bytes(reversed(range(32)))).decode("ascii")


@pytest.fixture(autouse=True)
def clean_encryption_route_env(monkeypatch):
    for name in (
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS",
        "VAULT_ADDR",
        "VAULT_NAMESPACE",
        "VAULT_TRANSIT_MOUNT",
        "VAULT_ROLE_ID",
        "VAULT_SECRET_ID",
        "DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY",
        "VAULT_TIMEOUT_SECONDS",
        "REQUIRE_AUTH",
    ):
        monkeypatch.delenv(name, raising=False)
    crypto.reset_content_encryption_manager_for_tests()
    yield
    crypto.reset_content_encryption_manager_for_tests()


def _enable_static_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _static_key())
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID", "test-key")
    crypto.reset_content_encryption_manager_for_tests()


def _enable_vault(monkeypatch) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY", "reports")
    crypto.reset_content_encryption_manager_for_tests()


async def _ready_unless_async_jobs_are_static_missing(**kwargs):
    if not kwargs["dask_available"] or kwargs["job_store"] is None or not kwargs["scheduler_address"]:
        return {"reason": "async_jobs_unavailable", "db": "unchecked"}
    return None


async def _build_jobs_app(
    monkeypatch,
    tmp_path,
    *,
    job_output=None,
    job_status="success",
    job_error=None,
    submitted_job=None,
) -> FastAPI:
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs import access
    from deep_researcher_api.jobs import event_store
    from deep_researcher_api.jobs import submit

    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_is_readable_regular_file", MagicMock(return_value=True))
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", AsyncMock())
    monkeypatch.setattr(
        jobs_routes,
        "_probe_async_job_readiness",
        AsyncMock(side_effect=_ready_unless_async_jobs_are_static_missing),
    )

    async def _no_op_reaper(*_args, **_kwargs):
        return None

    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", _no_op_reaper)
    monkeypatch.setattr(
        jobs_routes,
        "require_verified_principal",
        lambda: Principal(type="jwt", sub="user-1", email="user@example.com"),
    )
    monkeypatch.setattr(event_store.EventStore, "_ensure_table_exists", MagicMock())

    if submitted_job is not None:
        # register_job_routes imports this helper after the patch and captures the mock.
        monkeypatch.setattr(submit, "submit_agent_job", submitted_job)

    agent_config = AgentConfig(
        class_path="deep_researcher_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        config_name="deep_research_agent",
        description="Test deep researcher",
    )
    monkeypatch.setattr(jobs_routes, "get_agent_config", lambda _agent_type: agent_config)

    job = SimpleNamespace(
        job_id="job-1",
        status=job_status,
        error=job_error,
        output=job_output,
        created_at=datetime.now(UTC),
    )
    job_store = SimpleNamespace(get_job=AsyncMock(return_value=job), update_status=AsyncMock())
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    access._job_access_schema_initialized.clear()

    worker = SimpleNamespace(
        _dask_available=True,
        _job_store=job_store,
        _scheduler_address="tcp://localhost:8786",
        _db_url=db_url,
        _config_file_path="config.yml",
        _log_level=20,
        _use_dask_threads=False,
        _front_end_config=SimpleNamespace(expiry_seconds=86400),
    )
    builder = MagicMock()
    builder.get_function_config.return_value = SimpleNamespace(tools=[], exclude_tools=[])
    builder.get_tools = AsyncMock(return_value=[])

    app = FastAPI()
    await jobs_routes.register_job_routes(app, builder, worker)
    return app


def _build_assembled_worker_app(
    monkeypatch, tmp_path, *, dask_available: bool = True, db_url: str | None = None
) -> FastAPI:
    """Build the Deep Researcher Agent worker app while preserving NAT-before-DeepResearcher route ordering."""
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api import plugin
    from deep_researcher_api.jobs import access
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.data_models.config import Config
    from nat.data_models.config import GeneralConfig
    from nat.front_ends.fastapi import fastapi_front_end_plugin_worker as worker_module
    from nat.plugins.eval.fastapi import routes as eval_routes
    from nat.plugins.mcp.client import fastapi_routes as mcp_routes

    monkeypatch.setenv("NAT_CONFIG_FILE", "config.yml")
    monkeypatch.delenv("NAT_DASK_SCHEDULER_ADDRESS", raising=False)
    monkeypatch.setattr(plugin, "_load_validators_from_entry_points", lambda: [])
    monkeypatch.setattr(plugin.DeepResearcherAPIWorker, "_install_signal_handlers", lambda _self: None)
    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_is_readable_regular_file", MagicMock(return_value=True))
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", AsyncMock())
    monkeypatch.setattr(
        jobs_routes,
        "_probe_async_job_readiness",
        AsyncMock(side_effect=_ready_unless_async_jobs_are_static_missing),
    )
    monkeypatch.setattr(access, "ensure_job_access_table", MagicMock())
    monkeypatch.setattr(
        worker_module.FastApiFrontEndPluginWorker,
        "_create_session_manager",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(worker_module.FastApiFrontEndPluginWorker, "add_default_route", AsyncMock())
    for route_helper in (
        "add_authorization_route",
        "add_execution_routes",
        "add_monitor_route",
        "add_static_files_route",
    ):
        monkeypatch.setattr(worker_module, route_helper, AsyncMock())
    monkeypatch.setattr(eval_routes, "add_evaluate_routes", AsyncMock())
    monkeypatch.setattr(mcp_routes, "add_mcp_client_tool_list_route", AsyncMock())

    async def _no_op_reaper(*_args, **_kwargs):
        return None

    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", _no_op_reaper)

    builder = MagicMock()
    builder.get_function_config.return_value = SimpleNamespace(tools=[], exclude_tools=[])
    builder.get_tools = AsyncMock(return_value=[])

    class BuilderContext:
        async def __aenter__(self):
            return builder

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(WorkflowBuilder, "from_config", lambda _config: BuilderContext())

    front_end = plugin.DeepResearcherAPIConfig(db_url=db_url or f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}")
    worker = plugin.DeepResearcherAPIWorker(Config(general=GeneralConfig(front_end=front_end)))
    worker._dask_available = dask_available
    worker._job_store = object() if dask_available else None
    worker._scheduler_address = "tcp://localhost:8786"
    worker._db_url = front_end.db_url
    return worker.build_app()


def _get_health_routes(app: FastAPI) -> list[APIRoute]:
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/health" and "GET" in route.methods
    ]


def _get_liveness_routes(app: FastAPI) -> list[APIRoute]:
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/live" and "GET" in route.methods
    ]


@pytest.mark.asyncio
async def test_health_returns_503_when_vault_readiness_failed(monkeypatch, tmp_path):
    _enable_vault(monkeypatch)

    class FailingVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            raise crypto.ContentEncryptionUnavailable("vault down")

    monkeypatch.setattr(crypto, "_VaultTransitClient", FailingVault)
    app = await _build_jobs_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")
        liveness_response = client.get("/live")

    assert response.status_code == 503
    body = response.json()
    assert body["encryption"]["mode"] == "vault"
    assert body["encryption"]["ready"] is False
    assert body["encryption"]["encrypt_ready"] is False
    assert body["encryption"]["decrypt_ready"] is False
    assert liveness_response.status_code == 200
    assert liveness_response.json() == {"status": "alive"}


def test_assembled_worker_serves_encryption_health_route(monkeypatch, tmp_path):
    _enable_vault(monkeypatch)

    class FailingVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            raise crypto.ContentEncryptionUnavailable("vault down")

    monkeypatch.setattr(crypto, "_VaultTransitClient", FailingVault)
    app = _build_assembled_worker_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")
        liveness_response = client.get("/live")
        health_routes = _get_health_routes(app)
        liveness_routes = _get_liveness_routes(app)

    assert len(health_routes) == 1
    assert health_routes[0].endpoint.__module__ == "deep_researcher_api.routes.jobs"
    assert len(liveness_routes) == 1
    assert liveness_routes[0].endpoint.__module__ == "deep_researcher_api.routes.jobs"
    assert liveness_response.status_code == 200
    assert liveness_response.json() == {"status": "alive"}
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["encryption"]["mode"] == "vault"
    assert body["encryption"]["ready"] is False


def test_assembled_worker_keeps_liveness_when_async_jobs_are_unavailable(monkeypatch, tmp_path):
    app = _build_assembled_worker_app(monkeypatch, tmp_path, dask_available=False)

    with TestClient(app) as client:
        liveness_response = client.get("/live")
        readiness_response = client.get("/health")
        openapi = client.get("/openapi.json").json()

    assert liveness_response.status_code == 200
    assert liveness_response.json() == {"status": "alive"}
    assert readiness_response.status_code == 503
    assert readiness_response.json() == {
        "status": "degraded",
        "dask_available": False,
        "db": "unchecked",
        "reason": "async_jobs_unavailable",
    }
    assert len(_get_liveness_routes(app)) == 1
    assert len(_get_health_routes(app)) == 1
    assert list(openapi["paths"]["/live"]) == ["get"]
    assert list(openapi["paths"]["/health"]) == ["get"]


@pytest.mark.parametrize("mode", ["off", "key"])
def test_assembled_worker_health_route_reports_ready_encryption(monkeypatch, tmp_path, mode):
    if mode == "key":
        _enable_static_key(monkeypatch)
    app = _build_assembled_worker_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")
        openapi = client.get("/openapi.json").json()
        health_routes = _get_health_routes(app)

    assert len(health_routes) == 1
    assert health_routes[0].endpoint.__module__ == "deep_researcher_api.routes.jobs"
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["encryption"] == {"mode": mode, "ready": True}
    assert list(openapi["paths"]["/live"]) == ["get"]
    assert list(openapi["paths"]["/health"]) == ["get"]


def test_assembled_worker_health_returns_503_when_db_unreachable_on_fresh_process(monkeypatch, tmp_path):
    """A fresh process with an empty async-engine cache must still ping the DB.

    Regression: the health check previously reported ``db: no_engine`` (HTTP 200)
    whenever no async engine happened to be cached yet, so a fresh pod could pass
    Helm readiness straight through a database outage.
    """
    from sqlalchemy.exc import OperationalError

    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs.event_store import EventStore

    real_readiness_probe = jobs_routes._probe_async_job_readiness
    app = _build_assembled_worker_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        # The process started cleanly, but by the time the uncached readiness
        # probe runs the database is unavailable.
        EventStore._async_engine_cache.clear()

        def _unreachable(_db_url):
            raise OperationalError("SELECT 1", {}, Exception("database is unavailable"))

        monkeypatch.setattr(jobs_routes, "_probe_async_job_readiness", real_readiness_probe)
        monkeypatch.setattr(EventStore, "_get_or_create_async_engine", _unreachable)

        response = client.get("/health")
        liveness_response = client.get("/live")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["db"] == "unreachable"
    # Liveness must stay decoupled from dependency health.
    assert liveness_response.status_code == 200
    assert liveness_response.json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_health_returns_503_when_vault_decrypt_readiness_failed(monkeypatch, tmp_path):
    _enable_vault(monkeypatch)

    class DecryptDeniedVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            return b"d" * crypto.DEK_BYTES, crypto.WrappedDEK(
                wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek"
            )

        def unwrap_dek(self, wrapped_dek, *, operation):
            raise crypto.ContentEncryptionUnavailable("decrypt denied")

    monkeypatch.setattr(crypto, "_VaultTransitClient", DecryptDeniedVault)
    app = await _build_jobs_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["encryption"]["mode"] == "vault"
    assert body["encryption"]["ready"] is False
    assert body["encryption"]["encrypt_ready"] is True
    assert body["encryption"]["decrypt_ready"] is False
    assert body["encryption"]["reason"] == "vault_decrypt_unavailable"


@pytest.mark.asyncio
async def test_submit_rejects_when_encryption_readiness_failed(monkeypatch, tmp_path):
    _enable_vault(monkeypatch)
    submitted_job = AsyncMock(return_value="job-1")

    class FailingVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            raise crypto.ContentEncryptionUnavailable("vault down")

    monkeypatch.setattr(crypto, "_VaultTransitClient", FailingVault)
    app = await _build_jobs_app(monkeypatch, tmp_path, submitted_job=submitted_job)

    with TestClient(app) as client:
        response = client.post("/v1/jobs/async/submit", json={"agent_type": "deep_researcher", "input": "query"})

    assert response.status_code == 503
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_rejects_when_vault_decrypt_readiness_failed(monkeypatch, tmp_path):
    _enable_vault(monkeypatch)
    submitted_job = AsyncMock(return_value="job-1")

    class DecryptDeniedVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            return b"d" * crypto.DEK_BYTES, crypto.WrappedDEK(
                wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek"
            )

        def unwrap_dek(self, wrapped_dek, *, operation):
            raise crypto.ContentEncryptionUnavailable("decrypt denied")

    monkeypatch.setattr(crypto, "_VaultTransitClient", DecryptDeniedVault)
    app = await _build_jobs_app(monkeypatch, tmp_path, submitted_job=submitted_job)

    with TestClient(app) as client:
        response = client.post("/v1/jobs/async/submit", json={"agent_type": "deep_researcher", "input": "query"})

    assert response.status_code == 503
    submitted_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_uses_authorized_submission_when_encryption_is_ready(monkeypatch, tmp_path):
    _enable_static_key(monkeypatch)
    submitted_job = AsyncMock(return_value="job-1")
    app = await _build_jobs_app(monkeypatch, tmp_path, submitted_job=submitted_job)

    with TestClient(app) as client:
        response = client.post("/v1/jobs/async/submit", json={"agent_type": "deep_researcher", "input": "query"})

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    submitted_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_openapi_documents_encryption_failures(monkeypatch, tmp_path):
    app = await _build_jobs_app(monkeypatch, tmp_path)

    responses = app.openapi()["paths"]["/v1/jobs/async/submit"]["post"]["responses"]

    assert responses["400"]["description"] == ("Unknown, internal-only, or unconfigured agent type, or invalid request")
    assert responses["413"]["description"] == "Deep-research input exceeds the configured payload limit"
    assert responses["429"]["description"] == "Per-principal active-job or submission-rate limit reached"
    assert responses["422"]["description"] == "One or more unknown or agent-unavailable data source IDs"
    assert responses["500"]["description"] == (
        "Content encryption configuration is invalid, async job authorization persistence failed, "
        "or agent/tool configuration lookup failed unexpectedly"
    )
    assert responses["503"]["description"] == (
        "Content encryption, Dask scheduler, admission database, or deployment job capacity is unavailable"
    )


@pytest.mark.asyncio
async def test_report_authorizes_before_decrypting(monkeypatch, tmp_path):
    _enable_static_key(monkeypatch)
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    app = await _build_jobs_app(monkeypatch, tmp_path, job_output='{"report":"plaintext"}')

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/report")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_report_returns_500_for_plaintext_violation(monkeypatch, tmp_path):
    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path, job_output='{"report":"plaintext"}')

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/report")

    assert response.status_code == 500
    assert response.json()["detail"] == "Final report data is invalid"


@pytest.mark.parametrize("job_output", ["", {}])
@pytest.mark.asyncio
async def test_report_returns_500_for_falsey_plaintext_violation(monkeypatch, tmp_path, job_output):
    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path, job_output=job_output)

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/report")

    assert response.status_code == 500
    assert response.json()["detail"] == "Final report data is invalid"


@pytest.mark.asyncio
async def test_report_returns_500_for_malformed_envelope(monkeypatch, tmp_path):
    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path, job_output="aiqenc:not-json")

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/report")

    assert response.status_code == 500
    assert response.json()["detail"] == "Final report data is invalid"


@pytest.mark.asyncio
async def test_report_returns_503_when_vault_decrypt_is_unavailable(monkeypatch, tmp_path):
    _enable_vault(monkeypatch)

    class UnwrapFailingVault:
        def __init__(self, _config):
            pass

        def generate_data_key(self, *, operation):
            return b"0" * 32, crypto.WrappedDEK(wrap="vault", kid="transit/reports", wrapped_dek="vault:v1:dek")

        def unwrap_dek(self, wrapped_dek, *, operation):
            raise crypto.ContentEncryptionUnavailable("vault down")

    envelope = crypto.encode_envelope(
        {
            "v": crypto.ENVELOPE_VERSION,
            "alg": crypto.CONTENT_ALGORITHM,
            "wrap": "vault",
            "kid": "transit/reports",
            "aad_hint": crypto.job_output_aad("job-1"),
            "wrapped_dek": "vault:v1:dek",
            "nonce": base64.urlsafe_b64encode(b"1" * 12).decode("ascii").rstrip("="),
            "ciphertext": base64.urlsafe_b64encode(b"ciphertext").decode("ascii").rstrip("="),
            "tag": base64.urlsafe_b64encode(b"2" * 16).decode("ascii").rstrip("="),
        }
    )
    monkeypatch.setattr(crypto, "_VaultTransitClient", UnwrapFailingVault)
    app = await _build_jobs_app(monkeypatch, tmp_path, job_output=envelope)

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/report")

    assert response.status_code == 503
    assert response.json()["detail"] == "Content encryption is unavailable"


@pytest.mark.asyncio
async def test_report_decrypts_encrypted_final_output(monkeypatch, tmp_path):
    _enable_static_key(monkeypatch)
    stored = crypto.create_job_content_cipher("job-1").encrypt_output_json(
        json.dumps(
            {
                "report": "secret",
                "parent_job_id": "parent-job",
                "interaction_action": "edit",
                "result_kind": "report",
            }
        )
    )
    app = await _build_jobs_app(monkeypatch, tmp_path, job_output=stored)

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/report")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "has_report": True,
        "report": "secret",
        "parent_job_id": "parent-job",
        "interaction_action": "edit",
        "result_kind": "report",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome_reason", "actionable_error"),
    [
        (
            "no_sources_selected",
            "No data sources are selected. Select at least one data source and run the research again.",
        ),
        (
            "no_source_results",
            "The selected data sources returned no results. "
            "Try rephrasing the question or selecting different data sources.",
        ),
    ],
)
async def test_source_failure_status_is_actionable_and_preserved_report_is_available(
    monkeypatch, tmp_path, outcome_reason, actionable_error
):
    _enable_static_key(monkeypatch)
    stored = crypto.create_job_content_cipher("job-1").encrypt_output_json(
        json.dumps(
            {
                "report": "# Preserved generated answer",
                "outcome_reason": outcome_reason,
            }
        )
    )
    app = await _build_jobs_app(
        monkeypatch,
        tmp_path,
        job_output=stored,
        job_status="failure",
        job_error=actionable_error,
    )

    with TestClient(app) as client:
        status_response = client.get("/v1/jobs/async/job/job-1")
        report_response = client.get("/v1/jobs/async/job/job-1/report")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "failure"
    assert status_response.json()["error"] == actionable_error
    assert report_response.status_code == 200
    assert report_response.json()["has_report"] is True
    assert report_response.json()["report"] == "# Preserved generated answer"


@pytest.mark.asyncio
async def test_state_returns_500_for_invalid_encrypted_event_data(monkeypatch, tmp_path):
    from deep_researcher_api.jobs.event_store import EventStore

    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    cipher = crypto.create_job_content_cipher("job-1")
    EventStore(db_url, "job-1", content_cipher=cipher).store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _other_static_key())
    crypto.reset_content_encryption_manager_for_tests()

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/state")

    assert response.status_code == 500
    assert response.json()["detail"] == "Job state data is invalid"


@pytest.mark.asyncio
async def test_state_returns_503_when_event_decrypt_is_unavailable(monkeypatch, tmp_path):
    from deep_researcher_api.jobs.event_store import EventStore

    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    cipher = crypto.create_job_content_cipher("job-1")
    EventStore(db_url, "job-1", content_cipher=cipher).store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )

    def unavailable_decrypt(*_args, **_kwargs):
        raise crypto.ContentEncryptionUnavailable("vault down")

    monkeypatch.setattr(crypto, "decrypt_event_field", unavailable_decrypt)

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/job-1/state")

    assert response.status_code == 503
    assert response.json()["detail"] == "Content encryption is unavailable"


@pytest.mark.asyncio
async def test_sse_emits_job_error_for_invalid_encrypted_event_data(monkeypatch, tmp_path):
    from deep_researcher_api.jobs.event_store import EventStore

    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    cipher = crypto.create_job_content_cipher("job-1")
    EventStore(db_url, "job-1", content_cipher=cipher).store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _other_static_key())
    crypto.reset_content_encryption_manager_for_tests()

    with TestClient(app) as client:
        with client.stream("GET", "/v1/jobs/async/job/job-1/stream") as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: job.error" in body
    assert "Job event data is invalid" in body
    assert "secret report" not in body


@pytest.mark.asyncio
async def test_sse_emits_job_error_when_event_decrypt_is_unavailable(monkeypatch, tmp_path):
    from deep_researcher_api.jobs.event_store import EventStore

    _enable_static_key(monkeypatch)
    app = await _build_jobs_app(monkeypatch, tmp_path)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    cipher = crypto.create_job_content_cipher("job-1")
    EventStore(db_url, "job-1", content_cipher=cipher).store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )

    def unavailable_decrypt(*_args, **_kwargs):
        raise crypto.ContentEncryptionUnavailable("vault down")

    monkeypatch.setattr(crypto, "decrypt_event_field", unavailable_decrypt)

    with TestClient(app) as client:
        with client.stream("GET", "/v1/jobs/async/job/job-1/stream") as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: job.error" in body
    assert "Content encryption is unavailable" in body
    assert "secret report" not in body
