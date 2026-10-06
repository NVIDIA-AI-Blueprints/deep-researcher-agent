# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for async-job startup, readiness, and submit gating."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from httpx import ASGITransport
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from deep_researcher_agent.auth import Principal


def _successful_agent_execution(*_args, **_kwargs) -> None:
    """Stand in for provider-backed agent execution on the real Dask worker."""


def _job_store(db_url: str):
    from nat.front_ends.fastapi.async_jobs.job_store import JobStore

    store = JobStore(scheduler_address="tcp://scheduler.invalid:8786", db_url=db_url)
    store._dask_client = MagicMock()
    store._dask_client.sync.return_value = {"workers": {}}
    return store


def _submit_routes(app: FastAPI) -> list[APIRoute]:
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/jobs/async/submit" and "POST" in route.methods
    ]


def _worker(*, job_store=None, config_path: str = "", db_url: str = "sqlite:///unused.db") -> SimpleNamespace:
    return SimpleNamespace(
        _dask_available=job_store is not None,
        _job_store=job_store,
        _scheduler_address="tcp://scheduler.invalid:8786" if job_store is not None else None,
        _db_url=db_url,
        _config_file_path=config_path,
        _log_level=20,
        _use_dask_threads=True,
        _front_end_config=SimpleNamespace(expiry_seconds=86400),
    )


def _builder() -> MagicMock:
    builder = MagicMock()
    builder.get_function_config.return_value = SimpleNamespace(tools=[], exclude_tools=[])
    builder.get_tools = AsyncMock(return_value=[])
    return builder


@pytest.fixture
def route_dependencies(monkeypatch):
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs import submit

    principal = Principal(type="jwt", sub="user-1", email="user@example.com")
    submit_job = AsyncMock(return_value="job-1")
    encryption_submit = AsyncMock()

    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", AsyncMock())
    monkeypatch.setattr(jobs_routes, "require_verified_principal", MagicMock(return_value=principal))
    monkeypatch.setattr(submit, "submit_agent_job", submit_job)
    monkeypatch.setattr(
        "deep_researcher_api.jobs.crypto.require_content_encryption_ready_for_submission_async",
        encryption_submit,
    )

    return jobs_routes, submit_job, encryption_submit


@pytest.mark.asyncio
async def test_bootstrap_initializes_clean_sqlite_and_accepts_empty_scheduler_worker_map(tmp_path):
    import deep_researcher_api.routes.jobs as jobs_routes

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    store = _job_store(db_url)

    await jobs_routes._bootstrap_async_job_storage(db_url, store)

    assert jobs_routes._REQUIRED_ASYNC_JOB_TABLES <= await jobs_routes._table_names(db_url)
    assert await store.get_job(jobs_routes._READINESS_JOB_ID) is None
    assert (
        await jobs_routes._probe_async_job_readiness(
            dask_available=True,
            job_store=store,
            scheduler_address="tcp://scheduler.invalid:8786",
            db_url=db_url,
            config_path=str(config_path),
            submit_route_registered=True,
        )
        is None
    )
    store._dask_client.sync.assert_called_once_with(
        store._dask_client.scheduler.identity,
        callback_timeout=jobs_routes._ASYNC_JOB_READINESS_TIMEOUT_SECONDS,
    )
    store._dask_client.sync.reset_mock()
    assert await jobs_routes._probe_async_job_readiness(
        dask_available=True,
        job_store=store,
        scheduler_address="tcp://scheduler.invalid:8786",
        db_url=db_url,
        config_path=str(config_path),
        submit_route_registered=False,
    ) == {"reason": "async_jobs_unavailable", "db": "ok"}


@pytest.mark.asyncio
async def test_bootstrap_does_not_alter_incomplete_nat_schema(tmp_path):
    import deep_researcher_api.routes.jobs as jobs_routes

    db_path = tmp_path / "jobs.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE job_info (job_id VARCHAR PRIMARY KEY)"))
    store = _job_store(db_url)

    with pytest.raises(OperationalError, match="no such column|has no column|job_info"):
        await jobs_routes._bootstrap_async_job_storage(db_url, store)

    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(job_info)"))}
    assert columns == {"job_id"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table",
    ["job_info", "job_access", "job_events", "artifacts", "deep_research_admission"],
)
async def test_probe_reports_dropped_required_table_as_schema_unavailable(tmp_path, table):
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs.event_store import EventStore

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    store = _job_store(db_url)
    await jobs_routes._bootstrap_async_job_storage(db_url, store)
    engine = EventStore._get_or_create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE {table}"))

    assert await jobs_routes._probe_async_job_readiness(
        dask_available=True,
        job_store=store,
        scheduler_address="tcp://scheduler.invalid:8786",
        db_url=db_url,
        config_path=str(config_path),
        submit_route_registered=True,
    ) == {"reason": "async_jobs_unavailable", "db": "schema_unavailable"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "column",
    [
        "job_id",
        "reservation_token",
        "owner_auth_type",
        "owner_subject",
        "admitted_at",
        "reservation_expires_at",
    ],
)
async def test_probe_reports_incomplete_admission_schema_as_unavailable(tmp_path, column):
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs.event_store import EventStore

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    store = _job_store(db_url)
    await jobs_routes._bootstrap_async_job_storage(db_url, store)
    engine = EventStore._get_or_create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text(f"ALTER TABLE deep_research_admission RENAME COLUMN {column} TO missing_{column}"))

    assert await jobs_routes._probe_async_job_readiness(
        dask_available=True,
        job_store=store,
        scheduler_address="tcp://scheduler.invalid:8786",
        db_url=db_url,
        config_path=str(config_path),
        submit_route_registered=True,
    ) == {"reason": "async_jobs_unavailable", "db": "schema_unavailable"}


@pytest.mark.asyncio
async def test_job_access_schema_drift_degrades_health_and_blocks_submit(route_dependencies, tmp_path):
    from deep_researcher_api.jobs.access import create_job_access
    from deep_researcher_api.jobs.event_store import EventStore

    jobs_routes, submit_job, encryption_submit = route_dependencies
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    store = _job_store(db_url)
    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        _builder(),
        _worker(job_store=store, config_path=str(config_path), db_url=db_url),
    )

    async def _persist_job_access(**kwargs):
        await asyncio.to_thread(create_job_access, "job-1", kwargs["principal"], db_url)
        return "job-1"

    submit_job.side_effect = _persist_job_access
    engine = EventStore._get_or_create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE job_access RENAME COLUMN owner_subject TO missing_owner_subject"))

    with TestClient(app) as client:
        health = client.get("/health")
        submit = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "shallow_researcher", "input": "question"},
        )

    assert health.status_code == 503
    assert health.json() == {
        "status": "degraded",
        "dask_available": True,
        "db": "schema_unavailable",
        "reason": "async_jobs_unavailable",
    }
    assert submit.status_code == 503
    assert submit.json() == {"detail": "Async job submission is currently unavailable"}
    submit_job.assert_not_awaited()
    encryption_submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_probe_reports_unreachable_database(monkeypatch, tmp_path):
    import deep_researcher_api.routes.jobs as jobs_routes

    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    monkeypatch.setattr(jobs_routes, "_table_names", AsyncMock(side_effect=OSError("database unavailable")))

    assert await jobs_routes._probe_async_job_readiness(
        dask_available=True,
        job_store=MagicMock(),
        scheduler_address="tcp://scheduler.invalid:8786",
        db_url="sqlite+aiosqlite:///unreachable.db",
        config_path=str(config_path),
        submit_route_registered=True,
    ) == {"reason": "async_jobs_unavailable", "db": "unreachable"}


@pytest.mark.asyncio
async def test_probe_reports_scheduler_failure_and_timeout(monkeypatch, tmp_path):
    import deep_researcher_api.routes.jobs as jobs_routes

    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    monkeypatch.setattr(
        jobs_routes,
        "_table_names",
        AsyncMock(return_value=set(jobs_routes._REQUIRED_ASYNC_JOB_TABLES)),
    )
    monkeypatch.setattr(
        "deep_researcher_api.jobs.access.validate_job_access_table",
        MagicMock(),
    )
    monkeypatch.setattr(
        "deep_researcher_api.jobs.admission.validate_deep_research_admission_table",
        MagicMock(),
    )

    store.dask_client.sync.side_effect = OSError("scheduler unavailable")
    assert await jobs_routes._probe_async_job_readiness(
        dask_available=True,
        job_store=store,
        scheduler_address="tcp://scheduler.invalid:8786",
        db_url="sqlite+aiosqlite:///unused.db",
        config_path=str(config_path),
        submit_route_registered=True,
    ) == {"reason": "async_jobs_unavailable", "db": "ok"}

    def _slow_scheduler_info():
        time.sleep(0.05)
        return {"workers": {}}

    store.dask_client.sync.side_effect = lambda _rpc, **_kwargs: _slow_scheduler_info()
    monkeypatch.setattr(jobs_routes, "_ASYNC_JOB_READINESS_TIMEOUT_SECONDS", 0.01)
    assert await jobs_routes._probe_async_job_readiness(
        dask_available=True,
        job_store=store,
        scheduler_address="tcp://scheduler.invalid:8786",
        db_url="sqlite+aiosqlite:///unused.db",
        config_path=str(config_path),
        submit_route_registered=True,
    ) == {"reason": "async_jobs_unavailable", "db": "ok"}


@pytest.mark.asyncio
async def test_static_failure_registers_one_guarded_submit_with_validation_and_auth_precedence(
    route_dependencies,
):
    jobs_routes, submit_job, encryption_submit = route_dependencies
    app = FastAPI()
    await jobs_routes.register_job_routes(app, _builder(), _worker())

    assert len(_submit_routes(app)) == 1
    with TestClient(app) as client:
        assert client.post("/v1/jobs/async/submit", json={"agent_type": "deep_researcher"}).status_code == 422
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "question"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Async job submission is currently unavailable"}
    submit_job.assert_not_awaited()
    encryption_submit.assert_not_awaited()

    jobs_routes.require_verified_principal.side_effect = HTTPException(401, "Authentication required")
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "question"},
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


@pytest.mark.asyncio
async def test_missing_config_health_and_submit_are_deterministic(route_dependencies, tmp_path):
    jobs_routes, submit_job, _encryption_submit = route_dependencies
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    store = _job_store(db_url)
    store.get_job = AsyncMock(return_value=None)
    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        _builder(),
        _worker(job_store=store, config_path=str(tmp_path / "missing.yml"), db_url=db_url),
    )

    assert len(_submit_routes(app)) == 1
    with TestClient(app) as client:
        health = client.get("/health")
        submit = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "question"},
        )

    assert health.status_code == 503
    assert health.json() == {
        "status": "degraded",
        "dask_available": True,
        "db": "unchecked",
        "reason": "configuration_missing",
    }
    assert submit.status_code == 503
    assert submit.json() == {"detail": "Async job submission is currently unavailable"}
    store.get_job.assert_not_awaited()
    store._dask_client.sync.assert_not_called()
    submit_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_failure_aborts_startup_and_recovery_registers_full_routes(
    route_dependencies,
    monkeypatch,
    tmp_path,
):
    jobs_routes, submit_job, encryption_submit = route_dependencies
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    store = _job_store(db_url)
    bootstrap = AsyncMock(side_effect=[RuntimeError("schema"), None])
    readiness = AsyncMock(return_value=None)
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", bootstrap)
    monkeypatch.setattr(jobs_routes, "_probe_async_job_readiness", readiness)
    worker = _worker(job_store=store, config_path=str(config_path), db_url=db_url)

    with pytest.raises(RuntimeError, match="schema"):
        await jobs_routes.register_job_routes(FastAPI(), _builder(), worker)

    recovered_app = FastAPI()
    await jobs_routes.register_job_routes(
        recovered_app,
        _builder(),
        worker,
    )

    route_paths = {route.path for route in recovered_app.routes if isinstance(route, APIRoute)}
    assert "/v1/jobs/async/job/{job_id}" in route_paths
    assert "/v1/jobs/async/job/{job_id}/report" in route_paths
    assert len(_submit_routes(recovered_app)) == 1

    with TestClient(recovered_app) as client:
        health = client.get("/health")
        submit = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "question"},
        )

    assert health.status_code == 200
    assert submit.status_code == 200
    assert bootstrap.await_count == 2
    assert readiness.await_count == 2
    submit_job.assert_awaited_once()
    encryption_submit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param({"reason": "async_jobs_unavailable", "db": "unreachable"}, id="database"),
        pytest.param({"reason": "async_jobs_unavailable", "db": "ok"}, id="scheduler"),
    ],
)
async def test_dynamic_readiness_loss_blocks_then_recovers(
    route_dependencies,
    monkeypatch,
    tmp_path,
    failure,
):
    jobs_routes, submit_job, encryption_submit = route_dependencies
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    store = _job_store(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}")
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", AsyncMock())
    readiness = AsyncMock(side_effect=[failure, None])
    monkeypatch.setattr(jobs_routes, "_probe_async_job_readiness", readiness)
    app = FastAPI()
    await jobs_routes.register_job_routes(
        app,
        _builder(),
        _worker(job_store=store, config_path=str(config_path), db_url=f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"),
    )

    with TestClient(app) as client:
        unavailable_response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "question"},
        )
        assert unavailable_response.status_code == 503
        assert unavailable_response.json() == {"detail": "Async job submission is currently unavailable"}
        submit_job.assert_not_awaited()
        encryption_submit.assert_not_awaited()

        recovered_response = client.post(
            "/v1/jobs/async/submit",
            json={"agent_type": "deep_researcher", "input": "question"},
        )

    assert recovered_response.status_code == 200
    assert recovered_response.json()["job_id"] == "job-1"
    assert recovered_response.json()["status"] == "submitted"
    assert readiness.await_count == 2
    submit_job.assert_awaited_once()
    encryption_submit.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_zero_worker_scheduler_is_ready_and_scales_for_http_submissions(monkeypatch, tmp_path):
    from dask.distributed import LocalCluster

    import deep_researcher_agent.auth
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs import submit
    from nat.front_ends.fastapi.async_jobs import job_store as job_store_module

    stores = []

    class TrackingJobStore(job_store_module.JobStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            stores.append(self)

    cluster = await asyncio.to_thread(
        LocalCluster,
        n_workers=0,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n", encoding="utf-8")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    principal = Principal(type="jwt", sub="user-1", email="user@example.com")

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", cluster.scheduler_address)
    monkeypatch.setenv("NAT_JOB_STORE_DB_URL", db_url)
    monkeypatch.setenv("NAT_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("NAT_USE_DASK_THREADS", "1")
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", AsyncMock())
    monkeypatch.setattr(jobs_routes, "require_verified_principal", MagicMock(return_value=principal))
    monkeypatch.setattr(deep_researcher_agent.auth, "get_auth_token", lambda: None)
    monkeypatch.setattr(submit, "run_agent_job", _successful_agent_execution)
    monkeypatch.setattr(job_store_module, "JobStore", TrackingJobStore)

    store = TrackingJobStore(scheduler_address=cluster.scheduler_address, db_url=db_url)
    worker = _worker(job_store=store, config_path=str(config_path), db_url=db_url)
    worker._scheduler_address = cluster.scheduler_address
    app = FastAPI()

    try:
        await jobs_routes.register_job_routes(app, _builder(), worker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200, health.text
            scheduler_info = await asyncio.to_thread(store.dask_client.scheduler_info)
            assert scheduler_info["workers"] == {}

            await asyncio.to_thread(cluster.scale, 1)
            await asyncio.to_thread(store.dask_client.wait_for_workers, 1, timeout=10)
            for agent_type in ("shallow_researcher", "deep_researcher"):
                job_id = f"readiness-{agent_type}"
                response = await client.post(
                    "/v1/jobs/async/submit",
                    json={"agent_type": agent_type, "input": "question", "job_id": job_id},
                )
                assert response.status_code == 200, response.text
                persisted = await store.get_job(job_id)
                assert persisted is not None
                assert persisted.status == "submitted"
    finally:
        for tracked_store in stores:
            if getattr(tracked_store, "_dask_client", None) is not None:
                await asyncio.to_thread(tracked_store._dask_client.close)
        await asyncio.to_thread(cluster.close)
