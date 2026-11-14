# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP route tests for async job endpoints in deep_researcher_api.routes.jobs.

These tests register the real nested route handlers via ``register_job_routes``
and exercise the actual ``get_job_report`` closure. Only external I/O is mocked
(``job_store.get_job``, background tasks, table bootstrap). Auth and access
control run through the real ``require_verified_principal`` and
``authorize_job_access`` implementations.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi import HTTPException

from deep_researcher_agent.auth import Principal
from deep_researcher_api.jobs.access import create_job_access
from deep_researcher_api.jobs.access import ensure_job_access_table

# Routes tested in this file:
# get_job_report            - /{job_id}/report (GET)
# get_job_status            - /{job_id} (GET)
# TODO: get_job_artifacts   - /{job_id}/state (GET)
#
# Routes that modify or create new jobs are not tested here.

# These tests bypass the FastAPI router and call the endpoint functions directly.
# This allows us to test the endpoint code without the complexity of routing network calls.


async def _call_endpoint(app: FastAPI, path: str, method: str, *args) -> Any:
    """Call the endpoint function directly and return the response."""
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return await route.endpoint(*args)  # pyright: ignore[reportAttributeAccessIssue]
    raise AssertionError(f"Route not found: {method} {path}")


@pytest.fixture
async def job_report_app(monkeypatch, tmp_path):
    """Register real job routes; mock only job_store I/O and startup side effects."""
    import deep_researcher_api.jobs.access as job_access_mod
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs import event_store

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setattr(job_access_mod, "get_current_principal", lambda: None)

    job_access_mod._job_access_schema_initialized.clear()
    event_store.EventStore._tables_initialized.clear()

    # This job object can be modified before calling _call_endpoint.
    # This allows us to test the endpoint code with different job states.
    job = SimpleNamespace(
        status="success",
        output={"report": "# Research Report\n\nFindings here."},
        error=None,
        created_at=None,
    )
    job_store = MagicMock()
    job_store.get_job = AsyncMock(return_value=job)

    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", AsyncMock())
    monkeypatch.setattr(jobs_routes, "_is_readable_regular_file", MagicMock(return_value=True))
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", AsyncMock())
    monkeypatch.setattr(jobs_routes, "_probe_async_job_readiness", AsyncMock(return_value=None))
    monkeypatch.setattr(event_store.EventStore, "_ensure_table_exists", MagicMock())

    # Mock the worker FastApiFrontEndPluginWorker object.
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

    app = FastAPI()
    await jobs_routes.register_job_routes(app, MagicMock(), worker)  # pyright: ignore[reportArgumentType]
    return app, job, job_store, db_url


@pytest.mark.asyncio
async def test_get_job_report_returns_report(job_report_app):
    """Test that the get_job_report endpoint returns the correct report."""
    app, job, job_store, _db_url = job_report_app

    response = await _call_endpoint(app, "/v1/jobs/async/job/{job_id}/report", "GET", "job-1")

    assert response.model_dump() == {
        "job_id": "job-1",
        "has_report": True,
        "report": job.output["report"],
        "parent_job_id": None,
        "interaction_action": None,
        "result_kind": None,
    }
    job_store.get_job.assert_awaited_once_with("job-1")


@pytest.mark.asyncio
async def test_get_job_report_has_report_false_when_missing(job_report_app):
    """Test that the get_job_report endpoint returns False when the report is missing."""
    app, job, job_store, _db_url = job_report_app
    job.output = {}

    response = await _call_endpoint(app, "/v1/jobs/async/job/{job_id}/report", "GET", "job-1")

    assert response.model_dump() == {
        "job_id": "job-1",
        "has_report": False,
        "report": None,
        "parent_job_id": None,
        "interaction_action": None,
        "result_kind": None,
    }
    job_store.get_job.assert_awaited_once_with("job-1")


@pytest.mark.asyncio
async def test_get_job_report_includes_interaction_metadata(job_report_app):
    """Test that the get_job_report endpoint includes interaction metadata."""
    app, job, job_store, _db_url = job_report_app
    job.output = {
        "report": "# Revised",
        "parent_job_id": "parent-job-1",
        "interaction_action": "edit",
        "result_kind": "report",
    }

    response = await _call_endpoint(app, "/v1/jobs/async/job/{job_id}/report", "GET", "child-job-1")

    assert response.model_dump() == {
        "job_id": "child-job-1",
        "has_report": True,
        "report": "# Revised",
        "parent_job_id": "parent-job-1",
        "interaction_action": "edit",
        "result_kind": "report",
    }
    job_store.get_job.assert_awaited_once_with("child-job-1")


@pytest.mark.asyncio
async def test_get_job_report_not_found(job_report_app):
    """Test that the get_job_report endpoint returns a 404 when the job is not found."""
    app, _job, job_store, _db_url = job_report_app
    job_store.get_job = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as e:
        await _call_endpoint(app, "/v1/jobs/async/job/{job_id}/report", "GET", "missing-job")
        assert e.value.status_code == 404
        assert e.value.detail == "Job not found: missing-job"
        job_store.get_job.assert_awaited_once_with("missing-job")


@pytest.mark.asyncio
async def test_get_job_report_requires_verified_principal(job_report_app, monkeypatch):
    """Test that the get_job_report endpoint requires a verified principal."""
    app, _job, job_store, _db_url = job_report_app
    import deep_researcher_api.jobs.access as job_access_mod

    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setattr(job_access_mod, "get_current_principal", lambda: None)

    with pytest.raises(HTTPException) as e:
        await _call_endpoint(app, "/v1/jobs/async/job/{job_id}/report", "GET", "job-1")
        assert e.value.status_code == 404
        assert e.value.detail == "Job not found: job-1"

        job_store.get_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_job_report_denies_cross_user_when_auth_required(job_report_app, monkeypatch):
    """Test that the get_job_report endpoint denies cross-user access when authentication is required."""
    app, _job, job_store, db_url = job_report_app
    import deep_researcher_api.jobs.access as job_access_mod

    owner = Principal(type="jwt", sub="owner-1", email="owner@example.com")
    intruder = Principal(type="jwt", sub="intruder-1", email="intruder@example.com")

    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setattr(job_access_mod, "get_current_principal", lambda: intruder)
    ensure_job_access_table(db_url)
    create_job_access("job-1", owner, db_url)

    with pytest.raises(HTTPException) as e:
        await _call_endpoint(app, "/v1/jobs/async/job/{job_id}/report", "GET", "job-1")
        assert e.value.status_code == 404
        assert e.value.detail == "Job not found: job-1"
        job_store.get_job.assert_awaited_once_with("job-1")


@pytest.mark.asyncio
async def test_get_job_status_success(job_report_app):
    """Test that the get_job_status endpoint returns the correct status."""
    app, job, job_store, _db_url = job_report_app

    response = await _call_endpoint(app, "/v1/jobs/async/job/{job_id}", "GET", "job-1")

    assert response.model_dump() == {
        "job_id": "job-1",
        "status": "success",
        "agent_type": None,
        "error": None,
        "created_at": None,
    }
    job_store.get_job.assert_awaited_once_with("job-1")


@pytest.mark.asyncio
async def test_get_job_status_error(job_report_app):
    """Test that the get_job_status endpoint returns the correct status when an error occurs."""
    app, job, job_store, _db_url = job_report_app
    job.error = "An error occurred"
    job.status = "error"

    response = await _call_endpoint(app, "/v1/jobs/async/job/{job_id}", "GET", "job-1")

    assert response.model_dump() == {
        "job_id": "job-1",
        "status": "error",
        "agent_type": None,
        "error": "An error occurred",
        "created_at": None,
    }
    job_store.get_job.assert_awaited_once_with("job-1")


@pytest.fixture
async def artifact_download_app(job_report_app, monkeypatch, tmp_path):
    """Exercise real SQL metadata and S3 streaming through the HTTP route."""
    from deep_researcher_agent.agents.deep_researcher.sandbox import artifacts

    app, _job, _job_store, _db_url = job_report_app
    client = MagicMock()
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    body = MagicMock()
    body.iter_chunks.return_value = iter([b"artifact bytes"])
    client.get_object.return_value = {"Body": body}
    blob_store = artifacts.S3ArtifactBlobStore(bucket="test-artifacts", client=client)
    store = artifacts.SqlArtifactStore(f"sqlite:///{tmp_path / 'artifacts.db'}", blob_store=blob_store)
    artifact = artifacts.Artifact(
        artifact_id="art_" + "a" * 32,
        job_id="job-1",
        kind=artifacts.ArtifactKind.IMAGE,
        mime_type="image/png",
        filename="chart.png",
        sandbox_path="/tmp/chart.png",
        storage_uri="",
        sha256="a" * 64,
        size_bytes=14,
    )
    store.put(artifact, b"artifact bytes")
    monkeypatch.setattr(artifacts, "build_artifact_store", lambda _url: store)
    return app, client, body, artifact


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "status"), [("missing", 404), ("denied", 500), ("access_denied", 500), (None, 200)])
async def test_artifact_download_errors_before_response_start(artifact_download_app, error, status):
    """The lazy SQL/S3 stack must resolve failures before committing HTTP 200."""
    import httpx
    from botocore.exceptions import ClientError

    app, s3, body, artifact = artifact_download_app
    private_location = "s3://private-bucket/private-key"
    if error == "missing":
        s3.get_object.side_effect = s3.exceptions.NoSuchKey(private_location)
    elif error == "denied":
        s3.get_object.side_effect = PermissionError(private_location)
    elif error == "access_denied":
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": private_location}}, "GetObject"
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get(f"/v1/jobs/async/job/job-1/artifacts/{artifact.artifact_id}/content")

    assert response.status_code == status
    assert "private-bucket" not in response.text
    assert "private-key" not in response.text
    if error is None:
        assert response.content == b"artifact bytes"
        body.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [b"one", b"two"]])
async def test_artifact_response_preserves_empty_and_chunked_content(chunks):
    """Prefetch preserves empty artifacts and every chunk without buffering all bytes."""
    from deep_researcher_api.routes.artifact_response import ArtifactStreamingResponse

    closed = []

    def content():
        """Record cleanup of the simulated provider stream."""
        try:
            yield from chunks
        finally:
            closed.append(True)

    sent = []

    async def send(message):
        """Capture the response status and bytes sent through ASGI."""
        sent.append(message)

    await ArtifactStreamingResponse(content())({"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send)
    assert sent[0]["status"] == 200
    assert b"".join(message.get("body", b"") for message in sent) == b"".join(chunks)
    assert closed == [True]


@pytest.mark.asyncio
async def test_artifact_response_does_not_send_second_status_on_late_error():
    """A read error after headers closes the iterator without attempting a 404."""
    from starlette.requests import ClientDisconnect

    from deep_researcher_api.routes.artifact_response import ArtifactStreamingResponse

    closed = []

    def content():
        """Record cleanup of the simulated provider stream."""
        try:
            yield b"first"
            raise FileNotFoundError("late read failure")
        finally:
            closed.append(True)

    sent = []

    async def send(message):
        """Capture the response status and bytes sent through ASGI."""
        sent.append(message)

    with pytest.raises(ClientDisconnect):
        await ArtifactStreamingResponse(content())({"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send)
    assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [200]
    assert closed == [True]


@pytest.mark.asyncio
async def test_artifact_response_closes_prefetch_when_cancelled():
    """Cancellation waits for a blocked first read and then closes its resource."""
    import threading

    import anyio

    from deep_researcher_api.routes.artifact_response import ArtifactStreamingResponse

    started = threading.Event()
    release = threading.Event()
    closed = []

    def content():
        """Record cleanup of the simulated provider stream."""
        try:
            started.set()
            assert release.wait(2)
            yield b"first"
        finally:
            closed.append(True)

    async def run():
        """Run the response under the task group's cancellation scope."""
        await ArtifactStreamingResponse(content())(
            {"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), AsyncMock()
        )

    async with anyio.create_task_group() as group:
        group.start_soon(run)
        assert await anyio.to_thread.run_sync(started.wait, 2)
        group.cancel_scope.cancel()
        release.set()
    assert closed == [True]
