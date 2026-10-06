# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_researcher_agent.auth import Principal


@pytest.fixture
async def report_edit_app(monkeypatch):
    """Build a minimal async-jobs app with patched storage and submission side effects."""
    import deep_researcher_agent.auth
    import deep_researcher_api.routes.jobs as jobs_routes
    from deep_researcher_api.jobs import access
    from deep_researcher_api.jobs import event_store
    from deep_researcher_api.jobs import submit

    principal = Principal(type="jwt", sub="user-1", email="user@example.com")
    parent_job = SimpleNamespace(
        status="success",
        output={
            "report": "# Parent Report\n\nKeep this.\n\n## Sources\n\n[1] https://example.com/source",
        },
        error=None,
        created_at=None,
    )
    job_store = MagicMock()
    authorize_job_access = AsyncMock(return_value=parent_job)
    submit_agent_job = AsyncMock(return_value="child-job-1")

    monkeypatch.setattr(jobs_routes, "_start_periodic_cleanup", MagicMock())
    monkeypatch.setattr(jobs_routes, "_reap_ghost_jobs", AsyncMock())
    monkeypatch.setattr(jobs_routes, "_is_readable_regular_file", MagicMock(return_value=True))
    monkeypatch.setattr(jobs_routes, "_bootstrap_async_job_storage", AsyncMock())
    monkeypatch.setattr(jobs_routes, "_probe_async_job_readiness", AsyncMock(return_value=None))
    monkeypatch.setattr(jobs_routes, "require_verified_principal", lambda: principal)
    monkeypatch.setattr(access, "authorize_job_access", authorize_job_access)
    monkeypatch.setattr(access, "ensure_job_access_table", MagicMock())
    monkeypatch.setattr(event_store.EventStore, "_ensure_table_exists", MagicMock())
    monkeypatch.setattr(event_store.EventStore, "get_events_async", AsyncMock(return_value=[]))
    monkeypatch.setattr(submit, "submit_agent_job", submit_agent_job)
    monkeypatch.setattr(deep_researcher_agent.auth, "get_auth_token", lambda: "token-1")

    worker = SimpleNamespace(
        _dask_available=True,
        _job_store=job_store,
        _scheduler_address="tcp://localhost:8786",
        _db_url="sqlite:///./test.db",
        _config_file_path="config.yml",
        _log_level=20,
        _use_dask_threads=False,
        _front_end_config=SimpleNamespace(expiry_seconds=86400),
    )
    builder = MagicMock()

    app = FastAPI()
    await jobs_routes.register_job_routes(app, builder, worker)
    return app, parent_job, authorize_job_access, submit_agent_job, principal, job_store


@pytest.mark.asyncio
async def test_report_edit_authorizes_parent_and_submits_internal_child(report_edit_app):
    app, parent_job, authorize_job_access, submit_agent_job, principal, job_store = report_edit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Remove the final paragraph."},
        )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "child-job-1",
        "parent_job_id": "parent-job-1",
        "status": "submitted",
        "agent_type": "report_rewriter",
    }
    authorize_job_access.assert_awaited_once_with(job_store, "sqlite:///./test.db", "parent-job-1", principal)
    submit_agent_job.assert_awaited_once()
    kwargs = submit_agent_job.await_args.kwargs
    assert kwargs["agent_type"] == "report_rewriter"
    assert kwargs["input_text"] == "Remove the final paragraph."
    assert kwargs["owner"] == "user@example.com"
    assert kwargs["principal"] == principal
    assert kwargs["data_sources"] == []
    assert kwargs["auth_token"] == "token-1"
    assert kwargs["initial_files"]["/shared/original_report.md"] == parent_job.output["report"]
    assert kwargs["initial_files"]["/shared/edit_instruction.txt"] == "Remove the final paragraph."
    assert json.loads(kwargs["initial_files"]["/shared/parent_report_context.json"])["parent_job_id"] == "parent-job-1"
    assert kwargs["output_metadata"] == {
        "parent_job_id": "parent-job-1",
        "interaction_action": "edit",
        "result_kind": "report",
    }


@pytest.mark.asyncio
async def test_report_edit_decrypts_parent_output_before_submitting_child(report_edit_app, monkeypatch):
    """Encrypted parent reports remain available to the report-follow-up workflow."""
    import base64

    from deep_researcher_api.jobs import crypto

    app, parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv(
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(b"k" * crypto.DEK_BYTES).decode(),
    )
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID", "test-key")
    crypto.reset_content_encryption_manager_for_tests()
    parent_report = "# Encrypted parent\n\n## Sources\n\n[1] https://example.com/source\n"
    parent_job.output = crypto.create_job_content_cipher("parent-job-1").encrypt_output_json(
        json.dumps({"report": parent_report})
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/jobs/async/job/parent-job-1/report/edit",
                json={"input": "Shorten it."},
            )
    finally:
        crypto.reset_content_encryption_manager_for_tests()

    assert response.status_code == 200
    assert submit_agent_job.await_args.kwargs["initial_files"]["/shared/original_report.md"] == parent_report.strip()


@pytest.mark.asyncio
async def test_report_edit_rejects_plaintext_parent_in_encrypted_mode(report_edit_app, monkeypatch):
    """Report follow-up fails closed when encrypted mode encounters plaintext parent output."""
    import base64

    from deep_researcher_api.jobs import crypto

    app, parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv(
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(b"k" * crypto.DEK_BYTES).decode(),
    )
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID", "test-key")
    crypto.reset_content_encryption_manager_for_tests()
    parent_job.output = {"report": "# Plaintext parent"}

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/jobs/async/job/parent-job-1/report/edit",
                json={"input": "Shorten it."},
            )
    finally:
        crypto.reset_content_encryption_manager_for_tests()

    assert response.status_code == 500
    assert response.json()["detail"] == "Parent report data is invalid"
    submit_agent_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_edit_returns_503_when_parent_decrypt_is_unavailable(report_edit_app, monkeypatch):
    from deep_researcher_api.jobs import crypto
    from deep_researcher_api.jobs import report_context

    app, _parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    monkeypatch.setattr(
        report_context,
        "read_job_output_async",
        AsyncMock(side_effect=crypto.ContentEncryptionUnavailable("vault unavailable")),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Shorten it."},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Content encryption is unavailable"
    submit_agent_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_edit_returns_503_when_submit_encryption_is_unavailable(report_edit_app, caplog):
    from deep_researcher_api.jobs.crypto import ContentEncryptionUnavailable

    app, _parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    submit_agent_job.side_effect = ContentEncryptionUnavailable("vault unavailable")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Shorten it."},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Content encryption is not ready"
    assert "vault unavailable" not in caplog.text
    submit_agent_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_edit_returns_500_when_submit_encryption_config_is_invalid(report_edit_app, caplog):
    from deep_researcher_api.jobs.crypto import ContentEncryptionConfigError

    app, _parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    submit_agent_job.side_effect = ContentEncryptionConfigError("sensitive configuration detail")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Shorten it."},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Content encryption configuration is invalid"
    assert "sensitive configuration detail" not in caplog.text
    submit_agent_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_edit_rejects_incomplete_parent(report_edit_app):
    app, parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    parent_job.status = "running"

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Remove the final paragraph."},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "Parent job is not complete: parent-job-1"
    submit_agent_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_edit_rejects_parent_without_durable_report(report_edit_app):
    app, parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    parent_job.output = {}

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Remove the final paragraph."},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "Parent job has no durable report: parent-job-1"
    submit_agent_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_edit_denies_cross_user_access_with_404(report_edit_app):
    """A non-owner is rejected (404 from authorization) before any child job is submitted."""
    from fastapi import HTTPException

    app, _parent_job, authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    authorize_job_access.side_effect = HTTPException(status_code=404, detail="Job not found: parent-job-1")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Remove the final paragraph."},
        )

    assert response.status_code == 404
    submit_agent_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_edit_rejects_blank_input(report_edit_app):
    """Whitespace-only edit instructions are rejected at the boundary (422), not submitted."""
    app, _parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "   "},
        )

    assert response.status_code == 422
    submit_agent_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_edit_job_id_collision_returns_409(report_edit_app):
    """A caller-supplied job_id that collides returns 409, never a 500/data-loss path."""
    from deep_researcher_api.jobs.submit import JobIdConflictError

    app, _parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app
    submit_agent_job.side_effect = JobIdConflictError("Job already exists: existing-child")

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Remove the final paragraph.", "job_id": "existing-child"},
        )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_report_edit_submits_internal_agent_with_allow_internal(report_edit_app):
    """The endpoint must opt in to the internal report_rewriter agent explicitly."""
    app, _parent_job, _authorize_job_access, submit_agent_job, _principal, _job_store = report_edit_app

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/async/job/parent-job-1/report/edit",
            json={"input": "Shorten it."},
        )

    assert response.status_code == 200
    assert submit_agent_job.await_args.kwargs["allow_internal"] is True


@pytest.mark.asyncio
async def test_job_report_response_includes_report_interaction_metadata(report_edit_app):
    app, child_job, authorize_job_access, _submit_agent_job, principal, job_store = report_edit_app
    child_job.output = {
        "report": "# Revised",
        "parent_job_id": "parent-job-1",
        "interaction_action": "edit",
        "result_kind": "report",
    }

    with TestClient(app) as client:
        response = client.get("/v1/jobs/async/job/child-job-1/report")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "child-job-1",
        "has_report": True,
        "report": "# Revised",
        "parent_job_id": "parent-job-1",
        "interaction_action": "edit",
        "result_kind": "report",
    }
    # The GET report route must authorize the caller for THIS job id (the fixture
    # mock returns the same object for any id, so prove the access check happened).
    authorize_job_access.assert_awaited_once_with(job_store, "sqlite:///./test.db", "child-job-1", principal)
