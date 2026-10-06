# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""submit_agent_job() must run the per-user MCP auth preflight before enqueue.

This guards the programmatic submit path (e.g. the chat researcher's async
deep-research submit), which bypasses the REST route's 409 preflight. The check
lives in submit_agent_job so both paths share one chokepoint.
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from deep_researcher_agent.auth import Principal
from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import reset_registry
from deep_researcher_api.jobs import submit as submit_mod
from deep_researcher_api.jobs.event_store import EventStore
from deep_researcher_api.mcp_auth import active as active_mod
from deep_researcher_api.mcp_auth.nat_provider import NatMcpAuthProvider
from deep_researcher_api.mcp_auth.nat_provider import OAuthSourceSettings
from deep_researcher_api.mcp_auth.nat_provider import _auth_result_from_token
from deep_researcher_api.mcp_auth.preflight import McpAuthRequiredError
from deep_researcher_api.mcp_auth.provider import principal_user_id
from nat.authentication.token_storage import InMemoryTokenStorage

PRINCIPAL = Principal(type="jwt", sub="user-1", email="u@example.com")


def _settings() -> OAuthSourceSettings:
    return OAuthSourceSettings(
        source_id="gdrive",
        mcp_server_id="gdrive",
        provider="google",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        client_id="client-123",
        client_secret="secret",  # pragma: allowlist secret
        scopes=["drive.readonly"],
        redirect_uri="https://deep-researcher.example/v1/auth/mcp/gdrive/callback",
    )


class _FakeJobStore:
    submitted = False

    def __init__(self, **_kwargs):
        pass

    def ensure_job_id(self, job_id):
        return job_id or "job-1"

    async def submit_job(self, *, job_id, expiry_seconds, job_fn, job_args):
        _FakeJobStore.submitted = True


@pytest.fixture(autouse=True)
def registry():
    reset_registry()
    populate_from_config(
        [
            {"id": "web_search", "name": "Web Search", "description": "x"},
            {
                "id": "gdrive",
                "name": "Google Drive",
                "description": "Drive",
                "requires_auth": True,
                "per_user_auth": {"required": True, "provider": "google", "mcp_server_id": "gdrive"},
            },
        ]
    )
    yield
    reset_registry()


@pytest.fixture
def patched(monkeypatch, tmp_path):
    import nat.front_ends.fastapi.async_jobs.job_store as js_mod

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setenv("NAT_JOB_STORE_DB_URL", db_url)
    monkeypatch.setattr(js_mod, "JobStore", _FakeJobStore)
    monkeypatch.setattr(
        submit_mod,
        "get_agent_config",
        lambda _t: SimpleNamespace(class_path="pkg.mod.Agent", config_name="deep_research_agent", public=True),
    )
    monkeypatch.setattr(submit_mod, "create_job_access", MagicMock())
    _FakeJobStore.submitted = False
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE job_info (job_id TEXT PRIMARY KEY, status TEXT, is_expired BOOLEAN DEFAULT 0)"))
        conn.commit()

    store = InMemoryTokenStorage()
    provider = NatMcpAuthProvider(settings_by_source={"gdrive": _settings()}, token_storage_resolver=lambda _s: store)
    active_mod.set_active_mcp_auth_provider(provider)
    yield store
    active_mod.set_active_mcp_auth_provider(None)


def _submit(data_sources):
    return asyncio.run(
        submit_mod.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="u@example.com",
            principal=PRINCIPAL,
            data_sources=data_sources,
        )
    )


def test_blocks_when_protected_source_not_connected(patched):
    with pytest.raises(McpAuthRequiredError) as exc:
        _submit(["gdrive"])
    assert _FakeJobStore.submitted is False  # never enqueued
    assert [s.source_id for s in exc.value.response.sources] == ["gdrive"]
    assert "gdrive" in str(exc.value)  # user-facing message names the source


def test_blocks_when_data_sources_none_and_protected_disconnected(patched):
    # data_sources=None means "any tool" -> every protected source must be connected.
    with pytest.raises(McpAuthRequiredError):
        _submit(None)
    assert _FakeJobStore.submitted is False


def test_allows_unprotected_only_selection(patched):
    job_id = _submit(["web_search"])  # no protected source selected
    assert job_id == "job-1"
    assert _FakeJobStore.submitted is True


def test_allows_when_protected_source_connected(patched):
    store = patched
    expires = datetime.now(UTC) + timedelta(hours=1)
    asyncio.run(
        store.store(
            principal_user_id(PRINCIPAL),
            _auth_result_from_token({"access_token": "t", "expires_at": expires.timestamp()}),
        )
    )
    job_id = _submit(["gdrive"])
    assert job_id == "job-1"
    assert _FakeJobStore.submitted is True


def test_no_active_provider_skips_guard(patched):
    # When MCP auth is not configured in this process, there is nothing to enforce.
    active_mod.set_active_mcp_auth_provider(None)
    job_id = _submit(["gdrive"])
    assert job_id == "job-1"
    assert _FakeJobStore.submitted is True
