# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_researcher_agent.auth import Principal
from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import reset_registry
from deep_researcher_api.mcp_auth.nat_provider import NatMcpAuthProvider
from deep_researcher_api.mcp_auth.nat_provider import OAuthSourceSettings
from deep_researcher_api.mcp_auth.nat_provider import _auth_result_from_token
from deep_researcher_api.mcp_auth.provider import principal_user_id
from deep_researcher_api.routes import auth as auth_routes
from nat.authentication.token_storage import InMemoryTokenStorage

PRINCIPAL = Principal(type="jwt", sub="user-1")


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
def client(monkeypatch):
    store = InMemoryTokenStorage()
    provider = NatMcpAuthProvider(settings_by_source={"gdrive": _settings()}, token_storage_resolver=lambda _s: store)
    monkeypatch.setattr(auth_routes, "require_verified_principal", lambda: PRINCIPAL)
    app = FastAPI()
    auth_routes.register_mcp_auth_routes(app, provider)
    return TestClient(app), provider, store


def test_status_not_connected_offers_connect_url(client):
    tc, _, _ = client
    resp = tc.get("/v1/auth/mcp/gdrive/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_connected"
    assert body["connect_url"] == "/v1/auth/mcp/gdrive/connect"


def test_status_unknown_source_404(client):
    tc, _, _ = client
    assert tc.get("/v1/auth/mcp/web_search/status").status_code == 404  # not a protected source
    assert tc.get("/v1/auth/mcp/nope/status").status_code == 404


def test_connect_returns_auth_url_structured(client):
    tc, _, _ = client
    resp = tc.post("/v1/auth/mcp/gdrive/connect")
    assert resp.status_code == 200
    body = resp.json()
    # The auth_url is returned in a structured response (not UI-only) for all clients.
    assert body["status"] == "auth_required"
    assert body["auth_url"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")


def test_connect_when_already_connected(client):
    tc, provider, store = client
    import asyncio

    expires = datetime.now(UTC) + timedelta(hours=1)
    asyncio.run(
        store.store(
            principal_user_id(PRINCIPAL),
            _auth_result_from_token({"access_token": "t", "expires_at": expires.timestamp()}),
        )
    )
    body = tc.post("/v1/auth/mcp/gdrive/connect").json()
    assert body["status"] == "connected" and body.get("auth_url") is None


def test_connect_then_callback_completes(client):
    tc, provider, store = client
    auth_url = tc.post("/v1/auth/mcp/gdrive/connect").json()["auth_url"]
    state = auth_url.split("state=")[1].split("&")[0]

    # Stub the token exchange on the pending flow.
    flow = provider._pending[state]
    expires = datetime.now(UTC) + timedelta(hours=1)
    flow.client.fetch_token = AsyncMock(
        return_value={"access_token": "fresh", "refresh_token": "r", "expires_at": expires.timestamp()}
    )
    flow.client.aclose = AsyncMock()

    cb = tc.get(f"/v1/auth/mcp/gdrive/callback?code=abc&state={state}")
    assert cb.status_code == 200
    assert "Connected" in cb.text
    assert tc.get("/v1/auth/mcp/gdrive/status").json()["status"] == "connected"


def test_callback_unknown_state_renders_error(client):
    tc, _, _ = client
    cb = tc.get("/v1/auth/mcp/gdrive/callback?code=abc&state=bogus")
    assert cb.status_code == 400
    assert "expired" in cb.text.lower()


def test_callback_provider_denied(client):
    tc, _, _ = client
    cb = tc.get("/v1/auth/mcp/gdrive/callback?error=access_denied&state=whatever")
    assert cb.status_code == 400
    assert "denied" in cb.text.lower()


def test_callback_error_is_html_escaped(client):
    """A provider-controlled `error` must not inject markup into the Deep Researcher Agent origin."""
    tc, _, _ = client
    payload = "</p><script>document.body.dataset.pwned=1</script><p>"
    cb = tc.get("/v1/auth/mcp/gdrive/callback", params={"error": payload, "state": "whatever"})
    assert cb.status_code == 400
    # The raw script tag must not appear; the escaped form must.
    assert "<script>document.body.dataset.pwned=1</script>" not in cb.text
    assert "&lt;script&gt;" in cb.text


def test_preflight_blocks_disconnected_with_409(client, monkeypatch):
    import asyncio

    from deep_researcher_api.routes import jobs as jobs_routes

    _, provider, _ = client
    result = asyncio.run(jobs_routes._preflight_mcp_auth(provider, PRINCIPAL, ["gdrive", "web_search"]))
    assert result is not None and result.status_code == 409
    body = json.loads(result.body)
    assert body["error"] == "mcp_auth_required"
    assert len(body["sources"]) == 1
    src = body["sources"][0]
    assert src["source_id"] == "gdrive"
    assert src["connect_url"] == "/v1/auth/mcp/gdrive/connect"
    assert src["auth_url"].startswith("https://accounts.google.com")


def test_preflight_allows_when_connected(client):
    import asyncio

    from deep_researcher_api.routes import jobs as jobs_routes

    _, provider, store = client
    expires = datetime.now(UTC) + timedelta(hours=1)
    asyncio.run(
        store.store(
            principal_user_id(PRINCIPAL),
            _auth_result_from_token({"access_token": "t", "expires_at": expires.timestamp()}),
        )
    )
    result = asyncio.run(jobs_routes._preflight_mcp_auth(provider, PRINCIPAL, ["gdrive"]))
    assert result is None  # connected -> no block
