# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from deep_researcher_agent.auth import Principal
from deep_researcher_api.mcp_auth.nat_provider import NatMcpAuthProvider
from deep_researcher_api.mcp_auth.nat_provider import OAuthSourceSettings
from deep_researcher_api.mcp_auth.nat_provider import _auth_result_from_token
from deep_researcher_api.mcp_auth.provider import ProtectedSourceAuthProvider
from deep_researcher_api.mcp_auth.provider import principal_user_id
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
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        redirect_uri="https://deep-researcher.example/v1/auth/mcp/gdrive/callback",
    )


@pytest.fixture
def provider() -> tuple[NatMcpAuthProvider, InMemoryTokenStorage]:
    store = InMemoryTokenStorage()
    prov = NatMcpAuthProvider(
        settings_by_source={"gdrive": _settings()},
        token_storage_resolver=lambda _s: store,
    )
    return prov, store


def test_satisfies_protocol(provider):
    prov, _ = provider
    assert isinstance(prov, ProtectedSourceAuthProvider)


def test_status_not_connected_initially(provider):
    prov, _ = provider
    state = asyncio.run(prov.get_status(PRINCIPAL, "gdrive"))
    assert state.status == "not_connected"


def test_status_error_for_unconfigured_source(provider):
    prov, _ = provider
    state = asyncio.run(prov.get_status(PRINCIPAL, "not-configured"))
    assert state.status == "error" and state.last_error


def test_start_auth_mints_provider_url_with_pkce_and_state(provider):
    prov, _ = provider
    challenge = asyncio.run(prov.start_auth(PRINCIPAL, "gdrive"))
    assert challenge.auth_url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert f"state={challenge.state}" in challenge.auth_url
    assert "code_challenge=" in challenge.auth_url
    assert "client_id=client-123" in challenge.auth_url
    assert challenge.state in prov._pending  # pending flow registered for the callback
    # Back-compat: with no resource configured, no resource indicator is added.
    assert "resource=" not in challenge.auth_url


def test_start_auth_includes_resource_when_set():
    # Mirrors NAT's MCPOAuth2Provider, which adds the RFC 8707 resource indicator to
    # the authorize request (authorization_kwargs={"resource": ...}).
    from urllib.parse import parse_qs
    from urllib.parse import urlparse

    settings = _settings().model_copy(update={"resource": "https://maas.example/maas/gdrive/mcp"})
    prov = NatMcpAuthProvider(
        settings_by_source={"gdrive": settings},
        token_storage_resolver=lambda _s: InMemoryTokenStorage(),
    )
    challenge = asyncio.run(prov.start_auth(PRINCIPAL, "gdrive"))
    params = parse_qs(urlparse(challenge.auth_url).query)
    assert params.get("resource") == ["https://maas.example/maas/gdrive/mcp"]


def test_connected_after_token_stored(provider):
    prov, store = provider
    expires = datetime.now(UTC) + timedelta(hours=1)
    asyncio.run(
        store.store(
            principal_user_id(PRINCIPAL),
            _auth_result_from_token({"access_token": "tok", "refresh_token": "r", "expires_at": expires.timestamp()}),
        )
    )
    state = asyncio.run(prov.get_status(PRINCIPAL, "gdrive"))
    assert state.status == "connected"
    assert abs((state.expires_at - expires).total_seconds()) < 2


def test_expired_token_reports_expired(provider):
    prov, store = provider
    past = datetime.now(UTC) - timedelta(minutes=5)
    asyncio.run(
        store.store(
            principal_user_id(PRINCIPAL),
            _auth_result_from_token({"access_token": "old", "expires_at": past.timestamp()}),
        )
    )
    state = asyncio.run(prov.get_status(PRINCIPAL, "gdrive"))
    assert state.status == "expired"


def test_require_connected_blocks_only_disconnected_protected(provider):
    prov, _ = provider
    blocked = asyncio.run(prov.require_connected(PRINCIPAL, ["gdrive", "web_search"]))
    # web_search is not configured/protected -> ignored; gdrive blocked with an auth_url
    assert [c.source_id for c in blocked] == ["gdrive"]
    assert blocked[0].auth_url.startswith("https://accounts.google.com")


def test_complete_callback_exchanges_and_stores(provider, monkeypatch):
    prov, store = provider
    challenge = asyncio.run(prov.start_auth(PRINCIPAL, "gdrive"))

    # Stub the network token exchange on the pending flow's authlib client.
    flow = prov._pending[challenge.state]
    expires = datetime.now(UTC) + timedelta(hours=1)
    flow.client.fetch_token = AsyncMock(
        return_value={"access_token": "fresh", "refresh_token": "r", "expires_at": expires.timestamp()}
    )
    flow.client.aclose = AsyncMock()

    completed = asyncio.run(
        prov.complete_callback(challenge.state, f"https://deep-researcher.example/cb?code=abc&state={challenge.state}")
    )
    assert completed == "gdrive"
    # Token now retrievable under the principal's key -> status connected
    state = asyncio.run(prov.get_status(PRINCIPAL, "gdrive"))
    assert state.status == "connected"
    assert challenge.state not in prov._pending


def test_complete_callback_unknown_state_raises(provider):
    prov, _ = provider
    with pytest.raises(KeyError):
        asyncio.run(prov.complete_callback("nope", "https://deep-researcher.example/cb?code=x&state=nope"))


def test_prune_closes_expired_flow_client():
    # An abandoned challenge (started, never completed) must have its
    # AsyncOAuth2Client closed when pruned, otherwise it leaks httpx connections.
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    prov = NatMcpAuthProvider(
        settings_by_source={"gdrive": _settings()},
        token_storage_resolver=lambda _s: InMemoryTokenStorage(),
        challenge_ttl=timedelta(minutes=5),
        now=lambda: clock[0],
    )
    challenge = asyncio.run(prov.start_auth(PRINCIPAL, "gdrive"))
    closer = AsyncMock()
    prov._pending[challenge.state].client.aclose = closer

    # Advance past the TTL so the flow is expired, then trigger a prune.
    clock[0] = clock[0] + timedelta(minutes=10)
    asyncio.run(prov.start_auth(PRINCIPAL, "gdrive"))

    closer.assert_awaited_once()
    assert challenge.state not in prov._pending
