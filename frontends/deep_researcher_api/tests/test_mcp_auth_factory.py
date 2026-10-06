# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import reset_registry
from deep_researcher_api.mcp_auth.factory import build_mcp_auth_provider


class _FakeProvider:
    """Stand-in for NAT's MCPOAuth2Provider with discovery already resolved."""

    def __init__(self, *, discover_ok: bool = True, client_id: str | None = "client-xyz"):
        self.config = SimpleNamespace(
            server_url="",  # empty -> factory probe is skipped (offline test)
            redirect_uri="https://deep-researcher.example/v1/auth/mcp/gdrive/callback",
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
            client_id=client_id,
            client_secret="shh",  # pragma: allowlist secret
            use_pkce=True,
            token_endpoint_auth_method="client_secret_post",
            token_storage_object_store="mcp_token_store",
        )
        self._discover_ok = discover_ok
        self._cached_endpoints = None
        self._cached_credentials = None

    async def _discover_and_register(self, response=None):
        if not self._discover_ok:
            raise RuntimeError("server unreachable")
        self._cached_endpoints = SimpleNamespace(
            authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
        )
        self._cached_credentials = SimpleNamespace(client_id="client-xyz", client_secret="shh")


class _FakeBuilder:
    def __init__(self, provider):
        self._provider = provider

    async def get_auth_provider(self, name):
        if self._provider is None:
            raise KeyError(name)
        return self._provider

    async def get_object_store_client(self, name):
        return SimpleNamespace(name=name)  # ObjectStoreTokenStorage only needs an object


@pytest.fixture(autouse=True)
def _registry():
    reset_registry()
    yield
    reset_registry()


def _register_gdrive():
    populate_from_config(
        [
            {
                "id": "gdrive",
                "name": "Google Drive",
                "requires_auth": True,
                "per_user_auth": {
                    "required": True,
                    "provider": "google",
                    "mcp_server_id": "gdrive",
                    "auth_provider": "mcp_oauth2_gdrive",
                },
            },
        ]
    )


def test_factory_resolves_settings_from_nat_provider():
    _register_gdrive()
    provider = build_from(_FakeBuilder(_FakeProvider()))
    assert provider.is_protected("gdrive")
    settings = provider.settings_by_source["gdrive"]
    assert settings.authorization_url == "https://accounts.google.com/o/oauth2/v2/auth"
    assert settings.token_url == "https://oauth2.googleapis.com/token"
    assert settings.client_id == "client-xyz"
    assert settings.redirect_uri.endswith("/v1/auth/mcp/gdrive/callback")
    assert settings.scopes == ["https://www.googleapis.com/auth/drive.readonly"]


def test_factory_skips_source_when_provider_missing():
    _register_gdrive()
    provider = build_from(_FakeBuilder(None))  # get_auth_provider raises
    assert not provider.is_protected("gdrive")  # left unconfigured -> status will be 'error'


def test_factory_skips_source_when_discovery_fails():
    _register_gdrive()
    provider = build_from(_FakeBuilder(_FakeProvider(discover_ok=False, client_id=None)))
    assert not provider.is_protected("gdrive")


def test_factory_skips_second_source_sharing_token_store():
    """Two protected sources must not share one token-storage object store.

    NAT keys tokens per user only, so a shared store would let the sources
    overwrite each other's credentials. The factory fails closed: the first
    source claims the store, the second is skipped (left unconfigured).
    """
    populate_from_config(
        [
            {
                "id": "gdrive",
                "name": "Google Drive",
                "requires_auth": True,
                "per_user_auth": {"required": True, "mcp_server_id": "gdrive", "auth_provider": "mcp_oauth2_gdrive"},
            },
            {
                "id": "notion",
                "name": "Notion",
                "requires_auth": True,
                "per_user_auth": {"required": True, "mcp_server_id": "notion", "auth_provider": "mcp_oauth2_notion"},
            },
        ]
    )
    # Both auth providers resolve to a store named "mcp_token_store" (the fake's
    # default), so the second source collides with the first.
    provider = build_from(_FakeBuilder(_FakeProvider()))
    assert provider.is_protected("gdrive")
    assert not provider.is_protected("notion")


def test_factory_ignores_unprotected_sources():
    populate_from_config([{"id": "web_search", "name": "Web", "description": "x"}])
    provider = build_from(_FakeBuilder(_FakeProvider()))
    assert provider.settings_by_source == {}


def build_from(builder):
    return asyncio.run(build_mcp_auth_provider(builder))
