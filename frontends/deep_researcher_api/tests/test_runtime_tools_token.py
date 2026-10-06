# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from deep_researcher_api.mcp_auth import runtime_tools
from deep_researcher_api.mcp_auth.runtime_tools import PerUserMcpSourceUnavailableError
from nat.authentication.token_storage import InMemoryTokenStorage
from nat.builder.context import ContextState
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred

USER = "verified:alice"


def _auth(*, expired: bool) -> AuthResult:
    delta = timedelta(hours=-1) if expired else timedelta(hours=1)
    return AuthResult(
        credentials=[BearerTokenCred(token=SecretStr("tok"))],
        token_expires_at=datetime.now(UTC) + delta,
    )


async def _check(stored: AuthResult | None) -> tuple[bool, bool]:
    """Returns (usable, token_still_present_after)."""
    store = InMemoryTokenStorage()
    ContextState.get().user_id.set(USER)
    if stored is not None:
        await store.store(USER, stored)

    async def _fake_resolve(builder, cfg, source_id):
        return store

    # _token_usable does `from .factory import _resolve_token_storage`, so patch it there.
    with patch("deep_researcher_api.mcp_auth.factory._resolve_token_storage", _fake_resolve):
        usable = await runtime_tools._token_usable(builder=None, cfg=None, source_id="gdrive")
    still_there = (await store.retrieve(USER)) is not None
    return usable, still_there


def test_valid_token_is_usable_and_kept():
    usable, still_there = asyncio.run(_check(_auth(expired=False)))
    assert usable is True
    assert still_there is True


def test_expired_token_is_not_usable_and_invalidated():
    # The core fix: an expired token must be reported unusable AND deleted, so the
    # next get_status flips the card to Reconnect instead of false "connected".
    usable, still_there = asyncio.run(_check(_auth(expired=True)))
    assert usable is False
    assert still_there is False


def test_missing_token_is_not_usable():
    usable, still_there = asyncio.run(_check(None))
    assert usable is False
    assert still_there is False


def _gdrive_source():
    return SimpleNamespace(
        id="gdrive",
        per_user_auth=SimpleNamespace(required=True, auth_provider="mcp_oauth2_gdrive", tool_overrides={}),
    )


class _Builder:
    async def get_auth_provider(self, name):
        return SimpleNamespace(config=SimpleNamespace(server_url="https://mcp.example/mcp"))


async def _open(data_sources):
    async with AsyncExitStack() as stack:
        return await runtime_tools.open_per_user_mcp_tools(
            builder=_Builder(), data_sources=data_sources, exit_stack=stack
        )


async def _not_usable(*_args, **_kwargs):
    return False


def test_explicitly_selected_unusable_source_fails_closed():
    # A source the caller named specifically that we can't resolve (expired/missing
    # token) must raise rather than silently answering without it.
    with (
        patch("deep_researcher_api.mcp_auth.runtime_tools.get_all_sources", return_value=[_gdrive_source()]),
        patch("deep_researcher_api.mcp_auth.runtime_tools._token_usable", _not_usable),
    ):
        with pytest.raises(PerUserMcpSourceUnavailableError) as exc:
            asyncio.run(_open(["gdrive"]))
    assert "gdrive" in str(exc.value)


def test_all_sources_mode_skips_unusable_without_raising():
    # When data_sources is None ("all"), stay best-effort: an unusable source is
    # skipped, not fatal.
    with (
        patch("deep_researcher_api.mcp_auth.runtime_tools.get_all_sources", return_value=[_gdrive_source()]),
        patch("deep_researcher_api.mcp_auth.runtime_tools._token_usable", _not_usable),
    ):
        tools = asyncio.run(_open(None))
    assert tools == []
