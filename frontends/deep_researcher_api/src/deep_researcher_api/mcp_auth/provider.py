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

"""Adapter boundary for protected MCP source auth.

This Protocol is the single seam between Deep Researcher Agent's product control plane and NAT's
MCP OAuth mechanics. All NAT calls live behind it (see ``nat_provider``) so the
route handlers and submit preflight never import NAT auth internals directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

from deep_researcher_agent.auth import Principal

AuthStatus = Literal["connected", "not_connected", "expired", "error"]


def principal_user_id(principal: Principal) -> str:
    """Canonical per-user key for NAT token storage.

    This MUST match the ``user_id`` that the headless worker sets on the NAT
    ``Context`` at job time, otherwise a token connected here will not be found
    when ``per_user_mcp_client`` looks it up during execution. Keying by
    ``type:sub`` keeps anonymous/no-auth principals distinct from verified ones.
    """
    return f"{principal.type}:{principal.sub}"


@dataclass(frozen=True)
class SourceAuthState:
    """Current per-user auth state for a single source."""

    status: AuthStatus
    expires_at: datetime | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class SourceAuthChallenge:
    """Result of starting (or reusing) an auth challenge for a source.

    ``auth_url`` is the provider login URL to hand to the client. ``state`` is
    the opaque OAuth state bound to (principal, source) used by the callback to
    complete the flow. ``expires_at`` is the challenge's expiry, not the token's.
    """

    source_id: str
    auth_url: str
    state: str
    expires_at: datetime | None = None


@runtime_checkable
class ProtectedSourceAuthProvider(Protocol):
    """Product-facing interface Deep Researcher Agent depends on; NAT lives behind the impl."""

    async def get_status(self, principal: Principal, source_id: str) -> SourceAuthState:
        """Return the current per-user auth state for ``source_id`` (read-only)."""
        ...

    async def start_auth(self, principal: Principal, source_id: str) -> SourceAuthChallenge:
        """Start or resume the OAuth flow and return a provider login URL."""
        ...

    async def require_connected(
        self,
        principal: Principal,
        source_ids: list[str],
    ) -> list[SourceAuthChallenge]:
        """Return a challenge per *blocked* source; empty list means all connected.

        Used by submit preflight. Implementations should attempt to include an
        ``auth_url`` when one can be safely minted, but may return a challenge
        with an empty ``auth_url`` if only ``connect_url`` can be offered.
        """
        ...
