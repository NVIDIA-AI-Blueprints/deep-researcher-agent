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

"""NAT-backed implementation of :class:`ProtectedSourceAuthProvider`.

This implementation never invokes NAT's blocking, browser-opening MCP auth
flow. Instead it mints the provider authorization URL with ``authlib`` (the
same library NAT uses), completes the ``code -> token`` exchange in Deep Researcher Agent's own
callback, and writes the resulting token into NAT's *public* token storage in
the exact shape NAT writes (``AuthResult`` with a ``BearerTokenCred`` and the
raw token dict incl. ``refresh_token``). The headless job-time
``per_user_mcp_client`` then finds that token via ``token_storage.retrieve``
and never has to authenticate interactively.

The per-source token storage MUST be a shared, persistent backend (an
``ObjectStore``) — the connect endpoint runs in the API process while the job
runs in a separate worker process, so in-memory storage would not be visible
across the boundary.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pkce
from authlib.integrations.httpx_client import AsyncOAuth2Client
from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr

from deep_researcher_agent.auth import Principal
from nat.authentication.token_storage import TokenStorageBase
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred

from .provider import SourceAuthChallenge
from .provider import SourceAuthState
from .provider import principal_user_id

logger = logging.getLogger(__name__)

_DEFAULT_CHALLENGE_TTL = timedelta(seconds=300)


class OAuthSourceSettings(BaseModel):
    """OAuth2 settings for one protected MCP source.

    These mirror the fields NAT's ``mcp_oauth2`` provider uses. ``mcp_server_id``
    is the NAT auth-provider/server key — the token written here must be
    retrievable by the same key the job-time provider uses (which is the per-user
    key within that server's token storage).
    """

    source_id: str
    mcp_server_id: str
    provider: str | None = None
    authorization_url: str
    token_url: str
    client_id: str
    client_secret: SecretStr | None = None
    scopes: list[str] = Field(default_factory=list)
    redirect_uri: str
    use_pkce: bool = True
    token_endpoint_auth_method: str = "client_secret_post"
    resource: str | None = Field(
        default=None,
        description=(
            "OAuth resource indicator (RFC 8707 / RFC 9728) added to the authorization request, matching NAT's "
            "MCPOAuth2Provider. Derived from the protected-resource metadata's `resource` (falling back to the "
            "server URL). Authorization-request only — NAT does not send it at token exchange."
        ),
    )


@dataclass
class _PendingFlow:
    source_id: str
    user_id: str
    client: AsyncOAuth2Client
    verifier: str | None
    settings: OAuthSourceSettings
    expires_at: datetime


@dataclass
class NatMcpAuthProvider:
    """Concrete :class:`ProtectedSourceAuthProvider` backed by NAT primitives.

    Args:
        settings_by_source: OAuth settings per protected source id.
        token_storage_resolver: Maps settings -> the NAT ``TokenStorageBase`` for
            that source. In production this resolves a shared ``ObjectStore``; in
            tests it can return an ``InMemoryTokenStorage`` per source.
        challenge_ttl: How long a minted ``auth_url`` / pending flow stays valid.
        now: Clock injection for tests.
    """

    settings_by_source: dict[str, OAuthSourceSettings]
    token_storage_resolver: Callable[[OAuthSourceSettings], TokenStorageBase]
    challenge_ttl: timedelta = _DEFAULT_CHALLENGE_TTL
    now: Callable[[], datetime] = field(default_factory=lambda: lambda: datetime.now(UTC))

    # DEPLOYMENT CONSTRAINT — single API replica (or sticky sessions):
    # ``_pending`` holds in-PROCESS OAuth flow state (PKCE verifier + authlib
    # client) keyed by the OAuth ``state``. The browser hits /connect on one
    # replica and the provider redirects /callback back to the API; the callback
    # MUST land on the SAME process that minted the state, or complete_callback
    # raises "Unknown or expired auth state". The Helm chart pins the backend to
    # replicas: 1, so this holds today. Scaling the API beyond one replica
    # requires either session affinity on the ingress/service (route a user's
    # /connect and /callback to the same pod) or moving pending-flow state to a
    # shared store (e.g. Redis). The token store is already shared/cross-process;
    # only this short-lived (challenge_ttl) pending state is process-local.
    _pending: dict[str, _PendingFlow] = field(default_factory=dict)
    _storage_cache: dict[str, TokenStorageBase] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ── storage helpers ──
    def _storage(self, settings: OAuthSourceSettings) -> TokenStorageBase:
        cached = self._storage_cache.get(settings.source_id)
        if cached is None:
            cached = self.token_storage_resolver(settings)
            self._storage_cache[settings.source_id] = cached
        return cached

    def is_protected(self, source_id: str) -> bool:
        """Whether this source is configured for per-user MCP OAuth."""
        return source_id in self.settings_by_source

    # ── ProtectedSourceAuthProvider ──
    async def get_status(self, principal: Principal, source_id: str) -> SourceAuthState:
        settings = self.settings_by_source.get(source_id)
        if settings is None:
            # Not configured for OAuth — treat as an error so the UI can surface it
            # rather than silently claiming "connected".
            return SourceAuthState(status="error", last_error="Source is not configured for MCP OAuth")

        user_id = principal_user_id(principal)
        try:
            auth_result = await self._storage(settings).retrieve(user_id)
        except Exception as exc:  # storage backend failure — report, don't crash the listing
            logger.warning("Token storage read failed for source=%s: %s", source_id, exc)
            return SourceAuthState(status="error", last_error="Could not read auth state")

        if auth_result is None or not auth_result.credentials:
            return SourceAuthState(status="not_connected")
        if auth_result.is_expired():
            return SourceAuthState(status="expired", expires_at=auth_result.token_expires_at)
        return SourceAuthState(status="connected", expires_at=auth_result.token_expires_at)

    async def start_auth(self, principal: Principal, source_id: str) -> SourceAuthChallenge:
        settings = self.settings_by_source.get(source_id)
        if settings is None:
            raise ValueError(f"Source '{source_id}' is not configured for MCP OAuth")

        user_id = principal_user_id(principal)
        state = secrets.token_urlsafe(24)

        client = AsyncOAuth2Client(
            client_id=settings.client_id,
            client_secret=(settings.client_secret.get_secret_value() if settings.client_secret else None),
            redirect_uri=settings.redirect_uri,
            scope=" ".join(settings.scopes) if settings.scopes else None,
            code_challenge_method="S256" if settings.use_pkce else None,
            token_endpoint_auth_method=settings.token_endpoint_auth_method,
        )

        verifier = challenge = None
        if settings.use_pkce:
            verifier, challenge = pkce.generate_pkce_pair()

        # RFC 8707 resource indicator on the authorize request, matching NAT's
        # MCPOAuth2Provider (authorization_kwargs={"resource": ...}). Authorize-only:
        # NAT does not send it at token exchange, so complete_callback omits it too.
        extra = {"resource": settings.resource} if settings.resource else {}
        auth_url, _ = client.create_authorization_url(
            settings.authorization_url,
            state=state,
            code_verifier=verifier if settings.use_pkce else None,
            code_challenge=challenge if settings.use_pkce else None,
            **extra,
        )

        expires_at = self.now() + self.challenge_ttl
        async with self._lock:
            stale = self._prune_locked()
            self._pending[state] = _PendingFlow(
                source_id=source_id,
                user_id=user_id,
                client=client,
                verifier=verifier,
                settings=settings,
                expires_at=expires_at,
            )
        await self._aclose_flows(stale)
        logger.info("Started MCP auth challenge for source=%s user=%s state=%s", source_id, user_id, state[:8])
        return SourceAuthChallenge(source_id=source_id, auth_url=auth_url, state=state, expires_at=expires_at)

    async def require_connected(
        self,
        principal: Principal,
        source_ids: list[str],
    ) -> list[SourceAuthChallenge]:
        blocked: list[SourceAuthChallenge] = []
        for source_id in source_ids:
            if not self.is_protected(source_id):
                continue  # unprotected / unknown sources never block submission
            state = await self.get_status(principal, source_id)
            if state.status == "connected":
                continue
            # Best-effort: mint an auth_url so the client can act immediately. If
            # minting fails, still report the source as blocked (connect_url only).
            try:
                challenge = await self.start_auth(principal, source_id)
            except Exception as exc:
                logger.warning("Could not mint auth_url during preflight for source=%s: %s", source_id, exc)
                challenge = SourceAuthChallenge(source_id=source_id, auth_url="", state="")
            blocked.append(challenge)
        return blocked

    # ── callback completion (Deep Researcher Agent owns the redirect route) ──
    async def complete_callback(self, state: str, authorization_response_url: str) -> str:
        """Exchange the callback's code for a token and persist it. Returns source_id.

        Raises ``KeyError`` for an unknown/expired state and propagates token
        exchange errors to the caller (the route maps them to an HTML error).
        """
        async with self._lock:
            stale = self._prune_locked()
            flow = self._pending.pop(state, None)
        await self._aclose_flows(stale)
        if flow is None:
            raise KeyError("Unknown or expired auth state")

        try:
            token = await flow.client.fetch_token(
                url=flow.settings.token_url,
                authorization_response=authorization_response_url,
                code_verifier=flow.verifier,
                state=state,
            )
        finally:
            await flow.client.aclose()

        auth_result = _auth_result_from_token(token)
        await self._storage(flow.settings).store(flow.user_id, auth_result)
        logger.info("Completed MCP auth for source=%s user=%s", flow.source_id, flow.user_id)
        return flow.source_id

    def _prune_locked(self) -> list[_PendingFlow]:
        """Pop expired flows and return them so the caller can close their clients.

        Returns the popped flows rather than discarding them: each holds an
        ``AsyncOAuth2Client`` (an httpx client pool) that must be ``aclose()``d to
        avoid leaking connection state. Pruning runs under ``self._lock`` and is
        synchronous, so the async close happens outside the lock via
        :meth:`_aclose_flows`.
        """
        now = self.now()
        expired = [s for s, f in self._pending.items() if f.expires_at <= now]
        return [self._pending.pop(s) for s in expired]

    @staticmethod
    async def _aclose_flows(flows: list[_PendingFlow]) -> None:
        for flow in flows:
            try:
                await flow.client.aclose()
            except Exception as exc:  # best-effort cleanup; never fail the caller
                logger.debug("Error closing expired auth flow client: %s", exc)


def _auth_result_from_token(token: dict) -> AuthResult:
    """Build a NAT ``AuthResult`` in the same shape NAT's OAuth provider writes."""
    expires_at: datetime | None = None
    if token.get("expires_at"):
        expires_at = datetime.fromtimestamp(float(token["expires_at"]), tz=UTC)
    access_token = token.get("access_token")
    if not access_token:
        raise ValueError("Token response missing access_token")
    return AuthResult(
        credentials=[BearerTokenCred(token=SecretStr(access_token))],
        token_expires_at=expires_at,
        raw=dict(token),
    )
