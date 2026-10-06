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

"""Build a :class:`NatMcpAuthProvider` from the registry + NAT's mcp_oauth2 config.

The NAT ``authentication: mcp_oauth2`` provider is the single source of truth.
A protected source's registry entry points at it via ``per_user_auth.auth_provider``
(falling back to ``mcp_server_id``); Deep Researcher Agent resolves that provider through the
builder and derives:

  * the shared token storage (from ``token_storage_object_store``) — so the token
    Deep Researcher Agent writes at connect time is the same one the job-time ``per_user_mcp_client``
    reads;
  * ``redirect_uri`` / ``scopes`` / ``client_id`` / ``use_pkce``;
  * the authorize/token endpoints, via NAT's own discovery (RFC 8414 / RFC 9728
    well-known), reusing NAT rather than a parallel config surface.

Deployment notes for the cross-process design:
  1. ``token_storage_object_store`` must name a shared, persistent object store
     (the API process writes the token; a separate worker process reads it).
     This is REQUIRED.
  2. ``client_id`` (manual registration) is OPTIONAL. Without it, NAT performs
     dynamic client registration (DCR) — the connect flow works, but the refresh
     token is bound to the connect-process's registered client, so the worker
     cannot silently refresh it. That degrades gracefully to the "expired ->
     Reconnect" UX. Set a fixed (e.g. ECI public) ``client_id`` only if you want
     silent cross-process refresh.

Endpoint discovery runs once at startup per source and is guarded: if it fails
(server unreachable, no well-known), the source is left unconfigured and its
status surfaces as ``error`` rather than crashing route registration.
"""

from __future__ import annotations

import logging

import httpx

from deep_researcher_agent.common.data_source_registry import get_all_sources
from nat.authentication.token_storage import InMemoryTokenStorage
from nat.authentication.token_storage import ObjectStoreTokenStorage
from nat.authentication.token_storage import TokenStorageBase

from .nat_provider import NatMcpAuthProvider
from .nat_provider import OAuthSourceSettings

logger = logging.getLogger(__name__)


async def _resolve_token_storage(builder, cfg, source_id: str) -> TokenStorageBase:
    """Resolve the token storage for a source from the mcp_oauth2 config.

    Uses the configured object store (shared across processes) when present;
    otherwise falls back to a process-local in-memory store with a loud warning
    (dev only — tokens will not be visible to job workers).
    """
    object_store_name = getattr(cfg, "token_storage_object_store", None)
    if object_store_name:
        object_store = await builder.get_object_store_client(object_store_name)
        return ObjectStoreTokenStorage(object_store)
    logger.warning(
        "Source '%s': mcp_oauth2 provider has no token_storage_object_store; using a process-local "
        "in-memory token store. Tokens connected via the API will NOT be visible to job workers. "
        "Set token_storage_object_store to a shared object store for real deployments.",
        source_id,
    )
    return InMemoryTokenStorage()


async def _probe_for_oauth_challenge(server_url: str) -> httpx.Response | None:
    """Send an unauthenticated MCP request to elicit the 401 + WWW-Authenticate.

    MCP servers point at their authorization server via the 401's RFC 9728
    ``resource_metadata`` hint. Returns the 401 response (for NAT discovery) or
    None if the server didn't challenge (NAT then falls back to well-known).
    """
    if not server_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                server_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                headers={"Accept": "application/json, text/event-stream"},
            )
        if response.status_code == 401:
            return response
        logger.debug("MCP probe of %s returned %s (no 401 challenge)", server_url, response.status_code)
    except Exception as exc:
        logger.warning("MCP auth probe failed for %s: %s", server_url, exc)
    return None


async def _resolve_oauth_settings(source_id: str, pua, nat_provider, cfg) -> OAuthSourceSettings | None:
    """Derive DeepResearcher-side OAuth settings from the NAT provider, discovering endpoints.

    Returns None (source left unconfigured -> status 'error') if endpoints or a
    client_id cannot be resolved.

    NAT private-API surface (review on every nvidia-nat upgrade; pinned to 1.8.0).
    All accesses below are defensive (``getattr`` defaults + the surrounding
    ``try/except``), so a renamed/removed member degrades to status 'error' with a
    warning rather than crashing:

      * ``nat_provider._discover_and_register(response=...)`` -- coroutine; runs
        well-known discovery + RFC 7591 dynamic client registration, populating the
        cached members below.
      * ``nat_provider._cached_endpoints`` -- object | None; ``.authorization_url``
        and ``.token_url`` (str-able).
      * ``nat_provider._cached_credentials`` -- object | None; ``.client_id`` and
        ``.client_secret`` (str).
      * ``nat_provider._effective_scopes`` -- Iterable[str] | None; scopes resolved
        from protected-resource metadata.
      * ``nat_provider._discoverer`` -- object | None; ``._resource_from_metadata``
        (str | None) is the RFC 9728 resource identifier.
    """
    redirect_uri = str(cfg.redirect_uri) if getattr(cfg, "redirect_uri", None) else ""
    scopes = list(getattr(cfg, "scopes", None) or [])
    client_id = getattr(cfg, "client_id", None)
    client_secret = getattr(cfg, "client_secret", None)
    server_url = str(getattr(cfg, "server_url", "") or "")
    # OAuth resource indicator (RFC 8707/9728); default to the server URL, matching
    # NAT's `resource = _resource_from_metadata or server_url`.
    resource = server_url or None

    authorization_url = token_url = None
    # Reuse NAT's discovery (well-known metadata + RFC 7591 DCR when no client_id
    # is set). This module is the explicit NAT integration seam, so reaching
    # discovery internals is acceptable (version-pinned to the resolved nat release).
    # MCP servers (e.g. NVIDIA MaaS) advertise their AS via the 401 WWW-Authenticate
    # header (RFC 9728 resource_metadata), not the root well-known, so we probe for
    # that 401 first; without it unauthenticated discovery cannot locate the AS.
    try:
        challenge = await _probe_for_oauth_challenge(str(getattr(cfg, "server_url", "") or ""))
        await nat_provider._discover_and_register(response=challenge)  # noqa: SLF001 — NAT seam
        endpoints = getattr(nat_provider, "_cached_endpoints", None)
        credentials = getattr(nat_provider, "_cached_credentials", None)
        if endpoints is not None:
            authorization_url = str(endpoints.authorization_url)
            token_url = str(endpoints.token_url)
        if credentials is not None and not client_id:
            client_id = credentials.client_id
            client_secret = credentials.client_secret
        # Prefer scopes resolved by discovery (protected-resource metadata) when
        # the config didn't pin any.
        discovered_scopes = getattr(nat_provider, "_effective_scopes", None)
        if discovered_scopes:
            scopes = list(discovered_scopes)
        # RFC 9728 resource identifier from protected-resource metadata, mirroring
        # NAT: `_discoverer._resource_from_metadata or server_url` (auth_provider.py).
        discoverer = getattr(nat_provider, "_discoverer", None)
        resource = getattr(discoverer, "_resource_from_metadata", None) or resource
    except Exception as exc:
        logger.warning("Source '%s': OAuth endpoint discovery failed: %s", source_id, exc)

    if not (authorization_url and token_url and client_id):
        logger.warning(
            "Source '%s': could not resolve OAuth endpoints/client_id from provider '%s'; "
            "connect will be unavailable. Ensure the mcp_oauth2 server is reachable and client_id is set.",
            source_id,
            pua.auth_provider or pua.mcp_server_id or source_id,
        )
        return None

    return OAuthSourceSettings(
        source_id=source_id,
        mcp_server_id=pua.mcp_server_id or source_id,
        provider=pua.provider,
        authorization_url=authorization_url,
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
        redirect_uri=redirect_uri,
        use_pkce=bool(getattr(cfg, "use_pkce", True)),
        token_endpoint_auth_method=(getattr(cfg, "token_endpoint_auth_method", None) or "client_secret_post"),
        resource=resource,
    )


async def build_mcp_auth_provider(builder) -> NatMcpAuthProvider:
    """Construct the provider from registry metadata + NAT mcp_oauth2 config."""
    settings_by_source: dict[str, OAuthSourceSettings] = {}
    storages: dict[str, TokenStorageBase] = {}
    # Guard against credential cross-contamination: NAT's ObjectStoreTokenStorage
    # keys tokens by user only (``tokens/{sha256(user_id)}``), so two protected
    # sources sharing one token-storage object store would overwrite each other's
    # credentials for the same user. Deep Researcher Agent can't fix this with a per-source key
    # prefix, because NAT's job-time per_user_mcp_client reads the token itself via
    # the source's mcp_oauth2 provider (unprefixed) — a prefix here would desync
    # that read. So each protected source needs its OWN token-storage bucket; fail
    # closed (skip the later source) when one is reused rather than silently
    # clobbering tokens.
    claimed_stores: dict[str, str] = {}  # object_store name -> first source_id to claim it

    for source in get_all_sources():
        pua = source.per_user_auth
        if pua is None or not pua.required:
            continue
        ref = pua.auth_provider or pua.mcp_server_id or source.id
        try:
            nat_provider = await builder.get_auth_provider(ref)
        except Exception as exc:
            logger.warning(
                "Source '%s' declares per_user_auth but NAT auth provider '%s' is not configured: %s",
                source.id,
                ref,
                exc,
            )
            continue

        cfg = getattr(nat_provider, "config", None)
        if cfg is None:
            logger.warning("Source '%s': auth provider '%s' has no config; skipping", source.id, ref)
            continue

        object_store_name = getattr(cfg, "token_storage_object_store", None)
        if object_store_name:
            prior = claimed_stores.get(object_store_name)
            if prior is not None:
                logger.error(
                    "Source '%s' shares token_storage_object_store '%s' with source '%s'. NAT keys tokens "
                    "per user only, so sharing one store lets these sources overwrite each other's "
                    "credentials. Give each protected source its own token-storage object store (distinct "
                    "bucket). Skipping '%s' — it will surface as unconfigured until this is resolved.",
                    source.id,
                    object_store_name,
                    prior,
                    source.id,
                )
                continue
            claimed_stores[object_store_name] = source.id

        try:
            storage = await _resolve_token_storage(builder, cfg, source.id)
        except Exception as exc:
            logger.error("Source '%s': could not resolve token storage: %s", source.id, exc)
            continue

        settings = await _resolve_oauth_settings(source.id, pua, nat_provider, cfg)
        if settings is None:
            continue

        settings_by_source[source.id] = settings
        storages[source.id] = storage

    if settings_by_source:
        logger.info("MCP auth configured for sources: %s", ", ".join(sorted(settings_by_source)))
    return NatMcpAuthProvider(
        settings_by_source=settings_by_source,
        token_storage_resolver=lambda s: storages[s.source_id],
    )
