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

"""Per-user MCP auth routes: status, connect, and OAuth callback.

Deep Researcher Agent owns these routes (the control plane). The actual OAuth mechanics live
behind :class:`ProtectedSourceAuthProvider`. The callback completes NAT's token
exchange and persists the token, then closes the popup.
"""

from __future__ import annotations

import html
import json
import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse

from deep_researcher_agent.common.data_source_registry import get_source

from ..jobs.access import require_verified_principal
from ..mcp_auth.models import SourceAuthStatusResponse
from ..mcp_auth.models import SourceConnectResponse
from ..mcp_auth.nat_provider import NatMcpAuthProvider
from ..mcp_auth.serialize import connect_url_for

logger = logging.getLogger(__name__)

_CALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family: sans-serif; padding: 2rem; text-align: center;">
<p>{message}</p>
<script>
  try {{ if (window.opener) window.opener.postMessage(
    {{ type: "mcp-auth", source_id: {source_id_js}, ok: {ok_js} }}, "*"); }} catch (e) {{}}
  setTimeout(function () {{ window.close(); }}, 400);
</script>
</body></html>"""


def _callback_page(source_id: str, *, ok: bool, message: str) -> HTMLResponse:
    # `message` can carry a provider-controlled value (e.g. the OAuth `error`
    # query param reflected by the callback), so HTML-escape it before it lands in
    # the page body — otherwise a crafted `error` executes script in Deep Researcher Agent's origin.
    # `source_id` is a validated registry id and json.dumps escapes it for the JS
    # string context.
    page = _CALLBACK_HTML.format(
        title="Connected" if ok else "Authentication failed",
        message=html.escape(message),
        source_id_js=json.dumps(source_id),
        ok_js="true" if ok else "false",
    )
    return HTMLResponse(content=page, status_code=200 if ok else 400, headers={"Cache-Control": "no-cache"})


def _require_protected_source(source_id: str):
    """Return the registry source or raise 404 if it isn't a protected MCP source."""
    source = get_source(source_id)
    if source is None or source.per_user_auth is None or not source.per_user_auth.required:
        raise HTTPException(404, f"Unknown protected data source: {source_id}")
    return source


def register_mcp_auth_routes(app: FastAPI, provider: NatMcpAuthProvider) -> None:
    """Register the per-user MCP auth routes against ``provider``."""

    @app.get(
        "/v1/auth/mcp/{source_id}/status",
        response_model=SourceAuthStatusResponse,
        tags=["mcp auth"],
        summary="Get per-user auth status for a protected source",
    )
    async def mcp_auth_status(source_id: str) -> SourceAuthStatusResponse:
        _require_protected_source(source_id)
        principal = require_verified_principal()
        state = await provider.get_status(principal, source_id)
        connect_url = connect_url_for(source_id) if state.status != "connected" else None
        return SourceAuthStatusResponse(
            source_id=source_id,
            status=state.status,
            expires_at=state.expires_at,
            connect_url=connect_url,
            last_error=state.last_error,
        )

    @app.post(
        "/v1/auth/mcp/{source_id}/connect",
        response_model=SourceConnectResponse,
        tags=["mcp auth"],
        summary="Start (or resume) the OAuth flow for a protected source",
    )
    async def mcp_auth_connect(source_id: str) -> SourceConnectResponse:
        _require_protected_source(source_id)
        principal = require_verified_principal()

        # If already connected, don't mint a new challenge.
        state = await provider.get_status(principal, source_id)
        if state.status == "connected":
            return SourceConnectResponse(source_id=source_id, status="connected")

        try:
            challenge = await provider.start_auth(principal, source_id)
        except ValueError as exc:
            # Source declared but not configured for OAuth in this deployment.
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to start MCP auth for source=%s", source_id)
            raise HTTPException(502, "Could not start authentication flow") from exc

        return SourceConnectResponse(
            source_id=source_id,
            status="auth_required",
            auth_url=challenge.auth_url,
            expires_at=challenge.expires_at,
        )

    @app.get(
        "/v1/auth/mcp/{source_id}/callback",
        tags=["mcp auth"],
        summary="OAuth redirect callback for a protected source",
        include_in_schema=False,
    )
    async def mcp_auth_callback(source_id: str, request: Request) -> HTMLResponse:
        _require_protected_source(source_id)
        state = request.query_params.get("state")
        if not state:
            return _callback_page(source_id, ok=False, message="Missing state. Please restart the connection.")

        error = request.query_params.get("error")
        if error:
            return _callback_page(source_id, ok=False, message=f"Authorization was denied: {error}")

        try:
            completed_source = await provider.complete_callback(state, str(request.url))
        except KeyError:
            return _callback_page(source_id, ok=False, message="This connection link has expired. Please try again.")
        except Exception:
            logger.exception("MCP auth callback failed for source=%s", source_id)
            return _callback_page(source_id, ok=False, message="Authentication failed. Please try again.")

        if completed_source != source_id:
            # State was bound to a different source — refuse to cross the streams.
            logger.warning("Callback source mismatch: path=%s flow=%s", source_id, completed_source)
            return _callback_page(source_id, ok=False, message="Connection mismatch. Please try again.")

        return _callback_page(source_id, ok=True, message="Connected. You can close this window.")

    logger.info("Registered /v1/auth/mcp/{source_id} status, connect, and callback routes")
