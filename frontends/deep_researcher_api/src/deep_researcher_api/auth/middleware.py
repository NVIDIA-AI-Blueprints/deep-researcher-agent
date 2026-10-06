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

"""Raw ASGI authentication middleware.

Three independent checks:
--------------------------------------------------
1. **Path allowlist** — blocks unexposed paths on external requests.
   *Always* enforced, regardless of ``REQUIRE_AUTH``.
2. **Token validation** — rejects missing / invalid JWT credentials.
   Controlled by the ``require_auth`` constructor arg (``REQUIRE_AUTH`` env var).
3. **Caller type detection** — sets ``user.type`` for downstream logic
   (e.g. skip the clarifier for headless callers).
   *Always* applied, even when auth is disabled.

CONTEXTVAR PROPAGATION
-----------------------
The middleware stores the resolved user dict in a ``ContextVar`` so that
NAT workflow functions (which do not receive the ASGI ``Request`` object)
can read the caller type without framework coupling.

    from deep_researcher_api.auth.middleware import get_current_user
    user = get_current_user()          # {"type": "jwt"|"internal"|"anonymous"|...}
    skip = user.get("skip_clarifier")  # True / False

ENVIRONMENT VARIABLES
----------------------
``DEEP_RESEARCHER_EXTERNAL_HOSTNAMES``
    Comma-separated list of external-facing hostnames.  Requests that arrive
    with a ``Host`` header matching one of these are treated as external (auth
    + path filter applied).

``REQUIRE_AUTH``
    ``"true"`` / ``"false"`` (case-insensitive).  When ``false``, token
    validation is skipped but the path filter and caller-type detection still
    run.  Defaults to ``"false"`` (safe default; configure ``"true"`` in
    production).

``DEEP_RESEARCHER_JWT_ISSUER``
    OIDC issuer URL used for JWT signature verification
    (e.g. ``https://accounts.google.com``).  Required when ``REQUIRE_AUTH=true``.

``DEEP_RESEARCHER_JWT_AUDIENCE``
    Optional ``aud`` claim to verify.  Leave unset to skip audience
    verification.

``DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE``
    Controls whether verified user identity is attached to active trace spans.
    Supported values:
    - ``none``: do not attach any user identity tags
    - ``id``: attach pseudonymous stable identifiers only (``enduser.id``, ``deep-researcher.user.id``,
      ``deep-researcher.auth.type``)
    - ``full``: attach identifiers plus ``deep-researcher.user.email`` and
      ``deep-researcher.user.name`` when present
    Defaults to ``none``.

``DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET``
    Secret used to derive pseudonymous trace user IDs from verified subjects
    when ``DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE`` is ``id`` or ``full``. If unset,
    user identity tagging is disabled even when a mode is configured.

``DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE``
    Controls whether a pseudonymous client identifier is attached to trace spans.
    Supported values:
    - ``none``: do not attach a client identifier
    - ``ip``: derive a pseudonymous client ID from the request IP
    Defaults to ``none``.

``DEEP_RESEARCHER_TRACE_CLIENT_ID_HMAC_SECRET``
    Secret used to derive pseudonymous client IDs. Falls back to
    ``DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET`` when unset.

``DEEP_RESEARCHER_TRACE_CLIENT_IP_HEADERS``
    Comma-separated header names to check for the client IP before falling back
    to the ASGI client address. Defaults to ``x-real-ip,x-forwarded-for``.
"""

import inspect
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from starlette.types import ASGIApp
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from . import utils as auth_utils
from .request_trace import get_request_trace_tags
from .request_trace import install_request_trace_span_injection
from .request_trace import request_trace_tag_context
from .utils import _load_trace_client_id_mode
from .utils import _load_trace_client_id_secret
from .utils import _load_trace_client_ip_headers
from .utils import _load_trace_user_identity_mode
from .utils import _load_trace_user_identity_secret
from .utils import attach_request_to_active_trace
from .utils import build_request_trace_tags as _build_request_trace_tags
from .utils import is_headless_request

logger = logging.getLogger(__name__)

# Backwards-compatible aliases for tests and internal helper imports.
_build_pseudonymous_trace_user_id = auth_utils._build_pseudonymous_trace_user_id
_build_pseudonymous_trace_client_id = auth_utils._build_pseudonymous_trace_client_id

# ---------------------------------------------------------------------------
# ContextVar — carries the resolved caller identity through the call stack
# ---------------------------------------------------------------------------
_current_user: ContextVar[dict[str, Any]] = ContextVar(
    "_current_user",
    default={"type": "internal", "skip_clarifier": False},
)


def get_current_user() -> dict[str, Any]:
    """Return the caller identity dict set by ``AuthMiddleware`` for this request.

    Returns the default ``{"type": "internal", "skip_clarifier": False}`` when
    the middleware is not registered or when called outside a request context.
    """
    return _current_user.get()


def get_current_trace_tags() -> dict[str, str]:
    """Return request trace tags resolved by ``AuthMiddleware`` for this request."""
    return get_request_trace_tags()


@contextmanager
def user_context(user: dict[str, Any]):
    """Temporarily bind a resolved caller identity in the auth ContextVar."""
    token = _current_user.set(user)
    try:
        yield
    finally:
        _current_user.reset(token)


def build_request_trace_tags(
    headers: dict[bytes, bytes],
    scope: Scope,
    user: dict[str, Any],
    *,
    external_hostnames: set[str] | None = None,
) -> dict[str, str]:
    """Build request trace tags using the same policy as ``AuthMiddleware``."""
    is_external = is_external_request(headers, external_hostnames)
    trust_access_channel_override = (not is_external) or bool(user.get("sub"))
    return _build_request_trace_tags(
        headers,
        scope,
        user,
        trust_access_channel_override=trust_access_channel_override,
        user_identity_mode=_load_trace_user_identity_mode(),
        user_identity_secret=_load_trace_user_identity_secret(),
        client_id_mode=_load_trace_client_id_mode(),
        client_id_secret=_load_trace_client_id_secret(),
        client_ip_headers=_load_trace_client_ip_headers(),
    )


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

# Paths reachable from outside the cluster.  Any external request whose path
# does NOT match one of these receives 404.  Prefix entries must end with "/".
EXTERNAL_ALLOWED_PATHS: list[str] = [
    "/live",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/chat",
    "/chat/stream",
    "/v1/chat/completions",
    "/v1/data_sources",
    "/v1/jobs/async/agents",
    "/v1/jobs/async/submit",
    "/v1/jobs/async/job/",  # prefix — matches /v1/jobs/async/job/{id}/*
    "/v1/auth/mcp/",  # prefix — per-user MCP auth: {id}/status, {id}/connect, {id}/callback
]

# External paths that require no token (monitoring, etc.)
AUTH_EXEMPT_PATHS: set[str] = {"/live", "/health", "/docs", "/redoc", "/openapi.json"}


def _is_oauth_callback_path(path: str) -> bool:
    """The MCP OAuth redirect callback (``/v1/auth/mcp/{source_id}/callback``).

    It must be auth-exempt: the provider redirects the user's browser here with no
    Deep Researcher Agent token. It is secured by the unguessable OAuth ``state`` (bound to the
    principal + source when the flow was started via /connect), not by a request
    token. ``/status`` and ``/connect`` are NOT exempt — they need the principal.
    """
    return path.startswith("/v1/auth/mcp/") and path.endswith("/callback")


def _load_external_hostnames() -> set[str]:
    """Read ``DEEP_RESEARCHER_EXTERNAL_HOSTNAMES`` env var; fall back to staging hostname."""
    env = os.getenv("DEEP_RESEARCHER_EXTERNAL_HOSTNAMES", "")
    names = {h.strip() for h in env.split(",") if h.strip()}
    return names


def is_external_request(headers: dict[bytes, bytes], external_hostnames: set[str] | None = None) -> bool:
    """Return ``True`` when the Host header matches an external-facing hostname."""
    host = headers.get(b"host", b"").decode().split(":")[0]
    return host in (external_hostnames or _load_external_hostnames())


def extract_auth_token(headers: dict[bytes, bytes]) -> str | None:
    """Extract a bearer token or idToken cookie from ASGI headers."""
    auth = headers.get(b"authorization", b"").decode()
    if auth.startswith("Bearer "):
        return auth[7:]

    cookie = headers.get(b"cookie", b"").decode()
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("idToken="):
            return part[8:]
    return None


async def validate_token_with_error(token: str, validators: list) -> tuple[dict[str, Any] | None, str | None]:
    """Try validators in order and preserve the most specific auth failure code."""
    last_error: str | None = "token_invalid"
    for validator in validators:
        if validator.can_handle(token):
            has_error_validator = "validate_with_error" in getattr(validator, "__dict__", {}) or hasattr(
                type(validator), "validate_with_error"
            )
            if has_error_validator:
                result = validator.validate_with_error(token)
                if inspect.isawaitable(result):
                    user, error_code = await result
                else:
                    user, error_code = result
            else:
                result = await validator.validate(token)
                if isinstance(result, tuple) and len(result) == 2:
                    user, error_code = result
                else:
                    user, error_code = result, None if result is not None else "token_invalid"
            if user is not None:
                return (user, None)
            if error_code:
                last_error = error_code
    logger.debug("Token rejected by all %d configured validator(s)", len(validators))
    return (None, last_error)


async def validate_token_with_validators(token: str, validators: list) -> dict[str, Any] | None:
    """Try validators in order and return the first successful identity dict."""
    user, _ = await validate_token_with_error(token, validators)
    return user


def detect_internal_caller(headers: dict[bytes, bytes]) -> dict[str, Any]:
    """Classify an internal request without validating any presented token."""
    token = extract_auth_token(headers)
    headless = is_headless_request(headers)
    if token:
        return {"type": "unverified_jwt", "token": token, "skip_clarifier": headless}
    return {"type": "internal", "skip_clarifier": headless}


async def resolve_request_user(
    headers: dict[bytes, bytes],
    *,
    validators: list,
    require_auth: bool,
    external_hostnames: set[str] | None = None,
) -> tuple[dict[str, Any] | None, int | None, bool, str | None]:
    """Resolve request identity from validated credentials when present.

    Returns a tuple of:
      - resolved user dict, or None when the request should be rejected
      - HTTP status code to use on rejection, or None on success
      - whether the request was classified as external
      - machine-readable auth error code, or None on success
    """
    is_external = is_external_request(headers, external_hostnames)
    token = extract_auth_token(headers)

    if token:
        user, error_code = await validate_token_with_error(token, validators)
        if user is not None:
            if is_headless_request(headers):
                user["skip_clarifier"] = True
            return user, None, is_external, None

        if is_external and require_auth:
            return None, 401, is_external, error_code or "token_invalid"

    if is_external:
        if not require_auth:
            return {"type": "anonymous", "skip_clarifier": True}, None, is_external, None
        return None, 401, is_external, "token_missing"

    return detect_internal_caller(headers), None, is_external, None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AuthMiddleware:
    """
    Raw ASGI middleware that enforces path filtering and token validation
    without buffering response bodies.

    Provider-agnostic: accepts any list of ``TokenValidator`` instances.
    Each validator implements ``can_handle()`` + ``validate()``.  The first
    validator that returns a non-None result wins.

    Register via FastAPI's ``add_middleware``

    Args:
        app: The inner ASGI application.
        validators: Ordered list of token validators to try.
        require_auth: When ``True``, external requests must carry a valid token.
        external_hostnames: Hostnames considered "external".
            Defaults to ``DEEP_RESEARCHER_EXTERNAL_HOSTNAMES`` env var.
    """

    def __init__(
        self,
        app: ASGIApp,
        validators: list | None = None,
        require_auth: bool = False,
        external_hostnames: set[str] | None = None,
    ) -> None:
        self.app = app
        self.require_auth = require_auth
        self._validators: list = validators or []
        self._external_hostnames: set[str] = external_hostnames or _load_external_hostnames()
        self._trace_user_identity_mode: str = _load_trace_user_identity_mode()
        self._trace_user_identity_secret: str | None = _load_trace_user_identity_secret()
        self._trace_client_id_mode: str = _load_trace_client_id_mode()
        self._trace_client_id_secret: str | None = _load_trace_client_id_secret()
        self._trace_client_ip_headers: list[str] = _load_trace_client_ip_headers()
        install_request_trace_span_injection()

        if require_auth and not self._validators:
            logger.warning(
                "REQUIRE_AUTH=true but no validators are configured — all authenticated requests will be rejected."
            )

        logger.info(
            "AuthMiddleware: require_auth=%s, validators=%s, external_hostnames=%s",
            require_auth,
            [type(v).__name__ for v in self._validators],
            self._external_hostnames,
        )
        logger.info(
            "AuthMiddleware trace user identity mode=%s secret_configured=%s",
            self._trace_user_identity_mode,
            bool(self._trace_user_identity_secret),
        )
        logger.info(
            "AuthMiddleware trace client id mode=%s secret_configured=%s ip_headers=%s",
            self._trace_client_id_mode,
            bool(self._trace_client_id_secret),
            self._trace_client_ip_headers,
        )

    # ------------------------------------------------------------------
    # ASGI entry point
    # ------------------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        path: str = scope["path"]
        is_external = self._is_external(headers)

        # External requests still honor the public path allowlist before auth.
        if is_external and not self._path_allowed(path):
            await self._send_json(send, 404, {"detail": "Not found"})
            return

        if is_external and (path in AUTH_EXEMPT_PATHS or _is_oauth_callback_path(path)):
            user = {"type": "anonymous", "skip_clarifier": True}
            await self._call_app(scope, receive, send, headers, user)
            return

        user, error_status, _, error_code = await resolve_request_user(
            headers,
            validators=self._validators,
            require_auth=self.require_auth,
            external_hostnames=self._external_hostnames,
        )
        if user is None:
            detail = {
                "token_missing": "Missing auth token",
                "token_expired": "Token has expired",
                "token_invalid": "Invalid auth token",
            }.get(error_code or "", "Authentication failed")
            await self._send_json(send, error_status or 401, {"detail": detail, "error": error_code or "token_invalid"})
            return

        await self._call_app(scope, receive, send, headers, user)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_app(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        headers: dict[bytes, bytes],
        user: dict[str, Any],
    ) -> None:
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["user"] = user
        is_external = self._is_external(headers)
        trust_access_channel_override = (not is_external) or bool(user.get("sub"))
        request_trace_tags = attach_request_to_active_trace(
            headers,
            scope,
            user,
            trust_access_channel_override=trust_access_channel_override,
            user_identity_mode=self._trace_user_identity_mode,
            user_identity_secret=self._trace_user_identity_secret,
            client_id_mode=self._trace_client_id_mode,
            client_id_secret=self._trace_client_id_secret,
            client_ip_headers=self._trace_client_ip_headers,
        )

        with user_context(user), request_trace_tag_context(request_trace_tags):
            await self.app(scope, receive, send)

    def _is_external(self, headers: dict[bytes, bytes]) -> bool:
        return is_external_request(headers, self._external_hostnames)

    def _path_allowed(self, path: str) -> bool:
        for allowed in EXTERNAL_ALLOWED_PATHS:
            if allowed.endswith("/"):
                if path.startswith(allowed) or path == allowed.rstrip("/"):
                    return True
            elif path == allowed:
                return True
        return False

    def _extract_token(self, headers: dict[bytes, bytes]) -> str | None:
        return extract_auth_token(headers)

    def _is_headless(self, headers: dict[bytes, bytes]) -> bool:
        return is_headless_request(headers)

    async def _validate_token(self, token: str) -> tuple[dict[str, Any] | None, str | None]:
        """Try each validator in order, returning ``(user, None)`` or ``(None, error_code)``."""
        return await validate_token_with_error(token, self._validators)

    def _detect_internal_caller(self, headers: dict[bytes, bytes]) -> dict[str, Any]:
        """Classify an internal request without validating the token."""
        return detect_internal_caller(headers)

    @staticmethod
    async def _send_json(send: Send, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(payload)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
