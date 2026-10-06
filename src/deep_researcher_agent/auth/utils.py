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

"""Shared authentication utilities for token retrieval and request identity.

These utilities can be used by any tool or agent to get auth tokens.
Trusted request identity must come from verified middleware context, not from
raw JWT payload decoding.

Token sources checked in priority order by ``get_auth_token()``:

1. **Registered token fetchers** — job-scoped sources such as the Dask worker
   auth context.
2. **Deep Researcher Agent auth middleware context** — the request token already validated by the
   configured ``TokenValidator``.
3. **NAT/DeepResearcher Context cookies** — ``idToken`` cookie set by the frontend auth layer
   (server / web-UI mode).
4. **NAT/DeepResearcher Context Authorization header** — ``Authorization: Bearer <jwt>`` sent
   by API callers who authenticate with a JWT directly.
"""

import base64
import json
import logging
import threading
from collections.abc import Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_token_fetchers: list[tuple[int, Callable[[], str | None]]] = []
_fetcher_lock = threading.Lock()


def register_token_fetcher(fetcher: Callable[[], str | None], priority: int = 0) -> None:
    """Register an additional token source.

    Registered fetchers are tried in priority order (highest first) before
    the default Context cookie lookup. The first fetcher that returns a
    non-None token wins.

    Duplicate fetchers (same callable identity) are silently ignored.

    Args:
        fetcher: Callable that returns a token string or None.
        priority: Higher priority fetchers are tried first. Default: 0.
    """
    with _fetcher_lock:
        if any(f is fetcher for _, f in _token_fetchers):
            logger.debug("Token fetcher already registered, skipping duplicate")
            return
        _token_fetchers.append((priority, fetcher))
        _token_fetchers.sort(key=lambda x: x[0], reverse=True)
    logger.debug("Registered token fetcher (priority=%d), total fetchers: %d", priority, len(_token_fetchers))


def unregister_token_fetcher(fetcher: Callable[[], str | None]) -> None:
    """Remove a previously registered token fetcher.

    Matches by callable identity (``is`` check). No-op if the fetcher
    is not currently registered.

    Args:
        fetcher: The same callable object that was passed to
            :func:`register_token_fetcher`.
    """
    with _fetcher_lock:
        before = len(_token_fetchers)
        _token_fetchers[:] = [(p, f) for p, f in _token_fetchers if f is not fetcher]
        removed = before - len(_token_fetchers)
    if removed:
        logger.debug("Unregistered token fetcher, total fetchers: %d", len(_token_fetchers))


def clear_token_fetchers() -> None:
    """Remove all registered token fetchers.

    Warning: This is intended for test isolation only. Calling in production
    will silently remove all registered auth sources.
    """
    with _fetcher_lock:
        _token_fetchers.clear()


class UserInfo(BaseModel):
    email: str | None = None
    name: str | None = None


class Principal(BaseModel):
    type: str
    sub: str
    email: str | None = None
    name: str | None = None


def decode_unverified_jwt_payload(token: str) -> dict:
    """Decode the payload section of a JWT token without verification."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token format")

        payload = parts[1]
        padding = len(payload) % 4
        if padding:
            payload += "=" * (4 - padding)

        decoded_bytes = base64.urlsafe_b64decode(payload)
        return json.loads(decoded_bytes)
    except Exception as e:
        logger.error("Failed to decode JWT token: %s", e)
        return {}


def get_user_info_from_unverified_token(id_token: str) -> UserInfo:
    """Extract user display fields from an unverified JWT token."""
    payload = decode_unverified_jwt_payload(id_token)

    email = payload.get("email")
    name = (
        payload.get("name") or payload.get("given_name") or payload.get("preferred_username") or payload.get("nickname")
    )

    return UserInfo(email=email, name=name)


def decode_jwt_payload(token: str) -> dict:
    """Deprecated alias for unverified JWT payload decoding."""
    logger.warning("decode_jwt_payload() is deprecated; use decode_unverified_jwt_payload()")
    return decode_unverified_jwt_payload(token)


def get_user_info_from_token(id_token: str) -> UserInfo:
    """Deprecated alias for unverified JWT display field extraction."""
    logger.warning("get_user_info_from_token() is deprecated; use get_user_info_from_unverified_token()")
    return get_user_info_from_unverified_token(id_token)


def get_auth_token() -> str | None:
    """
    Return a token from the first available source.

    Sources checked in order:

    1. Registered token fetchers in priority order (highest first).
    2. Deep Researcher Agent auth middleware's current request token.
    3. NAT ``Context`` cookies — ``idToken`` key (server / web-UI mode).
    4. NAT ``Context`` Authorization header — ``Bearer <jwt>`` (API callers with JWT).

    Returns:
        ID token string, or ``None`` if no valid token is available.
    """
    # Try registered fetchers first (highest priority first).
    # Iterate a snapshot so concurrent register_token_fetcher() calls
    # don't mutate the list mid-iteration.
    for _priority, fetcher in list(_token_fetchers):
        try:
            token = fetcher()
            if token:
                logger.debug("Token provided by registered fetcher")
                return token
        except Exception as e:
            logger.debug("Registered token fetcher failed: %s", e)

    # Custom Deep Researcher Agent FastAPI routes run under AuthMiddleware but outside a NAT
    # SessionManager session, so their headers/cookies are not represented in
    # NAT Context. TokenValidator's contract preserves the validated raw token
    # in the middleware user context; use that request-scoped value before
    # falling back to NAT Context.
    try:
        from deep_researcher_api.auth.middleware import get_current_user

        current_user = get_current_user()
        middleware_token = current_user.get("token") if isinstance(current_user, dict) else None
        if isinstance(middleware_token, str) and middleware_token.strip():
            logger.debug("Using token from Deep Researcher Agent auth middleware context")
            return middleware_token.strip()
    except Exception:
        logger.debug("Failed to retrieve token from Deep Researcher Agent auth middleware context")

    # Fall back to NAT Context cookies and headers.
    try:
        from nat.builder.context import Context

        context_metadata = Context.get().metadata

        # 1. NAT Context cookie (browser / web-UI mode)
        if context_metadata and context_metadata.cookies:
            id_token = context_metadata.cookies.get("idToken")
            if id_token:
                logger.debug("Using token from Context cookies")
                return id_token.strip()

        # 2. NAT Context Authorization header (API callers with JWT)
        if context_metadata and context_metadata.headers:
            auth_header = context_metadata.headers.get("authorization", "")
            if auth_header.startswith("Bearer eyJ"):
                logger.debug("Using token from Authorization header")
                return auth_header[7:].strip()
    except Exception as e:
        logger.debug("Failed to retrieve token from Context: %s", e)

    return None


def get_current_principal() -> Principal | None:
    """Return the verified current request principal from middleware context."""
    try:
        from deep_researcher_api.auth.middleware import get_current_user
    except ImportError:
        logger.debug("Verified request principal unavailable: deep_researcher_api.auth.middleware not importable")
        return None
    except Exception as e:
        logger.debug("Verified request principal unavailable: %s", e)
        return None

    try:
        current_user = get_current_user()
    except Exception as e:
        logger.debug("Failed to read current user from middleware context: %s", e)
        return None

    if not isinstance(current_user, dict):
        logger.debug("Ignoring non-dict current user context")
        return None

    principal_type = current_user.get("type")
    sub = current_user.get("sub")
    if not principal_type or not sub:
        # Middleware may expose anonymous/internal callers or raw JWTs captured
        # on internal traffic. Without a verified subject, the identity is not trusted.
        return None

    return Principal(
        type=str(principal_type),
        sub=str(sub),
        email=current_user.get("email"),
        name=current_user.get("name"),
    )


def get_verified_current_user() -> Principal | None:
    """Return the verified current request principal from middleware context."""
    return get_current_principal()


def get_current_user_info() -> UserInfo | None:
    """
    Get trusted current user information from verified middleware context.

    Returns:
        ``UserInfo`` with email / name, or ``None`` if no verified principal is available.
    """
    principal = get_current_principal()
    if principal is None:
        return None
    return UserInfo(email=principal.email, name=principal.name)
