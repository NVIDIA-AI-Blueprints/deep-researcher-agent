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

"""Utilities for auth-related request classification and trace enrichment."""

import hmac
import logging
import os
from hashlib import sha256
from importlib import import_module
from typing import Any

from starlette.types import Scope

logger = logging.getLogger(__name__)

TRACE_USER_IDENTITY_MODES = frozenset({"none", "id", "full"})
TRACE_CLIENT_ID_MODES = frozenset({"none", "ip"})
TRACE_ACCESS_CHANNELS = frozenset(
    {
        "ui",
        "api",
        "headless",
        "anonymous",
        "internal",
        "unknown",
    }
)
TRACE_REDACTION_HEADER = "x-deep-researcher-telemetry-redact"


def _load_trace_user_identity_mode() -> str:
    """Read and normalize the trace user identity tagging mode."""
    raw = os.getenv("DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE", "none").strip().lower()
    aliases = {
        "0": "none",
        "false": "none",
        "off": "none",
        "1": "id",
        "true": "id",
        "on": "id",
        "basic": "id",
    }
    mode = aliases.get(raw, raw)
    if mode not in TRACE_USER_IDENTITY_MODES:
        logger.warning(
            "Invalid DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE=%r; expected one of %s. Falling back to 'none'.",
            raw,
            sorted(TRACE_USER_IDENTITY_MODES),
        )
        return "none"
    return mode


def _load_trace_user_identity_secret() -> str | None:
    """Read the secret used to pseudonymize trace user identity."""
    secret = os.getenv("DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET", "").strip()
    return secret or None


def _load_trace_client_id_mode() -> str:
    """Read and normalize the trace client-id tagging mode."""
    raw = os.getenv("DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE", "none").strip().lower()
    aliases = {
        "0": "none",
        "false": "none",
        "off": "none",
        "1": "ip",
        "true": "ip",
        "on": "ip",
    }
    mode = aliases.get(raw, raw)
    if mode not in TRACE_CLIENT_ID_MODES:
        logger.warning(
            "Invalid DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE=%r; expected one of %s. Falling back to 'none'.",
            raw,
            sorted(TRACE_CLIENT_ID_MODES),
        )
        return "none"
    return mode


def _load_trace_client_id_secret() -> str | None:
    """Read the secret used to pseudonymize trace client identity."""
    secret = os.getenv("DEEP_RESEARCHER_TRACE_CLIENT_ID_HMAC_SECRET", "").strip()
    if secret:
        return secret
    return _load_trace_user_identity_secret()


def _load_trace_client_ip_headers() -> list[str]:
    """Read preferred client IP headers in lookup order."""
    raw = os.getenv("DEEP_RESEARCHER_TRACE_CLIENT_IP_HEADERS", "x-real-ip,x-forwarded-for")
    return [header.strip().lower() for header in raw.split(",") if header.strip()]


def is_headless_request(headers: dict[bytes, bytes]) -> bool:
    """Return ``True`` for headless callers that should skip the clarifier."""
    return headers.get(b"x-deep-researcher-mode", b"").decode().lower() == "headless"


def _build_pseudonymous_trace_user_id(principal_type: str, principal_sub: str, secret: str | None) -> str | None:
    """Return a stable pseudonymous trace user ID derived from the verified principal."""
    if not secret:
        return None

    payload = f"{principal_type}:{principal_sub}".encode()
    return hmac.new(secret.encode(), payload, sha256).hexdigest()


def _build_pseudonymous_trace_client_id(client_value: str, secret: str | None) -> str | None:
    """Return a stable pseudonymous client ID derived from client network identity."""
    if not secret or not client_value:
        return None

    return hmac.new(secret.encode(), f"client:{client_value}".encode(), sha256).hexdigest()


def _extract_auth_transport(headers: dict[bytes, bytes]) -> str:
    """Return the request auth transport shape."""
    auth = headers.get(b"authorization", b"").decode()
    if auth.startswith("Bearer "):
        return "bearer"

    cookie = headers.get(b"cookie", b"").decode()
    for part in cookie.split(";"):
        if part.strip().startswith("idToken="):
            return "cookie"
    return "none"


def _extract_header_text(headers: dict[bytes, bytes], name: str) -> str | None:
    """Return a decoded header value for a lowercase header name."""
    value = headers.get(name.encode())
    if not value:
        return None
    decoded = value.decode().strip()
    return decoded or None


def _extract_client_ip(headers: dict[bytes, bytes], scope: Scope, header_names: list[str]) -> str | None:
    """Return the best available client IP from trusted headers or ASGI scope."""
    for header_name in header_names:
        value = _extract_header_text(headers, header_name)
        if not value:
            continue
        candidate = value.split(",")[0].strip() if header_name == "x-forwarded-for" else value
        if candidate:
            return candidate

    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        host = client[0]
        if isinstance(host, str) and host.strip():
            return host.strip()
    return None


def _extract_explicit_access_channel(headers: dict[bytes, bytes], *, allow_override: bool) -> str | None:
    """Return an explicit low-cardinality access-channel override header when present."""
    if not allow_override:
        return None

    value = _extract_header_text(headers, "x-deep-researcher-access-channel")
    if not value:
        return None
    channel = value.lower()
    if channel in TRACE_ACCESS_CHANNELS:
        return channel
    logger.warning("Ignoring unsupported X-DeepResearcher-Access-Channel=%r", value)
    return None


def _infer_access_channel(
    headers: dict[bytes, bytes],
    user: dict[str, Any],
    auth_transport: str,
    *,
    allow_explicit_override: bool,
) -> str:
    """Infer a low-cardinality access channel for observability."""
    if explicit := _extract_explicit_access_channel(headers, allow_override=allow_explicit_override):
        return explicit

    caller_type = str(user.get("type") or "")
    headless = is_headless_request(headers)
    if auth_transport == "cookie":
        return "ui"
    if headless and auth_transport == "bearer":
        return "headless"
    if caller_type == "anonymous":
        return "anonymous"
    if caller_type == "internal":
        return "internal"
    if auth_transport == "bearer":
        return "api"
    return "unknown"


def _is_verified_trace_user(user: dict[str, Any]) -> bool:
    """Return whether the resolved request user is a verified authenticated principal."""
    principal_sub = user.get("sub")
    principal_type = str(user.get("type") or "")
    return bool(principal_sub and principal_type not in {"anonymous", "internal", "unverified_jwt"})


def _build_trace_user_tags(user: dict[str, Any], mode: str, secret: str | None) -> dict[str, str]:
    """Return trace tags for verified user identities according to policy."""
    if mode == "none" or not _is_verified_trace_user(user):
        return {}

    principal_sub = user.get("sub")
    principal_type = user.get("type")
    if not principal_sub or not principal_type:
        return {}

    pseudonymous_id = _build_pseudonymous_trace_user_id(str(principal_type), str(principal_sub), secret)
    if pseudonymous_id is None:
        logger.warning(
            "DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE=%s "
            "but DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET is not set; "
            "skipping user identity trace tags.",
            mode,
        )
        return {}

    tags = {
        "enduser.id": pseudonymous_id,
        "deep-researcher.user.id": pseudonymous_id,
        "deep-researcher.auth.type": str(principal_type),
    }
    if mode == "full":
        if email := user.get("email"):
            tags["deep-researcher.user.email"] = str(email)
        if name := user.get("name"):
            tags["deep-researcher.user.name"] = str(name)
    return tags


def _build_common_trace_tags(
    headers: dict[bytes, bytes],
    scope: Scope,
    user: dict[str, Any],
    *,
    trust_access_channel_override: bool,
    client_id_mode: str,
    client_id_secret: str | None,
    client_ip_headers: list[str],
) -> dict[str, str]:
    """Return always-on low-risk trace tags describing request origin and auth shape."""
    auth_transport = _extract_auth_transport(headers)
    tags = {
        "deep-researcher.caller.type": str(user.get("type") or "unknown"),
        "deep-researcher.auth.transport": auth_transport,
        "deep-researcher.auth.verified": "true" if _is_verified_trace_user(user) else "false",
        "deep-researcher.access.channel": _infer_access_channel(
            headers,
            user,
            auth_transport,
            allow_explicit_override=trust_access_channel_override,
        ),
    }
    if (_extract_header_text(headers, TRACE_REDACTION_HEADER) or "").lower() == "true":
        tags["deep-researcher.telemetry.redact"] = "true"

    if client_id_mode == "ip":
        client_ip = _extract_client_ip(headers, scope, client_ip_headers)
        client_id = _build_pseudonymous_trace_client_id(client_ip or "", client_id_secret)
        if client_id:
            tags["deep-researcher.client.id"] = client_id

    return tags


def _tag_current_ddtrace_span(tags: dict[str, str]) -> None:
    """Attach user tags to the active Datadog span when ddtrace is installed."""
    if not tags:
        return

    try:
        tracer = import_module("ddtrace").tracer
        span = tracer.current_span()
    except Exception:
        return

    if span is None:
        return

    for key, value in tags.items():
        try:
            span.set_tag(key, value)
        except Exception:
            logger.debug("Failed to set Datadog span tag %s", key, exc_info=True)


def _tag_current_otel_span(tags: dict[str, str]) -> None:
    """Attach user tags to the active OpenTelemetry span when present."""
    if not tags:
        return

    try:
        span = import_module("opentelemetry.trace").get_current_span()
    except Exception:
        return

    for key, value in tags.items():
        try:
            span.set_attribute(key, value)
        except Exception:
            logger.debug("Failed to set OpenTelemetry span attribute %s", key, exc_info=True)


def build_request_trace_tags(
    headers: dict[bytes, bytes],
    scope: Scope,
    user: dict[str, Any],
    *,
    trust_access_channel_override: bool,
    user_identity_mode: str,
    user_identity_secret: str | None,
    client_id_mode: str,
    client_id_secret: str | None,
    client_ip_headers: list[str],
) -> dict[str, str]:
    """Build request classification and optional pseudonymous identity trace tags."""
    tags = _build_common_trace_tags(
        headers,
        scope,
        user,
        trust_access_channel_override=trust_access_channel_override,
        client_id_mode=client_id_mode,
        client_id_secret=client_id_secret,
        client_ip_headers=client_ip_headers,
    )
    tags.update(_build_trace_user_tags(user, user_identity_mode, user_identity_secret))
    return tags


def attach_request_to_active_trace(
    headers: dict[bytes, bytes],
    scope: Scope,
    user: dict[str, Any],
    *,
    trust_access_channel_override: bool,
    user_identity_mode: str,
    user_identity_secret: str | None,
    client_id_mode: str,
    client_id_secret: str | None,
    client_ip_headers: list[str],
) -> dict[str, str]:
    """Attach request classification and optional pseudonymous identity to active trace spans."""
    tags = build_request_trace_tags(
        headers,
        scope,
        user,
        trust_access_channel_override=trust_access_channel_override,
        user_identity_mode=user_identity_mode,
        user_identity_secret=user_identity_secret,
        client_id_mode=client_id_mode,
        client_id_secret=client_id_secret,
        client_ip_headers=client_ip_headers,
    )

    _tag_current_ddtrace_span(tags)
    _tag_current_otel_span(tags)
    return tags
