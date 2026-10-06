# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Request-scoped privacy controls for Relay observability payloads."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import nemo_relay

from .config import RelayRedactionConfig

_PRIVACY_SANITIZER = "deep-researcher-request-privacy"
_request_privacy_enabled: ContextVar[bool] = ContextVar("deep_researcher_relay_request_privacy", default=False)


@contextmanager
def request_privacy_context(enabled: bool) -> Iterator[None]:
    """Apply payload redaction to Relay events emitted by the current request."""
    token = _request_privacy_enabled.set(enabled)
    try:
        yield
    finally:
        _request_privacy_enabled.reset(token)


def request_privacy_from_tags(tags: dict[str, str], tag: str = "deep-researcher.telemetry.redact") -> bool:
    """Resolve the trusted request privacy decision carried with trace tags."""
    return str(tags.get(tag, "")).strip().lower() == "true"


def _redact_event_fields(
    _event: Any,
    fields: nemo_relay.EventSanitizeFields,
    *,
    attributes: tuple[str, ...],
) -> nemo_relay.EventSanitizeFields:
    """Replace only configured fields that Relay supplied for this event."""
    if not _request_privacy_enabled.get():
        return fields
    sanitized = dict(fields)
    for attribute in attributes:
        if attribute in sanitized:
            sanitized[attribute] = None
    metadata = sanitized.get("metadata")
    if isinstance(metadata, dict):
        metadata = {**metadata, "deep-researcher.telemetry.redacted": True}
        if "otel.status_description" in metadata:
            metadata["otel.status_description"] = "[REDACTED]"
        sanitized["metadata"] = metadata
    return nemo_relay.EventSanitizeFields(**sanitized)


def _redact_llm_request(request: Any, _context: Any) -> Any:
    return None if _request_privacy_enabled.get() else request


def _redact_llm_response(response: Any, _context: Any) -> Any:
    return None if _request_privacy_enabled.get() else response


def _redact_tool_payload(_tool_name: str, payload: Any) -> Any:
    return "[REDACTED]" if _request_privacy_enabled.get() else payload


def deregister_privacy_sanitizers() -> None:
    """Remove Deep Researcher Agent's process-global Relay privacy sanitizers."""
    guardrails = nemo_relay.guardrails
    guardrails.deregister_scope_sanitize_start(_PRIVACY_SANITIZER)
    guardrails.deregister_scope_sanitize_end(_PRIVACY_SANITIZER)
    guardrails.deregister_mark_sanitize(_PRIVACY_SANITIZER)
    guardrails.deregister_llm_sanitize_request(_PRIVACY_SANITIZER)
    guardrails.deregister_llm_sanitize_response(_PRIVACY_SANITIZER)
    guardrails.deregister_tool_sanitize_request(_PRIVACY_SANITIZER)
    guardrails.deregister_tool_sanitize_response(_PRIVACY_SANITIZER)


def register_privacy_sanitizers(config: RelayRedactionConfig) -> None:
    """Register observation-only sanitizers for request privacy decisions."""
    deregister_privacy_sanitizers()
    if not config.enabled:
        return

    attributes = tuple(config.request_privacy_attributes)

    def sanitize_event(event: Any, fields: nemo_relay.EventSanitizeFields) -> nemo_relay.EventSanitizeFields:
        return _redact_event_fields(event, fields, attributes=attributes)

    guardrails = nemo_relay.guardrails
    guardrails.register_scope_sanitize_start(_PRIVACY_SANITIZER, 10, sanitize_event)
    guardrails.register_scope_sanitize_end(_PRIVACY_SANITIZER, 10, sanitize_event)
    guardrails.register_mark_sanitize(_PRIVACY_SANITIZER, 10, sanitize_event)
    guardrails.register_llm_sanitize_request(_PRIVACY_SANITIZER, 10, _redact_llm_request)
    guardrails.register_llm_sanitize_response(_PRIVACY_SANITIZER, 10, _redact_llm_response)
    guardrails.register_tool_sanitize_request(_PRIVACY_SANITIZER, 10, _redact_tool_payload)
    guardrails.register_tool_sanitize_response(_PRIVACY_SANITIZER, 10, _redact_tool_payload)
