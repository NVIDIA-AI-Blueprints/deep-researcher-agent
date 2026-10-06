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

import inspect
import json
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages

from deep_researcher_agent.common import parse_data_sources

from .request_context import ChatRequestContext
from .request_context import DatabaseName


def trim_message_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Trim messages to a maximum number of tokens."""
    return trim_messages(
        messages=[m.model_dump() for m in messages],
        max_tokens=max_tokens,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
    )


def _normalize_enum_value(value: Any) -> str | None:
    """Extract string value from enum or return as-is if already a string."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _is_text_type(type_value: Any) -> bool:
    """Check if type value represents 'text', handling both strings and enums."""
    normalized = _normalize_enum_value(type_value)
    return normalized is not None and normalized.lower() == "text"


def _is_user_role(role_value: Any) -> bool:
    """Check if role value represents 'user', handling both strings and enums."""
    normalized = _normalize_enum_value(role_value)
    return normalized is not None and normalized.lower() == "user"


def _extract_text_from_message(message: Any) -> str | None:
    if message is None:
        return None
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        content_value = getattr(message, "content")
        if isinstance(content_value, str):
            return content_value
        if isinstance(content_value, list):
            parts = []
            for item in content_value:
                if hasattr(item, "type") and _is_text_type(getattr(item, "type")):
                    text_value = getattr(item, "text", None)
                    if text_value:
                        parts.append(str(text_value))
                elif isinstance(item, dict) and _is_text_type(item.get("type")):
                    text_value = item.get("text")
                    if text_value:
                        parts.append(str(text_value))
            if parts:
                return "\n".join(parts).strip()
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and _is_text_type(item.get("type")):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            if parts:
                return "\n".join(parts).strip()
        if isinstance(content, str):
            return content
        text_value = message.get("text")
        if isinstance(text_value, str):
            return text_value
    return None


def _clean_optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _get_declared_object_field(payload: Any, name: str) -> Any:
    """Read an object field without triggering dynamic fallback attributes."""
    pydantic_extra = getattr(payload, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict) and name in pydantic_extra:
        return pydantic_extra[name]
    try:
        inspect.getattr_static(payload, name)
    except AttributeError:
        return None
    return getattr(payload, name)


def _extract_context_from_text(text: str) -> ChatRequestContext:
    if not text:
        return ChatRequestContext(query_text="")
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            payload = json.loads(trimmed)
        except json.JSONDecodeError:
            return ChatRequestContext(query_text=text)
        if isinstance(payload, dict):
            query_text = payload.get("query") or payload.get("text")
            if isinstance(query_text, str) and query_text.strip():
                return ChatRequestContext(
                    query_text=query_text.strip(),
                    data_sources=parse_data_sources(payload.get("data_sources")),
                    active_report_job_id=_clean_optional_string(payload.get("active_report_job_id")),
                    database_name=payload.get("database_name"),
                )
    return ChatRequestContext(query_text=text)


def _extract_query_context(payload: Any) -> ChatRequestContext:
    """Extract query text, data sources, report context, and database scope from payloads.

    Returns:
        ChatRequestContext.
        - data_sources is None if not specified, meaning use all configured tools
        - data_sources is a list if explicitly specified (use only those)
        - active_report_job_id is optional report context for router decisions
    """
    if isinstance(payload, dict):
        content = payload.get("content", {}) if isinstance(payload.get("content"), dict) else {}
        # Use `is None` (not `or`): an explicit empty list means "no data-source tools"
        # and must be preserved rather than falling through to the nested/inline value.
        data_sources = parse_data_sources(payload.get("data_sources"))
        if data_sources is None:
            data_sources = parse_data_sources(content.get("data_sources"))
        active_report_job_id = _clean_optional_string(payload.get("active_report_job_id")) or _clean_optional_string(
            content.get("active_report_job_id")
        )
        database_name = payload.get("database_name") if "database_name" in payload else content.get("database_name")
        messages = content.get("messages", [])
        query_text = None
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and _is_user_role(msg.get("role")):
                    query_text = _extract_text_from_message(msg)
                    if query_text:
                        break
            if not query_text:
                query_text = _extract_text_from_message(messages[-1])
        if not query_text:
            query_text = _extract_text_from_message(payload.get("message")) or _extract_text_from_message(
                payload.get("text")
            )
        if query_text:
            inline_context = _extract_context_from_text(query_text)
            query_text = inline_context.query_text
            if data_sources is None:
                data_sources = inline_context.data_sources
            active_report_job_id = active_report_job_id or inline_context.active_report_job_id
            if database_name is None:
                database_name = inline_context.database_name
        return ChatRequestContext(
            query_text=query_text or "",
            data_sources=data_sources,
            active_report_job_id=active_report_job_id,
            database_name=database_name,
        )

    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        data_sources = parse_data_sources(getattr(payload, "data_sources", None))
        active_report_job_id = _clean_optional_string(getattr(payload, "active_report_job_id", None))
        database_name = _get_declared_object_field(payload, "database_name")
        query_text = None
        for msg in reversed(messages):
            if _is_user_role(getattr(msg, "role", None)):
                query_text = _extract_text_from_message(msg)
                if query_text:
                    break
        if not query_text and messages:
            query_text = _extract_text_from_message(messages[-1])
        if query_text:
            inline_context = _extract_context_from_text(query_text)
            query_text = inline_context.query_text
            if data_sources is None:
                data_sources = inline_context.data_sources
            active_report_job_id = active_report_job_id or inline_context.active_report_job_id
            if database_name is None:
                database_name = inline_context.database_name
        return ChatRequestContext(
            query_text=query_text or "",
            data_sources=data_sources,
            active_report_job_id=active_report_job_id,
            database_name=database_name,
        )

    return _extract_context_from_text(str(payload))


def _extract_database_name_from_request_metadata(metadata: Any) -> DatabaseName | None:
    """Recover database scope retained by NAT's original HTTP request metadata."""
    payload = getattr(metadata, "payload", None)
    if not isinstance(payload, dict):
        return None
    return _extract_query_context(payload).database_name


def _extract_query_and_sources(payload: Any) -> tuple[str, list[str] | None]:
    """Compatibility wrapper returning only query text and data sources."""

    context = _extract_query_context(payload)
    return (context.query_text, context.data_sources)
