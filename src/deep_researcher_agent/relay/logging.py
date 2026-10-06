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

"""Developer-safe console logging for NeMo Relay lifecycle events."""

from __future__ import annotations

import logging
import threading
from typing import Any

import nemo_relay

from deep_researcher_agent.common.callbacks import BOLD
from deep_researcher_agent.common.callbacks import CYAN
from deep_researcher_agent.common.callbacks import DIM
from deep_researcher_agent.common.callbacks import GREEN
from deep_researcher_agent.common.callbacks import MAGENTA
from deep_researcher_agent.common.callbacks import RED
from deep_researcher_agent.common.callbacks import RESET
from deep_researcher_agent.common.callbacks import RESET_ALL
from deep_researcher_agent.common.callbacks import YELLOW
from deep_researcher_agent.common.logging_utils import log_content_metadata

logger = logging.getLogger(__name__)

SUBSCRIBER_NAME = "deep-researcher-relay-logging"
_depths: dict[str, int] = {}
_depth_lock = threading.RLock()


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    value = getattr(event, name, default)
    return value if value is not None else default


def _nested(value: Any, *path: str | int) -> Any:
    for part in path:
        if isinstance(part, int) and isinstance(value, list) and len(value) > part:
            value = value[part]
        elif isinstance(part, str) and isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _event_depth(event: Any, phase: str) -> int:
    uuid = str(_event_value(event, "uuid", ""))
    parent_uuid = str(_event_value(event, "parent_uuid", ""))
    with _depth_lock:
        if phase == "start":
            depth = _depths.get(parent_uuid, -1) + 1
            _depths[uuid] = depth
            return depth
        depth = _depths.pop(uuid, _depths.get(parent_uuid, -1) + 1)
        return depth


def _log_agent(name: str, phase: str, indent: str, status: str | None) -> None:
    if "agent" not in name.lower() and "subagent" not in name.lower():
        return
    if phase == "start":
        logger.info("%s%s[Chain Start] %s%s", indent, CYAN, name, RESET)
    elif status == "ERROR":
        logger.error("%s%s[Chain Error] %s%s", indent, RED, name, RESET)
    else:
        logger.info("%s%s[Chain End] %s%s", indent, CYAN, name, RESET)


def _log_llm(event: Any, name: str, phase: str, status: str | None) -> None:
    data = _event_value(event, "data")
    profile = _event_value(event, "category_profile", {}) or {}
    annotated = profile.get("annotated_response", {}) if isinstance(profile, dict) else {}
    if phase == "start":
        logger.info("-" * 30)
        logger.info("%s[AGENT]%s %s", BOLD, RESET_ALL, name)
        if data is not None:
            logger.info("%sAgent input: %s%s", YELLOW, log_content_metadata(data), RESET)
        return
    if status == "ERROR":
        logger.error("%s[LLM Error] %s%s", RED, name, RESET)
        return

    reasoning = _nested(data, "generations", 0, 0, "message", "additional_kwargs", "reasoning_content")
    response = annotated.get("message") if isinstance(annotated, dict) else None
    response = response or _nested(data, "generations", 0, 0, "message", "content")
    tool_calls = annotated.get("tool_calls") if isinstance(annotated, dict) else None
    tool_calls = (
        tool_calls
        or _nested(data, "generations", 0, 0, "message", "tool_calls")
        or _nested(data, "generations", 0, 0, "message", "additional_kwargs", "tool_calls")
    )
    response_metadata = _nested(data, "generations", 0, 0, "message", "response_metadata") or {}
    usage = annotated.get("usage", {}) if isinstance(annotated, dict) else {}
    if not usage and isinstance(response_metadata, dict):
        usage = response_metadata.get("token_usage", {}) or {}
    model = annotated.get("model") if isinstance(annotated, dict) else None
    model = model or (profile.get("model_name") if isinstance(profile, dict) else None)
    if not model and isinstance(response_metadata, dict):
        model = next(
            (response_metadata[key] for key in ("model_name", "model", "model_id") if response_metadata.get(key)),
            None,
        )
    if reasoning:
        logger.info("%s[Reasoning] %s%s", MAGENTA, log_content_metadata(reasoning), RESET_ALL)
    if response:
        logger.info("%s[Agent Response] %s%s", CYAN, log_content_metadata(response), RESET)
    if isinstance(tool_calls, list) and tool_calls:
        logger.info("%s[Tool Calls] %d tool(s) requested%s", GREEN, len(tool_calls), RESET)
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None
            tool_name = tool_name or (function.get("name") if isinstance(function, dict) else None) or "unknown"
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else None
            tool_args = tool_args if tool_args is not None else tool_call.get("arguments")
            tool_args = tool_args if tool_args is not None else function.get("arguments", {})
            logger.info("%s  → %s%s", GREEN, tool_name, RESET)
            logger.info("%s    Args: %s%s", DIM, log_content_metadata(tool_args), RESET_ALL)
    if usage:
        logger.info(
            "%s[Tokens] prompt=%s, completion=%s, model=%s%s",
            DIM,
            usage.get("prompt_tokens", "N/A"),
            usage.get("completion_tokens", "N/A"),
            model or "unknown",
            RESET_ALL,
        )
    logger.info("-" * 30)


def _log_tool(event: Any, name: str, phase: str, status: str | None) -> None:
    data = _event_value(event, "data")
    if phase == "start":
        logger.info("%s[Tool Start] %s%s", GREEN, name, RESET)
        if data is not None:
            logger.info("%s  Input: %s%s", DIM, log_content_metadata(data), RESET_ALL)
    elif status == "ERROR":
        logger.error("%s[Tool Error] %s%s", RED, name, RESET)
    else:
        logger.info("%s[Tool Result] %s%s", GREEN, log_content_metadata(data), RESET)


def log_event(event: Any) -> None:
    """Render sanitized Relay events in Deep Researcher Agent's established callback format."""
    category = str(_event_value(event, "category", ""))
    kind = str(_event_value(event, "kind", ""))
    name = str(_event_value(event, "name", "unknown"))
    phase = str(_event_value(event, "scope_category", "event"))
    metadata = _event_value(event, "metadata", {}) or {}
    status = metadata.get("otel.status_code") if isinstance(metadata, dict) else None

    if kind != "scope":
        logger.debug("[Relay Event] %s category=%s", name, category)
        return

    indent = "  " * _event_depth(event, phase)
    if category == "llm":
        _log_llm(event, name, phase, status)
    elif category == "tool":
        _log_tool(event, name, phase, status)
    elif category == "agent":
        _log_agent(name, phase, indent, status)


def register_logging_subscriber() -> None:
    """Register the process-global Deep Researcher Agent Relay logger exactly once."""
    nemo_relay.subscribers.deregister(SUBSCRIBER_NAME)
    with _depth_lock:
        _depths.clear()
    nemo_relay.subscribers.register(SUBSCRIBER_NAME, log_event)
