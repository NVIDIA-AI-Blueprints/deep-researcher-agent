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

"""Shared utilities for data source handling across agents."""

import logging
from typing import Any

from langchain_core.messages import BaseMessage

from .data_source_registry import get_source
from .data_source_registry import get_source_id_for_tool

# Default to web_search when no data sources specified
DEFAULT_DATA_SOURCES: list[str] = ["web_search"]

logger = logging.getLogger(__name__)


def parse_data_sources(raw: Any) -> list[str] | None:
    """Parse data sources from various input formats.

    Args:
        raw: Can be None, a list of strings, or a comma-separated string.

    Returns:
        - None if input is None (not specified, use all tools)
        - Empty list [] if input was explicitly empty (no data-source tools)
        - List of data source IDs if specified
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        if len(raw) == 0:
            return []
        parsed = [str(value).strip() for value in raw]
        return [value for value in parsed if value] or []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        parsed = [value.strip() for value in raw.split(",")]
        return [value for value in parsed if value] or []
    return None


def filter_tools_by_sources(tools: list[Any], data_sources: list[str] | None) -> list[Any]:
    """Filter tools based on selected data sources.

    Uses the tool->source map built at startup from config ``data_source`` fields.
    Tools without a mapping (e.g. "think", calculator) are always included.
    Source ID matching is case-insensitive.

    Args:
        tools: List of LangChain tools.
        data_sources: List of selected data source IDs, None for all data-source
            tools, or [] for no data-source tools.

    Returns:
        Filtered list of tools matching the selected data sources.
    """
    if data_sources is None:
        return tools

    selected = {source_id.strip().lower() for source_id in data_sources if source_id.strip()}
    filtered = []
    for tool in tools:
        name = getattr(tool, "name", "")
        source_id = get_source_id_for_tool(name)
        if source_id is None:
            # Not a data source tool (e.g., "think", calculator) -> always include.
            filtered.append(tool)
        elif source_id.lower() in selected:
            filtered.append(tool)
    return filtered


def all_mapped_tools_filtered_out(
    tools: list[Any],
    selected_tools: list[Any],
    data_sources: list[str] | None,
) -> bool:
    """Return True when filtering dropped every data-source-mapped tool.

    Useful for emitting a diagnostic when a caller passed ``data_sources`` but
    the filter produced no mapped tools (e.g. ``data_sources=[]`` with mapped
    tools configured, or ``data_sources=["unknown"]`` that matched nothing).
    Returns False when ``data_sources is None`` (no filtering requested) or
    when the original tool list had no mapped tools to filter in the first
    place.

    Args:
        tools: Full tool list before filtering.
        selected_tools: Tool list after ``filter_tools_by_sources``.
        data_sources: The ``data_sources`` argument passed to the filter.

    Returns:
        True if ``data_sources`` was specified, ``tools`` contained at least
        one mapped tool, and zero mapped tools survived the filter.
    """
    if data_sources is None:
        return False
    had_mapped = any(get_source_id_for_tool(getattr(t, "name", "")) is not None for t in tools)
    if not had_mapped:
        return False
    still_has_mapped = any(get_source_id_for_tool(getattr(t, "name", "")) is not None for t in selected_tools)
    return not still_has_mapped


def extract_messages_and_sources(payload: Any) -> tuple[list[BaseMessage], list[str] | None]:
    """Extract messages and data sources from a payload.

    Args:
        payload: Can be a dict with 'payload' key, a dict with 'messages', or a list.

    Returns:
        Tuple of (messages, data_sources).

    Raises:
        ValueError: If payload format is invalid.
    """
    if isinstance(payload, dict):
        if "payload" in payload and isinstance(payload["payload"], dict):
            payload = payload["payload"]
        messages = payload.get("messages")
        if isinstance(messages, list):
            return messages, parse_data_sources(payload.get("data_sources"))
    if isinstance(payload, list):
        return payload, None
    raise ValueError("Invalid payload format: expected dict with 'messages' or list")


def format_data_source_tools(data_sources: list[str]) -> list[dict[str, str]]:
    """Format data sources as tool info for meta chatter.

    Looks up display metadata from the registry first; falls back to
    title-cased IDs for unregistered sources.

    Args:
        data_sources: List of data source IDs.

    Returns:
        List of tool info dicts with 'name' and 'description'.
    """
    tools_info: list[dict[str, str]] = []
    for source_id in data_sources:
        meta = get_source(source_id)
        if meta:
            tools_info.append({"name": meta.name, "description": meta.description})
        else:
            label = source_id.replace("_", " ").title()
            tools_info.append({"name": label, "description": f"Search {source_id.replace('_', ' ')}."})
    return tools_info
