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

"""Convert NeMo Relay ATOF JSONL events into tokenomics request profiles."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .pricing import PricingRegistry
from .profile import PHASE_ORCHESTRATOR
from .profile import PHASE_PLANNER
from .profile import PHASE_RESEARCHER
from .profile import PhaseStats
from .profile import RequestProfile

logger = logging.getLogger(__name__)


def _event_uuid(event: dict[str, Any]) -> str | None:
    value = event.get("uuid")
    return value if isinstance(value, str) and value else None


def _timestamp(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _integer(*values: Any) -> int:
    for value in values:
        if isinstance(value, int | float):
            return int(value)
    return 0


def _number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
    return None


def _question(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("input", "query", "question"):
            if key in data:
                return _question(data[key])
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict):
                    content = message.get("content") or _nested(message, "data", "content")
                    if isinstance(content, str):
                        return content
    return "" if data is None else json.dumps(data, ensure_ascii=False, default=str)


def _load_events(path: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid ATOF JSON on line %d", line_number)
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _phase_for(event: dict[str, Any], starts: dict[str, dict[str, Any]]) -> str:
    parent_uuid = event.get("parent_uuid")
    visited: set[str] = set()
    while isinstance(parent_uuid, str) and parent_uuid not in visited:
        visited.add(parent_uuid)
        parent = starts.get(parent_uuid)
        if parent is None:
            break
        name = str(parent.get("name") or "").lower()
        if name == PHASE_PLANNER:
            return PHASE_PLANNER
        if name == "researcher-agent" or (parent.get("category") == "agent" and "researcher" in name):
            return PHASE_RESEARCHER
        parent_uuid = parent.get("parent_uuid")
    return PHASE_ORCHESTRATOR


def _root_uuid(event: dict[str, Any], starts: dict[str, dict[str, Any]]) -> str | None:
    event_uuid = _event_uuid(event)
    parent_uuid = event.get("parent_uuid")
    current = event_uuid if event_uuid in starts else parent_uuid
    if not isinstance(current, str) or not current:
        return None
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        parent = starts.get(current, {}).get("parent_uuid")
        if not isinstance(parent, str) or parent not in starts:
            return current
        current = parent
    return None


def _usage(event: dict[str, Any]) -> tuple[int, int, int, int, float | None]:
    profile = event.get("category_profile")
    profile = profile if isinstance(profile, dict) else {}
    annotated = profile.get("annotated_response")
    annotated = annotated if isinstance(annotated, dict) else {}
    usage = annotated.get("usage") or profile.get("usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    cost = usage.get("cost") or annotated.get("cost") or profile.get("cost")
    cost_total = (
        _number(cost.get("total"), cost.get("total_cost"), cost.get("usd")) if isinstance(cost, dict) else _number(cost)
    )
    return (
        _integer(usage.get("prompt_tokens"), usage.get("input_tokens")),
        _integer(usage.get("cached_tokens"), _nested(input_details, "cached_tokens")),
        _integer(usage.get("completion_tokens"), usage.get("output_tokens")),
        _integer(usage.get("reasoning_tokens"), _nested(output_details, "reasoning_tokens")),
        cost_total,
    )


def _model(event: dict[str, Any]) -> str:
    profile = event.get("category_profile") or {}
    annotated = profile.get("annotated_response") if isinstance(profile, dict) else {}
    for value in (
        annotated.get("model") if isinstance(annotated, dict) else None,
        annotated.get("model_name") if isinstance(annotated, dict) else None,
        profile.get("model_name") if isinstance(profile, dict) else None,
        event.get("name"),
    ):
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _parse_request(
    request_index: int,
    root: dict[str, Any],
    events: list[dict[str, Any]],
    starts: dict[str, dict[str, Any]],
    pricing: PricingRegistry,
) -> RequestProfile:
    ends = {
        event_uuid: event
        for event in events
        if (event_uuid := _event_uuid(event)) is not None
        if event.get("kind") == "scope" and event.get("scope_category") == "end"
    }
    root_end = ends.get(str(root.get("uuid")), {})
    duration_s = max(0.0, _timestamp(root_end.get("timestamp")) - _timestamp(root.get("timestamp")))
    phase_model_stats: dict[tuple[str, str], PhaseStats] = {}
    model_call_counters: dict[str, int] = {}
    llm_call_events: list[dict[str, Any]] = []
    tool_call_events: list[dict[str, Any]] = []
    tool_calls: dict[str, int] = {}

    for start in events:
        if start.get("kind") != "scope" or start.get("scope_category") != "start":
            continue
        end = ends.get(str(start.get("uuid")))
        if end is None:
            continue
        category = start.get("category")
        dur_s = max(0.0, _timestamp(end.get("timestamp")) - _timestamp(start.get("timestamp")))
        if category == "tool":
            name = str(start.get("name") or "unknown")
            tool_calls[name] = tool_calls.get(name, 0) + 1
            tool_call_events.append(
                {"tool": name, "dur_s": round(dur_s, 3), "cost_usd": pricing.get_tool(name).cost_per_call}
            )
            continue
        if category != "llm":
            continue

        model = _model(end)
        prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens, relay_cost = _usage(end)
        phase = _phase_for(start, starts)
        key = (phase, model)
        stats = phase_model_stats.setdefault(key, PhaseStats(phase=phase, model=model))
        try:
            price = pricing.get(model)
            calculated_cost = price.cost(prompt_tokens, cached_tokens, completion_tokens)
            savings = price.cache_savings(cached_tokens)
        except KeyError:
            calculated_cost = savings = 0.0
            if relay_cost is None:
                logger.warning("No price for model %r and no Relay cost; cost will be 0", model)
        cost = relay_cost if relay_cost is not None else calculated_cost
        stats.llm_calls += 1
        stats.prompt_tokens += prompt_tokens
        stats.cached_tokens += cached_tokens
        stats.completion_tokens += completion_tokens
        stats.cost_usd += cost
        stats.cache_savings_usd += savings
        call_index = model_call_counters.get(model, 0)
        model_call_counters[model] = call_index + 1
        llm_call_events.append(
            {
                "uuid": start.get("uuid"),
                "isl": prompt_tokens,
                "osl": completion_tokens,
                "cached": cached_tokens,
                "reasoning": reasoning_tokens,
                "dur_s": round(dur_s, 3),
                "tps": round(completion_tokens / dur_s, 2) if dur_s else 0.0,
                "model": model,
                "phase": phase,
                "call_idx": call_index,
            }
        )

    phases = list(phase_model_stats.values())
    return RequestProfile(
        request_index=request_index,
        question=_question(root.get("data")),
        duration_s=duration_s,
        phases=phases,
        tool_calls=tool_calls,
        llm_call_events=llm_call_events,
        tool_call_events=tool_call_events,
        total_llm_calls=sum(phase.llm_calls for phase in phases),
        total_prompt_tokens=sum(phase.prompt_tokens for phase in phases),
        total_cached_tokens=sum(phase.cached_tokens for phase in phases),
        total_completion_tokens=sum(phase.completion_tokens for phase in phases),
        total_cost_usd=sum(phase.cost_usd for phase in phases),
        total_tool_cost_usd=sum(event["cost_usd"] for event in tool_call_events),
        total_cache_savings_usd=sum(phase.cache_savings_usd for phase in phases),
    )


def parse_trace(path: str, pricing: PricingRegistry) -> list[RequestProfile]:
    """Parse Relay ATOF JSONL into one profile per workflow root scope."""
    events = _load_events(path)
    starts = {
        event_uuid: event
        for event in events
        if (event_uuid := _event_uuid(event)) is not None
        if event.get("kind") == "scope" and event.get("scope_category") == "start"
    }
    explicit_roots = [
        event for event in starts.values() if _nested(event, "metadata", "deep-researcher.component.type") == "workflow"
    ]
    roots = explicit_roots or [
        event
        for event in starts.values()
        if not isinstance(event.get("parent_uuid"), str) or event.get("parent_uuid") not in starts
    ]
    roots.sort(key=lambda event: _timestamp(event.get("timestamp")))

    events_by_root: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if (root_uuid := _root_uuid(event, starts)) is not None:
            events_by_root.setdefault(root_uuid, []).append(event)

    profiles: list[RequestProfile] = []
    for request_index, root in enumerate(roots):
        root_uuid = root["uuid"]
        request_events = events_by_root.get(root_uuid, [])
        try:
            profiles.append(_parse_request(request_index, root, request_events, starts, pricing))
        except Exception as exc:
            logger.warning(
                "Failed to parse Relay request; skipping (request_index=%d, error_type=%s)",
                request_index,
                type(exc).__name__,
            )
    return profiles
