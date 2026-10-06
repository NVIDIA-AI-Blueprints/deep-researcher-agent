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

"""Tests for Relay ATOF tokenomics post-processing."""

import json
from pathlib import Path

import pytest

from deep_researcher_agent.tokenomics import atof_adapter
from deep_researcher_agent.tokenomics.atof_adapter import parse_trace
from deep_researcher_agent.tokenomics.pricing import PricingRegistry
from deep_researcher_agent.tokenomics.profile import PHASE_ORCHESTRATOR
from deep_researcher_agent.tokenomics.profile import PHASE_PLANNER
from deep_researcher_agent.tokenomics.profile import PHASE_RESEARCHER


def _pricing() -> PricingRegistry:
    return PricingRegistry.from_dict(
        {
            "models": {
                "test-model": {
                    "input_per_1m_tokens": 1.0,
                    "cached_input_per_1m_tokens": 0.5,
                    "output_per_1m_tokens": 2.0,
                }
            },
            "tools": {"search": {"cost_per_call": 0.01}},
        }
    )


def _scope(
    uuid: str,
    category: str,
    name: str,
    phase: str,
    timestamp: str,
    *,
    parent_uuid: str = "ambient",
    data: object = None,
    metadata: dict | None = None,
    category_profile: dict | None = None,
) -> dict:
    return {
        "atof_version": "0.1",
        "kind": "scope",
        "uuid": uuid,
        "parent_uuid": parent_uuid,
        "category": category,
        "name": name,
        "scope_category": phase,
        "timestamp": timestamp,
        "data": data,
        "metadata": metadata,
        "category_profile": category_profile,
    }


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8")


def test_parse_trace_uses_real_agent_ancestry_and_relay_cost(tmp_path: Path) -> None:
    root_metadata = {"deep-researcher.component.type": "workflow", "session_id": "session-1"}
    usage = {
        "annotated_response": {
            "model": "test-model",
            "usage": {
                "prompt_tokens": 100,
                "cached_tokens": 20,
                "completion_tokens": 40,
                "reasoning_tokens": 7,
                "cost": {"total": 0.25},
            },
        }
    }
    events = [
        _scope(
            "root",
            "function",
            "workflow",
            "start",
            "2026-01-01T00:00:00Z",
            data={"query": "why?"},
            metadata=root_metadata,
        ),
        _scope("orch", "llm", "test-model", "start", "2026-01-01T00:00:01Z", parent_uuid="root"),
        _scope("orch", "llm", "test-model", "end", "2026-01-01T00:00:02Z", parent_uuid="root", category_profile=usage),
        _scope("planner", "agent", "planner-agent", "start", "2026-01-01T00:00:02Z", parent_uuid="root"),
        _scope("plan-llm", "llm", "test-model", "start", "2026-01-01T00:00:03Z", parent_uuid="planner"),
        _scope(
            "plan-llm",
            "llm",
            "test-model",
            "end",
            "2026-01-01T00:00:04Z",
            parent_uuid="planner",
            category_profile=usage,
        ),
        _scope("planner", "agent", "planner-agent", "end", "2026-01-01T00:00:04Z", parent_uuid="root"),
        _scope("researcher", "agent", "researcher-agent", "start", "2026-01-01T00:00:04Z", parent_uuid="root"),
        _scope("research-llm", "llm", "test-model", "start", "2026-01-01T00:00:05Z", parent_uuid="researcher"),
        _scope(
            "research-llm",
            "llm",
            "test-model",
            "end",
            "2026-01-01T00:00:06Z",
            parent_uuid="researcher",
            category_profile=usage,
        ),
        _scope("search", "tool", "search", "start", "2026-01-01T00:00:06Z", parent_uuid="researcher"),
        _scope("search", "tool", "search", "end", "2026-01-01T00:00:07Z", parent_uuid="researcher"),
        _scope("researcher", "agent", "researcher-agent", "end", "2026-01-01T00:00:07Z", parent_uuid="root"),
        _scope("root", "function", "workflow", "end", "2026-01-01T00:00:08Z", metadata=root_metadata),
    ]
    path = tmp_path / "relay.atof.jsonl"
    _write(path, events)

    profiles = parse_trace(str(path), _pricing())

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.question == "why?"
    assert profile.duration_s == pytest.approx(8.0)
    assert profile.total_llm_calls == 3
    assert profile.total_cost_usd == pytest.approx(0.75)
    assert profile.total_tool_cost_usd == pytest.approx(0.01)
    assert {event["phase"] for event in profile.llm_call_events} == {
        PHASE_ORCHESTRATOR,
        PHASE_PLANNER,
        PHASE_RESEARCHER,
    }
    assert profile.llm_call_events[0]["reasoning"] == 7


def test_parse_trace_skips_invalid_json_and_uses_catalog_fallback(tmp_path: Path) -> None:
    usage = {"annotated_response": {"model": "test-model", "usage": {"input_tokens": 100, "output_tokens": 50}}}
    events = [
        _scope(
            "root",
            "function",
            "workflow",
            "start",
            "2026-01-01T00:00:00Z",
            metadata={"deep-researcher.component.type": "workflow"},
        ),
        _scope("llm", "llm", "test-model", "start", "2026-01-01T00:00:01Z", parent_uuid="root"),
        _scope("llm", "llm", "test-model", "end", "2026-01-01T00:00:02Z", parent_uuid="root", category_profile=usage),
        _scope("root", "function", "workflow", "end", "2026-01-01T00:00:03Z"),
    ]
    path = tmp_path / "relay.atof.jsonl"
    _write(path, events)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("not-json\n")

    profile = parse_trace(str(path), _pricing())[0]

    assert profile.total_prompt_tokens == 100
    assert profile.total_completion_tokens == 50
    assert profile.total_cost_usd == pytest.approx(0.0002)


def test_parse_trace_ignores_non_string_identifiers(tmp_path: Path) -> None:
    events = [
        _scope(
            "root",
            "function",
            "workflow",
            "start",
            "2026-01-01T00:00:00Z",
            metadata={"deep-researcher.component.type": "workflow"},
        ),
        _scope("root", "function", "workflow", "end", "2026-01-01T00:00:01Z"),
        {"kind": "scope", "scope_category": "start", "uuid": ["invalid"], "parent_uuid": {"invalid": True}},
    ]
    path = tmp_path / "relay.atof.jsonl"
    _write(path, events)

    profiles = parse_trace(str(path), _pricing())

    assert len(profiles) == 1


def test_parse_trace_tolerates_non_mapping_category_profile(tmp_path: Path) -> None:
    events = [
        _scope(
            "root",
            "function",
            "workflow",
            "start",
            "2026-01-01T00:00:00Z",
            metadata={"deep-researcher.component.type": "workflow"},
        ),
        _scope("llm", "llm", "test-model", "start", "2026-01-01T00:00:01Z", parent_uuid="root"),
        _scope("llm", "llm", "test-model", "end", "2026-01-01T00:00:02Z", parent_uuid="root"),
        _scope("root", "function", "workflow", "end", "2026-01-01T00:00:03Z"),
    ]
    events[2]["category_profile"] = ["malformed"]
    path = tmp_path / "relay.atof.jsonl"
    _write(path, events)

    profiles = parse_trace(str(path), _pricing())

    assert len(profiles) == 1
    assert profiles[0].total_llm_calls == 1


def test_parse_trace_failure_log_excludes_exception_content(tmp_path: Path, monkeypatch, caplog) -> None:
    events = [
        _scope(
            "root",
            "function",
            "workflow",
            "start",
            "2026-01-01T00:00:00Z",
            metadata={"deep-researcher.component.type": "workflow"},
        )
    ]
    path = tmp_path / "relay.atof.jsonl"
    _write(path, events)

    def fail(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("customer-secret")

    monkeypatch.setattr(atof_adapter, "_parse_request", fail)

    assert parse_trace(str(path), _pricing()) == []
    assert "RuntimeError" in caplog.text
    assert "customer-secret" not in caplog.text
