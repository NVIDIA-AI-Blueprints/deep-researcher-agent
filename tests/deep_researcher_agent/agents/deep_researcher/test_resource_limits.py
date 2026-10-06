# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable deep-research resource-limit ceilings."""

import pytest
from pydantic import ValidationError

from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_FINAL_REPORT_BYTES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_EXECUTION_SECONDS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_INPUT_CHARS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_NOTE_BYTES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_PLAN_BYTES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_QUERIES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_QUERY_CHARS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_SOURCE_ROUTING_BYTES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_SOURCE_TOOL_CALLS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_STATE_FILE_COUNT
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_TODO_ITEM_CHARS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_TODO_ITEMS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_TOTAL_RESEARCH_NOTE_BYTES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_TOTAL_STATE_BYTES
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_TOTAL_TODO_CHARS
from deep_researcher_agent.agents.deep_researcher.resource_limits import DeepResearchResourceLimits
from deep_researcher_agent.agents.deep_researcher.resource_limits import StateBudgetLedger
from deep_researcher_agent.agents.deep_researcher.resource_limits import state_backend_file_sizes


def test_resource_limit_defaults_equal_non_disableable_security_ceiling():
    """Default values are the largest configuration accepted by the schema."""
    limits = DeepResearchResourceLimits()

    assert limits.model_dump() == {
        "max_input_chars": DEFAULT_MAX_RESEARCH_INPUT_CHARS,
        "max_execution_seconds": DEFAULT_MAX_RESEARCH_EXECUTION_SECONDS,
        "max_plan_bytes": DEFAULT_MAX_RESEARCH_PLAN_BYTES,
        "max_source_routing_bytes": DEFAULT_MAX_SOURCE_ROUTING_BYTES,
        "max_final_report_bytes": DEFAULT_MAX_FINAL_REPORT_BYTES,
        "max_state_file_count": DEFAULT_MAX_STATE_FILE_COUNT,
        "max_total_state_bytes": DEFAULT_MAX_TOTAL_STATE_BYTES,
        "max_research_queries": DEFAULT_MAX_RESEARCH_QUERIES,
        "max_total_query_chars": DEFAULT_MAX_RESEARCH_QUERY_CHARS,
        "max_research_note_bytes": DEFAULT_MAX_RESEARCH_NOTE_BYTES,
        "max_total_research_note_bytes": DEFAULT_MAX_TOTAL_RESEARCH_NOTE_BYTES,
        "max_source_tool_calls": DEFAULT_MAX_SOURCE_TOOL_CALLS,
        "max_consecutive_source_tool_failures": DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES,
        "max_todo_items": DEFAULT_MAX_TODO_ITEMS,
        "max_todo_item_chars": DEFAULT_MAX_TODO_ITEM_CHARS,
        "max_total_todo_chars": DEFAULT_MAX_TOTAL_TODO_CHARS,
    }


@pytest.mark.parametrize(
    ("field", "ceiling"),
    [
        ("max_input_chars", DEFAULT_MAX_RESEARCH_INPUT_CHARS),
        ("max_execution_seconds", DEFAULT_MAX_RESEARCH_EXECUTION_SECONDS),
        ("max_plan_bytes", DEFAULT_MAX_RESEARCH_PLAN_BYTES),
        ("max_source_routing_bytes", DEFAULT_MAX_SOURCE_ROUTING_BYTES),
        ("max_final_report_bytes", DEFAULT_MAX_FINAL_REPORT_BYTES),
        ("max_state_file_count", DEFAULT_MAX_STATE_FILE_COUNT),
        ("max_total_state_bytes", DEFAULT_MAX_TOTAL_STATE_BYTES),
        ("max_research_queries", DEFAULT_MAX_RESEARCH_QUERIES),
        ("max_total_query_chars", DEFAULT_MAX_RESEARCH_QUERY_CHARS),
        ("max_research_note_bytes", DEFAULT_MAX_RESEARCH_NOTE_BYTES),
        ("max_total_research_note_bytes", DEFAULT_MAX_TOTAL_RESEARCH_NOTE_BYTES),
        ("max_source_tool_calls", DEFAULT_MAX_SOURCE_TOOL_CALLS),
        ("max_consecutive_source_tool_failures", DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES),
        ("max_todo_items", DEFAULT_MAX_TODO_ITEMS),
        ("max_todo_item_chars", DEFAULT_MAX_TODO_ITEM_CHARS),
        ("max_total_todo_chars", DEFAULT_MAX_TOTAL_TODO_CHARS),
    ],
)
def test_resource_limits_reject_one_above_each_security_ceiling(field, ceiling):
    """Configuration can tighten each limit but cannot raise the security envelope."""
    with pytest.raises(ValidationError):
        DeepResearchResourceLimits.model_validate({field: ceiling + 1})


def test_resource_limits_accept_downward_overrides_and_are_frozen():
    """Operators may reduce limits, but a running job cannot mutate them."""
    limits = DeepResearchResourceLimits(
        max_input_chars=1024,
        max_execution_seconds=60,
        max_plan_bytes=4096,
        max_source_routing_bytes=2048,
        max_final_report_bytes=8192,
        max_state_file_count=8,
        max_total_state_bytes=16_384,
        max_research_queries=2,
        max_total_query_chars=512,
        max_research_note_bytes=2048,
        max_total_research_note_bytes=4096,
        max_source_tool_calls=8,
        max_consecutive_source_tool_failures=4,
        max_todo_items=5,
        max_todo_item_chars=128,
        max_total_todo_chars=512,
    )

    assert limits.max_research_queries == 2
    with pytest.raises(ValidationError):
        limits.max_research_queries = 3


def test_state_backend_file_sizes_handles_route_forms_and_payload_shapes():
    """Only StateBackend-routed seeds count when a sandbox backs the workspace."""
    files = {
        "/shared/plan.json": "é",
        "/output.md": b"abc",
        "/research_note_01.json": {"content": ["one", "two"]},
        "/workspace/large.bin": b"x" * 100,
        "/skills/private.txt": "ignored",
    }

    assert state_backend_file_sizes(files, sandbox_enabled=True) == {
        "/shared/plan.json": 2,
        "/shared/output.md": 3,
        "/shared/research_note_01.json": 7,
    }
    assert state_backend_file_sizes(files, sandbox_enabled=False)["/workspace/large.bin"] == 100


def test_state_budget_rejects_duplicate_route_aliases_before_graph_construction():
    """A prefixed and route-local seed cannot ambiguously target the same state file."""
    with pytest.raises(ValueError, match="duplicate aliases"):
        StateBudgetLedger(
            limits=DeepResearchResourceLimits(),
            files={"/shared/output.md": "first", "/output.md": "second"},
            sandbox_enabled=True,
        )


def test_state_budget_enforces_seeded_file_count_and_aggregate_bytes():
    """Seeded StateBackend content is bounded before any graph/backend mutation."""
    with pytest.raises(ValueError, match="2-file limit"):
        StateBudgetLedger(
            limits=DeepResearchResourceLimits(max_state_file_count=2),
            files={"/a": "1", "/b": "2", "/c": "3"},
            sandbox_enabled=False,
        )
    with pytest.raises(ValueError, match="4-byte aggregate limit"):
        StateBudgetLedger(
            limits=DeepResearchResourceLimits(max_total_state_bytes=4),
            files={"/a": "ééé"},
            sandbox_enabled=False,
        )


def test_state_budget_reservation_is_replacement_aware_and_rollback_safe():
    """Runtime state reservations enforce totals and restore capacity after failure."""
    limits = DeepResearchResourceLimits(max_state_file_count=2, max_total_state_bytes=6)
    ledger = StateBudgetLedger(
        limits=limits,
        files={"/shared/plan.json": "1234"},
        sandbox_enabled=True,
    )

    replacement = ledger.reserve([("/shared/plan.json", b"12")])
    second = ledger.reserve([("/shared/output.md", b"3456")])
    with pytest.raises(ValueError, match="aggregate limit"):
        ledger.reserve([("/shared/output.md", b"34567")])

    ledger.rollback(second)
    latest = ledger.reserve([("/shared/output.md", b"3456")])
    ledger.rollback(second)  # Superseded reservation must not corrupt later accounting.
    with pytest.raises(ValueError, match="aggregate limit"):
        ledger.reserve([("/shared/plan.json", b"123")])

    ledger.rollback(latest)
    ledger.rollback(replacement)
    restored = ledger.reserve([("/shared/output.md", b"34")])
    with pytest.raises(ValueError, match="aggregate limit"):
        ledger.reserve([("/shared/output.md", b"345")])
    ledger.rollback(restored)


def test_state_budget_reserve_uses_non_sandbox_path_canonicalization():
    """Route-local mutations replace route-local seeds when StateBackend owns every path."""
    ledger = StateBudgetLedger(
        limits=DeepResearchResourceLimits(max_state_file_count=3, max_total_state_bytes=3),
        files={"/output.md": "12"},
        sandbox_enabled=False,
    )

    replacement = ledger.reserve([("/output.md", b"123")])

    with pytest.raises(ValueError, match="aggregate limit"):
        ledger.reserve([("/output.md", b"1234")])
    ledger.rollback(replacement)
    # /output.md must be restored to 2 bytes; only then does one more byte fit
    # the 3-byte aggregate cap.
    ledger.reserve([("/notes.md", b"1")])
    with pytest.raises(ValueError, match="aggregate limit"):
        ledger.reserve([("/extra.md", b"1")])
