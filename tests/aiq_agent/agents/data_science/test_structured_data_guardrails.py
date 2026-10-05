# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for request-local structured-data tool-call controls."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage

from aiq_agent.agents.data_science.utils.analysis_runtime import begin_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import end_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import get_analysis_run
from aiq_agent.agents.data_science.utils.structured_data_guardrails import StructuredDataCallBudget
from aiq_agent.agents.data_science.utils.structured_data_guardrails import StructuredDataCallGuardMiddleware

CATALOG_TOOL = "ontology__catalog_search"
SQL_TOOL = "ontology__text_to_sql"


def _middleware(
    *,
    catalog_calls: int | None = None,
    text_to_sql_calls: int | None = None,
) -> StructuredDataCallGuardMiddleware:
    return StructuredDataCallGuardMiddleware(
        provider="ontology",
        catalog_tools=frozenset({CATALOG_TOOL}),
        text_to_sql_tools=frozenset({SQL_TOOL}),
        budget=StructuredDataCallBudget(
            catalog_calls=catalog_calls,
            text_to_sql_calls=text_to_sql_calls,
        ),
    )


def _request(tool_name: str, call_id: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": tool_name, "id": call_id, "args": args})


@pytest.mark.asyncio
async def test_exact_repeated_call_is_cached_and_diagnosed() -> None:
    middleware = _middleware(catalog_calls=2, text_to_sql_calls=2)
    handler = AsyncMock(
        return_value=ToolMessage(
            content='{"request_id":"r1","rows":[{"value":3}],"truncated":false}',
            tool_call_id="call-1",
            name=SQL_TOOL,
        )
    )
    token = middleware.begin_run()
    try:
        first = await middleware.awrap_tool_call(
            _request(SQL_TOOL, "call-1", {"question": "Total revenue", "database_name": "db"}),
            handler,
        )
        second = await middleware.awrap_tool_call(
            _request(SQL_TOOL, "call-2", {"question": "Total revenue", "database_name": "db"}),
            handler,
        )
        summary = middleware.summarize_run()
    finally:
        middleware.end_run(token)

    assert handler.await_count == 1
    assert first.tool_call_id == "call-1"
    assert second.tool_call_id == "call-2"
    assert summary["text_to_sql_calls"] == 1
    assert summary["cache_hits"] == 1
    assert summary["records"][0]["row_count"] == 1


@pytest.mark.asyncio
async def test_distinct_calls_hit_hard_budget_without_invoking_handler() -> None:
    middleware = _middleware(catalog_calls=1)
    handler = AsyncMock(
        return_value=ToolMessage(
            content='{"request_id":"r1","coverage":1.0,"candidates":[]}',
            tool_call_id="call-1",
            name=CATALOG_TOOL,
        )
    )
    token = middleware.begin_run()
    try:
        await middleware.awrap_tool_call(
            _request(CATALOG_TOOL, "call-1", {"question": "Revenue", "database_name": "db"}),
            handler,
        )
        blocked = await middleware.awrap_tool_call(
            _request(CATALOG_TOOL, "call-2", {"question": "Customers", "database_name": "db"}),
            handler,
        )
        summary = middleware.summarize_run()
    finally:
        middleware.end_run(token)

    assert handler.await_count == 1
    assert blocked.status == "error"
    assert json.loads(str(blocked.content))["code"] == "aiq_ontology_call_budget_exhausted"
    assert summary["records"][-1]["status"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_non_finite_sql_payload_is_not_registered_for_python() -> None:
    middleware = _middleware(text_to_sql_calls=1)
    original = ToolMessage(
        content='{"request_id":"r1","rows":[{"value":NaN}]}',
        tool_call_id="call-1",
        name=SQL_TOOL,
    )
    handler = AsyncMock(return_value=original)
    analysis_token = begin_analysis_run()
    guard_token = middleware.begin_run()
    try:
        result = await middleware.awrap_tool_call(
            _request(SQL_TOOL, "call-1", {"question": "Value", "database_name": "db"}),
            handler,
        )
        analysis_state = get_analysis_run()
        assert analysis_state is not None
    finally:
        middleware.end_run(guard_token)
        await end_analysis_run(analysis_token)

    assert result is original
    assert analysis_state.structured_results == []


@pytest.mark.asyncio
async def test_unassigned_tool_passes_through_without_accounting() -> None:
    middleware = _middleware(catalog_calls=1, text_to_sql_calls=1)
    expected = ToolMessage(content="result", tool_call_id="other-1", name="other_tool")
    handler = AsyncMock(return_value=expected)
    token = middleware.begin_run()
    try:
        result = await middleware.awrap_tool_call(_request("other_tool", "other-1", {"query": "value"}), handler)
        summary = middleware.summarize_run()
    finally:
        middleware.end_run(token)

    assert result is expected
    assert handler.await_count == 1
    assert summary["catalog_calls"] == 0
    assert summary["text_to_sql_calls"] == 0
    assert summary["cache_hits"] == 0


@pytest.mark.asyncio
async def test_error_response_is_not_cached_for_a_retry() -> None:
    middleware = _middleware(text_to_sql_calls=2)
    handler = AsyncMock(
        side_effect=[
            ToolMessage(
                content='{"status":"error","retryable":true}',
                tool_call_id="call-1",
                name=SQL_TOOL,
                status="error",
            ),
            ToolMessage(
                content='{"request_id":"r2","rows":[{"value":4}]}',
                tool_call_id="call-2",
                name=SQL_TOOL,
            ),
        ]
    )
    args = {"question": "Revenue", "database_name": "db"}
    token = middleware.begin_run()
    try:
        first = await middleware.awrap_tool_call(_request(SQL_TOOL, "call-1", args), handler)
        second = await middleware.awrap_tool_call(_request(SQL_TOOL, "call-2", args), handler)
    finally:
        middleware.end_run(token)

    assert first.status == "error"
    assert second.status == "success"
    assert handler.await_count == 2


@pytest.mark.asyncio
async def test_successful_sql_result_gets_stable_python_reference() -> None:
    middleware = _middleware(text_to_sql_calls=2)
    handler = AsyncMock(
        return_value=ToolMessage(
            content='{"request_id":"r1","sql":"SELECT value","rows":[{"value":3}],"truncated":false}',
            tool_call_id="call-1",
            name=SQL_TOOL,
        )
    )
    analysis_token = begin_analysis_run()
    guard_token = middleware.begin_run()
    try:
        result = await middleware.awrap_tool_call(
            _request(SQL_TOOL, "call-1", {"question": "Value", "database_name": "db"}),
            handler,
        )
        payload = json.loads(str(result.content))
        analysis_state = get_analysis_run()
        assert analysis_state is not None
        manifest = json.loads(analysis_state.manifest_path.read_text(encoding="utf-8"))
    finally:
        middleware.end_run(guard_token)
        await end_analysis_run(analysis_token)

    assert payload["analysis_ref"] == "structured_1"
    assert "analysis_rows('structured_1')" in payload["analysis_hint"]
    assert manifest["results"][0]["ref"] == "structured_1"
    assert manifest["results"][0]["provider"] == "ontology"
    assert manifest["results"][0]["tool_name"] == SQL_TOOL
    assert manifest["results"][0]["row_count"] == 1
