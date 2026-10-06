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

"""Tests for researcher-facing source tool adapters."""

import asyncio
from contextlib import suppress
from datetime import datetime
from typing import Annotated
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool

from deep_researcher_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from deep_researcher_agent.agents.deep_researcher.tools import source_tool_batching
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import SourceToolBudgetExceeded
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import SourceToolCallBudget
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import SourceToolCircuitOpen
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import SourceToolConcurrencyLimiter
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import activate_source_tool_budget
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import adapt_source_tools_for_research
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import reset_source_tool_budget
from deep_researcher_agent.agents.deep_researcher.tools.source_tool_batching import source_tool_result_failed


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("Error: provider rate limited the request", True),
        ("Error 432: provider capacity exhausted", True),
        ("Search failed with status 429", True),
        ('{"status_code": 429, "detail": "rate limited"}', True),
        ({"status": 503, "error": "unavailable"}, True),
        ({}, True),
        ([], True),
        ({"status": "error"}, True),
        ({"timestamp": datetime(2026, 8, 5)}, False),
        ({"content": b"bytes"}, False),
        ({"artifact": object()}, False),
        ({"status": "error", "timestamp": datetime(2026, 8, 5)}, True),
        ("Search returned no results", True),
        ("An article explaining HTTP error 429 handling patterns.", False),
        ('<Document href="https://example.com">Evidence</Document>', False),
    ],
)
def test_source_tool_result_failure_classifier_is_conservative(result: object, expected: bool):
    assert source_tool_result_failed(result) is expected


@pytest.mark.asyncio
async def test_batch_circuit_stops_queued_provider_calls_after_first_failure():
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        return "Error: provider unavailable"

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=2,
    )[0]
    token = activate_source_tool_budget(2, max_consecutive_failures=1)
    try:
        output = await wrapped.ainvoke({"queries": ["first", "second"]})
        assert output == "ERROR: Source batch returned no citable results."
        assert calls == ["first"]
        with pytest.raises(SourceToolCircuitOpen, match="circuit is open after 1 consecutive"):
            await wrapped.ainvoke({"queries": "third"})
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_late_success_cannot_reset_open_circuit_failure_count():
    budget = SourceToolCallBudget(max_calls=2, max_consecutive_failures=1)
    success_in_flight = asyncio.Event()
    release_success = asyncio.Event()

    async def finish_success() -> None:
        success_in_flight.set()
        await release_success.wait()
        await budget.record_result("valid evidence")

    success_task = asyncio.create_task(finish_success())
    await success_in_flight.wait()
    with pytest.raises(SourceToolCircuitOpen, match="opened after 1 consecutive"):
        await budget.record_result("Error: provider unavailable")
    release_success.set()
    await success_task

    assert budget.circuit_open is True
    assert budget.consecutive_failures == 1
    with pytest.raises(SourceToolCircuitOpen, match="is open after 1 consecutive"):
        await budget.consume()


@pytest.mark.asyncio
async def test_batch_retains_success_that_finishes_after_sibling_opens_circuit(monkeypatch: pytest.MonkeyPatch):
    good_started = asyncio.Event()
    release_good = asyncio.Event()
    circuit_opened = asyncio.Event()

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        if query == "good":
            good_started.set()
            await release_good.wait()
            return "Late evidence at https://example.test/late-good"
        await good_started.wait()
        return "Error: provider unavailable"

    record_result = source_tool_batching._record_source_tool_result

    async def record_and_signal(result: object) -> None:
        try:
            await record_result(result)
        except SourceToolCircuitOpen:
            circuit_opened.set()
            raise

    monkeypatch.setattr(source_tool_batching, "_record_source_tool_result", record_and_signal)
    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=2,
    )[0]
    token = activate_source_tool_budget(2, max_consecutive_failures=1)
    try:
        batch_task = asyncio.create_task(wrapped.ainvoke({"queries": ["good", "bad"]}))
        await circuit_opened.wait()
        release_good.set()
        output = await batch_task

        assert "## Query: good" in output
        assert "https://example.test/late-good" in output
        assert "## Query: bad" not in output
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_batch_preserves_completed_success_when_sibling_opens_circuit():
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        if query == "bad":
            return "Error: provider unavailable"
        return "Evidence at https://example.test/good"

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=2,
    )[0]
    token = activate_source_tool_budget(2, max_consecutive_failures=1)
    try:
        output = await wrapped.ainvoke({"queries": ["good", "bad"]})

        assert calls == ["good", "bad"]
        assert "## Query: good" in output
        assert "https://example.test/good" in output
        assert "## Query: bad" not in output
        assert "provider unavailable" not in output
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_result", [{}, [], {"status": "error"}])
async def test_batch_filters_structured_failures_from_citable_output(failed_result: object):
    @tool
    async def search_tool(query: str) -> object:
        """Search a source."""
        return failed_result

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=1,
    )[0]

    output = await wrapped.ainvoke({"queries": "query"})

    assert output == "ERROR: Source batch returned no citable results."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_result",
    ["Error: provider unavailable", {}, {"status": "error"}],
)
async def test_throttled_wrapper_sanitizes_classified_source_failures(failed_result: object):
    @tool
    async def search_tool(query: str, limit: int) -> object:
        """Search a source with a result limit."""
        return failed_result

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=1,
    )[0]

    output = await wrapped.ainvoke({"query": "query", "limit": 1})

    assert output == "ERROR: Source batch returned no citable results."


@pytest.mark.asyncio
@pytest.mark.parametrize("batchable", [True, False])
async def test_provider_exception_counts_toward_source_circuit(batchable: bool):
    calls = 0

    if batchable:

        @tool
        async def search_tool(query: str) -> str:
            """Search a source."""
            nonlocal calls
            calls += 1
            raise TimeoutError("provider timed out")

        invocation = {"queries": "query"}
    else:

        @tool
        async def search_tool(query: str, limit: int) -> str:
            """Search a source."""
            nonlocal calls
            calls += 1
            raise TimeoutError("provider timed out")

        invocation = {"query": "query", "limit": 1}

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=2,
    )[0]
    token = activate_source_tool_budget(2, max_consecutive_failures=2)
    try:
        if batchable:
            assert await wrapped.ainvoke(invocation) == "ERROR: Source batch returned no citable results."
        else:
            assert await wrapped.ainvoke(invocation) == "ERROR: Source batch returned no citable results."
        if batchable:
            assert await wrapped.ainvoke(invocation) == "ERROR: Source batch returned no citable results."
        else:
            with pytest.raises(SourceToolCircuitOpen, match="opened after 2 consecutive"):
                await wrapped.ainvoke(invocation)
        assert calls == 2
        with pytest.raises(SourceToolCircuitOpen, match="circuit is open after 2 consecutive"):
            await wrapped.ainvoke(invocation)
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_source_failure_wave_opens_circuit_before_another_provider_call():
    calls = 0

    @tool
    async def search_tool(query: str, limit: int) -> str:
        """Search a source."""
        nonlocal calls
        calls += 1
        return "Error 432: provider capacity exhausted"

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=1,
    )[0]
    token = activate_source_tool_budget(20, max_consecutive_failures=3)
    try:
        for _ in range(2):
            assert await wrapped.ainvoke({"query": "q", "limit": 1}) == (
                "ERROR: Source batch returned no citable results."
            )
        with pytest.raises(SourceToolCircuitOpen, match="opened after 3 consecutive"):
            await wrapped.ainvoke({"query": "q", "limit": 1})
        with pytest.raises(SourceToolCircuitOpen, match="circuit is open"):
            await wrapped.ainvoke({"query": "q", "limit": 1})
        assert calls == 3
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_successful_source_result_resets_consecutive_failure_count():
    results = iter(["Error: transient", "valid evidence", "Error: transient", "Error: transient"])

    @tool
    async def search_tool(query: str, limit: int) -> str:
        """Search a source."""
        return next(results)

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=1,
    )[0]
    token = activate_source_tool_budget(20, max_consecutive_failures=2)
    try:
        assert await wrapped.ainvoke({"query": "q1", "limit": 1}) == (
            "ERROR: Source batch returned no citable results."
        )
        assert await wrapped.ainvoke({"query": "q2", "limit": 1}) == "valid evidence"
        assert await wrapped.ainvoke({"query": "q3", "limit": 1}) == (
            "ERROR: Source batch returned no citable results."
        )
        with pytest.raises(SourceToolCircuitOpen, match="opened after 2 consecutive"):
            await wrapped.ainvoke({"query": "q4", "limit": 1})
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_batch_wrapper_single_string_calls_original_once():
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        return f"result for {query}"

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=3,
    )

    wrapped = result[0]
    output = await wrapped.ainvoke({"queries": "alpha"})

    assert wrapped.name == "search_tool"
    assert calls == ["alpha"]
    assert "## Query: alpha" in output
    assert "result for alpha" in output


@pytest.mark.asyncio
async def test_batch_wrapper_list_calls_original_once_per_item():
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        return f"https://example.test/{query}"

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=3,
        max_batch_size=3,
    )

    output = await result[0].ainvoke({"queries": ["alpha", "beta", "gamma"]})

    assert sorted(calls) == ["alpha", "beta", "gamma"]
    assert "## Query: alpha" in output
    assert "## Query: beta" in output
    assert "## Query: gamma" in output
    assert "https://example.test/beta" in output


@pytest.mark.asyncio
async def test_batch_wrapper_represents_partial_failures_per_item():
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        if query == "bad":
            raise RuntimeError("backend unavailable")
        return f"ok {query}"

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=3,
    )

    output = await result[0].ainvoke({"queries": ["good", "bad"]})

    assert sorted(calls) == ["bad", "good"]
    assert "## Query: good" in output
    assert "ok good" in output
    assert "## Query: bad" not in output
    assert "backend unavailable" not in output


@pytest.mark.asyncio
async def test_batch_wrapper_rejects_oversized_tool_batches_without_calling_original():
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        return query

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=1,
    )

    output = await result[0].ainvoke({"queries": ["a", "b"]})

    assert calls == []
    assert "ERROR: search_tool accepts at most 1 queries per batch" in output


@pytest.mark.asyncio
async def test_source_call_budget_accepts_exact_batch_items_and_rejects_one_more():
    """Every batch item counts as one provider call and overage is atomic."""
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        return query

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=3,
        max_batch_size=3,
    )[0]
    token = activate_source_tool_budget(3)
    try:
        await wrapped.ainvoke({"queries": ["a", "b", "c"]})

        assert sorted(calls) == ["a", "b", "c"]
        with pytest.raises(SourceToolBudgetExceeded, match=r"3/3 calls used"):
            await wrapped.ainvoke({"queries": "d"})
        assert sorted(calls) == ["a", "b", "c"]
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_source_call_budget_is_shared_by_batchable_and_non_batchable_tools():
    """All source adapter shapes consume the same job-local ledger."""
    calls: list[str] = []

    @tool
    async def batchable(query: str) -> str:
        """Search a source."""
        calls.append(f"batch:{query}")
        return query

    @tool
    async def non_batchable(query: str, limit: int) -> str:
        """Search a source with a result limit."""
        calls.append(f"plain:{query}:{limit}")
        return query

    wrapped = {
        item.name: item
        for item in adapt_source_tools_for_research(
            [batchable, non_batchable],
            source_tool_names={"batchable", "non_batchable"},
            max_concurrent_source_tool_calls=2,
            max_batch_size=2,
        )
    }
    token = activate_source_tool_budget(2)
    try:
        await wrapped["batchable"].ainvoke({"queries": "a"})
        await wrapped["non_batchable"].ainvoke({"query": "b", "limit": 1})

        with pytest.raises(SourceToolBudgetExceeded):
            await wrapped["non_batchable"].ainvoke({"query": "c", "limit": 1})
        assert calls == ["batch:a", "plain:b:1"]
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_inferred_schema_source_tool_is_budgeted_and_throttled():
    """BaseTool schemas inferred from _run signatures cannot bypass either guard."""
    calls: list[str] = []
    active = 0
    max_seen = 0

    class InferredSchemaSourceTool(BaseTool):
        name: str = "inferred_source"
        description: str = "Source tool whose args_schema is inferred by BaseTool."

        def _run(self, query: str, limit: int = 1) -> str:
            raise NotImplementedError

        async def _arun(self, query: str, limit: int = 1) -> str:
            nonlocal active, max_seen
            calls.append(query)
            active += 1
            max_seen = max(max_seen, active)
            await asyncio.sleep(0.01)
            active -= 1
            return f"{query}:{limit}"

    original = InferredSchemaSourceTool()
    assert original.args_schema is None
    wrapped = adapt_source_tools_for_research(
        [original],
        source_tool_names={original.name},
        max_concurrent_source_tool_calls=1,
        max_batch_size=2,
    )[0]

    token = activate_source_tool_budget(2)
    try:
        results = await asyncio.gather(
            wrapped.ainvoke({"query": "a", "limit": 1}),
            wrapped.ainvoke({"query": "b", "limit": 2}),
        )
        assert results == ["a:1", "b:2"]
        assert max_seen == 1

        with pytest.raises(SourceToolBudgetExceeded, match=r"2/2 calls used"):
            await wrapped.ainvoke({"query": "c", "limit": 3})
        assert sorted(calls) == ["a", "b"]
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_source_adapters_use_model_facing_schema_without_injected_arguments():
    """Injected arguments stay runtime-forwardable but never enter the model schema."""
    calls: list[str] = []

    @tool
    async def batchable(
        query: str,
    ) -> str:
        """Search a source that is safe to expose as a batch wrapper."""
        calls.append(f"batch:{query}")
        return query

    @tool
    async def injected_single(
        query: str,
        credential: Annotated[str, InjectedToolArg],
    ) -> str:
        """Search a source with one model argument and one required injected argument."""
        calls.append(f"injected:{query}:{credential}")
        return query

    @tool
    async def throttled(
        query: str,
        limit: int,
        credential: Annotated[str, InjectedToolArg],
    ) -> str:
        """Search a source with multiple model arguments and an injected credential."""
        calls.append(f"plain:{query}:{limit}:{credential}")
        return query

    wrapped = {
        item.name: item
        for item in adapt_source_tools_for_research(
            [batchable, injected_single, throttled],
            source_tool_names={"batchable", "injected_single", "throttled"},
            max_concurrent_source_tool_calls=1,
            max_batch_size=2,
        )
    }

    assert set(wrapped["batchable"].tool_call_schema.model_fields) == {"queries"}
    assert set(wrapped["injected_single"].tool_call_schema.model_fields) == {"query"}
    assert set(wrapped["throttled"].tool_call_schema.model_fields) == {"query", "limit"}
    assert set(wrapped["injected_single"].get_input_schema().model_fields) == {"query", "credential"}
    assert set(wrapped["throttled"].get_input_schema().model_fields) == {"query", "limit", "credential"}
    assert "credential" not in wrapped["injected_single"].tool_call_schema.model_fields
    assert "credential" not in wrapped["throttled"].tool_call_schema.model_fields

    await wrapped["batchable"].ainvoke({"queries": "alpha"})
    await wrapped["injected_single"].ainvoke({"query": "beta", "credential": "runtime-secret"})
    await wrapped["throttled"].ainvoke({"query": "gamma", "limit": 2, "credential": "runtime-secret"})
    assert calls == [
        "batch:alpha",
        "injected:beta:runtime-secret",
        "plain:gamma:2:runtime-secret",
    ]


@pytest.mark.asyncio
async def test_source_call_budget_reservation_is_concurrency_safe():
    """Concurrent wrappers cannot race past the remaining provider-call budget."""
    calls: list[str] = []
    release = asyncio.Event()

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        await release.wait()
        return query

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=1,
    )[0]
    token = activate_source_tool_budget(1)
    try:
        first = asyncio.create_task(wrapped.ainvoke({"queries": "a"}))
        await asyncio.sleep(0)
        second = asyncio.create_task(wrapped.ainvoke({"queries": "b"}))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

        assert len(calls) == 1
        assert sum(isinstance(result, SourceToolBudgetExceeded) for result in results) == 1
    finally:
        reset_source_tool_budget(token)


@pytest.mark.asyncio
async def test_source_call_budget_is_fresh_per_activation_and_reset_restores_parent():
    """Nested jobs receive isolated ledgers and reset restores the caller context exactly."""
    calls: list[str] = []

    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        calls.append(query)
        return query

    wrapped = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=1,
    )[0]
    outer = activate_source_tool_budget(1)
    try:
        await wrapped.ainvoke({"queries": "outer"})
        inner = activate_source_tool_budget(1)
        try:
            await wrapped.ainvoke({"queries": "inner"})
        finally:
            reset_source_tool_budget(inner)

        with pytest.raises(SourceToolBudgetExceeded):
            await wrapped.ainvoke({"queries": "outer-overage"})
    finally:
        reset_source_tool_budget(outer)

    await wrapped.ainvoke({"queries": "unbudgeted-caller"})
    assert calls == ["outer", "inner", "unbudgeted-caller"]


@pytest.mark.asyncio
async def test_source_registry_captures_urls_from_wrapped_tool_output():
    @tool
    async def search_tool(query: str) -> str:
        """Search a source."""
        return f"{query}: https://example.test/source"

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=2,
    )
    output = await result[0].ainvoke({"queries": ["alpha"]})

    middleware = SourceRegistryMiddleware(source_tool_names={"search_tool"})
    request = MagicMock()
    request.tool_call = {"name": "search_tool"}
    handler = AsyncMock(return_value=ToolMessage(content=output, tool_call_id="tc1"))

    await middleware.awrap_tool_call(request, handler)

    sources = middleware.registry.all_sources()
    assert len(sources) == 1
    assert sources[0].url == "https://example.test/source"


@pytest.mark.asyncio
async def test_incompatible_multi_arg_source_tool_keeps_schema_and_is_throttled():
    @tool
    async def search_tool(query: str, limit: int) -> str:
        """Search a source."""
        return f"{query}:{limit}"

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=2,
        max_batch_size=3,
    )
    wrapped = result[0]

    assert wrapped.name == "search_tool"
    assert wrapped.args == search_tool.args
    assert await wrapped.ainvoke({"query": "alpha", "limit": 5}) == "alpha:5"


@pytest.mark.asyncio
async def test_shared_limiter_caps_underlying_calls_across_wrapped_tools():
    active = 0
    max_seen = 0

    async def _recorded_result(query: str) -> str:
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        active -= 1
        return query

    @tool
    async def search_a(query: str) -> str:
        """Search source A."""
        return await _recorded_result(query)

    @tool
    async def search_b(query: str) -> str:
        """Search source B."""
        return await _recorded_result(query)

    result = adapt_source_tools_for_research(
        [search_a, search_b],
        source_tool_names={"search_a", "search_b"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=3,
    )
    wrapped_tools = {wrapped.name: wrapped for wrapped in result}

    await asyncio.gather(
        wrapped_tools["search_a"].ainvoke({"queries": ["a1", "a2"]}),
        wrapped_tools["search_b"].ainvoke({"queries": ["b1", "b2"]}),
    )

    assert max_seen == 1


@pytest.mark.asyncio
async def test_shared_limiter_caps_non_batchable_source_tools():
    active = 0
    max_seen = 0

    @tool
    async def search_tool(query: str, limit: int) -> str:
        """Search a source."""
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        active -= 1
        return f"{query}:{limit}"

    result = adapt_source_tools_for_research(
        [search_tool],
        source_tool_names={"search_tool"},
        max_concurrent_source_tool_calls=1,
        max_batch_size=3,
    )

    await asyncio.gather(*(result[0].ainvoke({"query": f"q{i}", "limit": i}) for i in range(3)))

    assert max_seen == 1


@pytest.mark.asyncio
async def test_limiter_caps_concurrent_blocks():
    limiter = SourceToolConcurrencyLimiter(1)
    active = 0
    max_seen = 0

    async def hold_slot():
        nonlocal active, max_seen
        async with limiter.limit():
            active += 1
            max_seen = max(max_seen, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(hold_slot() for _ in range(3)))

    assert max_seen == 1


@pytest.mark.asyncio
async def test_limiter_releases_after_exception():
    limiter = SourceToolConcurrencyLimiter(1)

    async def fail_with_slot():
        async with limiter.limit():
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await fail_with_slot()

    async with asyncio.timeout(0.1):
        async with limiter.limit():
            pass


@pytest.mark.asyncio
async def test_limiter_timeout_does_not_release_unacquired_slot():
    limiter = SourceToolConcurrencyLimiter(1, acquire_timeout=0.01)

    async with limiter.limit():
        with pytest.raises(TimeoutError, match="Timed out waiting for a source-tool concurrency slot"):
            async with limiter.limit():
                pass

        with pytest.raises(TimeoutError, match="Timed out waiting for a source-tool concurrency slot"):
            async with limiter.limit():
                pass

    async with asyncio.timeout(0.1):
        async with limiter.limit():
            pass


@pytest.mark.asyncio
async def test_limiter_releases_after_cancellation():
    limiter = SourceToolConcurrencyLimiter(1)

    async def hold_slot():
        async with limiter.limit():
            await asyncio.sleep(1)

    task = asyncio.create_task(hold_slot())
    await asyncio.sleep(0.01)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    async with asyncio.timeout(0.1):
        async with limiter.limit():
            pass
