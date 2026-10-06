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

"""Researcher-facing source tool adapters."""

from __future__ import annotations

import asyncio
import json
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from contextvars import Token
from dataclasses import dataclass
from dataclasses import field

from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from deep_researcher_agent.common.citation_verification import is_non_citable_status_output
from deep_researcher_agent.relay import ainvoke_tool_with_relay

from ..resource_limits import DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES

DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS = 5
DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE = 4
DEFAULT_SOURCE_TOOL_CONCURRENCY_TIMEOUT = 120.0
_SOURCE_TOOL_FAILURE_OUTPUT = "ERROR: Source batch returned no citable results."


class SourceToolBudgetExceeded(RuntimeError):
    """Raised before an external call would exceed the per-job budget."""


class SourceToolCircuitOpen(RuntimeError):
    """Raised when a provider failure wave opens the per-job source circuit."""


def source_tool_result_failed(result: object) -> bool:
    """Return whether a source result is provider status rather than evidence.

    The classifier is deliberately conservative: it recognizes whole-result
    error/status shapes and leading provider errors, but does not treat an
    arbitrary article mentioning an HTTP error as a failed tool call.
    """
    if result is None:
        return True
    if isinstance(result, (dict, list)) and not result:
        return True
    if getattr(result, "status", None) == "error":
        return True
    text = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
    text = text.strip()
    if not text:
        return True
    return is_non_citable_status_output(text)


@dataclass
class SourceToolCallBudget:
    """Concurrency-safe per-job counter inherited by nested researcher tasks."""

    max_calls: int
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES
    used_calls: int = 0
    consecutive_failures: int = 0
    circuit_open: bool = False
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        self._lock = asyncio.Lock()

    async def consume(self, count: int = 1) -> None:
        """Reserve calls atomically before invoking an external provider."""
        if count < 1:
            raise ValueError("count must be >= 1")
        async with self._lock:
            if self.circuit_open:
                raise SourceToolCircuitOpen(
                    f"Source-tool circuit is open after {self.consecutive_failures} consecutive provider failures"
                )
            if self.used_calls + count > self.max_calls:
                raise SourceToolBudgetExceeded(
                    f"Source-tool call budget exhausted ({self.used_calls}/{self.max_calls} calls used)"
                )
            self.used_calls += count

    async def ensure_circuit_closed(self) -> None:
        """Reject a queued invocation if an earlier provider call opened the circuit."""
        async with self._lock:
            if self.circuit_open:
                raise SourceToolCircuitOpen(
                    f"Source-tool circuit is open after {self.consecutive_failures} consecutive provider failures"
                )

    async def record_result(self, result: object) -> None:
        """Update consecutive-failure state and open the circuit when exhausted."""
        failed = source_tool_result_failed(result)
        async with self._lock:
            if self.circuit_open:
                return
            if not failed:
                self.consecutive_failures = 0
                return

            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.circuit_open = True
                raise SourceToolCircuitOpen(
                    f"Source-tool circuit opened after {self.consecutive_failures} consecutive provider failures"
                )


_source_tool_budget: ContextVar[SourceToolCallBudget | None] = ContextVar(
    "deep_researcher_source_tool_budget",
    default=None,
)


def activate_source_tool_budget(
    max_calls: int,
    *,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES,
) -> Token[SourceToolCallBudget | None]:
    """Install a fresh budget for one deep-research run."""
    return _source_tool_budget.set(
        SourceToolCallBudget(max_calls=max_calls, max_consecutive_failures=max_consecutive_failures)
    )


def reset_source_tool_budget(token: Token[SourceToolCallBudget | None]) -> None:
    """Restore the caller's prior budget context."""
    _source_tool_budget.reset(token)


async def _consume_source_tool_budget(count: int = 1) -> None:
    budget = _source_tool_budget.get()
    if budget is not None:
        await budget.consume(count)


async def _record_source_tool_result(result: object) -> None:
    budget = _source_tool_budget.get()
    if budget is not None:
        await budget.record_result(result)


async def _ensure_source_tool_circuit_closed() -> None:
    budget = _source_tool_budget.get()
    if budget is not None:
        await budget.ensure_circuit_closed()


class SourceToolConcurrencyLimiter:
    """Shared per-event-loop limiter for source tool calls."""

    def __init__(
        self,
        max_concurrent: int,
        *,
        acquire_timeout: float | None = DEFAULT_SOURCE_TOOL_CONCURRENCY_TIMEOUT,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if acquire_timeout is not None and acquire_timeout <= 0:
            raise ValueError("acquire_timeout must be > 0 or None")
        self.max_concurrent = max_concurrent
        self.acquire_timeout = acquire_timeout
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            self._semaphores[loop] = semaphore
        return semaphore

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """Acquire one source-tool slot and release it on success, failure, or cancellation."""
        semaphore = self._get_semaphore()
        acquired = False
        try:
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=self.acquire_timeout)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Timed out waiting for a source-tool concurrency slot after {self.acquire_timeout} seconds"
                ) from exc
            acquired = True
            yield
        finally:
            if acquired:
                semaphore.release()


class BatchSourceToolInput(BaseModel):
    """Input schema for batch-capable same-name source tool wrappers."""

    model_config = ConfigDict(extra="forbid")

    queries: str | list[str] = Field(description="One query/input string, or a list of query/input strings.")


def _single_string_input_field(tool: BaseTool) -> str | None:
    """Return the sole string model field when no injected fields would be dropped."""
    model_fields = getattr(tool.tool_call_schema, "model_fields", None)
    full_fields = getattr(tool.get_input_schema(), "model_fields", None)
    if not model_fields or len(model_fields) != 1 or not full_fields or set(full_fields) != set(model_fields):
        return None

    name, field = next(iter(model_fields.items()))
    if field.annotation is str:
        return name
    return None


def _format_batch_tool_output(results: list[tuple[str, str | None, str | None]]) -> str:
    """Render successful evidence without making failed-only batches citable."""
    parts: list[str] = []
    for query, output, error in results:
        if error is None and output:
            parts.append(f"## Query: {query}\n{output}")
    if not parts:
        return _SOURCE_TOOL_FAILURE_OUTPUT
    return "\n\n---\n\n".join(parts)


def _make_batch_source_tool(
    original_tool: BaseTool,
    *,
    input_field_name: str,
    limiter: SourceToolConcurrencyLimiter,
    max_batch_size: int,
) -> BaseTool:
    """Create a same-name wrapper that fans out list input to the original tool."""

    async def _run_batch(queries: str | list[str]) -> str:
        query_list = [queries] if isinstance(queries, str) else list(queries)
        if not query_list:
            return "No queries provided."
        if len(query_list) > max_batch_size:
            return (
                f"ERROR: {original_tool.name} accepts at most {max_batch_size} queries per batch. "
                f"Received {len(query_list)}."
            )
        await _consume_source_tool_budget(len(query_list))

        async def _call_one(query: str) -> tuple[str, str | None, str | None]:
            try:
                async with limiter.limit():
                    await _ensure_source_tool_circuit_closed()
                    try:
                        result = await ainvoke_tool_with_relay(original_tool, {input_field_name: query})
                    except SourceToolCircuitOpen:
                        raise
                    except Exception:  # noqa: BLE001 - represented as per-item failure for the LLM
                        await _record_source_tool_result(None)
                        return query, None, "Source request failed."
                    failed = source_tool_result_failed(result)
                    await _record_source_tool_result(result)
                if failed:
                    return query, None, "Source request failed."
                return query, str(result), None
            except SourceToolCircuitOpen:
                return query, None, "Source request skipped because the source circuit is open."

        results = await asyncio.gather(*(_call_one(query) for query in query_list))
        return _format_batch_tool_output(results)

    description = (
        f"{original_tool.description}\n\n"
        "Batch mode: pass `queries` as either a single string or a list of strings. "
        "Each item is run as one underlying source-tool call and returned in grouped sections."
    )
    return StructuredTool.from_function(
        coroutine=_run_batch,
        name=original_tool.name,
        description=description,
        args_schema=BatchSourceToolInput,
    )


def _make_throttled_source_tool(
    original_tool: BaseTool,
    *,
    limiter: SourceToolConcurrencyLimiter,
) -> BaseTool:
    """Create a same-name wrapper that throttles calls to a non-batchable source tool."""

    async def _run_throttled(**kwargs) -> object:
        await _consume_source_tool_budget()
        async with limiter.limit():
            await _ensure_source_tool_circuit_closed()
            try:
                result = await ainvoke_tool_with_relay(original_tool, kwargs)
            except SourceToolCircuitOpen:
                raise
            except Exception:
                await _record_source_tool_result(None)
                return _SOURCE_TOOL_FAILURE_OUTPUT
            await _record_source_tool_result(result)
            if source_tool_result_failed(result):
                return _SOURCE_TOOL_FAILURE_OUTPUT
        return result

    # Retain injected fields for runtime validation/forwarding. BaseTool derives
    # tool_call_schema from this full schema and hides InjectedToolArg fields from
    # the model-facing contract.
    args_schema = getattr(original_tool, "args_schema", None) or original_tool.get_input_schema()

    return StructuredTool.from_function(
        coroutine=_run_throttled,
        name=original_tool.name,
        description=original_tool.description,
        args_schema=args_schema,
        return_direct=original_tool.return_direct,
        response_format=original_tool.response_format,
    )


def adapt_source_tools_for_research(
    tools: list[BaseTool],
    *,
    source_tool_names: set[str],
    max_concurrent_source_tool_calls: int = DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
    max_batch_size: int = DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
) -> list[BaseTool]:
    """Return researcher-facing tools with source calls internally throttled.

    Compatible single-string source tools are upgraded to same-name batch-capable
    tools. Other source tools keep their original schema and receive a
    throttle-only wrapper. Non-source tools are returned unchanged.
    """
    adapted_tools: list[BaseTool] = []
    limiter = SourceToolConcurrencyLimiter(max_concurrent_source_tool_calls)
    for candidate in tools:
        if candidate.name not in source_tool_names:
            adapted_tools.append(candidate)
            continue

        input_field_name = _single_string_input_field(candidate)
        if input_field_name is None:
            adapted_tools.append(_make_throttled_source_tool(candidate, limiter=limiter))
            continue

        adapted_tools.append(
            _make_batch_source_tool(
                candidate,
                input_field_name=input_field_name,
                limiter=limiter,
                max_batch_size=max_batch_size,
            )
        )

    return adapted_tools
