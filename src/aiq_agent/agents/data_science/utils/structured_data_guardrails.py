# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local structured-data serialization, budgets, caching, and diagnostics."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from contextvars import Token
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from .analysis_runtime import register_structured_result


@dataclass(frozen=True, slots=True)
class StructuredDataCallBudget:
    """Optional catalog and text-to-SQL limits plus exact-call caching."""

    catalog_calls: int | None = None
    text_to_sql_calls: int | None = None
    cache_repeated_calls: bool = True


@dataclass(frozen=True, slots=True)
class StructuredDataCallRecord:
    """Compact, non-sensitive evidence-gain diagnostic for one call."""

    tool_name: str
    status: str
    cached: bool
    row_count: int | None = None
    candidate_count: int | None = None
    coverage: float | None = None
    truncated: bool | None = None


@dataclass(slots=True)
class _StructuredDataRunState:
    provider: str
    catalog_tools: frozenset[str]
    text_to_sql_tools: frozenset[str]
    budget: StructuredDataCallBudget
    counts: dict[str, int] = field(default_factory=lambda: {"catalog": 0, "text_to_sql": 0})
    cache: dict[str, ToolMessage] = field(default_factory=dict)
    records: list[StructuredDataCallRecord] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_CURRENT_RUN: ContextVar[_StructuredDataRunState | None] = ContextVar(
    "current_data_science_structured_data_run",
    default=None,
)


def _cache_key(tool_name: str, args: Any) -> str:
    try:
        serialized = json.dumps(args, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        serialized = repr(args)
    return f"{tool_name}:{serialized}"


def _record_from_message(tool_name: str, message: ToolMessage, *, cached: bool) -> StructuredDataCallRecord:
    try:
        payload = json.loads(str(message.content or ""))
    except json.JSONDecodeError:
        payload = None
    status = str(getattr(message, "status", None) or "success")
    row_count = candidate_count = None
    coverage = truncated = None
    if isinstance(payload, dict):
        status = str(payload.get("status") or status)
        rows = payload.get("rows")
        candidates = payload.get("candidates")
        row_count = len(rows) if isinstance(rows, list) else None
        candidate_count = len(candidates) if isinstance(candidates, list) else None
        coverage = payload.get("coverage") if isinstance(payload.get("coverage"), (int, float)) else None
        truncated = payload.get("truncated") if isinstance(payload.get("truncated"), bool) else None
    return StructuredDataCallRecord(
        tool_name=tool_name,
        status=status,
        cached=cached,
        row_count=row_count,
        candidate_count=candidate_count,
        coverage=coverage,
        truncated=truncated,
    )


class StructuredDataCallGuardMiddleware(AgentMiddleware):
    """Guard explicitly assigned catalog and text-to-SQL tools for one provider."""

    def __init__(
        self,
        *,
        provider: str,
        catalog_tools: frozenset[str],
        text_to_sql_tools: frozenset[str],
        budget: StructuredDataCallBudget,
    ) -> None:
        overlap = catalog_tools & text_to_sql_tools
        if overlap:
            raise ValueError(f"structured-data guard tool roles overlap: {', '.join(sorted(overlap))}")
        self.provider = provider
        self.catalog_tools = catalog_tools
        self.text_to_sql_tools = text_to_sql_tools
        self.budget = budget

    def begin_run(self) -> Token[_StructuredDataRunState | None]:
        """Install isolated accounting for one async request."""

        return _CURRENT_RUN.set(
            _StructuredDataRunState(
                provider=self.provider,
                catalog_tools=self.catalog_tools,
                text_to_sql_tools=self.text_to_sql_tools,
                budget=self.budget,
            )
        )

    @staticmethod
    def summarize_run() -> dict[str, Any]:
        """Return compact counters suitable for tracing and tests."""

        state = _CURRENT_RUN.get()
        if state is None:
            return {}
        return {
            "provider": state.provider,
            "catalog_calls": state.counts["catalog"],
            "text_to_sql_calls": state.counts["text_to_sql"],
            "cache_hits": sum(record.cached for record in state.records),
            "records": [
                {
                    "tool_name": record.tool_name,
                    "status": record.status,
                    "cached": record.cached,
                    "row_count": record.row_count,
                    "candidate_count": record.candidate_count,
                    "coverage": record.coverage,
                    "truncated": record.truncated,
                }
                for record in state.records
            ],
        }

    @staticmethod
    def end_run(token: Token[_StructuredDataRunState | None]) -> None:
        """Restore the prior request-local accounting context."""

        _CURRENT_RUN.reset(token)

    def _role_for(self, tool_name: str) -> str | None:
        if tool_name in self.catalog_tools:
            return "catalog"
        if tool_name in self.text_to_sql_tools:
            return "text_to_sql"
        return None

    @staticmethod
    def _register_sql_evidence(
        state: _StructuredDataRunState,
        tool_call: dict[str, Any],
        message: ToolMessage,
    ) -> ToolMessage:
        """Persist exact SQL rows and annotate the model-facing receipt."""

        try:
            payload = json.loads(str(message.content or ""))
        except json.JSONDecodeError:
            return message
        if not isinstance(payload, dict):
            return message
        args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
        tool_name = str(tool_call.get("name") or message.name or "")
        reference = register_structured_result(
            provider=state.provider,
            tool_name=tool_name,
            question=str(args.get("question") or ""),
            database_name=str(args["database_name"]) if args.get("database_name") is not None else None,
            payload=payload,
        )
        if reference is None:
            return message
        payload["analysis_ref"] = reference
        payload["analysis_hint"] = (
            f"Use analysis_rows('{reference}') in the Python tool to load these exact rows; do not copy them manually."
        )
        try:
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return message
        return message.model_copy(update={"content": content})

    async def awrap_tool_call(self, request, handler):
        """Guard assigned tools while leaving predictive and unrelated tools unchanged."""

        tool_call = request.tool_call if isinstance(request.tool_call, dict) else {}
        tool_name = str(tool_call.get("name") or "")
        role = self._role_for(tool_name)
        if role is None:
            return await handler(request)

        state = _CURRENT_RUN.get()
        if state is None:
            return await handler(request)

        async with state.lock:
            cache_key = _cache_key(tool_name, tool_call.get("args"))
            if state.budget.cache_repeated_calls and cache_key in state.cache:
                cached = state.cache[cache_key].model_copy(
                    update={"tool_call_id": tool_call.get("id", f"{state.provider}-cache-hit"), "name": tool_name}
                )
                state.records.append(_record_from_message(tool_name, cached, cached=True))
                return cached

            limit = state.budget.catalog_calls if role == "catalog" else state.budget.text_to_sql_calls
            if limit is not None and state.counts[role] >= limit:
                blocked = ToolMessage(
                    content=json.dumps(
                        {
                            "status": "error",
                            "code": f"aiq_{state.provider}_call_budget_exhausted",
                            "message": (
                                f"The request-local {tool_name} limit of {limit} has been reached. "
                                "Use collected evidence and synthesize a bounded answer."
                            ),
                            "retryable": False,
                        },
                        separators=(",", ":"),
                    ),
                    tool_call_id=tool_call.get("id", f"{state.provider}-budget-exhausted"),
                    name=tool_name,
                    status="error",
                )
                state.records.append(StructuredDataCallRecord(tool_name, "budget_exhausted", False))
                return blocked

            state.counts[role] += 1
            result = await handler(request)
            if isinstance(result, ToolMessage):
                if role == "text_to_sql":
                    result = self._register_sql_evidence(state, tool_call, result)
                record = _record_from_message(tool_name, result, cached=False)
                state.records.append(record)
                if state.budget.cache_repeated_calls and record.status != "error":
                    state.cache[cache_key] = result
            return result


__all__ = [
    "StructuredDataCallBudget",
    "StructuredDataCallGuardMiddleware",
]
