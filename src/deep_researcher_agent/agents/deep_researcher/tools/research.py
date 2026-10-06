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

"""Researcher runnable and batched research tool construction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any
from typing import cast

import nemo_relay
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.relay import agent_scope

from ..custom_middleware import ResearcherBudgetExhaustedError
from ..models import EvidenceJudgment
from ..models import ResearchGap
from ..models import ResearchNotes
from ..models import ResearchQuery
from ..resource_limits import DeepResearchResourceLimits
from ..resource_limits import StateBudgetLedger

_NO_TOOL_RUNTIME = cast(ToolRuntime, None)
logger = logging.getLogger(__name__)


class _MissingStructuredResponseError(ValueError):
    """Raised when a researcher worker returns no structured response."""


_NOTE_SLUG_MAX_LENGTH = 64
RESEARCHER_AGENT_NAME = "researcher-agent"


def _exhausted_research_notes(query: ResearchQuery) -> ResearchNotes:
    """Build the deterministic fallback when the reserved finalization turn fails."""
    return ResearchNotes(
        query_topic=query.query[:120],
        target_components=list(query.target_components),
        summary="Research for this query was cut short because its model-call budget was exhausted.",
        findings=[],
        gaps=[
            ResearchGap(
                description=f"No grounded evidence was finalized for: {query.query}",
                impact="Target components for this query may be unsupported in the final answer.",
                suggested_follow_up_queries=[query.query],
            )
        ],
        sources=[],
        narrative_notes="The researcher did not return structured notes during its reserved finalization turn.",
        language="en",
        evidence_judgment=EvidenceJudgment(
            relevance_score=0,
            confidence="low",
            rationale="The model-call budget was exhausted before verified findings were finalized.",
        ),
    )


def format_research_request(query: ResearchQuery) -> str:
    """Create the single-query researcher task text used by the batch tool."""
    query_json = json.dumps(query.model_dump(mode="json"), indent=2, ensure_ascii=False)
    return (
        "Batch research invocation. Execute this ResearchQuery and return a structured ResearchNotes response. "
        "Do not call write_file or edit_file; run_research_batch will persist the returned ResearchNotes under "
        "/shared/ after you return.\n\n"
        "ResearchQuery JSON:\n"
        f"{query_json}"
    )


def researcher_invoke_state(query: ResearchQuery, runtime: ToolRuntime | None) -> dict[str, Any]:
    """Build nested researcher state, carrying parent files for StateBackend-backed skills."""
    invoke_state: dict[str, Any] = {
        "messages": [HumanMessage(content=format_research_request(query))],
    }
    parent_state = getattr(runtime, "state", None) if runtime is not None else None
    if isinstance(parent_state, dict) and "files" in parent_state:
        invoke_state["files"] = parent_state["files"]
    return invoke_state


def researcher_invoke_config(runtime: ToolRuntime | None, callbacks: list[Any]) -> dict[str, Any]:
    """Build child-run config while preserving the active callback lineage."""
    config = dict(runtime.config) if runtime is not None else {}
    config.pop("run_id", None)
    config.pop("configurable", None)
    config["run_name"] = RESEARCHER_AGENT_NAME
    if not config.get("callbacks") and callbacks:
        config["callbacks"] = callbacks
    return config


async def _run_research_query(
    *,
    query: ResearchQuery,
    researcher_runnable: Any,
    runtime: ToolRuntime | None,
    callbacks: list[Any],
    semaphore: asyncio.Semaphore,
) -> ResearchNotes:
    """Run one researcher worker and return its structured notes."""
    async with semaphore:
        with agent_scope(RESEARCHER_AGENT_NAME, input_value=query) as lifecycle:
            try:
                result = await researcher_runnable.ainvoke(
                    researcher_invoke_state(query, runtime),
                    config=researcher_invoke_config(runtime, callbacks),
                )
            except (ModelCallLimitExceededError, ResearcherBudgetExhaustedError):
                logger.warning(
                    "Researcher worker exhausted its model-call budget (query_%s)",
                    log_content_metadata(query.query),
                )
                return _exhausted_research_notes(query)
            except Exception as exc:  # noqa: BLE001 - captured as per-item failure
                logger.warning(
                    "Researcher worker failed (error_type=%s, query_%s)",
                    type(exc).__name__,
                    log_content_metadata(query.query),
                )
                raise RuntimeError("researcher worker failed") from exc

            try:
                structured = result.get("structured_response") if isinstance(result, dict) else None
                if structured is None:
                    raise _MissingStructuredResponseError("researcher worker did not return structured ResearchNotes")
                note = ResearchNotes.model_validate(structured)
            except _MissingStructuredResponseError:
                raise
            except Exception as exc:  # noqa: BLE001 - captured as per-item failure
                logger.warning(
                    "Researcher worker returned invalid ResearchNotes (error_type=%s, query_%s)",
                    type(exc).__name__,
                    log_content_metadata(query.query),
                )
                raise ValueError("researcher worker returned invalid ResearchNotes") from exc

            lifecycle.output = note
            return note


def _research_note_slug(text: str) -> str:
    """Return a compact filesystem-safe slug for a research note."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    slug = slug[:_NOTE_SLUG_MAX_LENGTH].strip("_")
    return slug or "research_note"


def _research_note_path(query: ResearchQuery, note: ResearchNotes, index: int) -> str:
    """Build a stable /shared path for a returned research note."""
    digest_input = json.dumps(query.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:8]
    slug = _research_note_slug(note.query_topic or query.query)
    return f"/shared/research_note_{index:02d}_{slug}_{digest}.json"


def _research_note_files(queries: list[ResearchQuery], notes: list[ResearchNotes]) -> list[tuple[str, bytes]]:
    """Serialize returned research notes as shared JSON files."""
    return [
        (
            _research_note_path(query, note, index),
            json.dumps(note.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False).encode("utf-8"),
        )
        for index, (query, note) in enumerate(zip(queries, notes, strict=False), start=1)
    ]


def _persist_research_notes(
    *,
    backend: Any | None,
    note_files: list[tuple[str, bytes]],
    state_budget: StateBudgetLedger,
) -> None:
    """Persist returned ResearchNotes into parent /shared state."""
    if backend is None or not note_files:
        return

    reservation = state_budget.reserve(note_files)
    try:
        responses = backend.upload_files(note_files)
    except Exception:
        state_budget.rollback(reservation)
        raise
    errors = [f"{response.path}: {response.error}" for response in responses if getattr(response, "error", None)]
    if errors:
        state_budget.rollback(reservation)
        raise RuntimeError(f"failed to persist research note file(s): {'; '.join(errors)}")


async def _run_research_queries(
    *,
    queries: list[ResearchQuery],
    researcher_runnable: Any,
    runtime: ToolRuntime | None,
    callbacks: list[Any],
    max_concurrency: int,
) -> tuple[list[ResearchQuery], list[ResearchNotes], list[str]]:
    """Run researcher workers concurrently and collect successful query/note pairs plus surfaced errors."""
    semaphore = asyncio.Semaphore(min(max_concurrency, len(queries)))
    tasks = [
        asyncio.create_task(
            _run_research_query(
                query=query,
                researcher_runnable=researcher_runnable,
                runtime=runtime,
                callbacks=callbacks,
                semaphore=semaphore,
            ),
            context=nemo_relay.fork_asyncio_context(),
        )
        for query in queries
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_queries: list[ResearchQuery] = []
    notes: list[ResearchNotes] = []
    errors: list[str] = []
    for query, raw_result in zip(queries, raw_results, strict=False):
        if isinstance(raw_result, BaseException):
            error = str(raw_result) or raw_result.__class__.__name__
            errors.append(error)
        else:
            successful_queries.append(query)
            notes.append(raw_result)
    return successful_queries, notes, errors


def build_research_batch_tool(
    *,
    researcher_runnable: Any,
    callbacks: list[Any],
    max_research_concurrency: int,
    resource_limits: DeepResearchResourceLimits | None = None,
    backend: Any | None = None,
    state_budget: StateBudgetLedger | None = None,
    source_registry_middleware: Any | None = None,
) -> BaseTool:
    """Build an orchestrator-only tool that runs researcher tasks concurrently."""
    limits = resource_limits or DeepResearchResourceLimits()
    state_budget = state_budget or StateBudgetLedger(limits=limits, files={}, sandbox_enabled=True)
    ledger_lock = asyncio.Lock()
    consumed_queries = 0
    consumed_query_chars = 0
    persisted_note_count = 0
    persisted_note_bytes = 0

    @tool
    async def run_research_batch(
        queries: list[ResearchQuery],
        runtime: ToolRuntime = _NO_TOOL_RUNTIME,
    ) -> str:
        """Run planned research queries in parallel and return ResearchNotes JSON."""
        nonlocal consumed_queries
        nonlocal consumed_query_chars
        nonlocal persisted_note_bytes
        nonlocal persisted_note_count

        if not queries:
            return "[]"

        if len(queries) > max_research_concurrency:
            raise ValueError(
                f"run_research_batch accepts at most {max_research_concurrency} curated queries. "
                f"Received {len(queries)}. Rank, merge, or drop lower-priority queries and call again."
            )
        batch_query_chars = sum(
            len(query.query) + sum(len(subquery) for subquery in query.subqueries) for query in queries
        )
        async with ledger_lock:
            if consumed_queries + len(queries) > limits.max_research_queries:
                raise ValueError(f"run_research_batch exceeds the {limits.max_research_queries}-query per-job limit")
            if consumed_query_chars + batch_query_chars > limits.max_total_query_chars:
                raise ValueError(
                    f"run_research_batch exceeds the {limits.max_total_query_chars}-character aggregate query limit"
                )
            consumed_queries += len(queries)
            consumed_query_chars += batch_query_chars

        successful_queries, notes, errors = await _run_research_queries(
            queries=queries,
            researcher_runnable=researcher_runnable,
            runtime=runtime,
            callbacks=callbacks,
            max_concurrency=max_research_concurrency,
        )
        note_files = _research_note_files(successful_queries, notes)
        batch_note_bytes = sum(len(content) for _, content in note_files)
        oversized_notes = [path for path, content in note_files if len(content) > limits.max_research_note_bytes]
        if oversized_notes:
            raise ValueError(f"ResearchNotes exceeds the {limits.max_research_note_bytes}-byte per-note limit")
        async with ledger_lock:
            # Each accepted ResearchQuery can yield at most one ResearchNotes file.
            # Reusing the consumed-query ceiling therefore enforces a job-wide note
            # count no greater than max_research_queries (20 at the security cap).
            if persisted_note_count + len(note_files) > limits.max_research_queries:
                raise ValueError(f"ResearchNotes exceeds the {limits.max_research_queries}-note per-job limit")
            if persisted_note_bytes + batch_note_bytes > limits.max_total_research_note_bytes:
                raise ValueError(
                    f"ResearchNotes exceeds the {limits.max_total_research_note_bytes}-byte aggregate per-job limit"
                )
            persisted_note_count += len(note_files)
            persisted_note_bytes += batch_note_bytes

        try:
            _persist_research_notes(backend=backend, note_files=note_files, state_budget=state_budget)
        except Exception:
            async with ledger_lock:
                persisted_note_count -= len(note_files)
                persisted_note_bytes -= batch_note_bytes
            raise
        if source_registry_middleware is not None:
            source_registry_middleware.register_research_note_sources(notes)

        if errors:
            retained_detail = ""
            if notes:
                retained_actions = []
                if source_registry_middleware is not None:
                    retained_actions.append("registered")
                if backend is not None:
                    retained_actions.append("persisted under /shared/")
                retained_text = " and ".join(retained_actions) if retained_actions else "retained"
                retained_detail = (
                    f" {len(notes)} successful researcher worker(s) were {retained_text}; "
                    "resubmit only the failed queries."
                )
            raise RuntimeError(
                f"run_research_batch failed for {len(errors)} of {len(queries)} researcher worker(s). "
                f"Errors: {'; '.join(errors)}.{retained_detail}"
            )

        return json.dumps(
            [note.model_dump(mode="json", exclude_none=True) for note in notes],
            indent=2,
            ensure_ascii=False,
        )

    return run_research_batch
