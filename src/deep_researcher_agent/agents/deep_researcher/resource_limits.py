# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-budget contract for one deep-research job."""

from __future__ import annotations

import posixpath
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

DEFAULT_MAX_RESEARCH_INPUT_CHARS = 32_768
DEFAULT_MAX_RESEARCH_EXECUTION_SECONDS = 3600.0
DEFAULT_MAX_RESEARCH_PLAN_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_SOURCE_ROUTING_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_FINAL_REPORT_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_STATE_FILE_COUNT = 64
DEFAULT_MAX_TOTAL_STATE_BYTES = 24 * 1024 * 1024
DEFAULT_MAX_RESEARCH_QUERIES = 20
DEFAULT_MAX_RESEARCH_QUERY_CHARS = 10_000
DEFAULT_MAX_RESEARCH_NOTE_BYTES = 512 * 1024
DEFAULT_MAX_TOTAL_RESEARCH_NOTE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_SOURCE_TOOL_CALLS = 100
DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES = 8
DEFAULT_MAX_TODO_ITEMS = 20
DEFAULT_MAX_TODO_ITEM_CHARS = 2048
DEFAULT_MAX_TOTAL_TODO_CHARS = 10_000


class DeepResearchExecutionTimeout(TimeoutError):
    """Raised only when the deep-research job wall-clock budget expires."""


_SHARED_ROUTE = "/shared/"
_ROUTE_LOCAL_STATE_FILES = frozenset(
    {
        "/edit_instruction.txt",
        "/original_report.md",
        "/output.md",
        "/parent_report_context.json",
        "/plan.json",
        "/source_routing.json",
    }
)


def _normalized_state_path(path: object) -> str | None:
    """Return a canonical absolute virtual path, or None for an invalid key."""
    if not isinstance(path, str) or not path:
        return None
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return normalized if normalized.startswith("/") else f"/{normalized}"


def _canonical_state_backend_path(path: object, *, sandbox_enabled: bool) -> str | None:
    """Map a virtual or route-local StateBackend key to its canonical shared path.

    Without a sandbox, StateBackend is the default backend and every seeded file
    belongs to graph state. With a sandbox, only ``/shared/`` is StateBackend
    backed. DeepAgents strips that route before storing files, so resumed state
    may contain route-local keys such as ``/plan.json`` or
    ``/research_note_*.json``. Known route-local state keys are accepted while
    sandbox-backed paths (for example ``/workspace/...``) are excluded.
    """
    normalized = _normalized_state_path(path)
    if normalized is None:
        return None
    if not sandbox_enabled:
        return normalized
    if normalized == _SHARED_ROUTE.rstrip("/"):
        return normalized
    if normalized.startswith(_SHARED_ROUTE):
        return normalized
    if normalized in _ROUTE_LOCAL_STATE_FILES or (
        normalized.startswith("/research_note_") and normalized.endswith(".json") and normalized.count("/") == 1
    ):
        return f"/shared{normalized}"
    return None


def _state_file_content_bytes(entry: object) -> int:
    """Return the exact stored payload size for a DeepAgents state-file entry."""
    if isinstance(entry, Mapping):
        entry = entry.get("content")
    if isinstance(entry, bytes):
        return len(entry)
    if isinstance(entry, str):
        return len(entry.encode("utf-8"))
    if isinstance(entry, list) and all(isinstance(line, str) for line in entry):
        return len("\n".join(entry).encode("utf-8"))
    raise ValueError("Deep-research state files must contain text or bytes payloads")


def state_backend_file_sizes(
    files: Mapping[str, Any],
    *,
    sandbox_enabled: bool,
) -> dict[str, int]:
    """Return canonical StateBackend paths and payload sizes for seeded files."""
    sizes: dict[str, int] = {}
    for path, entry in files.items():
        canonical = _canonical_state_backend_path(path, sandbox_enabled=sandbox_enabled)
        if canonical is None:
            continue
        if canonical in sizes:
            raise ValueError(f"Deep-research state contains duplicate aliases for {canonical}")
        sizes[canonical] = _state_file_content_bytes(entry)
    return sizes


@dataclass(frozen=True)
class _StateBudgetReservation:
    """Opaque rollback token for one prospective state mutation."""

    changes: tuple[tuple[str, int, int | None], ...]


class StateBudgetLedger:
    """Thread-safe accounting for the bounded StateBackend filesystem."""

    def __init__(
        self,
        *,
        limits: DeepResearchResourceLimits,
        files: Mapping[str, Any],
        sandbox_enabled: bool,
    ) -> None:
        self._limits = limits
        self._sandbox_enabled = sandbox_enabled
        self._sizes = state_backend_file_sizes(files, sandbox_enabled=sandbox_enabled)
        self._generations = {path: 0 for path in self._sizes}
        self._next_generation = 1
        self._lock = threading.Lock()
        self._validate_totals(self._sizes)

    def _validate_totals(self, sizes: Mapping[str, int]) -> None:
        if len(sizes) > self._limits.max_state_file_count:
            raise ValueError(f"Deep-research shared state exceeds the {self._limits.max_state_file_count}-file limit")
        if sum(sizes.values()) > self._limits.max_total_state_bytes:
            raise ValueError(
                f"Deep-research shared state exceeds the {self._limits.max_total_state_bytes}-byte aggregate limit"
            )

    @staticmethod
    def _mutation_size(content: object) -> int:
        if isinstance(content, bytes):
            return len(content)
        if isinstance(content, str):
            return len(content.encode("utf-8"))
        raise ValueError("Deep-research state mutations must contain text or bytes payloads")

    def reserve(self, files: list[tuple[str, bytes | str]]) -> _StateBudgetReservation:
        """Reserve aggregate capacity before a StateBackend mutation."""
        if not files:
            return _StateBudgetReservation(changes=())

        proposed: dict[str, int] = {}
        for path, content in files:
            canonical = _canonical_state_backend_path(path, sandbox_enabled=self._sandbox_enabled)
            if canonical is None:
                raise ValueError("Deep-research state mutation must target /shared/")
            if canonical in proposed:
                raise ValueError(f"Deep-research state mutation contains duplicate path {canonical}")
            proposed[canonical] = self._mutation_size(content)

        with self._lock:
            prospective = {**self._sizes, **proposed}
            self._validate_totals(prospective)
            changes: list[tuple[str, int, int | None]] = []
            for path, size in proposed.items():
                generation = self._next_generation
                self._next_generation += 1
                previous = self._sizes.get(path)
                self._sizes[path] = size
                self._generations[path] = generation
                changes.append((path, generation, previous))
            return _StateBudgetReservation(changes=tuple(changes))

    def rollback(self, reservation: _StateBudgetReservation) -> None:
        """Release a failed reservation without clobbering a later mutation."""
        with self._lock:
            for path, generation, previous in reservation.changes:
                if self._generations.get(path) != generation:
                    continue
                if previous is None:
                    self._sizes.pop(path, None)
                    self._generations.pop(path, None)
                else:
                    self._sizes[path] = previous
                    self._generations[path] = self._next_generation
                    self._next_generation += 1


class DeepResearchResourceLimits(BaseModel):
    """Hard per-job limits enforced before expensive work or state mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_chars: int = Field(
        default=DEFAULT_MAX_RESEARCH_INPUT_CHARS,
        ge=1,
        le=DEFAULT_MAX_RESEARCH_INPUT_CHARS,
        description="Maximum aggregate query and clarification characters accepted in one deep-research request.",
    )
    max_execution_seconds: float = Field(
        default=DEFAULT_MAX_RESEARCH_EXECUTION_SECONDS,
        gt=0,
        le=DEFAULT_MAX_RESEARCH_EXECUTION_SECONDS,
        description="Wall-clock timeout for one deep-research graph execution.",
    )
    max_plan_bytes: int = Field(
        default=DEFAULT_MAX_RESEARCH_PLAN_BYTES,
        ge=1,
        le=DEFAULT_MAX_RESEARCH_PLAN_BYTES,
        description="Maximum serialized size of the structured research plan.",
    )
    max_source_routing_bytes: int = Field(
        default=DEFAULT_MAX_SOURCE_ROUTING_BYTES,
        ge=1,
        le=DEFAULT_MAX_SOURCE_ROUTING_BYTES,
        description="Maximum serialized size of the structured source-routing response.",
    )
    max_final_report_bytes: int = Field(
        default=DEFAULT_MAX_FINAL_REPORT_BYTES,
        ge=1,
        le=DEFAULT_MAX_FINAL_REPORT_BYTES,
        description="Maximum UTF-8 serialized size of the writer-owned final report.",
    )
    max_state_file_count: int = Field(
        default=DEFAULT_MAX_STATE_FILE_COUNT,
        ge=1,
        le=DEFAULT_MAX_STATE_FILE_COUNT,
        description="Maximum number of files held in the job's StateBackend filesystem.",
    )
    max_total_state_bytes: int = Field(
        default=DEFAULT_MAX_TOTAL_STATE_BYTES,
        ge=1,
        le=DEFAULT_MAX_TOTAL_STATE_BYTES,
        description="Maximum aggregate payload bytes held in the job's StateBackend filesystem.",
    )
    max_research_queries: int = Field(
        default=DEFAULT_MAX_RESEARCH_QUERIES,
        ge=1,
        le=DEFAULT_MAX_RESEARCH_QUERIES,
        description="Maximum researcher queries consumed across all batches in one job.",
    )
    max_total_query_chars: int = Field(
        default=DEFAULT_MAX_RESEARCH_QUERY_CHARS,
        ge=1,
        le=DEFAULT_MAX_RESEARCH_QUERY_CHARS,
        description="Maximum aggregate query and subquery characters across one job.",
    )
    max_research_note_bytes: int = Field(
        default=DEFAULT_MAX_RESEARCH_NOTE_BYTES,
        ge=1,
        le=DEFAULT_MAX_RESEARCH_NOTE_BYTES,
        description="Maximum serialized size of one ResearchNotes result.",
    )
    max_total_research_note_bytes: int = Field(
        default=DEFAULT_MAX_TOTAL_RESEARCH_NOTE_BYTES,
        ge=1,
        le=DEFAULT_MAX_TOTAL_RESEARCH_NOTE_BYTES,
        description="Maximum aggregate serialized ResearchNotes bytes in one job.",
    )
    max_source_tool_calls: int = Field(
        default=DEFAULT_MAX_SOURCE_TOOL_CALLS,
        ge=1,
        le=DEFAULT_MAX_SOURCE_TOOL_CALLS,
        description="Maximum Deep Researcher Agent source-tool attempts and concrete batch items in one job.",
    )
    max_consecutive_source_tool_failures: int = Field(
        default=DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES,
        ge=1,
        le=DEFAULT_MAX_CONSECUTIVE_SOURCE_TOOL_FAILURES,
        description="Consecutive failed source-tool results allowed before the per-job circuit opens.",
    )
    max_todo_items: int = Field(
        default=DEFAULT_MAX_TODO_ITEMS,
        ge=1,
        le=DEFAULT_MAX_TODO_ITEMS,
        description="Maximum top-level orchestrator todo items in one state replacement.",
    )
    max_todo_item_chars: int = Field(
        default=DEFAULT_MAX_TODO_ITEM_CHARS,
        ge=1,
        le=DEFAULT_MAX_TODO_ITEM_CHARS,
        description="Maximum characters in one top-level orchestrator todo item.",
    )
    max_total_todo_chars: int = Field(
        default=DEFAULT_MAX_TOTAL_TODO_CHARS,
        ge=1,
        le=DEFAULT_MAX_TOTAL_TODO_CHARS,
        description="Maximum aggregate todo content characters in one top-level state replacement.",
    )
