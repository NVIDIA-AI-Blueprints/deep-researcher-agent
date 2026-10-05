# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local analytical artifacts and lifecycle management."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import tempfile
from contextvars import ContextVar
from contextvars import Token
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisRunState:
    """Mutable resources that belong to exactly one DS Agent request."""

    run_id: str
    temporary_directory: tempfile.TemporaryDirectory[str]
    root: Path
    manifest_path: Path
    structured_results: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    force_finalization: bool = False
    finalization_instruction: str | None = None


_CURRENT_ANALYSIS_RUN: ContextVar[AnalysisRunState | None] = ContextVar(
    "current_data_science_analysis_run",
    default=None,
)


def _write_manifest(state: AnalysisRunState) -> bool:
    payload = {"version": 1, "results": state.structured_results}
    staging = state.manifest_path.with_suffix(".tmp")
    try:
        staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(state.manifest_path)
    except OSError:
        logger.exception("Failed to persist the request-local structured-data manifest")
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove an incomplete request-local structured-data manifest")
        return False
    return True


def begin_analysis_run() -> Token[AnalysisRunState | None]:
    """Create and install one isolated analytical runtime for this async request."""

    temporary_directory = tempfile.TemporaryDirectory(prefix="aiq-ds-analysis-", ignore_cleanup_errors=True)
    root = Path(temporary_directory.name)
    manifest_path = root / "structured-results.json"
    state = AnalysisRunState(
        run_id=str(uuid4()),
        temporary_directory=temporary_directory,
        root=root,
        manifest_path=manifest_path,
    )
    _write_manifest(state)
    return _CURRENT_ANALYSIS_RUN.set(state)


def get_analysis_run() -> AnalysisRunState | None:
    """Return the current request's analytical runtime, if one is active."""

    return _CURRENT_ANALYSIS_RUN.get()


def register_structured_result(
    *,
    provider: str,
    tool_name: str,
    question: str,
    database_name: str | None,
    payload: dict[str, Any],
) -> str | None:
    """Persist one successful structured SQL response and return its stable reference."""

    state = get_analysis_run()
    if state is None or payload.get("status") == "error" or not isinstance(payload.get("rows"), list):
        return None

    reference = f"structured_{len(state.structured_results) + 1}"
    result_path = state.root / f"{reference}.json"
    try:
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        result_path.write_text(serialized, encoding="utf-8")
    except (OSError, TypeError, ValueError):
        logger.exception("Failed to persist a request-local structured-data result")
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove an incomplete request-local structured-data result")
        return None
    columns = payload.get("columns")
    column_names = [
        str(column.get("name"))
        for column in columns or []
        if isinstance(column, dict) and column.get("name") is not None
    ]
    if not column_names and payload["rows"] and isinstance(payload["rows"][0], dict):
        column_names = [str(name) for name in payload["rows"][0]]
    result = {
        "ref": reference,
        "provider": provider,
        "tool_name": tool_name,
        "question": question,
        "database_name": database_name,
        "request_id": payload.get("request_id"),
        "row_count": len(payload["rows"]),
        "columns": column_names,
        "truncated": bool(payload.get("truncated", False)),
        "path": str(result_path),
    }
    state.structured_results.append(result)
    if not _write_manifest(state):
        state.structured_results.remove(result)
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove an unregistered request-local structured-data result")
        return None
    return reference


async def end_analysis_run(token: Token[AnalysisRunState | None]) -> None:
    """Close request-owned resources, remove artifacts, and restore the prior context."""

    state = get_analysis_run()
    try:
        if state is not None:
            cancellation: BaseException | None = None
            try:
                for resource in reversed(list(state.resources.values())):
                    closer = getattr(resource, "aclose", None) or getattr(resource, "close", None)
                    if closer is None:
                        continue
                    try:
                        result = closer()
                        if inspect.isawaitable(result):
                            await result
                    except BaseException as exc:  # cleanup must continue through cancellation
                        logger.exception("Failed to close a request-local analysis resource")
                        if isinstance(exc, asyncio.CancelledError):
                            cancellation = cancellation or exc
            finally:
                try:
                    await asyncio.to_thread(state.temporary_directory.cleanup)
                except BaseException as exc:  # cleanup must not replace the agent outcome
                    logger.exception("Failed to remove request-local analysis artifacts")
                    if isinstance(exc, asyncio.CancelledError):
                        cancellation = cancellation or exc
            if cancellation is not None:
                raise cancellation
    finally:
        _CURRENT_ANALYSIS_RUN.reset(token)


__all__ = [
    "AnalysisRunState",
    "begin_analysis_run",
    "end_analysis_run",
    "get_analysis_run",
    "register_structured_result",
]
