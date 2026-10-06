# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Deep Researcher Agent MCP todo progress from LangGraph checkpoint storage."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import asyncpg
import msgpack

from deep_researcher_agent.common.logging_utils import log_identifier_ref

from .db_url import normalize_postgres_url

logger = logging.getLogger(__name__)

_DEFAULT_MIN_POOL_SIZE = 1
_DEFAULT_MAX_POOL_SIZE = 5
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The current chat workflow invokes DeepAgents inside the outer graph's
# deep_research node. With the inner graph's default checkpointer=None,
# LangGraph inherits the outer checkpointer and writes a
# deep_research:<task-id> namespace. Deeper subagents add pipe-delimited
# segments, which the query below deliberately excludes.
# Canonical 8-4-4-4-12 hex UUID. LangGraph mints checkpoint_ids as time-sortable
# UUIDv6, which is why ``ORDER BY checkpoint_id DESC`` returns the latest
# checkpoint. The column is plain TEXT with no enforcement, so we validate the
# returned id against this shape and fail soft if the upstream scheme ever
# changes (see ``_get_todos``).
_CHECKPOINT_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_READ_TIMEOUT_SECONDS = 1.0
_LOG_RATE_LIMIT_SECONDS = 30.0
_CANONICAL_STATUSES = {"pending", "in_progress", "completed"}
_STATUS_ALIASES = {
    "todo": "pending",
    "open": "pending",
    "doing": "in_progress",
    "active": "in_progress",
    "done": "completed",
    "complete": "completed",
}
_CONTENT_KEYS = ("content", "title", "text", "description")


@dataclass(frozen=True)
class TodoItem:
    content: str
    status: str
    id: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {"content": self.content, "status": self.status}
        if self.id is not None:
            data["id"] = self.id
        return data


class CheckpointTodoReader:
    """Read latest top-level deep-research todos for an MCP job thread."""

    def __init__(
        self,
        db_url: str,
        *,
        schema: str = "public",
        min_pool_size: int = _DEFAULT_MIN_POOL_SIZE,
        max_pool_size: int = _DEFAULT_MAX_POOL_SIZE,
    ) -> None:
        schema_ident = _quote_identifier(schema)
        self._db_url = normalize_postgres_url(db_url, label="checkpoint todo DB URL")
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._checkpoints_table = f"{schema_ident}.checkpoints"
        self._checkpoint_blobs_table = f"{schema_ident}.checkpoint_blobs"
        self._pool: asyncpg.Pool | None = None
        self._owns_pool = False
        self._last_log: dict[str, float] = {}

    def bind_pool(self, pool: asyncpg.Pool) -> None:
        """Borrow an externally-owned pool instead of opening a second one.

        The job store and this reader hit the same database, so the JobManager
        wires the store's warm pool in here at startup. A borrowed pool is not
        closed by :meth:`close` — its owner is responsible for that.
        """
        self._pool = pool
        self._owns_pool = False

    async def start(self) -> None:
        # Only stand up our own pool if one was not already bound (see
        # ``bind_pool``). When borrowing the store's pool, this is a no-op.
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._db_url,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
            self._owns_pool = True

    async def close(self) -> None:
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
        self._pool = None
        self._owns_pool = False

    async def get_todos(self, thread_id: str) -> list[dict[str, str]]:
        """Return latest top-level deep-research todos for ``thread_id``."""
        try:
            return await self._get_todos(thread_id)
        except TimeoutError:
            # asyncpg raises asyncio.TimeoutError (== builtin TimeoutError on
            # 3.11+) when the per-query ``timeout`` elapses; it cancels the
            # statement on the same connection and leaves it reusable.
            self._warn_rate_limited("timeout", f"read exceeded {_READ_TIMEOUT_SECONDS}s")
            return []
        except Exception as exc:  # noqa: BLE001 - polling progress must remain fail-soft
            self._warn_rate_limited(
                type(exc).__name__,
                f"database read failed for thread_ref={log_identifier_ref(thread_id)}",
            )
            return []

    async def _get_todos(self, thread_id: str) -> list[dict[str, str]]:
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            WITH latest AS (
                SELECT DISTINCT ON (thread_id, checkpoint_ns)
                       thread_id,
                       checkpoint_ns,
                       checkpoint_id,
                       checkpoint->'channel_versions'->>'todos' AS todos_version
                  FROM {self._checkpoints_table}
                 WHERE thread_id = $1
                   AND checkpoint->'channel_versions'->>'todos' IS NOT NULL
                   AND checkpoint_ns LIKE 'deep_research:%'
                   AND position('|' in checkpoint_ns) = 0
                 ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
            )
            SELECT b.type, b.blob, latest.checkpoint_id
              FROM latest
              JOIN {self._checkpoint_blobs_table} b
                ON b.thread_id = latest.thread_id
               AND b.checkpoint_ns = latest.checkpoint_ns
               AND b.channel = 'todos'
               AND b.version = latest.todos_version
             ORDER BY latest.checkpoint_id DESC
             LIMIT 1
            """,
            thread_id,
            timeout=_READ_TIMEOUT_SECONDS,
        )
        if row is None:
            return []
        checkpoint_id = row["checkpoint_id"]
        if not _is_sortable_checkpoint_id(checkpoint_id):
            # "Latest" relies on checkpoint_id sorting chronologically (LangGraph
            # UUIDv6). If the upstream id scheme ever changes, DESC could surface
            # the wrong checkpoint — fail soft rather than report stale progress.
            self._warn_rate_limited(
                "checkpoint_id_format",
                f"unexpected checkpoint_id {checkpoint_id!r}; todo ordering assumes time-sortable UUIDs",
            )
            return []
        return [todo.to_dict() for todo in decode_todos_value(row["type"], row["blob"])]

    def _warn_rate_limited(self, error_class: str, message: str) -> None:
        now = time.monotonic()
        last = self._last_log.get(error_class, 0.0)
        if now - last < _LOG_RATE_LIMIT_SECONDS:
            return
        self._last_log[error_class] = now
        logger.warning("Checkpoint todo read failed (%s): %s", error_class, message)

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("CheckpointTodoReader.start() must be called before use")
        return self._pool


def decode_todos_value(blob_type: str | None, blob: bytes | None) -> list[TodoItem]:
    if blob is None:
        return []
    raw: Any | None = None
    if blob_type == "msgpack":
        try:
            raw = msgpack.unpackb(blob, raw=False)
        except Exception:
            raw = None

    if raw is None:
        try:
            raw = json.loads(blob)
        except Exception:
            logger.debug("Unable to decode checkpoint todos blob", exc_info=True)
            return []

    return normalize_todos(raw)


def normalize_todos(raw: Any) -> list[TodoItem]:
    """Normalize checkpoint todo objects to a stable MCP-facing shape."""
    if not isinstance(raw, (list, tuple)):
        return []

    todos: list[TodoItem] = []
    for index, item in enumerate(raw):
        content = _displayable_content(item)
        if content is None:
            logger.debug("Dropping todo item with no displayable content at index %d", index)
            continue

        item_id = _field_value(item, "id")
        todos.append(
            TodoItem(
                content=content,
                status=_normalize_status(_field_value(item, "status")),
                id=item_id if isinstance(item_id, str) else None,
            )
        )
    return todos


def _decode_todo_blob(blob_type: str | None, blob: bytes | None) -> list[dict[str, str]]:
    """Compatibility wrapper used by older tests."""
    return [todo.to_dict() for todo in decode_todos_value(blob_type, blob)]


def _normalize_status(status: Any) -> str:
    if not isinstance(status, str) or not status:
        return "pending"
    if status in _CANONICAL_STATUSES:
        return status
    canonical = _STATUS_ALIASES.get(status.lower())
    if canonical is not None:
        return canonical
    logger.debug("Unknown todo status encountered: %s", status)
    return status


def _displayable_content(item: Any) -> str | None:
    for key in _CONTENT_KEYS:
        value = _field_value(item, key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _field_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _is_sortable_checkpoint_id(value: Any) -> bool:
    """True if ``value`` is a canonical UUID, the shape our DESC ordering assumes."""
    return isinstance(value, str) and _CHECKPOINT_ID_RE.fullmatch(value) is not None


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid Postgres identifier: {value!r}")
    return f'"{value}"'
