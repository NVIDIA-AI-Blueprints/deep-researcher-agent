# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint todo reader tests."""

from __future__ import annotations

import os
import re
import uuid
import warnings
from types import MethodType
from types import SimpleNamespace
from typing import Any
from typing import TypedDict
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import msgpack
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from deep_researcher_agent.common.logging_utils import log_identifier_ref
from deep_researcher_mcp import checkpoint_todos as checkpoint_todos_module
from deep_researcher_mcp.checkpoint_todos import CheckpointTodoReader
from deep_researcher_mcp.checkpoint_todos import TodoItem
from deep_researcher_mcp.checkpoint_todos import _decode_todo_blob
from deep_researcher_mcp.checkpoint_todos import decode_todos_value
from deep_researcher_mcp.checkpoint_todos import normalize_todos
from deep_researcher_mcp.db_url import normalize_postgres_url
from deep_researcher_mcp.db_url import require_test_database_url

_DB_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@pytest.fixture()
async def postgres_url() -> str:
    db_url = os.getenv("DEEP_RESEARCHER_MCP_TEST_DB_URL")
    if not db_url:
        pytest.skip("set DEEP_RESEARCHER_MCP_TEST_DB_URL to run Postgres checkpoint todo tests")
    try:
        await _ensure_database(db_url)
        await _reset_checkpoint_tables(db_url)
    except (OSError, asyncpg.PostgresError) as exc:
        message = f"local Postgres test database is not available ({type(exc).__name__})"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        pytest.skip(message)

    yield db_url

    try:
        await _reset_checkpoint_tables(db_url)
    except (OSError, asyncpg.PostgresError):
        pass


def test_decode_todo_blob_accepts_list_of_dicts() -> None:
    todos = [{"content": "Plan", "status": "in_progress"}]

    assert _decode_todo_blob("msgpack", msgpack.packb(todos, use_bin_type=True)) == todos


def test_decode_todo_blob_normalizes_displayable_items() -> None:
    payload = [{"title": "Plan", "status": "todo", "id": "t-1"}, "not-a-dict", {"content": "Draft"}]

    assert _decode_todo_blob("msgpack", msgpack.packb(payload, use_bin_type=True)) == [
        {"content": "Plan", "status": "pending", "id": "t-1"},
        {"content": "Draft", "status": "pending"},
    ]


def test_decode_todo_blob_rejects_non_list_payload() -> None:
    assert _decode_todo_blob("msgpack", msgpack.packb({"content": "Plan"}, use_bin_type=True)) == []


def test_decode_todo_blob_rejects_invalid_msgpack() -> None:
    assert _decode_todo_blob("msgpack", b"not-msgpack") == []


def test_decode_todo_blob_rejects_unsupported_blob_type() -> None:
    todos = [{"content": "Plan", "status": "in_progress"}]

    assert _decode_todo_blob("json", msgpack.packb(todos, use_bin_type=True)) == []


def test_decode_todos_value_supports_json_fallback() -> None:
    blob = b'[{"description": "Write report", "status": "done"}]'

    assert decode_todos_value(None, blob) == [TodoItem(content="Write report", status="completed")]


def test_decode_todos_value_with_no_blob_is_empty() -> None:
    assert decode_todos_value("msgpack", None) == []


def test_decode_todos_value_accepts_target_langgraph_serializer() -> None:
    payload = [{"content": "Research sources", "status": "in_progress"}]
    blob_type, blob = JsonPlusSerializer().dumps_typed(payload)

    assert decode_todos_value(blob_type, blob) == [
        TodoItem(content="Research sources", status="in_progress"),
    ]


def test_normalize_todos_preserves_unknown_status() -> None:
    assert normalize_todos([{"content": "Blocked item", "status": "blocked"}]) == [
        TodoItem(content="Blocked item", status="blocked")
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, "pending"),
        ("", "pending"),
        ("pending", "pending"),
        ("in_progress", "in_progress"),
        ("completed", "completed"),
        ("todo", "pending"),
        ("OPEN", "pending"),
        ("Doing", "in_progress"),
        ("active", "in_progress"),
        ("DONE", "completed"),
        ("complete", "completed"),
        ("blocked", "blocked"),
        (7, "pending"),
    ],
)
def test_normalize_todos_exact_status_alias_contract(status: Any, expected: str) -> None:
    assert normalize_todos([{"content": "Task", "status": status}]) == [
        TodoItem(content="Task", status=expected),
    ]


def test_normalize_todos_content_priority_id_and_drop_contract() -> None:
    payload = [
        {
            "content": "  preferred content  ",
            "title": "title",
            "text": "text",
            "description": "description",
            "status": "done",
            "id": "",
        },
        {"content": " ", "title": "Title fallback", "id": 123},
        {"content": None, "title": "", "text": "Text fallback"},
        {"description": "Description fallback"},
        {"content": " ", "title": None, "text": 42, "description": ""},
        "not-an-object",
    ]

    assert normalize_todos(payload) == [
        TodoItem(content="  preferred content  ", status="completed", id=""),
        TodoItem(content="Title fallback", status="pending"),
        TodoItem(content="Text fallback", status="pending"),
        TodoItem(content="Description fallback", status="pending"),
    ]


def test_normalize_todos_accepts_tuple_and_attribute_objects() -> None:
    payload = (
        SimpleNamespace(title="Object task", status=False, id="object-1"),
        {"text": "Mapping task", "status": "active"},
    )

    assert normalize_todos(payload) == [
        TodoItem(content="Object task", status="pending", id="object-1"),
        TodoItem(content="Mapping task", status="in_progress"),
    ]


def test_checkpoint_warning_is_rate_limited_per_error_class(
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    reader = CheckpointTodoReader("postgresql://localhost/db")
    times = iter((100.0, 101.0, 102.0, 131.0))
    monkeypatch.setattr(checkpoint_todos_module.time, "monotonic", lambda: next(times))
    caplog.set_level("WARNING", logger="deep_researcher_mcp.checkpoint_todos")

    reader._warn_rate_limited("timeout", "first")
    reader._warn_rate_limited("timeout", "suppressed")
    reader._warn_rate_limited("database", "different class")
    reader._warn_rate_limited("timeout", "after window")

    assert [record.getMessage() for record in caplog.records] == [
        "Checkpoint todo read failed (timeout): first",
        "Checkpoint todo read failed (database): different class",
        "Checkpoint todo read failed (timeout): after window",
    ]


def test_checkpoint_reader_rejects_invalid_schema_identifier() -> None:
    with pytest.raises(ValueError, match="Invalid Postgres identifier"):
        CheckpointTodoReader("postgresql://localhost/db", schema="public;drop table checkpoints")


def test_checkpoint_reader_builds_quoted_table_names() -> None:
    reader = CheckpointTodoReader("postgresql://localhost/db", schema="public")

    assert reader._checkpoints_table == '"public".checkpoints'
    assert reader._checkpoint_blobs_table == '"public".checkpoint_blobs'


@pytest.mark.asyncio
async def test_checkpoint_reader_returns_latest_top_level_todos(postgres_url: str) -> None:
    reader = CheckpointTodoReader(postgres_url)
    await reader.start()
    thread_id = str(uuid.uuid4())
    top_ns = "deep_research:top"
    nested_ns = "deep_research:top|tools:nested"
    older_todos = [{"content": "Plan", "status": "pending"}]
    latest_todos = [{"content": "Plan", "status": "completed"}]
    nested_todos = [{"content": "Nested", "status": "in_progress"}]

    conn = await asyncpg.connect(postgres_url)
    try:
        await conn.execute(
            """
            INSERT INTO public.checkpoints (
                thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
            )
            VALUES
              ($1, $2, '00000000-0000-0000-0000-000000000001', NULL, NULL,
               '{"channel_versions":{"todos":"00000000000000000000000000000001.0.1"}}'::jsonb, '{}'::jsonb),
              ($1, $2, '00000000-0000-0000-0000-000000000002', NULL, NULL,
               '{"channel_versions":{"todos":"00000000000000000000000000000002.0.1"}}'::jsonb, '{}'::jsonb),
              ($1, $3, '00000000-0000-0000-0000-000000000003', NULL, NULL,
               '{"channel_versions":{"todos":"00000000000000000000000000000003.0.1"}}'::jsonb, '{}'::jsonb)
            """,
            thread_id,
            top_ns,
            nested_ns,
        )
        await conn.executemany(
            """
            INSERT INTO public.checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
            VALUES ($1, $2, 'todos', $3, 'msgpack', $4)
            """,
            [
                (
                    thread_id,
                    top_ns,
                    "00000000000000000000000000000001.0.1",
                    msgpack.packb(older_todos, use_bin_type=True),
                ),
                (
                    thread_id,
                    top_ns,
                    "00000000000000000000000000000002.0.1",
                    msgpack.packb(latest_todos, use_bin_type=True),
                ),
                (
                    thread_id,
                    nested_ns,
                    "00000000000000000000000000000003.0.1",
                    msgpack.packb(nested_todos, use_bin_type=True),
                ),
            ],
        )

        assert await reader.get_todos(thread_id) == latest_todos
    finally:
        await conn.close()
        await reader.close()


@pytest.mark.asyncio
async def test_checkpoint_reader_timeout_returns_empty(caplog) -> None:
    reader = CheckpointTodoReader("postgresql://localhost/db")

    async def timing_out_get_todos(self, thread_id: str) -> list[dict[str, str]]:
        # asyncpg's per-query timeout raises asyncio.TimeoutError (== builtin
        # TimeoutError on 3.11+) once the statement is cancelled server-side.
        del self, thread_id
        raise TimeoutError

    reader._get_todos = MethodType(timing_out_get_todos, reader)  # type: ignore[method-assign]
    caplog.set_level("WARNING", logger="deep_researcher_mcp.checkpoint_todos")

    assert await reader.get_todos("job-1") == []
    assert "Checkpoint todo read failed (timeout)" in caplog.text


@pytest.mark.asyncio
async def test_checkpoint_reader_error_does_not_log_thread_capability(caplog) -> None:
    reader = CheckpointTodoReader("postgresql://localhost/db")
    capability_id = str(uuid.uuid4())

    async def failing_get_todos(self, thread_id: str) -> list[dict[str, str]]:
        del self
        raise RuntimeError(f"database rejected thread {thread_id}")

    reader._get_todos = MethodType(failing_get_todos, reader)  # type: ignore[method-assign]
    caplog.set_level("WARNING", logger="deep_researcher_mcp.checkpoint_todos")

    assert await reader.get_todos(capability_id) == []
    assert capability_id not in caplog.text
    assert log_identifier_ref(capability_id) in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_get_todos_applies_per_query_timeout() -> None:
    """The read must bound itself with asyncpg's native query timeout (so the
    statement is cancelled on its own connection), not an outer wait_for."""
    reader = CheckpointTodoReader("postgresql://localhost/db")
    captured: dict[str, object] = {}

    class _FakePool:
        async def fetchrow(self, query: str, *args, timeout=None):
            # No row -> get_todos should short-circuit to [] (implicit None return).
            captured["query"] = query
            captured["args"] = args
            captured["timeout"] = timeout

    reader._pool = _FakePool()  # type: ignore[assignment]

    assert await reader.get_todos("thread-1") == []
    assert captured["timeout"] == 1.0
    assert captured["args"] == ("thread-1",)


def test_is_sortable_checkpoint_id_accepts_uuid_rejects_other() -> None:
    from deep_researcher_mcp.checkpoint_todos import _is_sortable_checkpoint_id

    assert _is_sortable_checkpoint_id("1ef9c8e0-1234-6abc-9def-0123456789ab")
    assert _is_sortable_checkpoint_id("00000000-0000-0000-0000-000000000001")
    assert not _is_sortable_checkpoint_id("not-a-uuid")
    assert not _is_sortable_checkpoint_id("0000000000000000")
    assert not _is_sortable_checkpoint_id(None)
    assert not _is_sortable_checkpoint_id(12345)


@pytest.mark.asyncio
async def test_target_langgraph_inherits_deep_research_todo_checkpoint_namespace() -> None:
    class _InnerState(TypedDict):
        todos: list[dict[str, str]]

    inner_builder = StateGraph(_InnerState)

    async def write_todos(state: _InnerState) -> dict[str, list[dict[str, str]]]:
        del state
        return {"todos": [{"content": "Plan sections", "status": "in_progress"}]}

    inner_builder.add_node("write_todos", write_todos)
    inner_builder.add_edge(START, "write_todos")
    inner_builder.add_edge("write_todos", END)
    inner_graph = inner_builder.compile()

    class _OuterState(TypedDict):
        result: str

    outer_builder = StateGraph(_OuterState)

    async def deep_research(state: _OuterState) -> dict[str, str]:
        del state
        await inner_graph.ainvoke({"todos": []})
        return {"result": "done"}

    outer_builder.add_node("deep_research", deep_research)
    outer_builder.add_edge(START, "deep_research")
    outer_builder.add_edge("deep_research", END)
    checkpointer = InMemorySaver()
    outer_graph = outer_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "job-123"}}

    await outer_graph.ainvoke({"result": ""}, config=config)

    nested_todo_checkpoints = [
        item
        for item in checkpointer.list(config)
        if item.config["configurable"].get("checkpoint_ns", "").startswith("deep_research:")
        and "|" not in item.config["configurable"]["checkpoint_ns"]
        and "todos" in item.checkpoint["channel_values"]
    ]
    assert nested_todo_checkpoints


@pytest.mark.asyncio
async def test_checkpoint_reader_fails_soft_on_non_sortable_checkpoint_id(postgres_url: str, caplog) -> None:
    reader = CheckpointTodoReader(postgres_url)
    await reader.start()
    thread_id = str(uuid.uuid4())
    todos = [{"content": "Plan", "status": "pending"}]

    conn = await asyncpg.connect(postgres_url)
    try:
        await conn.execute(
            """
            INSERT INTO public.checkpoints (
                thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
            )
            VALUES ($1, 'deep_research:top', 'not-a-sortable-id', NULL, NULL,
                    '{"channel_versions":{"todos":"v1"}}'::jsonb, '{}'::jsonb)
            """,
            thread_id,
        )
        await conn.execute(
            """
            INSERT INTO public.checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
            VALUES ($1, 'deep_research:top', 'todos', 'v1', 'msgpack', $2)
            """,
            thread_id,
            msgpack.packb(todos, use_bin_type=True),
        )
        caplog.set_level("WARNING", logger="deep_researcher_mcp.checkpoint_todos")

        assert await reader.get_todos(thread_id) == []
        assert "unexpected checkpoint_id" in caplog.text
    finally:
        await conn.close()
        await reader.close()


async def _ensure_database(db_url: str) -> None:
    db_url = require_test_database_url(db_url, label="DEEP_RESEARCHER_MCP_TEST_DB_URL")
    maintenance_url, db_name = _maintenance_url(db_url)
    conn = await asyncpg.connect(maintenance_url)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if not exists:
            await conn.execute(f"CREATE DATABASE {_quote_database_name(db_name)}")
    finally:
        await conn.close()


async def _reset_checkpoint_tables(db_url: str) -> None:
    db_url = require_test_database_url(db_url, label="DEEP_RESEARCHER_MCP_TEST_DB_URL")
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("DROP TABLE IF EXISTS public.checkpoint_blobs")
        await conn.execute("DROP TABLE IF EXISTS public.checkpoints")
        await conn.execute(
            """
            CREATE TABLE public.checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE public.checkpoint_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BYTEA,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            )
            """
        )
    finally:
        await conn.close()


def _maintenance_url(db_url: str) -> tuple[str, str]:
    parts = urlsplit(normalize_postgres_url(db_url, label="DEEP_RESEARCHER_MCP_TEST_DB_URL"))
    db_name = parts.path.lstrip("/") or "postgres"
    maintenance = urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))
    return maintenance, db_name


def _quote_database_name(db_name: str) -> str:
    if _DB_NAME_RE.fullmatch(db_name):
        return f'"{db_name}"'
    return '"' + db_name.replace('"', '""') + '"'
