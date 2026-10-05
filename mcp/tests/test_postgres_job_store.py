# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Postgres-backed MCP job store tests."""

from __future__ import annotations

import asyncio
import os
import re
import uuid
import warnings
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest

from deep_researcher_agent.agents.chat_researcher.models import WorkflowSuccess
from deep_researcher_mcp.db_url import normalize_postgres_url
from deep_researcher_mcp.db_url import require_test_database_url
from deep_researcher_mcp.job_store import JobStore
from deep_researcher_mcp.jobs import JobManager

_DB_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _Runner:
    def __init__(
        self,
        *,
        depth: str = "shallow",
        gate: asyncio.Event | None = None,
        result: str = "research answer",
    ) -> None:
        self.depth = depth
        self.gate = gate
        self.result = result
        self.run_calls: list[tuple[str, str]] = []

    async def classify(self, query: str) -> dict[str, Any]:
        del query
        return {
            "user_intent": {"intent": "research"},
            "depth_decision": {"decision": self.depth},
        }

    async def run_query(self, query: str, *, conversation_id: str, depth: str | None = None) -> WorkflowSuccess:
        self.run_calls.append((query, conversation_id))
        if self.gate is not None:
            await self.gate.wait()
        return WorkflowSuccess(result=self.result)


@pytest.fixture()
async def postgres_url() -> str:
    db_url = os.getenv("DEEP_RESEARCHER_MCP_TEST_DB_URL")
    if not db_url:
        pytest.skip("set DEEP_RESEARCHER_MCP_TEST_DB_URL to run Postgres MCP job store tests")
    try:
        await _ensure_database(db_url)
        await _reset_schema(db_url)
    except (OSError, asyncpg.PostgresError) as exc:
        message = f"local Postgres test database is not available ({type(exc).__name__})"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        pytest.skip(message)

    yield db_url

    try:
        await _reset_schema(db_url)
    except (OSError, asyncpg.PostgresError):
        pass


@pytest.mark.asyncio
async def test_schema_init_is_idempotent_and_concurrent(postgres_url: str) -> None:
    stores = [JobStore(postgres_url, min_pool_size=1, max_pool_size=1) for _ in range(4)]
    try:
        await asyncio.gather(*(store.init() for store in stores))
        await asyncio.gather(*(store.init() for store in stores))

        conn = await asyncpg.connect(postgres_url)
        try:
            assert await conn.fetchval("SELECT to_regclass('public.mcp_jobs')") == "mcp_jobs"
            assert await conn.fetchval("SELECT to_regclass('public.mcp_schema_migrations')") == "mcp_schema_migrations"
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'mcp_jobs' AND column_name = 'poll_count'"
                )
                == 1
            )
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name = 'mcp_jobs'
                 ORDER BY ordinal_position
                """
            )
            assert [tuple(row) for row in columns] == [
                ("job_id", "uuid", "NO"),
                ("principal", "text", "NO"),
                ("query", "text", "NO"),
                ("depth", "text", "NO"),
                ("state", "text", "NO"),
                ("result", "text", "YES"),
                ("error", "text", "YES"),
                ("poll_count", "integer", "NO"),
                ("runner_id", "text", "YES"),
                ("heartbeat_at", "timestamp with time zone", "YES"),
                ("created_at", "timestamp with time zone", "NO"),
                ("updated_at", "timestamp with time zone", "NO"),
                ("expires_at", "timestamp with time zone", "NO"),
            ]
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM public.mcp_schema_migrations WHERE component = 'aiq_maas_mcp' AND version = 1"
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM public.mcp_schema_migrations WHERE component = 'aiq_maas_mcp' AND version = 2"
                )
                == 1
            )

            anonymous_job_id = await stores[0].create(
                principal="anonymous",
                query="capability query",
                depth="shallow",
                state="queued",
            )
            assert str(uuid.UUID(anonymous_job_id)) == anonymous_job_id
            assert uuid.UUID(anonymous_job_id).version == 4
            assert (
                await conn.fetchval(
                    "SELECT principal FROM public.mcp_jobs WHERE job_id = $1::uuid",
                    anonymous_job_id,
                )
                == "anonymous"
            )
            lifetime = await conn.fetchrow(
                "SELECT created_at, expires_at FROM public.mcp_jobs WHERE job_id = $1::uuid",
                anonymous_job_id,
            )
            assert lifetime is not None
            assert lifetime["expires_at"] - lifetime["created_at"] == timedelta(seconds=86_400)
        finally:
            await conn.close()
    finally:
        await asyncio.gather(*(store.close() for store in stores))


@pytest.mark.asyncio
async def test_job_manager_can_submit_on_one_instance_and_poll_from_another(postgres_url: str) -> None:
    gate = asyncio.Event()
    manager_a = JobManager(
        _Runner(gate=gate, result="eventual answer"),
        JobStore(postgres_url),
        runner_id="pod-a",
        heartbeat_interval_seconds=0.05,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )
    manager_b = JobManager(
        _Runner(),
        JobStore(postgres_url),
        runner_id="pod-b",
        heartbeat_interval_seconds=0.05,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )

    await manager_a.start()
    await manager_b.start()
    try:
        submitted = await manager_a.submit("query", "principal-a")

        polled = await manager_b.poll(submitted["job_id"], "principal-a")
        assert polled["state"] in {"queued", "running"}

        gate.set()
        completed = await manager_a.wait_for_completion(submitted["job_id"], "principal-a", timeout=2)

        assert completed == {
            "job_id": submitted["job_id"],
            "depth": "shallow",
            "state": "complete",
            "result": "eventual answer",
        }
        completed_status = await manager_b.poll(submitted["job_id"], "principal-a")
        assert completed_status == {
            "job_id": submitted["job_id"],
            "depth": "shallow",
            "state": "complete",
            "todos": [],
        }
        assert "result" not in completed_status
        assert await manager_b.get_final_report(submitted["job_id"], "principal-a") == completed
    finally:
        await manager_a.stop()
        await manager_b.stop()


@pytest.mark.asyncio
async def test_deep_poll_count_persists_across_instances(postgres_url: str) -> None:
    gate = asyncio.Event()
    manager_a = JobManager(
        _Runner(depth="deep", gate=gate),
        JobStore(postgres_url),
        runner_id="pod-a",
        heartbeat_interval_seconds=0.05,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )
    manager_b = JobManager(
        _Runner(depth="deep"),
        JobStore(postgres_url),
        runner_id="pod-b",
        heartbeat_interval_seconds=0.05,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )

    await manager_a.start()
    await manager_b.start()
    try:
        submitted = await manager_a.submit("query", "principal-a")

        cadences = []
        for _ in range(10):
            polled = await manager_b.poll(submitted["job_id"], "principal-a")
            assert polled["depth"] == "deep"
            assert polled["state"] in {"queued", "running"}
            cadences.append(polled["next_poll_after_seconds"])

        assert cadences == [180] * 10
        assert sum(cadences) == 30 * 60
        conn = await asyncpg.connect(postgres_url)
        try:
            assert (
                await conn.fetchval(
                    "SELECT poll_count FROM public.mcp_jobs WHERE job_id = $1::uuid",
                    submitted["job_id"],
                )
                == 10
            )
        finally:
            await conn.close()
    finally:
        gate.set()
        await manager_a.stop()
        await manager_b.stop()


@pytest.mark.asyncio
async def test_poll_count_and_timestamps_preserve_stale_reconciliation_semantics(postgres_url: str) -> None:
    store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    await store.init()
    try:
        job_id = await store.create(
            principal="principal-a",
            query="query",
            depth="deep",
            state="queued",
        )
        await store.mark_running(job_id, "runner-a")
        before_poll = await store.get(job_id)
        assert before_poll is not None
        assert before_poll.heartbeat_at is not None

        await asyncio.sleep(0.02)
        wrong_principal = await store.record_poll(job_id, "principal-b")
        assert wrong_principal is None
        after_poll = await store.record_poll(job_id, "principal-a")
        assert after_poll is not None
        assert after_poll.poll_count == 1
        assert after_poll.updated_at == before_poll.updated_at
        assert after_poll.heartbeat_at == before_poll.heartbeat_at

        await asyncio.sleep(0.02)
        await store.heartbeat(job_id, "wrong-runner")
        wrong_runner = await store.get(job_id)
        assert wrong_runner is not None
        assert wrong_runner.updated_at == before_poll.updated_at
        assert wrong_runner.heartbeat_at == before_poll.heartbeat_at

        await store.heartbeat(job_id, "runner-a")
        after_heartbeat = await store.get(job_id)
        assert after_heartbeat is not None
        assert after_heartbeat.updated_at > before_poll.updated_at
        assert after_heartbeat.heartbeat_at > before_poll.heartbeat_at

        await store.update(job_id, state="complete", result="answer")
        terminal_poll = await store.record_poll(job_id, "principal-a")
        assert terminal_poll is not None
        assert terminal_poll.poll_count == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_record_poll_does_not_expose_terminal_job_to_wrong_principal(postgres_url: str) -> None:
    store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    await store.init()
    try:
        job_id = await store.create(
            principal="principal-a",
            query="q",
            depth="shallow",
            state="complete",
            result="answer",
        )
        result = await store.record_poll(job_id, "principal-b")
        assert result is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_state_transitions_are_guarded_by_state_and_runner(postgres_url: str) -> None:
    store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    await store.init()
    try:
        job_id = await store.create(
            principal="principal-a",
            query="query",
            depth="shallow",
            state="queued",
        )

        assert await store.mark_running(job_id, "runner-a") is True
        running = await store.get(job_id)
        assert running is not None
        assert running.state == "running"
        assert running.runner_id == "runner-a"

        assert await store.mark_running(job_id, "runner-b") is False
        still_running = await store.get(job_id)
        assert still_running is not None
        assert still_running.state == "running"
        assert still_running.runner_id == "runner-a"

        assert (
            await store.update(
                job_id,
                state="failed",
                error="wrong runner",
                from_states=("running",),
                runner_id="runner-b",
            )
            is False
        )
        assert (
            await store.update(
                job_id,
                state="complete",
                result="wrong state",
                from_states=("queued",),
                runner_id="runner-a",
            )
            is False
        )

        before_complete = await store.get(job_id)
        assert before_complete is not None
        assert before_complete.state == "running"
        assert before_complete.error is None
        assert before_complete.result is None

        assert (
            await store.update(
                job_id,
                state="complete",
                result="answer",
                from_states=("running",),
                runner_id="runner-a",
            )
            is True
        )
        complete = await store.get(job_id)
        assert complete is not None
        assert complete.state == "complete"
        assert complete.result == "answer"

        assert (
            await store.update(
                job_id,
                state="failed",
                error="late failure",
                from_states=("running",),
                runner_id="runner-a",
            )
            is False
        )
        terminal = await store.get(job_id)
        assert terminal is not None
        assert terminal.state == "complete"
        assert terminal.result == "answer"
        assert terminal.error is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failure_transition_updates_only_queued_or_owned_running_jobs(postgres_url: str) -> None:
    store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    await store.init()
    try:
        queued_id = await store.create(
            principal="principal-a",
            query="queued",
            depth="shallow",
            state="queued",
        )
        owned_id = await store.create(
            principal="principal-a",
            query="owned",
            depth="shallow",
            state="queued",
        )
        foreign_id = await store.create(
            principal="principal-a",
            query="foreign",
            depth="shallow",
            state="queued",
        )
        complete_id = await store.create(
            principal="principal-a",
            query="complete",
            depth="shallow",
            state="complete",
            result="answer",
        )
        failed_id = await store.create(
            principal="principal-a",
            query="failed",
            depth="shallow",
            state="failed",
        )

        assert await store.mark_running(owned_id, "runner-a") is True
        assert await store.mark_running(foreign_id, "runner-b") is True

        assert await store.mark_failed_if_queued_or_owned(queued_id, "runner-a", "claim failed") is True
        assert await store.mark_failed_if_queued_or_owned(owned_id, "runner-a", "run failed") is True
        assert await store.mark_failed_if_queued_or_owned(foreign_id, "runner-a", "wrong owner") is False
        assert await store.mark_failed_if_queued_or_owned(complete_id, "runner-a", "late failure") is False
        assert await store.mark_failed_if_queued_or_owned(failed_id, "runner-a", "replacement failure") is False

        queued = await store.get(queued_id)
        owned = await store.get(owned_id)
        foreign = await store.get(foreign_id)
        complete = await store.get(complete_id)
        failed = await store.get(failed_id)

        assert queued is not None and (queued.state, queued.error, queued.runner_id) == (
            "failed",
            "claim failed",
            None,
        )
        assert owned is not None and (owned.state, owned.error, owned.runner_id) == (
            "failed",
            "run failed",
            "runner-a",
        )
        assert foreign is not None and (foreign.state, foreign.error, foreign.runner_id) == (
            "running",
            None,
            "runner-b",
        )
        assert complete is not None and (complete.state, complete.result, complete.error) == (
            "complete",
            "answer",
            None,
        )
        assert failed is not None and (failed.state, failed.error) == ("failed", None)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_shutdown_cancellation_is_persisted_and_visible_after_restart(postgres_url: str) -> None:
    gate = asyncio.Event()
    manager = JobManager(
        _Runner(gate=gate),
        JobStore(postgres_url, min_pool_size=1, max_pool_size=1),
        runner_id="stopping-pod",
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )
    await manager.start()
    submitted = await manager.submit("query", "anonymous")
    job_id = submitted["job_id"]
    for _ in range(100):
        job = await manager._store.get(job_id)
        if job is not None and job.state == "running":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("background job did not enter running state")

    await manager.stop()

    restarted_store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    await restarted_store.init()
    try:
        persisted = await restarted_store.get(job_id)
        assert persisted is not None
        assert persisted.state == "failed"
        assert persisted.error == "Research task was cancelled before completion."
    finally:
        await restarted_store.close()


@pytest.mark.asyncio
async def test_expired_job_deletion_becomes_stable_not_found(postgres_url: str) -> None:
    store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    manager = JobManager(
        _Runner(),
        store,
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )
    await manager.start()
    try:
        job_id = await store.create(
            principal="anonymous",
            query="query",
            depth="shallow",
            state="complete",
            result="answer",
            ttl_seconds=-1,
        )
        await _ensure_checkpoint_tables(postgres_url)
        unrelated_thread_id = str(uuid.uuid4())
        await _seed_checkpoint_thread(postgres_url, job_id)
        await _seed_checkpoint_thread(postgres_url, unrelated_thread_id)

        assert await store.delete_expired() == 1
        assert await manager.poll(job_id, "anonymous") == {
            "state": "not_found",
            "error": "job_not_found",
        }
        assert await manager.get_final_report(job_id, "anonymous") == {
            "state": "not_found",
            "error": "job_not_found",
        }
        assert await _checkpoint_row_counts(postgres_url, job_id) == (0, 0, 0)
        assert await _checkpoint_row_counts(postgres_url, unrelated_thread_id) == (1, 1, 1)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_expired_running_job_is_preserved_until_terminal(postgres_url: str) -> None:
    store = JobStore(postgres_url, min_pool_size=1, max_pool_size=1)
    await store.init()
    try:
        job_id = await store.create(
            principal="anonymous",
            query="query",
            depth="deep",
            state="running",
            ttl_seconds=-1,
        )
        await _seed_checkpoint_thread(postgres_url, job_id)

        assert await store.delete_expired() == 0
        job = await store.get(job_id)
        assert job is not None
        assert job.state == "running"
        assert await _checkpoint_row_counts(postgres_url, job_id) == (1, 1, 1)

        writer = await asyncpg.connect(postgres_url)
        try:
            await writer.execute(
                """
                INSERT INTO public.checkpoints (
                    thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata
                )
                VALUES ($1, '', 'checkpoint-2', '{}', '{}')
                """,
                job_id,
            )
        finally:
            await writer.close()

        assert await store.update(job_id, state="complete", from_states=("running",)) is True
        assert await store.delete_expired() == 1
        assert await store.get(job_id) is None
        assert await _checkpoint_row_counts(postgres_url, job_id) == (0, 0, 0)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_expired_job_cleanup_is_idempotent_across_stores(postgres_url: str) -> None:
    stores = [JobStore(postgres_url, min_pool_size=1, max_pool_size=1) for _ in range(2)]
    await asyncio.gather(*(store.init() for store in stores))
    try:
        job_id = await stores[0].create(
            principal="anonymous",
            query="query",
            depth="shallow",
            state="complete",
            result="answer",
            ttl_seconds=-1,
        )
        await _seed_checkpoint_thread(postgres_url, job_id)

        deleted = await asyncio.gather(*(store.delete_expired() for store in stores))

        assert sorted(deleted) == [0, 1]
        assert await stores[0].get(job_id) is None
        assert await _checkpoint_row_counts(postgres_url, job_id) == (0, 0, 0)
    finally:
        await asyncio.gather(*(store.close() for store in stores))


@pytest.mark.asyncio
async def test_stale_jobs_and_expired_jobs_are_cleaned_up(postgres_url: str) -> None:
    store = JobStore(postgres_url)
    await store.init()
    try:
        old = datetime.now(UTC) - timedelta(hours=2)
        fresh = datetime.now(UTC)
        stale_running_id = str(uuid.uuid4())
        stale_queued_id = str(uuid.uuid4())
        fresh_running_id = str(uuid.uuid4())
        expired_id = str(uuid.uuid4())

        conn = await asyncpg.connect(postgres_url)
        try:
            await conn.executemany(
                """
                INSERT INTO public.mcp_jobs (
                    job_id, principal, query, depth, state, result, error,
                    runner_id, heartbeat_at, created_at, updated_at, expires_at
                )
                VALUES ($1::uuid, 'principal-a', 'query', 'shallow', $2, NULL, NULL,
                        $3, $4, $5, $6, $7)
                """,
                [
                    (stale_running_id, "running", "dead-pod", old, old, old, fresh + timedelta(hours=1)),
                    (stale_queued_id, "queued", None, None, old, old, fresh + timedelta(hours=1)),
                    (fresh_running_id, "running", "live-pod", fresh, fresh, fresh, fresh + timedelta(hours=1)),
                    (expired_id, "complete", None, None, old, old, old),
                ],
            )
        finally:
            await conn.close()

        assert await store.mark_stale_running_failed(stale_after_seconds=60, error="stale") == 2
        assert (await store.get(stale_running_id)).state == "failed"
        assert (await store.get(stale_queued_id)).state == "failed"
        assert (await store.get(fresh_running_id)).state == "running"

        assert await store.delete_expired() == 1
        assert await store.get(expired_id) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_periodic_reconciler_reaps_stale_running_job(postgres_url: str) -> None:
    """The background reconciler must fail a stale `running` job without a restart.

    Regression guard for the startup-only reconciliation bug: a job whose owning
    pod died (heartbeat gone silent) must be marked `failed` by the periodic
    sweep loop, so a polling client stops seeing `running` forever.
    """
    manager = JobManager(
        _Runner(),
        JobStore(postgres_url),
        runner_id="live-pod",
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0.05,
        stale_job_after_seconds=60,
    )
    await manager.start()
    try:
        # Seed a dead-pod job directly: running, last heartbeat 2h ago.
        old = datetime.now(UTC) - timedelta(hours=2)
        stale_id = str(uuid.uuid4())
        conn = await asyncpg.connect(postgres_url)
        try:
            await conn.execute(
                """
                INSERT INTO public.mcp_jobs (
                    job_id, principal, query, depth, state, result, error,
                    runner_id, heartbeat_at, created_at, updated_at, expires_at
                )
                VALUES ($1::uuid, 'principal-a', 'q', 'deep', 'running', NULL, NULL,
                        'dead-pod', $2, $2, $2, $3)
                """,
                stale_id,
                old,
                datetime.now(UTC) + timedelta(hours=1),
            )
        finally:
            await conn.close()

        # Wait for the periodic reconciler to fire.
        for _ in range(40):
            await asyncio.sleep(0.05)
            polled = await manager.poll(stale_id, "principal-a")
            if polled["state"] == "failed":
                break

        polled = await manager.poll(stale_id, "principal-a")
        assert polled["state"] == "failed"
        assert "interrupted" in polled["error"].lower()
        assert polled["todos"] == []
        report = await manager.get_final_report(stale_id, "principal-a")
        assert report["state"] == "failed"
        assert "interrupted" in report["error"].lower()
        assert "result" not in report
    finally:
        await manager.stop()


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


async def _reset_schema(db_url: str) -> None:
    db_url = require_test_database_url(db_url, label="DEEP_RESEARCHER_MCP_TEST_DB_URL")
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("DROP TABLE IF EXISTS public.mcp_jobs")
        await conn.execute("DROP TABLE IF EXISTS public.mcp_schema_migrations")
    finally:
        await conn.close()
    await _ensure_checkpoint_tables(db_url)


async def _ensure_checkpoint_tables(db_url: str) -> None:
    """Create the production LangGraph tables needed by the expiry test."""
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public.checkpoints (
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
            CREATE TABLE IF NOT EXISTS public.checkpoint_blobs (
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
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public.checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                blob BYTEA NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
            """
        )
    finally:
        await conn.close()


async def _seed_checkpoint_thread(db_url: str, thread_id: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            """
            INSERT INTO public.checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
            VALUES ($1, '', 'checkpoint-1', '{}', '{}')
            """,
            thread_id,
        )
        await conn.execute(
            """
            INSERT INTO public.checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
            VALUES ($1, '', 'channel', 'version-1', 'bytes', $2)
            """,
            thread_id,
            b"checkpoint blob",
        )
        await conn.execute(
            """
            INSERT INTO public.checkpoint_writes (
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob
            )
            VALUES ($1, '', 'checkpoint-1', 'task-1', 0, 'channel', 'bytes', $2)
            """,
            thread_id,
            b"checkpoint write",
        )
    finally:
        await conn.close()


async def _checkpoint_row_counts(db_url: str, thread_id: str) -> tuple[int, int, int]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM public.checkpoints WHERE thread_id = $1) AS checkpoints,
                (SELECT COUNT(*) FROM public.checkpoint_blobs WHERE thread_id = $1) AS blobs,
                (SELECT COUNT(*) FROM public.checkpoint_writes WHERE thread_id = $1) AS writes
            """,
            thread_id,
        )
        assert rows is not None
        return rows["checkpoints"], rows["blobs"], rows["writes"]
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
