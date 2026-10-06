# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres-backed job store for async MCP research queries.

One row is stored per submitted query. Jobs are partitioned by a generic
principal so cross-principal reads can be refused. The public no-auth runtime
uses the constant principal ``anonymous`` to preserve this schema, which means
the job UUID is the public server's bearer capability. This store is the shared
MCP submit/poll ledger used by replicas; it lives in the same Postgres database
as Deep Researcher Agent checkpointing but uses separate MCP-owned tables.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Literal

import asyncpg

from .db_url import normalize_postgres_url

JobState = Literal["queued", "running", "complete", "failed"]
JobDepth = Literal["shallow", "deep", "meta"]

_DEFAULT_TTL_SECONDS = 24 * 3600
_DEFAULT_MIN_POOL_SIZE = 1
_DEFAULT_MAX_POOL_SIZE = 5
_SCHEMA_LOCK_ID = 742190880248061112
# Preserve the reference component ID so an in-place upgrade does not create a
# second migration history for the same physical tables.
_MIGRATION_COMPONENT = "aiq_maas_mcp"
_BASE_MIGRATION_VERSION = 1
_POLL_COUNT_MIGRATION_VERSION = 2
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Job:
    job_id: str
    principal: str
    query: str
    depth: JobDepth
    state: JobState
    result: str | None
    error: str | None
    poll_count: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    runner_id: str | None = None
    heartbeat_at: datetime | None = None


class JobStore:
    """Async Postgres wrapper for the MCP jobs table."""

    def __init__(
        self,
        db_url: str,
        *,
        schema: str = "public",
        min_pool_size: int = _DEFAULT_MIN_POOL_SIZE,
        max_pool_size: int = _DEFAULT_MAX_POOL_SIZE,
    ) -> None:
        self._db_url = normalize_postgres_url(db_url, label="MCP job store URL")
        self._schema = schema
        self._schema_ident = _quote_identifier(schema)
        self._jobs_table = f"{self._schema_ident}.mcp_jobs"
        self._migrations_table = f"{self._schema_ident}.mcp_schema_migrations"
        # MCP runs each job with its UUID as the LangGraph ``thread_id``. Keep
        # these table names alongside the ledger names so expiry can remove the
        # complete MCP-owned thread, rather than leaving research state behind
        # after the public job capability has expired.
        self._checkpoints_table = f"{self._schema_ident}.checkpoints"
        self._checkpoint_blobs_table = f"{self._schema_ident}.checkpoint_blobs"
        self._checkpoint_writes_table = f"{self._schema_ident}.checkpoint_writes"
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._db_url,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _SCHEMA_LOCK_ID)
                await self._ensure_schema(conn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool | None:
        """The store's connection pool (``None`` until :meth:`init`).

        Exposed so the checkpoint todo reader can borrow this warm pool instead
        of opening a second one against the same database.
        """
        return self._pool

    async def create(
        self,
        *,
        principal: str,
        query: str,
        depth: JobDepth,
        state: JobState,
        result: str | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds)
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._jobs_table} (
                job_id, principal, query, depth, state, result, error,
                created_at, updated_at, expires_at
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, NULL, $7, $7, $8)
            """,
            job_id,
            principal,
            query,
            depth,
            state,
            result,
            now,
            expires_at,
        )
        return job_id

    async def mark_running(self, job_id: str, runner_id: str) -> bool:
        pool = self._require_pool()
        status = await pool.execute(
            f"""
            UPDATE {self._jobs_table}
               SET state = 'running',
                   runner_id = $2,
                   heartbeat_at = NOW(),
                   updated_at = NOW()
              WHERE job_id = $1::uuid
                AND state = 'queued'
            """,
            job_id,
            runner_id,
        )
        return _rows_changed(status) == 1

    async def mark_failed_if_queued_or_owned(self, job_id: str, runner_id: str, error: str) -> bool:
        """Fail a queued job or a running job owned by ``runner_id``.

        A connection failure during :meth:`mark_running` can be ambiguous: the
        transition may not have happened, or PostgreSQL may have committed it
        before the acknowledgement was lost. Match both safe states in one
        statement without overwriting terminal work or another runner's job.
        """
        pool = self._require_pool()
        status = await pool.execute(
            f"""
            UPDATE {self._jobs_table}
               SET state = 'failed',
                   error = $3,
                   updated_at = NOW()
             WHERE job_id = $1::uuid
               AND (
                   state = 'queued'
                   OR (state = 'running' AND runner_id = $2)
               )
            """,
            job_id,
            runner_id,
            error,
        )
        return _rows_changed(status) == 1

    async def heartbeat(self, job_id: str, runner_id: str) -> None:
        pool = self._require_pool()
        await pool.execute(
            f"""
            UPDATE {self._jobs_table}
               SET heartbeat_at = NOW(),
                   updated_at = NOW()
             WHERE job_id = $1::uuid
               AND runner_id = $2
               AND state = 'running'
            """,
            job_id,
            runner_id,
        )

    async def update(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        result: str | None = None,
        error: str | None = None,
        from_states: tuple[JobState, ...] | None = None,
        runner_id: str | None = None,
    ) -> bool:
        assignments = ["updated_at = NOW()"]
        vals: list[object] = [job_id]
        predicates = ["job_id = $1::uuid"]
        if state is not None:
            vals.append(state)
            assignments.append(f"state = ${len(vals)}")
        if result is not None:
            vals.append(result)
            assignments.append(f"result = ${len(vals)}")
        if error is not None:
            vals.append(error)
            assignments.append(f"error = ${len(vals)}")
        if from_states is not None:
            vals.append(list(from_states))
            predicates.append(f"state = ANY(${len(vals)}::text[])")
        if runner_id is not None:
            vals.append(runner_id)
            predicates.append(f"runner_id = ${len(vals)}")

        pool = self._require_pool()
        status = await pool.execute(
            f"UPDATE {self._jobs_table} SET {', '.join(assignments)} WHERE {' AND '.join(predicates)}",
            *vals,
        )
        return _rows_changed(status) == 1

    async def get(self, job_id: str) -> Job | None:
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT job_id::text AS job_id, principal, query, depth, state, result, error,
                   poll_count, created_at, updated_at, expires_at, runner_id, heartbeat_at
              FROM {self._jobs_table}
             WHERE job_id = $1::uuid
            """,
            job_id,
        )
        if row is None:
            return None
        return _row_to_job(row)

    async def record_poll(self, job_id: str, principal: str) -> Job | None:
        """Increment poll_count for an active job and return the latest row.

        Polling must not refresh updated_at/heartbeat_at; those timestamps drive
        stale-job reconciliation, so changing them here would let client polling
        keep an orphaned job alive indefinitely.
        """
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            UPDATE {self._jobs_table}
               SET poll_count = poll_count + 1
             WHERE job_id = $1::uuid
               AND principal = $2
               AND state IN ('queued', 'running')
            RETURNING job_id::text AS job_id, principal, query, depth, state, result, error,
                      poll_count, created_at, updated_at, expires_at, runner_id, heartbeat_at
            """,
            job_id,
            principal,
        )
        if row is not None:
            return _row_to_job(row)
        # Don't fall back to get(): it has no principal filter and would expose terminal
        # jobs owned by a different principal. Repeat the SELECT with AND principal = $2.
        row = await pool.fetchrow(
            f"""
            SELECT job_id::text AS job_id, principal, query, depth, state, result, error,
                   poll_count, created_at, updated_at, expires_at, runner_id, heartbeat_at
              FROM {self._jobs_table}
             WHERE job_id = $1::uuid
               AND principal = $2
            """,
            job_id,
            principal,
        )
        return _row_to_job(row) if row is not None else None

    async def delete_expired(self) -> int:
        """Delete expired MCP jobs and their LangGraph checkpoint threads.

        A workflow may outlive its fixed job TTL, so only terminal ledger rows
        are eligible. The ledger job ID is the workflow's checkpoint
        ``thread_id``. Select eligible rows under lock before removing their
        checkpoint writes, blobs, and snapshots, then delete the ledger rows in
        the same transaction. ``SKIP LOCKED`` makes concurrent sweepers on
        separate MCP replicas divide expired jobs without blocking one another.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    f"""
                    SELECT job_id::text AS job_id
                      FROM {self._jobs_table}
                     WHERE expires_at < NOW()
                       AND state IN ('complete', 'failed')
                     FOR UPDATE SKIP LOCKED
                    """
                )
                job_ids = [row["job_id"] for row in rows]
                if not job_ids:
                    return 0

                for table in (
                    self._checkpoint_writes_table,
                    self._checkpoint_blobs_table,
                    self._checkpoints_table,
                ):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ANY($1::text[])",
                        job_ids,
                    )

                status = await conn.execute(
                    f"DELETE FROM {self._jobs_table} WHERE job_id = ANY($1::uuid[])",
                    job_ids,
                )
                return _rows_changed(status)

    async def mark_stale_running_failed(self, *, stale_after_seconds: int, error: str) -> int:
        pool = self._require_pool()
        status = await pool.execute(
            f"""
            UPDATE {self._jobs_table}
               SET state = 'failed',
                   error = $2,
                   updated_at = NOW()
             WHERE state IN ('queued', 'running')
               AND COALESCE(heartbeat_at, updated_at) < NOW() - ($1::double precision * INTERVAL '1 second')
            """,
            stale_after_seconds,
            error,
        )
        return _rows_changed(status)

    async def _ensure_schema(self, conn: asyncpg.Connection) -> None:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema_ident}")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._migrations_table} (
                component TEXT NOT NULL,
                version INTEGER NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (component, version)
            )
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._jobs_table} (
                job_id UUID PRIMARY KEY,
                principal TEXT NOT NULL,
                query TEXT NOT NULL,
                depth TEXT NOT NULL CHECK (depth IN ('shallow', 'deep', 'meta')),
                state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'complete', 'failed')),
                result TEXT,
                error TEXT,
                poll_count INTEGER NOT NULL DEFAULT 0,
                runner_id TEXT,
                heartbeat_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        await conn.execute(
            f"ALTER TABLE {self._jobs_table} ADD COLUMN IF NOT EXISTS poll_count INTEGER NOT NULL DEFAULT 0"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_mcp_jobs_principal_job_id ON {self._jobs_table}(principal, job_id)"
        )
        await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_mcp_jobs_expires_at ON {self._jobs_table}(expires_at)")
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_mcp_jobs_state_updated_at ON {self._jobs_table}(state, updated_at)"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_mcp_jobs_runner_state ON {self._jobs_table}(runner_id, state)"
        )
        await conn.execute(
            f"""
            INSERT INTO {self._migrations_table} (component, version)
            VALUES ($1, $2)
            ON CONFLICT (component, version) DO NOTHING
            """,
            _MIGRATION_COMPONENT,
            _BASE_MIGRATION_VERSION,
        )
        await conn.execute(
            f"""
            INSERT INTO {self._migrations_table} (component, version)
            VALUES ($1, $2)
            ON CONFLICT (component, version) DO NOTHING
            """,
            _MIGRATION_COMPONENT,
            _POLL_COUNT_MIGRATION_VERSION,
        )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("JobStore.init() must be called before use")
        return self._pool


def _row_to_job(row: asyncpg.Record) -> Job:
    return Job(
        job_id=row["job_id"],
        principal=row["principal"],
        query=row["query"],
        depth=row["depth"],
        state=row["state"],
        result=row["result"],
        error=row["error"],
        poll_count=row["poll_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        runner_id=row["runner_id"],
        heartbeat_at=row["heartbeat_at"],
    )


def _rows_changed(status: str) -> int:
    try:
        return int(status.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return 0


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid Postgres identifier: {value!r}")
    return f'"{value}"'
