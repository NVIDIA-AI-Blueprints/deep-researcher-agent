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

"""DeepResearcher-owned async job access control helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from deep_researcher_agent.auth import Principal
from deep_researcher_agent.auth import get_current_principal

logger = logging.getLogger(__name__)

_job_access_schema_initialized: set[str] = set()

# Statuses that mean a job no longer holds a live sandbox. Anything else (running,
# pending, submitted, etc.) counts as active for the concurrency guard.
_TERMINAL_STATUS_SQL = "('success','failure','failed','interrupted','cancelled','completed','error')"

_JOB_ACCESS_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_job_access_owner ON job_access(owner_auth_type, owner_subject)"
_JOB_ACCESS_CONVERSATION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_job_access_conversation ON job_access(conversation_id)"
)
_JOB_ACCESS_SELECT_SQL = text(
    "SELECT job_id, owner_auth_type, owner_subject, owner_email, conversation_id, created_at "
    "FROM job_access WHERE job_id = :job_id"
)
_VALIDATE_JOB_ACCESS_SCHEMA_SQL = text(
    "SELECT job_id, owner_auth_type, owner_subject, owner_email, conversation_id, agent_type, "
    "submission_token, submission_expires_at, created_at FROM job_access WHERE 1 = 0"
)
# Most recent completed, non-expired report job for a conversation. The owner predicate is
# applied only when REQUIRE_AUTH=true, mirroring authorize_job_access (which skips ownership
# under REQUIRE_AUTH=false). This keeps the fallback consistent with the auth gate and avoids
# brittle owner-subject matching for the synthesized no-auth principal.
# Agent types whose successful output is a full report eligible for follow-up. Excludes
# non-report agents (e.g. shallow_researcher); legacy rows with NULL agent_type are allowed.
_REPORT_PRODUCING_AGENTS = ("deep_researcher", "report_rewriter")
_REPORT_AGENT_FILTER = (
    "AND (ja.agent_type IS NULL OR ja.agent_type IN (" + ", ".join(f"'{a}'" for a in _REPORT_PRODUCING_AGENTS) + ")) "
)
_LATEST_REPORT_JOB_BASE = (
    "SELECT ja.job_id FROM job_access ja "
    "JOIN job_info ji ON ja.job_id = ji.job_id "
    "WHERE ja.conversation_id = :conversation_id "
    "AND ji.status = 'success' "
    "AND ji.is_expired IS NOT TRUE " + _REPORT_AGENT_FILTER
)
_LATEST_REPORT_JOB_SQL_ANY = text(_LATEST_REPORT_JOB_BASE + "ORDER BY ji.created_at DESC LIMIT 1")
_LATEST_REPORT_JOB_SQL_OWNED = text(
    _LATEST_REPORT_JOB_BASE
    + "AND ja.owner_auth_type = :owner_auth_type AND ja.owner_subject = :owner_subject "
    + "ORDER BY ji.created_at DESC LIMIT 1"
)
_JOB_ACCESS_DELETE_SQL = text("DELETE FROM job_access WHERE job_id = :job_id")
_JOB_ACCESS_RESERVATION_DELETE_SQL = text(
    "DELETE FROM job_access WHERE job_id = :job_id AND submission_token = :submission_token"
)
_JOB_ACCESS_RESERVATION_RENEW_SQL = text(
    "UPDATE job_access SET submission_expires_at = :submission_expires_at "
    "WHERE job_id = :job_id AND submission_token = :submission_token"
)
_JOB_ACCESS_CLEANUP_SQL = text(
    "DELETE FROM job_access "
    "WHERE job_id NOT IN (SELECT job_id FROM job_info WHERE is_expired IS NOT TRUE) "
    "AND (submission_token IS NULL OR submission_expires_at <= :now)"
)
_JOB_INFO_DELETE_SQL = text("DELETE FROM job_info WHERE job_id = :job_id")
_JOB_EVENTS_DELETE_SQL = text("DELETE FROM job_events WHERE job_id = :job_id")


class JobAccessConflictError(RuntimeError):
    """A submission cannot reserve ownership for an already-used job ID."""


def _is_postgres(db_url: str) -> bool:
    """Return whether the database URL targets PostgreSQL."""
    return db_url.startswith("postgres")


def ensure_job_access_table(db_url: str) -> None:
    """Create the DeepResearcher-owned job access table if it does not exist."""
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        conn.commit()
    validate_job_access_table(db_url)


def validate_job_access_table(db_url: str) -> None:
    """Raise unless the access table exposes its full storage contract."""
    with _job_access_connection(db_url) as conn:
        conn.execute(_VALIDATE_JOB_ACCESS_SCHEMA_SQL)


def create_job_access(
    job_id: str,
    principal: Principal,
    db_url: str,
    conversation_id: str | None = None,
    agent_type: str | None = None,
    *,
    submission_token: str | None = None,
    submission_expires_at: float | None = None,
) -> None:
    """Persist owner metadata, optionally as an insert-only pre-enqueue reservation.

    Normal callers retain the historical upsert behavior. Submission code passes
    an opaque token and expiry, which makes ownership durable *before* handing a
    task to Dask without allowing a colliding caller to overwrite an existing
    job's owner.
    """
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        params = _principal_params(
            job_id,
            principal,
            conversation_id,
            agent_type,
            submission_token=submission_token,
            submission_expires_at=submission_expires_at,
        )
        try:
            if submission_token is None:
                conn.execute(_job_access_upsert_sql(db_url), params)
            else:
                if submission_expires_at is None:
                    raise ValueError("submission_expires_at is required with submission_token")
                # An expired, never-enqueued reservation is safe to reclaim. A
                # live/expired job_info row is never touched by this path.
                conn.execute(
                    text(
                        "DELETE FROM job_access "
                        "WHERE job_id = :job_id "
                        "AND submission_token IS NOT NULL "
                        "AND submission_expires_at <= :now "
                        "AND job_id NOT IN (SELECT job_id FROM job_info)"
                    ),
                    {"job_id": job_id, "now": time.time()},
                )
                if conn.execute(
                    text("SELECT 1 FROM job_info WHERE job_id = :job_id"),
                    {"job_id": job_id},
                ).first():
                    conn.rollback()
                    raise JobAccessConflictError(f"Job already exists: {job_id}")
                conn.execute(_job_access_reservation_insert_sql(), params)
            conn.commit()
        except IntegrityError as exc:
            conn.rollback()
            raise JobAccessConflictError(f"Job access already exists: {job_id}") from exc


def renew_job_access_reservation(
    job_id: str,
    submission_token: str,
    db_url: str,
    submission_expires_at: float,
) -> bool:
    """Extend an in-flight ownership reservation iff its opaque token matches."""
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        result = conn.execute(
            _JOB_ACCESS_RESERVATION_RENEW_SQL,
            {
                "job_id": job_id,
                "submission_token": submission_token,
                "submission_expires_at": submission_expires_at,
            },
        )
        conn.commit()
        return (result.rowcount or 0) == 1


def release_job_access_reservation(job_id: str, submission_token: str, db_url: str) -> bool:
    """Delete a failed pre-enqueue reservation iff its opaque token matches."""
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        result = conn.execute(
            _JOB_ACCESS_RESERVATION_DELETE_SQL,
            {"job_id": job_id, "submission_token": submission_token},
        )
        conn.commit()
        return (result.rowcount or 0) == 1


def get_latest_report_job_for_conversation(
    conversation_id: str | None, principal: Principal, db_url: str
) -> str | None:
    """Return the most recent completed report job submitted in this conversation by this caller.

    Used as the server-side default for report follow-up when the client does not supply an
    explicit ``active_report_job_id``. Returns None (degrade to fresh research) for an empty
    conversation id, no match, or any storage error — it must never raise into the request path.
    """
    if not conversation_id:
        return None
    enforce_owner = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"
    if enforce_owner and principal is None:
        return None
    params: dict[str, str] = {"conversation_id": conversation_id}
    if enforce_owner:
        sql = _LATEST_REPORT_JOB_SQL_OWNED
        params["owner_auth_type"] = principal.type
        params["owner_subject"] = principal.sub
    else:
        sql = _LATEST_REPORT_JOB_SQL_ANY
    try:
        with _job_access_connection(db_url) as conn:
            _ensure_job_access_schema(conn, db_url)
            row = conn.execute(sql, params).first()
            return row[0] if row else None
    except Exception as e:
        logger.debug("Conversation report-job lookup failed for %s: %s", conversation_id, type(e).__name__)
        return None


def get_job_access(job_id: str, db_url: str) -> dict[str, Any] | None:
    """Return job access metadata for a job."""
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        row = conn.execute(_JOB_ACCESS_SELECT_SQL, {"job_id": job_id}).mappings().first()
        return dict(row) if row is not None else None


def delete_job_access(job_id: str, db_url: str) -> int:
    """Delete job access metadata for a specific job."""
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        result = conn.execute(_JOB_ACCESS_DELETE_SQL, {"job_id": job_id})
        conn.commit()
        return result.rowcount or 0


def cleanup_job_access(db_url: str, conn: Connection | None = None) -> int:
    """Delete expired-job rows and expired pre-enqueue reservations."""
    params = {"now": time.time()}
    if conn is not None:
        _ensure_job_access_schema(conn, db_url)
        result = conn.execute(_JOB_ACCESS_CLEANUP_SQL, params)
        return result.rowcount or 0

    with _job_access_connection(db_url) as owned_conn:
        _ensure_job_access_schema(owned_conn, db_url)
        result = owned_conn.execute(_JOB_ACCESS_CLEANUP_SQL, params)
        owned_conn.commit()
        return result.rowcount or 0


def rollback_job_submission(job_id: str, db_url: str) -> None:
    """Best-effort rollback when ownership persistence fails after NAT job creation.

    The submit path must not return an ownerless job ID. If job submission creates
    NAT metadata but `job_access` cannot be written, remove the partial job state.
    """
    from .event_store import EventStore

    EventStore._ensure_table_exists(db_url)
    with _job_access_connection(db_url) as conn:
        _ensure_job_access_schema(conn, db_url)
        conn.execute(_JOB_ACCESS_DELETE_SQL, {"job_id": job_id})
        conn.execute(_JOB_EVENTS_DELETE_SQL, {"job_id": job_id})
        conn.execute(_JOB_INFO_DELETE_SQL, {"job_id": job_id})
        conn.commit()


def count_active_jobs_for_owner(db_url: str, principal: Principal) -> int | None:
    """Count an owner's non-terminal, non-expired jobs.

    Returns ``None`` if the count cannot be computed (e.g. the NAT ``job_info``
    schema differs); callers should fail open so a query mismatch never blocks
    legitimate submissions. Used by the submit-path sandbox concurrency guard.
    """
    try:
        with _job_access_connection(db_url) as conn:
            _ensure_job_access_schema(conn, db_url)
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM job_access ja JOIN job_info ji ON ja.job_id = ji.job_id "
                    "WHERE ja.owner_auth_type = :t AND ja.owner_subject = :s "
                    "AND (ji.is_expired IS NOT TRUE) "
                    f"AND lower(ji.status) NOT IN {_TERMINAL_STATUS_SQL}"
                ),
                {"t": principal.type, "s": principal.sub},
            ).scalar()
            return int(row or 0)
    except Exception as exc:  # noqa: BLE001 - guard must fail open, never block submits
        logger.warning("Could not count active jobs for owner; allowing submit: %s", exc)
        return None


def count_active_jobs_global(db_url: str) -> int | None:
    """Count all non-terminal, non-expired jobs (global capacity guard).

    Returns ``None`` on query failure so callers fail open.
    """
    try:
        with _job_access_connection(db_url) as conn:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM job_info "
                    f"WHERE (is_expired IS NOT TRUE) AND lower(status) NOT IN {_TERMINAL_STATUS_SQL}"
                )
            ).scalar()
            return int(row or 0)
    except Exception as exc:  # noqa: BLE001 - guard must fail open, never block submits
        logger.warning("Could not count active jobs globally; allowing submit: %s", exc)
        return None


def _make_no_auth_principal(owner: str | None = None) -> Principal:
    """Synthesize a principal for deployments with auth disabled (REQUIRE_AUTH=false).

    Uses the middleware caller type as the principal type.  When an owner
    identifier is provided it becomes the subject (useful for programmatic
    job submission); otherwise the caller type is used as a stable subject.
    """
    try:
        from deep_researcher_api.auth.middleware import get_current_user

        current_user = get_current_user()
    except Exception:
        current_user = {}

    principal_type = str(current_user.get("type") or "anonymous")
    subject = owner if owner else principal_type
    email = owner if owner and "@" in owner else None
    return Principal(type=principal_type, sub=subject, email=email)


def require_verified_principal() -> Principal:
    """Return the verified request principal or raise a safe auth error.

    When auth is disabled (REQUIRE_AUTH != true), synthesizes a principal
    from the middleware caller identity so no-auth deployments can still
    access async jobs.
    """
    principal = get_current_principal()
    if principal is not None:
        return principal

    if os.environ.get("REQUIRE_AUTH", "false").lower() == "true":
        raise HTTPException(403, "Verified principal required for async job access")

    return _make_no_auth_principal()


async def authorize_job_access(job_store: Any, db_url: str, job_id: str, principal: Principal) -> Any:
    """Load a job, enforcing ownership when auth is enabled.

    When REQUIRE_AUTH=false, ownership is not enforced — any caller may access
    any existing job.  Ownership records are still written at submit time for
    audit purposes and to support future auth enablement without data migration.
    """
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    if os.environ.get("REQUIRE_AUTH", "false").lower() != "true":
        return job

    loop = asyncio.get_running_loop()
    access = await loop.run_in_executor(None, get_job_access, job_id, db_url)
    if access is None:
        raise HTTPException(404, f"Job not found: {job_id}")

    if not _principal_matches_access(principal, access):
        raise HTTPException(404, f"Job not found: {job_id}")

    return job


def _principal_matches_access(principal: Principal, access: Mapping[str, Any]) -> bool:
    """Return whether a principal matches a job-access row's owner identity."""
    return principal.type == access.get("owner_auth_type") and principal.sub == access.get("owner_subject")


def _job_access_connection(db_url: str):
    """Open a sync connection on the shared event-store engine for the URL."""
    from .event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    return engine.connect()


def _ensure_job_access_schema(conn: Connection, db_url: str) -> None:
    """Create the ``job_access`` schema, caching only committed initialization.

    Most callers pass a fresh connection, so this helper owns and commits the
    DDL transaction before any job-level work begins. A caller may also pass a
    connection with an active transaction (for example, coordinated cleanup).
    In that case the caller retains transaction ownership and the URL is not
    cached: a later rollback may undo transactional DDL on PostgreSQL.
    """
    if db_url in _job_access_schema_initialized:
        return
    caller_owns_transaction = conn.in_transaction()
    conn.execute(text(_job_access_table_sql(db_url)))
    _ensure_extra_columns(conn, db_url)
    conn.execute(text(_JOB_ACCESS_INDEX_SQL))
    conn.execute(text(_JOB_ACCESS_CONVERSATION_INDEX_SQL))
    if not caller_owns_transaction:
        conn.commit()
        _job_access_schema_initialized.add(db_url)


def _ensure_extra_columns(conn: Connection, db_url: str) -> None:
    """Add optional metadata columns to a pre-existing job_access table.

    CREATE TABLE IF NOT EXISTS won't add columns to an existing table. Idempotent across upgrades:
    Postgres supports ADD COLUMN IF NOT EXISTS; SQLite does not, so check PRAGMA table_info first.
    Migration failure is fatal because the submission-token columns are part of
    the admission and ownership invariant; the API must not report ready with a
    partially upgraded table.
    """
    columns = (
        ("conversation_id", "VARCHAR"),
        ("agent_type", "VARCHAR"),
        ("submission_token", "VARCHAR"),
        ("submission_expires_at", "DOUBLE PRECISION"),
    )
    if _is_postgres(db_url):
        for col, column_type in columns:
            conn.execute(text(f"ALTER TABLE job_access ADD COLUMN IF NOT EXISTS {col} {column_type}"))
        return

    existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_access)")).fetchall()}
    for col, column_type in columns:
        if col not in existing:
            conn.execute(text(f"ALTER TABLE job_access ADD COLUMN {col} {column_type}"))


def _job_access_table_sql(db_url: str) -> str:
    """Return the ``CREATE TABLE`` SQL for ``job_access``, dialect-aware for the URL."""
    created_at_type = (
        "TIMESTAMP WITH TIME ZONE DEFAULT NOW()" if _is_postgres(db_url) else "DATETIME DEFAULT CURRENT_TIMESTAMP"
    )
    return (
        "CREATE TABLE IF NOT EXISTS job_access ("
        "  job_id VARCHAR PRIMARY KEY,"
        "  owner_auth_type VARCHAR NOT NULL,"
        "  owner_subject VARCHAR NOT NULL,"
        "  owner_email VARCHAR,"
        "  conversation_id VARCHAR,"
        "  agent_type VARCHAR,"
        "  submission_token VARCHAR,"
        "  submission_expires_at DOUBLE PRECISION,"
        f"  created_at {created_at_type}"
        ")"
    )


def _job_access_upsert_sql(db_url: str):
    """Return the dialect-appropriate upsert statement for ``job_access``."""
    cols = (
        "job_id, owner_auth_type, owner_subject, owner_email, conversation_id, agent_type, "
        "submission_token, submission_expires_at"
    )
    vals = (
        ":job_id, :owner_auth_type, :owner_subject, :owner_email, :conversation_id, :agent_type, "
        ":submission_token, :submission_expires_at"
    )
    postgres_upsert = (
        f"INSERT INTO job_access ({cols}) VALUES ({vals}) "
        "ON CONFLICT(job_id) DO UPDATE SET "
        "owner_auth_type = excluded.owner_auth_type, "
        "owner_subject = excluded.owner_subject, "
        "owner_email = excluded.owner_email, "
        "conversation_id = excluded.conversation_id, "
        "agent_type = excluded.agent_type, "
        "submission_token = NULL, "
        "submission_expires_at = NULL"
    )
    sqlite_upsert = f"INSERT OR REPLACE INTO job_access ({cols}) VALUES ({vals})"
    return text(postgres_upsert if _is_postgres(db_url) else sqlite_upsert)


def _job_access_reservation_insert_sql():
    """Return the insert-only statement used before a task is enqueued."""
    return text(
        "INSERT INTO job_access ("
        "job_id, owner_auth_type, owner_subject, owner_email, conversation_id, agent_type, "
        "submission_token, submission_expires_at"
        ") VALUES ("
        ":job_id, :owner_auth_type, :owner_subject, :owner_email, :conversation_id, :agent_type, "
        ":submission_token, :submission_expires_at"
        ")"
    )


def _principal_params(
    job_id: str,
    principal: Principal,
    conversation_id: str | None = None,
    agent_type: str | None = None,
    *,
    submission_token: str | None = None,
    submission_expires_at: float | None = None,
) -> dict[str, str | float | None]:
    """Return SQL bind params for a job's owner identity, conversation, and agent type."""
    return {
        "job_id": job_id,
        "owner_auth_type": principal.type,
        "owner_subject": principal.sub,
        "owner_email": principal.email,
        "conversation_id": conversation_id,
        "agent_type": agent_type,
        "submission_token": submission_token,
        "submission_expires_at": submission_expires_at,
    }
