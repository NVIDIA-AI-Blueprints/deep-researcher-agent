# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic admission control for asynchronous deep-research jobs.

Every deep-research submission passes through :func:`reserve_deep_research_job`
before it is handed to Dask.  A durable reservation makes concurrent
submissions visible before NAT creates ``job_info`` and Deep Researcher Agent creates
``job_access``.  Admission is serialized with a transaction-scoped advisory
lock on PostgreSQL and ``BEGIN IMMEDIATE`` on SQLite.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_INPUT_CHARS
from deep_researcher_agent.auth import Principal

from .access import _TERMINAL_STATUS_SQL
from .access import _ensure_job_access_schema
from .event_store import EventStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_CHARS = DEFAULT_MAX_RESEARCH_INPUT_CHARS
DEFAULT_MAX_ACTIVE_PER_PRINCIPAL = 5
DEFAULT_MAX_ACTIVE_GLOBAL = 50
DEFAULT_MAX_SUBMISSIONS_PER_MINUTE = 20
DEFAULT_RESERVATION_TTL_SECONDS = 30

_RATE_WINDOW_SECONDS = 60
_PG_ADMISSION_LOCK_ID = 0x41495141_444D4954  # "AIQADMIT"
_schema_lock = threading.Lock()
_schema_initialized: set[str] = set()

_CREATE_ADMISSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS deep_research_admission (
    job_id VARCHAR PRIMARY KEY,
    reservation_token VARCHAR NOT NULL,
    owner_auth_type VARCHAR NOT NULL,
    owner_subject VARCHAR NOT NULL,
    admitted_at DOUBLE PRECISION NOT NULL,
    reservation_expires_at DOUBLE PRECISION NOT NULL
)
"""
_CREATE_OWNER_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_deep_research_admission_owner
ON deep_research_admission(owner_auth_type, owner_subject, admitted_at)
"""
_DELETE_RESERVATION_SQL = text(
    "DELETE FROM deep_research_admission WHERE job_id = :job_id AND reservation_token = :reservation_token"
)
_RENEW_RESERVATION_SQL = text(
    "UPDATE deep_research_admission SET reservation_expires_at = :reservation_expires_at "
    "WHERE job_id = :job_id AND reservation_token = :reservation_token"
)
_INSERT_RESERVATION_SQL = text(
    "INSERT INTO deep_research_admission "
    "(job_id, reservation_token, owner_auth_type, owner_subject, admitted_at, reservation_expires_at) "
    "VALUES ("
    ":job_id, :reservation_token, :owner_auth_type, :owner_subject, :admitted_at, :reservation_expires_at"
    ")"
)
_VALIDATE_ADMISSION_SCHEMA_SQL = text(
    "SELECT job_id, reservation_token, owner_auth_type, owner_subject, admitted_at, reservation_expires_at "
    "FROM deep_research_admission WHERE 1 = 0"
)
_CLEANUP_RESERVATIONS_SQL = text(
    "DELETE FROM deep_research_admission "
    "WHERE admitted_at < :rate_cutoff "
    "AND ("
    "  (reservation_expires_at <= :now AND job_id NOT IN (SELECT job_id FROM job_info)) "
    "  OR job_id IN ("
    "    SELECT job_id FROM job_info "
    "    WHERE is_expired IS TRUE OR lower(status) IN " + _TERMINAL_STATUS_SQL + ""
    "  )"
    ")"
)
_COUNT_RECENT_SQL = text(
    "SELECT COUNT(*) FROM deep_research_admission "
    "WHERE owner_auth_type = :owner_auth_type "
    "AND owner_subject = :owner_subject "
    "AND admitted_at >= :rate_cutoff"
)
_COUNT_ACTIVE_FOR_OWNER_SQL = text(
    "SELECT COUNT(*) FROM ("
    "  SELECT dra.job_id "
    "  FROM deep_research_admission dra "
    "  LEFT JOIN job_info ji ON ji.job_id = dra.job_id "
    "  WHERE dra.owner_auth_type = :owner_auth_type "
    "    AND dra.owner_subject = :owner_subject "
    "    AND ("
    "      (ji.job_id IS NULL AND dra.reservation_expires_at > :now) "
    "      OR ("
    "        ji.job_id IS NOT NULL "
    "        AND ji.is_expired IS NOT TRUE "
    "        AND lower(ji.status) NOT IN " + _TERMINAL_STATUS_SQL + ""
    "      )"
    "    ) "
    "  UNION "
    "  SELECT ja.job_id "
    "  FROM job_access ja "
    "  JOIN job_info ji ON ji.job_id = ja.job_id "
    "  WHERE ja.owner_auth_type = :owner_auth_type "
    "    AND ja.owner_subject = :owner_subject "
    "    AND (ja.agent_type IS NULL OR ja.agent_type = 'deep_researcher') "
    "    AND ji.is_expired IS NOT TRUE "
    "    AND lower(ji.status) NOT IN " + _TERMINAL_STATUS_SQL + ""
    ") active_jobs"
)
_COUNT_ACTIVE_GLOBAL_SQL = text(
    "SELECT COUNT(*) FROM ("
    "  SELECT dra.job_id "
    "  FROM deep_research_admission dra "
    "  LEFT JOIN job_info ji ON ji.job_id = dra.job_id "
    "  WHERE (ji.job_id IS NULL AND dra.reservation_expires_at > :now) "
    "     OR ("
    "       ji.job_id IS NOT NULL "
    "       AND ji.is_expired IS NOT TRUE "
    "       AND lower(ji.status) NOT IN " + _TERMINAL_STATUS_SQL + ""
    "     ) "
    "  UNION "
    "  SELECT ji.job_id "
    "  FROM job_info ji "
    "  LEFT JOIN job_access ja ON ja.job_id = ji.job_id "
    "  WHERE (ja.job_id IS NULL OR ja.agent_type IS NULL OR ja.agent_type = 'deep_researcher') "
    "    AND ji.is_expired IS NOT TRUE "
    "    AND lower(ji.status) NOT IN " + _TERMINAL_STATUS_SQL + ""
    ") active_jobs"
)


class JobAdmissionError(RuntimeError):
    """Base class for safe, transport-neutral admission failures."""

    status_code = 503
    public_message = "Deep research is temporarily unavailable. Please try again shortly."
    retry_after_seconds: int | None = None


class JobInputTooLargeError(JobAdmissionError):
    """The deep-research input exceeds the configured admission bound."""

    status_code = 413

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.public_message = f"Deep research input exceeds the {max_chars}-character limit."
        super().__init__(self.public_message)


class JobSubmissionRateExceededError(JobAdmissionError):
    """The principal exhausted its per-minute submission budget."""

    status_code = 429
    public_message = "Deep research submission rate limit reached. Please try again shortly."
    retry_after_seconds = _RATE_WINDOW_SECONDS


class JobPrincipalCapacityExceededError(JobAdmissionError):
    """The principal already owns the maximum number of active jobs."""

    status_code = 429
    public_message = "Active deep research job limit reached. Wait for a running job to finish."
    retry_after_seconds = 30


class JobGlobalCapacityExceededError(JobAdmissionError):
    """The deployment has reached its active-job capacity."""

    status_code = 503
    public_message = "The server is at deep research capacity. Please try again shortly."
    retry_after_seconds = 30


class JobAdmissionUnavailableError(JobAdmissionError):
    """The admission store could not make a safe decision."""


class JobAdmissionConflictError(RuntimeError):
    """The requested job ID already has an admission record."""


@dataclass(frozen=True)
class DeepResearchAdmissionLimits:
    """Validated admission limits loaded from deployment configuration."""

    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_active_per_principal: int = DEFAULT_MAX_ACTIVE_PER_PRINCIPAL
    max_active_global: int = DEFAULT_MAX_ACTIVE_GLOBAL
    max_submissions_per_minute: int = DEFAULT_MAX_SUBMISSIONS_PER_MINUTE
    reservation_ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS

    @classmethod
    def from_env(cls) -> DeepResearchAdmissionLimits:
        return cls(
            max_input_chars=_positive_int_env(
                "DEEP_RESEARCHER_MAX_DEEP_RESEARCH_INPUT_CHARS",
                DEFAULT_MAX_INPUT_CHARS,
                maximum=DEFAULT_MAX_RESEARCH_INPUT_CHARS,
            ),
            max_active_per_principal=_positive_int_env(
                "DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_PER_PRINCIPAL",
                DEFAULT_MAX_ACTIVE_PER_PRINCIPAL,
            ),
            max_active_global=_positive_int_env(
                "DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_GLOBAL",
                DEFAULT_MAX_ACTIVE_GLOBAL,
            ),
            max_submissions_per_minute=_positive_int_env(
                "DEEP_RESEARCHER_MAX_DEEP_RESEARCH_SUBMISSIONS_PER_MINUTE",
                DEFAULT_MAX_SUBMISSIONS_PER_MINUTE,
            ),
            reservation_ttl_seconds=DEFAULT_RESERVATION_TTL_SECONDS,
        )


def _positive_int_env(name: str, default: int, *, maximum: int | None = None) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        return default
    if value < 1:
        logger.warning("%s must be positive; using default %d", name, default)
        return default
    if maximum is not None and value > maximum:
        logger.warning("%s cannot exceed %d; using hard maximum", name, maximum)
        return maximum
    return value


def validate_deep_research_input(
    input_text: str,
    limits: DeepResearchAdmissionLimits | None = None,
) -> DeepResearchAdmissionLimits:
    """Reject oversized input before any database or Dask work."""
    resolved_limits = limits or DeepResearchAdmissionLimits.from_env()
    if len(input_text) > resolved_limits.max_input_chars:
        raise JobInputTooLargeError(resolved_limits.max_input_chars)
    return resolved_limits


async def reserve_deep_research_job(
    *,
    db_url: str,
    job_id: str,
    principal: Principal,
    limits: DeepResearchAdmissionLimits,
) -> str:
    """Atomically reserve capacity and return its opaque ownership token."""
    reservation_token = secrets.token_urlsafe(24)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            _reserve_deep_research_job_sync,
            db_url,
            job_id,
            reservation_token,
            principal,
            limits,
            time.time(),
        )
        return reservation_token
    except JobAdmissionError:
        raise
    except JobAdmissionConflictError:
        raise
    except Exception as exc:
        logger.warning("Deep research admission store unavailable (error_type=%s)", type(exc).__name__)
        raise JobAdmissionUnavailableError from exc


async def renew_deep_research_job_reservation(
    *,
    db_url: str,
    job_id: str,
    reservation_token: str,
    ttl_seconds: float,
) -> bool:
    """Extend an in-flight reservation iff its opaque token still matches."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            _renew_deep_research_job_reservation_sync,
            db_url,
            job_id,
            reservation_token,
            time.time() + ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - the submit lease fails closed
        logger.warning("Could not renew deep research admission reservation (error_type=%s)", type(exc).__name__)
        return False


async def release_deep_research_job_reservation(
    *,
    db_url: str,
    job_id: str,
    reservation_token: str,
) -> bool:
    """Release a reservation iff the caller still owns its opaque token."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            _release_deep_research_job_reservation_sync,
            db_url,
            job_id,
            reservation_token,
        )
    except Exception as exc:  # noqa: BLE001 - expiry cleanup is the fail-safe
        logger.warning(
            "Could not release failed deep research admission reservation (error_type=%s)",
            type(exc).__name__,
        )
        return False


def ensure_deep_research_admission_table(db_url: str) -> None:
    """Create admission state eagerly so startup fails before reporting ready."""
    _ensure_admission_schema(db_url)
    validate_deep_research_admission_table(db_url)


def validate_deep_research_admission_table(db_url: str) -> None:
    """Raise unless the admission table exposes every column used by submission."""
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        conn.execute(_VALIDATE_ADMISSION_SCHEMA_SQL)


def is_deep_research_reservation_current(db_url: str, job_id: str, reservation_token: str) -> bool:
    """Return whether a worker still owns the admission fencing token."""
    _ensure_admission_schema(db_url)
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        return (
            conn.execute(
                text(
                    "SELECT 1 FROM deep_research_admission "
                    "WHERE job_id = :job_id AND reservation_token = :reservation_token"
                ),
                {"job_id": job_id, "reservation_token": reservation_token},
            ).first()
            is not None
        )


def _reserve_deep_research_job_sync(
    db_url: str,
    job_id: str,
    reservation_token: str,
    principal: Principal,
    limits: DeepResearchAdmissionLimits,
    now: float,
) -> None:
    _ensure_admission_schema(db_url)
    engine = EventStore._get_or_create_sync_engine(db_url)
    rate_cutoff = now - _RATE_WINDOW_SECONDS

    with engine.connect() as conn:
        _begin_serialized_transaction(conn, db_url)
        try:
            params = {
                "owner_auth_type": principal.type,
                "owner_subject": principal.sub,
                "now": now,
                "rate_cutoff": rate_cutoff,
            }
            conn.execute(_CLEANUP_RESERVATIONS_SQL, params)

            recent = int(conn.execute(_COUNT_RECENT_SQL, params).scalar() or 0)
            if recent >= limits.max_submissions_per_minute:
                conn.commit()
                raise JobSubmissionRateExceededError

            owner_active = int(conn.execute(_COUNT_ACTIVE_FOR_OWNER_SQL, params).scalar() or 0)
            if owner_active >= limits.max_active_per_principal:
                conn.commit()
                raise JobPrincipalCapacityExceededError

            global_active = int(conn.execute(_COUNT_ACTIVE_GLOBAL_SQL, params).scalar() or 0)
            if global_active >= limits.max_active_global:
                conn.commit()
                raise JobGlobalCapacityExceededError

            try:
                conn.execute(
                    _INSERT_RESERVATION_SQL,
                    {
                        **params,
                        "job_id": job_id,
                        "reservation_token": reservation_token,
                        "admitted_at": now,
                        "reservation_expires_at": now + limits.reservation_ttl_seconds,
                    },
                )
            except IntegrityError as exc:
                conn.rollback()
                raise JobAdmissionConflictError(f"Job admission already exists: {job_id}") from exc

            conn.commit()
        except (JobAdmissionError, JobAdmissionConflictError):
            raise
        except Exception:
            conn.rollback()
            raise


def _renew_deep_research_job_reservation_sync(
    db_url: str,
    job_id: str,
    reservation_token: str,
    reservation_expires_at: float,
) -> bool:
    _ensure_admission_schema(db_url)
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            _RENEW_RESERVATION_SQL,
            {
                "job_id": job_id,
                "reservation_token": reservation_token,
                "reservation_expires_at": reservation_expires_at,
            },
        )
        conn.commit()
        return (result.rowcount or 0) == 1


def _release_deep_research_job_reservation_sync(
    db_url: str,
    job_id: str,
    reservation_token: str,
) -> bool:
    _ensure_admission_schema(db_url)
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            _DELETE_RESERVATION_SQL,
            {"job_id": job_id, "reservation_token": reservation_token},
        )
        conn.commit()
        return (result.rowcount or 0) == 1


def _ensure_admission_schema(db_url: str) -> None:
    if db_url in _schema_initialized:
        return
    with _schema_lock:
        if db_url in _schema_initialized:
            return
        engine = EventStore._get_or_create_sync_engine(db_url)
        with engine.connect() as conn:
            _ensure_job_access_schema(conn, db_url)
            conn.execute(text(_CREATE_ADMISSION_TABLE_SQL))
            _ensure_admission_extra_columns(conn, db_url)
            conn.execute(text(_CREATE_OWNER_INDEX_SQL))
            conn.commit()
        _schema_initialized.add(db_url)


def _ensure_admission_extra_columns(conn: Connection, db_url: str) -> None:
    """Add the opaque-token column to an earlier admission-table revision."""
    if db_url.startswith("postgres"):
        conn.execute(text("ALTER TABLE deep_research_admission ADD COLUMN IF NOT EXISTS reservation_token VARCHAR"))
    else:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(deep_research_admission)")).fetchall()}
        if "reservation_token" not in columns:
            conn.execute(text("ALTER TABLE deep_research_admission ADD COLUMN reservation_token VARCHAR"))

    # Rows created by the pre-token implementation cannot be released safely.
    # Removing them is fail-safe because this migration runs during process
    # startup, before this process can have an in-flight submitter for them.
    conn.execute(text("DELETE FROM deep_research_admission WHERE reservation_token IS NULL"))
    if db_url.startswith("postgres"):
        conn.execute(text("ALTER TABLE deep_research_admission ALTER COLUMN reservation_token SET NOT NULL"))


def _begin_serialized_transaction(conn: Connection, db_url: str) -> None:
    if db_url.startswith("postgres"):
        conn.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": _PG_ADMISSION_LOCK_ID},
        )
        return
    if db_url.startswith("sqlite"):
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        return
    raise RuntimeError("Deep research admission supports only PostgreSQL and SQLite job stores")
