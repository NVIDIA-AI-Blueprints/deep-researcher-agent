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

"""Tests for the ghost-job reaper's stale-job detection (_find_stale_jobs).

These use a real SQLite database so the SQL runs exactly as it would in
production. The key regression: a job that entered RUNNING but never stored an
event (worker crash/OOM before the first event flush) must still be reaped;
before the fix, the INNER JOIN on job_events made such jobs invisible.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher_api.routes.jobs import GHOST_JOB_TIMEOUT_SECONDS  # noqa: E402
from deep_researcher_api.routes.jobs import _find_stale_jobs  # noqa: E402
from deep_researcher_api.routes.jobs import _mark_job_failed_if_running  # noqa: E402

RUNNING = "running"
FAILURE = "failure"
SUCCESS = "success"


def _make_db() -> str:
    """Create a temp SQLite DB with the job_info and job_events tables."""
    db_path = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"

    # job_info comes from NAT's ORM model.
    from nat.front_ends.fastapi.async_jobs.job_store import JobInfo

    engine = create_engine(db_url)
    JobInfo.__table__.metadata.create_all(engine)

    # job_events is created by deep-researcher's EventStore.
    from deep_researcher_api.jobs.event_store import EventStore

    EventStore._ensure_table_exists(db_url)
    return db_url


def _insert_job(
    db_url: str,
    job_id: str,
    *,
    status: str,
    updated_ago_seconds: float,
    created_ago_seconds: float | None = None,
) -> None:
    """Insert a job_info row. updated_at (the lease) and created_at ages differ
    when created_ago_seconds is given, to model a long-running-but-live job."""
    from nat.front_ends.fastapi.async_jobs.job_store import JobInfo

    now = datetime.now(UTC)
    updated = now - timedelta(seconds=updated_ago_seconds)
    created = now - timedelta(seconds=created_ago_seconds if created_ago_seconds is not None else updated_ago_seconds)
    engine = create_engine(db_url)
    with Session(engine) as s:
        s.add(JobInfo(job_id=job_id, status=status, expiry_seconds=3600, created_at=created, updated_at=updated))
        s.commit()


def _get_status(db_url: str, job_id: str) -> str | None:
    """Return the stored status for a job, or None if absent."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status FROM job_info WHERE job_id = :j"), {"j": job_id}).first()
    return row[0] if row else None


def _insert_event(db_url: str, job_id: str, *, created_ago_seconds: float) -> None:
    """Insert a job_events row with created_at set to now minus the given age."""
    ts = (datetime.now(UTC) - timedelta(seconds=created_ago_seconds)).strftime("%Y-%m-%d %H:%M:%S")
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO job_events (job_id, event_type, event_data, created_at) VALUES (:j, :t, :d, :c)"),
            {"j": job_id, "t": "test.event", "d": "{}", "c": ts},
        )


OLD = GHOST_JOB_TIMEOUT_SECONDS + 60
RECENT = 5


def test_zero_event_running_job_past_timeout_is_reaped():
    """A RUNNING job with no events, started long ago, is a ghost and reaped.

    This is the regression: the old INNER JOIN made zero-event jobs invisible.
    """
    db = _make_db()
    _insert_job(db, "ghost", status=RUNNING, updated_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == ["ghost"]


def test_zero_event_running_job_within_timeout_is_not_reaped():
    """A freshly-started RUNNING job with no events yet must not be reaped."""
    db = _make_db()
    _insert_job(db, "fresh", status=RUNNING, updated_ago_seconds=RECENT)
    assert _find_stale_jobs(db, RUNNING) == []


def test_running_job_with_stale_last_event_is_reaped():
    """Existing behavior preserved: events present but last one is old."""
    db = _make_db()
    _insert_job(db, "stalled", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "stalled", created_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == ["stalled"]


def test_running_job_with_recent_event_is_not_reaped():
    """A job actively emitting events is healthy, even if it started long ago."""
    db = _make_db()
    _insert_job(db, "active", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "active", created_ago_seconds=RECENT)
    assert _find_stale_jobs(db, RUNNING) == []


def test_non_running_job_is_never_reaped():
    """Only RUNNING jobs are candidates; a completed job is left alone."""
    db = _make_db()
    _insert_job(db, "done", status="success", updated_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == []


def test_missing_tables_returns_empty():
    """No job_info/job_events tables (fresh deployment) -> nothing to reap."""
    db_path = tempfile.mktemp(suffix=".db")
    assert _find_stale_jobs(f"sqlite:///{db_path}", RUNNING) == []


def test_mixed_fleet_reaps_only_ghosts():
    """A realistic mix: only the two ghosts (old zero-event + stalled) return."""
    db = _make_db()
    _insert_job(db, "ghost-zero", status=RUNNING, updated_ago_seconds=OLD)
    _insert_job(db, "fresh-zero", status=RUNNING, updated_ago_seconds=RECENT)
    _insert_job(db, "stalled", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "stalled", created_ago_seconds=OLD)
    _insert_job(db, "healthy", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "healthy", created_ago_seconds=RECENT)
    _insert_job(db, "done", status="success", updated_ago_seconds=OLD)

    assert sorted(_find_stale_jobs(db, RUNNING)) == ["ghost-zero", "stalled"]


# ---------------------------------------------------------------------------
# Cold-start lease + atomic conditional transition (issue #318 review follow-up)
# ---------------------------------------------------------------------------


def test_slow_init_job_with_fresh_lease_is_not_reaped():
    """A live worker in a long cold start refreshes its lease (updated_at), so a
    zero-event RUNNING job that STARTED long ago but was touched recently is not
    reaped — the regression AjayThorve flagged."""
    db = _make_db()
    # created long ago (slow init still running), but lease refreshed just now.
    _insert_job(db, "slow-init", status=RUNNING, updated_ago_seconds=RECENT, created_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == []


def test_stale_lease_zero_event_job_is_reaped():
    """Once the lease goes stale (worker dead), the zero-event ghost is reaped."""
    db = _make_db()
    _insert_job(db, "dead", status=RUNNING, updated_ago_seconds=OLD, created_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == ["dead"]


def test_conditional_transition_marks_running_job_failed():
    """The atomic transition flips a still-RUNNING job to FAILURE and reports it."""
    db = _make_db()
    _insert_job(db, "ghost", status=RUNNING, updated_ago_seconds=OLD)
    did = _mark_job_failed_if_running(db, "ghost", RUNNING, FAILURE, "timed out")
    assert did is True
    assert _get_status(db, "ghost") == FAILURE


def test_conditional_transition_does_not_clobber_terminal_job():
    """A job that reached SUCCESS between detection and reaping is left intact."""
    db = _make_db()
    _insert_job(db, "finished", status=SUCCESS, updated_ago_seconds=OLD)
    did = _mark_job_failed_if_running(db, "finished", RUNNING, FAILURE, "timed out")
    assert did is False
    assert _get_status(db, "finished") == SUCCESS  # unchanged


def test_conditional_transition_missing_job_is_noop():
    """Transitioning a non-existent job returns False without error."""
    db = _make_db()
    assert _mark_job_failed_if_running(db, "nope", RUNNING, FAILURE, "x") is False


def test_runner_lease_touch_refreshes_only_running_jobs():
    """The runner's lease bumps updated_at for a RUNNING job but not a terminal one."""
    from deep_researcher_api.jobs.runner import _touch_job_lease_sync

    db = _make_db()
    _insert_job(db, "run", status=RUNNING, updated_ago_seconds=OLD)
    _insert_job(db, "done", status=SUCCESS, updated_ago_seconds=OLD)

    # Before: the running job is stale and would be reaped.
    assert _find_stale_jobs(db, RUNNING) == ["run"]

    _touch_job_lease_sync(db, "run")  # live worker refreshes its lease
    _touch_job_lease_sync(db, "done")  # must be a no-op for a terminal job

    # After: the running job's lease is fresh, so it is no longer reapable;
    # the terminal job's timestamp was not resurrected.
    assert _find_stale_jobs(db, RUNNING) == []
    assert _get_status(db, "done") == SUCCESS


def test_lease_refresher_thread_bumps_updated_at():
    """The dedicated lease thread actually refreshes a running job's lease.

    Uses a 0s interval so the thread refreshes immediately, then stops it.
    """
    import threading
    import time

    from deep_researcher_api.jobs import runner

    db = _make_db()
    _insert_job(db, "run", status=RUNNING, updated_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == ["run"]  # stale before any refresh

    stop = threading.Event()
    # Small positive interval so the refresher touches the lease promptly
    # without a zero-interval busy loop hammering the DB until teardown.
    original = runner.LEASE_REFRESH_INTERVAL_SECONDS
    runner.LEASE_REFRESH_INTERVAL_SECONDS = 0.01
    t = threading.Thread(target=runner._run_lease_refresher, args=(db, "run", stop), daemon=True)
    try:
        t.start()
        # Poll until the lease is refreshed (thread is on a 0s loop).
        for _ in range(50):
            if _find_stale_jobs(db, RUNNING) == []:
                break
            time.sleep(0.02)
        assert _find_stale_jobs(db, RUNNING) == []  # lease is now fresh
    finally:
        stop.set()
        t.join(timeout=5)
        runner.LEASE_REFRESH_INTERVAL_SECONDS = original
    assert not t.is_alive()  # thread exits promptly on stop


def test_db_now_expr_handles_both_postgres_schemes():
    """Both postgresql:// and legacy postgres:// map to NOW(); sqlite to CURRENT_TIMESTAMP."""
    from deep_researcher_api.jobs.runner import _db_now_expr

    assert _db_now_expr("postgresql://u@h/db") == "NOW()"
    assert _db_now_expr("postgres://u@h/db") == "NOW()"
    assert _db_now_expr("sqlite:///x.db") == "CURRENT_TIMESTAMP"


def test_success_cas_writes_only_when_running():
    """The worker's success write is a compare-and-set: it writes for a running
    job but not for one the reaper already moved to a terminal state."""
    from deep_researcher_api.jobs.runner import _write_job_success_if_running_sync

    db = _make_db()
    _insert_job(db, "run", status=RUNNING, updated_ago_seconds=RECENT)
    assert _write_job_success_if_running_sync(db, "run", '{"report": "ok"}') is True
    assert _get_status(db, "run") == SUCCESS


def test_success_cas_does_not_resurrect_reaped_job():
    """A job already reaped to FAILURE is not resurrected by a late success write."""
    from deep_researcher_api.jobs.runner import _write_job_success_if_running_sync

    db = _make_db()
    _insert_job(db, "reaped", status=FAILURE, updated_ago_seconds=RECENT)
    assert _write_job_success_if_running_sync(db, "reaped", '{"report": "late"}') is False
    assert _get_status(db, "reaped") == FAILURE  # stays failed
