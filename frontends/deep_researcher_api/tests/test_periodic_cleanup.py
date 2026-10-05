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

"""Tests for periodic cleanup of expired jobs and old events."""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from deep_researcher_api.jobs.event_store import EventStore


@pytest.fixture
def db_url(tmp_path):
    """Return an async SQLite database URL."""
    return f"sqlite+aiosqlite:///{tmp_path / 'test_cleanup.db'}"


@pytest.fixture(autouse=True)
def clear_event_store_caches():
    """Clear EventStore caches between tests."""
    EventStore._tables_initialized.clear()
    yield
    EventStore._tables_initialized.clear()


def _backdate_events(db_url: str, hours: int, *, job_id: str | None = None, event_id: int | None = None):
    """Helper to backdate events in the test DB."""
    from sqlalchemy import text

    engine = EventStore._get_or_create_sync_engine(db_url)
    old_time = datetime.now(UTC) - timedelta(hours=hours)
    with engine.connect() as conn:
        params: dict = {"ts": old_time.replace(tzinfo=None)}
        if event_id is not None:
            query = "UPDATE job_events SET created_at = :ts WHERE id = :event_id"
            params["event_id"] = event_id
        elif job_id is not None:
            query = "UPDATE job_events SET created_at = :ts WHERE job_id = :job_id"
            params["job_id"] = job_id
        else:
            query = "UPDATE job_events SET created_at = :ts"
        conn.execute(text(query), params)
        conn.commit()


def _create_expired_job(db_url: str, job_id: str):
    """Helper to create a fake expired job in the job_info table."""
    from sqlalchemy import text

    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        # Ensure job_info table exists (simplified schema for tests)
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS job_info ("
                "  job_id TEXT PRIMARY KEY,"
                "  status TEXT,"
                "  config_file TEXT,"
                "  error TEXT,"
                "  output_path TEXT,"
                "  created_at DATETIME,"
                "  updated_at DATETIME,"
                "  expiry_seconds INTEGER,"
                "  output TEXT,"
                "  is_expired BOOLEAN DEFAULT 0"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT OR REPLACE INTO job_info "
                "(job_id, status, expiry_seconds, is_expired, created_at, updated_at) "
                "VALUES (:job_id, 'success', 3600, :is_expired, :ts, :ts)"
            ),
            {
                "job_id": job_id,
                "is_expired": True,
                "ts": datetime.now(UTC).replace(tzinfo=None),
            },
        )
        conn.commit()


# =========================================================================
# EventStore.cleanup_old_events (time-based)
# =========================================================================


class TestCleanupOldEvents:
    """Tests for EventStore.cleanup_old_events."""

    def test_cleanup_deletes_old_events(self, db_url):
        """Events older than retention period should be deleted."""
        store = EventStore(db_url, job_id="job-1")
        store.store({"type": "test.event", "data": {"msg": "hello"}})
        store.store({"type": "test.event", "data": {"msg": "world"}})

        _backdate_events(db_url, hours=2)

        deleted = EventStore.cleanup_old_events(db_url, retention_seconds=3600)
        assert deleted == 2
        assert len(EventStore.get_events(db_url, "job-1")) == 0

    def test_cleanup_preserves_recent_events(self, db_url):
        """Events within retention period should not be deleted."""
        store = EventStore(db_url, job_id="job-1")
        store.store({"type": "test.event", "data": {"msg": "recent"}})

        deleted = EventStore.cleanup_old_events(db_url, retention_seconds=3600)
        assert deleted == 0
        assert len(EventStore.get_events(db_url, "job-1")) == 1

    def test_cleanup_mixed_old_and_new(self, db_url):
        """Only old events should be deleted when both old and new exist."""
        store = EventStore(db_url, job_id="job-1")
        store.store({"type": "old.event", "data": {"msg": "old"}})
        store.store({"type": "new.event", "data": {"msg": "new"}})

        _backdate_events(db_url, hours=2, event_id=1)

        deleted = EventStore.cleanup_old_events(db_url, retention_seconds=3600)
        assert deleted == 1
        assert len(EventStore.get_events(db_url, "job-1")) == 1

    def test_cleanup_multiple_jobs(self, db_url):
        """Cleanup should delete old events across all jobs."""
        EventStore(db_url, job_id="job-1").store({"type": "test", "data": {}})
        EventStore(db_url, job_id="job-2").store({"type": "test", "data": {}})

        _backdate_events(db_url, hours=2)

        deleted = EventStore.cleanup_old_events(db_url, retention_seconds=3600)
        assert deleted == 2

    def test_cleanup_empty_table(self, db_url):
        """Cleanup on empty table should return 0."""
        EventStore(db_url, job_id="job-1")  # ensures table exists
        assert EventStore.cleanup_old_events(db_url, retention_seconds=3600) == 0


class TestCleanupOldEventsAsync:
    """Tests for EventStore.cleanup_old_events_async."""

    @pytest.mark.asyncio
    async def test_async_cleanup_delegates_to_sync(self, db_url):
        """Async cleanup should produce same results as sync."""
        store = EventStore(db_url, job_id="job-1")
        store.store({"type": "test", "data": {}})
        _backdate_events(db_url, hours=2)

        deleted = await EventStore.cleanup_old_events_async(db_url, retention_seconds=3600)
        assert deleted == 1


class TestCleanupJobEvents:
    """Tests for EventStore.cleanup_job_events (targeted per-job deletion)."""

    def test_cleanup_specific_job(self, db_url):
        """Should delete events for a specific job only."""
        EventStore(db_url, job_id="job-1").store({"type": "test", "data": {}})
        EventStore(db_url, job_id="job-1").store({"type": "test", "data": {}})
        EventStore(db_url, job_id="job-2").store({"type": "test", "data": {}})

        deleted = EventStore.cleanup_job_events(db_url, "job-1")
        assert deleted == 2
        assert len(EventStore.get_events(db_url, "job-2")) == 1


# =========================================================================
# _run_event_cleanup (coordinated: time-based + expired-job events)
# =========================================================================


class TestRunEventCleanup:
    """Tests for _run_event_cleanup coordinated cleanup cycle."""

    @pytest.mark.asyncio
    async def test_removes_events_for_expired_jobs(self, db_url):
        """Events for jobs marked is_expired=True in job_info should be deleted."""
        from deep_researcher_api.routes.jobs import _run_event_cleanup

        store = EventStore(db_url, job_id="expired-job")
        store.store({"type": "test", "data": {"msg": "should be deleted"}})

        _create_expired_job(db_url, "expired-job")

        await _run_event_cleanup(db_url, retention_seconds=86400, is_postgres=False)

        remaining = EventStore.get_events(db_url, "expired-job")
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_preserves_events_for_non_expired_jobs(self, db_url):
        """Events for jobs NOT marked expired should be preserved (if within retention)."""
        from deep_researcher_api.routes.jobs import _run_event_cleanup

        store = EventStore(db_url, job_id="active-job")
        store.store({"type": "test", "data": {"msg": "keep me"}})

        # Create job_info table but no expired jobs
        _create_expired_job(db_url, "some-other-job")

        await _run_event_cleanup(db_url, retention_seconds=86400, is_postgres=False)

        remaining = EventStore.get_events(db_url, "active-job")
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_combined_time_and_expired_cleanup(self, db_url):
        """Both time-based and expired-job cleanup should run in one cycle."""
        from deep_researcher_api.routes.jobs import _run_event_cleanup

        # Old event for a non-expired job (should be cleaned by time)
        store1 = EventStore(db_url, job_id="old-job")
        store1.store({"type": "test", "data": {}})
        _backdate_events(db_url, hours=2, job_id="old-job")

        # Recent event for an expired job (should be cleaned by expired-job logic)
        store2 = EventStore(db_url, job_id="expired-job")
        store2.store({"type": "test", "data": {}})
        _create_expired_job(db_url, "expired-job")

        # Recent event for a live job (should survive)
        store3 = EventStore(db_url, job_id="live-job")
        store3.store({"type": "test", "data": {}})

        await _run_event_cleanup(db_url, retention_seconds=3600, is_postgres=False)

        assert len(EventStore.get_events(db_url, "old-job")) == 0
        assert len(EventStore.get_events(db_url, "expired-job")) == 0
        assert len(EventStore.get_events(db_url, "live-job")) == 1

    @pytest.mark.asyncio
    async def test_sqlite_removes_expired_artifacts_after_event_cleanup(self, db_url):
        """SQLite releases event writes before artifact retention uses a second connection."""
        from sqlalchemy import text

        from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import Artifact
        from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import ArtifactKind
        from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store
        from deep_researcher_api.routes.jobs import _run_event_cleanup

        event_store = EventStore(db_url, job_id="old-job")
        event_store.store({"type": "test", "data": {}})
        _backdate_events(db_url, hours=2, job_id="old-job")
        _create_expired_job(db_url, "expired-other-job")

        artifact_store = build_artifact_store(db_url)
        stored = artifact_store.put(
            Artifact(
                artifact_id="art_" + "a" * 32,
                job_id="old-job",
                kind=ArtifactKind.TEXT,
                mime_type="text/plain",
                filename="result.txt",
                sandbox_path="/tmp/artifacts/result.txt",
                storage_uri="",
                sha256="b" * 64,
                size_bytes=6,
            ),
            b"result",
        )
        with artifact_store._engine.connect() as conn:
            conn.execute(
                text("UPDATE artifacts SET created_at = datetime('now', '-2 seconds') WHERE artifact_id = :id"),
                {"id": stored.artifact_id},
            )
            conn.commit()

        await _run_event_cleanup(db_url, retention_seconds=1, is_postgres=False)

        assert artifact_store.get(stored.job_id, stored.artifact_id) is None

    @pytest.mark.asyncio
    async def test_artifact_cleanup_error_does_not_log_endpoint(self, db_url, caplog):
        """Artifact retention failures expose the exception type, not configured endpoints."""
        from deep_researcher_api.routes.jobs import _run_event_cleanup

        _create_expired_job(db_url, "expired-job")
        EventStore(db_url, job_id="active-job").store({"type": "test", "data": {}})
        artifact_store = MagicMock()
        artifact_store.cleanup_old_artifacts.side_effect = RuntimeError("https://secret-artifacts.internal")
        caplog.set_level("DEBUG", logger="deep_researcher_api.routes.jobs")

        with patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.artifacts.build_artifact_store",
            return_value=artifact_store,
        ):
            await _run_event_cleanup(db_url, retention_seconds=3600, is_postgres=False)

        assert "RuntimeError" in caplog.text
        assert "secret-artifacts.internal" not in caplog.text

    @pytest.mark.asyncio
    async def test_postgres_non_leader_skips_artifact_cleanup(self):
        """Only the advisory-lock holder may delete retained artifacts."""
        from deep_researcher_api.routes.jobs import _run_event_cleanup

        lock_result = MagicMock()
        lock_result.scalar.return_value = False
        connection = MagicMock()
        connection.execute.return_value = lock_result
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection

        with (
            patch(
                "deep_researcher_api.jobs.event_store.EventStore._get_or_create_sync_engine",
                return_value=engine,
            ),
            patch(
                "deep_researcher_agent.agents.deep_researcher.sandbox.artifacts.build_artifact_store"
            ) as mock_artifact_store,
        ):
            await _run_event_cleanup(
                "postgresql+asyncpg://example.invalid/jobs",
                retention_seconds=3600,
                is_postgres=True,
            )

        mock_artifact_store.assert_not_called()
        connection.commit.assert_not_called()


# =========================================================================
# _cleanup_old_events_loop (background task)
# =========================================================================


class TestCleanupOldEventsLoop:
    """Tests for _cleanup_old_events_loop background task."""

    @pytest.mark.asyncio
    async def test_runs_immediately_on_startup(self):
        """The loop should run one cleanup cycle before the first sleep."""
        from deep_researcher_api.routes.jobs import _cleanup_old_events_loop

        calls = []

        async def mock_run(db_url, retention_seconds, is_postgres):
            calls.append(("run", retention_seconds))
            if len(calls) >= 2:
                raise asyncio.CancelledError()

        with patch("deep_researcher_api.routes.jobs._run_event_cleanup", side_effect=mock_run):
            task = asyncio.create_task(
                _cleanup_old_events_loop(
                    db_url="sqlite+aiosqlite:///test.db",
                    retention_seconds=3600,
                    interval_seconds=9999,  # long interval — shouldn't matter if startup run works
                )
            )
            # Give the startup run time to execute
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(calls) >= 1, "Should have run at least once immediately on startup"

    @pytest.mark.asyncio
    async def test_loop_survives_cleanup_errors(self):
        """The loop should continue running even if cleanup raises."""
        from deep_researcher_api.routes.jobs import _cleanup_old_events_loop

        call_count = 0

        async def mock_run(db_url, retention_seconds, is_postgres):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                pass  # startup run — succeeds
            elif call_count == 2:
                raise RuntimeError("DB connection failed")
            elif call_count >= 4:
                raise asyncio.CancelledError()

        with patch("deep_researcher_api.routes.jobs._run_event_cleanup", side_effect=mock_run):
            task = asyncio.create_task(
                _cleanup_old_events_loop(
                    db_url="sqlite+aiosqlite:///test.db",
                    retention_seconds=3600,
                    interval_seconds=0,
                )
            )
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        assert call_count >= 3, "Should have continued past the error"

    @pytest.mark.asyncio
    async def test_runs_job_cleanup_locally_without_dask_task(self):
        """Job expiry should run in the API process without occupying a Dask worker."""
        from deep_researcher_api.routes.jobs import _cleanup_old_events_loop

        mock_job_store = MagicMock()

        async def mock_job_cleanup():
            return 0

        mock_job_store.cleanup_expired_jobs.side_effect = mock_job_cleanup

        async def mock_event_cleanup(*_args):
            return None

        with patch("deep_researcher_api.routes.jobs._run_event_cleanup", side_effect=mock_event_cleanup):
            task = asyncio.create_task(
                _cleanup_old_events_loop(
                    db_url="sqlite+aiosqlite:///test.db",
                    retention_seconds=3600,
                    interval_seconds=9999,
                    job_store=mock_job_store,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            await task

        mock_job_store.cleanup_expired_jobs.assert_called_once_with()
        mock_job_store.dask_client.submit.assert_not_called()


class TestRunJobCleanup:
    """Tests for PostgreSQL cleanup leader election."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("lock_acquired", "expected_result"), [(True, 3), (False, None)])
    async def test_only_lock_holder_expires_jobs(self, lock_acquired, expected_result):
        from deep_researcher_api.routes.jobs import _run_job_cleanup

        job_store = MagicMock()
        job_store.cleanup_expired_jobs = AsyncMock(return_value=3)

        lock_result = MagicMock()
        lock_result.scalar.return_value = lock_acquired
        lock_session = MagicMock()
        lock_session.execute = AsyncMock(return_value=lock_result)
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=lock_session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        job_store.session.return_value = session_context

        assert await _run_job_cleanup(job_store, is_postgres=True) == expected_result

        if lock_acquired:
            job_store.cleanup_expired_jobs.assert_awaited_once_with()
        else:
            job_store.cleanup_expired_jobs.assert_not_awaited()


# =========================================================================
# _start_periodic_cleanup (orchestration)
# =========================================================================


class TestStartPeriodicCleanup:
    """Tests for _start_periodic_cleanup orchestration function."""

    def test_starts_one_local_cleanup_task_without_dask_submission(self):
        """Housekeeping should not reserve a worker in the shared Dask pool."""
        import deep_researcher_api.routes.jobs as jobs_module

        jobs_module._cleanup_task = None
        mock_job_store = MagicMock()
        cleanup_coroutine = object()
        mock_loop = MagicMock(return_value=cleanup_coroutine)

        with (
            patch("deep_researcher_api.routes.jobs._cleanup_old_events_loop", new=mock_loop),
            patch("deep_researcher_api.routes.jobs.asyncio.create_task") as mock_create_task,
        ):
            jobs_module._start_periodic_cleanup(
                job_store=mock_job_store,
                db_url="sqlite:///test.db",
                expiry_seconds=3600,
            )

        mock_loop.assert_called_once_with(
            "sqlite:///test.db",
            3600,
            1800,
            job_store=mock_job_store,
        )
        mock_create_task.assert_called_once_with(cleanup_coroutine)
        mock_job_store.dask_client.submit.assert_not_called()

    @pytest.mark.parametrize(
        ("expiry_seconds", "expected_interval"),
        [
            (604800, 3600),
            (60, 60),
        ],
    )
    def test_cleanup_interval_is_clamped(self, expiry_seconds, expected_interval):
        """Local cleanup interval should remain between one minute and one hour."""
        import deep_researcher_api.routes.jobs as jobs_module

        jobs_module._cleanup_task = None
        mock_job_store = MagicMock()
        mock_loop = MagicMock(return_value=object())

        with (
            patch("deep_researcher_api.routes.jobs._cleanup_old_events_loop", new=mock_loop),
            patch("deep_researcher_api.routes.jobs.asyncio.create_task"),
        ):
            jobs_module._start_periodic_cleanup(
                job_store=mock_job_store,
                db_url="sqlite:///test.db",
                expiry_seconds=expiry_seconds,
            )

        assert mock_loop.call_args.args[2] == expected_interval


# =========================================================================
# stop_periodic_cleanup (graceful shutdown)
# =========================================================================


class TestStopPeriodicCleanup:
    """Tests for stop_periodic_cleanup shutdown function."""

    @pytest.mark.asyncio
    async def test_cancels_running_task(self):
        """Should cancel the background cleanup task."""
        import deep_researcher_api.routes.jobs as jobs_module

        async def long_running():
            await asyncio.sleep(9999)

        jobs_module._cleanup_task = asyncio.create_task(long_running())
        assert not jobs_module._cleanup_task.done()

        await jobs_module.stop_periodic_cleanup()

        assert jobs_module._cleanup_task is None

    @pytest.mark.asyncio
    async def test_noop_when_no_task(self):
        """Should not raise if no task is running."""
        import deep_researcher_api.routes.jobs as jobs_module

        jobs_module._cleanup_task = None
        await jobs_module.stop_periodic_cleanup()  # should not raise


class TestCancelDaskTask:
    """Tests for cancelling submitted Dask jobs."""

    @pytest.mark.asyncio
    async def test_cancels_deterministic_future_key_without_variable_get(self, monkeypatch):
        from deep_researcher_api.routes.jobs import _cancel_dask_task

        calls: dict[str, object] = {}

        class FakeFuture:
            def __init__(self, key, client):
                self.key = key
                self.client = client

        class FakeClient:
            def __init__(self, scheduler_address, asynchronous):
                calls["scheduler_address"] = scheduler_address
                calls["asynchronous"] = asynchronous

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def cancel(self, futures, asynchronous, force):
                calls["cancelled_keys"] = [future.key for future in futures]
                calls["cancel_asynchronous"] = asynchronous
                calls["force"] = force

        fake_distributed = types.SimpleNamespace(Client=FakeClient, Future=FakeFuture)
        monkeypatch.setitem(sys.modules, "distributed", fake_distributed)

        assert await _cancel_dask_task("tcp://localhost:8786", "job-123") is True
        assert calls == {
            "scheduler_address": "tcp://localhost:8786",
            "asynchronous": True,
            "cancelled_keys": ["job-123-job"],
            "cancel_asynchronous": True,
            "force": True,
        }
