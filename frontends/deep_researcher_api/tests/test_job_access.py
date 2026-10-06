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

from __future__ import annotations

import asyncio
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from deep_researcher_agent.auth import Principal
from deep_researcher_api.jobs import access as job_access
from deep_researcher_api.jobs.access import JobAccessConflictError
from deep_researcher_api.jobs.access import authorize_job_access
from deep_researcher_api.jobs.access import cleanup_job_access
from deep_researcher_api.jobs.access import create_job_access
from deep_researcher_api.jobs.access import ensure_job_access_table
from deep_researcher_api.jobs.access import get_job_access
from deep_researcher_api.jobs.access import release_job_access_reservation
from deep_researcher_api.jobs.access import renew_job_access_reservation
from deep_researcher_api.jobs.event_store import EventStore


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path / 'test_job_access.db'}"


@pytest.fixture(autouse=True)
def clear_event_store_caches():
    EventStore._tables_initialized.clear()
    job_access._job_access_schema_initialized.clear()
    yield
    EventStore._tables_initialized.clear()
    job_access._job_access_schema_initialized.clear()


def _insert_job_info(
    db_url: str,
    job_id: str,
    *,
    is_expired: bool = False,
    status: str = "running",
    created_at: datetime | None = None,
) -> None:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
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
        ts = (created_at or datetime.now(UTC)).replace(tzinfo=None)
        conn.execute(
            text(
                "INSERT OR REPLACE INTO job_info "
                "(job_id, status, created_at, updated_at, expiry_seconds, is_expired) "
                "VALUES (:job_id, :status, :ts, :ts, 3600, :is_expired)"
            ),
            {"job_id": job_id, "status": status, "ts": ts, "is_expired": is_expired},
        )
        conn.commit()


class TestJobAccessStorage:
    def test_schema_initialization_commits_before_caching(self, monkeypatch):
        db_url = "postgresql://test"
        conn = MagicMock()
        conn.in_transaction.return_value = False
        monkeypatch.setattr(job_access, "_ensure_extra_columns", MagicMock())

        job_access._ensure_job_access_schema(conn, db_url)

        conn.commit.assert_called_once_with()
        assert db_url in job_access._job_access_schema_initialized

    def test_schema_initialization_in_caller_transaction_is_not_cached(self, monkeypatch):
        db_url = "postgresql://test"
        conn = MagicMock()
        conn.in_transaction.return_value = True
        monkeypatch.setattr(job_access, "_ensure_extra_columns", MagicMock())

        job_access._ensure_job_access_schema(conn, db_url)

        conn.commit.assert_not_called()
        assert db_url not in job_access._job_access_schema_initialized

    def test_schema_initialization_commit_failure_is_not_cached(self, monkeypatch):
        db_url = "postgresql://test"
        conn = MagicMock()
        conn.in_transaction.return_value = False
        conn.commit.side_effect = RuntimeError("commit failed")
        monkeypatch.setattr(job_access, "_ensure_extra_columns", MagicMock())

        with pytest.raises(RuntimeError, match="commit failed"):
            job_access._ensure_job_access_schema(conn, db_url)

        assert db_url not in job_access._job_access_schema_initialized

    def test_create_and_get_job_access(self, db_url):
        principal = Principal(type="jwt", sub="user-1", email="alice@example.com")

        create_job_access("job-1", principal, db_url)

        access = get_job_access("job-1", db_url)
        assert access is not None
        assert access["owner_auth_type"] == "jwt"
        assert access["owner_subject"] == "user-1"
        assert access["owner_email"] == "alice@example.com"

    def test_create_job_access_persists_conversation_id(self, db_url):
        principal = Principal(type="jwt", sub="user-1", email="alice@example.com")

        create_job_access("job-1", principal, db_url, conversation_id="conv-A")

        access = get_job_access("job-1", db_url)
        assert access is not None
        assert access["conversation_id"] == "conv-A"

    def test_pre_enqueue_access_reservation_is_insert_only_and_token_guarded(self, db_url):
        _insert_job_info(db_url, "schema-bootstrap")
        engine = EventStore._get_or_create_sync_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM job_info WHERE job_id = 'schema-bootstrap'"))
            conn.commit()

        principal = Principal(type="jwt", sub="user-1")
        create_job_access(
            "job-1",
            principal,
            db_url,
            submission_token="current-token",
            submission_expires_at=10_000_000_000,
        )

        assert renew_job_access_reservation("job-1", "stale-token", db_url, 10_000_000_001) is False
        assert release_job_access_reservation("job-1", "stale-token", db_url) is False
        assert get_job_access("job-1", db_url) is not None
        assert renew_job_access_reservation("job-1", "current-token", db_url, 10_000_000_001) is True
        assert release_job_access_reservation("job-1", "current-token", db_url) is True
        assert get_job_access("job-1", db_url) is None

    def test_pre_enqueue_access_reservation_never_overwrites_existing_job(self, db_url):
        victim = Principal(type="jwt", sub="victim")
        _insert_job_info(db_url, "victim-job")
        create_job_access("victim-job", victim, db_url)

        with pytest.raises(JobAccessConflictError):
            create_job_access(
                "victim-job",
                Principal(type="jwt", sub="attacker"),
                db_url,
                submission_token="attacker-token",
                submission_expires_at=10_000_000_000,
            )

        assert get_job_access("victim-job", db_url)["owner_subject"] == "victim"


class TestLatestReportJobForConversation:
    """Backend fallback: resolve the conversation's most recent completed report job."""

    def _seed(
        self,
        db_url,
        job_id,
        principal,
        conversation_id,
        *,
        status,
        is_expired=False,
        created_at=None,
        agent_type="deep_researcher",
    ):
        _insert_job_info(db_url, job_id, status=status, is_expired=is_expired, created_at=created_at)
        create_job_access(job_id, principal, db_url, conversation_id=conversation_id, agent_type=agent_type)

    def test_excludes_non_report_agent(self, db_url):
        """A newer non-report job (e.g. shallow_researcher) must not become the active report."""
        from datetime import timedelta

        from deep_researcher_api.jobs.access import get_latest_report_job_for_conversation

        p = Principal(type="jwt", sub="user-1")
        base = datetime.now(UTC)
        self._seed(
            db_url,
            "report",
            p,
            "conv-A",
            status="success",
            created_at=base - timedelta(hours=1),
            agent_type="deep_researcher",
        )
        self._seed(
            db_url, "shallow-newer", p, "conv-A", status="success", created_at=base, agent_type="shallow_researcher"
        )

        # The newer shallow job is excluded; the report job still wins.
        assert get_latest_report_job_for_conversation("conv-A", p, db_url) == "report"

    def test_returns_latest_success_for_conversation_and_owner(self, db_url):
        from datetime import timedelta

        from deep_researcher_api.jobs.access import get_latest_report_job_for_conversation

        p = Principal(type="jwt", sub="user-1")
        base = datetime.now(UTC)
        self._seed(db_url, "old", p, "conv-A", status="success", created_at=base - timedelta(hours=2))
        self._seed(db_url, "new", p, "conv-A", status="success", created_at=base - timedelta(minutes=1))

        assert get_latest_report_job_for_conversation("conv-A", p, db_url) == "new"

    def test_ignores_non_success_expired_and_other_conversation(self, db_url):
        from deep_researcher_api.jobs.access import get_latest_report_job_for_conversation

        p = Principal(type="jwt", sub="user-1")
        self._seed(db_url, "running", p, "conv-A", status="running")
        self._seed(db_url, "expired", p, "conv-A", status="success", is_expired=True)
        self._seed(db_url, "other-conv", p, "conv-B", status="success")

        assert get_latest_report_job_for_conversation("conv-A", p, db_url) is None

    def test_owner_enforced_when_auth_required(self, db_url, monkeypatch):
        from deep_researcher_api.jobs.access import get_latest_report_job_for_conversation

        monkeypatch.setenv("REQUIRE_AUTH", "true")
        owner = Principal(type="jwt", sub="user-1")
        self._seed(db_url, "owned-by-1", owner, "conv-A", status="success")

        intruder = Principal(type="jwt", sub="user-2")
        assert get_latest_report_job_for_conversation("conv-A", intruder, db_url) is None
        assert get_latest_report_job_for_conversation("conv-A", owner, db_url) == "owned-by-1"

    def test_owner_not_enforced_under_no_auth(self, db_url, monkeypatch):
        """Mirrors authorize_job_access: under REQUIRE_AUTH=false ownership is not enforced;
        conversation_id is the only boundary."""
        from deep_researcher_api.jobs.access import get_latest_report_job_for_conversation

        monkeypatch.setenv("REQUIRE_AUTH", "false")
        owner = Principal(type="anonymous", sub="anonymous")
        self._seed(db_url, "anon-job", owner, "conv-A", status="success")

        # A different synthesized principal still resolves the conversation's report.
        other = Principal(type="internal", sub="internal")
        assert get_latest_report_job_for_conversation("conv-A", other, db_url) == "anon-job"

    def test_none_for_empty_or_unknown_conversation(self, db_url):
        from deep_researcher_api.jobs.access import get_latest_report_job_for_conversation

        p = Principal(type="jwt", sub="user-1")
        assert get_latest_report_job_for_conversation("", p, db_url) is None
        assert get_latest_report_job_for_conversation("nope", p, db_url) is None

    def test_cleanup_removes_expired_and_orphaned_access_rows(self, db_url):
        ensure_job_access_table(db_url)
        principal = Principal(type="jwt", sub="user-1")
        create_job_access("live-job", principal, db_url)
        create_job_access("expired-job", principal, db_url)
        create_job_access("orphan-job", principal, db_url)

        _insert_job_info(db_url, "live-job", is_expired=False)
        _insert_job_info(db_url, "expired-job", is_expired=True)

        deleted = cleanup_job_access(db_url)

        assert deleted == 2
        assert get_job_access("live-job", db_url) is not None
        assert get_job_access("expired-job", db_url) is None
        assert get_job_access("orphan-job", db_url) is None


class TestAuthorizeJobAccess:
    @pytest.fixture(autouse=True)
    def _enforce_auth(self, monkeypatch):
        # authorize_job_access only consults the ownership table when REQUIRE_AUTH=true.
        # Pin it on so these tests deterministically exercise the enforced path instead
        # of silently passing/failing based on the ambient environment.
        monkeypatch.setenv("REQUIRE_AUTH", "true")

    def test_no_auth_mode_allows_any_caller(self, db_url, monkeypatch):
        """With auth disabled, ownership is intentionally not enforced (documented posture)."""
        monkeypatch.setenv("REQUIRE_AUTH", "false")
        _insert_job_info(db_url, "job-1")
        # No job_access row and a different principal -> still allowed under no-auth.
        job = SimpleNamespace(job_id="job-1", status="running", created_at=None, error=None)
        job_store = SimpleNamespace(get_job=AsyncMock(return_value=job))

        result = asyncio.run(authorize_job_access(job_store, db_url, "job-1", Principal(type="anonymous", sub="anon")))

        assert result is job

    def test_owner_can_access_job(self, db_url):
        _insert_job_info(db_url, "job-1")
        principal = Principal(type="jwt", sub="user-1")
        create_job_access("job-1", principal, db_url)
        job = SimpleNamespace(job_id="job-1", status="running", created_at=None, error=None)
        job_store = SimpleNamespace(get_job=AsyncMock(return_value=job))

        result = asyncio.run(authorize_job_access(job_store, db_url, "job-1", principal))

        assert result is job

    def test_cross_user_access_denied_with_404(self, db_url):
        _insert_job_info(db_url, "job-1")
        create_job_access("job-1", Principal(type="jwt", sub="user-1"), db_url)
        job = SimpleNamespace(job_id="job-1", status="running", created_at=None, error=None)
        job_store = SimpleNamespace(get_job=AsyncMock(return_value=job))

        with pytest.raises(HTTPException) as exc:
            asyncio.run(authorize_job_access(job_store, db_url, "job-1", Principal(type="jwt", sub="user-2")))

        assert exc.value.status_code == 404

    def test_missing_access_row_denied_with_404(self, db_url):
        _insert_job_info(db_url, "job-1")
        job = SimpleNamespace(job_id="job-1", status="running", created_at=None, error=None)
        job_store = SimpleNamespace(get_job=AsyncMock(return_value=job))

        with pytest.raises(HTTPException) as exc:
            asyncio.run(authorize_job_access(job_store, db_url, "job-1", Principal(type="jwt", sub="user-1")))

        assert exc.value.status_code == 404

    def test_missing_job_returns_404_before_access_check(self, db_url):
        job_store = SimpleNamespace(get_job=AsyncMock(return_value=None))

        with pytest.raises(HTTPException) as exc:
            asyncio.run(authorize_job_access(job_store, db_url, "job-1", Principal(type="jwt", sub="user-1")))

        assert exc.value.status_code == 404


class TestRequireVerifiedPrincipal:
    """The report follow-up surfaces depend on this resolving correctly per REQUIRE_AUTH."""

    def test_no_auth_synthesizes_a_usable_principal(self, monkeypatch):
        from deep_researcher_api.jobs.access import require_verified_principal

        monkeypatch.setenv("REQUIRE_AUTH", "false")
        principal = require_verified_principal()
        # No 401/403 in no-auth mode; a stable, non-None principal is returned so
        # chat/HTTP report follow-up works for anonymous callers.
        assert principal is not None
        assert principal.sub

    def test_auth_required_without_verified_principal_raises_403(self, monkeypatch):
        from deep_researcher_api.jobs.access import require_verified_principal

        monkeypatch.setenv("REQUIRE_AUTH", "true")
        with pytest.raises(HTTPException) as exc:
            require_verified_principal()
        assert exc.value.status_code == 403
