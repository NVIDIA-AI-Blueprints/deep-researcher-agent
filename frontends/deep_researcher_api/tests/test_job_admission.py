# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission-control invariants for asynchronous deep-research jobs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_INPUT_CHARS
from deep_researcher_agent.auth import Principal
from deep_researcher_api.jobs import access as job_access
from deep_researcher_api.jobs import admission
from deep_researcher_api.jobs.admission import DeepResearchAdmissionLimits
from deep_researcher_api.jobs.admission import JobAdmissionUnavailableError
from deep_researcher_api.jobs.admission import JobGlobalCapacityExceededError
from deep_researcher_api.jobs.admission import JobInputTooLargeError
from deep_researcher_api.jobs.admission import JobPrincipalCapacityExceededError
from deep_researcher_api.jobs.admission import JobSubmissionRateExceededError
from deep_researcher_api.jobs.admission import release_deep_research_job_reservation
from deep_researcher_api.jobs.admission import reserve_deep_research_job
from deep_researcher_api.jobs.admission import validate_deep_research_input
from deep_researcher_api.jobs.event_store import EventStore


@pytest.fixture
def db_url(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'test_job_admission.db'}"
    engine = EventStore._get_or_create_sync_engine(url)
    with engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE job_info (  job_id TEXT PRIMARY KEY,  status TEXT,  is_expired BOOLEAN DEFAULT 0)")
        )
        conn.commit()
    return url


@pytest.fixture(autouse=True)
def clear_schema_caches():
    admission._schema_initialized.clear()
    job_access._job_access_schema_initialized.clear()
    yield
    admission._schema_initialized.clear()
    job_access._job_access_schema_initialized.clear()


def _limits(**overrides) -> DeepResearchAdmissionLimits:
    values = {
        "max_input_chars": DEFAULT_MAX_RESEARCH_INPUT_CHARS,
        "max_active_per_principal": 5,
        "max_active_global": 50,
        "max_submissions_per_minute": 20,
        "reservation_ttl_seconds": 30,
    }
    values.update(overrides)
    return DeepResearchAdmissionLimits(**values)


def _set_job_status(db_url: str, job_id: str, status: str, *, is_expired: bool = False) -> None:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO job_info (job_id, status, is_expired) VALUES (:job_id, :status, :is_expired) "
                "ON CONFLICT(job_id) DO UPDATE SET status = excluded.status, is_expired = excluded.is_expired"
            ),
            {"job_id": job_id, "status": status, "is_expired": is_expired},
        )
        conn.commit()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param("4096", 4096, id="operator-can-lower"),
        pytest.param(str(DEFAULT_MAX_RESEARCH_INPUT_CHARS + 1), DEFAULT_MAX_RESEARCH_INPUT_CHARS, id="hard-cap"),
        pytest.param("0", DEFAULT_MAX_RESEARCH_INPUT_CHARS, id="non-positive-falls-back"),
        pytest.param("invalid", DEFAULT_MAX_RESEARCH_INPUT_CHARS, id="invalid-falls-back"),
    ],
)
def test_input_limit_never_exceeds_shared_hard_contract(monkeypatch, configured, expected):
    monkeypatch.setenv("DEEP_RESEARCHER_MAX_DEEP_RESEARCH_INPUT_CHARS", configured)

    assert DeepResearchAdmissionLimits.from_env().max_input_chars == expected


def test_input_limit_rejects_before_submission():
    limits = _limits(max_input_chars=4)

    with pytest.raises(JobInputTooLargeError) as exc_info:
        validate_deep_research_input("12345", limits)

    assert exc_info.value.status_code == 413
    assert exc_info.value.max_chars == 4


@pytest.mark.asyncio
async def test_per_principal_active_limit_is_enforced(db_url):
    principal = Principal(type="jwt", sub="alice")
    limits = _limits(max_active_per_principal=1)
    await reserve_deep_research_job(db_url=db_url, job_id="job-1", principal=principal, limits=limits)

    with pytest.raises(JobPrincipalCapacityExceededError):
        await reserve_deep_research_job(db_url=db_url, job_id="job-2", principal=principal, limits=limits)


@pytest.mark.asyncio
async def test_global_active_limit_is_enforced_across_principals(db_url):
    limits = _limits(max_active_per_principal=10, max_active_global=1)
    await reserve_deep_research_job(
        db_url=db_url,
        job_id="job-1",
        principal=Principal(type="jwt", sub="alice"),
        limits=limits,
    )

    with pytest.raises(JobGlobalCapacityExceededError):
        await reserve_deep_research_job(
            db_url=db_url,
            job_id="job-2",
            principal=Principal(type="jwt", sub="bob"),
            limits=limits,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_type", "blocks_deep_research"),
    [
        pytest.param("deep_researcher", True, id="deep-research-counts"),
        pytest.param("shallow_researcher", False, id="unrelated-agent-does-not-count"),
    ],
)
async def test_global_active_limit_counts_only_deep_research_jobs(db_url, agent_type, blocks_deep_research):
    """Existing job metadata consumes deep-research capacity only for deep-research jobs."""
    _set_job_status(db_url, "existing-job", "running")
    job_access.create_job_access(
        "existing-job",
        Principal(type="jwt", sub="existing-owner"),
        db_url,
        agent_type=agent_type,
    )
    reserve = reserve_deep_research_job(
        db_url=db_url,
        job_id="new-deep-job",
        principal=Principal(type="jwt", sub="new-owner"),
        limits=_limits(max_active_per_principal=10, max_active_global=1),
    )

    if blocks_deep_research:
        with pytest.raises(JobGlobalCapacityExceededError):
            await reserve
    else:
        await reserve


@pytest.mark.asyncio
async def test_global_active_limit_counts_missing_access_metadata_conservatively(db_url):
    """Unclassifiable active jobs fail closed instead of bypassing the deployment ceiling."""
    _set_job_status(db_url, "legacy-active-job", "running")

    with pytest.raises(JobGlobalCapacityExceededError):
        await reserve_deep_research_job(
            db_url=db_url,
            job_id="new-deep-job",
            principal=Principal(type="jwt", sub="new-owner"),
            limits=_limits(max_active_per_principal=10, max_active_global=1),
        )


@pytest.mark.asyncio
async def test_per_principal_limit_keeps_principals_independent(db_url):
    limits = _limits(max_active_per_principal=1, max_active_global=10)

    await reserve_deep_research_job(
        db_url=db_url,
        job_id="job-alice",
        principal=Principal(type="jwt", sub="alice"),
        limits=limits,
    )
    await reserve_deep_research_job(
        db_url=db_url,
        job_id="job-bob",
        principal=Principal(type="jwt", sub="bob"),
        limits=limits,
    )


@pytest.mark.asyncio
async def test_completed_job_still_consumes_sliding_rate_budget(db_url):
    principal = Principal(type="jwt", sub="alice")
    limits = _limits(max_submissions_per_minute=1)
    await reserve_deep_research_job(db_url=db_url, job_id="job-1", principal=principal, limits=limits)
    _set_job_status(db_url, "job-1", "success")

    with pytest.raises(JobSubmissionRateExceededError):
        await reserve_deep_research_job(db_url=db_url, job_id="job-2", principal=principal, limits=limits)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "is_expired"),
    [
        pytest.param("success", False, id="terminal"),
        pytest.param("running", True, id="expired"),
    ],
)
async def test_terminal_or_expired_job_releases_active_capacity(db_url, status, is_expired):
    principal = Principal(type="jwt", sub="alice")
    limits = _limits(max_active_per_principal=1, max_submissions_per_minute=100)
    await reserve_deep_research_job(db_url=db_url, job_id="job-1", principal=principal, limits=limits)
    _set_job_status(db_url, "job-1", status, is_expired=is_expired)

    await reserve_deep_research_job(db_url=db_url, job_id="job-2", principal=principal, limits=limits)


@pytest.mark.asyncio
async def test_release_on_submission_failure_frees_capacity(db_url):
    principal = Principal(type="jwt", sub="alice")
    limits = _limits(max_active_per_principal=1)
    token = await reserve_deep_research_job(db_url=db_url, job_id="job-1", principal=principal, limits=limits)

    await release_deep_research_job_reservation(db_url=db_url, job_id="job-1", reservation_token=token)
    await reserve_deep_research_job(db_url=db_url, job_id="job-2", principal=principal, limits=limits)


@pytest.mark.asyncio
async def test_stale_submitter_cannot_release_reused_job_id_reservation(db_url, monkeypatch):
    """An expired submitter must not delete a newer reservation for the same ID."""
    principal = Principal(type="jwt", sub="alice")
    limits = _limits(max_submissions_per_minute=100, reservation_ttl_seconds=1)
    old_token = await reserve_deep_research_job(
        db_url=db_url,
        job_id="reused-job",
        principal=principal,
        limits=limits,
    )

    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE deep_research_admission "
                "SET admitted_at = 0, reservation_expires_at = 0 "
                "WHERE job_id = 'reused-job'"
            )
        )
        conn.commit()

    monkeypatch.setattr(admission, "time", SimpleNamespace(time=lambda: 100.0))
    new_token = await reserve_deep_research_job(
        db_url=db_url,
        job_id="reused-job",
        principal=principal,
        limits=limits,
    )
    assert new_token != old_token

    released = await release_deep_research_job_reservation(
        db_url=db_url,
        job_id="reused-job",
        reservation_token=old_token,
    )

    assert released is False
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT reservation_token FROM deep_research_admission WHERE job_id = 'reused-job'")
            ).scalar_one()
            == new_token
        )


@pytest.mark.asyncio
async def test_concurrent_sqlite_submissions_cannot_oversubscribe(db_url):
    principal = Principal(type="jwt", sub="alice")
    limits = _limits(max_active_per_principal=1, max_submissions_per_minute=100)

    async def attempt(index: int) -> bool:
        try:
            await reserve_deep_research_job(
                db_url=db_url,
                job_id=f"job-{index}",
                principal=principal,
                limits=limits,
            )
        except JobPrincipalCapacityExceededError:
            return False
        return True

    admitted = await asyncio.gather(*(attempt(index) for index in range(8)))

    assert sum(admitted) == 1


@pytest.mark.asyncio
async def test_stalled_submit_renews_lease_and_keeps_capacity_reserved(db_url, monkeypatch):
    """A healthy-but-slow submitter remains counted after its original TTL."""
    import nat.front_ends.fastapi.async_jobs.job_store as job_store_module
    from deep_researcher_api.jobs import submit

    started = asyncio.Event()
    finish = asyncio.Event()
    limits = _limits(
        max_active_per_principal=1,
        max_active_global=10,
        max_submissions_per_minute=100,
        reservation_ttl_seconds=0.15,
    )

    class _StalledJobStore:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_job_id(self, job_id):
            return job_id

        async def submit_job(self, *args, **kwargs):
            started.set()
            await finish.wait()

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setenv("NAT_JOB_STORE_DB_URL", db_url)
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setattr(
        submit,
        "get_agent_config",
        lambda _agent_type: SimpleNamespace(
            public=True,
            class_path="example.DeepResearcher",
            config_name="deep_research_agent",
        ),
    )
    monkeypatch.setattr(submit, "validate_deep_research_input", lambda _input: limits)
    monkeypatch.setattr(job_store_module, "JobStore", _StalledJobStore)

    submit_task = asyncio.create_task(
        submit.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="alice@example.com",
            principal=Principal(type="jwt", sub="alice"),
            job_id="stalled-job",
            skip_encryption_readiness_check=True,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.sleep(0.4)

    with pytest.raises(JobPrincipalCapacityExceededError):
        await reserve_deep_research_job(
            db_url=db_url,
            job_id="second-job",
            principal=Principal(type="anonymous", sub="anonymous"),
            limits=limits,
        )

    finish.set()
    assert await asyncio.wait_for(submit_task, timeout=1) == "stalled-job"


@pytest.mark.asyncio
async def test_unsupported_admission_store_fails_closed():
    with pytest.raises(JobAdmissionUnavailableError):
        await reserve_deep_research_job(
            db_url="unsupported://admission-store",
            job_id="job-1",
            principal=Principal(type="jwt", sub="alice"),
            limits=_limits(),
        )


def test_postgres_admission_uses_transaction_scoped_advisory_lock():
    conn = MagicMock()

    admission._begin_serialized_transaction(conn, "postgresql+psycopg://db")

    statement, params = conn.execute.call_args.args
    assert "pg_advisory_xact_lock" in str(statement)
    assert params == {"lock_id": admission._PG_ADMISSION_LOCK_ID}


def test_no_auth_programmatic_owner_cannot_choose_quota_key(monkeypatch):
    from deep_researcher_api.jobs import submit

    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setattr(submit, "get_current_principal", lambda: None)

    resolved = submit._resolve_admission_principal(Principal(type="anonymous", sub="caller-controlled-owner"))

    assert resolved == Principal(type="anonymous", sub="anonymous")


def test_no_auth_still_shares_anonymous_budget_when_context_has_principal(monkeypatch):
    from deep_researcher_api.jobs import submit

    verified = Principal(type="jwt", sub="verified-user")
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setattr(submit, "get_current_principal", lambda: verified)

    assert submit._resolve_admission_principal(verified) == Principal(type="anonymous", sub="anonymous")


def test_auth_required_uses_verified_principal_for_quota_key(monkeypatch):
    from deep_researcher_api.jobs import submit

    verified = Principal(type="jwt", sub="verified-user")
    monkeypatch.setenv("REQUIRE_AUTH", "true")

    assert submit._resolve_admission_principal(verified) is verified


def _fake_job_store_factory(submit_job):
    class _FakeJobStore:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_job_id(self, job_id):
            return job_id or "generated-id"

        async def submit_job(self, *args, **kwargs):
            return await submit_job(*args, **kwargs)

    return _FakeJobStore


@pytest.fixture
def patched_submission(monkeypatch):
    import nat.front_ends.fastapi.async_jobs.job_store as job_store_module
    from deep_researcher_api.jobs import submit

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setattr(
        submit,
        "get_agent_config",
        lambda _agent_type: SimpleNamespace(
            public=True,
            class_path="example.DeepResearcher",
            config_name="deep_research_agent",
        ),
    )
    monkeypatch.setattr(submit, "get_current_principal", lambda: None)
    monkeypatch.setattr(submit, "create_job_access", MagicMock())
    monkeypatch.setattr(submit, "release_job_access_reservation", MagicMock())
    monkeypatch.setattr(submit, "renew_job_access_reservation", MagicMock(return_value=True))
    monkeypatch.setattr(submit, "reserve_deep_research_job", AsyncMock(return_value="admission-token"))
    monkeypatch.setattr(submit, "renew_deep_research_job_reservation", AsyncMock(return_value=True))
    monkeypatch.setattr(submit, "release_deep_research_job_reservation", AsyncMock())
    return submit, job_store_module


@pytest.mark.asyncio
async def test_submit_agent_job_reserves_before_enqueue(patched_submission, monkeypatch):
    submit, job_store_module = patched_submission
    ordering: list[str] = []

    async def reserve(**_kwargs):
        ordering.append("reserve")
        return "admission-token"

    async def enqueue(*_args, **_kwargs):
        ordering.append("enqueue")

    submit.reserve_deep_research_job.side_effect = reserve
    submit.create_job_access.side_effect = lambda *_args, **_kwargs: ordering.append("access")
    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(enqueue))

    await submit.submit_agent_job(
        agent_type="deep_researcher",
        input_text="query",
        owner="untrusted-owner",
        principal=Principal(type="anonymous", sub="untrusted-owner"),
        job_id="job-1",
        skip_encryption_readiness_check=True,
    )

    assert ordering == ["reserve", "access", "enqueue"]
    assert submit.reserve_deep_research_job.await_args.kwargs["principal"] == Principal(
        type="anonymous",
        sub="anonymous",
    )
    submit.release_deep_research_job_reservation.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_agent_job_rejection_stops_before_enqueue(patched_submission, monkeypatch):
    submit, job_store_module = patched_submission
    enqueue = AsyncMock()
    submit.reserve_deep_research_job.side_effect = JobPrincipalCapacityExceededError
    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(enqueue))

    with pytest.raises(JobPrincipalCapacityExceededError):
        await submit.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="untrusted-owner",
            principal=Principal(type="anonymous", sub="untrusted-owner"),
            job_id="job-1",
            skip_encryption_readiness_check=True,
        )

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_non_deep_agent_bypasses_admission(patched_submission, monkeypatch):
    submit, job_store_module = patched_submission
    enqueue = AsyncMock()
    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(enqueue))

    await submit.submit_agent_job(
        agent_type="shallow_researcher",
        input_text="query",
        owner="untrusted-owner",
        principal=Principal(type="anonymous", sub="untrusted-owner"),
        job_id="job-1",
        skip_encryption_readiness_check=True,
    )

    submit.reserve_deep_research_job.assert_not_awaited()
    submit.create_job_access.assert_called_once()
    enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_agent_job_failure_retains_reservations_when_enqueue_may_have_occurred(
    patched_submission,
    monkeypatch,
):
    submit, job_store_module = patched_submission

    async def fail_enqueue(*_args, **_kwargs):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(fail_enqueue))

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        await submit.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="untrusted-owner",
            principal=Principal(type="anonymous", sub="untrusted-owner"),
            job_id="job-1",
            skip_encryption_readiness_check=True,
        )

    submit.release_job_access_reservation.assert_not_called()
    submit.release_deep_research_job_reservation.assert_not_awaited()


@pytest.mark.asyncio
async def test_submission_lease_loss_cancels_only_inner_submit_task(patched_submission, monkeypatch):
    """Lease failure aborts enqueue without injecting cancellation into the caller."""
    submit, job_store_module = patched_submission
    started = asyncio.Event()
    inner_cancelled = asyncio.Event()

    async def stalled_enqueue(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            inner_cancelled.set()

    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(stalled_enqueue))
    monkeypatch.setattr(
        submit,
        "validate_deep_research_input",
        lambda _input: _limits(reservation_ttl_seconds=0.03),
    )
    submit.renew_job_access_reservation.return_value = False

    with pytest.raises(JobAdmissionUnavailableError, match="lease was lost"):
        await asyncio.wait_for(
            submit.submit_agent_job(
                agent_type="deep_researcher",
                input_text="query",
                owner="untrusted-owner",
                principal=Principal(type="anonymous", sub="untrusted-owner"),
                job_id="job-1",
                skip_encryption_readiness_check=True,
            ),
            timeout=1,
        )

    assert started.is_set()
    assert inner_cancelled.is_set()
    assert asyncio.current_task().cancelling() == 0
    submit.release_job_access_reservation.assert_not_called()
    submit.release_deep_research_job_reservation.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_submission_cancellation_is_preserved(patched_submission, monkeypatch):
    """External cancellation cancels the inner enqueue and remains CancelledError."""
    submit, job_store_module = patched_submission
    started = asyncio.Event()
    inner_cancelled = asyncio.Event()

    async def stalled_enqueue(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            inner_cancelled.set()

    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(stalled_enqueue))

    caller_task = asyncio.create_task(
        submit.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="untrusted-owner",
            principal=Principal(type="anonymous", sub="untrusted-owner"),
            job_id="job-1",
            skip_encryption_readiness_check=True,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    caller_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller_task

    assert inner_cancelled.is_set()
    submit.release_job_access_reservation.assert_not_called()
    submit.release_deep_research_job_reservation.assert_not_awaited()


@pytest.mark.asyncio
async def test_access_persistence_failure_stops_before_enqueue_and_releases_admission(
    patched_submission,
    monkeypatch,
):
    submit, job_store_module = patched_submission
    enqueue = AsyncMock()
    submit.create_job_access.side_effect = RuntimeError("database unavailable")
    monkeypatch.setattr(job_store_module, "JobStore", _fake_job_store_factory(enqueue))

    with pytest.raises(RuntimeError, match="database unavailable"):
        await submit.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="untrusted-owner",
            principal=Principal(type="anonymous", sub="untrusted-owner"),
            job_id="job-1",
            skip_encryption_readiness_check=True,
        )

    enqueue.assert_not_awaited()
    submit.release_job_access_reservation.assert_called_once_with(
        "job-1",
        "admission-token",
        "sqlite:///./data/jobs.db",
    )
    submit.release_deep_research_job_reservation.assert_awaited_once_with(
        db_url="sqlite:///./data/jobs.db",
        job_id="job-1",
        reservation_token="admission-token",
    )
