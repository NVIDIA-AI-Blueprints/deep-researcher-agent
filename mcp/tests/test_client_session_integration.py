# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real MCP 1.28.1 client coverage for the complete asynchronous protocol."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import uuid
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import httpx
import pytest
import uvicorn
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.version import LATEST_PROTOCOL_VERSION
from mcp.types import Implementation
from mcp.types import TextContent

from deep_researcher_agent.agents.chat_researcher.models import RESEARCH_WORKFLOW_FAILURE_ERROR
from deep_researcher_agent.agents.chat_researcher.models import WorkflowFailure
from deep_researcher_agent.agents.chat_researcher.models import WorkflowSuccess
from deep_researcher_mcp.db_url import normalize_postgres_url
from deep_researcher_mcp.db_url import require_test_database_url
from deep_researcher_mcp.job_store import Job
from deep_researcher_mcp.job_store import JobStore
from deep_researcher_mcp.jobs import JobManager
from deep_researcher_mcp.server import MCPRuntime
from deep_researcher_mcp.server import ServerSettings
from mcp import ClientSession

_DB_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPECTED_TOOLS = ["submit_query", "poll_query", "get_final_report"]
_QUESTION = "deterministic background research question"
_ANSWER = "deterministic research answer"


class _ProtocolRunner:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.background_started = asyncio.Event()
        self.background_completed = asyncio.Event()
        self.start_count = 0
        self.stop_count = 0
        self.classify_calls: list[str] = []
        self.run_calls: list[tuple[str, str]] = []

    async def start(self) -> None:
        self.start_count += 1

    async def stop(self) -> None:
        self.stop_count += 1

    async def classify(self, query: str) -> dict[str, dict[str, str]]:
        self.classify_calls.append(query)
        return {
            "user_intent": {"intent": "research"},
            "depth_decision": {"decision": "deep"},
        }

    async def run_query(self, query: str, *, conversation_id: str, depth: str | None = None) -> WorkflowSuccess:
        self.run_calls.append((query, conversation_id))
        self.background_started.set()
        await self.release.wait()
        self.background_completed.set()
        return WorkflowSuccess(result=_ANSWER)


class _ProtocolFailureRunner(_ProtocolRunner):
    async def run_query(self, query: str, *, conversation_id: str, depth: str | None = None) -> WorkflowFailure:
        self.run_calls.append((query, conversation_id))
        self.background_started.set()
        await self.release.wait()
        self.background_completed.set()
        return WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)


class _MemoryJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.init_count = 0
        self.close_count = 0

    @property
    def pool(self) -> None:
        return None

    async def init(self) -> None:
        self.init_count += 1

    async def close(self) -> None:
        self.close_count += 1

    async def create(
        self,
        *,
        principal: str,
        query: str,
        depth: str,
        state: str,
        result: str | None = None,
        ttl_seconds: int = 24 * 3600,
    ) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        self.jobs[job_id] = Job(
            job_id=job_id,
            principal=principal,
            query=query,
            depth=depth,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
            result=result,
            error=None,
            poll_count=0,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        return job_id

    async def mark_running(self, job_id: str, runner_id: str) -> bool:
        job = self.jobs[job_id]
        if job.state != "queued":
            return False
        now = datetime.now(UTC)
        job.state = "running"
        job.runner_id = runner_id
        job.heartbeat_at = now
        job.updated_at = now
        return True

    async def mark_failed_if_queued_or_owned(self, job_id: str, runner_id: str, error: str) -> bool:
        job = self.jobs[job_id]
        if job.state != "queued" and not (job.state == "running" and job.runner_id == runner_id):
            return False
        job.state = "failed"
        job.error = error
        job.updated_at = datetime.now(UTC)
        return True

    async def heartbeat(self, job_id: str, runner_id: str) -> None:
        job = self.jobs[job_id]
        if job.state == "running" and job.runner_id == runner_id:
            now = datetime.now(UTC)
            job.heartbeat_at = now
            job.updated_at = now

    async def update(
        self,
        job_id: str,
        *,
        state: str | None = None,
        result: str | None = None,
        error: str | None = None,
        from_states: tuple[str, ...] | None = None,
        runner_id: str | None = None,
    ) -> bool:
        job = self.jobs[job_id]
        if from_states is not None and job.state not in from_states:
            return False
        if runner_id is not None and job.runner_id != runner_id:
            return False
        if state is not None:
            job.state = state  # type: ignore[assignment]
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        job.updated_at = datetime.now(UTC)
        return True

    async def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def record_poll(self, job_id: str, principal: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is not None and job.principal == principal and job.state in ("queued", "running"):
            job.poll_count += 1
        return job

    async def delete_expired(self) -> int:
        now = datetime.now(UTC)
        expired = [job_id for job_id, job in self.jobs.items() if job.expires_at < now]
        for job_id in expired:
            del self.jobs[job_id]
        return len(expired)

    async def mark_stale_running_failed(self, *, stale_after_seconds: int, error: str) -> int:
        del stale_after_seconds, error
        return 0


def _settings(port: int) -> ServerSettings:
    return ServerSettings(
        host="127.0.0.1",
        port=port,
        path="/mcp",
        workers=1,
        log_level="CRITICAL",
        config_path=Path("/not-used/config.yml"),
        shallow_inline_wait_seconds=0,
        cors_origins=(),
        allowed_hosts=("127.0.0.1", "127.0.0.1:*"),
        allowed_origins=(),
    )


@asynccontextmanager
async def _serve_runtime(runtime_factory) -> AsyncIterator[tuple[MCPRuntime, str]]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    runtime = runtime_factory(port)
    config = uvicorn.Config(
        runtime.app,
        loop="asyncio",
        lifespan="on",
        log_level="critical",
        access_log=False,
        log_config=None,
        ws="none",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve(sockets=[listener]), name="deep-researcher-mcp-test-uvicorn")
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if server_task.done():
                    server_task.result()
                await asyncio.sleep(0.01)
        yield runtime, f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        finally:
            listener.close()


@asynccontextmanager
async def _client_session(
    endpoint: str,
    observed_headers: list[set[str]],
) -> AsyncIterator[tuple[ClientSession, Any, Any]]:
    async def capture_headers(request: httpx.Request) -> None:
        observed_headers.append(set(request.headers))

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(5),
        trust_env=False,
        event_hooks={"request": [capture_headers]},
    ) as http_client:
        async with streamable_http_client(
            endpoint,
            http_client=http_client,
            terminate_on_close=False,
        ) as (read_stream, write_stream, get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=5),
                client_info=Implementation(name="deep-researcher-mcp-integration-test", version="1.0"),
            ) as session:
                initialized = await session.initialize()
                yield session, initialized, get_session_id


def _structured(result) -> dict[str, Any]:
    assert result.isError is False
    assert isinstance(result.structuredContent, dict)
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == result.structuredContent
    return result.structuredContent


async def _exercise_complete_client_flow(
    runtime_factory,
    runner: _ProtocolRunner,
    manager: JobManager,
    store,
) -> str:
    observed_headers: list[set[str]] = []
    async with _serve_runtime(runtime_factory) as (runtime, endpoint):
        async with _client_session(endpoint, observed_headers) as (
            submit_session,
            initialized,
            get_submit_session_id,
        ):
            assert initialized.protocolVersion == LATEST_PROTOCOL_VERSION == "2025-11-25"
            assert initialized.serverInfo.name == "deep_researcher_deep_research"
            assert initialized.serverInfo.version == "1.28.1"
            assert initialized.capabilities.tools is not None
            assert initialized.capabilities.tools.listChanged is False
            assert get_submit_session_id() is None

            listed = await submit_session.list_tools()
            assert [tool.name for tool in listed.tools] == _EXPECTED_TOOLS
            assert listed.nextCursor is None
            for tool in listed.tools:
                assert tool.outputSchema == {
                    "additionalProperties": True,
                    "title": f"{tool.name}DictOutput",
                    "type": "object",
                }

            submitted = _structured(await submit_session.call_tool("submit_query", {"query": _QUESTION}))
            job_id = submitted["job_id"]
            assert submitted == {
                "job_id": job_id,
                "depth": "deep",
                "state": "queued",
                "estimated_duration_seconds": 180,
                "first_poll_after_seconds": 180,
            }
            assert uuid.UUID(job_id).version == 4
            await asyncio.wait_for(runner.background_started.wait(), timeout=5)
            background_task = manager._active_tasks[job_id]
            assert background_task.done() is False

        # The submit call and its entire client/HTTP transport are gone. The
        # process-owned JobManager task and persisted row must still be alive.
        assert background_task.done() is False
        running_job = await store.get(job_id)
        assert running_job is not None
        assert running_job.principal == "anonymous"
        assert running_job.state == "running"

        async with _client_session(endpoint, observed_headers) as (
            poll_session,
            second_initialized,
            get_poll_session_id,
        ):
            assert second_initialized.serverInfo.name == "deep_researcher_deep_research"
            assert get_poll_session_id() is None
            assert _structured(await poll_session.call_tool("poll_query", {"job_id": job_id})) == {
                "job_id": job_id,
                "depth": "deep",
                "state": "running",
                "next_poll_after_seconds": 180,
                "todos": [],
            }
            assert _structured(await poll_session.call_tool("get_final_report", {"job_id": job_id})) == {
                "job_id": job_id,
                "depth": "deep",
                "state": "not_ready",
                "error": "job_not_ready",
            }

            runner.release.set()
            await asyncio.wait_for(runner.background_completed.wait(), timeout=5)
            await asyncio.wait_for(asyncio.shield(background_task), timeout=5)
            assert background_task.cancelled() is False

            assert _structured(await poll_session.call_tool("poll_query", {"job_id": job_id})) == {
                "job_id": job_id,
                "depth": "deep",
                "state": "complete",
                "todos": [],
            }
            assert _structured(await poll_session.call_tool("get_final_report", {"job_id": job_id})) == {
                "job_id": job_id,
                "depth": "deep",
                "state": "complete",
                "result": _ANSWER,
            }

            missing_argument = await poll_session.call_tool("submit_query", {})
            assert missing_argument.isError is True
            assert missing_argument.structuredContent is None
            assert "Field required" in missing_argument.content[0].text

            wrong_type = await poll_session.call_tool("poll_query", {"job_id": 7})
            assert wrong_type.isError is True
            assert wrong_type.structuredContent is None
            assert "valid string" in wrong_type.content[0].text

            unknown_id = str(uuid.uuid4())
            assert _structured(
                await poll_session.call_tool(
                    "poll_query",
                    {"job_id": unknown_id, "unexpected_argument": "ignored"},
                )
            ) == {"state": "not_found", "error": "job_not_found"}

        completed_job = await store.get(job_id)
        assert completed_job is not None
        assert completed_job.state == "complete"
        assert completed_job.result == _ANSWER
        assert completed_job.poll_count == 1
        assert runner.classify_calls == [_QUESTION]
        assert runner.run_calls == [(_QUESTION, job_id)]
        assert runtime.ready is True

    assert runtime.ready is False
    assert runner.start_count == 1
    assert runner.stop_count == 1
    assert observed_headers
    for headers in observed_headers:
        assert "authorization" not in headers
    return job_id


@pytest.mark.asyncio
async def test_real_client_background_job_survives_submit_request() -> None:
    runner = _ProtocolRunner()
    store = _MemoryJobStore()
    manager = JobManager(
        runner,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        runner_id="protocol-test-runner",
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )

    def runtime_factory(port: int) -> MCPRuntime:
        return MCPRuntime(
            _settings(port),
            runner=runner,
            jobs_factory=lambda: manager,
            validate_startup=lambda: None,
        )

    await _exercise_complete_client_flow(runtime_factory, runner, manager, store)

    assert store.init_count == 1
    assert store.close_count == 1


@pytest.mark.asyncio
async def test_real_client_surfaces_structured_workflow_failure() -> None:
    runner = _ProtocolFailureRunner()
    store = _MemoryJobStore()
    manager = JobManager(
        runner,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        runner_id="protocol-failure-runner",
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )

    def runtime_factory(port: int) -> MCPRuntime:
        return MCPRuntime(
            _settings(port),
            runner=runner,
            jobs_factory=lambda: manager,
            validate_startup=lambda: None,
        )

    observed_headers: list[set[str]] = []
    async with _serve_runtime(runtime_factory) as (_runtime, endpoint):
        async with _client_session(endpoint, observed_headers) as (session, _initialized, _get_session_id):
            submitted = _structured(await session.call_tool("submit_query", {"query": _QUESTION}))
            job_id = submitted["job_id"]
            assert submitted == {
                "job_id": job_id,
                "depth": "deep",
                "state": "queued",
                "estimated_duration_seconds": 180,
                "first_poll_after_seconds": 180,
            }

            await asyncio.wait_for(runner.background_started.wait(), timeout=5)
            background_task = manager._active_tasks[job_id]
            runner.release.set()
            await asyncio.wait_for(runner.background_completed.wait(), timeout=5)
            await asyncio.wait_for(asyncio.shield(background_task), timeout=5)

            assert _structured(await session.call_tool("poll_query", {"job_id": job_id})) == {
                "job_id": job_id,
                "depth": "deep",
                "state": "failed",
                "error": RESEARCH_WORKFLOW_FAILURE_ERROR,
                "todos": [],
            }
            assert _structured(await session.call_tool("get_final_report", {"job_id": job_id})) == {
                "job_id": job_id,
                "depth": "deep",
                "state": "failed",
                "error": RESEARCH_WORKFLOW_FAILURE_ERROR,
            }

    failed_job = await store.get(job_id)
    assert failed_job is not None
    assert failed_job.state == "failed"
    assert failed_job.result is None
    assert failed_job.error == RESEARCH_WORKFLOW_FAILURE_ERROR
    assert store.init_count == 1
    assert store.close_count == 1


@pytest.fixture()
async def phase6_postgres_url() -> str:
    db_url = os.getenv("DEEP_RESEARCHER_MCP_TEST_DB_URL")
    if not db_url:
        pytest.skip("set DEEP_RESEARCHER_MCP_TEST_DB_URL to run the real-client Postgres parity test")
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
async def test_real_client_flow_persists_anonymous_job_across_manager_restart(
    phase6_postgres_url: str,
) -> None:
    runner = _ProtocolRunner()
    store = JobStore(phase6_postgres_url, min_pool_size=1, max_pool_size=2)
    manager = JobManager(
        runner,  # type: ignore[arg-type]
        store,
        runner_id="protocol-postgres-runner",
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )

    def runtime_factory(port: int) -> MCPRuntime:
        return MCPRuntime(
            _settings(port),
            runner=runner,
            jobs_factory=lambda: manager,
            validate_startup=lambda: None,
        )

    job_id = await _exercise_complete_client_flow(runtime_factory, runner, manager, store)

    # A fresh manager has no local task but must still retrieve the completed
    # report from the shared ledger after the serving runtime has shut down.
    restarted_store = JobStore(phase6_postgres_url, min_pool_size=1, max_pool_size=1)
    restarted = JobManager(
        _ProtocolRunner(),  # type: ignore[arg-type]
        restarted_store,
        runner_id="protocol-restarted-runner",
        heartbeat_interval_seconds=0,
        ttl_sweep_interval_seconds=0,
        stale_job_after_seconds=3600,
    )
    await restarted.start()
    try:
        assert await restarted.get_final_report(job_id, "anonymous") == {
            "job_id": job_id,
            "depth": "deep",
            "state": "complete",
            "result": _ANSWER,
        }
    finally:
        await restarted.stop()


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


def _maintenance_url(db_url: str) -> tuple[str, str]:
    parts = urlsplit(normalize_postgres_url(db_url, label="DEEP_RESEARCHER_MCP_TEST_DB_URL"))
    db_name = parts.path.lstrip("/") or "postgres"
    maintenance = urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))
    return maintenance, db_name


def _quote_database_name(db_name: str) -> str:
    if _DB_NAME_RE.fullmatch(db_name):
        return f'"{db_name}"'
    return '"' + db_name.replace('"', '""') + '"'
