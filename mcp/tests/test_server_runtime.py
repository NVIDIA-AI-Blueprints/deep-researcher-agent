# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FastMCP transport and worker-lifecycle tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp.shared.version import LATEST_PROTOCOL_VERSION

from deep_researcher_mcp import server


def _settings(tmp_path: Path, **overrides: Any) -> server.ServerSettings:
    values: dict[str, Any] = {
        "host": "0.0.0.0",
        "port": 9001,
        "path": "/mcp",
        "workers": 1,
        "log_level": "INFO",
        "config_path": tmp_path / "config.yml",
        "shallow_inline_wait_seconds": 30.0,
        "cors_origins": ("http://localhost:6274",),
        "allowed_hosts": ("localhost", "localhost:*"),
        "allowed_origins": ("http://localhost:*", "http://localhost:6274"),
    }
    values.update(overrides)
    return server.ServerSettings(**values)


class _Service:
    def __init__(self, name: str, events: list[str], *, fail_start: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start

    async def start(self) -> None:
        self.events.append(f"{self.name}:start")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed to start")

    async def stop(self) -> None:
        self.events.append(f"{self.name}:stop")


class _Jobs(_Service):
    def __init__(self, events: list[str], *, fail_start: bool = False) -> None:
        super().__init__("jobs", events, fail_start=fail_start)
        self.calls: list[tuple[Any, ...]] = []
        self.submit_result: dict[str, Any] = {
            "job_id": "job-1",
            "depth": "deep",
            "state": "queued",
            "estimated_duration_seconds": 180,
            "first_poll_after_seconds": 180,
        }
        self.inline_result: dict[str, Any] | None = None

    async def submit(self, query: str, principal: str) -> dict[str, Any]:
        self.calls.append(("submit", query, principal))
        return dict(self.submit_result)

    async def wait_for_completion(
        self,
        job_id: str,
        principal: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        self.calls.append(("wait", job_id, principal, timeout))
        return self.inline_result

    async def poll(self, job_id: str, principal: str) -> dict[str, Any]:
        self.calls.append(("poll", job_id, principal))
        return {"job_id": job_id, "depth": "deep", "state": "running", "todos": []}

    async def get_final_report(self, job_id: str, principal: str) -> dict[str, Any]:
        self.calls.append(("report", job_id, principal))
        return {"job_id": job_id, "depth": "deep", "state": "complete", "result": "done"}


class _CapabilityJobs(_Jobs):
    JOB_ID = "00000000-0000-4000-8000-000000000001"

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.submit_result = {
            "job_id": self.JOB_ID,
            "depth": "shallow",
            "state": "queued",
            "estimated_duration_seconds": 45,
            "first_poll_after_seconds": 5,
        }

    async def poll(self, job_id: str, principal: str) -> dict[str, Any]:
        self.calls.append(("poll", job_id, principal))
        if job_id != self.JOB_ID:
            return {"state": "not_found", "error": "job_not_found"}
        return {
            "job_id": job_id,
            "depth": "shallow",
            "state": "complete",
            "todos": [],
        }

    async def get_final_report(self, job_id: str, principal: str) -> dict[str, Any]:
        self.calls.append(("report", job_id, principal))
        if job_id != self.JOB_ID:
            return {"state": "not_found", "error": "job_not_found"}
        return {
            "job_id": job_id,
            "depth": "shallow",
            "state": "complete",
            "result": "capability result",
        }


def _initialize_request(request_id: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "deep-researcher-mcp-test", "version": "1.0"},
        },
    }


def _tool_call_request(request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _structured_tool_result(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    assert body["result"]["isError"] is False
    structured = body["result"].get("structuredContent")
    if structured is not None:
        return structured
    return json.loads(body["result"]["content"][0]["text"])


_MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


_BOUNDARY_FAILURE_CREDENTIAL = "mcp-boundary-credential-sentinel"  # pragma: allowlist secret
_BOUNDARY_FAILURE_HOST = "internal-db.sentinel.invalid"
_BOUNDARY_FAILURE_MESSAGE = f"postgresql://service:{_BOUNDARY_FAILURE_CREDENTIAL}@{_BOUNDARY_FAILURE_HOST}/jobs"


def test_settings_defaults_and_overrides(tmp_path: Path) -> None:
    defaults = server.ServerSettings.from_env({})
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 9001
    assert defaults.path == "/mcp"
    assert defaults.workers == 1
    assert defaults.log_level == "INFO"
    assert defaults.cors_origins == ("http://localhost:6274",)
    assert defaults.max_query_chars == 8000
    assert "localhost:*" in defaults.allowed_hosts
    assert "127.0.0.1:*" in defaults.allowed_hosts
    assert "http://localhost:*" in defaults.allowed_origins
    assert "http://localhost:6274" in defaults.allowed_origins

    config_path = tmp_path / "workflow.yml"
    overridden = server.ServerSettings.from_env(
        {
            "DEEP_RESEARCHER_MCP_HOST": "127.0.0.1",
            "DEEP_RESEARCHER_MCP_PORT": "9100",
            "DEEP_RESEARCHER_MCP_PATH": "/legacy/research/mcp/",
            "DEEP_RESEARCHER_MCP_WORKERS": "3",
            "DEEP_RESEARCHER_MCP_LOG_LEVEL": "warning",
            "DEEP_RESEARCHER_MCP_CONFIG": str(config_path),
            "DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS": "4.5",
            "DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS": "2000",
            "DEEP_RESEARCHER_MCP_CORS_ORIGINS": "http://localhost:6274, https://inspector.example",
            "DEEP_RESEARCHER_MCP_ALLOWED_HOSTS": "research.example.com,research.example.com:*",
            "DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS": "https://research.example.com",
        }
    )
    assert overridden == server.ServerSettings(
        host="127.0.0.1",
        port=9100,
        path="/legacy/research/mcp",
        workers=3,
        log_level="WARNING",
        config_path=config_path.resolve(),
        shallow_inline_wait_seconds=4.5,
        max_query_chars=2000,
        cors_origins=("http://localhost:6274", "https://inspector.example"),
        allowed_hosts=("research.example.com", "research.example.com:*"),
        allowed_origins=(
            "https://research.example.com",
            "http://localhost:6274",
            "https://inspector.example",
        ),
    )
    assert server.ServerSettings.from_env({"DEEP_RESEARCHER_MCP_CORS_ORIGINS": ""}).cors_origins == ()


def test_installed_layout_disables_checkout_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed package must not point defaults at site-packages paths."""
    site_packages = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    module_path = site_packages / "deep_researcher_mcp" / "server.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("")
    monkeypatch.setattr(server, "__file__", str(module_path))

    assert server._find_source_checkout_root() is None

    monkeypatch.setattr(server, "DEFAULT_CONFIG", None)
    with pytest.raises(ValueError, match="DEEP_RESEARCHER_MCP_CONFIG must point to a workflow config"):
        server.ServerSettings.from_env({})

    explicit = tmp_path / "workflow.yml"
    assert (
        server.ServerSettings.from_env({"DEEP_RESEARCHER_MCP_CONFIG": str(explicit)}).config_path == explicit.resolve()
    )

    monkeypatch.delenv("DEEP_RESEARCHER_MCP_ENV_FILE", raising=False)
    monkeypatch.setattr(server, "_DEFAULT_ENV_FILE", None)
    server._load_env_file()


def test_source_checkout_defaults_resolve_to_repo_paths() -> None:
    assert server._REPO_ROOT is not None
    assert server.DEFAULT_CONFIG == server._REPO_ROOT / "configs" / "config_mcp.yml"
    assert server.DEFAULT_CONFIG.is_file()
    assert server._DEFAULT_ENV_FILE == server._REPO_ROOT / "deploy" / ".env"


def test_load_env_file_uses_public_name_without_overriding_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "mcp.env"
    env_file.write_text(
        "PHASE4_FROM_FILE=loaded\nPHASE4_EXISTING=file-value\n"  # pragma: allowlist secret
    )
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_ENV_FILE", str(env_file))
    monkeypatch.setenv("PHASE4_EXISTING", "process-value")
    monkeypatch.delenv("PHASE4_FROM_FILE", raising=False)

    server._load_env_file()

    assert server.os.environ["PHASE4_FROM_FILE"] == "loaded"
    assert server.os.environ["PHASE4_EXISTING"] == "process-value"


def test_load_env_file_warns_for_missing_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    missing = tmp_path / "missing.env"
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_ENV_FILE", str(missing))

    server._load_env_file()

    assert str(missing) in caplog.text


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"DEEP_RESEARCHER_MCP_HOST": " "}, "DEEP_RESEARCHER_MCP_HOST"),
        ({"DEEP_RESEARCHER_MCP_PORT": "0"}, "DEEP_RESEARCHER_MCP_PORT"),
        ({"DEEP_RESEARCHER_MCP_PORT": "65536"}, "DEEP_RESEARCHER_MCP_PORT"),
        ({"DEEP_RESEARCHER_MCP_PORT": "not-a-port"}, "DEEP_RESEARCHER_MCP_PORT"),
        ({"DEEP_RESEARCHER_MCP_PATH": "mcp"}, "DEEP_RESEARCHER_MCP_PATH"),
        ({"DEEP_RESEARCHER_MCP_PATH": "/"}, "DEEP_RESEARCHER_MCP_PATH"),
        ({"DEEP_RESEARCHER_MCP_PATH": "//"}, "DEEP_RESEARCHER_MCP_PATH"),
        ({"DEEP_RESEARCHER_MCP_PATH": "/health"}, "reserved route"),
        ({"DEEP_RESEARCHER_MCP_PATH": "/live/"}, "reserved route"),
        ({"DEEP_RESEARCHER_MCP_PATH": "/mcp?debug=1"}, "DEEP_RESEARCHER_MCP_PATH"),
        ({"DEEP_RESEARCHER_MCP_PATH": "/mcp/{tenant}"}, "literal path"),
        ({"DEEP_RESEARCHER_MCP_WORKERS": "0"}, "DEEP_RESEARCHER_MCP_WORKERS"),
        ({"DEEP_RESEARCHER_MCP_LOG_LEVEL": "verbose"}, "DEEP_RESEARCHER_MCP_LOG_LEVEL"),
        ({"DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS": "-1"}, "DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS"),
        ({"DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS": "nan"}, "finite"),
        ({"DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS": "0"}, "DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS"),
        ({"DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS": "not-a-count"}, "DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS"),
        ({"DEEP_RESEARCHER_MCP_ALLOWED_HOSTS": ""}, "DEEP_RESEARCHER_MCP_ALLOWED_HOSTS"),
    ],
)
def test_settings_reject_invalid_values(environment: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        server.ServerSettings.from_env(environment)


def test_validate_startup_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("functions: {}\n")
    settings = _settings(tmp_path, config_path=config_path)
    monkeypatch.setenv("DEEP_RESEARCHER_CHECKPOINT_DB", "postgresql+asyncpg://db.example/deep_researcher_jobs")

    server._validate_startup_configuration(settings)

    assert server._resolve_checkpoint_db_url() == "postgresql://db.example/deep_researcher_jobs"
    assert server.os.environ["DEEP_RESEARCHER_CHECKPOINT_DB"] == "postgresql://db.example/deep_researcher_jobs"


def test_validate_startup_configuration_requires_config_and_postgres(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.delenv("DEEP_RESEARCHER_CHECKPOINT_DB", raising=False)

    with pytest.raises(ValueError, match="config does not exist"):
        server._validate_startup_configuration(settings)

    settings.config_path.write_text("functions: {}\n")
    with pytest.raises(ValueError, match="DEEP_RESEARCHER_CHECKPOINT_DB"):
        server._validate_startup_configuration(settings)


def test_job_manager_uses_one_normalized_checkpoint_dsn_for_ledger_and_todos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_CHECKPOINT_DB", "postgresql+asyncpg://db.example/deep_researcher_jobs")
    runtime = server.MCPRuntime(
        _settings(tmp_path),
        runner=_Service("runner", []),
        validate_startup=lambda: None,
    )

    manager = runtime._create_job_manager()

    assert manager._store._db_url == "postgresql://db.example/deep_researcher_jobs"
    assert manager._checkpoint_todo_reader._db_url == "postgresql://db.example/deep_researcher_jobs"
    assert manager._store._schema == "public"
    assert manager._checkpoint_todo_reader._checkpoints_table == '"public".checkpoints'


@pytest.mark.asyncio
async def test_fastmcp_settings_and_exact_tool_schemas(tmp_path: Path) -> None:
    runtime = server.MCPRuntime(_settings(tmp_path), validate_startup=lambda: None)

    assert runtime.mcp.name == "deep_researcher_deep_research"
    assert runtime.mcp.settings.host == "0.0.0.0"
    assert runtime.mcp.settings.port == 9001
    assert runtime.mcp.settings.streamable_http_path == "/mcp"
    assert runtime.mcp.settings.stateless_http is True
    assert runtime.mcp.settings.json_response is True
    assert runtime.mcp.settings.lifespan is None
    assert runtime.mcp.settings.auth is None
    assert runtime.mcp._token_verifier is None
    assert runtime.mcp._auth_server_provider is None
    assert runtime.mcp.settings.transport_security is not None
    assert runtime.mcp.settings.transport_security.enable_dns_rebinding_protection is True
    assert runtime.mcp.settings.transport_security.allowed_hosts == ["localhost", "localhost:*"]
    assert runtime.mcp.settings.transport_security.allowed_origins == [
        "http://localhost:*",
        "http://localhost:6274",
    ]
    for instruction in (
        "If submit_query returns state='complete', use result directly",
        "If submit_query returns state='queued', wait first_poll_after_seconds",
        "While poll_query returns state='queued' or state='running'",
        "wait next_poll_after_seconds before polling again",
        "If poll_query returns state='complete', call get_final_report(job_id)",
        "If poll_query returns state='failed' or state='not_found', stop polling",
        "Deep jobs always queue",
        "Shallow jobs may complete inline within 30 seconds",
        "poll_query is status-only",
        "fixed 180-second polling cadence",
        "todos=[] is normal and not an error",
        "Do not run separate web or paper research",
        "No Authorization header",
        "shared anonymous principal",
        "opaque bearer capability",
        "Anyone who possesses it",
    ):
        assert instruction in runtime.mcp.instructions

    listed_tools = await runtime.mcp.list_tools()
    assert [tool.name for tool in listed_tools] == ["submit_query", "poll_query", "get_final_report"]
    tools = {tool.name: tool for tool in listed_tools}
    assert set(tools) == {"submit_query", "poll_query", "get_final_report"}
    assert tools["submit_query"].inputSchema == {
        "properties": {"query": {"title": "Query", "type": "string"}},
        "required": ["query"],
        "title": "submit_queryArguments",
        "type": "object",
    }
    for name in ("poll_query", "get_final_report"):
        assert tools[name].inputSchema == {
            "properties": {"job_id": {"title": "Job Id", "type": "string"}},
            "required": ["job_id"],
            "title": f"{name}Arguments",
            "type": "object",
        }
    for name, tool in tools.items():
        assert tool.outputSchema == {
            "additionalProperties": True,
            "title": f"{name}DictOutput",
            "type": "object",
        }
        assert tool.annotations is None
        assert tool.title is None
        assert tool.meta is None
        assert tool.icons is None
        assert tool.execution is None

    schema_contract = []
    for tool in listed_tools:
        dumped = tool.model_dump(mode="json", by_alias=True, exclude_none=False)
        schema_contract.append(
            {
                field: dumped.get(field)
                for field in (
                    "name",
                    "inputSchema",
                    "outputSchema",
                    "annotations",
                    "title",
                    "_meta",
                    "icons",
                    "execution",
                )
            }
        )
    serialized_contract = json.dumps(schema_contract, sort_keys=True, separators=(",", ":")).encode()
    # Frozen public compatibility contract for FastMCP 1.28.1. Descriptions
    # use semantic assertions below so anonymous-capability wording stays clear.
    assert hashlib.sha256(serialized_contract).hexdigest() == (
        "81eba67fadd56e64b58a84b700b202841f8636c93c6cbf63752507c8bf5ca96a"  # pragma: allowlist secret
    )

    descriptions = {name: " ".join(tool.description.split()) for name, tool in tools.items()}
    expected_description_fragments = {
        "submit_query": (
            "Deep queries always queue",
            'state="complete": use result directly; do not call get_final_report',
            'state="queued": wait first_poll_after_seconds',
            'state="queued" or state="running"',
            "poll_query is status-only",
            "Do not run independent web or paper research",
            "return non-empty todos",
            "opaque bearer capability",
        ),
        "poll_query": (
            "status-only",
            "does not return the final report body",
            "wait next_poll_after_seconds",
            "fixed 180-second polling cadence",
            'state is "complete", stop polling and call get_final_report',
            'state is "failed" or "not_found", stop polling',
            "todos=[] is normal",
            "Do not run independent web or paper research",
            "opaque capability UUID",
        ),
        "get_final_report": (
            'only after poll_query(job_id) returns state="complete"',
            'Inline state="complete" responses from submit_query already include result',
            'state="not_ready" with error="job_not_ready"',
            'state="failed" with error',
            'returns state="not_found"',
            "possession of",
            "controls access",
        ),
    }
    for tool_name, fragments in expected_description_fragments.items():
        for fragment in fragments:
            assert fragment in descriptions[tool_name]

    assert list(inspect.signature(runtime.submit_query).parameters) == ["ctx", "query"]
    assert list(inspect.signature(runtime.poll_query).parameters) == ["ctx", "job_id"]
    assert list(inspect.signature(runtime.get_final_report).parameters) == ["ctx", "job_id"]


@pytest.mark.asyncio
async def test_outer_lifespan_owns_services_once_across_stateless_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runner = _Service("runner", events)
    jobs = _Jobs(events)
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=runner,
        jobs_factory=lambda: jobs,
        validate_startup=lambda: events.append("validate"),
    )

    session_manager = runtime.mcp.session_manager
    original_run = session_manager.run

    @asynccontextmanager
    async def counted_session_manager():
        events.append("session:start")
        async with original_run():
            yield
        events.append("session:stop")

    monkeypatch.setattr(session_manager, "run", counted_session_manager)

    transport = httpx.ASGITransport(app=runtime.app)
    async with runtime.app.router.lifespan_context(runtime.app):
        assert events == ["validate", "runner:start", "jobs:start", "session:start"]
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            first = await client.post("/mcp", headers=_MCP_HEADERS, json=_initialize_request(1))
            second = await client.post("/mcp", headers=_MCP_HEADERS, json=_initialize_request(2))
            listed = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
            )
            called = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "poll_query", "arguments": {"job_id": "job-1"}},
                },
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert listed.status_code == 200
        assert called.status_code == 200
        assert first.headers["content-type"].startswith("application/json")
        assert "mcp-session-id" not in first.headers
        assert first.json()["result"]["serverInfo"]["name"] == "deep_researcher_deep_research"
        assert {tool["name"] for tool in listed.json()["result"]["tools"]} == {
            "submit_query",
            "poll_query",
            "get_final_report",
        }
        assert called.json()["result"]["isError"] is False
        assert jobs.calls == [("poll", "job-1", "anonymous")]
        assert events == ["validate", "runner:start", "jobs:start", "session:start"]

    assert events == [
        "validate",
        "runner:start",
        "jobs:start",
        "session:start",
        "session:stop",
        "jobs:stop",
        "runner:stop",
    ]


@pytest.mark.asyncio
async def test_anonymous_uuid_capability_works_across_stateless_no_auth_clients(tmp_path: Path) -> None:
    events: list[str] = []
    jobs = _CapabilityJobs(events)
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    captured_headers: list[dict[bytes, bytes]] = []

    async def capture_headers_app(scope, receive, send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST":
            captured_headers.append(dict(scope["headers"]))
        await runtime.app(scope, receive, send)

    transport = httpx.ASGITransport(app=capture_headers_app)
    unknown_capability = str(uuid.uuid4())

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as submit_client:
            initialized = await submit_client.post("/mcp", headers=_MCP_HEADERS, json=_initialize_request(1))
            listed = await submit_client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            submitted = await submit_client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(3, "submit_query", {"query": "question"}),
            )

        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as poll_client:
            polled = await poll_client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(4, "poll_query", {"job_id": _CapabilityJobs.JOB_ID}),
            )

        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as report_client:
            reported = await report_client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(5, "get_final_report", {"job_id": _CapabilityJobs.JOB_ID}),
            )
            unknown = await report_client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(6, "poll_query", {"job_id": unknown_capability}),
            )
            spoofed = await report_client.post(
                "/mcp",
                headers={
                    **_MCP_HEADERS,
                    "authorization": "Bearer ignored",
                },
                json=_tool_call_request(7, "get_final_report", {"job_id": _CapabilityJobs.JOB_ID}),
            )

    responses = (initialized, listed, submitted, polled, reported, unknown, spoofed)
    assert all(response.status_code == 200 for response in responses)
    assert all("mcp-session-id" not in response.headers for response in responses)
    assert {tool["name"] for tool in listed.json()["result"]["tools"]} == {
        "submit_query",
        "poll_query",
        "get_final_report",
    }

    submitted_result = _structured_tool_result(submitted)
    assert submitted_result["job_id"] == _CapabilityJobs.JOB_ID
    assert uuid.UUID(submitted_result["job_id"]).version == 4
    assert submitted_result["state"] == "queued"
    assert _structured_tool_result(polled)["state"] == "complete"
    assert _structured_tool_result(reported)["result"] == "capability result"
    assert _structured_tool_result(unknown) == {"state": "not_found", "error": "job_not_found"}
    assert _structured_tool_result(spoofed)["result"] == "capability result"

    assert jobs.calls == [
        ("submit", "question", "anonymous"),
        ("wait", _CapabilityJobs.JOB_ID, "anonymous", 30.0),
        ("poll", _CapabilityJobs.JOB_ID, "anonymous"),
        ("report", _CapabilityJobs.JOB_ID, "anonymous"),
        ("poll", unknown_capability, "anonymous"),
        ("report", _CapabilityJobs.JOB_ID, "anonymous"),
    ]
    for headers in captured_headers[:-1]:
        assert b"authorization" not in headers
    assert captured_headers[-1][b"authorization"] == b"Bearer ignored"


@pytest.mark.asyncio
async def test_liveness_readiness_and_mcp_get_policy(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: _Jobs(events),
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        assert (await client.get("/live")).json() == {"status": "alive"}
        not_ready = await client.get("/health")
        assert not_ready.status_code == 503
        assert not_ready.json() == {"status": "not_ready"}

        mcp_get = await client.get("/mcp", headers={"accept": "text/event-stream"})
        assert mcp_get.status_code == 405
        assert mcp_get.content == b""
        assert mcp_get.headers["allow"] == "POST"

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            ready = await client.get("/health")
            assert ready.status_code == 200
            assert ready.json() == {"status": "ready"}

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        assert (await client.get("/health")).status_code == 503


@pytest.mark.asyncio
async def test_configured_mcp_path_routes_protocol_and_get_policy(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = server.MCPRuntime(
        _settings(tmp_path, path="/legacy/research/mcp", cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: _Jobs(events),
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            initialized = await client.post(
                "/legacy/research/mcp",
                headers=_MCP_HEADERS,
                json=_initialize_request(1),
            )
            rejected_get = await client.get(
                "/legacy/research/mcp",
                headers={"accept": "text/event-stream"},
            )
            old_path = await client.post("/mcp", headers=_MCP_HEADERS, json=_initialize_request(2))

    assert initialized.status_code == 200
    assert rejected_get.status_code == 405
    assert rejected_get.headers["allow"] == "POST"
    assert old_path.status_code == 404


@pytest.mark.asyncio
async def test_inspector_cors_is_explicitly_allowlisted(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = server.MCPRuntime(
        _settings(tmp_path),
        runner=_Service("runner", events),
        jobs_factory=lambda: _Jobs(events),
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)
    preflight_headers = {
        "origin": "http://localhost:6274",
        "access-control-request-method": "POST",
        "access-control-request-headers": "content-type",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        allowed = await client.options("/mcp", headers=preflight_headers)
        denied = await client.options(
            "/mcp",
            headers={**preflight_headers, "origin": "https://untrusted.example"},
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:6274"
    assert "access-control-allow-credentials" not in allowed.headers
    assert allowed.headers["access-control-allow-methods"] == "POST"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            initialized = await client.post(
                "/mcp",
                headers={**_MCP_HEADERS, "origin": "http://localhost:6274"},
                json=_initialize_request(1),
            )
            bad_origin = await client.post(
                "/mcp",
                headers={**_MCP_HEADERS, "origin": "https://untrusted.example"},
                json=_initialize_request(2),
            )
        async with httpx.AsyncClient(transport=transport, base_url="http://untrusted.example") as client:
            bad_host = await client.post("/mcp", headers=_MCP_HEADERS, json=_initialize_request(3))

    assert initialized.status_code == 200
    assert initialized.headers["access-control-allow-origin"] == "http://localhost:6274"
    assert bad_origin.status_code == 403
    assert bad_origin.text == "Invalid Origin header"
    assert bad_host.status_code == 421
    assert bad_host.text == "Invalid Host header"


@pytest.mark.asyncio
async def test_startup_failure_cleans_up_started_services(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: _Jobs(events, fail_start=True),
        validate_startup=lambda: events.append("validate"),
    )

    with pytest.raises(RuntimeError, match="jobs failed to start"):
        async with runtime.app.router.lifespan_context(runtime.app):
            raise AssertionError("startup should not yield")

    assert runtime.ready is False
    assert runtime.app.state.ready is False
    assert events == ["validate", "runner:start", "jobs:start", "jobs:stop", "runner:stop"]


@pytest.mark.asyncio
async def test_runner_start_failure_is_cleaned_up_before_session_manager_starts(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events, fail_start=True),
        jobs_factory=lambda: _Jobs(events),
        validate_startup=lambda: events.append("validate"),
    )

    with pytest.raises(RuntimeError, match="runner failed to start"):
        async with runtime.app.router.lifespan_context(runtime.app):
            raise AssertionError("startup should not yield")

    assert runtime.ready is False
    assert runtime.mcp.session_manager._has_started is False
    assert events == ["validate", "runner:start", "runner:stop"]


@pytest.mark.asyncio
async def test_tool_handlers_use_anonymous_principal_and_preserve_inline_behavior(tmp_path: Path) -> None:
    events: list[str] = []
    jobs = _Jobs(events)
    jobs.submit_result = {
        "job_id": "job-1",
        "depth": "shallow",
        "state": "queued",
        "estimated_duration_seconds": 10,
        "first_poll_after_seconds": 5,
    }
    runtime = server.MCPRuntime(
        _settings(tmp_path, shallow_inline_wait_seconds=4.5),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    runtime.jobs = jobs

    submitted = await runtime.submit_query(None, "question")  # type: ignore[arg-type]
    polled = await runtime.poll_query(None, "job-1")  # type: ignore[arg-type]
    report = await runtime.get_final_report(None, "job-1")  # type: ignore[arg-type]

    assert submitted == {
        "job_id": "job-1",
        "depth": "shallow",
        "state": "queued",
        "estimated_duration_seconds": 5,
        "first_poll_after_seconds": 0,
    }
    assert polled["state"] == "running"
    assert report["state"] == "complete"
    assert jobs.calls == [
        ("submit", "question", "anonymous"),
        ("wait", "job-1", "anonymous", 4.5),
        ("poll", "job-1", "anonymous"),
        ("report", "job-1", "anonymous"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_result",
    [
        {
            "job_id": "job-1",
            "depth": "shallow",
            "state": "complete",
            "result": "inline answer",
        },
        {
            "job_id": "job-1",
            "depth": "shallow",
            "state": "failed",
            "error": "inline failure",
        },
    ],
)
async def test_shallow_submit_returns_exact_inline_terminal_contract(
    tmp_path: Path,
    terminal_result: dict[str, Any],
) -> None:
    events: list[str] = []
    jobs = _Jobs(events)
    jobs.submit_result = {
        "job_id": "job-1",
        "depth": "shallow",
        "state": "queued",
        "estimated_duration_seconds": 10,
        "first_poll_after_seconds": 5,
    }
    jobs.inline_result = terminal_result
    runtime = server.MCPRuntime(
        _settings(tmp_path, shallow_inline_wait_seconds=30),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    runtime.jobs = jobs

    assert await runtime.submit_query(None, "question") == terminal_result  # type: ignore[arg-type]
    assert jobs.calls == [
        ("submit", "question", "anonymous"),
        ("wait", "job-1", "anonymous", 30.0),
    ]


@pytest.mark.asyncio
async def test_default_shallow_inline_timeout_returns_exact_zero_cadence_contract(tmp_path: Path) -> None:
    events: list[str] = []
    jobs = _Jobs(events)
    jobs.submit_result = {
        "job_id": "job-1",
        "depth": "shallow",
        "state": "queued",
        "estimated_duration_seconds": 10,
        "first_poll_after_seconds": 5,
    }
    runtime = server.MCPRuntime(
        _settings(tmp_path, shallow_inline_wait_seconds=30),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    runtime.jobs = jobs

    assert await runtime.submit_query(None, "question") == {  # type: ignore[arg-type]
        "job_id": "job-1",
        "depth": "shallow",
        "state": "queued",
        "estimated_duration_seconds": 0,
        "first_poll_after_seconds": 0,
    }


@pytest.mark.asyncio
async def test_submit_query_rejects_oversized_query_before_enqueue(tmp_path: Path) -> None:
    events: list[str] = []
    jobs = _Jobs(events)
    runtime = server.MCPRuntime(
        _settings(tmp_path, max_query_chars=10),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    runtime.jobs = jobs

    with pytest.raises(ValueError, match="11 characters exceeds the 10-character limit"):
        await runtime.submit_query(None, "x" * 11)  # type: ignore[arg-type]
    assert jobs.calls == []

    boundary = await runtime.submit_query(None, "x" * 10)  # type: ignore[arg-type]
    assert boundary["state"] == "queued"
    assert jobs.calls == [("submit", "x" * 10, "anonymous")]


@pytest.mark.asyncio
async def test_deep_submit_never_attempts_inline_wait(tmp_path: Path) -> None:
    events: list[str] = []

    class _DeepJobs(_Jobs):
        async def wait_for_completion(
            self,
            job_id: str,
            principal: str,
            timeout: float,
        ) -> dict[str, Any] | None:
            del job_id, principal, timeout
            raise AssertionError("deep jobs must never inline-wait")

    jobs = _DeepJobs(events)
    runtime = server.MCPRuntime(
        _settings(tmp_path, shallow_inline_wait_seconds=30),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    runtime.jobs = jobs

    assert await runtime.submit_query(None, "detailed research") == {  # type: ignore[arg-type]
        "job_id": "job-1",
        "depth": "deep",
        "state": "queued",
        "estimated_duration_seconds": 180,
        "first_poll_after_seconds": 180,
    }
    assert jobs.calls == [("submit", "detailed research", "anonymous")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_method", "tool_name", "arguments", "expected_call", "public_error"),
    [
        (
            "submit",
            "submit_query",
            {"query": "question"},
            ("question", "anonymous"),
            "Research query submission failed. Check server logs for details.",
        ),
        (
            "poll",
            "poll_query",
            {"job_id": _CapabilityJobs.JOB_ID},
            (_CapabilityJobs.JOB_ID, "anonymous"),
            "Research query status check failed. Check server logs for details.",
        ),
        (
            "get_final_report",
            "get_final_report",
            {"job_id": _CapabilityJobs.JOB_ID},
            (_CapabilityJobs.JOB_ID, "anonymous"),
            "Research report retrieval failed. Check server logs for details.",
        ),
    ],
    ids=("submit", "poll", "final-report"),
)
async def test_transport_sanitizes_job_service_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
    job_method: str,
    tool_name: str,
    arguments: dict[str, Any],
    expected_call: tuple[Any, ...],
    public_error: str,
) -> None:
    events: list[str] = []
    jobs = _CapabilityJobs(events)
    failed_calls: list[tuple[Any, ...]] = []

    async def fail_with_internal_details(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert not kwargs
        failed_calls.append(args)
        raise RuntimeError(_BOUNDARY_FAILURE_MESSAGE)

    monkeypatch.setattr(jobs, job_method, fail_with_internal_details)
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)
    caplog.set_level("ERROR", logger="deep_researcher_mcp.server")

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(1, tool_name, arguments),
            )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert result.get("structuredContent") is None
    assert public_error in result["content"][0]["text"]
    assert failed_calls == [expected_call]
    assert "RuntimeError" in caplog.text
    assert _CapabilityJobs.JOB_ID not in caplog.text
    for fragment in (_BOUNDARY_FAILURE_CREDENTIAL, _BOUNDARY_FAILURE_HOST):
        assert fragment not in response.text
        assert fragment not in caplog.text


@pytest.mark.asyncio
async def test_transport_sanitizes_submit_response_processing_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    class _ExplodingSubmitResult(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            del key, default
            raise RuntimeError(_BOUNDARY_FAILURE_MESSAGE)

    events: list[str] = []
    jobs = _CapabilityJobs(events)

    async def return_exploding_result(query: str, principal: str) -> dict[str, Any]:
        jobs.calls.append(("submit", query, principal))
        return _ExplodingSubmitResult(jobs.submit_result)

    monkeypatch.setattr(jobs, "submit", return_exploding_result)
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)
    caplog.set_level("ERROR", logger="deep_researcher_mcp.server")

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(1, "submit_query", {"query": "question"}),
            )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert result.get("structuredContent") is None
    assert "Research query submission failed. Check server logs for details." in result["content"][0]["text"]
    assert jobs.calls == [("submit", "question", "anonymous")]
    assert "submit_query response handling failed (RuntimeError)" in caplog.text
    assert _CapabilityJobs.JOB_ID not in caplog.text
    for fragment in (_BOUNDARY_FAILURE_CREDENTIAL, _BOUNDARY_FAILURE_HOST):
        assert fragment not in response.text
        assert fragment not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("submit_result", "expected_log"),
    [
        (
            {
                "depth": "shallow",
                "state": "queued",
                "estimated_duration_seconds": 45,
                "first_poll_after_seconds": 5,
            },
            "submit_query response handling failed (KeyError)",
        ),
        (
            {
                "job_id": "",
                "depth": "shallow",
                "state": "queued",
                "estimated_duration_seconds": 45,
                "first_poll_after_seconds": 5,
            },
            "submit_query response handling failed (TypeError)",
        ),
        (
            {
                "job_id": _CapabilityJobs.JOB_ID,
                "depth": "shallow",
                "state": "queued",
                "estimated_duration_seconds": _BOUNDARY_FAILURE_MESSAGE,
                "first_poll_after_seconds": 5,
            },
            "submit_query response handling failed (TypeError)",
        ),
    ],
    ids=("missing-job-id", "empty-job-id", "bad-estimate"),
)
async def test_transport_sanitizes_malformed_queued_submit_response(
    tmp_path: Path,
    caplog,
    submit_result: dict[str, Any],
    expected_log: str,
) -> None:
    events: list[str] = []
    jobs = _CapabilityJobs(events)
    jobs.submit_result = submit_result
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)
    caplog.set_level("ERROR", logger="deep_researcher_mcp.server")

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(1, "submit_query", {"query": "question"}),
            )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert result.get("structuredContent") is None
    assert "Research query submission failed. Check server logs for details." in result["content"][0]["text"]
    assert jobs.calls == [("submit", "question", "anonymous")]
    assert expected_log in caplog.text
    assert _CapabilityJobs.JOB_ID not in caplog.text
    for fragment in (_BOUNDARY_FAILURE_CREDENTIAL, _BOUNDARY_FAILURE_HOST):
        assert fragment not in response.text
        assert fragment not in caplog.text


@pytest.mark.asyncio
async def test_inline_wait_exception_preserves_queued_capability_without_exposing_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    events: list[str] = []
    jobs = _CapabilityJobs(events)

    async def fail_inline_wait(job_id: str, principal: str, timeout: float) -> dict[str, Any] | None:
        jobs.calls.append(("wait", job_id, principal, timeout))
        raise RuntimeError(_BOUNDARY_FAILURE_MESSAGE)

    monkeypatch.setattr(jobs, "wait_for_completion", fail_inline_wait)
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)
    caplog.set_level("ERROR", logger="deep_researcher_mcp.server")

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(1, "submit_query", {"query": "question"}),
            )

    assert response.status_code == 200
    assert _structured_tool_result(response) == jobs.submit_result
    assert jobs.calls == [
        ("submit", "question", "anonymous"),
        ("wait", _CapabilityJobs.JOB_ID, "anonymous", 30.0),
    ]
    assert "submit_query inline wait after enqueue failed (RuntimeError)" in caplog.text
    assert _CapabilityJobs.JOB_ID not in caplog.text
    for fragment in (_BOUNDARY_FAILURE_CREDENTIAL, _BOUNDARY_FAILURE_HOST):
        assert fragment not in response.text
        assert fragment not in caplog.text


@pytest.mark.asyncio
async def test_classifier_failure_becomes_mcp_tool_error_without_business_payload(tmp_path: Path) -> None:
    events: list[str] = []

    class _ClassifierFailureJobs(_Jobs):
        async def submit(self, query: str, principal: str) -> dict[str, Any]:
            self.calls.append(("submit", query, principal))
            raise RuntimeError("classifier unavailable")

    jobs = _ClassifierFailureJobs(events)
    runtime = server.MCPRuntime(
        _settings(tmp_path, cors_origins=()),
        runner=_Service("runner", events),
        jobs_factory=lambda: jobs,
        validate_startup=lambda: None,
    )
    transport = httpx.ASGITransport(app=runtime.app)

    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.post(
                "/mcp",
                headers=_MCP_HEADERS,
                json=_tool_call_request(1, "submit_query", {"query": "question"}),
            )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert result.get("structuredContent") is None
    assert "Research query submission failed. Check server logs for details." in result["content"][0]["text"]
    assert "classifier unavailable" not in result["content"][0]["text"]
    assert jobs.calls == [("submit", "question", "anonymous")]


def test_create_app_returns_fresh_worker_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_CORS_ORIGINS", "")

    first = server.create_app()
    second = server.create_app()

    assert first is not second
    assert first.state.deep_researcher_mcp_runtime is not second.state.deep_researcher_mcp_runtime
    assert (
        first.state.deep_researcher_mcp_runtime.mcp.session_manager
        is not second.state.deep_researcher_mcp_runtime.mcp.session_manager
    )


def test_main_uses_uvicorn_import_string(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: list[tuple[str, dict[str, Any]]] = []
    logging_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_PORT", "9100")
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_WORKERS", "3")
    monkeypatch.setenv("DEEP_RESEARCHER_MCP_LOG_LEVEL", "warning")
    monkeypatch.setattr(server.logging, "basicConfig", lambda **kwargs: logging_calls.append(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda app_path, **kwargs: calls.append((app_path, kwargs)))

    server.main()

    assert calls == [
        (
            "deep_researcher_mcp.server:app",
            {
                "host": "127.0.0.1",
                "port": 9100,
                "workers": 3,
                "log_level": "warning",
            },
        )
    ]
    assert logging_calls == [
        {
            "level": "WARNING",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "force": True,
        }
    ]
