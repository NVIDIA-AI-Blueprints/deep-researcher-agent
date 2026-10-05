# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FastMCP runtime for the Deep Researcher Agent research workflow.

The MCP transport is stateless, but the NAT workflow and background job
manager are process-scoped. An outer Starlette lifespan owns those long-lived
resources and the FastMCP session manager exactly once per Uvicorn worker.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Protocol
from typing import cast

from dotenv import load_dotenv
from mcp.server.fastmcp import Context
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.routing import Mount
from starlette.routing import Route

from .checkpoint_todos import CheckpointTodoReader
from .db_url import normalize_postgres_url
from .job_store import JobStore
from .jobs import JobManager
from .workflow_runner import WorkflowRunner

logger = logging.getLogger(__name__)


def _find_source_checkout_root() -> Path | None:
    """Return the Deep Researcher Agent checkout root, or None when running from an installed package."""
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "configs" / "config_mcp.yml").is_file():
        return candidate
    return None


_REPO_ROOT = _find_source_checkout_root()
_DEFAULT_ENV_FILE = _REPO_ROOT / "deploy" / ".env" if _REPO_ROOT else None
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "config_mcp.yml" if _REPO_ROOT else None
MCP_SERVER_NAME = "deep_researcher_deep_research"
ANONYMOUS_PRINCIPAL = "anonymous"
_PUBLIC_POLL_ERROR = "Research query status check failed. Check server logs for details."
_PUBLIC_REPORT_ERROR = "Research report retrieval failed. Check server logs for details."
_PUBLIC_SUBMIT_ERROR = "Research query submission failed. Check server logs for details."
_CHECKPOINT_DB_ENV_VAR = "DEEP_RESEARCHER_CHECKPOINT_DB"
_DEFAULT_INSPECTOR_ORIGIN = "http://localhost:6274"
_DEFAULT_ALLOWED_HOSTS = (
    "localhost",
    "localhost:*",
    "127.0.0.1",
    "127.0.0.1:*",
    "[::1]",
    "[::1]:*",
    "0.0.0.0",
    "0.0.0.0:*",
)
_DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:*",
    "http://127.0.0.1:*",
    "http://[::1]:*",
    "http://0.0.0.0:*",
)
_RESERVED_HTTP_PATHS = frozenset({"/health", "/live"})

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_LOG_LEVELS: tuple[LogLevel, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _load_env_file() -> None:
    """Load the public MCP dotenv file without overriding process variables."""
    configured_path = os.getenv("DEEP_RESEARCHER_MCP_ENV_FILE")
    path = Path(configured_path).expanduser() if configured_path else _DEFAULT_ENV_FILE
    if path is None:
        return
    if not path.exists():
        if configured_path:
            logger.warning("MCP env file does not exist: %s", path)
        return
    load_dotenv(dotenv_path=path, override=False)


_load_env_file()


class LifecycleService(Protocol):
    """The lifecycle surface shared by the workflow and job manager."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class JobService(LifecycleService, Protocol):
    """Job operations used by the three MCP tools."""

    async def submit(self, query: str, principal: str) -> dict[str, Any]: ...

    async def wait_for_completion(
        self,
        job_id: str,
        principal: str,
        timeout: float,
    ) -> dict[str, Any] | None: ...

    async def poll(self, job_id: str, principal: str) -> dict[str, Any]: ...

    async def get_final_report(self, job_id: str, principal: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ServerSettings:
    """Environment-derived process and transport settings."""

    host: str
    port: int
    path: str
    workers: int
    log_level: LogLevel
    config_path: Path
    shallow_inline_wait_seconds: float
    cors_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    max_query_chars: int = 8000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerSettings:
        values = os.environ if environ is None else environ

        host = values.get("DEEP_RESEARCHER_MCP_HOST", "0.0.0.0").strip()
        if not host:
            raise ValueError("DEEP_RESEARCHER_MCP_HOST must not be empty")

        port = _parse_int(
            values.get("DEEP_RESEARCHER_MCP_PORT"), name="DEEP_RESEARCHER_MCP_PORT", default=9001, minimum=1
        )
        if port > 65535:
            raise ValueError("DEEP_RESEARCHER_MCP_PORT must be at most 65535")

        path = _normalize_mcp_path(values.get("DEEP_RESEARCHER_MCP_PATH", "/mcp"))
        workers = _parse_int(
            values.get("DEEP_RESEARCHER_MCP_WORKERS"), name="DEEP_RESEARCHER_MCP_WORKERS", default=1, minimum=1
        )

        raw_log_level = values.get("DEEP_RESEARCHER_MCP_LOG_LEVEL", "INFO").strip().upper()
        if raw_log_level not in _LOG_LEVELS:
            raise ValueError("DEEP_RESEARCHER_MCP_LOG_LEVEL must be one of: " + ", ".join(_LOG_LEVELS))
        log_level = cast(LogLevel, raw_log_level)

        config_value = values.get("DEEP_RESEARCHER_MCP_CONFIG")
        if config_value:
            config_path = Path(config_value).expanduser().resolve()
        elif DEFAULT_CONFIG is not None:
            config_path = DEFAULT_CONFIG
        else:
            raise ValueError(
                "DEEP_RESEARCHER_MCP_CONFIG must point to a workflow config when deep_researcher_mcp is installed "
                "outside the Deep Researcher Agent source checkout"
            )

        inline_wait = _parse_float(
            values.get("DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS"),
            name="DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS",
            default=30.0,
            minimum=0.0,
        )

        max_query_chars = _parse_int(
            values.get("DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS"),
            name="DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS",
            default=8000,
            minimum=1,
        )

        raw_origins = values.get("DEEP_RESEARCHER_MCP_CORS_ORIGINS", _DEFAULT_INSPECTOR_ORIGIN)
        cors_origins = _parse_csv(raw_origins)

        raw_allowed_hosts = values.get("DEEP_RESEARCHER_MCP_ALLOWED_HOSTS", ",".join(_DEFAULT_ALLOWED_HOSTS))
        allowed_hosts = _parse_csv(raw_allowed_hosts)
        if not allowed_hosts:
            raise ValueError("DEEP_RESEARCHER_MCP_ALLOWED_HOSTS must contain at least one host")

        raw_allowed_origins = values.get("DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS", ",".join(_DEFAULT_ALLOWED_ORIGINS))
        # Any browser origin admitted by CORS must also pass FastMCP's actual
        # request validation. Keep the transport list as the union so a custom
        # Inspector origin cannot pass preflight and then fail its POST.
        allowed_origins = tuple(dict.fromkeys((*_parse_csv(raw_allowed_origins), *cors_origins)))

        return cls(
            host=host,
            port=port,
            path=path,
            workers=workers,
            log_level=log_level,
            config_path=config_path,
            shallow_inline_wait_seconds=inline_wait,
            cors_origins=cors_origins,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
            max_query_chars=max_query_chars,
        )


def _parse_int(value: str | None, *, name: str, default: int, minimum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _parse_float(value: str | None, *, name: str, default: float, minimum: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    return parsed


def _normalize_mcp_path(value: str) -> str:
    path = value.strip()
    if not path.startswith("/") or path == "/":
        raise ValueError("DEEP_RESEARCHER_MCP_PATH must be an absolute non-root URL path")
    if any(character in path for character in "?#{}"):
        raise ValueError(
            "DEEP_RESEARCHER_MCP_PATH must be a literal path without parameters, a query string, or a fragment"
        )
    path = path.rstrip("/")
    if not path:
        raise ValueError("DEEP_RESEARCHER_MCP_PATH must be an absolute non-root URL path")
    if path in _RESERVED_HTTP_PATHS:
        raise ValueError(f"DEEP_RESEARCHER_MCP_PATH conflicts with the reserved route: {path}")
    return path


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _normalized_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_checkpoint_db_url() -> str:
    value = _normalized_env_value(os.getenv(_CHECKPOINT_DB_ENV_VAR))
    if value is None:
        raise ValueError("MCP startup requires DEEP_RESEARCHER_CHECKPOINT_DB to be set to a Postgres DSN")
    return normalize_postgres_url(value, label=_CHECKPOINT_DB_ENV_VAR)


def _validate_startup_configuration(settings: ServerSettings) -> None:
    if not settings.config_path.is_file():
        raise ValueError(f"MCP workflow config does not exist: {settings.config_path}")

    # NAT and the MCP job ledger share the same Postgres deployment. Normalize
    # driver-qualified URLs once before either lifecycle is started.
    os.environ[_CHECKPOINT_DB_ENV_VAR] = _resolve_checkpoint_db_url()


def _mcp_instructions(inline_wait_seconds: float) -> str:
    return (
        "Deep Researcher Agent Deep Research MCP server. Protocol: "
        "1. Call submit_query(query). "
        "2. If submit_query returns state='complete', use result directly. "
        "3. If submit_query returns state='queued', wait first_poll_after_seconds, then call "
        "poll_query(job_id). "
        "4. While poll_query returns state='queued' or state='running', wait "
        "next_poll_after_seconds before polling again. "
        "5. If poll_query returns state='complete', call get_final_report(job_id) to fetch the report payload. "
        "6. If poll_query returns state='failed' or state='not_found', stop polling and surface the error. "
        f"Deep jobs always queue. Shallow jobs may complete inline within {inline_wait_seconds:g} seconds. "
        "poll_query is status-only and does not return the final report body. Deep jobs use a fixed "
        "180-second polling cadence. Found jobs may include best-effort todos progress hints; return non-empty "
        "todos to the user during polling. todos=[] is normal and not an error. Do not run separate web or paper "
        "research for the same question while waiting; "
        "the queued Deep Researcher Agent job is already doing that work. If the "
        "client supports background monitors or subagents, delegate only the polling loop and resume when the "
        "final report is ready. No Authorization header or token tool argument is required. Every request uses "
        "the shared anonymous principal, so a returned job_id is an opaque bearer capability rather than a "
        "per-user identifier. Anyone who possesses it can poll or retrieve that job until its expired database "
        "row is removed by periodic cleanup; keep it private and do not log or share it."
    )


def _log_public_tool_failure(operation: str, exc: Exception) -> None:
    logger.error("MCP %s failed (%s)", operation, type(exc).__name__)


def _public_tool_error(operation: str, public_message: str, exc: Exception) -> RuntimeError:
    _log_public_tool_failure(operation, exc)
    return RuntimeError(public_message)


class MCPRuntime:
    """One worker's FastMCP transport and long-lived Deep Researcher Agent services."""

    def __init__(
        self,
        settings: ServerSettings,
        *,
        runner: LifecycleService | None = None,
        jobs_factory: Callable[[], JobService] | None = None,
        validate_startup: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.runner = runner or WorkflowRunner(settings.config_path)
        self._jobs_factory = jobs_factory or self._create_job_manager
        self._validate_startup = validate_startup or (lambda: _validate_startup_configuration(settings))
        self.jobs: JobService | None = None
        self.ready = False

        self.mcp = FastMCP(
            name=MCP_SERVER_NAME,
            instructions=_mcp_instructions(settings.shallow_inline_wait_seconds),
            debug=False,
            log_level=settings.log_level,
            host=settings.host,
            port=settings.port,
            streamable_http_path=settings.path,
            json_response=True,
            stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(settings.allowed_hosts),
                allowed_origins=list(settings.allowed_origins),
            ),
        )
        self.mcp.tool()(self.submit_query)
        self.mcp.tool()(self.poll_query)
        self.mcp.tool()(self.get_final_report)
        self.app = self._build_app()

    def _create_job_manager(self) -> JobManager:
        checkpoint_db_url = _resolve_checkpoint_db_url()
        return JobManager(
            cast(WorkflowRunner, self.runner),
            JobStore(checkpoint_db_url),
            checkpoint_todo_reader=CheckpointTodoReader(checkpoint_db_url),
        )

    def _get_jobs(self) -> JobService:
        if self.jobs is None:
            raise RuntimeError("MCP job manager has not been initialized")
        return self.jobs

    async def submit_query(self, ctx: Context, query: str) -> dict[str, Any]:
        """Submit a research question.

        For meta queries and shallow queries that finish within the configured
        inline window, the final answer is returned with state="complete" and
        result. Deep queries always queue. Shallow queries that do not finish
        within the inline window return state="queued". If the inline wait
        itself fails after enqueue, the original queued response is returned
        so the caller retains the job capability and can poll it.

        Protocol:
          - If state="complete": use result directly; do not call get_final_report.
          - If state="failed": surface error and stop.
          - If state="queued": wait first_poll_after_seconds, then call poll_query(job_id).

        After the first poll, continue using poll_query while it returns
        state="queued" or state="running", waiting next_poll_after_seconds each
        time. poll_query is status-only; when it returns state="complete", call
        get_final_report(job_id) to fetch the final report.

        Do not run independent web or paper research for the same question
        while waiting; the background Deep Researcher Agent job is already doing that work. If
        the client supports background monitors or subagents, delegate only
        the polling loop and return non-empty todos to the user as best-effort
        progress hints.

        Args:
            query: The research question to investigate.

        Returns:
            A dict that always contains `job_id`, `depth`
            ("shallow"|"deep"|"meta"), and `state`. Plus one of:
              - `result` when state="complete"
              - `error` when state="failed"
              - `first_poll_after_seconds` and `estimated_duration_seconds`
                when state="queued"

            `job_id` is an opaque bearer capability. All callers share the
            anonymous principal, so anyone possessing the UUID can poll or
            retrieve the job until its expired database row is removed by
            periodic cleanup. Keep it private.
        """
        del ctx
        # Anonymous callers get a static, capability-free rejection before any
        # job is enqueued; the limit bounds per-request workflow input size.
        query_limit = self.settings.max_query_chars
        if len(query) > query_limit:
            raise ValueError(f"query is too long: {len(query)} characters exceeds the {query_limit}-character limit")
        try:
            jobs = self._get_jobs()
            submit_result = await jobs.submit(query, ANONYMOUS_PRINCIPAL)
        except Exception as exc:  # noqa: BLE001 - public MCP boundary must sanitize tool errors
            raise _public_tool_error("submit_query submission", _PUBLIC_SUBMIT_ERROR, exc) from None

        try:
            if submit_result.get("state") != "queued":
                return submit_result
            if submit_result["depth"] != "shallow":
                return submit_result
        except Exception as exc:  # noqa: BLE001 - public MCP boundary must sanitize tool errors
            raise _public_tool_error("submit_query response handling", _PUBLIC_SUBMIT_ERROR, exc) from None

        inline_wait = self.settings.shallow_inline_wait_seconds
        try:
            job_id = submit_result["job_id"]
            if not isinstance(job_id, str) or not job_id:
                raise TypeError("queued submit result job_id must be a non-empty string")
            remaining_estimate = max(
                0,
                int(submit_result["estimated_duration_seconds"] - inline_wait),
            )
        except Exception as exc:  # noqa: BLE001 - public MCP boundary must sanitize tool errors
            raise _public_tool_error("submit_query response handling", _PUBLIC_SUBMIT_ERROR, exc) from None

        try:
            inline = await jobs.wait_for_completion(
                job_id,
                ANONYMOUS_PRINCIPAL,
                timeout=inline_wait,
            )
            if inline is not None:
                return inline

            return {
                **submit_result,
                "first_poll_after_seconds": 0,
                "estimated_duration_seconds": remaining_estimate,
            }
        except Exception as exc:  # noqa: BLE001 - preserve an already-issued job capability
            _log_public_tool_failure("submit_query inline wait after enqueue", exc)
            return submit_result

    async def poll_query(self, ctx: Context, job_id: str) -> dict[str, Any]:
        """Check a research job's status.

        poll_query is status-only. It tells the client whether the job is still
        running or whether the final report is ready; it does not return the
        final report body.

        If state is "queued" or "running", wait next_poll_after_seconds before
        calling poll_query again. Deep jobs use a fixed 180-second polling
        cadence. If state is "complete", stop polling and call
        get_final_report(job_id). If state is "failed" or "not_found", stop
        polling and surface error.

        Found jobs include todos. These are best-effort progress hints for deep
        jobs. Return non-empty todos to the user during polling; todos=[] is
        normal when progress is unavailable, early, shallow, or queued.

        Do not run independent web or paper research for the same question
        while waiting. If the client supports background monitors or subagents,
        delegate only the polling loop, surface todos when useful, and resume
        once get_final_report(job_id) is ready.

        Args:
            job_id: The opaque capability UUID returned by submit_query. Anyone
                possessing it can poll this anonymously owned job until its
                expired database row is removed by periodic cleanup.
        """
        del ctx
        try:
            return await self._get_jobs().poll(job_id, ANONYMOUS_PRINCIPAL)
        except Exception as exc:  # noqa: BLE001 - public MCP boundary must sanitize tool errors
            raise _public_tool_error("poll_query", _PUBLIC_POLL_ERROR, exc) from None

    async def get_final_report(self, ctx: Context, job_id: str) -> dict[str, Any]:
        """Fetch a completed research job's final answer or report.

        Call this only after poll_query(job_id) returns state="complete".
        Inline state="complete" responses from submit_query already include
        result and do not need get_final_report.

        If the job is still queued or running, this returns state="not_ready"
        with error="job_not_ready"; use poll_query for cadence instead. If the
        job failed, this returns state="failed" with error. An unknown,
        malformed, or cleanup-deleted capability ID returns state="not_found".
        All callers share the anonymous principal, so possession of `job_id`,
        not caller identity, controls access. Keep the UUID private.

        Args:
            job_id: The opaque capability UUID returned by submit_query.
        """
        del ctx
        try:
            return await self._get_jobs().get_final_report(job_id, ANONYMOUS_PRINCIPAL)
        except Exception as exc:  # noqa: BLE001 - public MCP boundary must sanitize tool errors
            raise _public_tool_error("get_final_report", _PUBLIC_REPORT_ERROR, exc) from None

    def _build_app(self) -> Starlette:
        # FastMCP creates its session manager lazily here. The returned child
        # app is mounted at root so its configured streamable_http_path remains
        # exactly /mcp (or the configured replacement), without double-prefixing.
        mcp_http_app = self.mcp.streamable_http_app()

        @asynccontextmanager
        async def lifespan(app: Starlette):
            runner_started = False
            jobs_started = False

            self._validate_startup()
            if self.jobs is None:
                self.jobs = self._jobs_factory()
            jobs = self._get_jobs()

            try:
                runner_started = True
                await self.runner.start()
                jobs_started = True
                await jobs.start()

                # StreamableHTTPSessionManager.run() is one-shot and must span
                # request handling for this worker, not an individual stateless
                # request. Keeping it innermost cancels request tasks before
                # their workflow/job dependencies are stopped.
                async with self.mcp.session_manager.run():
                    self.ready = True
                    app.state.ready = True
                    yield
            finally:
                self.ready = False
                app.state.ready = False
                try:
                    if jobs_started:
                        await jobs.stop()
                finally:
                    if runner_started:
                        await self.runner.stop()

        async def liveness(_: Request) -> JSONResponse:
            return JSONResponse({"status": "alive"})

        async def readiness(_: Request) -> JSONResponse:
            if self.ready:
                return JSONResponse({"status": "ready"})
            return JSONResponse({"status": "not_ready"}, status_code=503)

        async def reject_mcp_get(_: Request) -> Response:
            # This JSON-only stateless server has no server-initiated events.
            # MCP permits rejecting the optional standalone SSE GET stream.
            return Response(status_code=405, headers={"Allow": "POST"})

        middleware: list[Middleware] = []
        if self.settings.cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=list(self.settings.cors_origins),
                    allow_credentials=False,
                    allow_methods=["POST"],
                    allow_headers=["Content-Type", "Mcp-Protocol-Version", "Mcp-Session-Id", "Last-Event-ID"],
                    expose_headers=["Mcp-Session-Id"],
                )
            )

        app = Starlette(
            debug=False,
            routes=[
                Route("/live", liveness, methods=["GET"]),
                Route("/health", readiness, methods=["GET"]),
                Route(self.settings.path, reject_mcp_get, methods=["GET"]),
                Mount("/", app=mcp_http_app),
            ],
            middleware=middleware,
            lifespan=lifespan,
        )
        app.state.ready = False
        app.state.deep_researcher_mcp_runtime = self
        return app


def create_app() -> Starlette:
    """Create an independent ASGI application for one Uvicorn worker."""
    runtime = MCPRuntime(ServerSettings.from_env())
    return runtime.app


# Export a conventional ASGI application and its FastMCP registry for ASGI
# servers, schema inspection, and tests. Launch ``app`` (or use ``main``), not
# ``mcp.run()``, because the outer application owns the Deep Researcher Agent lifecycle.
app = create_app()
_default_runtime: MCPRuntime = app.state.deep_researcher_mcp_runtime
mcp = _default_runtime.mcp
submit_query = _default_runtime.submit_query
poll_query = _default_runtime.poll_query
get_final_report = _default_runtime.get_final_report


def main() -> None:
    """Launch the outer Starlette application with Uvicorn."""
    settings = ServerSettings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    import uvicorn

    uvicorn.run(
        "deep_researcher_mcp.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
