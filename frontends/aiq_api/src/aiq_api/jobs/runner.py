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

"""
Agent-agnostic job runner.

Provides the Dask task function for running any registered agent with:
- NAT's JobStore for job metadata and status
- SSE event streaming for real-time UI updates
- Cancellation monitoring for graceful job termination
- Phoenix/OpenTelemetry observability via NAT's ExporterManager
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from aiq_agent.agents.deep_researcher.resource_limits import DeepResearchExecutionTimeout
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.logging_utils import log_content_metadata

from .callbacks import AgentEventCallback
from .callbacks import build_final_report_event
from .event_store import BatchingEventStore
from .event_store import EventStore

if TYPE_CHECKING:
    from .crypto import ContentEncryptionPolicyIdentity

logger = logging.getLogger(__name__)

_DEEP_RESEARCH_AGENT_KWARGS = frozenset(
    {
        "domain_catalog_path",
        "enable_source_router",
        "enable_citation_verification",
        "skills",
        "sandbox",
        "job_id",
        "max_research_concurrency",
        "max_concurrent_source_tool_calls",
        "max_source_tool_batch_size",
        "resource_limits",
    }
)
_CONFIGURABLE_AGENT_KWARGS = frozenset({"config", "job_id"})
_JOB_SCOPED_AGENT_KWARGS = frozenset({"job_id"})
_SHALLOW_RESEARCH_AGENT_KWARGS = frozenset({"max_tool_iterations", "enforce_citations"})
_DATA_SCIENCE_AGENT_KWARGS = frozenset(
    {
        "llm",
        "recursion_limit",
        "interaction_mode",
        "response_mode",
        "gsf_catalog_call_limit",
        "gsf_text_to_sql_call_limit",
        "gsf_cache_repeated_calls",
        "python_call_limit",
        "finalization_model_call_limit",
    }
)


@dataclass(frozen=True)
class JobTraceCorrelation:
    """Serializable correlation from a submitting request to an independent job trace."""

    session_id: str | None = None
    submission_trace_id: str | None = None
    submission_span_id: str | None = None
    request_trace_tags: dict[str, str] = field(default_factory=dict)


def _constructor_accepts_explicit_kwargs(agent_cls: type, kwarg_names: frozenset[str]) -> bool:
    """Return true when a class constructor explicitly declares all requested kwargs."""
    import inspect

    try:
        signature = inspect.signature(agent_cls)
    except (TypeError, ValueError):
        try:
            signature = inspect.signature(agent_cls.__init__)
        except (TypeError, ValueError):
            return False

    accepted_kwargs = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return kwarg_names.issubset(accepted_kwargs)


class CancellationMonitor:
    """
    Monitors job status for cancellation requests.

    Polls the job store at regular intervals and sets an asyncio.Event
    when the job status changes to INTERRUPTED.
    """

    def __init__(
        self,
        scheduler_address: str,
        db_url: str,
        job_id: str,
        poll_interval: float = 1.0,
    ):
        """Configure the job-status poller used to detect interruption.

        Args:
            scheduler_address: Dask scheduler address (unused by polling, kept for context).
            db_url: Database URL of the job store to poll.
            job_id: Job whose status is monitored.
            poll_interval: Seconds between status polls.
        """
        self.scheduler_address = scheduler_address
        self.db_url = db_url
        self.job_id = job_id
        self.poll_interval = poll_interval
        self._cancelled = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None

    @property
    def is_cancelled(self) -> bool:
        """Return whether the monitored job has been interrupted."""
        return self._cancelled.is_set()

    async def _poll_job_status(self) -> None:
        """Poll job status and set cancelled event if interrupted."""
        from nat.front_ends.fastapi.async_jobs.job_store import JobStatus
        from nat.front_ends.fastapi.async_jobs.job_store import JobStore

        job_store = JobStore(scheduler_address=self.scheduler_address, db_url=self.db_url)

        while not self._cancelled.is_set():
            try:
                job = await job_store.get_job(self.job_id)
                if job and job.status == JobStatus.INTERRUPTED.value:
                    logger.info("Cancellation detected for job %s", self.job_id)
                    self._cancelled.set()
                    break
            except Exception as e:
                logger.warning("Error checking job status for %s: %s", self.job_id, e)

            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        """Start the cancellation monitor background task."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._poll_job_status())
            logger.debug("Started cancellation monitor for job %s", self.job_id)

    def stop(self) -> None:
        """Stop the cancellation monitor."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
            logger.debug("Stopped cancellation monitor for job %s", self.job_id)

    def check(self) -> None:
        """Check if cancelled and raise CancelledError if so."""
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled by user")


# Interval for emitting heartbeat events
HEARTBEAT_INTERVAL_SECONDS = 30

# The ghost-job reaper treats a RUNNING job with no recent activity as a dead
# worker. But a worker can spend minutes in pre-event initialization (config,
# providers, tools, MCP, sandbox) before it stores its first event, so from the
# moment the job enters RUNNING we refresh a lightweight lease — job_info's
# updated_at, the same column the reaper falls back to for a zero-event job — on
# this interval. A slow-but-live worker keeps its lease fresh and is not reaped;
# a genuinely dead worker stops refreshing and its lease goes stale. Keep this
# well under GHOST_JOB_TIMEOUT_SECONDS so a live worker refreshes several times
# before the reaper's timeout.
LEASE_REFRESH_INTERVAL_SECONDS = 60
RELAY_STARTUP_TIMEOUT_SECONDS = 30


async def _ensure_relay_started_for_job(relay_config: Any, job_id: str) -> None:
    """Start Relay without allowing observability initialization to stall a job."""
    from aiq_agent.relay.bootstrap import ensure_started

    try:
        await asyncio.wait_for(ensure_started(relay_config), timeout=RELAY_STARTUP_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("Relay startup failed for job %s (error_type=%s)", job_id, type(exc).__name__)


def _resolve_job_relay_config(config: Any, function_config: Any) -> Any:
    """Prefer workflow Relay settings for a separately executed async agent."""
    workflow_config = getattr(config, "workflow", None)
    return getattr(workflow_config, "relay", None) or getattr(function_config, "relay", None)


def _db_now_expr(db_url: str) -> str:
    """Return the DB current-time SQL expression for this backend.

    Accepts both ``postgresql://`` and the legacy ``postgres://`` scheme.
    """
    return "NOW()" if db_url.startswith(("postgresql", "postgres")) else "CURRENT_TIMESTAMP"


def _touch_job_lease_sync(db_url: str, job_id: str) -> None:
    """Refresh the running-job lease by bumping job_info.updated_at.

    Scoped to ``status = 'running'`` so it can never resurrect the timestamp of
    a job that has already reached a terminal state.
    """
    from sqlalchemy import text

    from .event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    stmt = text(
        f"UPDATE job_info SET updated_at = {_db_now_expr(db_url)} WHERE job_id = :job_id AND status = 'running'"
    )
    with engine.begin() as conn:
        conn.execute(stmt, {"job_id": job_id})


def _write_job_success_if_running_sync(db_url: str, job_id: str, stored_output: str) -> bool:
    """Compare-and-set the job to SUCCESS with its output, only if still RUNNING.

    A single guarded ``UPDATE ... WHERE status = 'running'`` so a job the reaper
    already moved to a terminal state (e.g. it was reaped while slow to finish)
    is never resurrected. Returns True iff this call performed the write.
    """
    from sqlalchemy import text

    from .event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    stmt = text(
        f"UPDATE job_info SET status = 'success', output = :output, updated_at = {_db_now_expr(db_url)} "
        "WHERE job_id = :job_id AND status = 'running'"
    )
    with engine.begin() as conn:
        result = conn.execute(stmt, {"output": stored_output, "job_id": job_id})
        return (result.rowcount or 0) == 1


def _write_job_source_failure_if_running_sync(
    db_url: str,
    job_id: str,
    public_error: str,
    stored_output: str,
    final_report_event: dict[str, Any] | None = None,
    job_output_cipher: Any | None = None,
) -> bool:
    """Persist a source failure, then store its optional final-report event best-effort."""
    from sqlalchemy import text

    from .event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    stmt = text(
        f"UPDATE job_info SET status = 'failure', error = :error, output = :output, "
        f"updated_at = {_db_now_expr(db_url)} WHERE job_id = :job_id AND status = 'running'"
    )
    with engine.begin() as conn:
        result = conn.execute(
            stmt,
            {"error": public_error, "output": stored_output, "job_id": job_id},
        )
        if (result.rowcount or 0) != 1:
            return False

    if final_report_event is not None:
        try:
            EventStore(db_url, job_id, content_cipher=job_output_cipher).store(final_report_event)
        except Exception as exc:
            logger.warning(
                "Job %s source-failure final-report event write failed exception=%s",
                job_id,
                exc.__class__.__name__,
            )
    return True


def _write_job_failure_if_running_sync(db_url: str, job_id: str, public_error: str) -> bool:
    """Compare-and-set a generic failure without changing an existing terminal job."""
    from sqlalchemy import text

    from .event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    stmt = text(
        f"UPDATE job_info SET status = 'failure', error = :error, updated_at = {_db_now_expr(db_url)} "
        "WHERE job_id = :job_id AND status = 'running'"
    )
    with engine.begin() as conn:
        result = conn.execute(stmt, {"error": public_error, "job_id": job_id})
        return (result.rowcount or 0) == 1


async def _persist_empty_source_failure(
    *,
    error: EmptySourceRegistryError,
    job_output_cipher: Any,
    db_url: str,
    job_id: str,
    event_store: Any | None = None,
) -> bool:
    """Persist a typed source failure, falling back safely if output storage fails."""
    from .crypto import serialize_job_output_for_storage

    output = {
        "report": error.generated_answer,
        "outcome_reason": error.reason.value,
    }
    try:
        stored_output = serialize_job_output_for_storage(output, job_output_cipher)
        if event_store is not None and hasattr(event_store, "flush"):
            await asyncio.to_thread(event_store.flush)
        final_report_event = (
            build_final_report_event(error.generated_answer).to_sse_dict() if error.generated_answer else None
        )
        return await asyncio.get_running_loop().run_in_executor(
            None,
            _write_job_source_failure_if_running_sync,
            db_url,
            job_id,
            error.public_message,
            stored_output,
            final_report_event,
            job_output_cipher,
        )
    except Exception as exc:
        logger.warning(
            "Job %s source-failure output write failed exception=%s",
            job_id,
            exc.__class__.__name__,
        )
        sanitized_error = f"job failed ({type(exc).__name__}); check server logs for details"
        await asyncio.get_running_loop().run_in_executor(
            None,
            _write_job_failure_if_running_sync,
            db_url,
            job_id,
            sanitized_error,
        )
        return False


def _run_lease_refresher(db_url: str, job_id: str, stop_event: threading.Event) -> None:
    """Refresh the running-job lease on a dedicated thread until signalled.

    Runs in its own OS thread — not the worker event loop — so it cannot be
    starved by synchronous cold-start work (config load, agent import) that
    holds the loop. ``stop_event.wait`` returns True when the job ends (exit) or
    False on timeout (refresh, then loop).
    """
    while not stop_event.wait(LEASE_REFRESH_INTERVAL_SECONDS):
        try:
            _touch_job_lease_sync(db_url, job_id)
        except Exception as exc:  # noqa: BLE001 - a failed lease refresh must never kill the job
            logger.debug("Lease refresh for job %s failed: %s", job_id, exc)


async def run_with_cancellation(
    coro,
    monitor: CancellationMonitor,
    event_store: EventStore | BatchingEventStore | None = None,
) -> Any:
    """
    Run a coroutine with cancellation monitoring and periodic heartbeats.

    Emits job.heartbeat events every 30s so the SSE stream stays alive
    and the ghost job reaper can detect dead workers.
    Raises asyncio.CancelledError if the monitor detects cancellation.
    """
    import time

    task = asyncio.create_task(coro)
    monitor.start()
    start_time = time.monotonic()
    last_heartbeat = start_time

    try:
        while not task.done():
            if monitor.is_cancelled:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError("Job cancelled by user")

            now = time.monotonic()
            if event_store and (now - last_heartbeat) >= HEARTBEAT_INTERVAL_SECONDS:
                last_heartbeat = now
                event_store.store(
                    {
                        "type": "job.heartbeat",
                        "data": {"uptime_seconds": int(now - start_time)},
                    }
                )

            await asyncio.sleep(0.1)

        return task.result()
    finally:
        monitor.stop()


def _load_agent_class(agent_class_path: str) -> type:
    """
    Dynamically load an agent class from its module path.

    Args:
        agent_class_path: Full path like 'aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent'

    Returns:
        The agent class.

    Raises:
        ImportError: If the module or class cannot be found.
    """
    module_path, class_name = agent_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_middleware_for_listed_function(config: Any, function_name: str) -> list[str]:
    """Return middleware directly assigned to a configured NAT function.

    This reads only the function's explicit ``middleware`` list. Middleware that
    targets functions through its own config, such as ``workflow_functions``, is
    resolved separately because the Dask worker does not execute the registered
    NAT function. The worker adapter must explicitly translate those configured
    function targets into middleware around the direct agent call.
    """
    middleware_names: list[str] = []
    function_config = config.functions.get(function_name)
    if function_config is not None:
        middleware_names.extend(function_config.middleware or [])

    duplicates = {name for name in middleware_names if middleware_names.count(name) > 1}
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Middleware configured multiple times for worker function `{function_name}`: {duplicate_list}"
        )

    return middleware_names


def _validate_worker_middleware_names(config: Any, function_name: str, middleware_names: list[str]) -> None:
    """Ensure worker-selected middleware names exist in the loaded config."""
    missing_middleware = [name for name in middleware_names if name not in config.middleware]
    if missing_middleware:
        missing_list = ", ".join(sorted(missing_middleware))
        raise ValueError(f"Middleware configured for worker function `{function_name}` is not defined: {missing_list}")


def _get_middleware_for_worker_function(config: Any, function_name: str) -> list[str]:
    """Return all middleware that should apply to a worker-executed function.

    The Dask worker does not call the registered NAT function directly, so it
    must reconstruct the same middleware boundary from config. This includes
    middleware listed on the function itself and middleware that targets the
    function by name through ``workflow_functions``.
    """
    middleware_names = _get_middleware_for_listed_function(config, function_name)

    # Dask does not use the normal builder-created function callable, so
    # workflow_functions targets must be translated into worker middleware here.
    for middleware_name, middleware_config in config.middleware.items():
        workflow_functions = getattr(middleware_config, "workflow_functions", None)
        if isinstance(workflow_functions, dict) and function_name in workflow_functions:
            middleware_names.append(middleware_name)
        elif isinstance(workflow_functions, list) and function_name in workflow_functions:
            middleware_names.append(middleware_name)

    duplicates = {name for name in middleware_names if middleware_names.count(name) > 1}
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Middleware configured multiple times for worker function `{function_name}`: {duplicate_list}"
        )

    _validate_worker_middleware_names(config, function_name, middleware_names)
    return middleware_names


async def _register_middleware(builder: Any, config: Any, middleware_names: list[str]) -> None:
    """Ensure the worker builder can instantiate the selected middleware.

    The worker has its own builder instance. Registering middleware here makes
    the configured middleware available to the worker without changing which
    function call is wrapped.
    """
    for middleware_name in middleware_names:
        middleware_config = config.middleware[middleware_name]
        try:
            await builder.get_middleware(middleware_name)
        except ValueError:
            await builder.add_middleware(middleware_name, middleware_config)


async def _attach_middleware_to_function(builder: Any, config: Any, agent_config_name: str) -> None:
    """Register middleware needed by the configured function this worker represents.

    Actual invocation wrapping happens later, when the worker adapts the direct
    ``agent.run`` call into a NAT middleware chain.
    """
    function_config = config.functions.get(agent_config_name)
    if function_config is None:
        return

    middleware_names = _get_middleware_for_worker_function(config, agent_config_name)
    await _register_middleware(builder, config, middleware_names)


async def _run_with_configured_function_middleware(
    *,
    builder: Any,
    config: Any,
    function_name: str,
    function_config: Any,
    input_value: Any,
    call_next: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Apply NAT function middleware to the callable executed by the Dask worker.

    The async worker runs the agent instance directly instead of invoking the
    registered NAT function. This adapter preserves the configured function
    boundary by wrapping the worker callable with the middleware configured for
    that function name.
    """
    if config.functions.get(function_name) is None:
        return await call_next(input_value)

    middleware_names = _get_middleware_for_worker_function(config, function_name)
    if not middleware_names:
        return await call_next(input_value)

    from nat.middleware.function_middleware import FunctionMiddlewareChain
    from nat.middleware.middleware import FunctionMiddlewareContext

    middleware = await builder.get_middleware_list(middleware_names)
    context = FunctionMiddlewareContext(
        name=function_name,
        config=function_config,
        description=None,
        input_schema=type(input_value),
        single_output_schema=type(input_value),
        stream_output_schema=type(None),
    )
    wrapped_call = FunctionMiddlewareChain(middleware=middleware, context=context).build_single(call_next)
    return await wrapped_call(input_value)


async def _create_llm_provider(builder: Any, fn_config: Any) -> tuple[Any, Any]:
    """Create a role-aware LLM provider from a NAT function config."""
    from aiq_agent.common import LLMProvider
    from aiq_agent.common import LLMRole
    from nat.builder.framework_enum import LLMFrameworkEnum

    role_config_attrs = (
        (LLMRole.ORCHESTRATOR, "orchestrator_llm"),
        (LLMRole.ROUTER, "source_router_llm"),
        (LLMRole.PLANNER, "planner_llm"),
        (LLMRole.RESEARCHER, "researcher_llm"),
        (LLMRole.REPORT_WRITER, "writer_llm"),
    )
    llm_cache: dict[Any, Any] = {}
    role_llms = {}
    for role, config_attr in role_config_attrs:
        llm_ref = getattr(fn_config, config_attr, None)
        if llm_ref:
            if llm_ref not in llm_cache:
                llm_cache[llm_ref] = await builder.get_llm(llm_ref, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            role_llms[role] = llm_cache[llm_ref]

    default_llm = role_llms.get(LLMRole.ORCHESTRATOR)
    if default_llm is None:
        llm_ref = getattr(fn_config, "llm", None)
        if llm_ref:
            if llm_ref not in llm_cache:
                llm_cache[llm_ref] = await builder.get_llm(llm_ref, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            default_llm = llm_cache[llm_ref]

    provider = LLMProvider()
    provider.set_default(default_llm)
    for role, llm in role_llms.items():
        provider.configure(role, llm)

    return provider, default_llm


async def run_agent_job(
    configure_logging: bool,
    log_level: int,
    scheduler_address: str,
    db_url: str,
    config_file_path: str,
    job_id: str,
    input_text: str,
    agent_class_path: str,
    agent_config_name: str,
    trace_correlation: JobTraceCorrelation | None = None,
    available_documents: list[dict] | None = None,
    data_sources: list[str] | None = None,
    auth_token: str | None = None,
    content_encryption_policy: ContentEncryptionPolicyIdentity | None = None,
    initial_files: dict[str, Any] | None = None,
    output_metadata: dict[str, Any] | None = None,
    owner_user_id: str | None = None,
    admission_token: str | None = None,
    database_name: str | None = None,
):
    """
    Dask task to run any registered agent with cancellation support and telemetry.

    This function is submitted to Dask and runs in a worker process. It:
    - Uses NAT's JobStore for status tracking
    - Monitors for cancellation requests and gracefully terminates the agent
    - Exports telemetry to Phoenix/OpenTelemetry via NAT's ExporterManager
    - Starts an independent trace correlated to the submitting request and session

    Args:
        configure_logging: Whether to set up logging in the worker.
        log_level: Logging level to use.
        scheduler_address: Dask scheduler address.
        db_url: Database URL for job store and event store.
        config_file_path: Path to NAT config file.
        job_id: Unique job identifier.
        input_text: User input/query to run.
        agent_class_path: Full module path to agent class.
        agent_config_name: NAT config function name for the agent.
        trace_correlation: Session and submission identifiers used to correlate this independent job trace.
        available_documents: Optional list of document dicts with file_name and summary.
        data_sources: Optional list of allowed data sources to enforce in the worker.
        auth_token: Optional auth token propagated from the HTTP request for
            data sources that require authentication (requires_auth: true).
        content_encryption_policy: Non-secret policy identity captured by the
            submitting API process and required to match the worker configuration.
        initial_files: Optional DeepAgents virtual filesystem files to seed into state.
        output_metadata: Optional metadata to persist alongside the final report.
        owner_user_id: Canonical per-user key (``principal_user_id``), set on the NAT
            Context so per_user_mcp_client retrieves the token the owner connected
            via /v1/auth/mcp/{id}/connect.
        admission_token: Opaque deep-research fencing token captured at submit time.
    """

    trace_correlation = trace_correlation or JobTraceCorrelation()

    # Propagate auth token into the current async task's context so tools
    # can retrieve it via get_auth_token(). Uses a ContextVar so concurrent
    # jobs in the same Dask worker process don't leak tokens across tasks.
    _auth_token_reset = None
    _conversation_id_reset = None
    _user_id_reset = None
    context_state = None
    if auth_token:
        from ._auth_context import job_auth_token

        _auth_token_reset = job_auth_token.set(auth_token)

    from aiq_api.auth.request_trace import install_request_trace_span_injection
    from aiq_api.auth.request_trace import request_trace_tag_context

    install_request_trace_span_injection()

    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus
    from nat.front_ends.fastapi.async_jobs.job_store import JobStore
    from nat.runtime.loader import load_config

    if configure_logging:
        try:
            from nat.utils.log_utils import setup_logging

            setup_logging(log_level)
        except ImportError:
            import logging as std_logging

            std_logging.basicConfig(level=log_level)

    job_store: JobStore | None = None
    job_output_cipher = None
    cancellation_monitor: CancellationMonitor | None = None
    lease_stop: threading.Event | None = None
    lease_thread: threading.Thread | None = None
    event_store: EventStore | BatchingEventStore | None = None
    # Sandbox runtime is released on the terminal path; interrupted forces terminate() over close().
    sandbox_runtime: Any | None = None
    interrupted = False
    logger.info(
        "Dask worker received: agent=%s, config=%s, job_id=%s",
        agent_class_path,
        agent_config_name,
        job_id,
    )

    try:
        job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
        if admission_token is not None:
            from .admission import is_deep_research_reservation_current

            try:
                reservation_is_current = await asyncio.to_thread(
                    is_deep_research_reservation_current,
                    db_url,
                    job_id,
                    admission_token,
                )
            except Exception as exc:  # noqa: BLE001 - worker admission fails closed
                logger.warning(
                    "Could not verify admission fencing token for job %s (error_type=%s)",
                    job_id,
                    type(exc).__name__,
                )
                reservation_is_current = False
            if not reservation_is_current:
                logger.warning("Rejected job %s because its admission fencing token is no longer current", job_id)
                await job_store.update_status(job_id, JobStatus.FAILURE, error="submission admission lease lost")
                return
        try:
            from .crypto import ContentEncryptionError
            from .crypto import ContentEncryptionPolicyMismatch
            from .crypto import create_job_content_cipher
            from .crypto import require_content_encryption_policy

            require_content_encryption_policy(content_encryption_policy)
            job_output_cipher = create_job_content_cipher(job_id)
        except ContentEncryptionError as exc:
            logger.warning(
                "Job %s failed encryption policy/readiness before running exception=%s",
                job_id,
                exc.__class__.__name__,
            )
            error = (
                "content encryption policy mismatch"
                if isinstance(exc, ContentEncryptionPolicyMismatch)
                else "content encryption unavailable"
            )
            await job_store.update_status(job_id, JobStatus.FAILURE, error=error)
            return

        await job_store.update_status(job_id, JobStatus.RUNNING)

        # Start refreshing the reaper lease immediately, before the slow
        # initialization below stores any event, so a live worker in a long
        # cold start is not mistaken for a dead one. It runs on a dedicated
        # thread so synchronous init work can't starve it.
        lease_stop = threading.Event()
        lease_thread = threading.Thread(
            target=_run_lease_refresher,
            args=(db_url, job_id, lease_stop),
            name=f"job-lease-{job_id}",
            daemon=True,
        )
        lease_thread.start()

        cancellation_monitor = CancellationMonitor(
            scheduler_address=scheduler_address,
            db_url=db_url,
            job_id=job_id,
            poll_interval=1.0,
        )

        config = load_config(config_file_path)

        # Dynamically load the agent class
        agent_cls = _load_agent_class(agent_class_path)

        # Bind the submitted conversation before entering the builder context.
        # NAT function providers may run in child contexts created while tools
        # are built, so setting this later leaves knowledge tools on their
        # configured fallback collection for the entire async job.
        from nat.builder.context import ContextState

        context_state = ContextState.get()
        _conversation_id_reset = context_state.conversation_id.set(trace_correlation.session_id)
        # Always shadow the inherited identity, including for ownerless jobs,
        # so a reused worker context cannot expose a prior owner's MCP tokens.
        _user_id_reset = context_state.user_id.set(owner_user_id)

        async with WorkflowBuilder.from_config(config=config) as builder:
            await _attach_middleware_to_function(builder, config, agent_config_name)

            fn_config = builder.get_function_config(agent_config_name)
            relay_config = _resolve_job_relay_config(config, fn_config)
            if relay_config is not None:
                await _ensure_relay_started_for_job(relay_config, job_id)
            if getattr(fn_config, "type", None) == "deep_research_agent":
                from aiq_agent.agents.deep_researcher.register import DeepResearchAgentConfig
                from aiq_agent.agents.deep_researcher.register import resolve_deep_research_runtime_config

                if isinstance(fn_config, DeepResearchAgentConfig):
                    skills_config, sandbox_config = resolve_deep_research_runtime_config(fn_config, builder)
                    fn_config = fn_config.model_copy(update={"skills": skills_config, "sandbox": sandbox_config})

            provider, llm = await _create_llm_provider(builder, fn_config)

            # Resolve tools: use explicit list or auto-inherit from data_source_registry
            tool_refs = fn_config.tools
            if not tool_refs:
                from aiq_agent.common import get_all_tool_refs

                tool_refs = get_all_tool_refs()

            tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

            # Apply per-agent exclusions (e.g. deep_research excludes web_search_tool)
            if hasattr(fn_config, "exclude_tools") and fn_config.exclude_tools:
                excluded = set(fn_config.exclude_tools)
                tools = [t for t in tools if getattr(t, "name", "") not in excluded]

            if data_sources is not None:
                from aiq_agent.common import filter_tools_by_sources

                tools = filter_tools_by_sources(tools, data_sources)

            # Set up telemetry/observability for Phoenix and OpenTelemetry
            from nat.builder.context import Context
            from nat.data_models.intermediate_step import IntermediateStepPayload
            from nat.data_models.intermediate_step import IntermediateStepType
            from nat.data_models.intermediate_step import StreamEventData
            from nat.data_models.intermediate_step import TraceMetadata
            from nat.data_models.invocation_node import InvocationNode
            from nat.observability.exporter_manager import ExporterManager
            from nat.utils.reactive.subject import Subject

            telemetry_exporters = {
                name: configured.instance for name, configured in builder._telemetry_exporters.items()
            }
            exporter_manager = ExporterManager.from_exporters(telemetry_exporters)

            # A durable background job is an independent trace. The submitting
            # request is retained only as correlation metadata.
            context_state.workflow_run_id.set(job_id)

            workflow_trace_id = uuid.uuid4().int
            context_state.workflow_trace_id.set(workflow_trace_id)

            # Event stream for exporters to subscribe to
            event_stream = Subject()
            context_state.event_stream.set(event_stream)

            # Initialize span stack (triggers default ["root"])
            _ = context_state.active_span_id_stack

            # Set up span hierarchy metadata
            workflow_span_name = agent_config_name
            context_state.active_function.set(
                InvocationNode(
                    function_name=workflow_span_name,
                    function_id=job_id,
                )
            )

            context = Context(context_state)

            workflow_metadata = TraceMetadata(
                provided_metadata={
                    "workflow_run_id": job_id,
                    "workflow_trace_id": f"{workflow_trace_id:032x}",
                    "conversation_id": trace_correlation.session_id,
                    "agent": agent_class_path,
                    "submission_trace_id": trace_correlation.submission_trace_id,
                    "submission_span_id": trace_correlation.submission_span_id,
                }
            )

            # Run with telemetry - exporter must start before pushing events
            with request_trace_tag_context(trace_correlation.request_trace_tags):
                async with exporter_manager.start(context_state=context_state):
                    # Push WORKFLOW_START first so LLM/tool events become children
                    context.intermediate_step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            UUID=job_id,
                            event_type=IntermediateStepType.WORKFLOW_START,
                            name=workflow_span_name,
                            metadata=workflow_metadata,
                            data=StreamEventData(input=input_text),
                        )
                    )

                    callbacks: list[Any] = []

                    raw_event_store = EventStore(db_url, job_id, content_cipher=job_output_cipher)
                    event_store = BatchingEventStore(raw_event_store)
                    callbacks.append(AgentEventCallback(event_store))

                    # Resolve per-user MCP source tools for the job owner (Context.user_id
                    # set above); connections stay open via mcp_stack for the agent run.
                    # Best-effort: the helper never raises, so this can't break a job.
                    from contextlib import AsyncExitStack

                    from ..mcp_auth.runtime_tools import open_per_user_mcp_tools

                    async with AsyncExitStack() as mcp_stack:
                        mcp_tools = await open_per_user_mcp_tools(
                            builder=builder,
                            data_sources=data_sources,
                            exit_stack=mcp_stack,
                            wrapper_type=LLMFrameworkEnum.LANGCHAIN,
                        )
                        agent_tools = [*tools, *mcp_tools] if mcp_tools else tools

                        # Instantiate agent with callbacks
                        agent = _create_agent_instance(
                            agent_cls=agent_cls,
                            llm_provider=provider,
                            llm=llm,
                            tools=agent_tools,
                            fn_config=fn_config,
                            callbacks=callbacks,
                            job_id=job_id,
                            # Artifact harvesting rides 284's job store + event stream: the same db_url
                            # backs the SqlArtifactStore, and event_store.store carries artifact SSE
                            # events. Inert unless sandbox.artifact_capture is enabled in config.
                            artifact_db_url=db_url,
                            artifact_emit=event_store.store,
                        )

                        # Capture the runtime so the terminal path can release the sandbox. None for
                        # agents without a sandbox runtime; close()/terminate() are then no-ops.
                        sandbox_runtime = getattr(agent, "deepagents_runtime", None)

                        from aiq_agent.relay import run_workflow as run_relay_workflow

                        async def _execute_agent() -> Any:
                            return await _run_agent(
                                agent=agent,
                                input_text=input_text,
                                builder=builder,
                                config=config,
                                function_name=agent_config_name,
                                function_config=fn_config,
                                monitor=cancellation_monitor,
                                available_documents=available_documents,
                                data_sources=data_sources,
                                event_store=event_store,
                                initial_files=initial_files,
                                database_name=database_name,
                            )

                        result = await run_relay_workflow(
                            f"async_{agent_config_name.removesuffix('_agent')}_job",
                            _execute_agent,
                            session_id=trace_correlation.session_id,
                            input_value=input_text,
                            metadata={
                                "aiq.execution.mode": "async",
                                "aiq.job.id": job_id,
                                "aiq.agent.type": agent_config_name,
                                "aiq.submission.trace_id": trace_correlation.submission_trace_id,
                                "aiq.submission.span_id": trace_correlation.submission_span_id,
                            },
                        )

                    # Emit WORKFLOW_END event for Phoenix
                    context.intermediate_step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            UUID=job_id,
                            event_type=IntermediateStepType.WORKFLOW_END,
                            name=workflow_span_name,
                            metadata=workflow_metadata,
                            data=StreamEventData(output=_extract_result(result)),
                        )
                    )

                    # Signal event stream completion
                    event_stream.on_complete()

                    # Harvest artifacts (durable, idempotent) before SUCCESS so clients cannot
                    # stop streaming before the terminal metadata is persisted. Resource release
                    # is deferred to the finally block: the provider's close() is unbounded, so
                    # awaiting it here could strand a finished job in RUNNING if SDK cleanup hangs.
                    await asyncio.to_thread(
                        _harvest_sandbox_artifacts,
                        sandbox_runtime,
                        job_id=job_id,
                        interrupted=False,
                    )
                    if hasattr(event_store, "flush"):
                        event_store.flush()

                    # Extract report and update status inside the context manager
                    # so the UI sees completion before exporter flush and cleanup
                    report = _extract_result(result)
                    from .crypto import serialize_job_output_for_storage

                    if job_output_cipher is None:
                        raise RuntimeError("job output cipher was not initialized")
                    # Apply caller metadata first, then set the canonical report last so a
                    # stray "report" key in output_metadata can never overwrite the real report.
                    output = {**(output_metadata or {}), "report": report}
                    # Terminal state is immutable: write SUCCESS with a single
                    # compare-and-set (WHERE status='running'), so if the ghost
                    # reaper already marked this job FAILURE it is never
                    # resurrected. Serialize/encrypt exactly as update_job_output
                    # would, then do the guarded write.
                    try:
                        stored_output = serialize_job_output_for_storage(output, job_output_cipher)
                        wrote = await asyncio.get_running_loop().run_in_executor(
                            None, _write_job_success_if_running_sync, db_url, job_id, stored_output
                        )
                    except Exception as exc:
                        logger.warning(
                            "Job %s encrypted output write failed exception=%s",
                            job_id,
                            exc.__class__.__name__,
                        )
                        raise
                    if wrote:
                        logger.info("Job %s completed (report: %d chars)", job_id, len(report))
                    else:
                        logger.warning("Job %s already terminal; skipping success write", job_id)

    except asyncio.CancelledError:
        logger.info("Job %s cancelled", job_id)
        interrupted = True
        if event_store is None:
            event_store = BatchingEventStore(EventStore(db_url, job_id))

        await asyncio.to_thread(_teardown_sandbox, sandbox_runtime, job_id=job_id, interrupted=True)
        _store_terminal_event_best_effort(
            event_store,
            {
                "type": "job.cancelled",
                "data": {"reason": "cancelled by user"},
            },
        )

        if job_store:
            try:
                job = await job_store.get_job(job_id)
                if job and job.status != JobStatus.INTERRUPTED.value:
                    await job_store.update_status(job_id, JobStatus.INTERRUPTED, error="cancelled by user")
            except (ConnectionError, TimeoutError, RuntimeError):
                pass

    except EmptySourceRegistryError as e:
        logger.info("Job %s failed because no research sources were available (%s)", job_id, e.reason.value)
        if event_store is None:
            event_store = BatchingEventStore(EventStore(db_url, job_id, content_cipher=job_output_cipher))

        await asyncio.to_thread(_harvest_sandbox_artifacts, sandbox_runtime, job_id=job_id, interrupted=False)
        wrote = await _persist_empty_source_failure(
            error=e,
            job_output_cipher=job_output_cipher,
            db_url=db_url,
            job_id=job_id,
            event_store=event_store,
        )
        if wrote:
            logger.info("Job %s persisted source-failure outcome", job_id)
        else:
            logger.warning("Job %s already terminal or source-failure output persistence failed", job_id)

    except Exception as e:
        resource_timeout = isinstance(e, DeepResearchExecutionTimeout)
        if resource_timeout:
            # A timed-out graph may still have a blocking provider call running
            # outside the event loop. Terminate its sandbox before persisting the
            # failure so external execution cannot outlive the job deadline.
            interrupted = True
            await asyncio.to_thread(_teardown_sandbox, sandbox_runtime, job_id=job_id, interrupted=True)
        # Tracebacks include ``str(e)`` and can therefore expose provider
        # request content, credentials, or internal endpoints. Keep the error
        # type and a stable content fingerprint without emitting raw details.
        logger.error(
            "Job %s failed (error_type=%s detail_%s)",
            job_id,
            type(e).__name__,
            log_content_metadata(e),
        )
        if event_store is None:
            event_store = BatchingEventStore(EventStore(db_url, job_id))

        # Persist only the exception class name: raw messages can embed
        # credentials or internal hostnames, and both the event stream and the
        # stored status are surfaced to the job's caller.
        sanitized_error = f"job failed ({type(e).__name__}); check server logs for details"
        if not resource_timeout:
            await asyncio.to_thread(_harvest_sandbox_artifacts, sandbox_runtime, job_id=job_id, interrupted=False)
        _store_terminal_event_best_effort(
            event_store,
            {
                "type": "job.error",
                "data": {
                    "error": sanitized_error,
                    "error_type": type(e).__name__,
                },
            },
        )
        if job_store:
            await job_store.update_status(job_id, JobStatus.FAILURE, error=sanitized_error)

    finally:
        # Ensure terminal-path events are not left in the batch buffer.
        await _flush_event_store(event_store, job_id=job_id)
        if lease_stop is not None:
            lease_stop.set()
        if lease_thread is not None:
            await asyncio.to_thread(lease_thread.join, 5)
        if cancellation_monitor:
            cancellation_monitor.stop()
        # Idempotent fallback for failures before a terminal branch finalized the runtime.
        await asyncio.to_thread(_teardown_sandbox, sandbox_runtime, job_id=job_id, interrupted=interrupted)
        await _flush_event_store(event_store, job_id=job_id)
        # Restore job-scoped ContextVars for worker task reuse.
        if _user_id_reset is not None and context_state is not None:
            context_state.user_id.reset(_user_id_reset)
        if _conversation_id_reset is not None and context_state is not None:
            context_state.conversation_id.reset(_conversation_id_reset)
        if _auth_token_reset is not None:
            from ._auth_context import job_auth_token

            job_auth_token.reset(_auth_token_reset)


def _store_terminal_event_best_effort(event_store, event: dict) -> None:
    """Persist a terminal event without masking the job's terminal status."""
    try:
        event_store.store(event)
        if hasattr(event_store, "flush"):
            event_store.flush()
    except Exception as exc:
        logger.warning(
            "Failed to persist terminal event %s for job %s exception=%s",
            event.get("type", "unknown"),
            event_store.job_id,
            exc.__class__.__name__,
        )


def _harvest_sandbox_artifacts(sandbox_runtime: Any | None, *, job_id: str, interrupted: bool) -> None:
    """Persist captured artifacts on a terminal path without releasing the sandbox."""
    if sandbox_runtime is None:
        return
    finalize_artifacts = getattr(sandbox_runtime, "finalize_artifacts", None)
    if callable(finalize_artifacts):
        try:
            finalize_artifacts(interrupted=interrupted)
        except Exception as exc:  # noqa: BLE001 - artifact capture cannot replace the job result
            logger.warning(
                "Terminal artifact harvest failed for job %s exception=%s",
                job_id,
                exc.__class__.__name__,
            )


async def _flush_event_store(event_store: Any | None, *, job_id: str) -> None:
    """Flush terminal events off-loop without replacing the job result."""
    if event_store is None or not hasattr(event_store, "flush"):
        return
    try:
        await asyncio.to_thread(event_store.flush)
    except Exception as exc:  # noqa: BLE001 - terminal observability must not replace the job result
        logger.warning("Event store flush failed for job %s (%s)", job_id, type(exc).__name__)


def _teardown_sandbox(sandbox_runtime: Any | None, *, job_id: str, interrupted: bool) -> None:
    """Harvest artifacts and release sandbox resources on a terminal path.

    Prefers ``finalize(interrupted=...)`` when available. On the legacy fallback,
    interrupted/cancelled jobs call ``terminate()`` so a still-running ``execute`` is forcibly
    preempted; normal failure and success paths call ``close()`` gracefully. Both are idempotent.
    This runs off the event loop (``asyncio.to_thread``) so SDK cleanup cannot block the worker.
    """
    if sandbox_runtime is None:
        return
    _harvest_sandbox_artifacts(sandbox_runtime, job_id=job_id, interrupted=interrupted)
    finalize = getattr(sandbox_runtime, "finalize", None)
    if callable(finalize):
        try:
            if not finalize(interrupted=interrupted):
                logger.warning("Sandbox cleanup reported failure for job %s", job_id)
        except Exception as exc:  # noqa: BLE001 - cleanup must never replace the job result
            logger.warning("Sandbox cleanup failed for job %s (%s)", job_id, type(exc).__name__)
        return
    teardown = getattr(sandbox_runtime, "terminate", None) if interrupted else None
    if teardown is None:
        teardown = getattr(sandbox_runtime, "close", None)
    if teardown is None:
        return
    try:
        teardown()
    except Exception as exc:  # noqa: BLE001 - cleanup must never raise on the terminal path
        # Secret-safe: log only the exception type. A provider cleanup error can carry a
        # credential or internal hostname, which must never reach the logs (matches the
        # finalize_artifacts handler above).
        logger.warning("Sandbox cleanup failed for job %s (%s)", job_id, type(exc).__name__)


def _create_agent_instance(
    agent_cls: type,
    llm_provider,
    llm,
    tools: list,
    fn_config,
    callbacks: list,
    job_id: str | None = None,
    artifact_db_url: str | None = None,
    artifact_emit=None,
):
    """
    Create an agent instance, supporting different constructor patterns.

    Tries in order:
    1. DataScienceAgent explicit config pattern
    2. DeepResearcherAgent explicit config pattern
    3. llm_provider + tools + config/job_id pattern
    4. llm_provider + tools + job_id pattern
    5. ShallowResearcherAgent config pattern
    6. llm_provider + tools pattern
    7. llm + tools pattern (simpler agents)
    """
    from aiq_agent.agents.data_science.register import DataScienceAgentConfig
    from aiq_agent.agents.deep_researcher.register import DeepResearchAgentConfig
    from aiq_agent.agents.shallow_researcher.register import ShallowResearchAgentConfig

    if isinstance(fn_config, DataScienceAgentConfig) and _constructor_accepts_explicit_kwargs(
        agent_cls, _DATA_SCIENCE_AGENT_KWARGS
    ):
        # An async job has no channel back to the user, so a clarifying question
        # would strand the job. Clarification is a pre-submission concern (the
        # chat workflow's clarifier), exactly as it is for deep research.
        if fn_config.interaction_mode != "headless":
            logger.info(
                "Running data_science_agent headless for job %s (configured interaction_mode=%s)",
                job_id,
                fn_config.interaction_mode,
            )
        return agent_cls(
            llm=llm,
            tools=tools,
            callbacks=callbacks,
            recursion_limit=fn_config.recursion_limit,
            interaction_mode="headless",
            response_mode=fn_config.response_mode,
            gsf_catalog_call_limit=fn_config.gsf_catalog_call_limit,
            gsf_text_to_sql_call_limit=fn_config.gsf_text_to_sql_call_limit,
            gsf_cache_repeated_calls=fn_config.gsf_cache_repeated_calls,
            python_call_limit=fn_config.python_call_limit,
            finalization_model_call_limit=fn_config.finalization_model_call_limit,
        )

    if isinstance(fn_config, DeepResearchAgentConfig) and _constructor_accepts_explicit_kwargs(
        agent_cls, _DEEP_RESEARCH_AGENT_KWARGS
    ):
        return agent_cls(
            llm_provider=llm_provider,
            tools=tools,
            callbacks=callbacks,
            domain_catalog_path=fn_config.domain_catalog_path,
            enable_source_router=fn_config.enable_source_router,
            enable_citation_verification=fn_config.enable_citation_verification,
            skills=fn_config.skills,
            sandbox=fn_config.sandbox,
            job_id=job_id,
            artifact_db_url=artifact_db_url,
            artifact_emit=artifact_emit,
            max_research_concurrency=fn_config.max_research_concurrency,
            max_concurrent_source_tool_calls=fn_config.max_concurrent_source_tool_calls,
            max_source_tool_batch_size=fn_config.max_source_tool_batch_size,
            resource_limits=fn_config.resource_limits,
        )

    if _constructor_accepts_explicit_kwargs(agent_cls, _CONFIGURABLE_AGENT_KWARGS):
        try:
            return agent_cls(
                llm_provider=llm_provider,
                tools=tools,
                callbacks=callbacks,
                config=fn_config,
                job_id=job_id,
            )
        except TypeError:
            pass

    if _constructor_accepts_explicit_kwargs(agent_cls, _JOB_SCOPED_AGENT_KWARGS):
        try:
            return agent_cls(
                llm_provider=llm_provider,
                tools=tools,
                callbacks=callbacks,
                job_id=job_id,
            )
        except TypeError:
            pass

    if isinstance(fn_config, ShallowResearchAgentConfig) and _constructor_accepts_explicit_kwargs(
        agent_cls, _SHALLOW_RESEARCH_AGENT_KWARGS
    ):
        return agent_cls(
            llm_provider=llm_provider,
            tools=tools,
            max_tool_iterations=fn_config.max_tool_iterations,
            enforce_citations=fn_config.enforce_citations,
            callbacks=callbacks,
        )

    # Try the common llm_provider + tools pattern.
    try:
        return agent_cls(
            llm_provider=llm_provider,
            tools=tools,
            callbacks=callbacks,
        )
    except TypeError:
        pass

    # Try simpler llm + tools pattern
    try:
        return agent_cls(
            llm=llm,
            tools=tools,
            callbacks=callbacks,
        )
    except TypeError:
        pass

    # Fallback: just callbacks
    return agent_cls(callbacks=callbacks)


async def _run_agent(
    agent,
    input_text: str,
    monitor: CancellationMonitor,
    builder: Any | None = None,
    config: Any | None = None,
    function_name: str | None = None,
    function_config: Any | None = None,
    available_documents: list[dict] | None = None,
    data_sources: list[str] | None = None,
    event_store: EventStore | None = None,
    initial_files: dict[str, Any] | None = None,
    database_name: str | None = None,
) -> Any:
    """
    Run the agent, supporting different run() signatures.

    Tries:
    1. run(input_text: str) -> str (simple protocol)
    2. run(state) where state has messages (LangGraph pattern)
    """
    from langchain_core.messages import HumanMessage

    # Check if agent has a simple run(input_text) method
    if hasattr(agent, "run"):
        import inspect

        sig = inspect.signature(agent.run)
        params = list(sig.parameters.keys())

        # If first param is 'input_text' or 'query', use simple pattern
        if params and params[0] in ("input_text", "query", "input"):
            return await run_with_cancellation(
                agent.run(input_text),
                monitor,
                event_store=event_store,
            )

        # Otherwise assume state-based pattern
        # Try to find the agent's state class
        state_cls = _get_agent_state_class(agent)
        if state_cls:
            # Build state with available_documents if the class supports it
            state_kwargs = {"messages": [HumanMessage(content=input_text)]}
            # Only pass optional fields the state actually declares. data_sources also
            # flows to non-Pydantic states (legacy behavior); files needs a model field.
            has_fields = hasattr(state_cls, "model_fields")
            if data_sources is not None and (not has_fields or "data_sources" in state_cls.model_fields):
                state_kwargs["data_sources"] = data_sources
            if initial_files and has_fields and "files" in state_cls.model_fields:
                state_kwargs["files"] = initial_files
            # A router-resolved database scope is authoritative for the request, so the
            # worker must not fall back to the agent's configured default.
            if database_name and has_fields and "database_name" in state_cls.model_fields:
                state_kwargs["database_name"] = database_name
            if available_documents:
                # Convert dicts to AvailableDocument if the state class expects them
                try:
                    from aiq_agent.knowledge import AvailableDocument

                    state_kwargs["available_documents"] = [AvailableDocument(**doc) for doc in available_documents]
                    logger.debug(
                        "Dask worker passing %d available documents to agent state",
                        len(available_documents),
                    )
                except (ImportError, TypeError):
                    # AvailableDocument not available or state doesn't support it
                    pass
            state = state_cls(**state_kwargs)
        else:
            # Fallback: create a simple dict state
            state = {"messages": [HumanMessage(content=input_text)]}
            if data_sources is not None:
                state["data_sources"] = data_sources
            if initial_files:
                state["files"] = initial_files
            if database_name:
                state["database_name"] = database_name
            if available_documents:
                state["available_documents"] = available_documents

        async def call_next(current_state: Any) -> Any:
            return await run_with_cancellation(
                agent.run(current_state),
                monitor,
                event_store=event_store,
            )

        if builder is None or config is None or function_name is None or function_config is None:
            return await call_next(state)

        return await _run_with_configured_function_middleware(
            builder=builder,
            config=config,
            function_name=function_name,
            function_config=function_config,
            input_value=state,
            call_next=call_next,
        )

    raise TypeError(f"Agent {type(agent).__name__} does not have a run method")


def _get_agent_state_class(agent) -> type | None:
    """Try to find the state class for an agent."""
    agent_module = type(agent).__module__
    agent_name = type(agent).__name__

    # Try common patterns for state class names
    # e.g., DeepResearcherAgent -> DeepResearchAgentState, DeepResearcherAgentState
    state_name_patterns = [
        "AgentState",
        f"{agent_name}State",
        f"{agent_name.replace('Agent', '')}AgentState",  # DeepResearcher -> DeepResearcherAgentState
        f"{agent_name.replace('erAgent', '')}AgentState",  # DeepResearcherAgent -> DeepResearchAgentState
        "State",
    ]

    # Try models submodule first
    try:
        models_module = importlib.import_module(agent_module.replace(".agent", ".models"))
        for state_name in state_name_patterns:
            if hasattr(models_module, state_name):
                return getattr(models_module, state_name)

        # Also scan for any class ending with "State" that has a messages field
        for name in dir(models_module):
            if name.endswith("State") and not name.startswith("_"):
                cls = getattr(models_module, name)
                if isinstance(cls, type) and hasattr(cls, "model_fields"):
                    if "messages" in cls.model_fields:
                        return cls
    except (ImportError, AttributeError):
        pass

    # Try same module
    try:
        module = importlib.import_module(agent_module)
        for state_name in state_name_patterns:
            if hasattr(module, state_name):
                return getattr(module, state_name)
    except ImportError:
        pass

    return None


def _extract_result(result: Any) -> str:
    """Extract string result from various result formats."""
    # Direct string
    if isinstance(result, str):
        return result

    # State with messages
    if hasattr(result, "messages") and result.messages:
        last_msg = result.messages[-1]
        if hasattr(last_msg, "content"):
            return str(last_msg.content)

    # Dict with messages
    if isinstance(result, dict):
        if "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                return str(last_msg.content)
        if "report" in result:
            return str(result["report"])
        if "output" in result:
            return str(result["output"])

    return str(result) if result else ""


# Backwards compatibility alias
async def run_deep_research(
    configure_logging: bool,
    log_level: int,
    scheduler_address: str,
    db_url: str,
    config_file_path: str,
    job_id: str,
    input_text: str,
):
    """
    Legacy function for running deep research jobs.

    Preserved for backwards compatibility. New code should use run_agent_job directly.
    """
    from .crypto import get_content_encryption_policy_identity

    await run_agent_job(
        configure_logging=configure_logging,
        log_level=log_level,
        scheduler_address=scheduler_address,
        db_url=db_url,
        config_file_path=config_file_path,
        job_id=job_id,
        input_text=input_text,
        agent_class_path="aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        agent_config_name="deep_research_agent",
        content_encryption_policy=get_content_encryption_policy_identity(),
    )
