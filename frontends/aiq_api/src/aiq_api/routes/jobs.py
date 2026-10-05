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
Agent-agnostic async job API routes.

Routes:
    GET  /v1/jobs/async/agents                            - List available agent types
    POST /v1/jobs/async/submit                            - Submit a new job for any agent
    GET  /v1/jobs/async/job/{job_id}                      - Get job status
    GET  /v1/jobs/async/job/{job_id}/stream               - SSE stream from beginning
    GET  /v1/jobs/async/job/{job_id}/stream/{last_event_id} - SSE stream from event ID
    POST /v1/jobs/async/job/{job_id}/cancel               - Cancel running job
    GET  /v1/jobs/async/job/{job_id}/state                - Get artifacts from event store
    GET  /v1/jobs/async/job/{job_id}/report               - Get final report
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

from fastapi import Body
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from aiq_agent.common.data_source_registry import get_all_sources
from aiq_agent.common.data_source_registry import get_all_tool_refs
from aiq_agent.common.data_source_registry import get_source_id_for_tool
from nat.builder.framework_enum import LLMFrameworkEnum

from ..jobs.access import require_verified_principal
from ..mcp_auth.models import PerUserAuthInfo
from ..registry import AGENT_REGISTRY
from ..registry import get_agent_config

if TYPE_CHECKING:
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)

_ASYNC_JOB_READINESS_TIMEOUT_SECONDS = 3.0
_READINESS_JOB_ID = "__aiq_readiness_probe__"
_REQUIRED_ASYNC_JOB_TABLES = frozenset({"job_info", "job_access", "job_events", "artifacts", "deep_research_admission"})


def _is_readable_regular_file(path: str) -> bool:
    """Return whether ``path`` names a readable regular file."""
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as config_file:
            config_file.read(1)
    except OSError:
        return False
    return True


def _scheduler_info(job_store: Any) -> Any:
    """Perform one synchronous scheduler RPC (called from a worker thread)."""
    client = job_store.dask_client
    return client.sync(
        client.scheduler.identity,
        callback_timeout=_ASYNC_JOB_READINESS_TIMEOUT_SECONDS,
    )


async def _table_names(db_url: str) -> set[str]:
    """Return database table names through AI-Q's shared async engine."""
    from sqlalchemy import inspect

    from ..jobs.event_store import EventStore

    engine = EventStore._get_or_create_async_engine(db_url)
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))


async def _bootstrap_async_job_storage(db_url: str, job_store: Any) -> None:
    """Initialize owned tables and verify NAT's mapped ``job_info`` contract."""
    from nat.front_ends.fastapi.async_jobs.job_store import JobInfo

    from ..jobs.access import ensure_job_access_table
    from ..jobs.admission import ensure_deep_research_admission_table
    from ..jobs.event_store import EventStore

    engine = EventStore._get_or_create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(JobInfo.metadata.create_all, checkfirst=True)

    await asyncio.to_thread(ensure_job_access_table, db_url)
    await asyncio.to_thread(ensure_deep_research_admission_table, db_url)
    await asyncio.to_thread(_validate_artifact_store, db_url)
    await asyncio.to_thread(EventStore._ensure_table_exists, db_url)

    tables = await _table_names(db_url)
    missing_tables = _REQUIRED_ASYNC_JOB_TABLES - tables
    if missing_tables:
        raise RuntimeError(f"Required async-job tables are unavailable: {sorted(missing_tables)}")

    # A mapped read verifies every JobInfo column. ``create_all(checkfirst=True)``
    # intentionally creates a missing table but never mutates an incomplete one.
    await job_store.get_job(_READINESS_JOB_ID)


async def _probe_async_job_readiness(
    *,
    dask_available: bool,
    job_store: Any,
    scheduler_address: str | None,
    db_url: str,
    config_path: str,
    submit_route_registered: bool,
) -> dict[str, str] | None:
    """Run the uncached async-job readiness contract under one total budget."""
    db_status = "unchecked"

    async def _probe() -> dict[str, str] | None:
        nonlocal db_status

        if not dask_available or job_store is None or not scheduler_address:
            return {"reason": "async_jobs_unavailable", "db": db_status}

        if not await asyncio.to_thread(_is_readable_regular_file, config_path):
            return {"reason": "configuration_missing", "db": db_status}

        db_status = "unreachable"
        try:
            tables = await _table_names(db_url)
        except Exception as exc:
            logger.warning("Async-job readiness database connection failed error_type=%s", type(exc).__name__)
            return {"reason": "async_jobs_unavailable", "db": db_status}

        db_status = "schema_unavailable"
        missing_tables = _REQUIRED_ASYNC_JOB_TABLES - tables
        if missing_tables:
            logger.warning("Async-job readiness is missing required database tables: %s", sorted(missing_tables))
            return {"reason": "async_jobs_unavailable", "db": db_status}

        try:
            from ..jobs.access import validate_job_access_table

            await asyncio.to_thread(validate_job_access_table, db_url)
        except Exception as exc:
            logger.warning("Async-job readiness access schema read failed error_type=%s", type(exc).__name__)
            return {"reason": "async_jobs_unavailable", "db": db_status}

        try:
            from ..jobs.admission import validate_deep_research_admission_table

            await asyncio.to_thread(validate_deep_research_admission_table, db_url)
        except Exception as exc:
            logger.warning("Async-job readiness admission schema read failed error_type=%s", type(exc).__name__)
            return {"reason": "async_jobs_unavailable", "db": db_status}

        try:
            await job_store.get_job(_READINESS_JOB_ID)
        except Exception as exc:
            logger.warning("Async-job readiness JobStore mapped read failed error_type=%s", type(exc).__name__)
            return {"reason": "async_jobs_unavailable", "db": db_status}

        db_status = "ok"
        try:
            await asyncio.to_thread(_scheduler_info, job_store)
        except Exception as exc:
            logger.warning("Async-job readiness scheduler RPC failed error_type=%s", type(exc).__name__)
            return {"reason": "async_jobs_unavailable", "db": db_status}

        if not submit_route_registered:
            return {"reason": "async_jobs_unavailable", "db": db_status}

        return None

    try:
        async with asyncio.timeout(_ASYNC_JOB_READINESS_TIMEOUT_SECONDS):
            return await _probe()
    except TimeoutError:
        logger.warning("Async-job readiness probe exceeded %.1fs", _ASYNC_JOB_READINESS_TIMEOUT_SECONDS)
        return {"reason": "async_jobs_unavailable", "db": db_status}


def _remove_existing_health_routes(app: FastAPI) -> int:
    """Remove existing GET /health routes before installing AI-Q readiness."""
    existing_routes = [
        route
        for route in app.router.routes
        if isinstance(route, APIRoute) and route.path == "/health" and "GET" in route.methods
    ]
    for route in existing_routes:
        app.router.routes.remove(route)
    if existing_routes:
        logger.info("Replacing %d existing GET /health route(s) with AI-Q readiness", len(existing_routes))
    app.openapi_schema = None
    return len(existing_routes)


def _validate_artifact_store(db_url: str) -> None:
    """Validate configured artifact storage during API startup."""
    from aiq_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store

    build_artifact_store(db_url).validate()


def _int_env(name: str, default: int) -> int:
    """Read a non-negative integer ops knob from the environment.

    A missing, non-integer, or negative value falls back to ``default`` so a
    misconfigured cap can never silently invert into "block all submissions".
    """
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        return default
    if value < 0:
        logger.warning("%s=%d is negative; using default %d", name, value, default)
        return default
    return value


def _sandbox_caps_configured() -> bool:
    """Whether an operator has opted into sandbox concurrency caps via env.

    Default-off so the guard never adds a function-config lookup (or behavior change)
    to submits unless caps are explicitly configured.
    """
    return "AIQ_MAX_SANDBOXES_PER_PRINCIPAL" in os.environ or "AIQ_MAX_SANDBOXES_GLOBAL" in os.environ


def _sandbox_enabled(sandbox: Any) -> bool:
    """Return whether a resolved sandbox config is active."""
    if sandbox is None:
        return False
    return bool(getattr(sandbox, "enabled", True))


def _tool_config_uses_sandbox(builder: Any, fn_config: Any) -> bool:
    """Return whether any tool the agent can resolve owns a sandbox.

    Agents such as ``data_science`` never declare ``sandbox`` themselves: the
    analysis tool (``stateful_python``) holds the sandbox ``FunctionRef``. Walk
    the agent's effective tool refs so the submit-path cap covers those jobs
    too, mirroring how the worker resolves tools.
    """
    tool_refs = getattr(fn_config, "tools", None) or get_all_tool_refs()
    # exclude_tools holds exact runtime tool names while tool_refs holds function and
    # group references. These coincide for plain-function tools, which is every tool
    # that owns a sandbox today (stateful_python, registered under its function name).
    # A group reference such as `gsf` would not match a child name such as
    # `gsf__text_to_sql`, but no function group owns a sandbox. Matching exactly here
    # keeps this off the async builder.get_tools() path on every submit; if a group
    # ever owns a sandbox, resolve runtime names instead. Erring toward "uses a
    # sandbox" only over-applies an opt-in cap rather than letting one escape it.
    excluded = set(getattr(fn_config, "exclude_tools", None) or [])
    for tool_ref in tool_refs:
        if tool_ref in excluded:
            continue
        try:
            tool_config = builder.get_function_config(tool_ref)
        except Exception:  # noqa: BLE001 - an unresolvable ref cannot enable a sandbox
            continue
        sandbox = getattr(tool_config, "sandbox", None)
        if isinstance(sandbox, str):
            # A FunctionRef naming a separate sandbox function config.
            try:
                sandbox = builder.get_function_config(sandbox)
            except Exception:  # noqa: BLE001 - same rationale as above
                continue
        if _sandbox_enabled(sandbox):
            return True
    return False


def _agent_uses_sandbox(builder: Any, config_name: str) -> bool:
    """Return whether the agent reaches a sandbox directly or through a tool."""
    try:
        fn_config = builder.get_function_config(config_name)
    except Exception:  # noqa: BLE001 - missing/odd config means "no sandbox guard"
        return False
    if _sandbox_enabled(getattr(fn_config, "sandbox", None)):
        return True
    return _tool_config_uses_sandbox(builder, fn_config)


async def _enforce_sandbox_concurrency(db_url: str, principal: Any) -> None:
    """Reject submission when per-principal or global sandbox limits are reached.

    Option A: enforced at the API submit path so cost is stopped before a Dask worker
    spins up a sandbox. Counts fail open (None) so a query mismatch never blocks submits.
    Configurable via AIQ_MAX_SANDBOXES_PER_PRINCIPAL / AIQ_MAX_SANDBOXES_GLOBAL.
    """
    from ..jobs.access import count_active_jobs_for_owner
    from ..jobs.access import count_active_jobs_global

    per_principal = _int_env("AIQ_MAX_SANDBOXES_PER_PRINCIPAL", 5)
    global_cap = _int_env("AIQ_MAX_SANDBOXES_GLOBAL", 50)
    loop = asyncio.get_running_loop()

    owner_count = await loop.run_in_executor(None, count_active_jobs_for_owner, db_url, principal)
    if owner_count is not None and owner_count >= per_principal:
        raise HTTPException(
            429,
            f"Active job limit reached for this principal ({per_principal}). "
            "Wait for running jobs to finish before submitting more.",
        )

    global_count = await loop.run_in_executor(None, count_active_jobs_global, db_url)
    if global_count is not None and global_count >= global_cap:
        raise HTTPException(503, "Server is at sandbox capacity; please retry shortly.")


class JobSubmitRequest(BaseModel):
    """Request to submit an async job."""

    agent_type: str = Field(..., description="Agent type (e.g., 'deep_researcher')")
    input: str = Field(..., min_length=1, description="Input query for the agent")
    job_id: str | None = Field(
        None,
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=64,
        description="Optional custom job ID (auto-generated if omitted)",
    )
    expiry_seconds: int | None = Field(
        None,
        ge=600,
        le=604800,
        description="Job expiry in seconds (default from config, max 7 days)",
    )
    data_sources: list[str] | None = Field(
        None,
        description=(
            "Optional data source IDs to target. Omit or set null to use all data-source tools "
            "available to the chosen agent. When specific IDs are passed, unmapped utility tools "
            "(e.g., 'think') remain available. Pass an empty list to run the agent with no "
            "data-source tools; unmapped utility tools remain available."
        ),
    )

    @field_validator("input")
    @classmethod
    def _input_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Input must not be blank")
        return stripped


JOB_SUBMIT_EXAMPLES: dict[str, dict] = {
    "default": {
        "summary": "Default (all data sources)",
        "value": {
            "agent_type": "deep_researcher",
            "input": "What are the latest advances in quantum computing?",
            "expiry_seconds": 86400,
        },
    },
    "scoped": {
        "summary": "Scoped to specific data sources",
        "value": {
            "agent_type": "deep_researcher",
            "input": "What are the latest advances in quantum computing?",
            "data_sources": ["web_search"],
        },
    },
}


def _source_ids_by_lowercase() -> tuple[list[str], dict[str, str]]:
    """Return known source IDs and a lower-case lookup preserving canonical IDs.

    Assumes registry IDs are unique under ``.lower()``. The data source registry
    convention is snake_case (e.g. ``web_search``, ``knowledge_layer``); two IDs
    differing only by case would collapse here.
    """
    known_ids = sorted(source.id for source in get_all_sources())
    return known_ids, {source_id.lower(): source_id for source_id in known_ids}


def _get_configured_agent_function_config(builder: WorkflowBuilder, config_name: str) -> Any | None:
    """Return an active function config, or None for NAT's missing-function error."""
    try:
        return builder.get_function_config(config_name)
    except ValueError as exc:
        if str(exc) == f"Function `{config_name}` not found":
            return None
        raise


async def _get_agent_available_source_ids(builder: WorkflowBuilder, fn_config: Any) -> list[str]:
    """Return mapped source IDs with at least one effective tool for an agent config.

    This mirrors the async job runner's effective tool resolution: explicit
    `tools` wins and overrides registry refs, otherwise inherit all registry
    refs, resolve LangChain wrappers through the builder, then apply exact
    tool-name `exclude_tools`.

    A source is reported as available if at least one of its tools survives
    ``exclude_tools``; partial exclusion does not hide the source.

    Assumes agent configs registered for async submission expose typed
    ``tools`` and ``exclude_tools`` fields (see ``aiq_agent.agents.*.register``).
    A registered agent without these fields is a registration-time bug, not a
    runtime concern.
    """
    tool_refs = fn_config.tools or get_all_tool_refs()
    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    excluded = set(fn_config.exclude_tools or [])
    if excluded:
        tools = [tool for tool in tools if getattr(tool, "name", "") not in excluded]

    source_ids: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", "")
        if not name:
            continue
        sid = get_source_id_for_tool(name)
        if sid is not None:
            source_ids.add(sid)

    # Per-user MCP sources (e.g. Google Drive) contribute NO static tools — their
    # tools are resolved per-user at run time by open_per_user_mcp_tools, so they
    # never appear in the loop above. Treat a configured protected source as an
    # available runtime candidate so submit validation doesn't 422 it; connectivity
    # is enforced separately by the MCP auth preflight (409 mcp_auth_required).
    from aiq_agent.common.data_source_registry import get_all_sources

    for source in get_all_sources():
        pua = source.per_user_auth
        if pua is not None and pua.required:
            source_ids.add(source.id)
    return sorted(source_ids)


async def _validate_data_sources_for_agent(
    *,
    builder: WorkflowBuilder,
    agent_type: str,
    agent_config_name: str,
    fn_config: Any,
    data_sources: list[str] | None,
) -> None:
    """Raise HTTP 422 if requested sources are unknown or unavailable to the selected agent."""
    # Semantic fast path: omit/null/empty means "use all data-source tools available
    # to the chosen agent" (or, for empty list, "use no data-source tools"). In both
    # cases there is nothing for the caller to validate against, so we skip.
    #
    # Bonus: this also avoids a builder.get_tools() round-trip on the default code
    # path -- pinned by test_submit_job_forwards_omitted_data_sources_without_resolving_tools.
    if not data_sources:
        return

    known_ids, known_by_lower = _source_ids_by_lowercase()

    try:
        available_ids = await _get_agent_available_source_ids(builder, fn_config)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to validate data sources for agent %s using config %s",
            agent_type,
            agent_config_name,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to validate data sources for selected agent",
        ) from exc

    available_by_lower = {source_id.lower(): source_id for source_id in available_ids}

    # Single-pass partition: walk requested IDs once, deduping case-insensitively
    # and routing each unique ID into either "unknown to system" or "known but
    # unavailable to this agent." Preserves first-seen casing and request order.
    seen: set[str] = set()
    invalid_ids: list[str] = []
    unavailable_for_agent: list[str] = []
    for source_id in data_sources:
        key = source_id.lower()
        if key in seen:
            continue
        seen.add(key)
        if key not in known_by_lower:
            invalid_ids.append(source_id)
        elif key not in available_by_lower:
            unavailable_for_agent.append(source_id)

    if not invalid_ids and not unavailable_for_agent:
        return

    parts: list[str] = []
    if invalid_ids:
        parts.append(f"Unknown data source(s): {', '.join(invalid_ids)}")
    if unavailable_for_agent:
        parts.append(f"Data source(s) are not available for agent '{agent_type}': {', '.join(unavailable_for_agent)}")
    message = ". ".join(parts)

    # Echo back the caller's request annotated with which IDs were unknown vs
    # unavailable, plus the global registry list (which is also discoverable via
    # /v1/data_sources). The per-agent capability list is intentionally NOT
    # returned -- it's not exposed anywhere else and would reveal agent
    # capability boundaries.
    raise HTTPException(
        status_code=422,
        detail={
            "message": message,
            "invalid_ids": invalid_ids,
            "unavailable_for_agent": unavailable_for_agent,
            "known_ids": known_ids,
        },
    )


async def _preflight_mcp_auth(provider, principal, data_sources: list[str] | None):
    """Return a 409 JSONResponse if any selected protected source is not connected, else None.

    Thin HTTP wrapper over the shared :func:`evaluate_mcp_auth`; the same check
    runs inside ``submit_agent_job`` (raising instead) so programmatic submitters
    cannot bypass it. Source existence is validated earlier by
    ``_validate_data_sources_for_agent``.
    """
    from fastapi.responses import JSONResponse

    from ..mcp_auth.preflight import evaluate_mcp_auth

    body = await evaluate_mcp_auth(provider, principal, data_sources)
    if body is None:
        return None
    return JSONResponse(status_code=409, content=body.model_dump(mode="json"))


class JobStatusResponse(BaseModel):
    """Job status response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "job_id": "abc123",
                    "status": "submitted",
                    "agent_type": "deep_researcher",
                    "error": None,
                    "created_at": "2026-02-12T10:30:00Z",
                }
            ]
        }
    )

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(
        ...,
        description="Current status: submitted, running, success, failure, interrupted, not_found",
    )
    agent_type: str | None = Field(None, description="Agent type used for this job")
    error: str | None = Field(None, description="Error message if job failed")
    created_at: str | None = Field(None, description="Creation timestamp (ISO format)")


class JobStateResponse(BaseModel):
    """Job state response with artifacts."""

    job_id: str = Field(..., description="Unique job identifier")
    has_state: bool = Field(..., description="Whether state/artifacts are available")
    state: dict | None = Field(None, description="Internal job state")
    artifacts: dict | None = Field(None, description="Tool calls, outputs, and sources collected during execution")


class JobReportResponse(BaseModel):
    """Final report response."""

    job_id: str = Field(..., description="Unique job identifier")
    has_report: bool = Field(..., description="Whether the final report is available")
    report: str | None = Field(None, description="Final research report from the agent")
    parent_job_id: str | None = Field(None, description="Parent report job ID for report follow-up outputs")
    interaction_action: str | None = Field(None, description="Report interaction action that produced this output")
    result_kind: str | None = Field(None, description="Kind of result returned by the child job")


class ReportEditRequest(BaseModel):
    """Request to create a revised report from an existing completed report job."""

    input: str = Field(..., min_length=1, description="Edit instruction for the parent report")
    job_id: str | None = Field(
        None,
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=64,
        description="Optional custom child job ID (auto-generated if omitted)",
    )
    expiry_seconds: int | None = Field(
        None,
        ge=600,
        le=604800,
        description="Child job expiry in seconds (default from config, max 7 days)",
    )

    @field_validator("input")
    @classmethod
    def _input_not_blank(cls, value: str) -> str:
        # min_length=1 still allows whitespace-only; the report rewriter requires a
        # real instruction, so reject blank input at the boundary instead of
        # creating a guaranteed-failing child job.
        stripped = value.strip()
        if not stripped:
            raise ValueError("Edit instruction must not be blank")
        return stripped


class ReportEditResponse(BaseModel):
    """Response for an accepted report edit job."""

    job_id: str = Field(..., description="Child job identifier")
    parent_job_id: str = Field(..., description="Parent report job identifier")
    status: str = Field(..., description="Child job status")
    agent_type: str = Field(..., description="Internal agent type used for the child job")


class AgentInfo(BaseModel):
    """Information about a registered agent."""

    agent_type: str = Field(..., description="Agent identifier used in submit requests")
    description: str = Field(..., description="Human-readable description of the agent")


class AgentListResponse(BaseModel):
    """List of available agents."""

    agents: list[AgentInfo] = Field(..., description="Public agent types configured in the active workflow")


class DataSource(BaseModel):
    """Information about an available data source."""

    id: str = Field(..., description="Unique identifier for the data source")
    name: str = Field(..., description="Display name")
    description: str | None = Field(default=None, description="Human-readable description")
    default_enabled: bool = Field(
        default=True,
        description="Whether the source is toggled on by default in the UI (from registry metadata)",
    )
    requires_auth: bool = Field(default=False, description="Whether user authentication is required")
    per_user_auth: PerUserAuthInfo | None = Field(
        default=None,
        description="Per-user MCP OAuth state for a protected source (omitted for unprotected sources)",
    )


async def register_job_routes(app: FastAPI, builder: WorkflowBuilder, worker: FastApiFrontEndPluginWorker) -> None:
    """
    Register agent-agnostic async job routes.

    Uses NAT's JobStore for job metadata and Dask for distributed execution.
    The /v1/data_sources endpoint is always registered regardless of Dask availability.
    """
    import os

    from aiq_agent.common.data_source_registry import get_all_sources
    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

    from ..jobs.access import authorize_job_access
    from ..jobs.admission import JobAdmissionError
    from ..jobs.crypto import ContentEncryptionConfigError
    from ..jobs.crypto import ContentEncryptionInvalidData
    from ..jobs.crypto import ContentEncryptionUnavailable
    from ..jobs.crypto import get_content_encryption_health_async
    from ..jobs.crypto import read_job_output_async
    from ..jobs.crypto import require_content_encryption_ready_for_submission_async
    from ..jobs.crypto import validate_content_encryption_startup_async
    from ..jobs.event_store import EventStore
    from ..jobs.report_context import report_output_metadata
    from ..jobs.report_context import resolve_report_context
    from ..jobs.report_context import to_initial_files
    from ..jobs.submit import JobIdConflictError
    from ..jobs.submit import submit_agent_job as submit_authorized_job
    from ..mcp_auth.factory import build_mcp_auth_provider
    from ..mcp_auth.preflight import McpAuthRequiredError
    from ..mcp_auth.serialize import build_listing_auth_info
    from .auth import register_mcp_auth_routes

    # Per-user MCP auth control plane. The provider is shared by the data-source
    # listing, the status/connect/callback routes, and submit preflight so a flow
    # started via /connect can be completed by /callback in the same process.
    mcp_auth_provider = await build_mcp_auth_provider(builder)
    # Publish the provider process-wide so submit_agent_job() can run the same
    # connect-state preflight for programmatic submitters, not just this REST route.
    from ..mcp_auth.active import set_active_mcp_auth_provider

    set_active_mcp_auth_provider(mcp_auth_provider)
    register_mcp_auth_routes(app, mcp_auth_provider)

    await validate_content_encryption_startup_async()

    @app.get(
        "/live",
        tags=["health"],
        summary="Liveness check",
        description="Returns success while the API process is running; does not check external dependencies.",
    )
    async def liveness_check() -> dict[str, str]:
        """Report process liveness without coupling restarts to dependency health."""

        return {"status": "alive"}

    dask_available = getattr(worker, "_dask_available", False)
    job_store = getattr(worker, "_job_store", None)
    scheduler_address = getattr(worker, "_scheduler_address", None) or os.environ.get("NAT_DASK_SCHEDULER_ADDRESS")
    db_url = getattr(worker, "_db_url", None) or os.environ.get("NAT_JOB_STORE_DB_URL", "sqlite:///./data/jobs.db")
    config_path = getattr(worker, "_config_file_path", None) or os.environ.get("NAT_CONFIG_FILE", "")
    front_end_config = getattr(worker, "_front_end_config", None)
    default_expiry_seconds = getattr(front_end_config, "expiry_seconds", 86400) if front_end_config else 86400
    submit_route_registered = False

    # NAT registers its generic /health route first. Replace it before any
    # async-job prerequisite early return so /health always means readiness.
    _remove_existing_health_routes(app)

    @app.get(
        "/health",
        tags=["health"],
        summary="Readiness check",
        responses={503: {"description": "Async-job, database, or content-encryption dependency is unavailable"}},
    )
    async def health_check():
        """Readiness endpoint that validates async-job, DB, and encryption dependencies."""
        from fastapi.responses import JSONResponse

        result = {"status": "healthy", "dask_available": bool(dask_available), "db": "ok"}
        readiness_failure = await _probe_async_job_readiness(
            dask_available=dask_available,
            job_store=job_store,
            scheduler_address=scheduler_address,
            db_url=db_url,
            config_path=config_path,
            submit_route_registered=submit_route_registered,
        )
        if readiness_failure is not None:
            result["status"] = "degraded"
            result.update(readiness_failure)
            return JSONResponse(status_code=503, content=result)

        try:
            encryption = await get_content_encryption_health_async()
            result["encryption"] = encryption.to_health_dict()
            if encryption.mode != "off" and not encryption.ready:
                result["status"] = "degraded"
                return JSONResponse(status_code=503, content=result)
        except ContentEncryptionConfigError as exc:
            logger.warning("Health check encryption config failed exception=%s", exc.__class__.__name__)
            result["status"] = "degraded"
            result["encryption"] = {
                "mode": "invalid",
                "ready": False,
                "reason": "configuration_invalid",
                "exception_type": exc.__class__.__name__,
            }
            return JSONResponse(status_code=503, content=result)

        return result

    if not get_all_sources():
        logger.warning(
            "No data sources registered. Add a 'data_sources' function with "
            "_type: data_source_registry to your YAML config to enable "
            "data source toggles in the UI."
        )

    @app.get(
        "/v1/jobs/async/agents",
        response_model=AgentListResponse,
        tags=["async jobs"],
        summary="List available agents",
        description="Returns public agent types configured in the active workflow.",
    )
    async def list_agents() -> AgentListResponse:
        """List available agent types for async job submission."""
        agents = []
        for agent_type, config in AGENT_REGISTRY.items():
            if not config.public:
                continue
            if _get_configured_agent_function_config(builder, config.config_name) is None:
                continue
            agents.append(AgentInfo(agent_type=agent_type, description=config.description))
        return AgentListResponse(agents=agents)

    @app.get(
        "/v1/data_sources",
        response_model=list[DataSource],
        tags=["data sources"],
        summary="List data sources",
    )
    async def list_data_sources() -> list[DataSource]:
        """List available data sources, including the current user's per-source auth state."""
        principal = require_verified_principal()
        sources = []
        for source in get_all_sources():
            per_user_auth = await build_listing_auth_info(mcp_auth_provider, principal, source)
            sources.append(
                DataSource(
                    id=source.id,
                    name=source.name,
                    description=source.description,
                    default_enabled=source.default_enabled,
                    requires_auth=source.requires_auth,
                    per_user_auth=per_user_auth,
                )
            )
        return sources

    logger.info("Registered /v1/data_sources and /v1/jobs/async/agents routes")

    static_failure: str | None = None
    if not dask_available or job_store is None or not scheduler_address:
        logger.warning(
            "Dask not available - async job submission routes require NAT_DASK_SCHEDULER_ADDRESS"
            " and NAT_JOB_STORE_DB_URL"
        )
        static_failure = "async_jobs_unavailable"
    elif not _is_readable_regular_file(config_path):
        logger.error("Config file path is missing, unreadable, or not a regular file")
        static_failure = "configuration_missing"
    else:
        await _bootstrap_async_job_storage(db_url, job_store)

    if static_failure is not None:

        @app.post(
            "/v1/jobs/async/submit",
            response_model=JobStatusResponse,
            tags=["async jobs"],
            summary="Submit a new async job",
            description=(
                "Submit a research query to a registered agent. Returns a job ID for tracking progress via SSE stream."
            ),
            responses={503: {"description": "Async job submission is unavailable"}},
        )
        async def unavailable_submit_job(
            req: Annotated[JobSubmitRequest, Body(openapi_examples=JOB_SUBMIT_EXAMPLES)],
            conversation_id: Annotated[str | None, Header(alias="conversation-id")] = None,
        ) -> JobStatusResponse:
            """Preserve request validation and authentication while startup is unavailable."""
            require_verified_principal()
            raise HTTPException(503, "Async job submission is currently unavailable")

        logger.warning("Registered guarded async submit fallback: reason=%s", static_failure)
        return

    logger.info(
        "Registering async job routes: scheduler=%s, db=%s, expiry=%ds",
        scheduler_address,
        db_url[:50],
        default_expiry_seconds,
    )

    @app.post(
        "/v1/jobs/async/submit",
        response_model=JobStatusResponse,
        tags=["async jobs"],
        summary="Submit a new async job",
        description=(
            "Submit a research query to a registered agent. Returns a job ID for tracking progress via SSE stream."
        ),
        responses={
            400: {"description": "Unknown, internal-only, or unconfigured agent type, or invalid request"},
            413: {"description": "Deep-research input exceeds the configured payload limit"},
            409: {
                "description": (
                    "A custom job_id was supplied that collides with an existing job, or a selected "
                    "protected data source requires per-user OAuth connection"
                )
            },
            429: {"description": "Per-principal active-job or submission-rate limit reached"},
            422: {"description": "One or more unknown or agent-unavailable data source IDs"},
            500: {
                "description": (
                    "Content encryption configuration is invalid, async job authorization persistence failed, "
                    "or agent/tool configuration lookup failed unexpectedly"
                )
            },
            503: {
                "description": (
                    "Content encryption, Dask scheduler, admission database, or deployment job capacity is unavailable"
                )
            },
        },
    )
    async def submit_job(
        req: Annotated[JobSubmitRequest, Body(openapi_examples=JOB_SUBMIT_EXAMPLES)],
        conversation_id: Annotated[str | None, Header(alias="conversation-id")] = None,
    ) -> JobStatusResponse:
        """Submit a new async job for deep research or other registered agents."""
        try:
            agent_config = get_agent_config(req.agent_type)
        except KeyError as e:
            raise HTTPException(400, str(e))
        if not agent_config.public:
            raise HTTPException(400, f"Agent type is internal-only and cannot be submitted directly: {req.agent_type}")

        fn_config = _get_configured_agent_function_config(builder, agent_config.config_name)
        if fn_config is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_not_configured",
                    "message": f"Agent '{req.agent_type}' is not configured in the active workflow",
                    "agent_type": req.agent_type,
                    "config_name": agent_config.config_name,
                },
            )

        expiry = req.expiry_seconds if req.expiry_seconds is not None else default_expiry_seconds
        # Authenticate the caller (raises 401/403 if unverified). The returned principal
        # is also forwarded to submit_authorized_job(...) below for ownership recording.
        principal = require_verified_principal()
        readiness_failure = await _probe_async_job_readiness(
            dask_available=dask_available,
            job_store=job_store,
            scheduler_address=scheduler_address,
            db_url=db_url,
            config_path=config_path,
            submit_route_registered=submit_route_registered,
        )
        if readiness_failure is not None:
            logger.warning("Rejected async job submission because readiness failed: %s", readiness_failure["reason"])
            raise HTTPException(503, "Async job submission is currently unavailable")
        try:
            await require_content_encryption_ready_for_submission_async()
        except ContentEncryptionUnavailable as e:
            logger.warning(
                "Rejected async job submission because content encryption is unready: %s",
                e.__class__.__name__,
            )
            raise HTTPException(503, "Content encryption is not ready")
        except ContentEncryptionConfigError as e:
            logger.warning(
                "Rejected async job submission because content encryption config is invalid: %s",
                e.__class__.__name__,
            )
            raise HTTPException(500, "Content encryption configuration is invalid")

        validation_start = time.perf_counter()
        await _validate_data_sources_for_agent(
            builder=builder,
            agent_type=req.agent_type,
            agent_config_name=agent_config.config_name,
            fn_config=fn_config,
            data_sources=req.data_sources,
        )
        logger.info(
            "Validated data_sources for agent %s in %.1fms (requested=%s)",
            req.agent_type,
            (time.perf_counter() - validation_start) * 1000,
            len(req.data_sources) if req.data_sources is not None else "none",
        )

        # Sandbox concurrency / cost guard (Option A): cap concurrent sandbox-enabled
        # jobs per principal and globally, enforced at submit so cost is stopped before
        # a worker spins up. Opt-in (default-off) via AIQ_MAX_SANDBOXES_* env vars so the
        # default submit path stays lazy; fail-open if the active-job count is unknown.
        if _sandbox_caps_configured() and _agent_uses_sandbox(builder, agent_config.config_name):
            await _enforce_sandbox_concurrency(db_url, principal)

        # Preflight protected MCP sources: block before enqueue if a selected
        # protected source is not connected. When data_sources is None the job
        # may use any tool, so every protected source must be connected.
        mcp_block = await _preflight_mcp_auth(mcp_auth_provider, principal, req.data_sources)
        if mcp_block is not None:
            return mcp_block

        # Propagate auth token to Dask worker for requires_auth data sources
        from aiq_agent.auth import get_auth_token

        auth_token = get_auth_token()
        try:
            job_id = await submit_authorized_job(
                agent_type=req.agent_type,
                input_text=req.input,
                owner=principal.email or principal.sub,
                principal=principal,
                job_id=req.job_id,
                expiry_seconds=expiry,
                data_sources=req.data_sources,
                auth_token=auth_token,
                conversation_id=conversation_id,
                skip_encryption_readiness_check=True,
            )
        except ContentEncryptionUnavailable as e:
            logger.warning(
                "Failed to submit authorized job because content encryption is unready: %s",
                e.__class__.__name__,
            )
            raise HTTPException(503, "Content encryption is not ready")
        except ContentEncryptionConfigError as e:
            logger.warning(
                "Failed to submit authorized job because content encryption config is invalid: %s",
                e.__class__.__name__,
            )
            raise HTTPException(500, "Content encryption configuration is invalid")
        except JobIdConflictError:
            raise HTTPException(409, f"Job already exists: {req.job_id}")
        except JobAdmissionError as e:
            headers = {"Retry-After": str(e.retry_after_seconds)} if e.retry_after_seconds is not None else None
            raise HTTPException(status_code=e.status_code, detail=e.public_message, headers=headers)
        except McpAuthRequiredError as e:
            # submit_agent_job runs the same MCP preflight and raises if a selected
            # protected source became disconnected between the route preflight above
            # and enqueue. Surface the SAME 409 mcp_auth_required contract instead of
            # letting it fall through to the generic 500 handler.
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=409, content=e.response.model_dump(mode="json"))
        except RuntimeError as e:
            # The principal is resolved above, so a RuntimeError here is an
            # availability/config failure (e.g. scheduler not configured), not an
            # authorization error -- surface 503, not 403, and don't echo internals.
            logger.warning("Async job submission unavailable: %s", e)
            raise HTTPException(503, "Async job submission is currently unavailable")
        except Exception as e:
            logger.warning("Failed to submit authorized job: %s", e)
            raise HTTPException(500, "Failed to persist async job authorization metadata")

        logger.info(
            "Submitted %s job %s (expiry=%ds) for principal %s:%s",
            req.agent_type,
            job_id,
            expiry,
            principal.type,
            principal.sub,
        )
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.SUBMITTED.value,
            agent_type=req.agent_type,
        )

    submit_route_registered = True

    @app.post(
        "/v1/jobs/async/job/{job_id}/report/edit",
        response_model=ReportEditResponse,
        tags=["async jobs"],
        summary="Create a revised report from a completed report job",
        description=(
            "Authorize access to a completed parent report, reconstruct durable report context, "
            "and submit an internal child job that emits a full revised report."
        ),
        responses={
            404: {"description": "Parent job not found"},
            409: {"description": "Parent job is incomplete, has no durable report, or the child job_id collides"},
            422: {"description": "Request validation failed (e.g. blank edit instruction)"},
            500: {"description": "Parent report data is invalid or report edit submission failed"},
            503: {"description": "Content encryption or Dask scheduler is unavailable"},
        },
    )
    async def edit_job_report(job_id: str, req: ReportEditRequest) -> ReportEditResponse:
        """Create an internal report-rewrite child job from a completed parent report."""
        principal = require_verified_principal()
        parent_job = await authorize_job_access(job_store, db_url, job_id, principal)
        if getattr(parent_job, "status", None) != JobStatus.SUCCESS.value:
            raise HTTPException(409, f"Parent job is not complete: {job_id}")

        try:
            context = await resolve_report_context(parent_job, db_url, job_id)
        except ContentEncryptionUnavailable as e:
            logger.warning(
                "Parent report decrypt unavailable job_id=%s exception=%s",
                job_id,
                e.__class__.__name__,
            )
            raise HTTPException(503, "Content encryption is unavailable")
        except ContentEncryptionInvalidData as e:
            logger.warning(
                "Parent report persisted output invalid job_id=%s exception=%s",
                job_id,
                e.__class__.__name__,
            )
            raise HTTPException(500, "Parent report data is invalid")
        expiry = req.expiry_seconds if req.expiry_seconds is not None else default_expiry_seconds

        from aiq_agent.auth import get_auth_token

        auth_token = get_auth_token()
        try:
            child_job_id = await submit_authorized_job(
                agent_type="report_rewriter",
                input_text=req.input,
                owner=principal.email or principal.sub,
                principal=principal,
                job_id=req.job_id,
                expiry_seconds=expiry,
                data_sources=[],
                auth_token=auth_token,
                initial_files=to_initial_files(context, instruction=req.input),
                output_metadata=report_output_metadata(job_id, "edit"),
                allow_internal=True,
            )
        except ContentEncryptionUnavailable as e:
            logger.warning(
                "Report edit submission rejected because content encryption is unready parent_job_id=%s exception=%s",
                job_id,
                e.__class__.__name__,
            )
            raise HTTPException(503, "Content encryption is not ready")
        except ContentEncryptionConfigError as e:
            logger.warning(
                "Report edit submission rejected because content encryption config is invalid "
                "parent_job_id=%s exception=%s",
                job_id,
                e.__class__.__name__,
            )
            raise HTTPException(500, "Content encryption configuration is invalid")
        except JobIdConflictError:
            raise HTTPException(409, f"Job already exists: {req.job_id}")
        except RuntimeError as e:
            # Principal is resolved above; a RuntimeError here is an availability/config
            # failure (e.g. scheduler not configured), not an authorization error.
            logger.warning("Report edit submission unavailable for parent %s: %s", job_id, e)
            raise HTTPException(503, "Report edit submission is currently unavailable")
        except Exception as e:
            logger.warning("Failed to submit report edit job for parent %s: %s", job_id, e)
            raise HTTPException(500, "Failed to submit report edit job")

        logger.info("Submitted report edit child job %s for parent job %s", child_job_id, job_id)
        return ReportEditResponse(
            job_id=child_job_id,
            parent_job_id=job_id,
            status=JobStatus.SUBMITTED.value,
            agent_type="report_rewriter",
        )

    @app.get(
        "/v1/jobs/async/job/{job_id}",
        response_model=JobStatusResponse,
        tags=["async jobs"],
        summary="Get job status",
        description="Get the current status of an async job by its ID.",
        responses={404: {"description": "Job not found"}},
    )
    async def get_job_status(job_id: str) -> JobStatusResponse:
        """Get the current status of a job."""
        principal = require_verified_principal()
        job = await authorize_job_access(job_store, db_url, job_id, principal)

        return JobStatusResponse(
            job_id=job_id,
            status=job.status,
            error=job.error,
            created_at=job.created_at.isoformat() if job.created_at else None,
        )

    @app.get(
        "/v1/jobs/async/job/{job_id}/stream",
        tags=["async jobs"],
        summary="Stream job events",
        description=(
            "Server-Sent Events (SSE) stream of job progress from the beginning."
            " Includes tool calls, intermediate results, and the final report."
        ),
        responses={404: {"description": "Job not found"}},
    )
    async def stream_job_events(job_id: str) -> StreamingResponse:
        """SSE stream for job events from beginning."""
        principal = require_verified_principal()
        await authorize_job_access(job_store, db_url, job_id, principal)

        return StreamingResponse(
            _sse_generator(job_store, job_id, db_url, start_event_id=0),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/v1/jobs/async/job/{job_id}/stream/{last_event_id}",
        tags=["async jobs"],
        summary="Resume job event stream",
        description="Resume an SSE stream from a specific event ID. Use for reconnection after network interruption.",
        responses={404: {"description": "Job not found"}},
    )
    async def stream_job_events_from(job_id: str, last_event_id: int) -> StreamingResponse:
        """SSE stream for job events from specific event ID (for reconnection)."""
        principal = require_verified_principal()
        await authorize_job_access(job_store, db_url, job_id, principal)

        return StreamingResponse(
            _sse_generator(job_store, job_id, db_url, start_event_id=last_event_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/v1/jobs/async/job/{job_id}/cancel",
        tags=["async jobs"],
        summary="Cancel a running job",
        description="Request cancellation of a running job. The job status will be set to INTERRUPTED.",
        responses={
            400: {"description": "Job is not in RUNNING state"},
            404: {"description": "Job not found"},
        },
    )
    async def cancel_job(job_id: str) -> dict:
        """Cancel a running job."""
        principal = require_verified_principal()
        job = await authorize_job_access(job_store, db_url, job_id, principal)

        if job.status != JobStatus.RUNNING.value:
            raise HTTPException(400, f"Job not running: {job_id} (status: {job.status})")

        await job_store.update_status(job_id, JobStatus.INTERRUPTED, error="cancelled by user")

        event_store = EventStore(db_url, job_id)
        event_store.store(
            {
                "type": "job.cancellation_requested",
                "data": {"reason": "cancelled by user"},
            }
        )

        task_cancelled = await _cancel_dask_task(scheduler_address, job_id)

        logger.info("Cancel requested for job %s: status updated, task_cancelled=%s", job_id, task_cancelled)

        return {"job_id": job_id, "status": JobStatus.INTERRUPTED.value, "task_cancelled": task_cancelled}

    @app.get(
        "/v1/jobs/async/job/{job_id}/state",
        response_model=JobStateResponse,
        tags=["async jobs"],
        summary="Get job artifacts",
        description="Get tool calls, outputs, and sources collected during job execution.",
        responses={404: {"description": "Job not found"}},
    )
    async def get_job_state(job_id: str) -> JobStateResponse:
        """Get artifacts from event store."""
        principal = require_verified_principal()
        await authorize_job_access(job_store, db_url, job_id, principal)

        try:
            artifacts = await _get_job_artifacts(db_url, job_id)
        except ContentEncryptionUnavailable as e:
            logger.warning(
                "Job state decrypt unavailable job_id=%s exception=%s",
                job_id,
                e.__class__.__name__,
            )
            raise HTTPException(503, "Content encryption is unavailable")
        except ContentEncryptionInvalidData as e:
            logger.warning(
                "Job state persisted event data invalid job_id=%s exception=%s",
                job_id,
                e.__class__.__name__,
            )
            raise HTTPException(500, "Job state data is invalid")
        return JobStateResponse(
            job_id=job_id,
            has_state=artifacts is not None,
            state=None,
            artifacts=artifacts,
        )

    @app.get(
        "/v1/jobs/async/job/{job_id}/artifacts",
        tags=["async jobs"],
        summary="List durable artifacts",
        description="List generated artifacts (charts, CSVs, notebooks) harvested from the sandbox.",
        responses={404: {"description": "Job not found"}},
    )
    async def list_job_artifacts(job_id: str) -> dict:
        """List durable artifact metadata for a job (no bytes)."""
        from aiq_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store

        principal = require_verified_principal()
        await authorize_job_access(job_store, db_url, job_id, principal)

        store = build_artifact_store(db_url)
        artifacts = await asyncio.to_thread(store.list, job_id)
        # Exclude storage internals (storage_uri embeds the db_url, which may carry
        # credentials/hostnames; sandbox_path is an internal layout detail) from the
        # client-facing payload. Clients use the content endpoint, not these fields.
        return {
            "job_id": job_id,
            "artifacts": [a.model_dump(mode="json", exclude={"storage_uri", "sandbox_path"}) for a in artifacts],
        }

    @app.get(
        "/v1/jobs/async/job/{job_id}/artifacts/{artifact_id}/content",
        tags=["async jobs"],
        summary="Download artifact content",
        description="Stream the bytes of a single artifact. Job-ownership checks apply.",
        responses={404: {"description": "Job or artifact not found"}},
    )
    async def get_job_artifact_content(job_id: str, artifact_id: str) -> StreamingResponse:
        """Stream an artifact's bytes (auth-scoped to the owning job)."""
        from aiq_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store

        principal = require_verified_principal()
        await authorize_job_access(job_store, db_url, job_id, principal)

        store = build_artifact_store(db_url)
        artifact = await asyncio.to_thread(store.get, job_id, artifact_id)
        if artifact is None:
            raise HTTPException(404, f"Artifact not found: {artifact_id}")

        # The filename is sandbox-controlled; strip control chars and quotes so it cannot
        # break out of the header value (response-splitting / header injection).
        safe_filename = "".join(c for c in artifact.filename if c.isprintable() and c not in '"\\') or "artifact"
        # Starlette encodes header values as Latin-1, so a non-Latin-1 filename (emoji, CJK)
        # would raise UnicodeEncodeError. Provide an ASCII-only fallback plus an RFC 5987
        # filename* with the UTF-8 percent-encoded original for clients that support it.
        from urllib.parse import quote

        ascii_filename = safe_filename.encode("ascii", "ignore").decode() or "artifact"
        encoded_filename = quote(safe_filename, safe="")
        # Only magic-verified raster images may render inline; everything else (SVG, HTML,
        # notebooks, PDFs) is forced to download with nosniff to prevent stored-XSS if a
        # user opens the content URL directly in a browser.
        inline_safe = artifact.mime_type in {"image/png", "image/jpeg", "image/webp"}
        disposition = "inline" if inline_safe else "attachment"
        return StreamingResponse(
            store.open_bytes(job_id, artifact_id),
            media_type=artifact.mime_type,
            headers={
                "Content-Disposition": (
                    f"{disposition}; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get(
        "/v1/jobs/async/job/{job_id}/report",
        response_model=JobReportResponse,
        tags=["async jobs"],
        summary="Get final report",
        description="Get the final research report from a completed job.",
        responses={404: {"description": "Job not found"}},
    )
    async def get_job_report(job_id: str) -> JobReportResponse:
        """Get the final report from a completed job."""
        principal = require_verified_principal()
        job = await authorize_job_access(job_store, db_url, job_id, principal)

        output: dict[str, Any] = {}
        if job.output is not None:
            try:
                decoded_output = await read_job_output_async(job_id, job.output)
            except ContentEncryptionUnavailable as e:
                logger.warning(
                    "Final report decrypt unavailable job_id=%s exception=%s",
                    job_id,
                    e.__class__.__name__,
                )
                raise HTTPException(503, "Content encryption is unavailable")
            except ContentEncryptionInvalidData as e:
                logger.warning(
                    "Final report persisted output invalid job_id=%s exception=%s",
                    job_id,
                    e.__class__.__name__,
                )
                raise HTTPException(500, "Final report data is invalid")
            if isinstance(decoded_output, dict):
                output = decoded_output
        report = output.get("report")

        return JobReportResponse(
            job_id=job_id,
            has_report=bool(report),
            report=report,
            parent_job_id=output.get("parent_job_id"),
            interaction_action=output.get("interaction_action"),
            result_kind=output.get("result_kind"),
        )

    logger.info("Registered async job routes at /v1/jobs/async")

    # Start the ghost job reaper background task
    asyncio.create_task(_reap_ghost_jobs(job_store, db_url))

    # Run job metadata and event retention in the API process. A never-ending
    # cleanup coroutine submitted to shared Dask would permanently occupy a worker.
    _start_periodic_cleanup(job_store, db_url, default_expiry_seconds)


GHOST_JOB_TIMEOUT_SECONDS = 300  # 5 minutes without events = ghost job
GHOST_REAPER_INTERVAL_SECONDS = 60  # check every 60 seconds


def _find_stale_jobs(db_url: str, running_status: str) -> list[str]:
    """
    Sync helper to query for ghost jobs. Runs in a thread via run_in_executor
    to avoid blocking the async event loop with DB I/O.
    """
    from sqlalchemy import inspect
    from sqlalchemy import text

    from ..jobs.event_store import EventStore

    # Ensure job_events exists so the LEFT JOIN resolves (it need not have rows).
    EventStore._ensure_table_exists(db_url)
    engine = EventStore._get_or_create_sync_engine(db_url)
    inspector = inspect(engine)
    # The query is driven from job_info; without it there are no jobs to reap.
    if not inspector.has_table("job_info"):
        return []

    with engine.connect() as conn:
        # Drive from job_info with a LEFT JOIN so a RUNNING job that has not
        # persisted any events yet is still considered. That is exactly the
        # failure this reaper exists to catch: a worker that crashes/OOMs after
        # the job is marked RUNNING but before its first event is stored leaves
        # zero rows in job_events and would be invisible to an INNER JOIN,
        # sticking the job in RUNNING forever. COALESCE falls back to
        # job_info.updated_at (set when the job entered RUNNING) when there are
        # no events, so both cases share one staleness check.
        if db_url.startswith(("postgresql", "postgres")):
            stale_query = text(
                "SELECT ji.job_id FROM job_info ji "
                "LEFT JOIN job_events je ON je.job_id = ji.job_id "
                "WHERE ji.status = :running_status "
                "GROUP BY ji.job_id, ji.updated_at "
                "HAVING COALESCE(MAX(je.created_at), ji.updated_at) < NOW() - :timeout * INTERVAL '1 second'"
            )
            params = {"running_status": running_status, "timeout": GHOST_JOB_TIMEOUT_SECONDS}
        else:
            stale_query = text(
                "SELECT ji.job_id FROM job_info ji "
                "LEFT JOIN job_events je ON je.job_id = ji.job_id "
                "WHERE ji.status = :running_status "
                "GROUP BY ji.job_id, ji.updated_at "
                "HAVING COALESCE(MAX(je.created_at), ji.updated_at) < datetime('now', :timeout_interval)"
            )
            params = {
                "running_status": running_status,
                "timeout_interval": f"-{GHOST_JOB_TIMEOUT_SECONDS} seconds",
            }

        result = conn.execute(stale_query, params)
        return [row[0] for row in result]


def _mark_job_failed_if_running(db_url: str, job_id: str, running_status: str, failure_status: str, error: str) -> bool:
    """Atomically flip a job from RUNNING to FAILURE, only if still running.

    Returns True iff this call performed the transition. The ``WHERE status =
    running`` guard makes the write conditional in a single statement, so a job
    that reached a terminal state (e.g. a slow worker that finished) between
    detection and reaping is never clobbered.
    """
    from sqlalchemy import text

    from ..jobs.event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    now_expr = "NOW()" if db_url.startswith(("postgresql", "postgres")) else "CURRENT_TIMESTAMP"
    stmt = text(
        f"UPDATE job_info SET status = :failure, error = :error, updated_at = {now_expr} "
        "WHERE job_id = :job_id AND status = :running"
    )
    with engine.begin() as conn:
        result = conn.execute(
            stmt,
            {"failure": failure_status, "error": error, "job_id": job_id, "running": running_status},
        )
        return (result.rowcount or 0) == 1


async def _reap_ghost_jobs(job_store, db_url: str) -> None:
    """
    Background task that periodically marks stale RUNNING jobs as FAILURE.

    A job is considered "ghost" if it has been RUNNING for over
    GHOST_JOB_TIMEOUT_SECONDS with no new events in the job_events table, OR if
    it has been RUNNING that long without ever storing an event (measured from
    job_info.updated_at). This catches Dask worker crashes and OOM kills that
    bypass Python exception handling, including a crash before the first event
    is persisted.
    """
    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

    from ..jobs.event_store import EventStore

    logger.info(
        "Ghost job reaper started (timeout=%ds, interval=%ds)",
        GHOST_JOB_TIMEOUT_SECONDS,
        GHOST_REAPER_INTERVAL_SECONDS,
    )

    loop = asyncio.get_running_loop()

    while True:
        try:
            await asyncio.sleep(GHOST_REAPER_INTERVAL_SECONDS)

            stale_job_ids = await loop.run_in_executor(None, _find_stale_jobs, db_url, JobStatus.RUNNING.value)

            for stale_job_id in stale_job_ids:
                error_msg = "Job timed out (no heartbeat received from worker)"
                try:
                    transitioned = await loop.run_in_executor(
                        None,
                        _mark_job_failed_if_running,
                        db_url,
                        stale_job_id,
                        JobStatus.RUNNING.value,
                        JobStatus.FAILURE.value,
                        error_msg,
                    )
                    if not transitioned:
                        # The job left RUNNING between detection and reaping
                        # (e.g. a slow worker finished); leave its status intact.
                        logger.info("Ghost reap skipped %s: no longer running", stale_job_id)
                        continue
                    logger.warning(
                        "Reaped ghost job %s (no heartbeat for %ds)", stale_job_id, GHOST_JOB_TIMEOUT_SECONDS
                    )
                    event_store = EventStore(db_url, stale_job_id)
                    event_store.store(
                        {
                            "type": "job.error",
                            "data": {
                                "error": error_msg,
                                "error_type": "GhostJobTimeout",
                            },
                        }
                    )
                except Exception as e:
                    logger.warning("Failed to reap ghost job %s: %s", stale_job_id, e)

        except asyncio.CancelledError:
            logger.info("Ghost job reaper stopped")
            break
        except Exception as e:
            logger.warning("Ghost job reaper error: %s", e)


_cleanup_task: asyncio.Task | None = None
"""Module-level reference for graceful shutdown cancellation."""

# Advisory lock ID for PostgreSQL — ensures only one pod runs cleanup at a time.
# Arbitrary constant; change if it collides with another lock in your deployment.
_PG_ADVISORY_LOCK_ID = 0x41495143_4C45414E  # "AIQCLEAN" in hex


def _start_periodic_cleanup(job_store, db_url: str, expiry_seconds: int) -> None:
    """Start local cleanup of expired jobs, events, access rows, and artifacts."""
    global _cleanup_task

    # Cleanup interval: half the expiry time, clamped to [60s, 3600s].
    cleanup_interval = max(60, min(expiry_seconds // 2, 3600))

    # Keep housekeeping off the shared Dask cluster. NAT's periodic_cleanup is an
    # infinite Dask task; with one thread per worker it consumes an entire worker
    # slot for the lifetime of the deployment.
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
    _cleanup_task = asyncio.create_task(
        _cleanup_old_events_loop(
            db_url,
            expiry_seconds,
            cleanup_interval,
            job_store=job_store,
        )
    )
    logger.info(
        "Started local periodic job and event cleanup (interval=%ds, expiry=%ds)",
        cleanup_interval,
        expiry_seconds,
    )


async def stop_periodic_cleanup() -> None:
    """Cancel the local periodic cleanup task. Call from shutdown handler."""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None
        logger.info("Periodic cleanup task cancelled")


async def _cleanup_old_events_loop(
    db_url: str,
    retention_seconds: int,
    interval_seconds: int,
    *,
    job_store=None,
) -> None:
    """
    Periodically expire finished jobs and clean their retained data.

    PostgreSQL cleanup cycles use an advisory lock so multiple API replicas do
    not perform the same work. Job cleanup remains local to the API process and
    therefore does not consume a shared Dask worker slot.
    """

    is_postgres = db_url.startswith("postgres")

    logger.info(
        "Periodic cleanup task started (retention=%ds, interval=%ds, advisory_lock=%s)",
        retention_seconds,
        interval_seconds,
        is_postgres,
    )

    # Run once immediately on startup to catch anything that aged out during downtime.
    await _run_local_cleanup_cycle(job_store, db_url, retention_seconds, is_postgres)

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await _run_local_cleanup_cycle(job_store, db_url, retention_seconds, is_postgres)
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task stopped")
            break


async def _run_local_cleanup_cycle(job_store, db_url: str, retention_seconds: int, is_postgres: bool) -> None:
    """Run one isolated job-retention and event-retention cycle."""
    if job_store is not None:
        try:
            expired = await _run_job_cleanup(job_store, is_postgres)
            if expired:
                logger.info("Expired jobs cleaned up: %d", expired)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Job cleanup error: %s", e)

    try:
        await _run_event_cleanup(db_url, retention_seconds, is_postgres)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Event cleanup error: %s", e)


async def _run_job_cleanup(job_store, is_postgres: bool) -> int | None:
    """Expire finished jobs once, with one PostgreSQL replica elected per cycle."""
    if not is_postgres:
        return await job_store.cleanup_expired_jobs()

    from sqlalchemy import text

    # NAT's scoped session is keyed by asyncio task. Hold the election lock in
    # this task and run cleanup in a child task so it receives its own DB session.
    async with job_store.session() as lock_session:
        locked = (
            await lock_session.execute(
                text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                {"lock_id": _PG_ADVISORY_LOCK_ID},
            )
        ).scalar()
        if not locked:
            return None
        return await asyncio.create_task(job_store.cleanup_expired_jobs())


async def _run_event_cleanup(db_url: str, retention_seconds: int, is_postgres: bool) -> None:
    """
    Execute one cleanup cycle: time-based event pruning + removal of events for expired jobs.

    On PostgreSQL, acquires a transaction-level advisory lock (pg_try_advisory_xact_lock)
    so concurrent pods skip the cycle rather than doing redundant work. The lock is
    automatically released on commit/rollback, avoiding leak risks.
    """
    from ..jobs.access import cleanup_job_access
    from ..jobs.event_store import EventStore

    loop = asyncio.get_running_loop()

    def _do_cleanup() -> tuple[int, int, int, int]:
        """Delete expired retained data synchronously; return removal counts."""
        from sqlalchemy import text

        engine = EventStore._get_or_create_sync_engine(db_url)

        with engine.connect() as conn:
            # On PostgreSQL, acquire a transaction-level advisory lock. If another pod
            # already holds it, skip this cycle. The lock is automatically released
            # on commit/rollback — no manual unlock needed.
            if is_postgres:
                locked = conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                    {"lock_id": _PG_ADVISORY_LOCK_ID},
                ).scalar()
                if not locked:
                    return (0, 0, 0, 0)

            # 1. Time-based: delete events older than retention period
            if is_postgres:
                result = conn.execute(
                    text("DELETE FROM job_events WHERE created_at < NOW() - :seconds * INTERVAL '1 second'"),
                    {"seconds": retention_seconds},
                )
            else:
                result = conn.execute(
                    text("DELETE FROM job_events WHERE created_at < datetime('now', :interval)"),
                    {"interval": f"-{retention_seconds} seconds"},
                )
            time_deleted = result.rowcount

            # 2. Coordinated: delete events for jobs already marked expired in job_info.
            # This catches events that haven't aged out yet but whose parent job is
            # already expired (e.g. short-lived jobs with long event retention).
            expired_result = conn.execute(
                text("DELETE FROM job_events WHERE job_id IN (SELECT job_id FROM job_info WHERE is_expired = true)")
            )
            expired_deleted = expired_result.rowcount
            access_deleted = cleanup_job_access(db_url, conn=conn)

            # SQLite permits one writer at a time. Artifact cleanup uses its own
            # connection, so release this connection's write lock first.
            if not is_postgres:
                conn.commit()

            # Artifact retention shares the job expiry boundary. Keep it inside
            # the leader's advisory-lock transaction so non-leader replicas
            # cannot run the same destructive cleanup concurrently.
            artifacts_deleted = 0
            try:
                from aiq_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store

                artifacts_deleted = build_artifact_store(db_url).cleanup_old_artifacts(retention_seconds)
            except Exception as e:  # noqa: BLE001 - retention is best-effort
                logger.debug("Artifact cleanup skipped (%s)", type(e).__name__)

            if is_postgres:
                conn.commit()
            return (time_deleted, expired_deleted, access_deleted, artifacts_deleted)

    time_deleted, expired_deleted, access_deleted, artifacts_deleted = await loop.run_in_executor(None, _do_cleanup)

    if artifacts_deleted:
        logger.info("Artifact cleanup: %d old artifacts removed", artifacts_deleted)

    if time_deleted > 0 or expired_deleted > 0 or access_deleted > 0:
        logger.info(
            "Event cleanup: %d old events removed, %d events for expired jobs removed, %d access rows removed",
            time_deleted,
            expired_deleted,
            access_deleted,
        )


async def _cancel_dask_task(scheduler_address: str, job_id: str) -> bool:
    """
    Cancel a Dask task by job ID.

    Args:
        scheduler_address: Dask scheduler address.
        job_id: Job ID to cancel.

    Returns:
        True if a Dask cancellation request was sent, False otherwise.
    """
    try:
        from distributed import Client
        from distributed import Future

        async with Client(scheduler_address, asynchronous=True) as client:
            # NAT JobStore submits job futures with key ``{job_id}-job``. Targeting
            # the key directly avoids using Dask Variable.get as a maybe-exists
            # check, which logs scheduler-side timeout errors when the variable is
            # absent or slow to resolve.
            future = Future(f"{job_id}-job", client)
            await client.cancel([future], asynchronous=True, force=True)
            logger.info("Sent cancellation request for Dask task %s", future.key)
            return True
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning("Failed to cancel Dask task for job %s: %s", job_id, e)
    except Exception as e:
        logger.warning("Unexpected error cancelling Dask task for job %s: %s", job_id, e)
    return False


def _extract_event_metadata(event: dict) -> tuple[dict, dict]:
    """Extract data and metadata from an event dict."""
    data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
    if not metadata and isinstance(data, dict):
        metadata = data.get("metadata", {}) or {}
    return data, metadata


def _process_tool_start(event: dict, data: dict, metadata: dict, tool_call_map: dict[str, dict]) -> None:
    """Process a tool.start event and add to tool_call_map."""
    tool_id = data.get("id", "")
    inner_data = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
    tool_call_map[tool_id] = {
        "id": tool_id,
        "name": data.get("name", ""),
        "input": inner_data.get("input"),
        "output": None,
        "status": "running",
        "workflow": metadata.get("workflow"),
        "is_sandbox": bool(metadata.get("sandbox")),
        "timestamp": event.get("timestamp"),
    }


def _process_tool_end(event: dict, data: dict, metadata: dict, tool_call_map: dict[str, dict]) -> None:
    """Process a tool.end event and update tool_call_map."""
    tool_id = data.get("id", "")
    inner_data = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
    tool_output = inner_data.get("output")

    if tool_id in tool_call_map:
        tool_call_map[tool_id]["output"] = tool_output
        tool_call_map[tool_id]["status"] = "completed"
        tool_call_map[tool_id]["is_sandbox"] = tool_call_map[tool_id].get("is_sandbox") or bool(metadata.get("sandbox"))
    else:
        tool_call_map[tool_id] = {
            "id": tool_id,
            "name": data.get("name", ""),
            "input": None,
            "output": tool_output,
            "status": "completed",
            "workflow": metadata.get("workflow"),
            "is_sandbox": bool(metadata.get("sandbox")),
            "timestamp": event.get("timestamp"),
        }


def _normalize_url(url: str) -> str:
    """Normalize URL for consistent deduplication."""
    from urllib.parse import urlparse
    from urllib.parse import urlunparse

    try:
        parsed = urlparse(url)
        normalized_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                normalized_path,
                parsed.params,
                parsed.query,
                "",
            )
        )
    except Exception:
        return url


def _is_valid_url(url: str) -> bool:
    """Check if string is a valid HTTP/HTTPS URL."""
    return bool(url and url.lower().startswith(("http://", "https://")))


def _process_artifact_update(
    event: dict,
    data: dict,
    metadata: dict,
    outputs: list[dict],
    sources_found: set[str],
    sources_cited: set[str],
) -> None:
    """Process an artifact.update event and add to outputs."""
    artifact_type = data.get("type")
    content = data.get("content")

    # Track citation sources and uses for accurate counts (with validation)
    if artifact_type == "citation_source":
        url = data.get("url") or content
        if _is_valid_url(url):
            sources_found.add(_normalize_url(url))
    elif artifact_type == "citation_use":
        url = data.get("url") or content
        if _is_valid_url(url):
            sources_cited.add(_normalize_url(url))
    elif artifact_type == "output" and data.get("output_category") == "final_report":
        final_cited_urls = data.get("cited_urls")
        if isinstance(final_cited_urls, list):
            # The verified final report is authoritative. Intermediate LLM
            # output can contain citations that finalization later removes.
            sources_cited.clear()
            sources_cited.update(_normalize_url(url) for url in final_cited_urls if _is_valid_url(url))

    if content:
        outputs.append(
            {
                "type": artifact_type,
                "content": content,
                "name": event.get("name"),
                "workflow": metadata.get("workflow"),
                "timestamp": event.get("timestamp"),
                **{k: v for k, v in data.items() if k not in ("type", "content")},
            }
        )


async def _get_job_artifacts(db_url: str, job_id: str) -> dict | None:
    """
    Extract artifacts from stored events.

    Returns a simplified structure with all tool calls, outputs, and source counts.
    Frontend categorizes tools by name (task=subagent, write_todos=middleware, etc.).

    Args:
        db_url: Database URL for event store.
        job_id: Job ID to fetch artifacts for.

    Returns:
        Dict with 'tools', 'outputs', and 'sources' (counts), or None if no artifacts found.
    """
    from ..jobs.crypto import ContentEncryptionError
    from ..jobs.event_store import EventStore

    try:
        events = await EventStore.get_events_async(db_url, job_id, 0, 10000)
        if not events:
            return None

        tool_call_map: dict[str, dict] = {}
        outputs: list[dict] = []
        sources_found: set[str] = set()
        sources_cited: set[str] = set()

        for event in events:
            event_type = event.get("type", "")
            data, metadata = _extract_event_metadata(event)

            if event_type == "tool.start":
                _process_tool_start(event, data, metadata, tool_call_map)
            elif event_type == "tool.end":
                _process_tool_end(event, data, metadata, tool_call_map)
            elif event_type == "artifact.update":
                _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        tools = list(tool_call_map.values())
        result = {
            "tools": tools,
            "outputs": outputs,
            "sources": {
                "found": len(sources_found),
                "cited": len(sources_cited),
                "found_urls": list(sources_found),
                "cited_urls": list(sources_cited),
            },
        }
        return result if tools or outputs or sources_found else None

    except ContentEncryptionError:
        raise
    except (KeyError, TypeError) as e:
        logger.warning("Failed to parse artifacts for job %s: %s", job_id, e)
        return None
    except Exception as e:
        logger.warning("Failed to get artifacts for job %s: %s", job_id, e)
        return None


async def _sse_generator(job_store, job_id: str, db_url: str, start_event_id: int = 0):
    """
    Route to appropriate SSE generator based on database type.

    PostgreSQL: Uses LISTEN/NOTIFY for real-time push-based events (sub-10ms latency).
    SQLite: Uses polling (0.5s interval) since SQLite doesn't support pub-sub.
    """
    from ..jobs.crypto import ContentEncryptionInvalidData
    from ..jobs.crypto import ContentEncryptionUnavailable
    from ..jobs.event_store import EventStore

    if EventStore.is_postgres(db_url):
        try:
            async for event in _sse_generator_postgres(job_store, job_id, db_url, start_event_id):
                yield event
        except ContentEncryptionUnavailable as e:
            logger.warning("SSE encrypted event decrypt unavailable for job %s: %s", job_id, e.__class__.__name__)
            yield f"event: job.error\ndata: {json.dumps({'error': 'Content encryption is unavailable'})}\n\n"
        except ContentEncryptionInvalidData as e:
            logger.warning("SSE encrypted event data invalid for job %s: %s", job_id, e.__class__.__name__)
            yield f"event: job.error\ndata: {json.dumps({'error': 'Job event data is invalid'})}\n\n"
        except Exception as e:
            logger.warning("Pub-sub failed, falling back to polling: %s", e)
            async for event in _sse_generator_polling(job_store, job_id, db_url, start_event_id):
                yield event
    else:
        async for event in _sse_generator_polling(job_store, job_id, db_url, start_event_id):
            yield event


async def _sse_generator_postgres(job_store, job_id: str, db_url: str, start_event_id: int = 0):
    """
    PostgreSQL pub-sub based SSE generator - near-instant event delivery.

    Uses asyncpg LISTEN/NOTIFY for real-time push-based events.
    Achieves sub-10ms latency compared to 500ms polling interval.
    """
    import asyncio
    import time

    import asyncpg

    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

    from ..jobs.connection_manager import get_connection_manager
    from ..jobs.crypto import ContentEncryptionInvalidData
    from ..jobs.crypto import ContentEncryptionUnavailable
    from ..jobs.event_store import EventStore

    connection_manager = get_connection_manager()
    last_status = None
    last_event_id = start_event_id
    sequence_id = start_event_id
    terminal_statuses = {JobStatus.SUCCESS.value, JobStatus.FAILURE.value, JobStatus.INTERRUPTED.value}
    is_reconnect = start_event_id > 0
    # Emit an SSE keepalive comment after this many seconds of silence so an
    # upstream idle timeout (OpenShift router / edge / proxy) never closes the
    # connection. job.heartbeat only starts once the worker runs, so it does not
    # cover worker cold-start on the first request — this keepalive does.
    SSE_KEEPALIVE_INTERVAL = 15.0
    last_keepalive = time.monotonic()

    def format_sse(event_type: str, data: dict, event_id: int | None = None) -> str:
        """Format an SSE frame and advance (or set) the monotonic event sequence id."""
        nonlocal sequence_id
        if event_id is not None:
            sequence_id = event_id
        else:
            sequence_id += 1
        return f"id: {sequence_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"

    # LISTEN/NOTIFY needs a persistent session — incompatible with PgBouncer
    # transaction pooling. Use AIQ_LISTEN_DB_URL to point directly at PostgreSQL.
    import os

    listen_db_url = os.environ.get("AIQ_LISTEN_DB_URL", db_url)
    asyncpg_url = listen_db_url.replace("+psycopg2", "").replace("+asyncpg", "").replace("postgresql://", "postgres://")
    channel = f"job_events_{job_id.replace('-', '_')}"

    logger.info(f"SSE pub-sub stream starting for job_id={job_id}, channel={channel}")

    conn = None
    notification_queue: asyncio.Queue = asyncio.Queue()

    def notification_handler(connection, pid, channel_name, payload):
        """Enqueue a Postgres LISTEN/NOTIFY payload, dropping it if the queue is full."""
        try:
            notification_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("Notification queue full for job %s", job_id)

    try:
        conn = await asyncpg.connect(asyncpg_url)
        await conn.add_listener(channel, notification_handler)
        logger.info(f"SSE: Listening on channel {channel}")

        async with connection_manager.track_connection():
            job = await job_store.get_job(job_id)
            if not job:
                logger.warning(f"SSE pub-sub: Job {job_id} not found")
                yield format_sse("job.error", {"error": "Job not found"})
                return

            job_already_complete = job.status in terminal_statuses

            events = await EventStore.get_events_async(db_url, job_id, last_event_id, 10000)
            logger.info(
                f"SSE pub-sub: Fetched {len(events)} historical events for job {job_id} (after_id={last_event_id})"
            )

            for event in events:
                db_event_id = event.pop("_id", None)
                if db_event_id:
                    last_event_id = db_event_id
                event_type = event.pop("type", "event")
                yield format_sse(event_type, event, db_event_id)

            yield format_sse("stream.mode", {"mode": "pubsub", "channel": channel})

            # Reconciliation fetch: catch events that arrived while sending the historical batch.
            # The LISTEN handler may have queued notifications for some of these, but a direct
            # fetch ensures no gap between the historical batch and the live stream.
            reconcile_events = await EventStore.get_events_async(db_url, job_id, last_event_id, 1000)
            if reconcile_events:
                logger.info(f"SSE pub-sub: Reconciliation fetched {len(reconcile_events)} events for job {job_id}")
                for event in reconcile_events:
                    db_event_id = event.pop("_id", None)
                    if db_event_id:
                        last_event_id = db_event_id
                    event_type = event.pop("type", "event")
                    yield format_sse(event_type, event, db_event_id)

            if job_already_complete:
                last_status = job.status
                data = {"status": job.status}
                if job.error:
                    data["error"] = job.error
                if is_reconnect:
                    data["reconnected"] = True
                yield format_sse("job.status", data)
                logger.info(f"SSE pub-sub: Job {job_id} already complete, sent {len(events)} events")
                return

            while True:
                if connection_manager.is_shutting_down:
                    logger.info("SSE pub-sub stream closing for job %s due to server shutdown", job_id)
                    yield format_sse("job.shutdown", {"message": "Server shutting down"})
                    break

                try:
                    try:
                        payload = await asyncio.wait_for(notification_queue.get(), timeout=5.0)
                        notification_data = json.loads(payload)
                        event_id = notification_data.get("id")

                        if event_id and event_id > last_event_id:
                            event = await EventStore.get_event_by_id_async(db_url, event_id)
                            if event:
                                last_event_id = event_id
                                db_event_id = event.pop("_id", None)
                                event_type = event.pop("type", "event")
                                yield format_sse(event_type, event, db_event_id)
                    except TimeoutError:
                        # Fallback poll: catch events if NOTIFY was lost
                        fallback_events = await EventStore.get_events_async(db_url, job_id, last_event_id, 100)
                        for event in fallback_events:
                            db_event_id = event.pop("_id", None)
                            if db_event_id:
                                last_event_id = db_event_id
                            event_type = event.pop("type", "event")
                            yield format_sse(event_type, event, db_event_id)
                        # Keepalive during silent periods (e.g. worker cold-start) so an
                        # upstream idle timeout never closes the connection.
                        if fallback_events:
                            last_keepalive = time.monotonic()
                        elif (time.monotonic() - last_keepalive) >= SSE_KEEPALIVE_INTERVAL:
                            last_keepalive = time.monotonic()
                            yield ": keepalive\n\n"

                    job = await job_store.get_job(job_id)
                    if not job:
                        logger.warning(f"SSE pub-sub: Job {job_id} not found")
                        yield format_sse("job.error", {"error": "Job not found"})
                        break

                    if job.status != last_status:
                        last_status = job.status
                        logger.info(f"SSE pub-sub: Job {job_id} status changed to {job.status}")
                        data = {"status": job.status}
                        if job.error:
                            data["error"] = job.error
                        if is_reconnect:
                            data["reconnected"] = True
                            is_reconnect = False
                        yield format_sse("job.status", data)

                    if job.status in terminal_statuses:
                        await asyncio.sleep(0.5)
                        while not notification_queue.empty():
                            try:
                                payload = notification_queue.get_nowait()
                                notification_data = json.loads(payload)
                                event_id = notification_data.get("id")
                                if event_id and event_id > last_event_id:
                                    event = await EventStore.get_event_by_id_async(db_url, event_id)
                                    if event:
                                        last_event_id = event_id
                                        db_event_id = event.pop("_id", None)
                                        event_type = event.pop("type", "event")
                                        yield format_sse(event_type, event, db_event_id)
                            except asyncio.QueueEmpty:
                                break
                        break

                except asyncio.CancelledError:
                    logger.info("SSE pub-sub stream cancelled for job %s", job_id)
                    break
                except ContentEncryptionUnavailable as e:
                    logger.warning(
                        "SSE pub-sub encrypted event decrypt unavailable for job %s: %s",
                        job_id,
                        e.__class__.__name__,
                    )
                    yield format_sse("job.error", {"error": "Content encryption is unavailable"})
                    break
                except ContentEncryptionInvalidData as e:
                    logger.warning(
                        "SSE pub-sub encrypted event data invalid for job %s: %s",
                        job_id,
                        e.__class__.__name__,
                    )
                    yield format_sse("job.error", {"error": "Job event data is invalid"})
                    break
                except Exception as e:
                    logger.exception("SSE pub-sub stream error for job %s: %s", job_id, e)
                    yield format_sse("job.error", {"error": "Internal server error"})
                    break

    finally:
        if conn:
            try:
                await conn.remove_listener(channel, notification_handler)
                await conn.close()
                logger.info(f"SSE pub-sub: Closed connection for job {job_id}")
            except Exception as e:
                logger.warning(f"SSE pub-sub: Error closing connection for job {job_id}: {e}")


async def _sse_generator_polling(job_store, job_id: str, db_url: str, start_event_id: int = 0):
    """
    Polling-based SSE generator for SQLite and fallback scenarios.

    Replays historical events as fast as possible, then switches to live polling mode.
    Live mode uses a 0.5s polling interval and is suitable for local development with SQLite.
    Supports reconnection via start_event_id - replays events after that ID without delay.
    Supports graceful shutdown via the SSE connection manager.
    """
    import asyncio
    import time

    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

    from ..jobs.connection_manager import get_connection_manager
    from ..jobs.crypto import ContentEncryptionInvalidData
    from ..jobs.crypto import ContentEncryptionUnavailable
    from ..jobs.event_store import EventStore

    connection_manager = get_connection_manager()
    last_status = None
    last_event_id = start_event_id
    sequence_id = start_event_id
    terminal_statuses = {JobStatus.SUCCESS.value, JobStatus.FAILURE.value, JobStatus.INTERRUPTED.value}
    is_reconnect = start_event_id > 0
    in_replay_mode = True
    replay_mode_announced = False
    # Emit an SSE keepalive comment after this many seconds of silent live polling
    # so an upstream idle timeout never closes the connection (e.g. during worker
    # cold-start, before the first event or 30s heartbeat arrives).
    SSE_KEEPALIVE_INTERVAL = 15.0
    last_keepalive = time.monotonic()

    def format_sse(event_type: str, data: dict, event_id: int | None = None) -> str:
        """Format an SSE frame and advance (or set) the monotonic event sequence id."""
        nonlocal sequence_id
        if event_id is not None:
            sequence_id = event_id
        else:
            sequence_id += 1
        return f"id: {sequence_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"

    logger.info(
        f"SSE polling stream starting for job_id={job_id}, start_event_id={start_event_id}, db_url={db_url[:50]}"
    )

    async with connection_manager.track_connection():
        yield format_sse("stream.mode", {"mode": "polling", "interval_ms": 500})

        while True:
            if connection_manager.is_shutting_down:
                logger.info("SSE stream closing for job %s due to server shutdown", job_id)
                yield format_sse("job.shutdown", {"message": "Server shutting down"})
                break

            try:
                job = await job_store.get_job(job_id)
                if not job:
                    logger.warning(f"SSE: Job {job_id} not found")
                    yield format_sse("job.error", {"error": "Job not found"})
                    break

                # Replay mode drains historical events quickly without wait delays.
                # Live mode returns to regular polling cadence.
                if in_replay_mode:
                    limit = 10000 if job.status in terminal_statuses else 1000
                else:
                    limit = 10000 if job.status in terminal_statuses else 100
                events = await EventStore.get_events_async(db_url, job_id, last_event_id, limit)

                if events:
                    last_keepalive = time.monotonic()
                    logger.info(f"SSE: Fetched {len(events)} events for job {job_id} (after_id={last_event_id})")
                elif job.status in terminal_statuses:
                    logger.warning(f"SSE: No events found for completed job {job_id} (after_id={last_event_id})")

                for i, event in enumerate(events):
                    if connection_manager.is_shutting_down:
                        logger.info("SSE stream closing for job %s due to server shutdown (mid-batch)", job_id)
                        yield format_sse("job.shutdown", {"message": "Server shutting down"})
                        return

                    try:
                        db_event_id = event.pop("_id", None)
                        if db_event_id:
                            last_event_id = db_event_id
                        event_type = event.pop("type", "event")
                        sse_output = format_sse(event_type, event, db_event_id)
                        yield sse_output
                    except Exception as e:
                        logger.error(f"SSE: Failed to yield event {i} (id={db_event_id}): {e}", exc_info=True)

                # Transition to live mode after historical catch-up:
                # - no more events after current cursor, or
                # - fetched a partial replay batch (< limit), indicating we've reached the current tail.
                if in_replay_mode and (not events or len(events) < limit):
                    in_replay_mode = False
                    replay_mode_announced = True
                    logger.info(
                        "SSE: Replay complete for job %s at event_id=%s; switching to live mode", job_id, last_event_id
                    )
                    yield format_sse("stream.mode", {"mode": "live"})

                if job.status != last_status:
                    last_status = job.status
                    logger.info(f"SSE: Job {job_id} status changed to {job.status}")
                    data = {"status": job.status}
                    if job.error:
                        data["error"] = job.error
                    if is_reconnect:
                        data["reconnected"] = True
                        is_reconnect = False
                    yield format_sse("job.status", data)

                if job.status in terminal_statuses:
                    break

                # During replay we intentionally avoid polling delays so clients can catch up quickly.
                if in_replay_mode:
                    continue

                # If replay was completed in a prior iteration but stream.mode couldn't be emitted
                # (e.g., due to an exception path), emit it once before waiting.
                if not in_replay_mode and not replay_mode_announced:
                    replay_mode_announced = True
                    yield format_sse("stream.mode", {"mode": "live"})

                # Keepalive during silent live periods (e.g. worker cold-start) so the
                # idle-looking connection isn't closed by an upstream idle timeout.
                if (time.monotonic() - last_keepalive) >= SSE_KEEPALIVE_INTERVAL:
                    last_keepalive = time.monotonic()
                    yield ": keepalive\n\n"

                shutdown_signaled = await connection_manager.wait_or_shutdown(0.5)
                if shutdown_signaled:
                    logger.info("SSE stream closing for job %s due to server shutdown (during wait)", job_id)
                    yield format_sse("job.shutdown", {"message": "Server shutting down"})
                    break

            except asyncio.CancelledError:
                logger.info("SSE stream cancelled for job %s", job_id)
                break
            except ContentEncryptionUnavailable as e:
                logger.warning("SSE encrypted event decrypt unavailable for job %s: %s", job_id, e.__class__.__name__)
                yield format_sse("job.error", {"error": "Content encryption is unavailable"})
                break
            except ContentEncryptionInvalidData as e:
                logger.warning("SSE encrypted event data invalid for job %s: %s", job_id, e.__class__.__name__)
                yield format_sse("job.error", {"error": "Job event data is invalid"})
                break
            except Exception as e:
                logger.exception("SSE stream error for job %s: %s", job_id, e)
                yield format_sse("job.error", {"error": "Internal server error"})
                break
