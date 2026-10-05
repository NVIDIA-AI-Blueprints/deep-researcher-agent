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
Job submission utilities.

Provides functions to submit agent jobs to the Dask cluster.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from contextlib import suppress
from dataclasses import replace
from functools import partial
from typing import Any

from sqlalchemy.exc import IntegrityError

from aiq_agent.auth import Principal
from aiq_agent.auth import get_current_principal
from aiq_api.auth import get_current_trace_tags
from aiq_api.mcp_auth.provider import principal_user_id

from ..registry import get_agent_config
from .access import JobAccessConflictError
from .access import _make_no_auth_principal
from .access import create_job_access
from .access import release_job_access_reservation
from .access import renew_job_access_reservation
from .admission import DEFAULT_RESERVATION_TTL_SECONDS
from .admission import DeepResearchAdmissionLimits
from .admission import JobAdmissionConflictError
from .admission import JobAdmissionUnavailableError
from .admission import release_deep_research_job_reservation
from .admission import renew_deep_research_job_reservation
from .admission import reserve_deep_research_job
from .admission import validate_deep_research_input
from .runner import JobTraceCorrelation
from .runner import run_agent_job

logger = logging.getLogger(__name__)


class JobIdConflictError(RuntimeError):
    """Raised when a caller-supplied job_id collides with an existing job.

    Distinct from generic submission failures so callers can map it to HTTP 409
    instead of triggering the rollback path, which would delete the pre-existing
    (victim) job's durable state.
    """


class InternalAgentError(RuntimeError):
    """Raised when an internal-only (public=False) agent is submitted without opt-in.

    The submission helper is the real trust boundary; trusted internal callers must
    pass allow_internal=True explicitly rather than relying on every call site (and
    the HTTP route filter) to remember the gate.
    """


def _resolve_submission_principal(owner: str) -> Principal | None:
    """Resolve the best available principal for async job ownership.

    Used only when submit_agent_job is called programmatically without an
    explicit principal.  Verified middleware identity is preferred; falls back
    to a compatibility principal when auth is disabled.
    """
    principal = get_current_principal()
    if principal is not None:
        return principal

    if os.environ.get("REQUIRE_AUTH", "false").lower() == "true":
        return None

    return _make_no_auth_principal(owner)


def _resolve_admission_principal(principal: Principal) -> Principal:
    """Return the trusted identity used for deep-research quotas.

    In no-auth mode a programmatic caller controls ``owner`` and therefore the
    compatibility principal's subject.  Never use that value as a quota key.
    Every caller shares one anonymous trust-domain budget until authentication
    is required for the deployment.
    """
    if os.environ.get("REQUIRE_AUTH", "false").lower() == "true":
        return principal

    return Principal(type="anonymous", sub="anonymous")


def _get_job_trace_correlation() -> JobTraceCorrelation:
    """Capture identifiers that correlate an independent async-job trace to its submission."""
    try:
        from nat.builder.context import ContextState
    except ImportError:
        return JobTraceCorrelation(request_trace_tags=get_current_trace_tags())

    context_state = ContextState.get()
    workflow_trace_id = context_state.workflow_trace_id.get()
    active_stack = context_state.active_span_id_stack.get()
    active_span_id = active_stack[-1] if active_stack and active_stack[-1] != "root" else None
    return JobTraceCorrelation(
        session_id=context_state.conversation_id.get(),
        submission_trace_id=f"{workflow_trace_id:032x}" if workflow_trace_id is not None else None,
        submission_span_id=active_span_id,
        request_trace_tags=get_current_trace_tags(),
    )


async def submit_agent_job(
    agent_type: str,
    input_text: str,
    owner: str,
    principal: Principal | None = None,
    job_id: str | None = None,
    expiry_seconds: int = 86400,
    available_documents: list[dict] | None = None,
    data_sources: list[str] | None = None,
    auth_token: str | None = None,
    conversation_id: str | None = None,
    skip_encryption_readiness_check: bool = False,
    initial_files: dict[str, Any] | None = None,
    output_metadata: dict[str, Any] | None = None,
    allow_internal: bool = False,
    database_name: str | None = None,
) -> str:
    """
    Submit an agent job to the Dask cluster.

    This is the main entry point for submitting async jobs from application code.
    It looks up the agent configuration from the registry and submits the job.

    Args:
        agent_type: Agent type identifier (e.g., 'deep_researcher').
        input_text: The user's query/request.
        owner: Owner email for the job.
        principal: Verified principal that owns the job.
        job_id: Optional custom job ID.
        expiry_seconds: Job expiry time in seconds (default 24h).
        available_documents: Optional list of document dicts with file_name and summary.
        data_sources: Optional list of allowed data sources to enforce in the worker.
        auth_token: Optional auth token to propagate to the Dask worker for
            data sources that require authentication.
        conversation_id: Optional originating conversation ID. REST callers
            pass the ``conversation-id`` header explicitly because custom
            FastAPI routes do not run inside NAT's execution context.
        skip_encryption_readiness_check: Skip the internal readiness check when
            the caller already performed it off the event loop.
        initial_files: Optional DeepAgents virtual filesystem files to seed into worker state.
        output_metadata: Optional metadata persisted with the final job output.
        allow_internal: Allow trusted callers to submit an internal-only agent.
        database_name: Optional database scope the router resolved for this request.
            Scoped requests are authoritative, so the worker must rebuild agent state
            with the same scope rather than falling back to the configured default.

    Returns:
        The job ID.

    Raises:
        KeyError: If agent_type is not registered.
        RuntimeError: If Dask scheduler is not configured.

    Example:
        job_id = await submit_agent_job(
            agent_type="deep_researcher",
            input_text="Research quantum computing",
            owner="user@example.com",
            available_documents=[{"file_name": "doc.pdf", "summary": "A research paper"}],
        )
    """
    from nat.front_ends.fastapi.async_jobs.job_store import JobStore

    from .crypto import get_content_encryption_policy_identity
    from .crypto import require_content_encryption_ready_for_submission_async

    # Get agent configuration from registry
    agent_config = get_agent_config(agent_type)

    # Enforce the internal-agent gate at the submission boundary (defense in depth):
    # internal-only agents may only be launched by trusted callers that opt in.
    if not agent_config.public and not allow_internal:
        raise InternalAgentError(f"Agent type is internal-only and cannot be submitted directly: {agent_type}")

    admission_limits: DeepResearchAdmissionLimits | None = None
    if agent_type == "deep_researcher":
        admission_limits = validate_deep_research_input(input_text)

    # @environment_variable NAT_DASK_SCHEDULER_ADDRESS
    # @category Server
    # @type str
    # @required true
    # Dask scheduler address for async job submission.
    scheduler_address = os.environ.get("NAT_DASK_SCHEDULER_ADDRESS")

    # @environment_variable NAT_JOB_STORE_DB_URL
    # @category Server
    # @type str
    # @default sqlite:///./data/jobs.db
    # @required false
    # Database URL for job persistence (SQLite or PostgreSQL).
    db_url = os.environ.get("NAT_JOB_STORE_DB_URL", "sqlite:///./data/jobs.db")

    # @environment_variable NAT_CONFIG_FILE
    # @category Server
    # @type str
    # @required false
    # Path to NAT workflow config file used by Dask workers.
    config_path = os.environ.get("NAT_CONFIG_FILE", "")

    # @environment_variable NAT_FASTAPI_LOG_LEVEL
    # @category Server
    # @type int
    # @default 20
    # @required false
    # Python logging level for FastAPI workers (10=DEBUG, 20=INFO, 30=WARNING).
    log_level = int(os.environ.get("NAT_FASTAPI_LOG_LEVEL", "20"))

    # @environment_variable NAT_USE_DASK_THREADS
    # @category Server
    # @type bool
    # @default 0
    # @required false
    # Use Dask thread pool instead of process pool for workers. Set to 1 to enable.
    use_threads = os.environ.get("NAT_USE_DASK_THREADS", "0") == "1"

    if not scheduler_address:
        raise RuntimeError("Async job submission requires NAT_DASK_SCHEDULER_ADDRESS to be set")

    if not skip_encryption_readiness_check:
        await require_content_encryption_ready_for_submission_async()
    content_encryption_policy = get_content_encryption_policy_identity()

    # Auto-capture auth token if not explicitly provided
    if auth_token is None:
        from aiq_agent.auth import get_auth_token

        auth_token = get_auth_token()

    if principal is None:
        principal = _resolve_submission_principal(owner)
    if principal is None:
        raise RuntimeError("Verified current principal required for async job submission")
    admission_principal = _resolve_admission_principal(principal)

    # Preflight protected MCP sources before enqueue. The REST submit route also
    # does this (returning 409), but programmatic submitters — notably the chat
    # researcher's async deep-research path — call this directly and would otherwise
    # bypass the check, so this is the single chokepoint both paths share. Skipped
    # when no MCP auth provider is active in this process (nothing to enforce).
    from aiq_api.mcp_auth.active import get_active_mcp_auth_provider
    from aiq_api.mcp_auth.preflight import McpAuthRequiredError
    from aiq_api.mcp_auth.preflight import evaluate_mcp_auth

    mcp_provider = get_active_mcp_auth_provider()
    if mcp_provider is not None:
        block = await evaluate_mcp_auth(mcp_provider, principal, data_sources)
        if block is not None:
            raise McpAuthRequiredError(block)

    job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
    resolved_job_id = job_store.ensure_job_id(job_id)
    loop = asyncio.get_running_loop()
    admission_token: str | None = None

    if admission_limits is not None:
        try:
            admission_token = await reserve_deep_research_job(
                db_url=db_url,
                job_id=resolved_job_id,
                principal=admission_principal,
                limits=admission_limits,
            )
        except JobAdmissionConflictError as exc:
            raise JobIdConflictError(f"Job already exists: {resolved_job_id}") from exc

    submission_token = admission_token or secrets.token_urlsafe(24)
    reservation_ttl_seconds = (
        admission_limits.reservation_ttl_seconds if admission_limits is not None else DEFAULT_RESERVATION_TTL_SECONDS
    )
    trace_correlation = _get_job_trace_correlation()
    if conversation_id is not None:
        trace_correlation = replace(trace_correlation, session_id=conversation_id)
    submission_conversation_id = trace_correlation.session_id

    async def _release_submission_reservations() -> None:
        """Conditionally release only reservations owned by this submitter."""
        try:
            await loop.run_in_executor(
                None,
                release_job_access_reservation,
                resolved_job_id,
                submission_token,
                db_url,
            )
        except Exception as cleanup_error:  # noqa: BLE001 - expiry is the fail-safe
            logger.warning(
                "Could not release failed job-access reservation for %s (error_type=%s)",
                resolved_job_id,
                type(cleanup_error).__name__,
            )
        if admission_token is not None:
            await release_deep_research_job_reservation(
                db_url=db_url,
                job_id=resolved_job_id,
                reservation_token=admission_token,
            )

    # Ownership is durable before Dask can run the task. This removes the
    # ownerless-worker window that existed when job_access was written after
    # JobStore.submit_job returned.
    try:
        await loop.run_in_executor(
            None,
            partial(
                create_job_access,
                resolved_job_id,
                principal,
                db_url,
                submission_conversation_id,
                agent_type,
                submission_token=submission_token,
                submission_expires_at=time.time() + reservation_ttl_seconds,
            ),
        )
    except JobAccessConflictError as exc:
        if admission_token is not None:
            await release_deep_research_job_reservation(
                db_url=db_url,
                job_id=resolved_job_id,
                reservation_token=admission_token,
            )
        raise JobIdConflictError(f"Job already exists: {resolved_job_id}") from exc
    except asyncio.CancelledError:
        await asyncio.shield(_release_submission_reservations())
        raise
    except Exception:
        await _release_submission_reservations()
        raise

    lease_lost = asyncio.Event()

    async def _maintain_submission_lease() -> None:
        """Keep pre-enqueue rows live until NAT has made job_info durable."""
        renewal_interval = max(0.05, reservation_ttl_seconds / 3)
        try:
            while True:
                await asyncio.sleep(renewal_interval)
                expires_at = time.time() + reservation_ttl_seconds
                access_ok = await loop.run_in_executor(
                    None,
                    renew_job_access_reservation,
                    resolved_job_id,
                    submission_token,
                    db_url,
                    expires_at,
                )
                admission_ok = True
                if admission_token is not None:
                    admission_ok = await renew_deep_research_job_reservation(
                        db_url=db_url,
                        job_id=resolved_job_id,
                        reservation_token=admission_token,
                        ttl_seconds=reservation_ttl_seconds,
                    )
                if not access_ok or not admission_ok:
                    lease_lost.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - lease renewal must fail closed
            logger.warning(
                "Submission lease renewal failed for %s (error_type=%s)",
                resolved_job_id,
                type(exc).__name__,
            )
            lease_lost.set()

    lease_task = asyncio.create_task(
        _maintain_submission_lease(),
        name=f"submission-lease-{resolved_job_id}",
    )
    lease_lost_waiter = asyncio.create_task(
        lease_lost.wait(),
        name=f"submission-lease-loss-{resolved_job_id}",
    )

    async def _stop_submission_lease() -> None:
        for task in (lease_task, lease_lost_waiter):
            task.cancel()
        for task in (lease_task, lease_lost_waiter):
            with suppress(asyncio.CancelledError):
                await task

    submit_task = asyncio.create_task(
        job_store.submit_job(
            job_id=resolved_job_id,
            expiry_seconds=expiry_seconds,
            job_fn=run_agent_job,
            job_args=[
                not use_threads,  # configure_logging
                log_level,
                scheduler_address,
                db_url,
                config_path,
                resolved_job_id,
                input_text,
                agent_config.class_path,
                agent_config.config_name,
                trace_correlation,
                available_documents,
                data_sources,
                auth_token,
                content_encryption_policy,
                initial_files,
                output_metadata,
                principal_user_id(principal),
                admission_token,
                database_name,
            ],
        ),
        name=f"submit-job-{resolved_job_id}",
    )

    try:
        completed, _ = await asyncio.wait(
            {submit_task, lease_lost_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if lease_lost_waiter in completed:
            submit_task.cancel()
            await asyncio.gather(submit_task, return_exceptions=True)
            raise JobAdmissionUnavailableError("Submission lease was lost before the job became durable")
        await submit_task
    except IntegrityError as e:
        # A caller-supplied job_id collided with an existing job. NAT's _create_job
        # inserts job_info first, so the collision fails before any state of OURS is
        # created. The colliding job belongs to someone else — we must NOT run the
        # rollback path, which unconditionally deletes that job's info/events/access
        # rows. Surface a conflict so the route can return HTTP 409.
        await _stop_submission_lease()
        await _release_submission_reservations()
        logger.info("Rejected colliding job_id %s on async submit", resolved_job_id)
        raise JobIdConflictError(f"Job already exists: {resolved_job_id}") from e
    except asyncio.CancelledError:
        submit_task.cancel()
        await asyncio.shield(asyncio.gather(submit_task, return_exceptions=True))
        await asyncio.shield(_stop_submission_lease())
        logger.warning(
            "Submission of %s was cancelled; retaining its expiring ownership reservation because Dask enqueue "
            "may already have occurred",
            resolved_job_id,
        )
        raise
    except Exception:
        await _stop_submission_lease()
        # NAT may have enqueued a Dask Future before Variable.set (or another
        # later step) failed. Ownership already exists; retaining both rows is
        # safer than deleting metadata/releasing capacity under a running task.
        # If NAT never created job_info, the expiring reservations are reclaimed.
        logger.warning(
            "Async submission failed for %s; retaining expiring ownership/admission state because Dask enqueue "
            "may already have occurred",
            resolved_job_id,
        )
        raise
    else:
        await _stop_submission_lease()

    logger.info(
        "Submitted %s job %s for owner %s (%s:%s)",
        agent_type,
        resolved_job_id,
        owner,
        principal.type,
        principal.sub,
    )
    return resolved_job_id


# Backwards compatibility alias
async def submit_deep_research_job(
    input_text: str,
    owner: str,
    job_id: str | None = None,
    expiry_seconds: int = 86400,
) -> str:
    """
    Submit a deep research job.

    Legacy function preserved for backwards compatibility.
    New code should use submit_agent_job(agent_type="deep_researcher", ...).
    """
    return await submit_agent_job(
        agent_type="deep_researcher",
        input_text=input_text,
        owner=owner,
        job_id=job_id,
        expiry_seconds=expiry_seconds,
    )
