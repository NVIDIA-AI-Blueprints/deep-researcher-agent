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

"""The ``SandboxProvider`` base — the thin contract a sandbox backend must satisfy.

Design: force only the minimum (``_create_session`` + declared ``capabilities``).
Everything else — lazy creation, locking, idempotency-gated retry, and cleanup — is
shared here so a new provider implements just the SDK-specific parts. File tools
(``read_file``/``write_file``/``edit_file``/``ls``/``glob``) are inherited from
``BaseSandbox``, which builds them on top of ``execute``; providers never reimplement them.

This mirrors the knowledge-layer adapter philosophy: a small required surface, optional
capabilities with safe defaults, and provider-owned error classification.
"""

from __future__ import annotations

import logging
import shlex
import threading
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import TypeVar

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.protocol import FileDownloadResponse
from deepagents.backends.protocol import FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from .capabilities import SandboxCapabilities
from .config import job_scoped_artifact_dir
from .config import job_scoped_workdir
from .logging_utils import log_sandbox_failure

if TYPE_CHECKING:
    from .config import SandboxConfig

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Per-job workspace creation is a quick mkdir; bound it well under the sandbox lifetime.
_WORKSPACE_PREP_TIMEOUT = 60


class SandboxTerminatedError(RuntimeError):
    """Raised when an operation is attempted on a terminated (cancelled/closed) provider."""


class SandboxProvider(BaseSandbox, ABC):
    """Job-scoped, lazily-created sandbox backend behind a uniform contract.

    Subclasses implement provider-specific session creation and declare their
    capabilities. The base provides shared resilience (single-flight creation,
    a serialization lock around remote calls, and idempotency-gated retry driven
    by the provider's own :meth:`is_recoverable_error`).

    Attributes:
        provider_name: Registry key for this provider.
    """

    provider_name: str = "base"

    def __init__(self, config: SandboxConfig, job_id: str) -> None:
        """Initialize the provider.

        Args:
            config: Resolved sandbox configuration for the job.
            job_id: Async job identifier used to scope the sandbox identity.
        """
        self.config = config
        self.job_id = job_id
        self.sandbox_name = self._scoped_name(job_id)
        # Per-job workspace roots inside the (possibly shared/reused) sandbox. Computed via
        # the same helpers the runtime uses, so the directory the agent writes to and the
        # directory the harvest scans always agree.
        self.workdir = job_scoped_workdir(config.workdir, job_id)
        self.artifact_dir = job_scoped_artifact_dir(config.workdir, job_id)
        self._session: BaseSandbox | None = None
        # Operation lock: serializes remote calls + gated retry (held across the call).
        self._lock = threading.RLock()
        # State lock: guards the session reference and the terminated flag only. Held for
        # microseconds and NEVER across a remote call or session creation, so close()/
        # terminate() can tear down out-of-band without waiting on an in-flight execute.
        # Lock order is strictly operation-lock -> state-lock; teardown takes only the
        # state lock, so the two can never deadlock.
        self._state_lock = threading.Lock()
        self._terminated = False
        self._event_emit: Callable[[dict[str, object]], None] | None = None
        self._cleanup_failed = False
        self._cleanup_failure_reasons: set[str] = set()
        self._cleanup_state_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Required surface (the only things a provider must implement)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _create_session(self) -> BaseSandbox:
        """Create the underlying provider-specific ``BaseSandbox`` session.

        Implementations own the SDK calls (gateway connect, create/attach, image
        build, ready-wait). They must NOT silently attach to a sandbox owned by a
        prior job; collisions should produce a fresh, job-scoped sandbox.

        Returns:
            A concrete ``BaseSandbox`` (e.g. the langchain-modal / langchain-openshell adapter).
        """

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        """Return the security/lifecycle guarantees this provider can enforce."""

    # ------------------------------------------------------------------ #
    # Optional hooks with safe, conservative defaults (override to opt in)
    # ------------------------------------------------------------------ #
    @classmethod
    def _scoped_name(cls, job_id: str) -> str:
        """Translate a job id into a provider-legal, job-scoped sandbox name.

        Providers override to apply their own naming rules (length, charset).
        """
        return job_id

    def is_recoverable_error(self, exc: Exception) -> bool:
        """Classify an exception as a transient/stale-sandbox error worth retrying.

        Conservative default returns ``False`` so unknown providers never recreate
        and silently re-run against an empty sandbox. Providers override using their
        own SDK's typed exceptions rather than fragile string matching.
        """
        return False

    @contextmanager
    def try_operation_lease(self):
        """Yield whether the provider is idle without waiting behind an in-flight execute."""
        acquired = self._lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    def set_event_emitter(self, emit: Callable[[dict[str, object]], None] | None) -> None:
        """Attach the job-scoped event sink used for sanitized lifecycle events."""
        self._event_emit = emit

    def _emit_event(self, event: dict[str, object]) -> None:
        """Best-effort event delivery; observability must never break execution."""
        if self._event_emit is None:
            return
        try:
            self._event_emit(event)
        except Exception as exc:  # noqa: BLE001 - event persistence is non-critical
            log_sandbox_failure(
                logger,
                operation="event_emit",
                reason_code="event_emit_failed",
                exc=exc,
                provider=self.provider_name,
                sandbox=self.sandbox_name,
            )

    def close(self) -> None:
        """Release the underlying sandbox session, if any (idempotent).

        Tears the session down out-of-band (under the short state lock, never the
        operation lock) so cleanup never blocks behind an in-flight call. Unlike
        :meth:`terminate`, this does not permanently terminate the provider: a later
        operation may lazily recreate the session. Default delegates to the session's
        ``close`` when present.
        """
        with self._state_lock:
            session = self._session
            self._session = None
        self._safe_close(session)

    def terminate(self) -> None:
        """Forcibly stop any in-flight execution and release the sandbox (idempotent).

        Used on the cancellation/timeout path. Because teardown runs out-of-band (it does
        not take the operation lock), closing the underlying session interrupts a
        long-running ``execute`` rather than waiting for it to finish. Providers that can
        hard-kill a remote process should override :meth:`_terminate_session`.
        """
        with self._state_lock:
            session = self._session
            self._session = None
            self._terminated = True
        self._terminate_session(session)

    def _terminate_session(self, session: BaseSandbox | None) -> None:
        """Forcibly stop a session. Default closes it; providers may override to hard-kill."""
        self._safe_close(session)

    def _prepare_workspace(self, session: BaseSandbox) -> None:
        """Create the job-scoped workspace and artifact directories in a new session.

        Idempotent (``mkdir -p``); runs once per session creation so the per-job root
        exists before the first ``write_file``, even when the underlying sandbox is shared
        and reused across jobs. Best-effort: a failure here is logged rather than raised,
        since a genuine filesystem problem resurfaces on the first real write.
        """
        command = f"mkdir -p {shlex.quote(self.workdir)} {shlex.quote(self.artifact_dir)}"
        try:
            session.execute(command, timeout=self._clamp_timeout(_WORKSPACE_PREP_TIMEOUT))
        except Exception as exc:  # noqa: BLE001 - workspace prep is best-effort; real failures resurface on first write
            log_sandbox_failure(
                logger,
                operation="workspace_prepare",
                reason_code="workspace_prepare_failed",
                exc=exc,
                provider=self.provider_name,
                sandbox=self.sandbox_name,
            )

    def _safe_close(self, session: BaseSandbox | None) -> None:
        """Best-effort close of a session; never raises on the teardown path."""
        if session is not None and hasattr(session, "close"):
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must never raise on the terminal path
                self._record_cleanup_failure("session_close_failed")
                log_sandbox_failure(
                    logger,
                    operation="session_close",
                    reason_code="session_close_failed",
                    exc=exc,
                    provider=self.provider_name,
                    sandbox=self.sandbox_name,
                )

    def _record_cleanup_failure(self, reason_code: str) -> None:
        """Record a monotonic, thread-safe cleanup failure reason."""
        with self._cleanup_state_lock:
            self._cleanup_failed = True
            self._cleanup_failure_reasons.add(reason_code)

    @property
    def cleanup_succeeded(self) -> bool:
        """Whether every cleanup attempt, including retry teardown, completed without an error."""
        with self._cleanup_state_lock:
            return not self._cleanup_failed

    @property
    def cleanup_failure_reason_codes(self) -> tuple[str, ...]:
        """Return stable reason codes suitable for terminal lifecycle events."""
        with self._cleanup_state_lock:
            return tuple(sorted(self._cleanup_failure_reasons))

    @property
    def id(self) -> str:
        """Stable identifier: the live session id once created, else the scoped name."""
        with self._state_lock:
            session = self._session
        return session.id if session is not None else self.sandbox_name

    # ------------------------------------------------------------------ #
    # Byte/exec surface — shared resilience, delegated to the session
    # ------------------------------------------------------------------ #
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a command in the sandbox (non-idempotent; no recreate-and-retry).

        The per-call ``timeout`` is clamped to the configured sandbox lifetime
        (``config.timeout``). Agent-supplied timeouts are unreliable (e.g. a tool
        may pass milliseconds where the backend expects seconds), and a single
        ``execute`` should never outlive the sandbox or exceed a provider's hard
        cap, so we bound it rather than let the backend reject the call.
        """
        timeout = self._clamp_timeout(timeout)
        return self._call("execute", lambda s: s.execute(command, timeout=timeout), idempotent=False)

    def _clamp_timeout(self, timeout: int | None) -> int | None:
        """Bound a per-call timeout to ``config.timeout`` (the sandbox max lifetime)."""
        if timeout is None:
            return None
        ceiling = self.config.timeout
        if timeout > ceiling:
            logger.warning(
                "Sandbox %s execute timeout %ss exceeds configured limit %ss; clamping",
                self.sandbox_name,
                timeout,
                ceiling,
            )
            return ceiling
        return max(1, timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload input files into the sandbox (idempotent; safe to retry)."""
        return self._call("upload_files", lambda s: s.upload_files(files), idempotent=True)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox for artifact harvesting (idempotent)."""
        return self._call("download_files", lambda s: s.download_files(paths), idempotent=True)

    # ------------------------------------------------------------------ #
    # Internal lifecycle
    # ------------------------------------------------------------------ #
    def _session_or_create(self) -> BaseSandbox:
        """Return the live session, creating it once (single-flight).

        Called while the operation lock is held, so creation is serialized. Only the
        session-reference reads/writes take the short state lock (not the slow
        ``_create_session`` call), so a concurrent terminate/close can swap the session
        out without waiting for the in-flight remote call.
        """
        with self._state_lock:
            if self._terminated:
                raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} has been terminated")
            if self._session is not None:
                return self._session
        logger.info("Sandbox session init: provider=%s name=%s", self.provider_name, self.sandbox_name)
        created = self._create_session()
        self._prepare_workspace(created)
        with self._state_lock:
            if not self._terminated:
                self._session = created
                return created
        # Terminated mid-creation: discard the freshly created session rather than leak it.
        self._safe_close(created)
        raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} has been terminated")

    def _reset_session(self) -> None:
        """Drop and recreate the session (used only for idempotent recoverable retries)."""
        logger.warning(
            "Sandbox session RESET: provider=%s name=%s (prior in-sandbox files are lost)",
            self.provider_name,
            self.sandbox_name,
        )
        with self._state_lock:
            stale = self._session
            self._session = None
        self._safe_close(stale)
        created = self._create_session()
        self._prepare_workspace(created)
        with self._state_lock:
            if not self._terminated:
                self._session = created
                return
        self._safe_close(created)
        raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} has been terminated")

    def _call(self, op_name: str, fn: Callable[[BaseSandbox], _T], *, idempotent: bool) -> _T:
        """Run a remote call with the serialization lock and gated retry.

        The operation lock serializes calls into a single shared job sandbox to avoid
        filesystem races and reset-during-execute hazards. A concurrent ``terminate()``
        does not take this lock: it closes the session out-of-band, which interrupts the
        in-flight call; the resulting error is re-raised (never retried) once we observe
        the terminated flag. Retry otherwise happens only when the operation is idempotent
        AND the provider classifies the error as recoverable (fail-safe over fail-silent).
        """
        with self._lock:
            try:
                result = fn(self._session_or_create())
                # A concurrent terminate() can flip _terminated while the call was in
                # flight; if the close did not interrupt it (or the call won the race),
                # surface cancellation rather than returning a result for a terminated job.
                with self._state_lock:
                    if self._terminated:
                        raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} has been terminated")
                return result
            except Exception as exc:
                with self._state_lock:
                    terminated = self._terminated
                if terminated:
                    raise
                if idempotent and self.is_recoverable_error(exc):
                    logger.warning("Sandbox %s recoverable error on %s; recreating and retrying once", self.id, op_name)
                    self._reset_session()
                    return fn(self._session_or_create())
                raise
