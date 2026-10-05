# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless Python execution in one request-owned OpenShell sandbox."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import Protocol

logger = logging.getLogger(__name__)

_REFERENCE_PATTERN = re.compile(r"structured_[1-9][0-9]*")
_REMOTE_EVIDENCE_DIR = "evidence"
_REMOTE_REQUEST_DIR = "requests"
_RUNNER_FILENAME = "runner.py"


class _SandboxBackend(Protocol):
    """Small provider surface used by the stateless Python transport."""

    def execute(self, command: str, *, timeout: int | None = None) -> Any: ...

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[Any]: ...


class _SandboxRuntime(Protocol):
    """Request-owned OpenShell runtime and its provider backend."""

    @property
    def sandbox_backend(self) -> _SandboxBackend: ...

    @property
    def workdir(self) -> str: ...

    def close(self) -> None: ...

    def terminate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PythonRunnerLimits:
    """Operational bounds applied to each script and uploaded evidence set."""

    wall_timeout_seconds: float = 30.0
    max_code_chars: int = 50_000
    max_output_chars: int = 50_000
    max_evidence_bytes: int = 20_000_000
    max_memory_mb: int = 8_192
    max_cpu_seconds: int = 600
    max_processes: int = 256
    max_open_files: int = 256
    max_file_bytes: int = 100_000_000


def _runner_command(
    *,
    runner_path: str,
    manifest_path: str,
    request_path: str,
    response_path: str,
    limits: PythonRunnerLimits,
) -> list[str]:
    """Build one bounded command for a fresh Python process."""

    memory_bytes = limits.max_memory_mb * 1024 * 1024
    return [
        "/usr/bin/timeout",
        "--signal=KILL",
        f"{limits.wall_timeout_seconds:.3f}s",
        "/usr/bin/prlimit",
        f"--as={memory_bytes}:{memory_bytes}",
        f"--cpu={limits.max_cpu_seconds}:{limits.max_cpu_seconds}",
        f"--nproc={limits.max_processes}:{limits.max_processes}",
        f"--nofile={limits.max_open_files}:{limits.max_open_files}",
        f"--fsize={limits.max_file_bytes}:{limits.max_file_bytes}",
        "--",
        "python3",
        "-I",
        "-u",
        runner_path,
        manifest_path,
        request_path,
        str(limits.max_output_chars),
        response_path,
    ]


class OpenShellPythonRunner:
    """Execute each script in a fresh namespace inside one request sandbox."""

    def __init__(
        self,
        *,
        runtime: _SandboxRuntime,
        host_manifest_path: Path,
        host_evidence_root: Path,
        limits: PythonRunnerLimits | None = None,
    ) -> None:
        self.runtime = runtime
        self.host_manifest_path = host_manifest_path
        self.host_evidence_root = host_evidence_root.resolve()
        self.limits = limits or PythonRunnerLimits()
        self._backend = runtime.sandbox_backend
        self._remote_root = PurePosixPath(runtime.workdir)
        self._remote_evidence_dir = self._remote_root / _REMOTE_EVIDENCE_DIR
        self._remote_request_dir = self._remote_root / _REMOTE_REQUEST_DIR
        self._remote_manifest = self._remote_evidence_dir / "structured-results.json"
        self._remote_runner = self._remote_root / _RUNNER_FILENAME
        self._runner_path = Path(__file__).with_name(_RUNNER_FILENAME)
        self._lock = asyncio.Lock()
        self._closed = False
        self._request_sequence = 0

    async def execute(self, code: str) -> str:
        """Execute one self-contained bounded script and return a JSON response."""

        if not code.strip():
            return _json_response({"status": "error", "error": "code_required"})
        if len(code) > self.limits.max_code_chars:
            return _json_response({"status": "error", "error": "code_too_large"})

        async with self._lock:
            if self._closed:
                return _json_response({"status": "error", "error": "sandbox_closed"})
            try:
                request_path, response_path, log_path = await self._prepare_request(code)
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._execute_runner, request_path, response_path, log_path),
                    timeout=self.limits.wall_timeout_seconds + 10.0,
                )
            except asyncio.CancelledError:
                await self._terminate()
                raise
            except TimeoutError:
                await self._terminate()
                return _json_response({"status": "error", "error": "execution_timed_out"})
            except Exception as exc:  # noqa: BLE001 - provider failures are sanitized before reaching the model
                logger.warning("OpenShell Python execution failed (error_type=%s)", type(exc).__name__)
                await self._terminate()
                return _json_response({"status": "error", "error": "sandbox_execution_failed"})

            if result.exit_code in {124, 137}:
                await self._terminate()
                return _json_response({"status": "error", "error": "execution_timed_out"})
            if result.exit_code not in {None, 0}:
                await self._terminate()
                return _json_response({"status": "error", "error": "python_process_failed"})
            response_limit = (self.limits.max_output_chars * 2) + 25_000
            raw_output = str(result.output).strip()
            if len(raw_output) > response_limit:
                await self._terminate()
                return _json_response({"status": "error", "error": "python_response_too_large"})
            try:
                payload = json.loads(raw_output)
                if not isinstance(payload, dict):
                    raise TypeError("Python response must be an object")
                return _json_response(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                await self._terminate()
                return _json_response({"status": "error", "error": "invalid_python_response"})

    async def _prepare_request(self, code: str) -> tuple[str, str, str]:
        """Upload a trusted runner, exact structured-data receipts, and one code request."""

        manifest, evidence_files = self._evidence_bundle()
        self._request_sequence += 1
        request_path = self._remote_request_dir / f"script-{self._request_sequence}.json"
        response_path = self._remote_request_dir / f"script-{self._request_sequence}.response.json"
        log_path = self._remote_request_dir / f"script-{self._request_sequence}.log"
        request = _json_response({"code": code}).encode("utf-8")
        files = {
            str(self._remote_runner): self._runner_path.read_bytes(),
            str(self._remote_manifest): manifest,
            str(request_path): request,
            **evidence_files,
        }
        await asyncio.to_thread(self._upload, files)
        return str(request_path), str(response_path), str(log_path)

    def _evidence_bundle(self) -> tuple[bytes, dict[str, bytes]]:
        """Build a sandbox-local manifest from bounded request-owned structured-data receipts."""

        raw_manifest = json.loads(self.host_manifest_path.read_text(encoding="utf-8"))
        entries = raw_manifest.get("results") if isinstance(raw_manifest, dict) else None
        if not isinstance(entries, list):
            raise ValueError("invalid structured-data evidence manifest")

        total_bytes = 0
        remote_entries: list[dict[str, Any]] = []
        evidence_files: dict[str, bytes] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("invalid structured-data evidence entry")
            reference = entry.get("ref")
            if not isinstance(reference, str) or _REFERENCE_PATTERN.fullmatch(reference) is None:
                raise ValueError("invalid structured-data evidence reference")
            host_path = Path(str(entry.get("path") or "")).resolve()
            if not host_path.is_relative_to(self.host_evidence_root):
                raise ValueError("structured-data evidence path escaped its request root")
            content = host_path.read_bytes()
            json.loads(content)
            total_bytes += len(content)
            if total_bytes > self.limits.max_evidence_bytes:
                raise ValueError("structured-data evidence exceeds the configured upload limit")

            remote_path = self._remote_evidence_dir / f"{reference}.json"
            evidence_files[str(remote_path)] = content
            remote_entries.append(
                {
                    "ref": reference,
                    "provider": entry.get("provider"),
                    "tool_name": entry.get("tool_name"),
                    "question": entry.get("question"),
                    "database_name": entry.get("database_name"),
                    "request_id": entry.get("request_id"),
                    "row_count": entry.get("row_count"),
                    "columns": entry.get("columns"),
                    "truncated": bool(entry.get("truncated", False)),
                    "path": str(remote_path),
                }
            )

        manifest = _json_response({"version": 1, "results": remote_entries}).encode("utf-8")
        if total_bytes + len(manifest) > self.limits.max_evidence_bytes:
            raise ValueError("structured-data evidence manifest exceeds the configured upload limit")
        return manifest, evidence_files

    def _upload(self, files: dict[str, bytes]) -> None:
        responses = self._backend.upload_files(list(files.items()))
        if len(responses) != len(files) or any(getattr(response, "error", None) for response in responses):
            raise RuntimeError("OpenShell file upload failed")

    def _execute_runner(self, request_path: str, response_path: str, log_path: str) -> Any:
        command = _runner_command(
            runner_path=str(self._remote_runner),
            manifest_path=str(self._remote_manifest),
            request_path=request_path,
            response_path=response_path,
            limits=self.limits,
        )
        script = f"{shlex.join(command)} >{shlex.quote(log_path)} 2>&1 && /bin/cat {shlex.quote(response_path)}"
        return self._backend.execute(
            shlex.join(["/bin/sh", "-c", script]),
            timeout=math.ceil(self.limits.wall_timeout_seconds) + 5,
        )

    async def _terminate(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(self.runtime.terminate)
        except Exception as exc:  # noqa: BLE001 - cleanup cannot expose provider details
            logger.warning("OpenShell Python termination failed (error_type=%s)", type(exc).__name__)

    async def aclose(self) -> None:
        """Delete the request-owned OpenShell sandbox and its complete process tree."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                await asyncio.to_thread(self.runtime.close)
            except Exception as exc:  # noqa: BLE001 - cleanup cannot replace the agent result
                logger.warning("OpenShell Python cleanup failed (error_type=%s)", type(exc).__name__)


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
