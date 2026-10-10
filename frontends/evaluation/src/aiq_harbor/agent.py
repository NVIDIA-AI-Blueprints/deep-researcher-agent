# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.base import with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

DEFAULT_OUTPUT_FILE = "/workspace/answer.txt"
AUTO_OUTPUT_FILE = "auto"
TASK_OUTPUT_FILE_ENV = "AIQ_OUTPUT_FILE"
ALLOWED_OUTPUT_FILENAMES = {"answer.txt", "report.md"}
_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


class AiqHarborAgent(BaseInstalledAgent):
    """Run a preinstalled AI-Q workflow directly inside a Harbor environment."""

    SUPPORTS_ATIF = True
    SUPPORTS_WINDOWS = False

    _CONTAINER_RUNNER = "/installed-agent/aiq_runner.py"
    _CONTAINER_CONFIG = "/installed-agent/aiq-agent-config.yml"
    _INSTRUCTION_FILE = "/installed-agent/instruction.txt"
    _TRAJECTORY_FILE = "/logs/agent/trajectory.json"
    _JOB_FILE = "/logs/agent/aiq_job.json"
    _STATE_FILE = "/logs/agent/aiq_state.json"
    _EVENTS_FILE = "/logs/agent/aiq_events.jsonl"
    _SETUP_FILE = "/logs/agent/aiq_setup.json"
    _PID_FILE = "/logs/agent/aiq-agent.pid"

    def __init__(
        self,
        *args: Any,
        config_file: str | Path,
        output_file: str = DEFAULT_OUTPUT_FILE,
        python_path: str = "/app/.venv/bin/python",
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self._config_file = Path(config_file).expanduser()
        self._output_file = self._validate_output_file(output_file)
        self._python_path = self._validate_python_path(python_path)

    @staticmethod
    def name() -> str:
        return "aiq-harbor"

    def get_version_command(self) -> str | None:
        return f"{shlex.quote(self._python_path)} -c " + shlex.quote(
            "import importlib.metadata as m; print(m.version('aiq-agent'))"
        )

    async def install(self, environment: BaseEnvironment) -> None:
        """Verify the image-provided runtime; never install packages per trial."""
        command = f"test -x {shlex.quote(self._python_path)} && {shlex.quote(self._python_path)} -c " + shlex.quote(
            "import importlib.metadata as m; "
            "import aiq_agent, nat; "
            "print('aiq-agent=' + m.version('aiq-agent')); "
            "print('nvidia-nat=' + m.version('nvidia-nat'))"
        )
        result = await environment.exec(command=command)
        if result.return_code != 0:
            raise RuntimeError(
                "AI-Q runtime preflight failed; the task image must contain /app/.venv, "
                "aiq-agent, and nvidia-nat. " + self._exec_detail(result)
            )

    async def setup(self, environment: BaseEnvironment) -> None:
        self._validate_host_config()
        await super().setup(environment)

        directory_result = await environment.exec(
            command=(
                "mkdir -p /installed-agent /workspace /logs/agent && "
                "chmod 0755 /installed-agent && chmod 0777 /workspace /logs/agent"
            ),
            user="root",
        )
        if directory_result.return_code != 0:
            raise RuntimeError("Failed to prepare AI-Q runtime directories. " + self._exec_detail(directory_result))

        runner_path = Path(__file__).with_name("runner.py")
        await environment.upload_file(runner_path, self._CONTAINER_RUNNER)
        await environment.upload_file(self._config_file, self._CONTAINER_CONFIG)
        chmod_result = await environment.exec(
            command=f"chmod 0755 {shlex.quote(self._CONTAINER_RUNNER)}",
            user="root",
        )
        if chmod_result.return_code != 0:
            raise RuntimeError("Failed to make the AI-Q runner executable. " + self._exec_detail(chmod_result))

        result = await environment.exec(
            command=self._build_validate_command(),
            env=self._runtime_env(),
            cwd="/app",
        )
        await self._persist_exec_logs(environment, "aiq-setup", result)
        if result.return_code != 0:
            raise RuntimeError("AI-Q configuration preflight failed. " + self._exec_detail(result))

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context  # Harbor backfills it from synced logs in populate_context_post_run().
        await self._upload_text(environment, self._INSTRUCTION_FILE, instruction)
        session_id = str(uuid.uuid4())
        output_file = self._resolve_output_file(environment)
        command = self._build_run_command(session_id, output_file)
        try:
            result = await environment.exec(
                command=command,
                env=self._runtime_env(),
                cwd="/app",
            )
        except asyncio.CancelledError:
            await self._terminate_runner(environment)
            raise

        await self._persist_exec_logs(environment, "aiq-agent", result)
        if result.return_code != 0:
            raise self._classify_exec_error(command, result)

    def populate_context_post_run(self, context: AgentContext) -> None:
        job = self._read_json(self.logs_dir / "aiq_job.json")
        if job is None:
            job = self._read_json(self.logs_dir / "aiq_setup.json")
        if job is not None:
            context.metadata = {"aiq": job}

        trajectory = self._read_json(self.logs_dir / "trajectory.json")
        if trajectory is None:
            return
        metrics = trajectory.get("final_metrics")
        if not isinstance(metrics, dict):
            return
        context.n_input_tokens = int(metrics.get("total_prompt_tokens") or 0)
        context.n_output_tokens = int(metrics.get("total_completion_tokens") or 0)
        context.n_cache_tokens = int(metrics.get("total_cached_tokens") or 0)
        cost = metrics.get("total_cost_usd")
        context.cost_usd = float(cost) if cost is not None else None

    def _validate_host_config(self) -> None:
        if not self._config_file.is_file():
            raise FileNotFoundError(f"AI-Q config file not found: {self._config_file}")

    def _runtime_env(self) -> dict[str, str]:
        env = {str(key): str(value) for key, value in self._extra_env.items() if value is not None}
        env.setdefault("NVIDIA_NAT_LOG_LEVEL", os.environ.get("NVIDIA_NAT_LOG_LEVEL", "WARNING"))
        return env

    def _build_validate_command(self) -> str:
        return " ".join(
            [
                shlex.quote(self._python_path),
                shlex.quote(self._CONTAINER_RUNNER),
                "validate",
                "--config-file",
                shlex.quote(self._CONTAINER_CONFIG),
                "--metadata-output",
                shlex.quote(self._SETUP_FILE),
            ]
        )

    def _build_run_command(self, session_id: str, output_file: str) -> str:
        runner_command = " ".join(
            [
                shlex.quote(self._python_path),
                shlex.quote(self._CONTAINER_RUNNER),
                "run",
                "--config-file",
                shlex.quote(self._CONTAINER_CONFIG),
                "--instruction-file",
                shlex.quote(self._INSTRUCTION_FILE),
                "--output-file",
                shlex.quote(output_file),
                "--trajectory-output",
                shlex.quote(self._TRAJECTORY_FILE),
                "--metadata-output",
                shlex.quote(self._JOB_FILE),
                "--state-output",
                shlex.quote(self._STATE_FILE),
                "--events-output",
                shlex.quote(self._EVENTS_FILE),
                "--session-id",
                shlex.quote(session_id),
            ]
        )
        return f"echo $$ > {shlex.quote(self._PID_FILE)}; exec {runner_command}"

    async def _terminate_runner(self, environment: BaseEnvironment) -> None:
        try:
            await environment.exec(
                command=(
                    f"if test -s {shlex.quote(self._PID_FILE)}; then "
                    f"kill -TERM $(cat {shlex.quote(self._PID_FILE)}) 2>/dev/null || true; fi"
                ),
                user="root",
                timeout_sec=5,
            )
        except Exception:
            pass

    async def _persist_exec_logs(self, environment: BaseEnvironment, prefix: str, result: Any) -> None:
        await self._upload_text(
            environment,
            f"/logs/agent/{prefix}-stdout.txt",
            self._redact_sensitive(result.stdout or ""),
        )
        await self._upload_text(
            environment,
            f"/logs/agent/{prefix}-stderr.txt",
            self._redact_sensitive(result.stderr or ""),
        )

    async def _upload_text(self, environment: BaseEnvironment, target_path: str, value: str) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(value)
            source_path = Path(handle.name)
        try:
            await environment.upload_file(source_path, target_path)
        finally:
            source_path.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _exec_detail(self, result: Any, limit: int = 2000) -> str:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        detail = stderr or stdout or f"exit code {result.return_code}"
        return self._redact_sensitive(detail)[:limit]

    def _classify_exec_error(self, command: str, result: Any) -> NonZeroAgentExitCodeError:
        safe_result = SimpleNamespace(
            return_code=result.return_code,
            stdout=self._redact_sensitive(result.stdout or ""),
            stderr=self._redact_sensitive(result.stderr or ""),
        )
        return super()._classify_exec_error(command, safe_result)

    def _redact_sensitive(self, value: str) -> str:
        for name, secret in self._runtime_env().items():
            if not secret or not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS):
                continue
            value = value.replace(secret, f"${{{name}}}")
        return value

    @staticmethod
    def _validate_python_path(value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute():
            raise ValueError("python_path must be an absolute container path")
        return str(path)

    @staticmethod
    def _validate_output_file(value: str) -> str:
        if value == AUTO_OUTPUT_FILE:
            return value
        return AiqHarborAgent._validate_concrete_output_file(value)

    @staticmethod
    def _validate_concrete_output_file(value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or path.parent != PurePosixPath("/workspace"):
            raise ValueError("output_file must point directly under /workspace")
        if path.name not in ALLOWED_OUTPUT_FILENAMES:
            allowed = ", ".join(sorted(ALLOWED_OUTPUT_FILENAMES))
            raise ValueError(f"output_file filename must be one of: {allowed}")
        return str(path)

    def _resolve_output_file(self, environment: BaseEnvironment) -> str:
        if self._output_file != AUTO_OUTPUT_FILE:
            return self._output_file

        task_env_config = getattr(environment, "task_env_config", None)
        task_env = getattr(task_env_config, "env", None)
        value = task_env.get(TASK_OUTPUT_FILE_ENV) if isinstance(task_env, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                f"output_file={AUTO_OUTPUT_FILE!r} requires task.toml [environment.env] "
                f"{TASK_OUTPUT_FILE_ENV} to select the task artifact"
            )
        try:
            return self._validate_concrete_output_file(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid task-selected {TASK_OUTPUT_FILE_ENV}: {value!r}") from exc
