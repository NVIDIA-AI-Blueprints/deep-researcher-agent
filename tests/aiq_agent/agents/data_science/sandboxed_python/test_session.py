# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stateless OpenShell Python execution and structured-data evidence."""

import ast
import json
import sys
from pathlib import Path

import pytest
from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.protocol import FileUploadResponse

from aiq_agent.agents.data_science.sandboxed_python import runner
from aiq_agent.agents.data_science.sandboxed_python.session import OpenShellPythonRunner
from aiq_agent.agents.data_science.sandboxed_python.session import PythonRunnerLimits
from aiq_agent.agents.data_science.sandboxed_python.session import _runner_command


class _FakeBackend:
    def __init__(self, responses: list[ExecuteResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.uploads: list[dict[str, bytes]] = []
        self.commands: list[tuple[str, int | None]] = []

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.uploads.append(dict(files))
        return [FileUploadResponse(path=path, error=None) for path, _content in files]

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append((command, timeout))
        if self.responses:
            return self.responses.pop(0)
        return ExecuteResponse(output="", exit_code=0)


class _FakeRuntime:
    def __init__(self, backend: _FakeBackend) -> None:
        self.sandbox_backend = backend
        self.workdir = "/sandbox/data-science-job"
        self.closed = False
        self.terminated = False

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True


def _write_evidence(root: Path) -> Path:
    result_path = root / "structured_1.json"
    result_path.write_text(
        json.dumps({"sql": "SELECT value", "rows": [{"value": 2}, {"value": 5}]}),
        encoding="utf-8",
    )
    manifest_path = root / "structured-results.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "results": [
                    {
                        "ref": "structured_1",
                        "provider": "ontology",
                        "tool_name": "ontology__text_to_sql",
                        "question": "Values",
                        "database_name": "example",
                        "request_id": "r1",
                        "row_count": 2,
                        "columns": ["value"],
                        "truncated": False,
                        "path": str(result_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_runner_uses_a_fresh_namespace_for_every_script(tmp_path: Path) -> None:
    manifest_path = _write_evidence(tmp_path)
    first = runner.execute("frame = pd.DataFrame({'value': [1, 2, 3]})\nframe", manifest_path, 50_000)
    second = runner.execute("int(frame['value'].sum())", manifest_path, 50_000)

    assert first["status"] == "ok"
    assert "frame" in first["variables"]
    assert second["status"] == "error"
    assert second["error"] == "NameError"


def test_runner_exposes_exact_analysis_helpers(tmp_path: Path) -> None:
    manifest_path = _write_evidence(tmp_path)
    response = runner.execute(
        "df = analysis_rows('structured_1')\n(int(df['value'].sum()), analysis_sql('structured_1'))",
        manifest_path,
        50_000,
    )

    assert response["status"] == "ok"
    assert ast.literal_eval(response["result"]) == (7, "SELECT value")


def test_runner_serializes_system_exit_and_next_script_remains_usable(tmp_path: Path) -> None:
    manifest_path = _write_evidence(tmp_path)
    failed = runner.execute("value = 7\nexit()", manifest_path, 50_000)
    recovered = runner.execute("7 + 1", manifest_path, 50_000)

    assert failed["status"] == "error"
    assert failed["error"] == "SystemExit"
    assert recovered["status"] == "ok"
    assert recovered["result"] == "8"


def test_runner_main_rejects_incomplete_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner.py"])

    with pytest.raises(ValueError, match="Python runner requires"):
        runner.main()


def test_runner_output_is_capped_while_written(tmp_path: Path) -> None:
    response = runner.execute("print('x' * 100_000)", _write_evidence(tmp_path), 64)

    assert response["status"] == "ok"
    assert response["output"] == ("x" * 64) + "\n... output truncated ..."


def test_runner_command_applies_hard_resource_limits() -> None:
    command = _runner_command(
        runner_path="/sandbox/runner.py",
        manifest_path="/sandbox/evidence/manifest.json",
        request_path="/sandbox/requests/script-1.json",
        response_path="/sandbox/requests/script-1.response.json",
        limits=PythonRunnerLimits(
            wall_timeout_seconds=45,
            max_output_chars=50_000,
            max_memory_mb=2_048,
            max_cpu_seconds=120,
            max_processes=64,
            max_open_files=128,
            max_file_bytes=25_000_000,
        ),
    )

    assert command == [
        "/usr/bin/timeout",
        "--signal=KILL",
        "45.000s",
        "/usr/bin/prlimit",
        "--as=2147483648:2147483648",
        "--cpu=120:120",
        "--nproc=64:64",
        "--nofile=128:128",
        "--fsize=25000000:25000000",
        "--",
        "python3",
        "-I",
        "-u",
        "/sandbox/runner.py",
        "/sandbox/evidence/manifest.json",
        "/sandbox/requests/script-1.json",
        "50000",
        "/sandbox/requests/script-1.response.json",
    ]


@pytest.mark.asyncio
async def test_runner_uploads_only_trusted_runtime_request_and_evidence(tmp_path: Path) -> None:
    manifest_path = _write_evidence(tmp_path)
    response = json.dumps({"status": "ok", "result": "7", "output": "", "variables": ["df"]})
    backend = _FakeBackend(responses=[ExecuteResponse(output=response, exit_code=0)])
    runtime = _FakeRuntime(backend)
    python_runner = OpenShellPythonRunner(
        runtime=runtime,
        host_manifest_path=manifest_path,
        host_evidence_root=tmp_path,
    )
    try:
        result = json.loads(await python_runner.execute("analysis_rows('structured_1')['value'].sum()"))
    finally:
        await python_runner.aclose()

    assert result["status"] == "ok"
    assert result["result"] == "7"
    uploaded = backend.uploads[0]
    assert set(uploaded) == {
        "/sandbox/data-science-job/evidence/structured-results.json",
        "/sandbox/data-science-job/evidence/structured_1.json",
        "/sandbox/data-science-job/requests/script-1.json",
        "/sandbox/data-science-job/runner.py",
    }
    remote_manifest = json.loads(uploaded["/sandbox/data-science-job/evidence/structured-results.json"])
    assert remote_manifest["results"][0]["path"] == "/sandbox/data-science-job/evidence/structured_1.json"
    assert remote_manifest["results"][0]["provider"] == "ontology"
    assert str(tmp_path) not in uploaded["/sandbox/data-science-job/evidence/structured-results.json"].decode()
    assert "/usr/bin/prlimit" in backend.commands[0][0]
    assert "/bin/cat /sandbox/data-science-job/requests/script-1.response.json" in backend.commands[0][0]
    assert runtime.closed is True
    assert runtime.terminated is False


@pytest.mark.asyncio
async def test_runner_reuploads_authoritative_evidence_before_every_script(tmp_path: Path) -> None:
    manifest_path = _write_evidence(tmp_path)
    response = json.dumps({"status": "ok", "result": "7", "output": "", "variables": []})
    backend = _FakeBackend(
        responses=[
            ExecuteResponse(output=response, exit_code=0),
            ExecuteResponse(output=response, exit_code=0),
        ]
    )
    python_runner = OpenShellPythonRunner(
        runtime=_FakeRuntime(backend),
        host_manifest_path=manifest_path,
        host_evidence_root=tmp_path,
    )
    try:
        await python_runner.execute("analysis_rows('structured_1')")
        await python_runner.execute("analysis_rows('structured_1')")
    finally:
        await python_runner.aclose()

    assert len(backend.uploads) == 2
    for upload in backend.uploads:
        assert "/sandbox/data-science-job/evidence/structured_1.json" in upload
        assert "/sandbox/data-science-job/evidence/structured-results.json" in upload


@pytest.mark.asyncio
async def test_runner_terminates_sandbox_after_script_timeout(tmp_path: Path) -> None:
    manifest_path = _write_evidence(tmp_path)
    backend = _FakeBackend(responses=[ExecuteResponse(output="", exit_code=124)])
    runtime = _FakeRuntime(backend)
    python_runner = OpenShellPythonRunner(
        runtime=runtime,
        host_manifest_path=manifest_path,
        host_evidence_root=tmp_path,
        limits=PythonRunnerLimits(wall_timeout_seconds=1),
    )

    response = json.loads(await python_runner.execute("while True: pass"))

    assert response == {"status": "error", "error": "execution_timed_out"}
    assert runtime.terminated is True


@pytest.mark.asyncio
async def test_runner_rejects_evidence_outside_request_root(tmp_path: Path) -> None:
    root = tmp_path / "request"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"rows":[]}', encoding="utf-8")
    manifest = root / "structured-results.json"
    manifest.write_text(
        json.dumps({"version": 1, "results": [{"ref": "structured_1", "path": str(outside)}]}),
        encoding="utf-8",
    )
    runtime = _FakeRuntime(_FakeBackend())
    python_runner = OpenShellPythonRunner(
        runtime=runtime,
        host_manifest_path=manifest,
        host_evidence_root=root,
    )

    response = json.loads(await python_runner.execute("1 + 1"))

    assert response == {"status": "error", "error": "sandbox_execution_failed"}
    assert runtime.terminated is True
