# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for OpenShell provisioning and gateway lifecycle ownership."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATEWAY_SCRIPT = _REPO_ROOT / "scripts" / "openshell" / "start_openshell_gateway.sh"
_SETUP_SCRIPT = _REPO_ROOT / "scripts" / "openshell" / "setup_openshell.sh"
_E2E_SCRIPT = _REPO_ROOT / "scripts" / "start_e2e.sh"


def _fake_openshell(tmp_path: Path) -> tuple[Path, Path, Path]:
    binary = tmp_path / "openshell"
    log = tmp_path / "openshell.log"
    state = tmp_path / "sandbox.state"
    binary.write_text(
        """#!/bin/bash
set -euo pipefail
echo "$*" >>"$FAKE_LOG"
if [[ "$1 ${2:-}" == "gateway list" ]]; then
    echo "$FAKE_GATEWAYS_JSON"
elif [[ "$1 ${2:-}" == "gateway select" ]]; then
    exit 0
elif [[ "$1" == "status" ]]; then
    exit "${FAKE_STATUS_EXIT:-0}"
elif [[ "$1 ${2:-}" == "sandbox create" ]]; then
    name=""
    shift 2
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--name" ]]; then name="$2"; break; fi
        shift
    done
    echo "$name" >"$FAKE_STATE"
    [[ "${FAKE_CREATE_FAIL:-false}" != "true" ]]
elif [[ "$1 ${2:-}" == "sandbox list" ]]; then
    if [[ -s "$FAKE_STATE" ]]; then
        name="$(sed -n '1p' "$FAKE_STATE")"
        if [[ "${FAKE_NEVER_READY:-false}" == "true" ]]; then
            echo "$name Creating"
        else
            echo "$name Ready"
        fi
    fi
elif [[ "$1 ${2:-}" == "sandbox delete" ]]; then
    [[ "${FAKE_DELETE_FAIL:-false}" != "true" ]]
    rm -f "$FAKE_STATE"
fi
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary, log, state


def _run_launcher(
    tmp_path: Path,
    *,
    gateway: dict[str, object],
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    binary, log, state = _fake_openshell(tmp_path)
    python_wrapper = tmp_path / "python"
    python_wrapper.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == *check_openshell_readiness.py ]]; then
    echo "readiness-checker $*" >>"$FAKE_LOG"
    exit "${FAKE_READINESS_EXIT:-0}"
fi
if [[ "${1:-}" == *check_versions.py ]]; then
    echo "version-inspector $*" >>"$FAKE_LOG"
    exit "${FAKE_VERSION_EXIT:-0}"
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    env = os.environ.copy()
    for key in ("OPENSHELL_GATEWAY_ENDPOINT", "OPENSHELL_GATEWAY_INSECURE", "OPENSHELL_GATEWAY_LAUNCH_BIN"):
        env.pop(key, None)
    env.update(
        {
            "OPENSHELL_BIN": str(binary),
            "PYTHON_BIN": str(python_wrapper),
            "REAL_PYTHON": sys.executable,
            "FAKE_LOG": str(log),
            "FAKE_STATE": str(state),
            "FAKE_GATEWAYS_JSON": json.dumps([gateway]),
            "AIQ_OPENSHELL_STATUS_ATTEMPTS": "1",
            "AIQ_OPENSHELL_PROBE_ATTEMPTS": "1",
            "AIQ_OPENSHELL_DELETE_ATTEMPTS": "1",
            "AIQ_OPENSHELL_POLL_DELAY": "0",
        }
    )
    env.update(extra_env or {})
    result = subprocess.run(
        [
            str(_GATEWAY_SCRIPT),
            "--reuse-existing",
            "--gateway-name",
            "enterprise",
            "--policy-file",
            str(policy),
        ],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, log.read_text(encoding="utf-8") if log.exists() else ""


def test_authenticated_gateway_runs_mandatory_strict_readiness_check(tmp_path: Path) -> None:
    result, calls = _run_launcher(
        tmp_path,
        gateway={
            "name": "enterprise",
            "endpoint": "https://gateway.example.com",
            "type": "remote",
            "auth": "oidc",
            "active": False,
        },
        extra_env={"AIQ_OPENSHELL_WORKSPACE": "research"},
    )

    assert result.returncode == 0, result.stderr
    assert "gateway list -o json" in calls
    assert "gateway select enterprise" in calls
    assert calls.count("version-inspector") == 2
    assert "readiness-checker" in calls
    assert "--gateway-name enterprise" in calls
    assert "--workspace research" in calls
    assert "gateway add" not in calls


def test_plaintext_gateway_is_rejected_before_probe(tmp_path: Path) -> None:
    result, calls = _run_launcher(
        tmp_path,
        gateway={
            "name": "enterprise",
            "endpoint": "http://127.0.0.1:8080",
            "type": "local",
            "auth": "plaintext",
            "active": True,
        },
    )

    assert result.returncode != 0
    assert "not authenticated" in result.stderr
    assert "sandbox create" not in calls


def test_raw_gateway_launcher_is_rejected_independent_of_probe_mode(tmp_path: Path) -> None:
    result, calls = _run_launcher(
        tmp_path,
        gateway={
            "name": "enterprise",
            "endpoint": "https://127.0.0.1:8080",
            "type": "local",
            "auth": "mtls",
            "active": True,
        },
        extra_env={"OPENSHELL_GATEWAY_LAUNCH_BIN": "/usr/bin/openshell-gateway"},
    )

    assert result.returncode != 0
    assert "raw gateway" in result.stderr.lower()
    assert calls == ""


def test_failed_strict_readiness_check_stops_launcher(tmp_path: Path) -> None:
    result, calls = _run_launcher(
        tmp_path,
        gateway={
            "name": "enterprise",
            "endpoint": "https://127.0.0.1:8080",
            "type": "local",
            "auth": "mtls",
            "active": True,
        },
        extra_env={"FAKE_READINESS_EXIT": "1"},
    )

    assert result.returncode != 0
    assert "readiness-checker" in calls
    assert "strict readiness check failed" in result.stderr.lower()


def test_component_mismatch_stops_before_gateway_or_readiness_probe(tmp_path: Path) -> None:
    result, calls = _run_launcher(
        tmp_path,
        gateway={
            "name": "enterprise",
            "endpoint": "https://127.0.0.1:8080",
            "type": "local",
            "auth": "mtls",
            "active": True,
        },
        extra_env={"FAKE_VERSION_EXIT": "1"},
    )

    assert result.returncode != 0
    assert "component version check failed" in result.stderr.lower()
    assert "version-inspector" in calls
    assert "readiness-checker" not in calls
    assert "sandbox create" not in calls


def test_setup_is_provisioning_only_and_migrates_old_lifecycle_flags() -> None:
    source = _SETUP_SCRIPT.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "start_or_verify_gateway" not in source
    assert '"$OPENSHELL_BIN" sandbox create' not in source
    assert "--reinstall-package" not in source
    assert "uv sync --dev --inexact" in source
    assert 'uv pip install "deepagents' not in source
    assert 'export VIRTUAL_ENV="$VENV_DIR"' in source
    assert '*":$VENV_DIR/bin:"*' in source
    assert 'export AIQ_OPENSHELL_WORKSPACE="${AIQ_OPENSHELL_WORKSPACE:-default}"' in source
    assert 'export AIQ_OPENSHELL_WORKSPACE="default"' not in source

    result = subprocess.run(
        [str(_SETUP_SCRIPT), "--gateway-name", "old"],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "start_openshell_gateway.sh" in result.stderr


def test_setup_policy_declares_canonical_openshell_proxy_baseline() -> None:
    source = _SETUP_SCRIPT.read_text(encoding="utf-8")
    policy_template = source.split("emit_policy_header()", maxsplit=1)[1].split("emit_policy_entry()", maxsplit=1)[0]

    assert "    - /proc\n" in policy_template
    assert "/proc/self" not in policy_template


@pytest.mark.parametrize("version", ["latest", "0.0.72", "0.0.81"])
def test_setup_rejects_uncertified_openshell_versions(version: str) -> None:
    result = subprocess.run(
        [str(_SETUP_SCRIPT), "--openshell-version", version, "--skip-build", "--policy", "offline"],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not certified" in result.stderr
    assert "0.0.88" in result.stderr


def test_setup_resolves_docker_desktop_cli_and_credential_helper_together() -> None:
    source = _SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'docker_desktop_bin="/Applications/Docker.app/Contents/Resources/bin"' in source
    assert 'export PATH="$docker_desktop_bin:$PATH"' in source


def test_setup_local_demo_uses_explicit_runtime_override_without_config_copy() -> None:
    source = _SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "--local-demo" in source
    assert 'LANDLOCK_COMPATIBILITY="best_effort"' in source
    assert 'runtime_env="AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false "' in source
    assert "configs/config_openshell.local.yml" not in source


def test_e2e_gateway_start_is_explicit_opt_in() -> None:
    source = _E2E_SCRIPT.read_text(encoding="utf-8")
    result = subprocess.run(
        [str(_E2E_SCRIPT), "--help"],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--start-openshell-gateway" in result.stdout
    assert "START_OPENSHELL_GATEWAY=false" in source
    assert 'if [[ "$START_OPENSHELL_GATEWAY" != "true" ]]' in source
    assert '"$PROJECT_ROOT/scripts/openshell/start_openshell_gateway.sh"' in source
    assert '"$PROJECT_ROOT/scripts/openshell/check_versions.py"' in source
    assert source.index("check_openshell_component_versions\n") < source.index("start_backend\n")
    assert 'PYTHON_BIN="$VENV_DIR/bin/python"' in source
    assert 'NAT_BIN="$VENV_DIR/bin/nat"' in source
    assert '"$NAT_BIN" serve' in source
    assert 'BACKEND_PID=""' in source
    assert 'FRONTEND_PID=""' in source
    assert "${BACKEND_PID:-}" in source
    assert "${FRONTEND_PID:-}" in source
