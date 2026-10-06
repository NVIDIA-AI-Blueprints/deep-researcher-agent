# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the explicit packaged OpenShell gateway installer."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTALLER = _REPO_ROOT / "scripts" / "openshell" / "install_gateway.sh"
_CHECKSUM = "c15d6cb8090e1c7c8d79a320b5bcbdaf1c15c2363942d81e84b56e03b836249e"  # pragma: allowlist secret


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path, *, uid: int = 501, checksum: str = _CHECKSUM) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _write_executable(
        bin_dir / "id",
        f'#!/bin/bash\nif [[ "${{1:-}}" == \'-u\' ]]; then echo {uid}; else /usr/bin/id "$@"; fi\n',
    )
    _write_executable(
        bin_dir / "uname",
        "#!/bin/bash\nif [[ \"${1:-}\" == '-s' ]]; then echo Darwin; else echo arm64; fi\n",
    )
    _write_executable(
        bin_dir / "brew",
        """#!/bin/bash
if [[ "$*" == "list --formula --full-name" ]]; then echo nvidia/openshell/openshell; exit 0; fi
echo "brew $*" >>"$FAKE_LOG"
exit 0
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/bin/bash
echo "curl $*" >>"$FAKE_LOG"
[[ "${FAKE_DOWNLOAD_FAIL:-false}" != "true" ]] || exit 22
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then printf '#!/bin/sh\nexit 0\n' >"$2"; exit 0; fi
  shift
done
exit 2
""",
    )
    _write_executable(bin_dir / "shasum", f"#!/bin/bash\necho '{checksum}  $2'\n")
    _write_executable(
        bin_dir / "sh",
        '#!/bin/bash\necho "installer version=${OPENSHELL_VERSION:-unset} $*" >>"$FAKE_LOG"\n',
    )
    python_wrapper = bin_dir / "python"
    _write_executable(
        python_wrapper,
        """#!/bin/bash
if [[ "${1:-}" == *version_contract.py ]]; then
  case "${3:-}" in
    release-tag) echo v0.0.88 ;;
    installer-sha256) echo "$EXPECTED_SHA256" ;;
  esac
  exit 0
fi
if [[ "${1:-}" == *check_versions.py ]]; then
  echo "check-versions $*" >>"$FAKE_LOG"
  exit "${FAKE_VERIFY_EXIT:-0}"
fi
exec "$REAL_PYTHON" "$@"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HOME": str(tmp_path / "home"),
            "PYTHON_BIN": str(python_wrapper),
            "REAL_PYTHON": sys.executable,
            "FAKE_LOG": str(log),
            "EXPECTED_SHA256": _CHECKSUM,
        }
    )
    return env, log


def _run(tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    effective_env, _ = _environment(tmp_path) if env is None else (env, tmp_path / "calls.log")
    return subprocess.run(
        [str(_INSTALLER), *args],
        cwd=_REPO_ROOT,
        env=effective_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_is_non_mutating_and_names_certified_release(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)

    result = _run(tmp_path, "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert "v0.0.88" in result.stdout
    assert "no files, taps, packages, or services were changed" in result.stdout
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "curl " not in calls
    assert "installer " not in calls
    assert "check-versions " not in calls


def test_root_execution_is_refused(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path, uid=0)

    result = _run(tmp_path, "--yes", env=env)

    assert result.returncode != 0
    assert "not root" in result.stderr


def test_noninteractive_install_requires_explicit_yes(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)

    result = _run(tmp_path, env=env)

    assert result.returncode != 0
    assert "rerun with --yes" in result.stderr
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "curl " not in calls


def test_checksum_mismatch_stops_before_official_installer(tmp_path: Path) -> None:
    env, log = _environment(tmp_path, checksum="0" * 64)
    config = Path(env["HOME"]) / ".config" / "openshell" / "gateway.env"
    config.parent.mkdir(parents=True)
    config.write_text("SAFE_SETTING=unchanged\n", encoding="utf-8")

    result = _run(tmp_path, "--yes", "--colima", env=env)

    assert result.returncode != 0
    assert "gateway_installer_checksum_mismatch" in result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "curl " in calls
    assert "installer " not in calls
    assert config.read_text(encoding="utf-8") == "SAFE_SETTING=unchanged\n"


def test_download_failure_is_sanitized(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    env["FAKE_DOWNLOAD_FAIL"] = "true"

    result = _run(tmp_path, "--yes", env=env)

    assert result.returncode != 0
    assert "gateway_installer_download_failed" in result.stderr


def test_colima_configuration_preserves_unrelated_values(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)
    config = Path(env["HOME"]) / ".config" / "openshell" / "gateway.env"
    config.parent.mkdir(parents=True)
    config.write_text("SAFE_SETTING=keep\nDOCKER_HOST=old\n", encoding="utf-8")
    config.chmod(0o640)

    result = _run(tmp_path, "--yes", "--colima", env=env)

    assert result.returncode == 0, result.stderr
    assert config.read_text(encoding="utf-8").splitlines() == [
        "SAFE_SETTING=keep",
        "DOCKER_HOST=unix://" + env["HOME"] + "/.colima/default/docker.sock",
        "OPENSHELL_DRIVERS=docker",
    ]
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    calls = log.read_text(encoding="utf-8")
    assert "installer version=v0.0.88" in calls
    assert "check-versions " in calls


def test_ambiguous_formula_installation_is_refused(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)
    brew = Path(env["PATH"].split(":", maxsplit=1)[0]) / "brew"
    _write_executable(
        brew,
        """#!/bin/bash
if [[ "$*" == "list --formula --full-name" ]]; then
  printf '%s\n' nvidia/openshell/openshell deep-researcher/local-openshell/openshell
  exit 0
fi
echo "brew $*" >>"$FAKE_LOG"
""",
    )

    result = _run(tmp_path, "--yes", env=env)

    assert result.returncode != 0
    assert "ambiguous_gateway_installation" in result.stderr
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "curl " not in calls


def test_bare_formula_is_not_treated_as_official_from_registered_tap(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)
    brew = Path(env["PATH"].split(":", maxsplit=1)[0]) / "brew"
    _write_executable(
        brew,
        """#!/bin/bash
if [[ "$*" == "list --formula --full-name" ]]; then echo openshell; exit 0; fi
if [[ "$*" == "tap" ]]; then echo nvidia/openshell; exit 0; fi
echo "brew $*" >>"$FAKE_LOG"
""",
    )

    result = _run(tmp_path, "--yes", env=env)

    assert result.returncode != 0
    assert "ambiguous_gateway_installation" in result.stderr
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "curl " not in calls


def test_ambiguous_service_installation_is_refused(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)
    brew = Path(env["PATH"].split(":", maxsplit=1)[0]) / "brew"
    _write_executable(
        brew,
        """#!/bin/bash
if [[ "$*" == "list --formula --full-name" ]]; then echo nvidia/openshell/openshell; exit 0; fi
if [[ "$*" == "services list --json" ]]; then
  printf '%s\n' '[{"name":"openshell"},{"name":"deep-researcher/local-openshell/openshell"}]'
  exit 0
fi
if [[ "$*" == "tap" ]]; then echo nvidia/openshell; exit 0; fi
echo "brew $*" >>"$FAKE_LOG"
""",
    )

    result = _run(tmp_path, "--yes", env=env)

    assert result.returncode != 0
    assert "multiple OpenShell services" in result.stderr
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "curl " not in calls


def test_post_install_component_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    env["FAKE_VERIFY_EXIT"] = "1"

    result = _run(tmp_path, "--yes", env=env)

    assert result.returncode != 0
    assert "component verification failed" in result.stderr


def test_installer_source_contains_no_implicit_or_insecure_lifecycle_shortcuts() -> None:
    source = _INSTALLER.read_text(encoding="utf-8")

    assert "curl |" not in source
    assert "launchctl setenv" not in source
    assert "tap-new deep-researcher/" not in source
    assert "openshell-gateway" not in source
