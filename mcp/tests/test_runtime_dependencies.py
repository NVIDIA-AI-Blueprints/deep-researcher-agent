# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production dependency-consistency policy tests."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_runtime_dependencies.py"
_NAMESPACE = runpy.run_path(str(_SCRIPT), run_name="dependency_policy_test")
validate_dependency_records = _NAMESPACE["validate_dependency_records"]
verify_runtime_imports = _NAMESPACE["verify_runtime_imports"]
script_main = _NAMESPACE["main"]


def _records() -> list[dict[str, object]]:
    return [
        {
            "name": "langchain-litellm",
            "version": "0.6.6",
            "requires": ["cryptography>=46.0.5,<49.0.0"],
        },
        {
            "name": "nvidia-nat-core",
            "version": "1.8.0",
            "requires": ["cryptography<47,>=46.0.6"],
        },
        {
            "name": "oci",
            "version": "2.178.0",
            "requires": ["cryptography (<47.0.0,>=3.2.1)"],
        },
        {
            "name": "cryptography",
            "version": "50.0.0",
            "requires": [],
        },
    ]


def test_exact_security_overrides_are_visible() -> None:
    result = validate_dependency_records(_records(), {"cryptography": "50.0.0"})

    assert result == {
        "security_overrides": [
            "langchain-litellm==0.6.6 requires cryptography<49.0.0,>=46.0.5; using 50.0.0",
            "nvidia-nat-core==1.8.0 requires cryptography<47,>=46.0.6; using 50.0.0",
            "oci==2.178.0 requires cryptography<47.0.0,>=3.2.1; using 50.0.0",
        ]
    }


def test_unexpected_incompatibility_fails() -> None:
    records = _records()
    records.append({"name": "new-package", "version": "1.0", "requires": ["cryptography<40"]})

    with pytest.raises(ValueError, match="unexpected dependency incompatibilities"):
        validate_dependency_records(records, {"cryptography": "50.0.0"})


def test_changed_override_fails_closed() -> None:
    records = _records()
    records[0]["version"] = "1.8.1"

    with pytest.raises(ValueError, match="unexpected dependency incompatibilities"):
        validate_dependency_records(records, {"cryptography": "50.0.0"})


def test_resolved_override_must_be_removed_from_policy() -> None:
    records = _records()
    records[0]["requires"] = ["cryptography>=50.0.0"]

    with pytest.raises(ValueError, match="stale security override exceptions"):
        validate_dependency_records(records, {"cryptography": "50.0.0"})


def test_missing_dependency_fails() -> None:
    with pytest.raises(ValueError, match="missing installed dependency"):
        validate_dependency_records(_records(), {})


def test_verify_runtime_imports_accepts_frozen_environment() -> None:
    # The mcp test environment resolves from the same frozen lock as the
    # release image, so the release import canary must pass here too.
    verify_runtime_imports()


def test_verify_runtime_imports_rejects_release_pin_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import version as real_version

    def fake_version(name: str) -> str:
        if name == "mcp":
            return "0.0.0"
        return real_version(name)

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    with pytest.raises(ValueError, match="release pin mismatch: mcp==0.0.0, expected 1.28.1"):
        verify_runtime_imports()


def test_verify_runtime_imports_rejects_missing_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: ())
    with pytest.raises(ValueError, match="missing entry point: tavily_web_search in group nat.plugins"):
        verify_runtime_imports()


def test_main_verify_imports_reports_verification(capsys: pytest.CaptureFixture[str]) -> None:
    assert script_main(["--verify-imports"]) == 0
    output = capsys.readouterr().out
    assert "security_overrides" in output
    assert "Deep Researcher Agent MCP runtime verified" in output


def test_main_without_flag_skips_verification(capsys: pytest.CaptureFixture[str]) -> None:
    assert script_main([]) == 0
    assert "Deep Researcher Agent MCP runtime verified" not in capsys.readouterr().out


def test_main_rejects_unknown_arguments() -> None:
    with pytest.raises(SystemExit, match="unknown arguments"):
        script_main(["--verify-imports", "--bogus"])
