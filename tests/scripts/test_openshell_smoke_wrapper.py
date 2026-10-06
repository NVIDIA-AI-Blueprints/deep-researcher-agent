# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thin OpenShell live-acceptance wrapper."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER = _REPO_ROOT / "scripts" / "openshell" / "smoke_openshell_isolation.py"
_LIVE_TEST = "tests/deep_researcher_agent/agents/deep_researcher/sandbox/test_openshell_live.py"


@pytest.fixture
def wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_openshell_isolation", _WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_translates_arguments_and_preserves_environment(
    wrapper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        recorded.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("DEEP_RESEARCHER_EXISTING_SETTING", "preserved")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = wrapper.main(
        [
            "--gateway",
            "enterprise",
            "--workspace",
            "research",
            "--policy",
            "policy.yaml",
            "--image",
            "image:tag",
            "--expected-gateway-version",
            "0.0.88",
            "--allow-best-effort-landlock",
        ]
    )

    assert result == 0
    assert recorded["command"] == [sys.executable, "-m", "pytest", "-m", "integration", "-vv", _LIVE_TEST]
    assert recorded["cwd"] == _REPO_ROOT
    assert recorded["check"] is False
    env = recorded["env"]
    assert isinstance(env, dict)
    assert env["DEEP_RESEARCHER_EXISTING_SETTING"] == "preserved"
    assert env["DEEP_RESEARCHER_OPENSHELL_LIVE_TESTS"] == "1"
    assert env["DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME"] == "enterprise"
    assert env["DEEP_RESEARCHER_OPENSHELL_WORKSPACE"] == "research"
    assert env["DEEP_RESEARCHER_OPENSHELL_POLICY_FILE"] == "policy.yaml"
    assert env["DEEP_RESEARCHER_OPENSHELL_IMAGE"] == "image:tag"
    assert env["DEEP_RESEARCHER_OPENSHELL_EXPECTED_GATEWAY_VERSION"] == "0.0.88"
    assert env["DEEP_RESEARCHER_OPENSHELL_LIVE_ALLOW_BEST_EFFORT"] == "1"


def test_wrapper_uses_generated_policy_and_single_pytest_target(
    wrapper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEP_RESEARCHER_OPENSHELL_EXPECTED_GATEWAY_VERSION", raising=False)
    monkeypatch.delenv("DEEP_RESEARCHER_OPENSHELL_POLICY_FILE", raising=False)
    monkeypatch.setenv("DEEP_RESEARCHER_OPENSHELL_WORKSPACE", "")
    args = wrapper._args([])

    assert args.workspace == "default"
    assert args.policy == "configs/openshell/generated/deep-researcher-openshell-policy.yaml"
    assert args.expected_gateway_version is None
    assert wrapper._command()[-1] == _LIVE_TEST
    assert wrapper._command().count(_LIVE_TEST) == 1


def test_wrapper_propagates_pytest_exit_code(
    wrapper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=17))

    assert wrapper.main([]) == 17


def test_wrapper_does_not_import_openshell_sdk(wrapper: ModuleType) -> None:
    source = _WRAPPER.read_text(encoding="utf-8")

    assert "import openshell" not in source
    assert "import grpc" not in source
