# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenShell release contract and safe component diagnostics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "openshell"


def _load(name: str) -> ModuleType:
    path = _SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPT_DIR))
    return module


@pytest.fixture(scope="module")
def contract_module() -> ModuleType:
    return _load("version_contract")


@pytest.fixture(scope="module")
def inspector() -> ModuleType:
    _load("version_contract")
    return _load("check_versions")


def test_release_contract_is_pinned_to_certified_installer(contract_module: ModuleType) -> None:
    contract = contract_module.load_contract()

    assert contract.version == "0.0.88"
    assert contract.release_tag == "v0.0.88"
    assert (
        contract.installer_sha256
        == "c15d6cb8090e1c7c8d79a320b5bcbdaf1c15c2363942d81e84b56e03b836249e"  # pragma: allowlist secret
    )
    assert contract.adapter_version == "0.1.0"


def _patch_common(
    inspector: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gateway_type: str = "local",
    gateway_version: str | None = "0.0.88",
) -> None:
    monkeypatch.setattr(inspector.importlib.metadata, "version", lambda _name: "0.0.88")
    monkeypatch.setattr(inspector, "_cli_version", lambda _path: "0.0.88")
    monkeypatch.setattr(inspector, "_gateway_type", lambda _path, _name: gateway_type)
    monkeypatch.setattr(inspector, "_live_gateway_version", lambda _name: gateway_version)


def test_matching_local_components_are_accepted(inspector: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(inspector, monkeypatch)
    monkeypatch.setattr(
        inspector,
        "_homebrew_components",
        lambda: (["nvidia/openshell/openshell"], "nvidia/openshell/openshell", "0.0.88", "0.0.88"),
    )

    report = inspector.inspect_components(gateway_name="openshell", system="Darwin")

    assert report.reason_code is None
    assert report.live_gateway_version == "0.0.88"


def test_missing_local_package_prints_exact_installer(inspector: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(inspector, monkeypatch, gateway_version=None)
    monkeypatch.setattr(inspector, "_homebrew_components", lambda: ([], None, None, None))

    report = inspector.inspect_components(gateway_name="openshell", include_live=False, system="Darwin")

    assert report.reason_code == "packaged_gateway_missing"
    assert report.remediation == "./scripts/openshell/install_gateway.sh"


def test_matching_but_stopped_local_gateway_recommends_launcher(
    inspector: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(inspector, monkeypatch, gateway_version=None)
    monkeypatch.setattr(
        inspector,
        "_homebrew_components",
        lambda: (["nvidia/openshell/openshell"], "nvidia/openshell/openshell", "0.0.88", "0.0.88"),
    )

    report = inspector.inspect_components(gateway_name="openshell", system="Darwin")

    assert report.reason_code == "gateway_unavailable"
    assert "start_openshell_gateway.sh" in str(report.remediation)


@pytest.mark.parametrize(
    ("formula_version", "packaged_version"),
    [("0.0.72", "0.0.72"), ("0.0.88", "0.0.72")],
)
def test_stale_local_package_is_classified(
    inspector: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    formula_version: str,
    packaged_version: str,
) -> None:
    _patch_common(inspector, monkeypatch)
    monkeypatch.setattr(
        inspector,
        "_homebrew_components",
        lambda: (["nvidia/openshell/openshell"], "nvidia/openshell/openshell", formula_version, packaged_version),
    )

    report = inspector.inspect_components(gateway_name="openshell", system="Darwin")

    assert report.reason_code == "component_version_mismatch"


def test_multiple_local_installations_fail_as_ambiguous(inspector: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(inspector, monkeypatch)
    monkeypatch.setattr(
        inspector,
        "_homebrew_components",
        lambda: (["nvidia/openshell/openshell", "deep-researcher/local-openshell/openshell"], None, None, None),
    )

    report = inspector.inspect_components(gateway_name="openshell", system="Darwin")

    assert report.reason_code == "ambiguous_gateway_installation"


def test_bare_formula_is_preserved_as_ambiguous(inspector: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(inspector, monkeypatch)
    monkeypatch.setattr(inspector.shutil, "which", lambda _name: "/opt/homebrew/bin/brew")

    def fake_run(command: list[str]) -> str | None:
        args = command[1:]
        if args == ["list", "--formula", "--full-name"]:
            return "openshell"
        if args == ["services", "list", "--json"]:
            return "[]"
        if args == ["list", "--versions", "openshell"]:
            return "openshell 0.0.88"
        if args == ["--prefix", "openshell"]:
            return "/opt/homebrew/opt/openshell"
        return None

    monkeypatch.setattr(inspector, "_run", fake_run)
    monkeypatch.setattr(inspector, "_cli_version", lambda _path: "0.0.88")

    formulas, formula, formula_version, packaged_version = inspector._homebrew_components()

    assert formulas == ["openshell"]
    assert formula == "openshell"
    assert formula_version == "0.0.88"
    assert packaged_version == "0.0.88"

    report = inspector.inspect_components(gateway_name="openshell", system="Darwin")

    assert report.reason_code == "ambiguous_gateway_installation"


def test_remote_mismatch_never_recommends_local_install(inspector: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(inspector, monkeypatch, gateway_type="remote", gateway_version="0.0.72")

    report = inspector.inspect_components(gateway_name="enterprise", system="Darwin")

    assert report.reason_code == "remote_gateway_version_mismatch"
    assert "install_gateway.sh" not in str(report.remediation)
    assert "enterprise" in str(report.remediation)


def test_sdk_mismatch_recommends_deep_researcher_environment_repair(
    inspector: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(inspector, monkeypatch, gateway_type="remote")
    monkeypatch.setattr(inspector.importlib.metadata, "version", lambda _name: "0.0.72")

    report = inspector.inspect_components(gateway_name="enterprise", system="Linux")

    assert report.reason_code == "component_version_mismatch"
    assert "setup_openshell.sh --openshell-version 0.0.88" in str(report.remediation)


def test_json_output_contains_only_allowlisted_fields(
    inspector: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = inspector.ComponentReport(
        certified_version="0.0.88",
        sdk_version="0.0.88",
        virtualenv_cli_version="0.0.88",
        homebrew_formula=None,
        homebrew_formula_version=None,
        packaged_cli_version=None,
        live_gateway_version="0.0.88",
        gateway_type="remote",
        reason_code=None,
        remediation=None,
    )
    monkeypatch.setattr(inspector, "inspect_components", lambda **_kwargs: report)

    assert inspector.main(["--json", "--gateway-name", "enterprise"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "certified_version",
        "gateway_type",
        "homebrew_formula",
        "homebrew_formula_version",
        "live_gateway_version",
        "packaged_cli_version",
        "reason_code",
        "remediation",
        "sdk_version",
        "virtualenv_cli_version",
    }
