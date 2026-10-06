# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the allowlisted OpenShell components without exposing configuration data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from version_contract import load_contract

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:(?:-dev\.|\.dev)\d+(?:\+g[0-9a-fA-F]+)?)?")
_LOCAL_REMEDIATION = "./scripts/openshell/install_gateway.sh"
_OFFICIAL_FORMULA = "nvidia/openshell/openshell"


@dataclass(frozen=True)
class ComponentReport:
    """Safe, automation-friendly OpenShell component report."""

    certified_version: str
    sdk_version: str | None
    virtualenv_cli_version: str | None
    homebrew_formula: str | None
    homebrew_formula_version: str | None
    packaged_cli_version: str | None
    live_gateway_version: str | None
    gateway_type: str | None
    reason_code: str | None
    remediation: str | None


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _extract_version(value: str | None) -> str | None:
    match = _VERSION_PATTERN.search(value or "")
    return match.group(0) if match is not None else None


def _cli_version(binary: Path) -> str | None:
    if not binary.is_file():
        return None
    return _extract_version(_run([str(binary), "--version"]))


def _gateway_type(binary: Path, gateway_name: str) -> str | None:
    if not binary.is_file():
        return None
    payload = _run([str(binary), "gateway", "list", "-o", "json"])
    if payload is None:
        return None
    try:
        registrations = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(registrations, list):
        return None
    for registration in registrations:
        if isinstance(registration, dict) and registration.get("name") == gateway_name:
            value = registration.get("type")
            return value.lower() if isinstance(value, str) else None
    return None


def _homebrew_components() -> tuple[list[str], str | None, str | None, str | None]:
    brew = shutil.which("brew")
    if brew is None:
        return [], None, None, None
    formula_output = _run([brew, "list", "--formula", "--full-name"]) or ""
    formulas = sorted(
        {line.strip() for line in formula_output.splitlines() if line.strip().split("/")[-1] == "openshell"}
    )
    service_output = _run([brew, "services", "list", "--json"])
    if service_output:
        try:
            services = json.loads(service_output)
        except json.JSONDecodeError:
            services = []
        if isinstance(services, list):
            service_formulas = [
                entry["name"]
                for entry in services
                if isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                and entry["name"].split("/")[-1] == "openshell"
            ]
            if formulas:
                service_formulas = [name for name in service_formulas if "/" in name and name not in formulas]
            formulas.extend(service_formulas)
            formulas = sorted(set(formulas))
    if len(formulas) != 1:
        return formulas, None, None, None
    formula = formulas[0]
    formula_version = _extract_version(_run([brew, "list", "--versions", formula]))
    prefix = _run([brew, "--prefix", formula])
    packaged_cli = _cli_version(Path(prefix) / "bin" / "openshell") if prefix else None
    return formulas, formula, formula_version, packaged_cli


def _live_gateway_version(gateway_name: str) -> str | None:
    try:
        from openshell.sandbox import SandboxClient

        with SandboxClient.from_active_cluster(cluster=gateway_name) as client:
            return _extract_version(str(client.health().version))
    except Exception:  # noqa: BLE001 - diagnostics intentionally discard SDK and gateway details
        return None


def inspect_components(
    *,
    gateway_name: str,
    include_live: bool = True,
    system: str | None = None,
) -> ComponentReport:
    """Inspect versions and return one sanitized reason and remediation."""
    contract = load_contract()
    certified = contract.version
    try:
        sdk_version = _extract_version(importlib.metadata.version("openshell"))
    except importlib.metadata.PackageNotFoundError:
        sdk_version = None
    cli_path = _REPO_ROOT / ".venv" / "bin" / "openshell"
    cli_version = _cli_version(cli_path)
    gateway_type = _gateway_type(cli_path, gateway_name)
    current_system = system or platform.system()

    formulas: list[str] = []
    formula = formula_version = packaged_cli_version = None
    if current_system == "Darwin" and gateway_type != "remote":
        formulas, formula, formula_version, packaged_cli_version = _homebrew_components()
    gateway_version = _live_gateway_version(gateway_name) if include_live else None

    reason = None
    remediation = None
    local_macos = current_system == "Darwin" and gateway_type != "remote"
    if local_macos and (len(formulas) > 1 or (formula is not None and formula != _OFFICIAL_FORMULA)):
        reason = "ambiguous_gateway_installation"
    elif local_macos and formula is None:
        reason = "packaged_gateway_missing"
    elif sdk_version != certified or cli_version != certified:
        reason = "component_version_mismatch"
        remediation = (
            "Run ./scripts/openshell/setup_openshell.sh "
            f"--openshell-version {certified} to repair the Deep Researcher Agent environment."
        )
    elif local_macos and (formula_version != certified or packaged_cli_version != certified):
        reason = "component_version_mismatch"
    elif include_live and gateway_version is None:
        reason = "gateway_unavailable"
        if gateway_type == "remote":
            remediation = f"Restore registered gateway '{gateway_name}' through its external operator."
        else:
            remediation = "Run ./scripts/openshell/start_openshell_gateway.sh to start or verify the packaged service."
    elif include_live and gateway_version != certified:
        reason = "remote_gateway_version_mismatch" if gateway_type == "remote" else "component_version_mismatch"

    if reason is not None and remediation is None:
        if local_macos:
            remediation = _LOCAL_REMEDIATION
        else:
            remediation = f"Upgrade registered gateway '{gateway_name}' and its CLI/SDK to OpenShell {certified}."

    return ComponentReport(
        certified_version=certified,
        sdk_version=sdk_version,
        virtualenv_cli_version=cli_version,
        homebrew_formula=formula,
        homebrew_formula_version=formula_version,
        packaged_cli_version=packaged_cli_version,
        live_gateway_version=gateway_version,
        gateway_type=gateway_type,
        reason_code=reason,
        remediation=remediation,
    )


def _print_human(report: ComponentReport) -> None:
    values: list[tuple[str, Any]] = [
        ("Certified version", report.certified_version),
        ("Deep Researcher Agent SDK version", report.sdk_version),
        ("Deep Researcher Agent virtual-environment CLI version", report.virtualenv_cli_version),
        ("Homebrew formula", report.homebrew_formula),
        ("Homebrew formula version", report.homebrew_formula_version),
        ("Packaged CLI version", report.packaged_cli_version),
        ("Live gateway version", report.live_gateway_version),
    ]
    for label, value in values:
        print(f"{label}: {value or 'not detected'}")
    if report.reason_code:
        print(f"Reason: {report.reason_code}", file=sys.stderr)
        print(f"Remediation: {report.remediation}", file=sys.stderr)
    else:
        print("OpenShell component versions match the certified stack.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-name", default="openshell")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = inspect_components(gateway_name=args.gateway_name, include_live=not args.skip_live)
    if args.json:
        print(json.dumps(asdict(report), sort_keys=True))
    else:
        _print_human(report)
    return 1 if report.reason_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
