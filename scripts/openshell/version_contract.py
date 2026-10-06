# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate Deep Researcher Agent's certified OpenShell release contract."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXACT_DEPENDENCY = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)")


@dataclass(frozen=True)
class OpenShellContract:
    """Allowlisted release metadata used by setup and diagnostics."""

    version: str
    release_tag: str
    installer_sha256: str
    adapter_version: str | None


def load_contract(pyproject: Path | None = None) -> OpenShellContract:
    """Load the release contract and cross-check a published optional extra when present."""
    path = pyproject or (_REPO_ROOT / "pyproject.toml")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("deep-researcher", {}).get("openshell", {})
    release_tag = table.get("release-tag")
    adapter_version = table.get("adapter-version")
    checksum = table.get("installer-sha256")
    if not isinstance(release_tag, str) or re.fullmatch(r"v\d+\.\d+\.\d+", release_tag) is None:
        raise ValueError("invalid_openshell_release_tag")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("invalid_openshell_installer_checksum")
    if not isinstance(adapter_version, str) or re.fullmatch(r"\d+\.\d+\.\d+", adapter_version) is None:
        raise ValueError("invalid_openshell_adapter_version")

    version = release_tag.removeprefix("v")
    extra = data.get("project", {}).get("optional-dependencies", {}).get("openshell", [])
    exact = {}
    for requirement in extra:
        if isinstance(requirement, str) and (match := _EXACT_DEPENDENCY.match(requirement)) is not None:
            exact[match.group("name").lower()] = match.group("version")
    openshell_version = exact.get("openshell")
    if openshell_version is not None and openshell_version != version:
        raise ValueError("openshell_extra_version_mismatch")
    extra_adapter_version = exact.get("langchain-nvidia-openshell")
    if extra_adapter_version is not None and extra_adapter_version != adapter_version:
        raise ValueError("openshell_adapter_extra_version_mismatch")

    return OpenShellContract(
        version=version,
        release_tag=release_tag,
        installer_sha256=checksum,
        adapter_version=adapter_version,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("version", "release-tag", "installer-sha256", "adapter-version"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_contract()
    if args.json:
        print(json.dumps(asdict(contract), sort_keys=True))
    elif args.field:
        print(getattr(contract, args.field.replace("-", "_")) or "")
    else:
        print(contract.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
