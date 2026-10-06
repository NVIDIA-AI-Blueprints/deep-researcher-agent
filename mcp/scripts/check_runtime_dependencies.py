# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the production environment, including exact security overrides."""

from __future__ import annotations

import json
import sys
from importlib.metadata import distributions
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_ALLOWED_INCOMPATIBILITIES = {
    ("langchain-litellm", "0.6.6", "cryptography", "50.0.0", "<49.0.0,>=46.0.5"),
    ("nvidia-nat-core", "1.8.0", "cryptography", "50.0.0", "<47,>=46.0.6"),
    ("oci", "2.178.0", "cryptography", "50.0.0", "<47.0.0,>=3.2.1"),
}

# Release-image import canary: every module the server needs at runtime must be
# importable from the production closure, the release pins must match exactly,
# and the NAT plugin entry point must be discoverable. Importing deep_researcher_mcp.server
# requires DEEP_RESEARCHER_MCP_CONFIG (or a source checkout) because the module builds its
# default ASGI app at import time.
_REQUIRED_IMPORTS = (
    "deep_researcher_mcp",
    "deep_researcher_mcp.server",
    "deep_researcher_mcp.jobs",
    "deep_researcher_mcp.job_store",
    "deep_researcher_mcp.workflow_runner",
    "deep_researcher_agent.common",
    "tavily_web_search",
    "knowledge_layer",
    "asyncpg",
)
_PINNED_RELEASE_VERSIONS = {
    "mcp": "1.28.1",
    "nvidia-nat-core": "1.8.0",
}
_REQUIRED_ENTRY_POINTS = (("nat.plugins", "tavily_web_search"),)


def validate_dependency_records(
    records: list[dict[str, Any]],
    installed_versions: dict[str, str],
) -> dict[str, list[str]]:
    environment = default_environment()
    environment["extra"] = ""
    conflicts: set[tuple[str, str, str, str, str]] = set()

    for record in records:
        owner = canonicalize_name(str(record["name"]))
        owner_version = str(record["version"])
        for requirement_text in record.get("requires", []):
            requirement = Requirement(str(requirement_text))
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            installed_version = installed_versions.get(dependency)
            if installed_version is None:
                raise ValueError(f"missing installed dependency: {owner} requires {dependency}")
            if requirement.specifier and Version(installed_version) not in requirement.specifier:
                conflicts.add(
                    (
                        owner,
                        owner_version,
                        dependency,
                        installed_version,
                        str(requirement.specifier),
                    )
                )

    unexpected = sorted(conflicts - _ALLOWED_INCOMPATIBILITIES)
    if unexpected:
        raise ValueError(f"unexpected dependency incompatibilities: {unexpected}")
    stale = sorted(_ALLOWED_INCOMPATIBILITIES - conflicts)
    if stale:
        raise ValueError(f"stale security override exceptions: {stale}")

    return {
        "security_overrides": [
            f"{owner}=={owner_version} requires {dependency}{specifier}; using {installed_version}"
            for owner, owner_version, dependency, installed_version, specifier in sorted(conflicts)
        ]
    }


def validate_environment() -> dict[str, list[str]]:
    installed = list(distributions())
    installed_versions = {
        canonicalize_name(dist.metadata["Name"]): dist.version for dist in installed if dist.metadata.get("Name")
    }
    records = [
        {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "requires": list(dist.requires or ()),
        }
        for dist in installed
        if dist.metadata.get("Name")
    ]
    return validate_dependency_records(records, installed_versions)


def verify_runtime_imports() -> None:
    import importlib
    from importlib.metadata import entry_points
    from importlib.metadata import version

    for module_name in _REQUIRED_IMPORTS:
        importlib.import_module(module_name)
    for distribution_name, pinned in _PINNED_RELEASE_VERSIONS.items():
        installed = version(distribution_name)
        if installed != pinned:
            raise ValueError(f"release pin mismatch: {distribution_name}=={installed}, expected {pinned}")
    for group, name in _REQUIRED_ENTRY_POINTS:
        if not any(entry_point.name == name for entry_point in entry_points(group=group)):
            raise ValueError(f"missing entry point: {name} in group {group}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    unknown = [argument for argument in arguments if argument != "--verify-imports"]
    if unknown:
        raise SystemExit(f"unknown arguments: {unknown}")
    try:
        result = validate_environment()
        if "--verify-imports" in arguments:
            verify_runtime_imports()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    if "--verify-imports" in arguments:
        print("Deep Researcher Agent MCP runtime verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
