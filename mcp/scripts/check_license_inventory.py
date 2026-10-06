# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and validate release-license evidence for the MCP production closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from collections import Counter
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlsplit

_DIRECT_RUNTIME_DEPENDENCIES = {
    "deep-researcher-agent",
    "asyncpg",
    "langchain-core",
    "mcp",
    "msgpack",
    "nvidia-nat-core",
    "python-dotenv",
    "starlette",
    "tavily-web-search",
    "uvicorn",
}
_FORBIDDEN_COMPONENTS = {"maas-sdk", "mos-sdk"}
_LOCAL_SOURCE_COMPONENTS = {
    ("deep-researcher-agent", "2.2.0"): "../",
    ("knowledge-layer", "1.0.0"): "../sources/knowledge_layer",
    ("tavily-web-search", "1.0.0"): "../sources/tavily_web_search",
}
_APPROVED_GIT_SOURCES = {
    ("nemo-relay", "0.8.0"): (
        "https://github.com/NVIDIA/NeMo-Relay.git"
        "?rev=ffb24817442bac99212da0971b13bdad5bc4d84d"
        "#ffb24817442bac99212da0971b13bdad5bc4d84d"
    )
}
_MCP_LOCK_PATH = Path(__file__).resolve().parents[1] / "uv.lock"

# These distributions omit license metadata but bundle a license file. A version
# or file change intentionally fails until the new evidence is reviewed.
_BUNDLED_FILE_ONLY = {
    ("docx2txt", "0.9"): "36c734f24a70b2671daa3364d21aaecca6bce2a9e321d1fb884344e06026f8e7",  # pragma: allowlist secret
    (
        "py-rust-stemmers",
        "0.1.8",
    ): "423fde63e5092fb0300819d1e91d2696d6897cd239e737f665c18194e58d9744",  # pragma: allowlist secret
}

# CycloneDX includes marker-gated packages for other platforms/Python versions.
# They cannot be inspected in the Linux Python 3.13 release environment.
_PLATFORM_EXCLUDED = {
    ("async-timeout", "5.0.1"): "python_full_version <= '3.11.2'",
    ("pywin32", "312"): "sys_platform == 'win32'",
    ("win32-setctime", "1.2.0"): "sys_platform == 'win32'",
}

# Ambiguous metadata and NOTICE content remain explicit manual-review findings.
# Exact hashes prevent a classifier or NOTICE change from being silently waived.
_AMBIGUOUS_METADATA = {
    ("fastembed", "0.8.0"): {
        "fingerprint": "21bf08f6142501aaf76fba9e395c02d8e8a649695d3b3946c7c2c21af7796af0",  # pragma: allowlist secret
        "reason": "notice-content",
    },
    ("nemoguardrails", "0.21.0"): {
        "fingerprint": "33f3b85395081c050469df36739324ade184292728b08e95b8ced23541f75be6",  # pragma: allowlist secret
        "reason": "ambiguous-metadata",
    },
}

# LGPL packages are not treated as prohibited, but their exact current evidence
# is surfaced for organizational distribution-policy review.
_WEAK_COPYLEFT = {
    ("crc32c", "2.7.1"): "36d52a6109036f61280f7575c912971fca070425e49a8137e00341d6a2db7a0f",  # pragma: allowlist secret
    (
        "psycopg",
        "3.3.4",
    ): "129eb8549fc870d7b4bae205c4e15a73d869df0845cbaeafde36314733250d58",  # pragma: allowlist secret
    (
        "psycopg-binary",
        "3.3.4",
    ): "129eb8549fc870d7b4bae205c4e15a73d869df0845cbaeafde36314733250d58",  # pragma: allowlist secret
    (
        "psycopg-pool",
        "3.3.1",
    ): "129eb8549fc870d7b4bae205c4e15a73d869df0845cbaeafde36314733250d58",  # pragma: allowlist secret
}

_NOTICE_REVIEW_PATTERNS = {
    "cc-by-nc": re.compile(r"cc-by-nc", re.IGNORECASE),
    "gemma-terms": re.compile(r"gemma\s+(terms|is provided)", re.IGNORECASE),
    "noncommercial": re.compile(r"non[- ]?commercial", re.IGNORECASE),
    "proprietary": re.compile(r"proprietary", re.IGNORECASE),
}


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _license_files(dist: Any) -> list[dict[str, str]]:
    evidence: dict[str, dict[str, str]] = {}
    for relative in dist.files or ():
        name = Path(str(relative)).name.lower()
        if not name.startswith(("license", "licence", "copying", "notice")):
            continue
        path = dist.locate_file(relative)
        if not path.is_file():
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        text = payload.decode(errors="replace")
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")[:200]
        evidence.setdefault(
            digest,
            {
                "filename": Path(str(relative)).name,
                "sha256": digest,
                "first_line": first_line,
            },
        )
    return sorted(evidence.values(), key=lambda item: (item["filename"], item["sha256"]))


def _notice_flags(dist: Any) -> list[str]:
    flags: set[str] = set()
    for relative in dist.files or ():
        if not Path(str(relative)).name.lower().startswith("notice"):
            continue
        path = dist.locate_file(relative)
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        flags.update(label for label, pattern in _NOTICE_REVIEW_PATTERNS.items() if pattern.search(text))
    return sorted(flags)


def _license_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("license_expression") or ""),
            str(row.get("license_field") or ""),
            *[str(value) for value in row.get("license_classifiers", [])],
        ]
    )


def _is_weak_copyleft(value: str) -> bool:
    normalized = value.upper()
    return "LGPL" in normalized or "LESSER GENERAL PUBLIC LICENSE" in normalized


def _is_strong_copyleft(value: str) -> bool:
    normalized = value.upper()
    normalized = normalized.replace("LGPL", "").replace("LESSER GENERAL PUBLIC LICENSE", "")
    return (
        "AGPL" in normalized
        or "AFFERO" in normalized
        or "GENERAL PUBLIC LICENSE" in normalized
        or re.search(r"(^|[^A-Z])GPL(?:V|[-\s]|$)", normalized) is not None
    )


def _evidence_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "evidence_kind": row.get("evidence_kind"),
        "license_expression": row.get("license_expression"),
        "license_field": row.get("license_field"),
        "license_classifiers": row.get("license_classifiers", []),
        "license_files": sorted(
            (str(item.get("filename")), str(item.get("sha256"))) for item in row.get("license_files", [])
        ),
        "notice_review_flags": row.get("notice_review_flags", []),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def validate_sbom(sbom: dict[str, Any]) -> None:
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.5":
        raise ValueError("expected a CycloneDX 1.5 SBOM")
    root = sbom.get("metadata", {}).get("component", {})
    if (root.get("name"), root.get("version")) != ("deep-researcher-mcp-server", "0.1.0"):
        raise ValueError("SBOM root must be deep-researcher-mcp-server 0.1.0")

    components = sbom.get("components")
    if not isinstance(components, list):
        raise ValueError("SBOM must contain a component list")

    observed_local: set[tuple[str, str]] = set()
    for component in components:
        name = _canonicalize(str(component.get("name", "")))
        version = str(component.get("version") or "")
        if name in _FORBIDDEN_COMPONENTS:
            raise ValueError(f"forbidden private SDK dependency in SBOM: {name}=={version}")
        properties = {
            str(item.get("name")): str(item.get("value"))
            for item in component.get("properties", [])
            if item.get("name") is not None
        }
        record = (name, version)
        if component.get("purl") is None:
            if record not in _LOCAL_SOURCE_COMPONENTS or "uv:workspace:path" in properties:
                raise ValueError(f"unapproved local dependency source: {name}=={version}")
            observed_local.add(record)
            continue

        purl = str(component.get("purl"))
        expected_purl = f"pkg:pypi/{name}@{version}"
        if purl == expected_purl:
            continue

        parsed_purl = urlsplit(purl)
        approved_git_source = _APPROVED_GIT_SOURCES.get(record)
        qualifiers = parse_qs(parsed_purl.query, keep_blank_values=True, strict_parsing=True)
        base_purl = parsed_purl._replace(query="", fragment="").geturl()
        if (
            approved_git_source is not None
            and base_purl == expected_purl
            and qualifiers == {"vcs_url": [approved_git_source]}
            and not parsed_purl.fragment
        ):
            continue

        raise ValueError(f"dependency is not from the public PyPI source contract: {name}=={version}")

    if observed_local != set(_LOCAL_SOURCE_COMPONENTS):
        raise ValueError("SBOM local component set differs from the approved public source contract")


def validate_lock_sources(lock_path: Path = _MCP_LOCK_PATH) -> None:
    """Preserve path provenance that uv's CycloneDX export does not encode."""
    lock = tomllib.loads(lock_path.read_text())
    expected_local = {
        **_LOCAL_SOURCE_COMPONENTS,
        ("deep-researcher-mcp-server", "0.1.0"): ".",
    }
    observed_local: dict[tuple[str, str], str] = {}
    for package in lock.get("package", []):
        name = _canonicalize(str(package.get("name", "")))
        version = str(package.get("version") or "")
        source = package.get("source", {})
        if source.get("registry") is not None:
            if source != {"registry": "https://pypi.org/simple"}:
                raise ValueError(f"dependency uses an unapproved registry source: {name}=={version}")
            continue
        if source.get("git") is not None:
            if source == {"git": _APPROVED_GIT_SOURCES.get((name, version))}:
                continue
        editable = source.get("editable")
        if not isinstance(editable, str):
            raise ValueError(f"dependency uses an unapproved lock source: {name}=={version}")
        observed_local[(name, version)] = editable

    if observed_local != expected_local:
        raise ValueError("MCP lock local sources differ from the approved public source contract")


def build_inventory(sbom_path: Path) -> dict[str, Any]:
    sbom_bytes = sbom_path.read_bytes()
    sbom = json.loads(sbom_bytes)
    validate_sbom(sbom)
    rows: list[dict[str, Any]] = []

    source_components: list[dict[str, str | None]] = []
    for component in sbom.get("components", []):
        name = _canonicalize(str(component["name"]))
        locked_version = str(component.get("version") or "")
        properties = {
            str(item.get("name")): str(item.get("value"))
            for item in component.get("properties", [])
            if item.get("name") is not None
        }
        resolution_marker = properties.get("uv:package:marker")
        source_components.append(
            {"name": name, "locked_version": locked_version, "resolution_marker": resolution_marker}
        )
        row: dict[str, Any] = {
            "name": name,
            "locked_version": locked_version,
            "resolution_marker": resolution_marker,
        }
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            row.update(
                installed_version=None,
                version_matches=None,
                evidence_kind="not-installed-in-target-environment",
                license_expression=None,
                license_field=None,
                license_classifiers=[],
                license_files=[],
                notice_review_flags=[],
            )
        else:
            metadata = dist.metadata
            expression = (metadata.get("License-Expression") or "").strip() or None
            license_field = " ".join((metadata.get("License") or "").split())[:200] or None
            classifiers = [
                value.removeprefix("License :: ").strip()
                for value in metadata.get_all("Classifier", [])
                if value.startswith("License :: ")
            ]
            files = _license_files(dist)
            if expression:
                evidence_kind = "expression"
            elif classifiers:
                evidence_kind = "classifier"
            elif license_field:
                evidence_kind = "license-field"
            elif files:
                evidence_kind = "bundled-license-file-only"
            else:
                evidence_kind = "missing"
            row.update(
                installed_version=dist.version,
                version_matches=dist.version == locked_version,
                evidence_kind=evidence_kind,
                license_expression=expression,
                license_field=license_field,
                license_classifiers=classifiers,
                license_files=files,
                notice_review_flags=_notice_flags(dist),
            )
        rows.append(row)

    counts = Counter(row["evidence_kind"] for row in rows)
    return {
        "schema": "deep-researcher-mcp-license-inventory-v2",
        "source_sbom_sha256": hashlib.sha256(sbom_bytes).hexdigest(),
        "source_components": source_components,
        "summary": {
            "components": len(rows),
            "evidence_counts": dict(sorted(counts.items())),
            "version_mismatches": sum(row["version_matches"] is False for row in rows),
            "notice_review_components": sum(bool(row["notice_review_flags"]) for row in rows),
        },
        "components": rows,
    }


def validate_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    rows = inventory.get("components")
    if not isinstance(rows, list):
        raise ValueError("license inventory must contain a component list")

    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = _canonicalize(str(row.get("name", "")))
        if not name or name in by_name:
            raise ValueError(f"duplicate or empty license inventory component: {name!r}")
        by_name[name] = row

    source_components = inventory.get("source_components")
    if source_components is not None:
        source_rows = {
            (
                _canonicalize(str(row.get("name", ""))),
                str(row.get("locked_version") or ""),
                row.get("resolution_marker"),
            )
            for row in source_components
        }
        inventory_rows = {
            (name, str(row.get("locked_version") or ""), row.get("resolution_marker")) for name, row in by_name.items()
        }
        if source_rows != inventory_rows:
            raise ValueError("SBOM and license inventory component sets differ")

    missing_direct = sorted(_DIRECT_RUNTIME_DEPENDENCIES - set(by_name))
    if missing_direct:
        raise ValueError(f"direct runtime dependencies missing from license inventory: {missing_direct}")

    manual_review: list[dict[str, str]] = []
    platform_excluded: list[str] = []
    bundled_only: list[str] = []
    processed_file_only: set[tuple[str, str]] = set()
    processed_platform: set[tuple[str, str]] = set()
    processed_ambiguous: set[tuple[str, str]] = set()
    processed_weak: set[tuple[str, str]] = set()
    for name, row in sorted(by_name.items()):
        version = str(row.get("locked_version") or "")
        key = (name, version)
        evidence_kind = str(row.get("evidence_kind") or "")
        license_text = _license_text(row)
        notice_flags = set(row.get("notice_review_flags") or [])
        evidence_fingerprint = _evidence_fingerprint(row)

        if row.get("version_matches") is False:
            raise ValueError(f"installed version does not match SBOM for {name}")

        if evidence_kind == "not-installed-in-target-environment":
            expected_marker = _PLATFORM_EXCLUDED.get(key)
            if expected_marker is None or row.get("resolution_marker") != expected_marker:
                raise ValueError(f"unexpected component absent from target environment: {name}=={version}")
            processed_platform.add(key)
            platform_excluded.append(f"{name}=={version}")
            continue

        if evidence_kind in {"", "missing"}:
            raise ValueError(f"missing license evidence: {name}=={version}")

        if evidence_kind == "bundled-license-file-only":
            expected_fingerprint = _BUNDLED_FILE_ONLY.get(key)
            if expected_fingerprint is None or evidence_fingerprint != expected_fingerprint:
                raise ValueError(f"unreviewed bundled-only license evidence: {name}=={version}")
            processed_file_only.add(key)
            bundled_only.append(f"{name}=={version}")

        if _is_strong_copyleft(license_text):
            raise ValueError(f"strong-copyleft dependency metadata requires review: {name}=={version}")

        if _is_weak_copyleft(license_text):
            expected_fingerprint = _WEAK_COPYLEFT.get(key)
            if expected_fingerprint is None or evidence_fingerprint != expected_fingerprint:
                raise ValueError(f"unreviewed weak-copyleft evidence: {name}=={version}")
            processed_weak.add(key)
            manual_review.append({"component": f"{name}=={version}", "reason": "weak-copyleft"})

        proprietary = "proprietary" in license_text.lower()
        expected = _AMBIGUOUS_METADATA.get(key)
        if expected is not None:
            if evidence_fingerprint != expected["fingerprint"]:
                raise ValueError(f"reviewed license or NOTICE evidence changed for {name}=={version}")
            processed_ambiguous.add(key)
            manual_review.append({"component": f"{name}=={version}", "reason": str(expected["reason"])})
        elif proprietary or notice_flags:
            raise ValueError(f"unreviewed proprietary or NOTICE metadata: {name}=={version}")

    exception_sets = [
        (set(_BUNDLED_FILE_ONLY), processed_file_only),
        (set(_PLATFORM_EXCLUDED), processed_platform),
        (set(_AMBIGUOUS_METADATA), processed_ambiguous),
        (set(_WEAK_COPYLEFT), processed_weak),
    ]
    stale_exceptions = sorted(
        f"{name}=={version}"
        for expected_keys, processed_keys in exception_sets
        for name, version in expected_keys - processed_keys
    )
    if stale_exceptions:
        raise ValueError(f"stale or changed release-review exceptions: {stale_exceptions}")

    return {
        "components": len(by_name),
        "bundled_license_file_only": bundled_only,
        "manual_review_required": manual_review,
        "platform_excluded": platform_excluded,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sbom", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        validate_lock_sources()
        inventory = build_inventory(args.sbom)
        validation = validate_inventory(inventory)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    inventory["validation"] = validation
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
