# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-hygiene policy tests."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_license_inventory.py"
_NAMESPACE = runpy.run_path(str(_SCRIPT), run_name="license_policy_test")
_AMBIGUOUS_METADATA = _NAMESPACE["_AMBIGUOUS_METADATA"]
_BUNDLED_FILE_ONLY = _NAMESPACE["_BUNDLED_FILE_ONLY"]
_DIRECT_RUNTIME_DEPENDENCIES = _NAMESPACE["_DIRECT_RUNTIME_DEPENDENCIES"]
_PLATFORM_EXCLUDED = _NAMESPACE["_PLATFORM_EXCLUDED"]
_WEAK_COPYLEFT = _NAMESPACE["_WEAK_COPYLEFT"]
_APPROVED_GIT_SOURCES = _NAMESPACE["_APPROVED_GIT_SOURCES"]
_evidence_fingerprint = _NAMESPACE["_evidence_fingerprint"]
validate_inventory = _NAMESPACE["validate_inventory"]
validate_sbom = _NAMESPACE["validate_sbom"]
validate_lock_sources = _NAMESPACE["validate_lock_sources"]
_MCP_LOCK_PATH = _NAMESPACE["_MCP_LOCK_PATH"]

_TEST_RELAY_GIT_SOURCE = "https://github.com/NVIDIA/NeMo-Relay.git?rev=test-revision#test-revision"
_TEST_RELAY_VCS_QUALIFIER = "vcs_url=https://github.com/NVIDIA/NeMo-Relay.git%3Frev%3Dtest-revision%23test-revision"
_TEST_RELAY_PURL = f"pkg:pypi/nemo-relay@0.8.0?{_TEST_RELAY_VCS_QUALIFIER}"


def _row(name: str, version: str = "1.0", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "locked_version": version,
        "resolution_marker": None,
        "installed_version": version,
        "version_matches": True,
        "evidence_kind": "expression",
        "license_expression": "Apache-2.0",
        "license_field": None,
        "license_classifiers": [],
        "license_files": [],
        "notice_review_flags": [],
    }
    row.update(overrides)
    return row


def _inventory(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rows = [_row(name) for name in sorted(_DIRECT_RUNTIME_DEPENDENCIES)]
    for name, version in _BUNDLED_FILE_ONLY:
        row = _row(
            name,
            version,
            evidence_kind="bundled-license-file-only",
            license_expression=None,
            license_files=[
                {
                    "filename": "LICENSE.txt" if name == "docx2txt" else "LICENSE",
                    "sha256": f"{name}-license-hash",
                }
            ],
        )
        rows.append(row)
        monkeypatch.setitem(_BUNDLED_FILE_ONLY, (name, version), _evidence_fingerprint(row))
    for (name, version), marker in _PLATFORM_EXCLUDED.items():
        rows.append(
            _row(
                name,
                version,
                installed_version=None,
                version_matches=None,
                evidence_kind="not-installed-in-target-environment",
                license_expression=None,
                resolution_marker=marker,
            )
        )
    for (name, version), expected in _AMBIGUOUS_METADATA.items():
        row = _row(
            name,
            version,
            license_expression=None,
            license_field="Apache License",
            license_classifiers=["Other/Proprietary License"],
            license_files=[{"filename": "LICENSE", "sha256": f"{name}-license-hash"}],
            notice_review_flags=["cc-by-nc", "gemma-terms"] if expected["reason"] == "notice-content" else [],
        )
        rows.append(row)
        monkeypatch.setitem(expected, "fingerprint", _evidence_fingerprint(row))
    for name, version in _WEAK_COPYLEFT:
        row = _row(
            name,
            version,
            license_expression="LGPL-3.0-only",
            license_files=[{"filename": "LICENSE", "sha256": f"{name}-license-hash"}],
        )
        rows.append(row)
        monkeypatch.setitem(_WEAK_COPYLEFT, (name, version), _evidence_fingerprint(row))
    return {"components": rows}


def test_license_inventory_surfaces_exact_review_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(monkeypatch)
    validation = validate_inventory(inventory)

    assert validation["components"] == len(inventory["components"])
    assert validation["bundled_license_file_only"] == [
        "docx2txt==0.9",
        "py-rust-stemmers==0.1.8",
    ]
    assert {item["reason"] for item in validation["manual_review_required"]} == {
        "ambiguous-metadata",
        "notice-content",
        "weak-copyleft",
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row("new-package", evidence_kind="missing", license_expression=None), "missing license evidence"),
        (_row("new-package", license_expression="GPLv3"), "strong-copyleft dependency metadata"),
        (
            _row(
                "new-package",
                license_expression=None,
                license_classifiers=["Other/Proprietary License"],
            ),
            "unreviewed proprietary or NOTICE metadata",
        ),
    ],
)
def test_license_inventory_rejects_unreviewed_evidence(
    row: dict[str, Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(monkeypatch)
    inventory["components"].append(row)
    with pytest.raises(ValueError, match=message):
        validate_inventory(inventory)


def test_license_inventory_rejects_changed_reviewed_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(monkeypatch)
    fastembed = next(row for row in inventory["components"] if row["name"] == "fastembed")
    fastembed["license_files"] = [{"sha256": "changed"}]

    with pytest.raises(ValueError, match="reviewed license or NOTICE evidence changed"):
        validate_inventory(inventory)


def test_license_inventory_rejects_extra_reviewed_file(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(monkeypatch)
    docx2txt = next(row for row in inventory["components"] if row["name"] == "docx2txt")
    docx2txt["license_files"].append({"filename": "COPYING-GPL", "sha256": "new-hash"})

    with pytest.raises(ValueError, match="unreviewed bundled-only license evidence"):
        validate_inventory(inventory)


def test_license_inventory_rejects_weak_metadata_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(monkeypatch)
    psycopg = next(row for row in inventory["components"] if row["name"] == "psycopg")
    psycopg["license_expression"] = "LGPL-2.1-only"

    with pytest.raises(ValueError, match="unreviewed weak-copyleft evidence"):
        validate_inventory(inventory)


def test_license_inventory_requires_every_direct_runtime_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory(monkeypatch)
    inventory["components"] = [
        row for row in inventory["components"] if row["name"] != next(iter(_DIRECT_RUNTIME_DEPENDENCIES))
    ]

    with pytest.raises(ValueError, match="direct runtime dependencies missing"):
        validate_inventory(inventory)


def test_sbom_contract_accepts_the_exact_approved_local_component_set() -> None:
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "deep-researcher-mcp-server", "version": "0.1.0"}},
        "components": [
            {"name": "deep-researcher-agent", "version": "2.2.0"},
            {"name": "knowledge-layer", "version": "1.0.0"},
            {"name": "tavily-web-search", "version": "1.0.0"},
            {
                "name": "asyncpg",
                "version": "0.31.0",
                "purl": "pkg:pypi/asyncpg@0.31.0",
            },
        ],
    }

    validate_sbom(sbom)


def test_sbom_contract_accepts_exact_approved_vcs_qualified_purl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_APPROVED_GIT_SOURCES, ("nemo-relay", "0.8.0"), _TEST_RELAY_GIT_SOURCE)
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "deep-researcher-mcp-server", "version": "0.1.0"}},
        "components": [
            {"name": "deep-researcher-agent", "version": "2.2.0"},
            {"name": "knowledge-layer", "version": "1.0.0"},
            {"name": "tavily-web-search", "version": "1.0.0"},
            {
                "name": "nemo-relay",
                "version": "0.8.0",
                "purl": _TEST_RELAY_PURL,
            },
        ],
    }

    validate_sbom(sbom)


@pytest.mark.parametrize(
    "purl",
    [
        pytest.param(
            "pkg:pypi/nemo-relay@0.8.0?vcs_url=https://github.com/NVIDIA/NeMo-Relay.git%3Frev%3Dwrong%23wrong",
            id="different-revision",
        ),
        pytest.param(
            "pkg:pypi/asyncpg@0.31.0?vcs_url=https://github.com/MagicStack/asyncpg.git%3Frev%3Dabc%23abc",
            id="unapproved-package",
        ),
        pytest.param(
            f"{_TEST_RELAY_PURL}&subdirectory=python",
            id="extra-qualifier",
        ),
        pytest.param(
            f"{_TEST_RELAY_PURL}&{_TEST_RELAY_VCS_QUALIFIER}",
            id="duplicate-vcs-qualifier",
        ),
    ],
)
def test_sbom_contract_rejects_unapproved_vcs_qualified_purl(purl: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_APPROVED_GIT_SOURCES, ("nemo-relay", "0.8.0"), _TEST_RELAY_GIT_SOURCE)
    name, version = ("asyncpg", "0.31.0") if "asyncpg" in purl else ("nemo-relay", "0.8.0")
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "deep-researcher-mcp-server", "version": "0.1.0"}},
        "components": [
            {"name": "deep-researcher-agent", "version": "2.2.0"},
            {"name": "knowledge-layer", "version": "1.0.0"},
            {"name": "tavily-web-search", "version": "1.0.0"},
            {"name": name, "version": version, "purl": purl},
        ],
    }

    with pytest.raises(ValueError, match="dependency is not from the public PyPI source contract"):
        validate_sbom(sbom)


def test_sbom_contract_requires_every_approved_local_component() -> None:
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "deep-researcher-mcp-server", "version": "0.1.0"}},
        "components": [
            {"name": "deep-researcher-agent", "version": "2.2.0"},
            {"name": "knowledge-layer", "version": "1.0.0"},
        ],
    }

    with pytest.raises(ValueError, match="local component set differs"):
        validate_sbom(sbom)


def test_sbom_contract_rejects_non_pypi_dependency_source() -> None:
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "deep-researcher-mcp-server", "version": "0.1.0"}},
        "components": [{"name": "private-runtime", "version": "1.0"}],
    }

    with pytest.raises(ValueError, match="unapproved local dependency source"):
        validate_sbom(sbom)


def test_sbom_contract_rejects_private_sdk_name_from_public_index() -> None:
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "deep-researcher-mcp-server", "version": "0.1.0"}},
        "components": [
            {
                "name": "maas-sdk",
                "version": "2.4.1",
                "purl": "pkg:pypi/maas-sdk@2.4.1",
            }
        ],
    }

    with pytest.raises(ValueError, match="forbidden private SDK dependency"):
        validate_sbom(sbom)


# --- validate_lock_sources: MCP public-source contract ------------------------

# The exact approved lock contract: the four editable path sources plus a
# representative public-PyPI registry dependency. Each test copies this baseline
# and mutates a single entry to exercise one branch of the contract at a time.
_APPROVED_LOCK_PACKAGES = [
    ("deep-researcher-agent", "2.2.0", '{ editable = "../" }'),
    ("knowledge-layer", "1.0.0", '{ editable = "../sources/knowledge_layer" }'),
    ("tavily-web-search", "1.0.0", '{ editable = "../sources/tavily_web_search" }'),
    ("deep-researcher-mcp-server", "0.1.0", '{ editable = "." }'),
    ("asyncpg", "0.31.0", '{ registry = "https://pypi.org/simple" }'),
]


def _write_lock(tmp_path: Path, packages: list[tuple[str, str, str]]) -> Path:
    blocks = [
        f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {source}\n'
        for name, version, source in packages
    ]
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("\n".join(blocks))
    return lock_path


def _without(name: str) -> list[tuple[str, str, str]]:
    return [package for package in _APPROVED_LOCK_PACKAGES if package[0] != name]


def test_lock_sources_accepts_the_exact_approved_contract(tmp_path: Path) -> None:
    validate_lock_sources(_write_lock(tmp_path, _APPROVED_LOCK_PACKAGES))


@pytest.mark.parametrize(
    "source",
    [
        '{ registry = "https://private.example.com/simple" }',
        '{ registry = "https://pypi.org/simple", subdirectory = "vendor" }',
    ],
)
def test_lock_sources_reject_unapproved_registry(tmp_path: Path, source: str) -> None:
    packages = _APPROVED_LOCK_PACKAGES + [("some-dependency", "1.0.0", source)]
    with pytest.raises(ValueError, match="unapproved registry source"):
        validate_lock_sources(_write_lock(tmp_path, packages))


@pytest.mark.parametrize(
    "source",
    [
        '{ git = "https://github.com/example/some-dependency" }',
        '{ url = "https://example.com/some-dependency-1.0.0-py3-none-any.whl" }',
        '{ directory = "../vendor/some_dependency" }',
        "{ editable = true }",
        "{}",
    ],
)
def test_lock_sources_reject_non_editable_local_source(tmp_path: Path, source: str) -> None:
    packages = _APPROVED_LOCK_PACKAGES + [("some-dependency", "1.0.0", source)]
    with pytest.raises(ValueError, match="unapproved lock source"):
        validate_lock_sources(_write_lock(tmp_path, packages))


@pytest.mark.parametrize(
    "packages",
    [
        pytest.param(_without("deep-researcher-mcp-server"), id="missing-approved-source"),
        pytest.param(
            _APPROVED_LOCK_PACKAGES + [("exa-web-search", "1.0.0", '{ editable = "../sources/exa_web_search" }')],
            id="extra-unapproved-editable",
        ),
        pytest.param(
            _without("deep-researcher-agent") + [("deep-researcher-agent", "2.2.0", '{ editable = "../wrong" }')],
            id="approved-name-wrong-path",
        ),
        pytest.param(
            _without("deep-researcher-agent") + [("deep-researcher-agent", "9.9.9", '{ editable = "../" }')],
            id="approved-source-wrong-version",
        ),
    ],
)
def test_lock_sources_reject_local_set_mismatch(tmp_path: Path, packages: list[tuple[str, str, str]]) -> None:
    with pytest.raises(ValueError, match="local sources differ"):
        validate_lock_sources(_write_lock(tmp_path, packages))


@pytest.mark.parametrize(
    ("name", "version", "path"),
    [
        ("deep-researcher-agent", "2.2.0", ".."),
        ("deep-researcher-mcp-server", "0.1.0", "./"),
    ],
)
def test_lock_sources_do_not_normalize_editable_paths(tmp_path: Path, name: str, version: str, path: str) -> None:
    # Characterization: the contract compares editable strings literally, so `..`
    # is not treated as `../` and `./` is not treated as `.`. This pins the
    # intentional exact-match behavior; tolerant path normalization would be a
    # deliberate change to validate_lock_sources, not an accident.
    packages = _without(name) + [(name, version, f'{{ editable = "{path}" }}')]
    with pytest.raises(ValueError, match="local sources differ"):
        validate_lock_sources(_write_lock(tmp_path, packages))


def test_lock_sources_canonicalize_dependency_names(tmp_path: Path) -> None:
    # uv may emit either dashed or underscored project names; the approved
    # contract is keyed on canonical (dashed) names, so an underscored spelling
    # of an approved source must still satisfy the contract.
    packages = _without("tavily-web-search") + [
        ("tavily_web_search", "1.0.0", '{ editable = "../sources/tavily_web_search" }')
    ]
    validate_lock_sources(_write_lock(tmp_path, packages))


def test_committed_mcp_lock_satisfies_source_contract() -> None:
    # Regression guard: the real, committed MCP lock must always satisfy the
    # public-source contract. Fails fast at unit-test time instead of only in
    # the full CI license-inventory job.
    validate_lock_sources(_MCP_LOCK_PATH)
