# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Repo-local agent skills under .agents/skills must pass the skill validator.

This guards the maintainer skill set in CI's pytest job, in addition to the
pre-commit hook, since the skills-eval workflow is scoped to skills/ only.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
VALIDATOR = REPO_ROOT / "scripts" / "validate_skills.py"


def _load_validator():
    """Dynamically import scripts/validate_skills.py and return the module."""
    spec = importlib.util.spec_from_file_location("validate_skills", VALIDATOR)
    if spec is None:
        raise RuntimeError(f"failed to create import spec for {VALIDATOR}")
    if spec.loader is None:
        raise RuntimeError(f"import spec loader missing for {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve annotations under
    # `from __future__ import annotations` during dynamic import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_skills_are_valid():
    """Assert every skill bundle under .agents/skills passes structural validation."""
    validator = _load_validator()
    report = validator.validate_roots([SKILLS_ROOT])
    assert report.skills_checked >= 1, f"no skills found under {SKILLS_ROOT}"
    assert report.errors == [], "skill validation errors:\n" + "\n".join(report.errors)


def _make_bundle(root: Path, name: str, *, body: str = "", newline: str = "\n") -> Path:
    """Create a minimal, otherwise-valid skill bundle under root and return its directory."""
    bundle = root / name
    bundle.mkdir(parents=True)
    skill_md = (
        f"---{newline}"
        f"name: {name}{newline}"
        f"description: Temporary skill bundle for validator tests.{newline}"
        f"---{newline}{newline}"
        f"{body}{newline}"
    )
    # newline="" keeps our explicit line endings so the CRLF case stays CRLF on disk.
    (bundle / "SKILL.md").write_text(skill_md, encoding="utf-8", newline="")
    return bundle


def test_validator_flags_bundle_link_escape(tmp_path):
    """A relative bundle link that traverses out of its bundle directory is an error."""
    validator = _load_validator()
    root = tmp_path / "skills"
    _make_bundle(root, "deep-researcher-escape-skill", body="See [oops](references/../../secret.md).")
    report = validator.validate_roots([root])
    assert any("escapes the" in e for e in report.errors), report.errors


def test_validator_flags_unreadable_skill_md(tmp_path):
    """An unreadable SKILL.md is reported rather than crashing the validator."""
    validator = _load_validator()
    root = tmp_path / "skills"
    skill_md = _make_bundle(root, "deep-researcher-unreadable-skill") / "SKILL.md"
    skill_md.chmod(0o000)
    if os.access(skill_md, os.R_OK):  # e.g. running as root, where chmod can't block reads
        skill_md.chmod(0o644)
        pytest.skip("cannot make file unreadable as the current user")
    try:
        report = validator.validate_roots([root])
    finally:
        skill_md.chmod(0o644)
    assert any("Unable to read SKILL.md" in e for e in report.errors), report.errors


def test_validator_accepts_crlf_frontmatter(tmp_path):
    """CRLF line endings in frontmatter are valid; the regex matches `\\r?\\n` by design."""
    validator = _load_validator()
    root = tmp_path / "skills"
    _make_bundle(root, "deep-researcher-crlf-skill", body="# body", newline="\r\n")
    report = validator.validate_roots([root])
    assert report.errors == [], report.errors
