#!/usr/bin/env python3
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
"""Validate Deep Researcher Agent repo-local agent skill bundles.

Checks every skill directory under one or more roots (default ``.agents/skills``)
for a well-formed ``SKILL.md`` and bundle layout. Runs fully offline with no
network access so it is safe in pre-commit and CI.

Usage:
    python scripts/validate_skills.py [ROOT ...]

Exit code is 0 when every skill is valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import yaml

# Skill name contract: lowercase, hyphen-separated, ``deep-researcher-`` prefixed.
NAME_PREFIX = "deep-researcher-"
NAME_RE = re.compile(r"^deep-researcher-[a-z0-9]+(?:-[a-z0-9]+)*$")

# Names that predate the contract and are allowed to keep their existing form.
GRANDFATHERED_NAMES: set[str] = set()

# Anthropic Agent Skills cap the matching ``description`` at 1024 characters.
MAX_DESCRIPTION_CHARS = 1024

# Only relative links pointing into these in-bundle subdirectories are checked
# for on-disk existence. Anchors, external URLs, and links into the wider repo
# are intentionally left to markdown-link-check.
BUNDLE_DIRS = ("references", "scripts", "templates", "assets")

FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

DEFAULT_ROOTS = (".agents/skills",)


@dataclass
class Report:
    """Accumulates per-skill validation errors."""

    errors: list[str] = field(default_factory=list)
    skills_checked: int = 0

    def fail(self, skill: str, message: str) -> None:
        self.errors.append(f"{skill}: {message}")


def _parse_frontmatter(text: str) -> dict | None:
    """Return the parsed YAML frontmatter mapping, or None if absent/invalid."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _check_bundle_links(skill_md: Path, text: str, dir_name: str, report: Report) -> None:
    """Verify relative links into references/scripts/templates/assets resolve."""
    for target in MD_LINK_RE.findall(text):
        target = target.strip()
        # Skip external schemes and pure anchors.
        if target.startswith(("http://", "https://", "mailto:", "#")) or "://" in target:
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        first_segment = path_part.split("/", 1)[0]
        if first_segment not in BUNDLE_DIRS:
            continue
        candidate = (skill_md.parent / path_part).resolve(strict=False)
        bundle_root = (skill_md.parent / first_segment).resolve(strict=False)
        if not candidate.is_relative_to(bundle_root):
            report.fail(dir_name, f"SKILL.md link '{path_part}' escapes the '{first_segment}' bundle directory")
        elif not candidate.exists():
            report.fail(dir_name, f"SKILL.md links to missing bundle file '{path_part}'")


def validate_skill(skill_dir: Path, report: Report) -> None:
    """Validate a single skill directory in place, recording any failures."""
    dir_name = skill_dir.name
    report.skills_checked += 1

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        report.fail(dir_name, "missing SKILL.md")
        return

    try:
        text = skill_md.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as err:
        report.fail(dir_name, f"Unable to read SKILL.md: {err}")
        return

    frontmatter = _parse_frontmatter(text)
    if frontmatter is None:
        report.fail(dir_name, "SKILL.md frontmatter is missing or not valid YAML")
        return

    name = frontmatter.get("name")
    if not name or not isinstance(name, str):
        report.fail(dir_name, "frontmatter 'name' is missing")
    else:
        if name != dir_name:
            report.fail(dir_name, f"frontmatter name '{name}' does not match directory '{dir_name}'")
        if name not in GRANDFATHERED_NAMES and not NAME_RE.match(name):
            report.fail(dir_name, f"name '{name}' must be lowercase-hyphenated and start with '{NAME_PREFIX}'")

    description = frontmatter.get("description")
    if not description or not isinstance(description, str) or not description.strip():
        report.fail(dir_name, "frontmatter 'description' is missing or empty")
    elif len(description) > MAX_DESCRIPTION_CHARS:
        report.fail(dir_name, f"description is {len(description)} chars (limit {MAX_DESCRIPTION_CHARS})")

    _check_bundle_links(skill_md, text, dir_name, report)


def iter_skill_dirs(root: Path) -> list[Path]:
    """Return sorted skill directories (symlinks followed) directly under root."""
    return sorted(p for p in root.iterdir() if p.is_dir())


def validate_roots(roots: list[Path]) -> Report:
    """Validate every skill under each root and return the combined report."""
    report = Report()
    for root in roots:
        if not root.is_dir():
            report.errors.append(f"{root}: skills root does not exist")
            continue
        for skill_dir in iter_skill_dirs(root):
            validate_skill(skill_dir, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse root arguments, run validation, and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Skill root directories to validate (default: .agents/skills).",
    )
    args = parser.parse_args(argv)

    report = validate_roots([Path(r) for r in args.roots])

    if report.errors:
        print(f"Skill validation FAILED ({len(report.errors)} error(s)):", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Skill validation passed: {report.skills_checked} skill(s) OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
