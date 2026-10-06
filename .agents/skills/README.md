<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Researcher Agent Maintainer Skills

This directory is the maintainer/developer skill set for working **on** the Deep Researcher Agent
repository. These are repo-local Agent Skills: portable, task-scoped guidance
that a coding agent (Claude Code, Codex, Cursor, OpenCode, or any Agent
Skills-compatible tool) loads while a developer adds data sources, adds tools,
runs release QA, or prepares a PR.

They are **not** an in-product skill runtime. The user-facing Deep Researcher Agent application
remains a research blueprint built on the NeMo Agent Toolkit; nothing here is
loaded or executed by the deployed product. These skills exist only to help
coding agents and contributors work in this repository.

## Maintainer skills vs. API-consumer skills

Deep Researcher Agent has two distinct kinds of skill, separated by audience:

| | Maintainer skills | API-consumer skills |
| :-- | :-- | :-- |
| **Audience** | Developers changing the Deep Researcher Agent repo | Users calling a running Deep Researcher Agent server |
| **Location** | `.agents/skills/` (this directory) | top-level `skills/` |
| **Examples** | `deep-researcher-add-data-source`, `deep-researcher-add-tool`, `deep-researcher-configure-workflow`, `deep-researcher-release-qa`, `deep-researcher-prepare-pr`, `deep-researcher-customize-prompts-models`, `deep-researcher-maintain-ci` | `deep-researcher-deploy`, `deep-researcher-research` |
| **Assumes** | A repo checkout and dev toolchain | A reachable Deep Researcher Agent backend |

Consumer skills under `skills/` are authored to be self-contained and exportable
(for example, to the NVIDIA Skills catalog). They are surfaced to in-repo coding
agents through symlinks; do not move maintainer skills there.

## Layout

Each skill is a directory using the Agent Skills convention:

```text
.agents/skills/<skill-name>/
  SKILL.md                 # required: frontmatter + concise routed workflow
  references/              # optional: longer procedures, loaded only when needed
  scripts/                 # optional: deterministic, dependency-light helpers
  templates/               # optional: starter files or snippets
  assets/                  # optional: images, binaries, or other bundled assets
```

Keep `SKILL.md` concise and route long material into `references/`. See
[TEMPLATE.md](TEMPLATE.md) for the canonical starting point.

## Naming and frontmatter rules

These are enforced by [`scripts/validate_skills.py`](../../scripts/validate_skills.py):

- The directory name and the frontmatter `name` must match.
- `name` is lowercase, hyphen-separated, and prefixed with `deep-researcher-`.
- `description` is required and must stay under 1024 characters (the Agent Skills
  matching limit). Make it specific enough to route on.
- Links into a skill's own `references/`, `scripts/`, `templates/`, or `assets/`
  must resolve on disk.

Keep names stable once a skill is published or mirrored externally.

## Adding a skill

1. Copy [TEMPLATE.md](TEMPLATE.md) to `.agents/skills/<deep-researcher-skill-name>/SKILL.md`
   and fill in the frontmatter and sections.
2. Add a coding-agent compatibility symlink so the in-repo agent discovers it:

   ```bash
   ln -s ../../.agents/skills/<deep-researcher-skill-name> .claude/skills/<deep-researcher-skill-name>
   ```

3. Validate before committing:

   ```bash
   uv run python scripts/validate_skills.py .agents/skills
   ```

The validator also runs as a pre-commit hook and in CI's pytest job, so a
malformed skill fails fast.
