<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

This is the canonical template for Deep Researcher Agent maintainer skills. To create a skill:

  1. Copy the YAML frontmatter and body below into
     .agents/skills/<deep-researcher-skill-name>/SKILL.md
  2. Replace every <PLACEHOLDER> and the example content.
  3. Keep SKILL.md concise; move long procedures into references/*.md.
  4. Validate: uv run python scripts/validate_skills.py .agents/skills

This file lives at the directory root (not inside a skill folder), so the
validator does not treat it as a skill. Do not add a SKILL.md beside it.
-->

# Skill authoring template

Copy everything between the rulers below into your new `SKILL.md`.

---

```markdown
---
name: deep-researcher-<skill-name>
description: Use when <specific trigger>. Keep this under 1024 characters and specific enough for an agent to route on.
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags: "deep-researcher nemo-agent-toolkit <area>"
allowed-tools: Read Bash Edit
---

# <Human-Readable Skill Title>

One or two sentences on what this skill helps a coding agent accomplish in the
Deep Researcher Agent repository, and when it applies.

## Start Here

- Confirm the requested change type before editing.
- Read the authoritative Deep Researcher Agent files listed below.
- Preserve existing repo patterns; prefer the smallest change that fits.
- Do not print secrets or hard-code tokens.

## Authoritative References

List the Deep Researcher Agent files and docs that are the source of truth for this workflow.
Point at precise paths, not whole trees.

- `docs/source/<area>/<page>.md`: canonical walkthrough.
- `<path/to/existing/example>`: existing pattern to follow.

For longer procedures, link to local reference files (loaded only when needed):

- [references/<topic>.md](references/<topic>.md)

## Workflow

1. Inspect the current implementation.
2. Choose the smallest change that fits the existing pattern.
3. Make the code or config change.
4. Add or update focused tests.
5. Run the validation commands below.
6. Summarize changed files and evidence.

## Validation

Run the narrowest command first, then broaden if the change crosses shared
boundaries. State the expected result.

```bash
uv run pytest <path/to/scoped/tests>
uv run ruff check <path/to/changed/code>
uv run ruff format --check <path/to/changed/code>
```

Expected: tests pass and Ruff reports no lint or format failures for the changed
code.

## Common Mistakes

- <A specific, high-frequency mistake for this workflow and how to avoid it.>
- <Another concrete pitfall.>

## Related Skills

- `deep-researcher-release-qa`
- `deep-researcher-prepare-pr`
```

---

## Authoring rules

- Embed the essential workflow in the skill; link to docs for extra context, but
  do not require the agent to discover the entire docs tree.
- Prefer checklists and exact commands over prose.
- Keep references local to the skill bundle when the content is needed to
  complete the task.
- Avoid volatile model names in examples unless the skill tells the agent to
  verify the current config.
- Never include secrets, real tokens, or environment-specific hostnames.
- Keep scripts small, deterministic, and dependency-light.
