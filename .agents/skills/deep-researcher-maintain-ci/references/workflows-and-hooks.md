<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Workflows and hooks

Authoritative sources: the workflow files under `.github/workflows/` and
`CONTRIBUTING.md` "CI and Bot Workflow".

## Workflows

- `ci.yml` ("Deep Researcher Agent CI") — jobs: `pre-commit`, `test` (pytest with a coverage gate),
  `helm-lint` (deploy charts), `test-scripts`. The `pre-commit` job does **not**
  run the full hook set: it runs `ruff check .` / `ruff format --check .`
  separately and skips the push-stage pytest and Helm hooks when it runs
  `pre-commit run --all-files`, leaving those checks to their own jobs. The test
  job syncs the root Deep Researcher Agent environment from `uv.lock` and the independent MCP
  environment from `mcp/uv.lock`.
- `ui.yml` — jobs `install`, `lint`, `type-check`, `unit-test`, `build` for
  `frontends/ui/`.
- `skills-eval.yml` ("Skills Eval") — runs on `push` and `workflow_dispatch`. A
  `detect-changes` job path-gates the run; the trigger deliberately does not use a
  `paths:` filter (see the comment in the file). See the skill-eval-harness
  reference for the stages.
- `request-nvskills-ci.yml` — comment-triggered NVSkills CI request.

## The copy-pr-bot mirror flow

Per `CONTRIBUTING.md`: pushing a branch does not run CI. A maintainer or vetter
comments `/ok to test`; copy-pr-bot mirrors the PR to a `pull-request/<N>` branch,
and CI runs there. `/nvskills-ci` requests NVSkills validation; `/merge` requests
bot-driven merge once repository rules pass. The mirror behavior is configured in
`.github/copy-pr-bot.yaml`.

## Pre-commit hooks

`.pre-commit-config.yaml` is the source of truth. Default-stage hooks (run by a
plain `pre-commit run`) include: `ruff-check`, `ruff-format`, `uv-lock`,
`uv-lock-mcp`, `check-merge-conflict`, `check-added-large-files`, `check-yaml`,
`end-of-file-fixer`, `trailing-whitespace`, `detect-secrets`, `validate-skills`,
`clear-notebook-output-cells`, and `markdown-link-check`.

The root pytest, MCP pytest, and `helm-lint` hooks are `stages: [push]`, so a
default `pre-commit run` / `pre-commit run --all-files` **skips them**. Include
them explicitly with `pre-commit run --all-files --hook-stage push`. CI does not
run them via the `pre-commit` job either; they run as the dedicated `test` and
`helm-lint` jobs in `ci.yml`. The root pytest hook uses `uv.lock`; the MCP hook
invokes `uv run --project mcp --extra dev pytest mcp/tests` against
`mcp/uv.lock`.

## Validation

```bash
uv run pre-commit run --all-files                     # default-stage hooks
uv run pre-commit run --all-files --hook-stage push   # + pytest, helm-lint
actionlint .github/workflows/<file>.yml               # if installed
```

Expected: hooks pass (or only auto-fix), and any edited workflow is valid YAML
that `actionlint` accepts.
