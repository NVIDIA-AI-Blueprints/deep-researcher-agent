---
name: deep-researcher-release-qa
description: Use when validating an Deep Researcher Agent change before opening or merging a PR — choosing and running the right Python, frontend, docs, or eval checks for the surfaces you touched instead of one fixed command list.
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags: "deep-researcher nemo-agent-toolkit validation testing release-qa"
allowed-tools: Read Bash
---

# Deep Researcher Agent Release QA

Use this skill to validate a change in the Deep Researcher Agent repository before opening or
merging a pull request. The goal is to run the **narrowest set of checks that
covers what you touched**, then broaden only when a change crosses shared
boundaries — not to run every command every time.

## Start Here

- Identify which surfaces the change touches: backend Python (`src/`,
  `sources/`, `tests/`), web UI (`frontends/ui/`), docs (`docs/`), or evals
  (`frontends/benchmarks/`).
- Run the scoped checks for those surfaces first (see the matrix below).
- Broaden to the full suite only when the change crosses shared boundaries
  (for example editing `src/deep_researcher_agent/common/` or a config many agents load).
- Capture the exact commands and their output — `deep-researcher-prepare-pr` requires this
  as validation evidence.
- Never paste secrets or `deploy/.env` values into command output you share.

## Authoritative References

- [AGENTS.md](../../../AGENTS.md): the "Build, test, and validation commands"
  section is the source of truth for every command below.
- [CONTRIBUTING.md](../../../CONTRIBUTING.md): "Local Validation" — the exact
  commands to run and the requirement to include their output in the PR.
- `pyproject.toml` and `uv.lock`: root Ruff config (line length 120, rule sets
  `E,F,W,I,PL,UP`) and the root dev environment.
- `mcp/pyproject.toml` and `mcp/uv.lock`: the independent MCP runtime and dev
  environment. MCP is not a root dependency group.
- `frontends/ui/package.json`: the real `scripts` (`lint`, `type-check`,
  `test:ci`, `build`) — use these names, do not invent npm scripts.

For the full per-surface command list and expected results:

- [references/validation-matrix.md](references/validation-matrix.md)

## Workflow

1. List the changed paths (`git status`, `git diff --name-only`) and map them to
   surfaces.
2. Run the scoped Python, UI, docs, or eval checks for those surfaces.
3. If the change touches shared code or config, broaden to the full suite for
   that surface.
4. Re-run until lint, format, type, and tests all pass; fix failures rather than
   skipping checks.
5. Record the exact commands and their output for the PR description.

## Validation

Pick the block that matches the change. Run the narrowest first.

Backend Python (run from the repo root; the project uses `uv`):

```bash
uv run ruff check <changed paths>          # lint
uv run ruff format --check <changed paths> # format check
uv run pytest <scoped test paths>          # tests
```

MCP server (run from the repo root):

```bash
uv sync --project mcp --extra dev
uv run ruff check mcp
uv run ruff format --check mcp
uv run --project mcp --extra dev pytest mcp/tests
```

Broaden to the whole tree when the change crosses shared boundaries:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run --project mcp --extra dev pytest mcp/tests
```

Frontend (from `frontends/ui/`):

```bash
npm run lint
npm run type-check
npm run test:ci
```

Expected: Ruff reports no lint or format failures, tests pass, and the frontend
lint/type-check/test commands exit cleanly. For docs and eval changes, see the
matrix reference.

## Common Mistakes

- Running both complete root and MCP test suites for a one-package change
  instead of scoping to the touched paths first — slow, and it buries the
  relevant signal.
- Running MCP tests in the root environment; use `uv run --project mcp` so the
  check consumes the isolated MCP lock and dev extra.
- Skipping `ruff format --check` and pushing unformatted code that fails CI.
- Hand-reformatting unrelated code; only the changed code should move.
- Forgetting that `nat eval` needs `deploy/.env`; run evals via
  `dotenv -f deploy/.env run nat eval ...` (see the matrix reference).
- Inventing npm scripts; stick to the canonical checks `lint`, `type-check`,
  `test:ci`, and `build` defined in `frontends/ui/package.json`.

## Related Skills

- `deep-researcher-prepare-pr`
- `deep-researcher-add-tool`
- `deep-researcher-add-data-source`
