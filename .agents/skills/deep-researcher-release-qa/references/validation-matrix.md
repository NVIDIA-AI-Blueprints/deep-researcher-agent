<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Researcher Agent validation matrix

Map the change to a surface, run that surface's checks, and broaden only when
the change crosses shared boundaries. All commands come from `AGENTS.md`
("Build, test, and validation commands") and `CONTRIBUTING.md` ("Local
Validation"); this file is a convenience index, not a new source of truth.

## One-time environment setup

```bash
./scripts/setup.sh                    # one-time environment setup
uv sync --group dev                   # root Deep Researcher Agent development environment
uv sync --project mcp --extra dev     # independent MCP development environment
```

## Backend Python — `src/`, `sources/`, `tests/`

Scoped first (replace the paths with what you changed):

```bash
uv run ruff check sources/my_package
uv run ruff format --check sources/my_package
uv run pytest sources/my_package/tests
```

A `sources/*` package usually needs an editable install before its tests run:

```bash
uv pip install -e ./sources/my_package
```

Broaden to the whole tree when the change touches shared code (for example
`src/deep_researcher_agent/common/`) or a config many agents load:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Expected: Ruff reports no lint or format failures for the changed code, and the
selected tests pass. Ruff config (line length 120; rule sets `E,F,W,I,PL,UP`)
lives in `pyproject.toml`.

## MCP server — `mcp/`

```bash
uv sync --project mcp --extra dev
uv run ruff check mcp
uv run ruff format --check mcp
uv run --project mcp --extra dev pytest mcp/tests
```

Set `DEEP_RESEARCHER_MCP_TEST_DB_URL` to a disposable PostgreSQL database to include the
database-backed job, checkpoint, restart, and NAT-load tests. Those tests create
and drop tables, so never point them at a service or production database.

## Web UI — `frontends/ui/`

```bash
cd frontends/ui
npm ci            # only when dependencies changed or node_modules is stale
npm run lint
npm run type-check
npm run test:ci
npm run build     # when the change could affect the production build
```

Expected: each command exits cleanly. Stick to the canonical checks `lint`,
`type-check`, `test:ci`, and `build` from `package.json`; do not invent scripts.
Include a screenshot for user-visible changes.

## Docs — `docs/`

```bash
cd docs
make html         # Sphinx build via the provided Makefile
```

Expected: the Sphinx build completes without errors. Update pages under
`docs/source/` whenever behavior, configuration, or workflows change.

## Evals — `frontends/benchmarks/`

Evals use the NAT eval harness and need the deployment env file `deploy/.env`
(created during deployment; never commit it):

```bash
dotenv -f deploy/.env run nat eval \
  --config_file frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml
```

Other harness configs:

- `frontends/benchmarks/freshqa/configs/config_full_workflow.yml`
- `frontends/benchmarks/freshqa/configs/config_shallow_research_only.yml`
- `frontends/benchmarks/deepsearch_qa/configs/config_deepsearch_qa.yml`

Expected: the eval run completes and writes its results. Evals call models and
data sources, so they need the relevant API keys present in `deploy/.env`.

## Running the backend for a manual check

```bash
./scripts/start_cli.sh        # CLI mode (configs/config_cli_default.yml)
nat serve --config_file configs/config_cli_default.yml --port 8000
```

The backend API serves at `http://localhost:8000`.
