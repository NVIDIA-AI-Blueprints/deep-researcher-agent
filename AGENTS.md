<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Researcher Agent Guidance

Repository-global instructions for coding agents and for humans reviewing
agent-authored changes. These rules apply to every task in this repository.
Task-specific runbooks live in [`.agents/skills/`](.agents/skills/) — load the
relevant skill before starting a workflow it covers.

## Project overview

Deep Researcher Agent is an NVIDIA AI Blueprint: an enterprise research agent built on the
**NeMo Agent Toolkit (NAT)**. The deployed product is a research blueprint, not
a general skill runtime. New retrieval sources and tools are NAT functions;
agent behavior is driven by workflow YAML, Jinja2 prompts, and a data-source
registry — not by hard-coded logic.

Primary boundaries:

- Backend Python package: `src/deep_researcher_agent/`.
- Data-source and tool packages: `sources/` (each is its own package).
- Frontends and tooling: `frontends/` (web UI in `frontends/ui/`, eval harnesses
  in `frontends/benchmarks/`).
- Configs, deployment, docs: `configs/`, `deploy/`, `docs/`.

Stay inside this repository. If your workspace also contains adjacent repos
(for example a sibling NeMo-Relay checkout), do not edit them as part of an Deep Researcher Agent
change. Treat `sources/*` as independent packages: prefer the smallest change
scoped to the package you are touching.

## Repository structure

| Path | Purpose |
| :-- | :-- |
| `src/deep_researcher_agent/` | Backend agent, FastAPI extensions, auth, observability, knowledge |
| `sources/` | Data-source / tool packages (e.g. `tavily_web_search`, `google_scholar_paper_search`) |
| `configs/` | Workflow YAML configs (e.g. `config_cli_default.yml`) |
| `frontends/ui/` | Next.js / React / TypeScript / Tailwind / KUI web UI |
| `frontends/benchmarks/` | Eval harnesses: `freshqa`, `deepsearch_qa`, `deepresearch_bench` |
| `mcp/` | Independent MCP server uv project, lockfile, tests, and release assets |
| `deploy/` | Docker Compose and Helm/Kubernetes assets; `deploy/.env` for secrets |
| `docs/source/` | Sphinx documentation |
| `skills/` | API-consumer Agent Skills (`deep-researcher-deploy`, `deep-researcher-research`) |
| `.agents/skills/` | Maintainer Agent Skills (this contributor skill set) |
| `tests/` | Pytest suite |

## Build, test, and validation commands

Python (run from the repo root; the project uses `uv`):

```bash
./scripts/setup.sh                    # one-time environment setup
uv sync --group dev                   # root Deep Researcher Agent development environment
uv run ruff check .                   # lint
uv run ruff format --check .          # format check
uv run pytest                         # root and source-package tests
uv sync --project mcp --extra dev     # isolated MCP development environment
uv run --project mcp --extra dev pytest mcp/tests
```

Run the backend locally:

```bash
./scripts/start_cli.sh             # CLI mode (configs/config_cli_default.yml)
nat serve --config_file configs/config_cli_default.yml --port 8000
```

The backend API serves at `http://localhost:8000`.

Frontend (from `frontends/ui/`):

```bash
npm run lint
npm run type-check
npm run test:ci
```

Docs (from `docs/`): build with the provided `Makefile` (Sphinx).

Evals (from the repo root) use the NAT eval harness, for example:

```bash
dotenv -f deploy/.env run nat eval \
  --config_file frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml
```

Run the narrowest relevant command first; broaden to the full suite only when a
change crosses shared boundaries.

## Coding style and naming conventions

- Python is linted and formatted with **Ruff** (line length 120, target 3.11,
  rule sets `E,F,W,I,PL,UP`; isort `force-single-line`). Match the existing
  import and formatting style; do not hand-reformat unrelated code.
- New tools and data sources are NAT functions registered with
  `@register_function`; config schemas inherit from `FunctionBaseConfig`; YAML
  `_type` names come from the registered config class.
- Register new data sources in the `data_source_registry` so the UI can toggle
  them.
- `pre-commit` enforces the above plus secret detection, link checking, and
  skill validation. Install it and let it run before pushing.
- The root workspace excludes `mcp/`; maintain `uv.lock` and `mcp/uv.lock` as
  separate resolutions. The root uses the NAT-compatible
  `cryptography>=46.0.6,<47` range, while the frozen MCP release profile pins
  `cryptography==50.0.0`. Keep the 50.x override scoped to `mcp/`: it is uv
  lock policy and is not guaranteed by the published wheel metadata. The
  audited MCP profile is Linux x86_64 with CPython 3.13; other 64-bit source
  hosts are development-only. x86_64 macOS and 32-bit Windows are unsupported
  because `cryptography` 50 does not publish those wheels; run the release
  container on a supported 64-bit Linux/container host.

## Security and auth rules

- Never commit secrets, tokens, or environment-specific hostnames. Use
  environment variables and `SecretStr`; resolve API keys at runtime.
- Never print or log secret values, including in tool output or error messages.
- Missing-secret paths must degrade gracefully (stub/skip), not crash or leak.
- Respect authenticated data sources: honor `requires_auth`, per-user token
  pass-through, and backend token validators. Apply owner guardrails before
  loading protected report or artifact context into an agent.
- Do not weaken or bypass `AuthMiddleware`, validators, or auth gating without a
  prior design discussion.

## Frontend conventions

- The UI is Next.js / React / TypeScript / Tailwind with KUI components and an
  adapter-based structure under `frontends/ui/src/`. Reuse existing KUI
  components and visual patterns rather than introducing new ones.
- The backend is reached via the proxy and `BACKEND_URL`; preserve auth-aware UI
  states.
- Validate UI-affecting changes with `npm run lint`, `npm run type-check`, and
  `npm run test:ci`. Include a screenshot for visible changes.

## Documentation and design expectations

- Update the docs under `docs/source/` when behavior, configuration, or
  workflows change. Skills are a task-oriented layer above the docs — keep the
  docs canonical; do not duplicate full doc pages into skill text.
- For substantial behavior, auth, UI, or architecture changes, open a design
  discussion before coding rather than landing a large unreviewed change.

## Git and PR hygiene

- This repo uses a maintainer-reviewed PR workflow with **DCO sign-off**,
  code-owner review, copy-pr-bot mirroring, and GitHub Actions validation.
- Every commit must be signed off:

  ```bash
  git commit -s -m "Concise, scoped change"
  ```

  The commit must contain a `Signed-off-by: Your Name <your@email.com>` trailer.
  Commits without sign-off may be rejected.
- Keep PRs scoped: no unrelated files, no accidental generated artifacts, no
  secrets. Provide validation evidence (commands run and results).

## Skill usage

- Repository-global rules live in this `AGENTS.md`.
- Task-specific runbooks live in [`.agents/skills/`](.agents/skills/) (maintainer
  skills). See its [README](.agents/skills/README.md) and
  [TEMPLATE](.agents/skills/TEMPLATE.md) to use or add one.
- API-consumer skills for calling a running Deep Researcher Agent server live in `skills/`
  (`deep-researcher-research`, `deep-researcher-deploy`); they are a different audience and are not
  maintainer skills.
- These skills are guidance for coding agents working in this repository. They
  are not an in-product skill runtime and are not executed by the deployed
  Deep Researcher Agent application.
