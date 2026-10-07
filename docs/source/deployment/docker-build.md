<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Docker Build System

The Deep Researcher Agent blueprint uses a multi-stage Dockerfile (`deploy/Dockerfile`) that produces two build targets: a development image with the CLI and a lean release image for production.

## Multi-Stage Architecture

```mermaid
graph TD
    A["nvcr.io/nvidia/base/ubuntu:noble-20260217"] -->|Stage 1| B["builder"]
    B -->|Install CLI + debug UI| C["dev-builder"]
    C -->|Copy /app| D["dev"]
    B -->|Copy /app| E["release"]

    F["nvcr.io/nvidia/distroless/python:3.13-v4.0.8"] -->|Runtime base| D
    F -->|Runtime base| E

    style A fill:#e0e0e0,stroke:#666
    style F fill:#e0e0e0,stroke:#666
    style B fill:#fff3cd,stroke:#856404
    style C fill:#fff3cd,stroke:#856404
    style D fill:#d1ecf1,stroke:#0c5460
    style E fill:#d4edda,stroke:#155724
```

The build consists of four stages:

| Stage | Base image | Purpose |
|-------|-----------|---------|
| `builder` | `nvcr.io/nvidia/base/ubuntu:noble-20260217` | Installs Python 3.13.14, system dependencies, and all application packages. |
| `dev-builder` | `builder` | Extends builder with the CLI and debug UI packages. |
| `dev` | `nvcr.io/nvidia/distroless/python:3.13-v4.0.8` | Development runtime -- copies from `dev-builder`. |
| `release` | `nvcr.io/nvidia/distroless/python:3.13-v4.0.8` | Production runtime -- copies from `builder` (no CLI). |

## Builder Stage

The builder stage handles all compilation and package installation:

1. **System dependencies** -- Installs build tools, curl, git, and the system Python used to bootstrap `uv`; `uv` then installs the pinned Python 3.13.14 runtime.
2. **Virtual environment** -- Creates a venv at `/app/.venv` using `uv` and Python 3.13.14.
3. **Dependency installation** -- Runs `uv sync --frozen --extra pii --extra s3` against the root workspace's `uv.lock`, installing locked runtime dependencies, the `pii` and `s3` extras, and the default `dev` dependency group. The independent `mcp/` project is excluded from the root workspace, so the standalone MCP package and `mcp/uv.lock` are not part of this image.
4. **Workspace packages** -- Installs application packages with `uv pip install -e` (the root package uses `--no-deps`):
   - Root workspace package (`deep-researcher-agent`) using `uv pip install --no-sources --no-deps -e .`; `--no-sources` keeps workspace source overrides from coupling this image to packages that are intentionally absent from its build context.
   - `sources/google_scholar_paper_search` -- Google Scholar search
   - `sources/tavily_web_search` -- Tavily web search
   - `sources/exa_web_search` -- Exa web search
   - `sources/nimble_web_search` -- Nimble web search
   - `sources/firecrawl_web_search` -- Firecrawl web search
   - `sources/knowledge_layer[all]` -- Knowledge layer with all extras
   - `frontends/deep_researcher_api` -- [FastAPI](https://fastapi.tiangolo.com/) frontend
   - `psycopg[binary]>=3.0.0` -- PostgreSQL driver (psycopg v3, installed non-editable)
5. **File setup** -- Makes startup scripts executable, creates `/app/data`, and sets ownership to UID 1000.

Only runtime scripts (`deploy/entrypoint.py` and `deploy/start_web.py`) are copied from the `deploy/` directory. The full `deploy/` directory is excluded to avoid leaking `.env` files, Helm charts, compose files, or other development artifacts into the image.

## Standalone MCP Image

The MCP release has a separate multi-stage Dockerfile at `mcp/Dockerfile`. Build
it from the repository root so its local Deep Researcher Agent source dependencies are available:

```bash
docker build -t deep-researcher-mcp-server:local -f mcp/Dockerfile .
```

That builder syncs the frozen MCP project (`uv sync --project /app/mcp`) from
`mcp/uv.lock`, independently of the root image. The root image stays within
NAT's `cryptography<47` constraint; the audited MCP container profile pins
`cryptography==50.0.0` through an MCP-scoped uv override. The release-supported
platform is Linux x86_64 with CPython 3.13, which CI validates through the
frozen production environment and this container. Other 64-bit source hosts
are development-only; x86_64 macOS and 32-bit Windows are unsupported by the
`cryptography` 50 wheel matrix. Run the release container on a supported 64-bit
Linux/container host instead. See the
[MCP security policy](../../../mcp/SECURITY.md#platform-compatibility) for the
full contract. The repository-local dependency closure is not published as a
generic wheel.

## Dev Stage

The development image extends the builder with additional packages:

- `frontends/cli` -- Command-line interface (`nat` CLI commands)
- `frontends/debug` -- Debug UI for local development

### Build

```bash
docker build --target dev -t deep-researcher:dev -f deploy/Dockerfile .
```

### What Is Included

- All application packages plus CLI and debug UI.
- Python 3.13.14 application runtime on the NVIDIA distroless base image.
- Startup scripts (`entrypoint.py`, `start_web.py`).
- Runs as non-root user (UID 1000).

## Release Stage

The release image is built from the base `builder` stage (no CLI or debug packages):

### Build

```bash
docker build --target release -t deep-researcher:prod -f deploy/Dockerfile .
```

### What Is Included

- Application packages only (no CLI, no debug UI).
- `APP_ENV=production` set by default.
- Same non-root user and distroless base as the dev image.

## Build Commands Summary

| Target | Command | Use case |
|--------|---------|----------|
| `dev` | `docker build --target dev -t deep-researcher:dev -f deploy/Dockerfile .` | Local development, testing, CLI access |
| `release` | `docker build --target release -t deep-researcher:prod -f deploy/Dockerfile .` | Production deployment, CI/CD |

When using Docker Compose, the build target is controlled by the `BUILD_TARGET` variable:

```bash
# Dev build (default)
cd deploy/compose
docker compose --env-file ../.env -f docker-compose.yaml up -d --build

# Release build
BUILD_TARGET=release docker compose --env-file ../.env -f docker-compose.yaml up -d --build
```

## Base Images

| Image | Used in | Purpose |
|-------|---------|---------|
| `nvcr.io/nvidia/base/ubuntu:noble-20260217` | Builder stages | Full Ubuntu with package managers for compilation. |
| `nvcr.io/nvidia/distroless/python:3.13-v4.0.8` | Runtime stages (`dev`, `release`) | Minimal NVIDIA distroless image with Python 3.13.14. No shell, no package manager -- reduces attack surface. |

## Startup Scripts

The container entrypoint is `python /app/deploy/entrypoint.py`, which orchestrates the full startup sequence. There are two scripts involved:

### entrypoint.py -- [Dask](https://www.dask.org/) cluster launcher

`entrypoint.py` is the Docker `ENTRYPOINT`. It performs the following:

1. **Argument pass-through** -- If command-line arguments are provided, it `exec`s them directly (useful for running one-off commands in the container).
2. **Dask scheduler** -- Starts a `dask-scheduler` process on loopback at the
   configured port (default `8786`) with a loopback-only dashboard on port `8787`.
3. **Wait for scheduler** -- Polls the scheduler with a Dask `Client` for up to 30 attempts (1 second apart).
4. **Dask worker** -- Starts a `dask-worker` process connected to the scheduler,
   with its RPC and diagnostics listeners restricted to loopback.
5. **Environment variable** -- Sets `NAT_DASK_SCHEDULER_ADDRESS` so the web server can submit background jobs.
6. **Web server** -- Launches `start_web.py` as a subprocess.
7. **Signal handling** -- Installs SIGTERM/SIGINT handlers that gracefully shut down all three processes (web, worker, scheduler).

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | `/app/configs/config_web_default_llamaindex.yml` | Path to the NeMo Agent Toolkit workflow config. |
| `HOST` | `0.0.0.0` | Bind address for the web server. |
| `PORT` | `8000` | Bind port for the web server. |
| `DASK_SCHEDULER_PORT` | `8786` | Dask scheduler port. |
| `DASK_NWORKERS` | `1` | Number of Dask workers. |
| `DASK_NTHREADS` | `4` | Threads per Dask worker. |

### start_web.py -- FastAPI server (direct uvicorn)

`start_web.py` bypasses the standard `nat serve` command to avoid an asyncio event loop conflict between the NeMo Agent Toolkit runtime and FastAPI/Starlette's anyio event loop management.

It performs the following:

1. **Configure logging** -- Sets up structured logging matching `nat serve` behavior.
2. **Load config** -- Validates the NeMo Agent Toolkit YAML config using `nat.runtime.loader.load_config()`.
3. **Set environment** -- Writes `NAT_CONFIG_FILE` and `NAT_FRONT_END_WORKER` so NeMo Agent Toolkit's FastAPI app can find the config and worker class.
4. **Run uvicorn** -- Starts uvicorn directly with `loop="asyncio"`, letting uvicorn create and manage its own event loop.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | `/app/configs/config_web_frag.yml` | Path to the NeMo Agent Toolkit workflow config. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8000` | Bind port. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
