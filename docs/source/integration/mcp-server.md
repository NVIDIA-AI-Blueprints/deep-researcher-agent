<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Expose Deep Researcher Agent as an MCP Server

Deep Researcher Agent includes a standalone [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes the
research workflow to MCP clients over stateless Streamable HTTP. The server is a public FastMCP application; it
does not depend on a private transport SDK or the Deep Researcher Agent REST API.

The default endpoint is `http://localhost:9001/mcp`. It advertises exactly three tools:

1. `submit_query(query)` classifies and starts the work.
2. `poll_query(job_id)` returns status and best-effort progress.
3. `get_final_report(job_id)` returns the completed answer or report.

## Requirements and public configuration

The shipped `configs/config_mcp.yml` uses hosted NVIDIA NIM models and Tavily search. Both credentials are
required for this MCP configuration:

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | Hosted NIM inference |
| `TAVILY_API_KEY` | Public web search |
| `DEEP_RESEARCHER_CHECKPOINT_DB` | Shared PostgreSQL database for MCP jobs and LangGraph checkpoints |

The config intentionally excludes the Deep Researcher Agent API frontend, authentication providers, enterprise data sources,
paper search, and the second async-job layer. FastMCP owns the transport and the submit/poll lifecycle.

### Dependency isolation

MCP is a separate uv project with its own `mcp/pyproject.toml`, `mcp/uv.lock`, and environment. The root Deep Researcher Agent
workspace excludes it and keeps `cryptography>=46.0.6,<47`, which is compatible with NAT's declared requirements.
The frozen MCP release/container profile instead pins `cryptography==50.0.0` to replace vulnerable earlier
releases.

The release-supported platform is Linux x86_64 with CPython 3.13, using either the frozen MCP project in an Deep Researcher Agent
source checkout or the release container built from the repository root. CI validates the production environment
and container on that platform. Other 64-bit source hosts are development-only. `cryptography` 50 no longer ships
x86_64 macOS or 32-bit Windows wheels. Those platforms are unsupported by the frozen profile; run the Linux
release container on a supported 64-bit Linux/container host. See the
[MCP security policy](../../../mcp/SECURITY.md#platform-compatibility) for the validation required before claiming
another target.

Here, *standalone* describes the MCP process and transport boundary, not a generic Python wheel.
`deep-researcher-mcp-server` depends on `deep-researcher-agent`, `tavily-web-search`, and other packages supplied by this repository; that
complete dependency closure is not published to a Python package index. Its locally buildable wheel is an internal
implementation artifact and is marked `Private :: Do Not Upload`.

The 50.0.0 override is security hardening for those supported runtime paths, not a functional MCP protocol
requirement or published package constraint. Use `mcp/uv.lock` through the commands below, or use the release
container, to retain the audited dependency profile.

For a local source checkout, start PostgreSQL and then run:

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"  # pragma: allowlist secret
export TAVILY_API_KEY="your-tavily-api-key"  # pragma: allowlist secret
export DEEP_RESEARCHER_CHECKPOINT_DB="postgresql://deep_researcher:local_mcp_password@127.0.0.1:1234/deep_researcher_jobs"  # pragma: allowlist secret
uv sync --project mcp --frozen
uv run --project mcp --frozen deep-researcher-mcp-server
```

The component accepts these runtime settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEP_RESEARCHER_MCP_HOST` | `0.0.0.0` | Uvicorn bind host |
| `DEEP_RESEARCHER_MCP_PORT` | `9001` | Uvicorn bind port |
| `DEEP_RESEARCHER_MCP_PATH` | `/mcp` | Streamable HTTP path |
| `DEEP_RESEARCHER_MCP_WORKERS` | `1` | Independent workflow-owning worker processes |
| `DEEP_RESEARCHER_MCP_LOG_LEVEL` | `INFO` | Python and Uvicorn log level |
| `DEEP_RESEARCHER_MCP_CONFIG` | `configs/config_mcp.yml` | NAT workflow configuration |
| `DEEP_RESEARCHER_MCP_ENV_FILE` | `deploy/.env` | Optional dotenv file; process variables take precedence |
| `DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS` | `30` | Shallow-query inline wait window |
| `DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS` | `8000` | Maximum accepted `submit_query` query length in characters |
| `DEEP_RESEARCHER_MCP_CORS_ORIGINS` | `http://localhost:6274` | Browser CORS allowlist; an empty value disables CORS |
| `DEEP_RESEARCHER_MCP_ALLOWED_HOSTS` | Local hosts | Valid HTTP `Host` values |
| `DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS` | Local HTTP origins | Valid browser `Origin` values |

## Component and lifecycle

The repository directory is `mcp/`, the distribution is `deep-researcher-mcp-server`, and the Python import package is
`deep_researcher_mcp`. This avoids shadowing the third-party `mcp` package.

| Path | Responsibility |
|------|----------------|
| `mcp/src/deep_researcher_mcp/server.py` | FastMCP tools, Starlette routes, process lifecycle, and Uvicorn launcher |
| `mcp/src/deep_researcher_mcp/workflow_runner.py` | Long-lived NAT workflow and sessions |
| `mcp/src/deep_researcher_mcp/jobs.py` | Classification, background work, state rendering, polling, and cleanup |
| `mcp/src/deep_researcher_mcp/job_store.py` | Shared PostgreSQL job ledger |
| `mcp/src/deep_researcher_mcp/checkpoint_todos.py` | Best-effort LangGraph todo progress |
| `mcp/deploy/init-mcp-db.sql` | Idempotent job-ledger bootstrap |

An outer Starlette lifespan starts the NAT workflow, job manager, and MCP session manager once per Uvicorn worker.
Stateless MCP requests do not own those resources, so closing the request that submitted a query does not cancel
its background task. Multiple workers share jobs and checkpoints through PostgreSQL.

## Tool protocol

Tool results use the field `state`. The HTTP health routes use `status`; these fields are deliberately different.
All `job_id` values below are canonical lowercase UUIDv4 strings.

### Submit

Meta queries complete synchronously:

```json
{
  "job_id": "<uuid>",
  "depth": "meta",
  "state": "complete",
  "result": "<answer>"
}
```

A shallow query starts with a 10-second estimate and a 5-second first-poll hint. The server waits for the
configured inline window. If work finishes, it returns `state: "complete"` with `result`, or `state: "failed"`
with `error`. If the inline window expires, the task remains alive and the response is queued:

```json
{
  "job_id": "<uuid>",
  "depth": "shallow",
  "state": "queued",
  "estimated_duration_seconds": 0,
  "first_poll_after_seconds": 0
}
```

The zero values above are the result with the default 30-second inline wait. For a custom wait, the remaining
estimate is `max(0, int(10 - inline_wait_seconds))`.

If the inline wait encounters an infrastructure error after the shallow job has been enqueued, `submit_query`
returns the original queued payload instead. The estimate and first-poll hint are left unchanged so the caller
retains the `job_id` capability and can continue with `poll_query`.

Deep work never waits inline and always returns:

```json
{
  "job_id": "<uuid>",
  "depth": "deep",
  "state": "queued",
  "estimated_duration_seconds": 180,
  "first_poll_after_seconds": 180
}
```

### Poll

`poll_query` is status-only and never includes the final `result`. Queued and running jobs include a cadence and
`todos`; shallow cadence is 3 seconds and deep cadence is 180 seconds:

```json
{
  "job_id": "<uuid>",
  "depth": "deep",
  "state": "running",
  "next_poll_after_seconds": 180,
  "todos": [
    {"content": "Gather sources", "status": "in_progress", "id": "optional-id"}
  ]
}
```

`todos` is always present for a found job. An empty list is normal for queued, shallow, early, or unavailable
progress. Valid normalized statuses are `pending`, `in_progress`, and `completed`; an upstream status that cannot
be normalized is preserved.

When polling returns `state: "complete"`, stop polling and call `get_final_report`. A completed poll response has
`job_id`, `depth`, `state`, and `todos`, but no `result` or cadence. A failed response has the same fields plus
`error`, and no cadence.

### Final report and stable errors

A completed report is:

```json
{
  "job_id": "<uuid>",
  "depth": "deep",
  "state": "complete",
  "result": "<report>"
}
```

Calling `get_final_report` while a job is queued or running returns:

```json
{
  "job_id": "<uuid>",
  "depth": "deep",
  "state": "not_ready",
  "error": "job_not_ready"
}
```

Unknown, malformed, noncanonical, expired, or inaccessible capabilities return the same non-enumerating shape:

```json
{"state": "not_found", "error": "job_not_found"}
```

A failed final report returns `job_id`, `depth`, `state: "failed"`, and a sanitized `error`. Server exception
details are not exposed in logs or responses, and capability UUIDs are not written to logs.

Unexpected infrastructure exceptions during submission before a capability is returned, polling, or final-report
retrieval are returned as MCP tool errors with stable public messages. Internal exception text is never serialized;
the MCP boundary log entry records only the affected operation and exception type.

## Health and transport contracts

| Request | Response |
|---------|----------|
| `GET /live` | HTTP 200 and `{"status":"alive"}` whenever the process can serve requests |
| `GET /health` before startup completes | HTTP 503 and `{"status":"not_ready"}` |
| `GET /health` after workflow and job startup | HTTP 200 and `{"status":"ready"}` |
| `GET /mcp` | HTTP 405 with `Allow: POST` |
| MCP Streamable HTTP request | `POST /mcp` |

The MCP transport is stateless and uses JSON responses. A supported client initializes a new connection, lists
the tools, submits work, and may close completely before another connection polls the same UUID.

## Anonymous capability security

The server does not require an `Authorization` header. Every call uses the constant database principal
`anonymous`; the random job UUID is therefore a bearer capability. Anyone who obtains it can poll or retrieve the
job until its database row expires. Do not put job IDs in URLs, logs, analytics, or support messages.

`DEEP_RESEARCHER_MCP_ALLOWED_HOSTS` and `DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS` protect browser-reachable deployments from DNS rebinding and
untrusted web origins. CORS controls which browser applications may read responses. None of these settings
authenticate a user or create per-user authorization. Headless clients normally omit `Origin` but must send an
allowed `Host`.

An unauthenticated MCP endpoint can consume model/search quota. The server rejects `submit_query` requests whose
query exceeds `DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS` characters before any job is enqueued, but it does not rate-limit callers.
Do not expose it directly to an untrusted network. Use network policy, an authenticated reverse proxy or gateway,
request and rate limits, and deployment monitoring when running beyond a trusted local environment.

## Container deployment

Build and start the loopback-only MCP/PostgreSQL stack from the repository root:

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"  # pragma: allowlist secret
export TAVILY_API_KEY="your-tavily-api-key"  # pragma: allowlist secret
docker compose -f deploy/compose/docker-compose.mcp.yaml up --detach --build --wait
```

The MCP endpoint is `http://127.0.0.1:9001/mcp`; PostgreSQL is published at `127.0.0.1:1234`. The local stack uses
a fixed development-only database password. Production should inject `DEEP_RESEARCHER_CHECKPOINT_DB` from secret management
and should not reuse the local Compose database credentials.

The image contains no private package indexes, private deployment metadata, bundled certificate authority, or
in-process TLS configuration. Terminate HTTPS at a reverse proxy or platform ingress.

Run the supported-client smoke test:

```bash
uv run --project mcp --frozen python mcp/scripts/protocol_smoke.py --url http://127.0.0.1:9001/mcp
```

The smoke client rejects URLs containing userinfo, query parameters, or fragments before making a request and does
not echo the rejected URL.

Stop the local stack with:

```bash
docker compose -f deploy/compose/docker-compose.mcp.yaml down --volumes
```

## Public design differences

The behavioral contract is frozen in `mcp/REFERENCE_PARITY.md`. The intentional public differences are:

- FastMCP and an outer Starlette lifespan provide the transport and worker lifecycle.
- Authentication, token propagation, private feedback hooks, and registry metadata are not included.
- Public `DEEP_RESEARCHER_MCP_*`, `NVIDIA_API_KEY`, `TAVILY_API_KEY`, and `DEEP_RESEARCHER_CHECKPOINT_DB` settings replace private startup
  inputs.
- The default workflow uses public NIM and Tavily only; enterprise sources, private prompts/indexes, and paper
  search are not part of this configuration.
- Bundled CA material and in-process certificate behavior are excluded.
- Exact lowercase canonical UUID validation and capability-safe logging harden public error behavior.

The three tool schemas, state machine, polling cadence, PostgreSQL schema, background lifecycle, todo normalization,
and client-safe error shapes otherwise remain compatible.
