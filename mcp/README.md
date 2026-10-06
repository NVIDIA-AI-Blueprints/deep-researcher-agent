# Deep Researcher Agent MCP Server

This independent uv project exposes the Deep Researcher Agent research workflow through the Model Context Protocol (MCP).

For the canonical user and deployment guide, including exact JSON contracts, see
[Expose Deep Researcher Agent as an MCP Server](../docs/source/integration/mcp-server.md).

The component uses public FastMCP with stateless Streamable HTTP and JSON responses. An outer Starlette lifespan
keeps the NAT workflow, MCP session manager, and background job manager alive once per Uvicorn worker so research
continues after the request that submitted it has returned.

The repository directory is named `mcp`, while the Python package is `deep_researcher_mcp` so it does not shadow the
third-party `mcp` package.

## Supported distribution paths

The release-supported runtime is Linux x86_64 with CPython 3.13, either through the release container built from
the repository root or directly from an Deep Researcher Agent source checkout with the frozen `mcp/uv.lock`. CI validates both the
frozen production environment and the release container on that platform. Other 64-bit source hosts are
development-only and do not carry the audited release guarantee. In particular, `cryptography` 50 no longer ships
x86_64 macOS or 32-bit Windows wheels. Those platforms are unsupported by the frozen profile; run the Linux
release container on a supported 64-bit Linux/container host.

In this documentation, *standalone* describes the MCP process and transport boundary; it does not mean that
`deep-researcher-mcp-server` is a separately installable Python wheel.

The MCP project depends on `deep-researcher-agent`, `tavily-web-search`, and other packages supplied from this repository. That
complete dependency closure is not published to a Python package index. The wheel that local build tooling may
produce is therefore an internal implementation artifact: do not publish it or install it as a generic wheel. The
project's `Private :: Do Not Upload` classifier makes this distribution boundary explicit.

## Component layout

- `src/deep_researcher_mcp/workflow_runner.py` owns the long-lived NAT workflow.
- `src/deep_researcher_mcp/jobs.py` manages asynchronous research jobs and polling.
- `src/deep_researcher_mcp/job_store.py` persists the shared job ledger in Postgres.
- `src/deep_researcher_mcp/checkpoint_todos.py` reads best-effort todo progress from LangGraph checkpoints.
- `src/deep_researcher_mcp/db_url.py` validates and normalizes Postgres URLs.
- `src/deep_researcher_mcp/server.py` exposes the FastMCP transport and process lifecycle.
- `REFERENCE_PARITY.md` freezes the reference behavior, executable evidence, and intentional public deviations.

## Runtime

The server binds to `0.0.0.0:9001` by default; local clients connect to `http://localhost:9001/mcp`. It exposes
exactly three tools:

- `submit_query(query)`
- `poll_query(job_id)`
- `get_final_report(job_id)`

No Authorization header is required. `GET /mcp` deliberately returns `405`; this stateless JSON server does not
open the optional standalone SSE stream. `GET /live` reports process liveness, while `GET /health` returns `200`
only after the workflow and job manager have started.

### Anonymous capability model

Every request uses the same database principal, `anonymous`. A returned job UUID is therefore an opaque bearer
capability rather than a per-user identifier: any caller possessing it can poll that job and retrieve its final
report until its expired database row is removed by periodic cleanup. Unknown, malformed, and cleanup-deleted UUIDs
return `not_found`. There is no per-user isolation. Treat job UUIDs as secrets—do not share them or place them in
logs, URLs, analytics, or support messages.

The public component intentionally does not consume `Authorization` headers and does not retain an actor-token
propagation seam. Authentication can be added later as a separate extension, but unverified headers must not be
interpreted as identity. The retained `principal` database column is always populated with `anonymous` to preserve
the reference schema without a migration.

The local MCP Inspector origin `http://localhost:6274` is allowed by default for POST requests without browser
credentials. Set `DEEP_RESEARCHER_MCP_CORS_ORIGINS` to a comma-separated allowlist, or to an empty value to disable browser
CORS. FastMCP Host and Origin validation is always enabled; deployments using public DNS names must add them to
`DEEP_RESEARCHER_MCP_ALLOWED_HOSTS` and `DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS`.

These allowlists are not authentication or per-user authorization and do not change the UUID capability model.
Host validation protects locally or internally
reachable MCP servers from DNS-rebinding attacks, while Origin validation rejects actual browser requests from
untrusted websites. CORS separately controls which browser applications may read responses and is not a substitute
for server-side Origin validation. Headless MCP clients normally omit `Origin`, so they only need to send an allowed
`Host`. This protection is especially useful for an unauthenticated server because it prevents an arbitrary webpage
from submitting Deep Researcher Agent jobs through a service reachable from the user's browser.

```bash
DEEP_RESEARCHER_CHECKPOINT_DB=postgresql://localhost/deep_researcher_jobs \
DEEP_RESEARCHER_MCP_CONFIG=/path/to/config.yml \
uv run --project mcp --frozen deep-researcher-mcp-server
```

Runtime settings use only public component names:

| Variable | Default | Purpose |
|---|---:|---|
| `DEEP_RESEARCHER_MCP_HOST` | `0.0.0.0` | Uvicorn bind host |
| `DEEP_RESEARCHER_MCP_PORT` | `9001` | Uvicorn bind port |
| `DEEP_RESEARCHER_MCP_PATH` | `/mcp` | Streamable HTTP endpoint |
| `DEEP_RESEARCHER_MCP_WORKERS` | `1` | Independent workflow-owning workers |
| `DEEP_RESEARCHER_MCP_LOG_LEVEL` | `INFO` | Python/Uvicorn log level |
| `DEEP_RESEARCHER_MCP_CONFIG` | `configs/config_mcp.yml` (source checkout) | NAT workflow configuration; set explicitly by the release image |
| `DEEP_RESEARCHER_MCP_ENV_FILE` | `deploy/.env` (source checkout) | Optional dotenv file; existing process variables take precedence |
| `DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS` | `30` | Shallow-query inline wait window |
| `DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS` | `8000` | Maximum submitted query length in characters |
| `DEEP_RESEARCHER_MCP_CORS_ORIGINS` | `http://localhost:6274` | Browser origin allowlist |
| `DEEP_RESEARCHER_MCP_ALLOWED_HOSTS` | local loopback/bind hosts | Valid HTTP Host headers; configure deployment DNS names |
| `DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS` | local HTTP origins | Valid MCP Origin headers; browser CORS origins are added automatically |
| `DEEP_RESEARCHER_CHECKPOINT_DB` | required | Shared Postgres DSN for checkpoints and jobs |

The `DEEP_RESEARCHER_MCP_CONFIG` and `DEEP_RESEARCHER_MCP_ENV_FILE` path defaults are available only in an Deep Researcher Agent source checkout. The
supported release image sets `DEEP_RESEARCHER_MCP_CONFIG` to its bundled config and does not load a dotenv file by default.

The default `configs/config_mcp.yml` uses only public Deep Researcher Agent plugins: hosted NVIDIA NIM inference through
`NVIDIA_API_KEY` and Tavily web search through `TAVILY_API_KEY`. It intentionally has no API front end,
enterprise source, authentication provider, or second asynchronous research layer; FastMCP owns the transport and
submit/poll lifecycle.

The exact state, timing, response, persistence, and todo contracts and all
public design decisions are recorded in [`REFERENCE_PARITY.md`](REFERENCE_PARITY.md).

## Development checks

The root Deep Researcher Agent workspace and MCP project use separate environments and lockfiles. Use uv 0.11.25 or newer; the
validated CI and container toolchain pins uv 0.11.26 because the MCP lock policy uses scoped dependency overrides.

```bash
uv sync --group dev
uv sync --project mcp --extra dev
uv run ruff check mcp
uv run ruff format --check mcp
uv run --project mcp --extra dev pytest mcp/tests
```

Set `DEEP_RESEARCHER_MCP_TEST_DB_URL` to a disposable Postgres database whose name ends in `_test` (for example,
`deep_researcher_mcp_test`) to enable the ledger and checkpoint integration tests.

The root `uv.lock` keeps `cryptography>=46.0.6,<47` for compatibility with NAT. The audited MCP release profile is
resolved independently from `mcp/uv.lock` and pins `cryptography==50.0.0` to harden the bundled OpenSSL version.
That pin uses three reviewed transitive compatibility exceptions; it is not a functional MCP requirement or
published package constraint. Its audited support matrix is Linux x86_64 with CPython 3.13; see
[`SECURITY.md`](SECURITY.md#platform-compatibility) for host-platform limitations.

## Container deployment

The release image is built entirely from public Python and Debian packages. It contains no private transport SDK,
private package index, bundled certificate authority, or private deployment metadata. TLS certificates are intentionally
out of scope: for HTTPS, terminate TLS at a reverse proxy or platform ingress in front of the container.

Export the two public API credentials and start the isolated MCP/Postgres stack from the repository root:

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"  # pragma: allowlist secret
export TAVILY_API_KEY="your-tavily-api-key"  # pragma: allowlist secret
docker compose -f deploy/compose/docker-compose.mcp.yaml up --detach --build --wait
```

The defaults publish MCP at `http://127.0.0.1:9001/mcp` and Postgres at `127.0.0.1:1234`. Both host bindings are
loopback-only. This local stack intentionally uses a fixed development-only database password so an arbitrary raw
password cannot be interpolated into a URL incorrectly. Production deployments should supply
`DEEP_RESEARCHER_CHECKPOINT_DB` through their secret-management platform rather than reuse this local Compose file. Compose
persists the database in a named volume and waits for Postgres before starting MCP. To use an env file for Compose
interpolation, add `--env-file /path/to/file`; only the variables explicitly listed in the Compose service are
passed into the MCP container.

Run the supported-client, no-authentication smoke test:

```bash
uv run --project mcp --frozen python mcp/scripts/protocol_smoke.py --url http://127.0.0.1:9001/mcp
```

The smoke client rejects URLs containing userinfo, query parameters, or fragments before making a request and does
not echo the rejected URL.

The check waits for readiness, initializes MCP, asserts that exactly the three documented tools are advertised,
and calls `poll_query` for an unknown UUID without sending `Authorization`. It must return
`{"state":"not_found","error":"job_not_found"}` as a successful tool result.

Useful lifecycle commands are:

```bash
docker compose -f deploy/compose/docker-compose.mcp.yaml ps
docker compose -f deploy/compose/docker-compose.mcp.yaml logs --no-color deep-researcher-mcp
docker compose -f deploy/compose/docker-compose.mcp.yaml down
# Also delete local job/checkpoint data when a completely fresh database is wanted:
docker compose -f deploy/compose/docker-compose.mcp.yaml down --volumes
```

Deployment overrides all use public names:

| Variable | Default | Purpose |
|---|---:|---|
| `DEEP_RESEARCHER_MCP_IMAGE` | `deep-researcher-mcp-server:local` | Built/tagged release image |
| `DEEP_RESEARCHER_MCP_PUBLISHED_PORT` | `9001` | Loopback host port for MCP |
| `DEEP_RESEARCHER_MCP_POSTGRES_PORT` | `1234` | Loopback host port for Postgres |
| `DEEP_RESEARCHER_MCP_PATH` | `/mcp` | Streamable HTTP endpoint path |
| `DEEP_RESEARCHER_MCP_WORKERS` | `1` | Uvicorn worker count |
| `DEEP_RESEARCHER_MCP_LOG_LEVEL` | `INFO` | Server log level |
| `DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS` | `30` | Shallow-query inline wait window |
| `DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS` | `8000` | Maximum submitted query length in characters |
| `DEEP_RESEARCHER_MCP_CORS_ORIGINS` | `http://localhost:6274` | Browser CORS allowlist; explicitly empty disables CORS |
| `DEEP_RESEARCHER_MCP_ALLOWED_HOSTS` | local and Compose hostnames | DNS-rebinding Host allowlist |
| `DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS` | local HTTP origins | Browser Origin validation allowlist |

Allowed hosts and origins are request-boundary protections, not authentication and not certificate settings. They
matter because an unauthenticated service reachable from a browser should not accept a rebinding Host or requests
from arbitrary websites. Headless MCP clients normally send no `Origin`; public deployments only need to add their
actual DNS hostname and browser origin. CORS can be disabled explicitly when no browser client is used. The
Host allowlist remains enabled by design so DNS-rebinding protection cannot be disabled accidentally.
