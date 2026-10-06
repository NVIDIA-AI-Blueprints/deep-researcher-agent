# Deep Researcher Agent API

Unified API plugin for the Deep Researcher Agent blueprint: **Knowledge API** (collections, documents) and **Async Job API** (agent-agnostic jobs with SSE streaming).

## Quick Start


### Web Mode - Local Development

```bash
# Loads API keys from deploy/.env; NeMo Agent toolkit auto-creates:
# - Local Dask cluster
# - SQLite database at .tmp/job_store.db
dotenv -f deploy/.env run nat serve --config configs/config_web_frag.yml
```

### Production (PostgreSQL + Dask Cluster)

```bash
export NAT_DASK_SCHEDULER_ADDRESS="tcp://scheduler:8786"
export NAT_JOB_STORE_DB_URL="postgresql://user:pass@host:5432/dbname"
dotenv -f deploy/.env run nat serve --config configs/config_web_frag.yml
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  NeMo Agent toolkit layer                       │
│                  (Built-in infrastructure)                      │
│                                                                 │
│   nat.front_ends.fastapi/                                       │
│   ├── job_store.py      # JobStore, JobStatus, JobInfo         │
│   ├── async_job.py      # Dask task patterns                   │
│   └── config.py         # db_url, scheduler_address            │
│                                                                 │
│   Provides: Job tracking, Dask scheduling, SQLite/PostgreSQL   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Deep Researcher Agent API (extension layer)                   │
│                                                                 │
│   frontends/deep_researcher_api/src/deep_researcher_api/                                │
│   ├── jobs/                                                     │
│   │   ├── runner.py           # run_agent_job (generic)         │
│   │   ├── submit.py           # submit_agent_job                │
│   │   ├── callbacks.py        # SSE event generation            │
│   │   ├── event_store.py      # Persistent event storage       │
│   │   └── connection_manager.py  # SSE connection lifecycle     │
│   ├── routes/                                                  │
│   │   ├── jobs.py             # /v1/jobs/async (submit, stream) │
│   │   ├── collections.py     # /v1/collections (Knowledge API) │
│   │   └── documents.py       # /v1/documents (upload, ingest)   │
│   ├── models/requests.py     # Pydantic request/response models │
│   ├── registry.py            # Agent type registry              │
│   ├── plugin.py              # NAT FastAPI plugin (DeepResearcherAPIWorker)│
│   └── websocket_reconnect.py # SSE reconnect handling           │
│                                                                 │
│   Provides: Async jobs + SSE streaming, Knowledge API, event replay │
└─────────────────────────────────────────────────────────────────┘
```

## API Routes

Base path: `/v1/jobs/async`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/jobs/async/agents` | GET | List public agents configured in the active workflow |
| `/v1/jobs/async/submit` | POST | Submit a new job |
| `/v1/jobs/async/job/{id}` | GET | Get job status |
| `/v1/jobs/async/job/{id}/stream` | GET | SSE stream from beginning |
| `/v1/jobs/async/job/{id}/stream/{last_event_id}` | GET | SSE stream from event ID |
| `/v1/jobs/async/job/{id}/cancel` | POST | Cancel running job |
| `/v1/jobs/async/job/{id}/state` | GET | Get current UI state |
| `/v1/jobs/async/job/{id}/report` | GET | Get final report |

### List Available Agents

The default catalog registers both `deep_researcher` and `shallow_researcher`, but this
endpoint returns an agent only when its function config is present in the active
workflow. It does not check credentials, source connectivity, tool health, or Dask
readiness.

```bash
curl http://localhost:8000/v1/jobs/async/agents
```

Response:

```json
{
  "agents": [
    {"agent_type": "deep_researcher", "description": "Performs comprehensive multi-loop deep research"},
    {"agent_type": "shallow_researcher", "description": "Performs quick single-turn research"}
  ]
}
```

### Submit a Job

```bash
curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "deep_researcher", "input": "Research quantum computing trends in 2026"}'
```

Optional body fields: `job_id` (custom ID), `expiry_seconds` (600–604800), `data_sources` (subset of IDs from `/v1/data_sources`; `[]` disables data-source tools while leaving unmapped utility tools available; unknown IDs and sources unavailable to the selected agent return 422).

A public catalog agent that is absent from the active workflow returns `400` before a
job is created:

```json
{
  "detail": {
    "code": "agent_not_configured",
    "message": "Agent 'shallow_researcher' is not configured in the active workflow",
    "agent_type": "shallow_researcher",
    "config_name": "shallow_research_agent"
  }
}
```

Response:

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "submitted",
  "agent_type": "deep_researcher"
}
```

### Stream Events (SSE)

```bash
curl http://localhost:8000/v1/jobs/async/job/{job_id}/stream
```

#### Replay and Live Handoff

When a client reconnects to an in-progress job stream, the server replays historical events as fast as possible. The server then sends a `stream.mode` event to indicate that catch-up is complete and the stream has switched to live polling.

Expected handoff event:

```
event: stream.mode
data: {"mode":"live"}
```

The `stream.mode` live event is sent once, after historical replay is complete and before subsequent live events.

### Get Final Report

```bash
curl http://localhost:8000/v1/jobs/async/job/{job_id}/report
```

## SSE Event Types

Events streamed during job execution:

| Event | Description |
|-------|-------------|
| `stream.mode` | Stream state transition event. `{"mode":"live"}` signals replay is complete and live streaming has started |
| `job.status` | Job status changes (running, success, failure) |
| `workflow.start` / `workflow.end` | Workflow lifecycle |
| `llm.start` / `llm.chunk` / `llm.end` | LLM inference progress |
| `tool.start` / `tool.end` | Tool invocations |
| `artifact.update` | Todos, citations, output, and generated-file metadata (`content_url` for durable bytes) |
| `job.error` | Error occurred |

## Configuration

### Default Behavior

| Mode | Async Jobs | Database | Notes |
|------|------------|----------|-------|
| **CLI** (`nat run`) | No | None | Agents run via WebSocket |
| **Web** (`nat serve`) | Yes | `./jobs.db` (or `front_end.db_url`) | Auto-creates Dask + SQLite |
| **Production** | Yes | PostgreSQL | Set `NAT_JOB_STORE_DB_URL` or `front_end.db_url` |

### NAT Config File

```yaml
# configs/config_web_frag.yml
general:
  front_end:
    _type: deep_researcher_api
    runner_class: deep_researcher_api.plugin.DeepResearcherAPIWorker
    db_url: sqlite+aiosqlite:///./jobs.db   # Job store + event store
    expiry_seconds: 86400

functions:
  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_llm
    tools: [tavily_web_search]
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `NAT_DASK_SCHEDULER_ADDRESS` | Dask scheduler | Auto-created local |
| `NAT_JOB_STORE_DB_URL` | Job store + event store database | `sqlite+aiosqlite:///./jobs.db` (or via front_end.db_url) |


## Authentication

Authentication is handled by `AuthMiddleware`, which runs on every request.  It
is disabled by default and opt-in via the `REQUIRE_AUTH` environment variable.

### Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `REQUIRE_AUTH` | Enforce authentication on all non-exempt routes | `false` |
| `DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE` | Control whether verified user identity is attached to NAT spans: `none`, `id`, or `full` | `none` |
| `DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET` | Secret used to pseudonymize verified user IDs in traces when identity tagging is enabled | unset |
| `DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE` | Control whether a pseudonymous client identifier is attached to NAT spans: `none` or `ip` | `none` |
| `DEEP_RESEARCHER_TRACE_CLIENT_ID_HMAC_SECRET` | Secret used to pseudonymize client IDs when client tagging is enabled; falls back to the user-identity secret | unset |
| `DEEP_RESEARCHER_TRACE_CLIENT_IP_HEADERS` | Ordered comma-separated client IP headers to trust before falling back to the ASGI client address | `x-real-ip,x-forwarded-for` |

### Enabling authentication

Set `REQUIRE_AUTH=true` and register at least one validator before the server
starts.  The server raises a `RuntimeError` at startup if auth is required but
no validators are registered.

Two registration mechanisms are supported — pick whichever fits your deployment:

### NAT span request tagging

Auth middleware can enrich NAT-exported workflow spans with low-risk request
tags and optional pseudonymous identity tags. The resolved tags are propagated
through HTTP requests, WebSocket workflow execution, and async jobs.

Always-on NAT span tags:

- `nat.deep-researcher.caller.type`: resolved caller type from the auth middleware
- `nat.deep-researcher.auth.transport`: `bearer`, `cookie`, or `none`
- `nat.deep-researcher.auth.verified`: whether the request resolved to a verified principal
- `nat.deep-researcher.access.channel`: inferred or explicitly supplied access channel

Optional user identity tags are controlled by `DEEP_RESEARCHER_TRACE_USER_IDENTITY_MODE`:

- `none`: disable user identity tagging entirely
- `id`: tag only pseudonymous stable identifiers (`nat.enduser.id`, `nat.deep-researcher.user.id`, `nat.deep-researcher.auth.type`)
- `full`: tag pseudonymous identifiers plus `nat.deep-researcher.user.email` and `nat.deep-researcher.user.name` when present

Only verified principals are tagged. Anonymous, internal, and unverified JWT
fallback callers are never attached to traces.
Set `DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET` when using `id` or `full`; otherwise
identity tagging is skipped.

Optional client correlation is controlled by `DEEP_RESEARCHER_TRACE_CLIENT_ID_MODE`:

- `none`: disable client correlation
- `ip`: add `nat.deep-researcher.client.id` as an HMAC-derived pseudonymous identifier from
  the client IP

Set `DEEP_RESEARCHER_TRACE_CLIENT_ID_HMAC_SECRET` when using `ip`. If unset, the middleware
falls back to `DEEP_RESEARCHER_TRACE_USER_IDENTITY_HMAC_SECRET`. `DEEP_RESEARCHER_TRACE_CLIENT_IP_HEADERS`
controls which ingress headers are consulted first.

#### 1. Programmatic (`register_validator`)

Call `register_validator()` in any code that runs before `nat serve`:

```python
from deep_researcher_api.plugin import register_validator
from deep_researcher_api.auth.jwt_validator import JWTValidator

register_validator(JWTValidator(issuer_url="https://your-identity-provider.com"))
```

#### 2. Entry points (recommended for packages / submodule deployments)

Any installed Python package can contribute validators by declaring an
`deep_researcher_api.validators` entry point.  The function must return a list of validator
instances.

```toml
# pyproject.toml of your deployment package
[project.entry-points."deep_researcher_api.validators"]
my_provider = "mypackage.auth:get_validators"
```

```python
# mypackage/auth.py
def get_validators() -> list:
    from deep_researcher_api.auth.jwt_validator import JWTValidator
    return [JWTValidator(issuer_url="https://your-identity-provider.com")]
```

After `pip install` / `uv sync`, validators are discovered automatically — no
code changes to the server are needed.

### Writing a custom validator

A validator is any object with an async `validate(token: str) -> dict | None`
method.  Return a dict with at minimum `{"type": "<name>"}` on success, or
`None` if the token is not recognized by this validator.  The first validator
to return a non-`None` result wins.

```python
class MyValidator:
    async def validate(self, token: str) -> dict | None:
        payload = verify_token(token)   # your verification logic
        if payload is None:
            return None
        return {"type": "my_provider", "sub": payload["sub"], "token": token}
```

### Caller types and the clarifier

The middleware sets `skip_clarifier=True` for callers that cannot participate in
interactive back-and-forth (anonymous requests, headless API callers).  Pass the
`X-DeepResearcher-Mode: headless` request header to force this for any authenticated caller.

---

## Registering New Agents

Agents are registered by type so the job runner can load them from NAT config. Register at import time (e.g. in your NAT plugin or app startup):

```python
from deep_researcher_api.registry import register_agent

register_agent(
    agent_type="my_agent",
    class_path="my_package.agent.MyAgent",
    config_name="my_agent_config",
    description="My custom agent",
)
```

Requirements:
- **config_name** must match a NAT function in your YAML (e.g. `deep_research_agent`).
- The agent class must be constructable from that config and support the run signature expected by the runner (see `jobs/runner.py`).

## Knowledge API

When a `knowledge_retrieval` function is configured in your workflow, the plugin adds:

- **`/v1/collections`** – Create and list collections (uses the same ingestor as the knowledge tool).
- **`/v1/documents`** – Upload files and trigger ingestion.

Backend (LlamaIndex, Foundational RAG, etc.) is determined by the `knowledge_retrieval` config. If no knowledge function is configured, these routes are not registered.

## Debug Console

When the `deep_researcher_debug` package is installed, the plugin registers a debug console at **`http://localhost:8000/debug`**:

- Real-time SSE streaming
- Job submission and tracking
- State visualization (todos, subagents, sources, tool calls)
- Copy SSE streams for debugging

## Comparison with NAT's Built-in Async

| Feature | NAT `/generate/async` | `/v1/jobs/async` |
|---------|----------------------|------------|
| Job tracking | JobStore | Same JobStore |
| Dask scheduling | Yes | Yes |
| Database | SQLite/Postgres | Same |
| **Agent-agnostic** | No | Yes |
| **SSE streaming** | No | Real-time events |
| **Event replay** | No | Via `last_event_id` |

## Related Documentation

- [Project ARCHITECTURE](../../docs/source/architecture/overview.md) – Overall system architecture
