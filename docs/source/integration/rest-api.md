<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# REST API

The AI-Q blueprint exposes a REST API built on top of NeMo Agent Toolkit's built-in [FastAPI](https://fastapi.tiangolo.com/) infrastructure. The **AI-Q API** is an extension layer that adds agent-agnostic async job management with SSE streaming, knowledge management endpoints, and event replay capabilities.

The API is served when running in **web mode** (`nat serve`). CLI mode (`nat run`) uses WebSocket communication instead and does not expose these endpoints.

## Architecture

NeMo Agent Toolkit provides the core infrastructure: job tracking, [Dask](https://www.dask.org/) scheduling, and SQLite/PostgreSQL persistence. The AI-Q API plugin (`aiq_api`) extends this with:

- **Async Jobs API** -- submit research queries to any registered agent, track progress through SSE
- **Durable Artifact API** -- list metadata and stream generated files captured from configured sandboxes
- **Knowledge API** -- manage document collections and trigger ingestion (when a knowledge function is configured)
- **Event replay** -- reconnect to an in-progress job and replay historical events from any point

## Async Jobs API

Base path: `/v1/jobs/async`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/jobs/async/agents` | List public agents configured in the active workflow |
| `POST` | `/v1/jobs/async/submit` | Submit a new research job |
| `GET` | `/v1/jobs/async/job/{job_id}` | Get job status |
| `GET` | `/v1/jobs/async/job/{job_id}/stream` | SSE event stream from beginning |
| `GET` | `/v1/jobs/async/job/{job_id}/stream/{last_event_id}` | SSE stream from event ID (reconnection) |
| `POST` | `/v1/jobs/async/job/{job_id}/cancel` | Cancel a running job |
| `POST` | `/v1/jobs/async/job/{job_id}/report/edit` | Create a revised report from a completed report job |
| `GET` | `/v1/jobs/async/job/{job_id}/state` | Get event-derived tool calls, outputs, and citations |
| `GET` | `/v1/jobs/async/job/{job_id}/artifacts` | List durable sandbox artifact metadata |
| `GET` | `/v1/jobs/async/job/{job_id}/artifacts/{artifact_id}/content` | Stream one durable artifact's bytes |
| `GET` | `/v1/jobs/async/job/{job_id}/report` | Get final research report |
| `GET` | `/v1/data_sources` | List available data sources |
| `GET` | `/live` | Process liveness check (no dependency checks) |
| `GET` | `/health` | Dependency readiness check (database, Dask, and content encryption) |

### List Available Agents

Returns all **public** registered agent types configured in the active workflow. The
`deep_researcher` and `shallow_researcher` entries are part of the default catalog, but
each is returned only when its corresponding function config is present in the loaded
workflow. Discovery reflects configuration availability, not credentials, tool health,
source connectivity, Dask readiness, or other runtime health checks.

Internal-only agents (registered with `public=False`, for example the `report_rewriter`
used by report follow-up) are intentionally omitted from this list and are rejected by
`POST /v1/jobs/async/submit` with `400`.

```bash
curl http://localhost:8000/v1/jobs/async/agents
```

**Response:**

```json
{
  "agents": [
    {"agent_type": "deep_researcher", "description": "Performs comprehensive multi-loop deep research"},
    {"agent_type": "shallow_researcher", "description": "Performs quick single-turn research"},
    {"agent_type": "data_science", "description": "Analyzes enterprise structured data, documents, and web evidence in one autonomous loop"}
  ]
}
```

### Submit a Job

Submit a research query to a registered agent. Returns a job ID for tracking progress through SSE.

```bash
curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "deep_researcher",
    "input": "Research quantum computing trends in 2026"
  }'
```

**Request body (`JobSubmitRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_type` | `string` | Yes | Agent identifier (for example, `deep_researcher`, `shallow_researcher`, `data_science`) |
| `input` | `string` | Yes | Research query. Must be non-blank after trimming (whitespace-only is rejected with 422) |
| `job_id` | `string` | No | Custom job ID. Auto-generated UUID if omitted. Pattern: `[a-zA-Z0-9_-]`, max 64 chars |
| `expiry_seconds` | `integer` | No | Job expiry in seconds. Range: 600--604800 (10 min to 7 days). Default from config |
| `data_sources` | `list[string]` | No | Optional data source IDs (from `/v1/data_sources`) to scope the job. Omit or `null` for all data-source tools; `[]` for no data-source tools. Unmapped utility tools remain available. Unknown IDs or sources unavailable to the selected agent return 422 |

**Response (`JobStatusResponse`):**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "SUBMITTED",
  "agent_type": "deep_researcher"
}
```

**Error responses:**

| Status | Reason |
|--------|--------|
| `400` | Unknown, **internal-only**, or registered-but-unconfigured agent type, or invalid request |
| `413` | Deep-research `input` exceeds `AIQ_MAX_DEEP_RESEARCH_INPUT_CHARS` |
| `409` | A custom `job_id` collides with an existing job, or a selected protected source requires per-user authentication |
| `429` | The caller reached the active deep-research job limit or rolling per-minute submission limit. The response includes `Retry-After` |
| `422` | Validation error: blank/whitespace-only `input`, invalid request fields, unknown data source IDs, or sources unavailable to the selected agent. Data source errors include `message`, `invalid_ids`, `unavailable_for_agent`, and `known_ids` |
| `500` | Content encryption configuration is invalid, authorization persistence fails, or agent/tool configuration fails unexpectedly |
| `503` | Content encryption, Dask, the admission database, or deployment-wide job capacity is unavailable. Capacity responses include `Retry-After` |

Deep-research admission is enforced before enqueue for both REST submissions and
deep-research jobs launched from chat. Limits are serialized in the configured
`NAT_JOB_STORE_DB_URL`, so backend replicas sharing PostgreSQL cannot race past
the configured capacity. See [Production Considerations](../deployment/production.md#deep-research-admission-control)
for defaults and identity behavior.

A public catalog entry that is not configured in the active workflow is rejected before
source validation or job creation:

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

### Edit a Report (Report Follow-up)

Create a revised report from a **completed** report job. The caller is authorized against
the parent job, the durable report context is reconstructed, and an internal `report_rewriter`
child job is submitted that emits a full revised report. The parent report is never mutated.

```bash
curl -X POST http://localhost:8000/v1/jobs/async/job/{job_id}/report/edit \
  -H "Content-Type: application/json" \
  -d '{"input": "Make the executive summary shorter and remove the appendix."}'
```

**Request body (`ReportEditRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | `string` | Yes | Edit instruction for the parent report. Must be non-blank after trimming (whitespace-only is rejected with 422) |
| `job_id` | `string` | No | Custom child job ID. Auto-generated if omitted. Pattern: `[a-zA-Z0-9_-]`, max 64 chars |
| `expiry_seconds` | `integer` | No | Child job expiry in seconds. Range: 600--604800. Default from config |

**Response (`ReportEditResponse`):**

```json
{
  "job_id": "child-job-uuid",
  "parent_job_id": "parent-job-uuid",
  "status": "SUBMITTED",
  "agent_type": "report_rewriter"
}
```

Track the child job with the standard status/stream/report endpoints. Its `GET .../report`
response carries `parent_job_id`, `interaction_action` (`edit`), and `result_kind` (`report`).

**Error responses:**

| Status | Reason |
|--------|--------|
| `404` | Parent job not found (or not accessible to the caller when auth is enabled) |
| `409` | Parent job is incomplete, has no durable report, or the supplied child `job_id` collides |
| `422` | Validation error: blank/whitespace-only `input`, invalid child `job_id`, or invalid `expiry_seconds` |
| `500` | Failed to submit the report edit job |
| `503` | Dask scheduler not available |

#### Conversation-scoped default for chat follow-up

The chat surface (`POST /chat`) routes report follow-up (ask / edit / delta) using an
`active_report_job_id`. Clients may send it explicitly (it always wins), but when it is omitted
the server **defaults to the most recent completed report job in the request's conversation** —
identified by the `conversation-id` request header. This lets any client (CLI, API, or the UI on
reload) get report follow-up by simply reusing a stable `conversation-id` across turns, without
tracking report ids. Jobs record their originating `conversation-id` at submit time for this lookup.

When no `active_report_job_id` and no (or a brand-new) `conversation-id` are present, follow-up
degrades to fresh research. The lookup is authorized like every other job read: under
`REQUIRE_AUTH=true` it is scoped to the caller's own jobs; under `REQUIRE_AUTH=false` (the public
default) ownership is not enforced, so `conversation-id` is the only isolation boundary — keep it
unguessable in any shared/multi-user deployment, consistent with the other job endpoints in that mode.

### Get Job Status

```bash
curl http://localhost:8000/v1/jobs/async/job/{job_id}
```

**Response:**

```json
{
  "job_id": "abc123",
  "status": "RUNNING",
  "agent_type": "deep_researcher",
  "error": null,
  "created_at": "2026-02-12T10:30:00Z"
}
```

Job statuses: `SUBMITTED`, `RUNNING`, `SUCCESS`, `FAILURE`, `INTERRUPTED`.

Typed source-condition failures, such as running with no selected sources or receiving no
results from the selected sources, remain in `FAILURE`. Their `error` is an actionable
message describing how to correct the source selection or query. If the agent produced a
sanitized answer before detecting the source condition, that report remains available from
`GET /v1/jobs/async/job/{job_id}/report`. Unexpected failures continue to return a sanitized
error and do not expose internal exception details or plaintext output.

### Stream Events (SSE)

Stream real-time events from a running or completed job using Server-Sent Events.

```bash
# Stream from beginning
curl -N http://localhost:8000/v1/jobs/async/job/{job_id}/stream

# Reconnect from a specific event ID
curl -N http://localhost:8000/v1/jobs/async/job/{job_id}/stream/{last_event_id}
```

Each SSE message has the format:

```
id: 42
event: llm.chunk
data: {"content": "The latest advances..."}
```

#### Replay and Live Handoff

When a client connects (or reconnects) to a job stream, the server replays all historical events as fast as possible, then sends a `stream.mode` event to signal the transition to live streaming. The exact payload depends on the database backend:

- **SQLite (polling):** First sends `{"mode":"polling","interval_ms":500}`, then `{"mode":"live"}` after replay completes.
- **PostgreSQL (pub-sub):** Sends `{"mode":"pubsub","channel":"job_events_<job_id>"}` after replay completes.

After the transition event, new events are delivered in real time. For PostgreSQL backends, the server uses `LISTEN/NOTIFY` for sub-10ms latency. For SQLite, it polls at 500ms intervals.

### Cancel a Job

```bash
curl -X POST http://localhost:8000/v1/jobs/async/job/{job_id}/cancel
```

**Response:**

```json
{
  "job_id": "abc123",
  "status": "INTERRUPTED",
  "task_cancelled": true
}
```

| Status | Reason |
|--------|--------|
| `400` | Job is not in `RUNNING` state |
| `404` | Job not found |

### Get Event-Derived Job State

Returns accumulated tool calls, outputs, and source citations reconstructed from job
events. This is distinct from the durable sandbox artifact endpoints, which store file
metadata and bytes outside the event-derived state document.

```bash
curl http://localhost:8000/v1/jobs/async/job/{job_id}/state
```

**Response (`JobStateResponse`):**

```json
{
  "job_id": "abc123",
  "has_state": true,
  "state": null,
  "artifacts": {
    "tools": [
      {
        "id": "tool_123",
        "name": "tavily_web_search",
        "input": {"query": "quantum computing 2026"},
        "output": "...",
        "status": "completed",
        "workflow": "shallow_research_agent"
      }
    ],
    "outputs": [
      {
        "type": "citation_source",
        "content": "https://example.com/article",
        "workflow": "shallow_research_agent"
      }
    ],
    "sources": {
      "found": 12,
      "cited": 8,
      "found_urls": ["https://..."],
      "cited_urls": ["https://..."]
    }
  }
}
```

### Durable Sandbox Artifacts

Durable artifacts are generated files such as charts, CSVs, notebooks, or documents
harvested from a configured deep-research sandbox. Capture is opt-in: the deep researcher
must have a sandbox and `artifact_capture.enabled: true`, and the API/worker must be able
to open the artifact store. Successful `execute` calls checkpoint manifest-declared files.
Success and failure paths perform one idempotent final manifest-plus-directory scan before
cleanup. Cancellation performs that scan only when the provider is idle; a busy provider
is terminated without waiting and artifacts from earlier checkpoints remain durable.
Capture remains best-effort, so sandbox execution alone does not guarantee that every
generated file is persisted.

#### Live and Replayed File Events

After storing a file, the worker emits a metadata-only `artifact.update` event with nested
`data.type: "file"`. It includes the artifact and job IDs, display filename, kind, MIME type,
size, digest, optional title/caption/inline metadata, and the job-scoped content URL. It does
not contain file bytes, the storage URI, or the sandbox path. These stored events drive both
live delivery and replay into the web UI Files tab, whose **Open file** action uses the
job-scoped content endpoint below. When `artifact_id` is present, the UI derives that
same-origin path from the current job and artifact IDs instead of trusting an arbitrary
event URL. Rejected candidates emit `artifact.warning` with
`data.path` and `data.reason` instead. Refer to [Data Flow](../architecture/data-flow.md#event-structure)
for the canonical payload.

#### List Artifact Metadata

```bash
curl http://localhost:8000/v1/jobs/async/job/{job_id}/artifacts
```

**Response:**

```json
{
  "job_id": "abc123",
  "artifacts": [
    {
      "artifact_id": "a1b2c3d4",
      "job_id": "abc123",
      "kind": "image",
      "mime_type": "image/png",
      "filename": "market-share.png",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size_bytes": 184320,
      "title": "Market share",
      "caption": "Market share by vendor",
      "inline": true,
      "workflow": "researcher-agent",
      "source_tool_call_id": "call_123",
      "provenance": {
        "command": "python /tmp/chart.py",
        "script_sha256": null,
        "input_file_hashes": {},
        "package_snapshot": []
      },
      "created_at": "2026-07-08T12:00:00Z",
      "status": "available"
    }
  ]
}
```

Each item contains `artifact_id`, `job_id`, `kind`, `mime_type`, `filename`,
`sha256`, `size_bytes`, optional `title` and `caption`, `inline`, optional
`workflow` and `source_tool_call_id`, `provenance`, `created_at`, and `status`.
The response intentionally excludes `storage_uri` and `sandbox_path`; clients fetch
bytes through the content endpoint rather than learning storage credentials, hostnames,
or internal sandbox layout.

#### Get Artifact Content

```bash
curl -OJ http://localhost:8000/v1/jobs/async/job/{job_id}/artifacts/{artifact_id}/content
```

Both durable artifact endpoints first load the owning job. With `REQUIRE_AUTH=true`,
access is scoped to that job's owning principal. Missing or invalid authentication can
return `401` or `403`, depending on the configured authentication middleware and principal
gate; a cross-owner lookup is hidden as `404`. With the default `REQUIRE_AUTH=false`, job
ownership is not enforced, so any caller with a valid job ID can access its artifacts.
Treat no-auth mode as trusted-local development only; do not expose it on a shared or
untrusted network. Enable authentication before serving durable artifacts in multi-user or
externally reachable deployments.

The list endpoint returns `404` when the owning job is not found; the content endpoint
returns `404` when either the job or artifact is not found.

Artifact cleanup is not tied to each job's `expiry_seconds`. The background cleanup uses
one server-wide configured/default retention duration and compares it with each artifact's
`created_at`. Stored artifacts can therefore outlive a shorter-expiry job, while artifacts
for a longer-expiry job can be removed before that job expires. Do not rely on per-job
artifact retention alignment unless the runtime contract changes.

Raster images require matching content magic. PDF and allowed text/data formats such as
CSV, JSON, Markdown, and notebooks may fall back to allowlisted extension-based MIME
classification. Only magic-confirmed PNG, JPEG, and WebP images are served with
`Content-Disposition: inline`; SVG, HTML, notebooks, PDFs, and all other types are forced
to `attachment`. Every content response sets `X-Content-Type-Options: nosniff`.

### Get Final Report

```bash
curl http://localhost:8000/v1/jobs/async/job/{job_id}/report
```

**Response (`JobReportResponse`):**

```json
{
  "job_id": "abc123",
  "has_report": true,
  "report": "# Quantum Computing Trends in 2026\n\n...",
  "parent_job_id": null,
  "interaction_action": null,
  "result_kind": null
}
```

For report follow-up child jobs (refer to [Edit a Report](#edit-a-report-report-follow-up)),
`parent_job_id`, `interaction_action` (for example `edit`), and `result_kind` (for example
`report`) identify the originating report and interaction. They are `null` for root research
jobs.

## SSE Event Types

Events streamed during job execution. Refer to the [Data Flow](../architecture/data-flow.md) page for details on how these events are generated and consumed.

| Event | Description |
|-------|-------------|
| `stream.mode` | Stream state transition. In polling mode (SQLite), the server first sends `{"mode":"polling","interval_ms":500}` then `{"mode":"live"}` after replay. In pub-sub mode (PostgreSQL), the server sends `{"mode":"pubsub","channel":"..."}` after replay |
| `job.status` | Job status changes (`RUNNING`, `SUCCESS`, `FAILURE`, `INTERRUPTED`). May include `error` and `reconnected` fields |
| `job.error` | Error occurred during execution |
| `job.shutdown` | Server is shutting down gracefully |
| `job.heartbeat` | Periodic heartbeat from Dask worker (every 30s); keeps SSE connection alive |
| `job.cancelled` | Job was cancelled by user |
| `job.update` | Retry notification when a chain (LLM call) fails and is retried |
| `job.cancellation_requested` | Cancellation was requested by user |
| `workflow.start` / `workflow.end` | Workflow lifecycle boundaries |
| `llm.start` / `llm.chunk` / `llm.end` | LLM inference progress. `llm.chunk` contains streaming token content |
| `tool.start` / `tool.end` | Tool invocation lifecycle. Includes tool name, input, and output |
| `artifact.update` | Structured updates for todos, citations, output content, legacy text files, and durable generated-file metadata with a job-scoped content URL |
| `artifact.warning` | Durable file candidate was rejected; contains its sandbox path and rejection reason, but no file bytes |

Sandbox-generated files also use `artifact.update` with `data.type: "file"`.
Their metadata includes `artifact_id`, `job_id`, `file_path`, authenticated
`content_url`, `kind`, validated `mime_type`, `size_bytes`, `sha256`, `title`,
`caption`, and `inline`. File bytes are never included in SSE; fetch them through
the authenticated `content_url`.

## Agent Registration

Agents are registered by type so the async job runner can load them dynamically. Registration happens at import time (typically in a NeMo Agent Toolkit plugin module):

```python
from aiq_api.registry import register_agent

register_agent(
    agent_type="my_agent",
    class_path="my_package.agent.MyAgent",
    config_name="my_agent_config",
    description="My custom research agent",
)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `agent_type` | Short identifier used in submit requests (for example, `deep_researcher`) |
| `class_path` | Full module path to the agent class |
| `config_name` | Must match a function name in the NeMo Agent Toolkit YAML config (for example, `deep_research_agent`) |
| `description` | Human-readable description shown in the agent list |

The default agents (`deep_researcher` and `shallow_researcher`) are registered automatically when the `aiq_api` plugin loads. Registration adds them to the catalog; each agent must also have its `config_name` defined in the active workflow to appear in `GET /v1/jobs/async/agents` or be accepted by `POST /v1/jobs/async/submit`.

## Knowledge API

The Knowledge API endpoints are **conditionally registered** -- they appear only when a `knowledge_retrieval` function is configured in the workflow. The backend (LlamaIndex, Foundational RAG, etc.) is determined by the knowledge config.

### Collection Routing

Knowledge ingestion and retrieval select collections independently. Collection and document endpoints use the
collection named in the request path, while retrieval uses the `conversation-id` header when present and otherwise
falls back to the configured `collection_name`. To query a collection after ingesting into it, reuse the exact
collection name as the `conversation-id` header. A `conversation_id` field in the `/v1/chat/completions` JSON body is
not used for collection routing.

For UI behavior, environment-variable usage, and an end-to-end request example, see
[Collection Routing](../customization/knowledge-layer.md#collection-routing).

### Collection Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/collections` | Create a new collection |
| `GET` | `/v1/collections` | List all collections |
| `GET` | `/v1/collections/{name}` | Get collection details |
| `DELETE` | `/v1/collections/{name}` | Delete a collection and all its contents |
| `GET` | `/v1/knowledge/health` | Check knowledge backend health |

#### Create a Collection

```bash
curl -X POST http://localhost:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research-papers",
    "description": "Collection of ML research papers",
    "metadata": {}
  }'
```

#### List Collections

```bash
curl http://localhost:8000/v1/collections
```

### Document Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/collections/{collection_name}/documents` | Upload and ingest documents (returns job ID) |
| `GET` | `/v1/collections/{collection_name}/documents` | List documents in a collection |
| `DELETE` | `/v1/collections/{collection_name}/documents` | Delete documents by file ID |
| `GET` | `/v1/documents/{job_id}/status` | Get ingestion job status |

#### Upload Documents

Document upload is asynchronous. The endpoint returns a job ID that you poll for ingestion status.

The API streams uploads to private temporary files and enforces the deployment's
`FILE_UPLOAD_ACCEPTED_TYPES`, `FILE_UPLOAD_MAX_SIZE_MB`, and `FILE_UPLOAD_MAX_FILE_COUNT` settings.
It rejects requests that exceed the size or count limits with `413 Payload Too Large`, and rejects unsupported,
malformed, or content-mismatched files with `415 Unsupported Media Type`.

```bash
curl -X POST http://localhost:8000/v1/collections/research-papers/documents \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf"
```

**Response (202 Accepted):**

```json
{
  "job_id": "job_abc123",
  "file_ids": ["file_abc123", "file_def456"],
  "message": "Ingestion job submitted for 2 file(s)"
}
```

The application does not provide per-client upload rate limiting. Production operators must put the endpoint behind
an authenticated API gateway or ingress and configure a per-identity rate limit appropriate for their deployment.
As a starting point, use 10 upload requests per minute with a burst of 20, then tune against expected document sizes,
ingestion capacity, and tenant isolation requirements. Do not rely only on source IP when requests pass through shared
proxies or NAT. Configure the gateway's maximum request-body size slightly above `FILE_UPLOAD_MAX_SIZE_MB` as well, so
grossly oversized multipart requests are rejected before application-level parsing.

#### Delete Documents

```bash
curl -X DELETE http://localhost:8000/v1/collections/research-papers/documents \
  -H "Content-Type: application/json" \
  -d '{"file_ids": ["file_abc123", "file_def456"]}'
```

**Request body (`DeleteFilesRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_ids` | `list[string]` | Yes | List of file IDs to delete |

#### Check Ingestion Status

```bash
curl http://localhost:8000/v1/documents/{job_id}/status
```

### List Data Sources

Returns available data sources based on the configured tools.

```bash
curl http://localhost:8000/v1/data_sources
```

**Response:**

```json
[
  {
    "id": "web_search",
    "name": "Web Search",
    "description": "Search the web for real-time information."
  },
  {
    "id": "knowledge_layer",
    "name": "Knowledge Base",
    "description": "Search uploaded documents and files."
  }
]
```

The `knowledge_layer` entry only appears when a knowledge retrieval function is configured.

### Liveness and Readiness Checks

Use `/live` for process liveness probes. It returns success without checking the
database, Dask, or content-encryption dependencies.

```bash
curl http://localhost:8000/live
```

```json
{
  "status": "alive"
}
```

Use `/health` for readiness checks. It returns HTTP 503 when a required
dependency is unavailable.

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "dask_available": true,
  "db": "ok",
  "encryption": {
    "mode": "off",
    "ready": true
  }
}
```

## Configuration

The API is configured through the NeMo Agent Toolkit config file under `general.front_end`:

```yaml
general:
  front_end:
    _type: aiq_api
    runner_class: aiq_api.plugin.AIQAPIWorker
    db_url: ${NAT_JOB_STORE_DB_URL:-sqlite+aiosqlite:///./jobs.db}
    expiry_seconds: 86400  # 24 hours
    cors:
      allow_origin_regex: 'http://localhost(:\d+)?'
      allow_methods: [GET, POST, DELETE, OPTIONS]
      allow_headers: ["*"]
      allow_credentials: true
```

### Mode Comparison

| Mode | Command | Async Jobs | Database | API Available |
|------|---------|------------|----------|---------------|
| CLI | `nat run` | No | None | No |
| Web (local) | `nat serve` | Yes | SQLite (`./jobs.db`) | Yes |
| Production | `nat serve` | Yes | PostgreSQL | Yes |

### Database Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `NAT_JOB_STORE_DB_URL` | Job store + event store database | `sqlite+aiosqlite:///./jobs.db` |
| `NAT_DASK_SCHEDULER_ADDRESS` | Dask scheduler for distributed execution | Auto-created local cluster |

For production deployments, use PostgreSQL for both the job store and LISTEN/NOTIFY-based real-time SSE:

```bash
export NAT_JOB_STORE_DB_URL="postgresql+asyncpg://user:pass@host:5432/aiq_jobs"  # pragma: allowlist secret
export NAT_DASK_SCHEDULER_ADDRESS="tcp://scheduler:8786"
```

## Debug Console

When the `aiq_debug` package is installed, a debug console is available at `http://localhost:8000/debug` with:

- Real-time SSE streaming visualization
- Job submission and tracking
- State visualization (todos, subagents, sources, tool calls)
- Copy SSE streams for debugging
