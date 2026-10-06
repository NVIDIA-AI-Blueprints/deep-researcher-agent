# NeMo Retriever Knowledge Layer Backends

Deep Researcher Agent supports [NVIDIA NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever)
through two separately registered backends. `nemo_retriever` calls a deployed
Retriever REST API. The experimental `nemo_retriever_local` runs the pinned
Retriever library and LanceDB inside the Deep Researcher Agent Python process. See the canonical
[knowledge-layer guide](../../../../docs/source/customization/knowledge-layer.md)
for shared Knowledge API behavior.

## Contents

- [Choose a backend](#choose-a-backend)
- [Embedded local backend](#embedded-local-backend-experimental)
  - [Choose an extraction profile](#choose-an-extraction-profile)
  - [Install and start](#install-and-start-the-local-backend)
  - [Local configuration](#local-configuration)
  - [Local operating model](#local-operating-model)
- [Upload format and batch contract](#upload-format-and-batch-contract)
- [Deployed REST backend](#deployed-rest-backend)
  - [Service compatibility](#service-compatibility)
  - [Service configuration](#service-configuration)
  - [Connect to the service](#connect-to-the-service)
  - [Start Deep Researcher Agent](#start-deep-researcher-agent-with-the-service-backend)
  - [Service validation](#service-validation)
  - [Service troubleshooting](#service-troubleshooting)

## Choose a backend

Both backends implement the same Deep Researcher Agent Knowledge API. Choose based on the
deployment and operational model; this is not a runtime mode switch.

| Consideration | Embedded library: `nemo_retriever_local` | Deployed service: `nemo_retriever` |
|---|---|---|
| Topology | Deep Researcher Agent, Retriever, and LanceDB share one Python 3.12 process | Deep Researcher Agent calls a separately operated Retriever gateway |
| Infrastructure | No Retriever or vector-database service | Retriever service deployed with Docker Compose or Helm/Kubernetes |
| Storage | Embedded LanceDB in `NRL_LOCAL_DATA_DIR` | Storage and durability are owned by the Retriever deployment |
| Inference | Extraction and embedding can call remote endpoints | Inference is configured and operated by the Retriever service |
| Concurrency | One Deep Researcher Agent process per data directory | Supports shared service deployments and multiple Deep Researcher Agent clients |
| Best fit | Laptop evaluation, local development, and zero-deployment workflows | Shared, scaled, or centrally operated environments |
| Support level | Experimental; Apple Silicon macOS, Windows x64, and Linux x64 targets | Existing deployed-service integration |
| Retriever version | Pinned to [`c80f4a5189ee10b98cbdb93e2f853ceb7b699c3b`](https://github.com/NVIDIA/NeMo-Retriever/commit/c80f4a5189ee10b98cbdb93e2f853ceb7b699c3b) | No Python dependency; requires the compatible service contract described below |

## Embedded local backend (experimental)

`nemo_retriever_local` delegates extraction, schemas, collection lifecycle,
storage, and retrieval to the unmodified Retriever library pinned in
[`environments/nemo_retriever_local/pyproject.toml`](../../../../environments/nemo_retriever_local/pyproject.toml).
Deep Researcher Agent does not start a Retriever HTTP service, a vector-database server, Ray, or
containers for this path.

```text
Deep Researcher Agent process -> NeMo Retriever operators -> embedded LanceDB
                       |
                       +-> configured extraction and embedding endpoints
```

Document ingestion does not require a generative LLM. Normal research invokes
the registered `knowledge_search` tool and requires an OpenAI-compatible agent
LLM.

The embedded adapter is orchestration over Retriever's public library
contracts, not a parallel VectorDB implementation:

| Deep Researcher Agent owns | NeMo Retriever owns |
|---|---|
| Temporary staging, Deep Researcher Agent job/status models, bounded work admission, adapter lifecycle, and universal-schema normalization | Extraction planning and graph construction, inference helpers, canonical schemas, `CollectionWriteContext`, collection catalog, chunk identity, recovery, deletion, physical table selection, VDB writes, retrieval, ranking, and distance |

Deep Researcher Agent constructs LanceDB with the same non-overwriting, service-collection
configuration used by Retriever's VectorDB service, then calls
`IngestVdbOperator`, `RetrieveVdbOperator`, and Retriever collection APIs. It
does not issue direct LanceDB searches, reproduce Retriever schemas, or expose
physical table identifiers. A public embedded collection facade and a public
embedding-default resolver remain upstream promotion follow-ups; neither is
required for this pinned experimental integration.

### Choose an extraction profile

Deep Researcher Agent exposes the two extraction profiles supported by the pinned Retriever
revision: `fast-text` and `auto`. Neither profile is universally preferred.
Select a profile based on corpus characteristics, retrieval requirements,
ingestion latency objectives, and inference usage.

| Profile | Extraction scope | Operational characteristics | Use when |
|---|---|---|---|
| `fast-text` | PDF/document text through PDFium, followed by text embedding | Optimized for ingestion efficiency and lower inference usage; skips images, tables, charts, page images, Page Elements, and OCR | The corpus is text-centric and ingestion throughput or cost is the primary objective |
| `auto` (default) | Text, images, tables, charts, page rendering, Page Elements, and OCR | Optimized for broad extraction coverage and performs additional inference stages; Table Structure remains off unless `NRL_TABLE_STRUCTURE_INVOKE_URL` is configured | Visual or structured content contributes to retrieval quality and the additional processing is acceptable |

Chunk count and ingestion time depend on document content, endpoint load, and
network conditions. Compare the profiles with representative documents before
selecting a production setting. Because the profile is recorded in collection
ownership metadata, use a new collection or data directory when changing it.

### Install and start the local backend

The isolated environment requires Python `>=3.12,<3.13` and keeps Retriever
out of the normal `knowledge-layer[all]` installation.

```bash
uv sync --project environments/nemo_retriever_local --frozen
```

Retriever's default hosted extraction and embedding endpoints accept an NVIDIA
Build `nvapi-...` key. `NRL_INFERENCE_API_KEY` may use the same value as
`NVIDIA_API_KEY`; set a distinct value only when Retriever and the Deep Researcher Agent
LLM need different credentials. When unset, Retriever reuses `NVIDIA_API_KEY`
and then `NGC_API_KEY`.

Set `TAVILY_API_KEY` to enable the standard web-search tools. Without it, the
server and collection/document APIs remain available, while web-search calls
return the existing missing-key response.

```bash
uv run --project environments/nemo_retriever_local --frozen \
  dotenv -f deploy/.env run \
  nat serve --config_file configs/config_web_nemo_retriever_local.yml --port 8000
```

This starts the backend at `http://localhost:8000`. The web UI creates session
collection names automatically, and normal research queries the selected
collection through the registered `knowledge_search` tool.

### Local configuration

Start from
[`configs/config_web_nemo_retriever_local.yml`](../../../../configs/config_web_nemo_retriever_local.yml).
Retriever supplies its default hosted endpoint URLs, so configure individual
invoke URLs only for compatible external or self-hosted NIM overrides. The
table below lists every NeMo Retriever environment variable exposed by the
embedded Deep Researcher Agent backend.

| YAML field | Environment variable | Default | Purpose |
|---|---|---:|---|
| `backend_config.scope` | `NRL_SCOPE` | `local` | Logical collection scope |
| `backend_config.data_dir` | `NRL_LOCAL_DATA_DIR` | `.deep-researcher-data/nemo_retriever` | LanceDB, catalog, recovery, and staging root |
| `backend_config.profile` | `NRL_LOCAL_PROFILE` | `auto` | `auto` or `fast-text` extraction profile |
| `backend_config.inference_api_key` | `NRL_INFERENCE_API_KEY` | `NVIDIA_API_KEY`, then `NGC_API_KEY` | Optional explicit NVIDIA Build credential for extraction plus document/query embedding; it may match `NVIDIA_API_KEY` |
| `backend_config.page_elements_invoke_url` | `NRL_PAGE_ELEMENTS_INVOKE_URL` | Retriever default | Optional Page Elements endpoint override |
| `backend_config.ocr_invoke_url` | `NRL_OCR_INVOKE_URL` | Retriever default | Optional OCR endpoint override |
| `backend_config.table_structure_invoke_url` | `NRL_TABLE_STRUCTURE_INVOKE_URL` | unset | Enables and overrides Table Structure when configured |
| `backend_config.embed_invoke_url` | `NRL_EMBED_INVOKE_URL` | Retriever default | Optional embedding endpoint override |
| `backend_config.embed_model_name` | `NRL_EMBED_MODEL_NAME` | Retriever default | Optional embedding model override |
| `backend_config.embed_model_provider_prefix` | `NRL_EMBED_MODEL_PROVIDER_PREFIX` | Retriever default | Optional model provider-prefix override |
| `backend_config.collection_ttl_hours` | `NRL_COLLECTION_TTL_HOURS` | `24` | Expiration applied to new collections |

If `NRL_INFERENCE_API_KEY` is unset, the pinned Retriever revision falls back
to `NVIDIA_API_KEY` and then `NGC_API_KEY`. All configured Retriever inference
endpoints use the resolved Retriever credential; Deep Researcher Agent does not define a
separate key for each endpoint.

### Local operating model

- Collections, committed documents, chunks, and recovery markers survive an
  Deep Researcher Agent restart; v1 job history is process-local.
- A process-lifetime file lock permits one Deep Researcher Agent process per canonical data
  directory. Use separate directories for parallel processes.
- Document ingestion remains asynchronous, while the shipped profile runs full
  deep research inline on threaded Dask workers so the active Retriever runtime
  and LanceDB lock remain in one process. Detached, durable async research jobs
  require the deployed service backend.
- Interrupted work before a document write is not resumed as a job. Collection
  reconciliation repairs committed collection state at startup and periodically.
- Physical LanceDB table names and paths are not exposed through the Knowledge
  API. Retriever and LanceDB can still emit them in process logs, so treat logs
  as operationally sensitive.
- Initial targets are Apple Silicon macOS, Windows x64, and Linux x64 with
  remote inference. Python 3.13, Intel macOS, local GPU inference, and shared
  multi-process data directories are outside the initial support target.

## Upload format and batch contract

Deep Researcher Agent keeps its global upload allowlist at the common backend subset. To enable
PowerPoint for either NeMo Retriever backend, set this value in the shared
application environment—normally the ignored `deploy/.env`—so the UI and
backend apply the same policy:

```dotenv
FILE_UPLOAD_ACCEPTED_TYPES=.pdf,.docx,.pptx,.txt,.md
```

At the pinned local-library revision, Retriever plans `.pptx` through its
document/PDF extraction branch for both `auto` and `fast-text`. DOCX and PPTX
conversion requires `libreoffice` on `PATH`; deployed-service support also
depends on the dependencies installed in the service image.

Deep Researcher Agent validates each upload request atomically before creating a job. One
disallowed or malformed file rejects the entire request with HTTP 415 and no
files are submitted. After a job is accepted, upload or extraction failures
remain visible per file and the job can complete with partial success.

## Deployed REST backend

The `nemo_retriever` backend connects Deep Researcher Agent to an independently deployed NeMo
Retriever service through its public REST API. Deep Researcher Agent never imports the
`nemo-retriever` Python package, opens LanceDB, selects physical tables, or
configures extraction workers.

```text
Deep Researcher Agent -> knowledge_layer.nemo_retriever -> NRL public gateway REST API
```

NeMo Retriever owns authentication, extraction, OCR, tokenization, embedding,
indexing, durable collections, and document lifecycle. Deep Researcher Agent owns only logical
collection/document input, job polling, retrieval queries, and conversion to
the Deep Researcher Agent universal schemas.

### Service compatibility

This adapter targets the collection-management contract validated with the
immutable NeMo Retriever integration head
[`f3a0b418b7250fa8823ec44dea569b07e2b008cb`](https://github.com/NVIDIA/NeMo-Retriever/commit/f3a0b418b7250fa8823ec44dea569b07e2b008cb),
which contains the collection-management fixes and TXT/HTML service-mode
tokenizer support. The service must expose the job-scoped
ingestion routes under `/v1/ingest/job`. A 404 or 410 from version-probing job
creation or immediate document-upload routes is reported as an actionable
API-version mismatch. The same status from job polling remains an ordinary
missing or expired resource response.

TXT, HTML, PDF, Office, image, audio, and video support is determined by the
deployed NRL service and its configured dependencies. Deep Researcher Agent does not override
NRL extraction, OCR, tokenization, embedding, or indexing settings. Use a
service image containing the tokenizer fix before validating TXT or HTML.

### Service configuration

Start from [`configs/config_web_nemo_retriever.yml`](../../../../configs/config_web_nemo_retriever.yml).
The required deployment setting is an explicit workspace scope. The table
below lists every NeMo Retriever environment variable exposed by the service
adapter.

```bash
export NRL_BASE_URL=http://127.0.0.1:7670
export NRL_SCOPE=workspace-123
export NRL_API_TOKEN='replace-with-a-secret'  # omit only for an auth-disabled dev deployment
```

| YAML field | Environment variable | Default | Purpose |
|---|---|---:|---|
| `backend_config.base_url` | `NRL_BASE_URL` | `http://127.0.0.1:7670` | Public NRL gateway, not a worker or VectorDB pod |
| `backend_config.api_token` | `NRL_API_TOKEN` | unset | Optional bearer token stored as a secret |
| `backend_config.scope` | `NRL_SCOPE` | required | Logical workspace scope sent on every request |
| `backend_config.connect_timeout_s` | `NRL_CONNECT_TIMEOUT_S` | `30` | TCP/TLS connection timeout |
| `backend_config.request_timeout_s` | `NRL_REQUEST_TIMEOUT_S` | `300` | Request timeout, including uploads |
| `backend_config.max_retries` | `NRL_MAX_RETRIES` | `5` | Retries for reads and explicitly idempotent writes on transport errors, 429, and retryable 5xx |
| `backend_config.max_concurrency` | `NRL_MAX_CONCURRENCY` | `8` | Maximum concurrent multipart uploads |
| `backend_config.max_queued_uploads` | `NRL_MAX_QUEUED_UPLOADS` | `128` | Maximum admitted uploads waiting behind active multipart work; `0` disables queuing |
| `backend_config.verify_ssl` | `NRL_VERIFY_SSL` | `true` | Verify gateway certificates |
| `backend_config.ca_bundle` | `NRL_CA_BUNDLE` | unset | Optional enterprise CA bundle |
| `backend_config.collection_ttl_hours` | `NRL_COLLECTION_TTL_HOURS` | `24` | Expiration applied to new NRL collections |

One token and scope are used per Deep Researcher Agent deployment. Per-user NRL credential
forwarding is not supported. Tokens are never logged and physical NRL storage
identifiers are removed from mapped metadata.

### Connect to the service

From the NRL deployment directory, start the stack and confirm its core
services are healthy:

```bash
docker compose up -d
docker compose ps
```

With the default Compose project name `nrl`, the core rows should include:

```text
nrl-embed-1      healthy
nrl-vectordb-1   healthy
nrl-gateway-1    healthy
```

Container-name prefixes can differ when the Compose project name is
overridden. For local Docker, publish the NRL gateway and use its host port:

```bash
curl -fsS http://127.0.0.1:7670/v1/health
export NRL_BASE_URL=http://127.0.0.1:7670
```

For an NRL deployment on a remote development host, create a tunnel from the
Deep Researcher Agent workstation:

```bash
ssh -N -L 7670:127.0.0.1:7670 user@nrl-host
export NRL_BASE_URL=http://127.0.0.1:7670
```

For Kubernetes, route Deep Researcher Agent to the NRL gateway Service or an enterprise ingress.
Do not route Deep Researcher Agent directly to realtime, batch, or VectorDB pods:

```bash
export NRL_BASE_URL=https://nrl-gateway.example.com
export NRL_CA_BUNDLE=/etc/ssl/certs/enterprise-ca.pem
```

### Start Deep Researcher Agent with the service backend

```bash
./scripts/setup.sh
source .venv/bin/activate

uv run python .agents/skills/deep-researcher-configure-workflow/scripts/validate_config.py \
  configs/config_web_nemo_retriever.yml

./scripts/start_e2e.sh \
  --config_file configs/config_web_nemo_retriever.yml \
  2>&1 | tee deep-researcher-nrl-e2e.log
```

The existing Deep Researcher Agent collection APIs create scoped NRL collections, upload files,
return the real NRL job ID immediately after job creation, poll aggregate
status, list stable document IDs, and delete documents or collections.
Multipart uploads continue on a bounded background worker; upload failures are
reported through the same job-status API. Upload retries reuse deterministic
job and manifest identifiers, preventing duplicate processing after a lost
response. `attempt_id` is retained only in diagnostic job/file metadata;
pending status entries use manifest IDs, and `document_id` becomes the stable
Deep Researcher Agent file identity after NRL accepts a file.

The adapter admits at most `max_concurrency + max_queued_uploads` files at a
time. It reserves complete batches before NRL job creation: an individually
oversized batch returns HTTP 413, while temporary saturation returns HTTP 503
without `Retry-After`. Accepted requests retain the same REST calls, ordering,
retry behavior, and multipart concurrency.

Deep Researcher Agent forwards caller-supplied file metadata in the upload request, but the pinned
service baseline does not propagate that field into persisted chunk metadata.
Do not rely on service-mode file metadata for retrieval until NRL exposes that
capability; metadata filters remain unsupported by both adapters.

Query results preserve the order returned by NeMo Retriever and expose its native
vector distance as `Chunk.distance`. Lower values are closer; Deep Researcher Agent does not
normalize or re-rank these backend-specific values.

### Summary reconciliation

Document summaries live in Deep Researcher Agent's summary store, but NeMo Retriever owns document
lifetime, including server-side collection expiration. Collections therefore
disappear without Deep Researcher Agent observing a delete. On startup the ingestor reconciles the
two on a background thread: documents NRL serves gain a summary row, rows NRL no
longer backs are removed, and a collection's summaries are cleared once NRL
positively reports the collection as absent.

Reconciliation never blocks or fails startup, and a transport failure leaves the
store untouched rather than deleting summaries NRL could not confirm. The same
pass warms the document-ID-to-filename mapping that document deletes rely on,
and adopts the expirations NRL reports so cleanup also covers collections this
process did not create.

## Collection expiration

`nrl_collection_ttl_hours` is sent as an absolute expiration when Deep Researcher Agent creates a
collection, and NRL deletes the collection itself once that expiration passes.
A background thread retires only what Deep Researcher Agent keeps alongside it — the collection's
summaries and the adapter's cached document state — so agents stop being offered
documents that can no longer be retrieved. It runs every
`DEEP_RESEARCHER_TTL_CLEANUP_INTERVAL_SECONDS` (3600 by default), the shared knowledge-layer
setting; unlike the other backends, expiration comes from NRL rather than from
how long a collection sat idle. The deadline used is the one NRL last reported,
recorded when Deep Researcher Agent creates or updates a collection and refreshed by startup
reconciliation.

### Service validation

Run the contract tests without a service:

```bash
uv run pytest tests/knowledge_layer_tests/test_nemo_retriever_adapter.py
```

### Service troubleshooting

| Symptom | Meaning and action |
|---|---|
| Health or connection failure | Confirm `NRL_BASE_URL` reaches the public gateway from the Deep Researcher Agent process or pod. |
| HTTP 401 | Configure a valid `NRL_API_TOKEN`, or confirm the dev deployment explicitly allows unauthenticated access. |
| HTTP 403 | The token is not authorized for `NRL_SCOPE`; use the deployment's assigned workspace credential. |
| Collection/document 404 | Confirm the collection, stable document ID, and scope. Cross-scope resources intentionally return 404. |
| Job creation/upload 404/410 | Deep Researcher Agent and the NRL service expose incompatible collection-management API versions; upgrade the NRL chart/image. A polling 404/410 instead means that job is missing or expired. |
| HTTP 409 | An idempotency key or manifest entry was reused with different content, or the resource already exists. |
| Empty retrieval | Confirm ingestion completed, the configured collection matches the upload collection, and NRL returned indexed documents. |
| TXT/HTML ingestion failure | Deploy the documented compatible NRL revision or a released successor containing its service-mode tokenizer support. |
| TLS failure | Keep verification enabled and configure `NRL_CA_BUNDLE` for an enterprise CA. |
