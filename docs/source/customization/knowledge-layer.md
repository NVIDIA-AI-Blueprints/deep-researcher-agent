<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Knowledge Layer

A pluggable abstraction for document ingestion and retrieval. Swap backends without changing application code.

> **Looking to build a custom backend adapter?** Refer to the [SDK Reference](../reference/knowledge-layer-sdk.md) for data schemas, interfaces, and implementation examples.

## Key Features

- **Rich Output Schema** - `Chunk` model with 12 fields: content types, citations, images, structured data
- **Full Ingestion Pipeline** - `BaseIngestor` with async job tracking and status polling
- **Collection Management** - create/delete/list collections per session or use case
- **File Management** - upload/delete/list files with status tracking (UPLOADING -> INGESTING -> SUCCESS/FAILED)
- **Content Typing** - TEXT, TABLE, CHART, IMAGE enums for frontend rendering
- **Backend Agnostic** - Swap among LlamaIndex, hosted RAG Blueprint, Azure AI Search, and OpenSearch without core
  agent code changes

---

## Table of Contents

- [Available Backends](#available-backends)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [With YAML Config](#with-nemo-agent-toolkit-yaml-config---recommended)
  - [Collection Routing](#collection-routing)
  - [LlamaIndex Multimodal Controls](#llamaindex-multimodal-extraction-controls)
  - [Document Summaries](#document-summaries)
  - [Supported File Types](#supported-file-types)
  - [Programmatic Usage](#programmatic-usage)
- [Web UI Mode](#web-ui-mode)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

---

## Available Backends

| Backend | Config Name | Mode | Vector Store | Best For |
|---------|-------------|------|--------------|----------|
| `llamaindex` | `"llamaindex"` | Local Library | ChromaDB | Dev, prototyping, macOS/Linux |
| `foundational_rag` | `"foundational_rag"` | Hosted Service | Remote Milvus | Production, multi-user |
| `azure_ai_search` | `"azure_ai_search"` | Managed Service | Azure AI Search | Managed hybrid retrieval |
| `opensearch` | `"opensearch"` | External Service | OpenSearch k-NN index | Self-hosted OpenSearch, Amazon OpenSearch Service, or Serverless |
| `nemo_retriever` | `"nemo_retriever"` | External Service | NRL-managed VectorDB | Enterprise multimodal ingestion and retrieval through REST |
| `nemo_retriever_local` | `"nemo_retriever_local"` | Local Library, experimental | Embedded LanceDB | Zero-deployment NRL on targeted Python 3.12 laptops |

**Local Library Mode** - The retrieval library and vector store run in your Python process; configured model inference
may still use remote endpoints.
- **`llamaindex`** - LlamaIndex + ChromaDB. Lightweight, great for development. Works on macOS and Linux.
- **`nemo_retriever_local`** - NeMo Retriever + embedded LanceDB. Experimental on Python 3.12.

**External Service Modes** - Connect to deployed services. They require infrastructure but support shared, durable stores.
- **`foundational_rag`** - Connects to [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) through HTTP.
  - Tested with: **NVIDIA RAG Blueprint `v2.4.0`** (Helm chart `nvidia-blueprint-rag`)
  - [Deployment Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md)
  - Backend-specific documentation: `sources/knowledge_layer/src/foundational_rag/README.md`
- **`azure_ai_search`** - Stores client-generated embeddings in namespaced Azure AI Search indexes and supports
  vector, hybrid, and semantic-ranked retrieval.
- **`opensearch`** - Uses one vector index per Deep Researcher Agent collection with `none`, `basic`, or SigV4 authentication.
  - Supports self-hosted OpenSearch, Amazon OpenSearch Service (`es`), and Amazon OpenSearch Serverless (`aoss`).
  - Can ingest in the local process or dispatch ingestion to Dask workers.
  - Refer to [Amazon OpenSearch Serverless](../deployment/aws-opensearch-serverless.md) for the AOSS/EKS deployment path.
- **`nemo_retriever`** - Calls a separately deployed NeMo Retriever gateway through its public REST API.
  - NRL owns extraction, OCR, tokenization, embedding, indexing, and collection durability.
  - Deep Researcher Agent owns logical inputs, job polling, retrieval, and universal-schema mapping only.
  - See the backend operator guide at `sources/knowledge_layer/src/nemo_retriever/README.md`.

---

## Quick Start

Before you begin documentation ingestion and retrieval, run the following commands to install the backend knowledge layer.

> **Prerequisites:** Complete the main setup first (refer to the project `README.md`): clone repo, run `./scripts/setup.sh`, obtain API keys.

> **Tip:** Instead of exporting env vars each time, add them to `deploy/.env` and use `dotenv -f deploy/.env run <command>` to run any command with those vars loaded automatically.

```bash
# 1. Set up environment variables (add to deploy/.env to avoid exporting each time)
export NVIDIA_API_KEY=nvapi-your-key-here

# 2. Install backend (choose one)
uv pip install -e "sources/knowledge_layer[llamaindex]"        # Recommended for local dev - works on macOS/Linux
uv pip install -e "sources/knowledge_layer[foundational_rag]"  # Requires deployed server
uv pip install -e "sources/knowledge_layer[azure_ai_search]"   # Requires an Azure AI Search service
uv pip install -e "sources/knowledge_layer[opensearch]"        # Requires an OpenSearch endpoint
uv pip install -e "sources/knowledge_layer"                    # NeMo Retriever REST support is in base dependencies
```

> **New to Knowledge Layer?** Start with `llamaindex` - it requires no external services and works on macOS and Linux.

```bash
# 3. Verify
python -c "from deep_researcher_agent.knowledge import get_retriever; print('OK')"
```

---

## Usage

To use the knowledge layer, you can change the variables in the YAML config file.

### With NeMo Agent Toolkit (YAML Config) - Recommended

The `knowledge_retrieval` function is registered as a NeMo Agent Toolkit function type. **YAML config is the recommended single source of truth** for workflow configuration:

```yaml
# Example knowledge_retrieval function configuration
functions:
  knowledge_search:
    _type: knowledge_retrieval      # NeMo Agent Toolkit function type
    backend: llamaindex             # Required: which adapter to use
    collection_name: my_docs        # Retrieval fallback when no session context is present
    top_k: 5                        # Results to return

    # Summarization options (optional, all backends):
    # generate_summary: true                  # Generate one-sentence summary per document
    # summary_model: summary_llm                    # LLM reference from llms: section (required if generate_summary is true)
    # summary_db: sqlite+aiosqlite:///./summaries.db  # Summary storage (SQLite or PostgreSQL)

    # Backend-specific options (each backend uses different fields):
    chroma_dir: /tmp/chroma_data              # llamaindex only
    rag_url: http://localhost:8081/v1         # foundational_rag only
    ingest_url: http://localhost:8082/v1      # foundational_rag only
    timeout: 120                              # foundational_rag only
    # verify_ssl: true                        # foundational_rag only (set false for self-signed certs)

    # opensearch_url: http://localhost:9200   # opensearch only
    # opensearch_auth_type: none              # none, basic, or sigv4
    # opensearch_index_prefix: deep-researcher
    # opensearch_ingestion_mode: local        # local, dask, or auto
    # embed_model: nvidia/nemotron-3-embed-1b

    # backend_config:                         # selected adapter owns these fields
    #   base_url: http://127.0.0.1:7670       # nemo_retriever only
    #   api_token: ${NRL_API_TOKEN:-}
    #   scope: ${NRL_SCOPE}                   # required
    #   verify_ssl: true
    #   collection_ttl_hours: 24
```

You can also use environment variable substitution in YAML for deployment-specific values:

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: foundational_rag
    rag_url: ${RAG_SERVER_URL:-http://localhost:8081/v1}
    collection_name: ${COLLECTION_NAME:-default}
```

> **Note:** Each backend has different config options. Only the options matching your `backend` value are used - others are ignored (a warning will be logged). To add new config fields, edit `KnowledgeRetrievalConfig` in `sources/knowledge_layer/src/register.py`.

### Collection Routing

Deep Researcher Agent selects ingestion and retrieval collections independently. This routing policy applies consistently to all
shipped knowledge backends: LlamaIndex, Foundational RAG, Azure AI Search, and OpenSearch.

The storage mapping is backend-specific: LlamaIndex and Foundational RAG use named collections, OpenSearch maps each
collection to a physical index, and Azure AI Search isolates logical collections with `collection_id` filters inside
one Deep Researcher Agent-owned physical index.

| Usage | Ingestion target | Retrieval target |
|-------|------------------|------------------|
| Web UI | UI-created session collection (`s_<uuid>`) | Active UI session collection |
| API with a `conversation-id` header | Collection named in `/v1/collections/{collection_name}/documents` | `conversation-id` header value |
| API without conversation context | Collection named explicitly by the ingestion operation | Configured `collection_name` fallback |

`collection_name` controls only the retrieval fallback. It does not choose an API ingestion destination, and it does
not override an active UI session. Shipped profiles commonly populate it with
`${COLLECTION_NAME:-test_collection}`; the environment value is resolved when the workflow configuration is loaded.

Use `COLLECTION_NAME` for a deployment-wide retrieval default when API or CLI requests do not carry conversation
context. To select a collection for an individual HTTP request, pass that collection name in the `conversation-id`
header:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "conversation-id: research-papers" \
  -d '{"messages": [{"role": "user", "content": "Summarize the uploaded documents."}], "stream": false}'
```

For `/v1/chat/completions`, a `conversation_id` field in the JSON body is not used for collection routing. Use the
`conversation-id` header instead.

### Switching Backends

To switch backends, change the `backend` field and its corresponding options. Here are complete examples for each backend:

**LlamaIndex (ChromaDB) - macOS/Linux**
```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: llamaindex
    collection_name: my_docs
    top_k: 5
    chroma_dir: /tmp/chroma_data    # ChromaDB persistence directory
```

**Foundational RAG (Hosted Server)**
```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: foundational_rag
    collection_name: my_docs
    top_k: 5
    rag_url: http://your-server:8081/v1      # Rag server
    ingest_url: http://your-server:8082/v1   # Ingestion server
    timeout: 120
```

**Azure AI Search (Managed Service)**

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: my_docs
```

Set `AZURE_SEARCH_ENDPOINT` and `NVIDIA_API_KEY` in the environment. Setting
`AZURE_SEARCH_API_KEY` selects key authentication; otherwise Azure
`DefaultAzureCredential` is used. The workload identity needs `Search Service
Contributor` for index management and `Search Index Data Contributor` for
document ingestion and retrieval. Embedding defaults can be shared with the
LlamaIndex backend through `DEEP_RESEARCHER_EMBED_BASE_URL` and `DEEP_RESEARCHER_EMBED_MODEL`; set
`DEEP_RESEARCHER_EMBED_DIM` when changing the model dimensions. Set a deployment-unique
`DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX` when multiple Deep Researcher Agent deployments share a search
service.

Azure stores all logical collections in one physical index selected by the
prefix, schema version, embedding model, and dimension. Collection, file, and
chunk manifests enforce logical isolation. Retrieval is always hybrid, and
chunking is fixed at 1024 tokens with 128-token overlap.

Upload responses return canonical UUID file IDs. Same-name uploads coexist as
independent files. Collection cleanup uses `DEEP_RESEARCHER_COLLECTION_TTL_HOURS` (24 hours
by default) and `DEEP_RESEARCHER_TTL_CLEANUP_INTERVAL_SECONDS` (one hour by default),
matching the other knowledge backends.

**OpenSearch (Self-Hosted or AWS)**

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: opensearch
    collection_name: my_docs
    top_k: 5
    opensearch_url: ${OPENSEARCH_URL:-http://localhost:9200}
    opensearch_auth_type: ${OPENSEARCH_AUTH_TYPE:-none}
    opensearch_aws_region: ${AWS_REGION:-us-east-1}
    opensearch_aws_service: ${OPENSEARCH_AWS_SERVICE:-aoss}
    opensearch_index_prefix: ${OPENSEARCH_INDEX_PREFIX:-deep-researcher}
    opensearch_embedding_dim: ${OPENSEARCH_EMBEDDING_DIM:-2048}
    opensearch_ingestion_mode: ${OPENSEARCH_INGESTION_MODE:-auto}
    opensearch_dask_scheduler_address: ${NAT_DASK_SCHEDULER_ADDRESS:-}
    embed_model: ${DEEP_RESEARCHER_EMBED_MODEL:-nvidia/nemotron-3-embed-1b}
    embed_base_url: ${DEEP_RESEARCHER_EMBED_BASE_URL:-https://integrate.api.nvidia.com/v1}
```

Use `opensearch_auth_type: none` only with a protected local development endpoint. Configure `basic` or `sigv4`
authentication for every remote, shared, or production OpenSearch deployment. For basic authentication, set
`OPENSEARCH_USERNAME` and `OPENSEARCH_PASSWORD`. For AWS, use `sigv4` and set `opensearch_aws_service` to `es` or
`aoss`.

The embedding model's output dimension must match `opensearch_embedding_dim` (environment variable
`OPENSEARCH_EMBEDDING_DIM`, default `2048`) before the collection index is created. For example, if a test embedding
response contains 2,048 values, keep the default; if it contains 1,024 values, set
`opensearch_embedding_dim: 1024` or `OPENSEARCH_EMBEDDING_DIM=1024` before creating the collection. Use a new
collection/index after changing dimensions because an existing `knn_vector` mapping cannot change its dimension.
The full shipped profile is
[`configs/config_web_opensearch.yml`](../../../configs/config_web_opensearch.yml).

#### Changing the embedding model

Persisted vector stores are tied to both the embedding model and its output dimension. Changing only
`DEEP_RESEARCHER_EMBED_MODEL` is not a compatible in-place update:

- **Chroma:** delete only the affected logical collection through the Knowledge API or UI, then re-ingest its
  documents. Configuring a new `DEEP_RESEARCHER_CHROMA_DIR` also creates an isolated store. Deleting the existing shared
  `DEEP_RESEARCHER_CHROMA_DIR` removes every named collection in that store and can destroy unrelated data.
- **OpenSearch:** set `OPENSEARCH_EMBEDDING_DIM` to the new model's exact output length, delete the existing Deep Researcher Agent
  collection/index, and re-ingest every document. Deep Researcher Agent rejects unmarked or incompatible indexes before ingestion or
  retrieval.
- **Azure AI Search:** model and dimension are part of the physical index identity; changing either creates an isolated
  index that must be populated by re-ingestion.

OpenSearch ingestion is text-only: it extracts text from PDF, DOCX, PPTX, and supported plain-text formats, but does not
perform LlamaIndex table/image/chart extraction. Distributed Dask ingestion also disables document-summary generation
because the configured summary LLM is not serialized to workers; use local ingestion when summaries are required.

**NeMo Retriever (External REST Service)**

Choose this backend when NeMo Retriever is deployed independently with Docker Compose or Helm/Kubernetes and Deep Researcher Agent
must connect to a shared service. Use the separately registered `nemo_retriever_local` backend below when the
Retriever library and LanceDB should instead run inside the Deep Researcher Agent process. The backend names are intentionally distinct;
there is no runtime mode switch between these two ownership models.

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: nemo_retriever
    collection_name: ${COLLECTION_NAME:-deep-researcher-nrl}
    top_k: 5
    generate_summary: false
    backend_config:
      base_url: ${NRL_BASE_URL:-http://127.0.0.1:7670}
      api_token: ${NRL_API_TOKEN:-}
      scope: ${NRL_SCOPE}
      max_concurrency: ${NRL_MAX_CONCURRENCY:-8}
      max_queued_uploads: ${NRL_MAX_QUEUED_UPLOADS:-128}
      verify_ssl: ${NRL_VERIFY_SSL:-true}
      collection_ttl_hours: ${NRL_COLLECTION_TTL_HOURS:-24}
```

Use [`configs/config_web_nemo_retriever.yml`](../../../configs/config_web_nemo_retriever.yml)
for the complete web workflow. The URL must identify the public NRL gateway,
not a realtime, batch, or VectorDB pod. One deployment token and explicit
workspace scope are sent on every scoped request. For a remote development
deployment, forward the gateway port with SSH; for Kubernetes, use the gateway
Service or an enterprise ingress and configure `NRL_CA_BUNDLE` when required.

The adapter returns NRL's job ID immediately after job creation and performs
bounded multipart uploads in the background. Upload and ingestion failures are
reported through job polling. Pending status entries use deterministic manifest
IDs; stable NRL `document_id` values replace them after each file is accepted.
The adapter admits complete batches before NRL job creation and bounds total
active plus queued files. Oversized batches return HTTP 413; temporary
saturation returns HTTP 503 without a `Retry-After` header.
Per-attempt IDs remain diagnostic metadata. Query filters are rejected until
the public NRL query contract supports them. Deep Researcher Agent does not expose NRL pipeline
tuning and does not consume physical VectorDB names or LanceDB locations.
Automatic transport retries are limited to reads and explicitly idempotent
writes. A 404/410 from version-probing job creation or immediate upload means
the service contract is incompatible; a later polling 404/410 means the job is
missing or expired.

`nrl_collection_ttl_hours` is sent as an absolute expiration when Deep Researcher Agent creates a
collection, and NRL deletes the expired collection itself. TTL cleanup clears the
document summaries and cached state Deep Researcher Agent holds for it, so agents stop being
offered documents NRL no longer serves. Expiration comes from NRL rather than
from how long a collection sat idle: the deadline used is the one NRL last
reported for the collection.

The tested service baseline is the immutable NeMo Retriever integration head
[`f3a0b418b7250fa8823ec44dea569b07e2b008cb`](https://github.com/NVIDIA/NeMo-Retriever/commit/f3a0b418b7250fa8823ec44dea569b07e2b008cb),
which contains the collection-management fixes and TXT/HTML service-mode tokenizer support. See the
backend operator guide at `sources/knowledge_layer/src/nemo_retriever/README.md`
for local Docker, SSH tunnel, Kubernetes, live validation, and troubleshooting.

**NeMo Retriever (Embedded Local, Experimental)**

`nemo_retriever_local` runs Deep Researcher Agent, pinned NeMo Retriever, and LanceDB in one Python 3.12 process. It starts no Retriever
or vector-database service and delegates extraction profiles, schemas, storage, and retrieval to NeMo Retriever. The
shipped profile defaults to scope `local`, data directory `.deep-researcher-data/nemo_retriever`, and NRL's unchanged `auto`
profile. This is zero deployment for Retriever and vector storage; extraction and embedding may still call remote
inference endpoints.

When using NRL's default hosted endpoints, authenticate with an NVIDIA Build `nvapi-...` key. `NRL_INFERENCE_API_KEY`
is an optional explicit Retriever credential, not a separate key type: it can use the same value as `NVIDIA_API_KEY`.
Deep Researcher Agent passes the resolved credential to NRL's extraction, document-embedding, and query-embedding calls. Set a distinct
value only when Retriever and the Deep Researcher Agent LLM need different credentials. If `NRL_INFERENCE_API_KEY` is unset,
pinned NRL falls back to `NVIDIA_API_KEY` and then `NGC_API_KEY`.

The default URLs are supplied by NRL, so they do not need to be configured in Deep Researcher Agent: Page Elements and OCR use the
hosted `ai.api.nvidia.com` services, while embedding uses `integrate.api.nvidia.com/v1/embeddings`. Set the corresponding
`NRL_*_INVOKE_URL` only for a compatible external or self-hosted NIM override. Table Structure stays disabled unless
`NRL_TABLE_STRUCTURE_INVOKE_URL` is configured. All configured NRL inference endpoints share the resolved
`NRL_INFERENCE_API_KEY`; Deep Researcher Agent does not define separate keys per endpoint.

Deep Researcher Agent exposes the two extraction profiles supported by the pinned Retriever revision: `auto` and `fast-text`. Neither
profile is universally preferred; select one based on corpus characteristics, retrieval requirements, ingestion
latency objectives, and inference usage.

| `NRL_LOCAL_PROFILE` | Extraction behavior | Operational characteristics |
|---------------------|---------------------|-----------------------------|
| `auto` (default) | NRL's unchanged profile: text, images, tables, charts, page rendering, Page Elements, and OCR. Table Structure remains off unless configured. The default embedding modality remains NRL's text modality. | Optimized for broad extraction coverage and performs additional inference stages |
| `fast-text` | PDF/document text through PDFium only; disables image, table, chart, page-image, Page Elements, and OCR stages, then embeds the extracted text. | Optimized for ingestion efficiency and lower inference usage |

Chunk count and timing vary with document content, endpoint load, and network conditions; compare representative
documents before selecting a production profile.

```bash
uv sync --project environments/nemo_retriever_local --frozen
uv run --project environments/nemo_retriever_local --frozen \
  dotenv -f deploy/.env run \
  nat serve --config_file configs/config_web_nemo_retriever_local.yml --port 8000
```

Document ingestion does not require a generative LLM. Normal research invokes the registered `knowledge_search` tool
and requires the configured agent LLM.

The web UI creates session collection names automatically. Use the collection returned by the collection API rather
than setting `COLLECTION_NAME` for normal UI operation.

The shipped profile exposes local overrides for the data directory, extraction profile, Page Elements, OCR, Table
Structure, embedding endpoint/model/provider prefix, inference key, and collection TTL. See
[`config_web_nemo_retriever_local.yml`](../../../configs/config_web_nemo_retriever_local.yml) for their environment
variable names. The default remains NRL `auto`; Table Structure remains off unless its endpoint is configured.

Collections, documents, chunks, and recovery markers survive restart. Job history is process-local, and interrupted
pre-write jobs do not. A process lock permits one Deep Researcher Agent process per data directory. The initial targets are Apple
Silicon macOS, Windows x64, and Linux x64 with remote inference; Intel macOS, Python 3.13, local GPU inference, and
shared multi-process storage are excluded.

The shipped local profile uses threaded Dask workers and runs full deep research inline so Deep Researcher Agent, Retriever, and the
LanceDB lock remain in one process. Document ingestion is still asynchronous. Detached, durable async research jobs
require the deployed service backend.

Deep Researcher Agent removes credentials, endpoint URLs, local paths, and physical table selectors from adapter errors and public API
responses. Pinned NRL and LanceDB can still write local data paths or physical table identifiers to process logs; treat
those logs as operationally sensitive.

#### LlamaIndex Multimodal Extraction Controls

By default, LlamaIndex ingests text only and uses the NVIDIA hosted embedding models. When `DEEP_RESEARCHER_EXTRACT_IMAGES` or `DEEP_RESEARCHER_EXTRACT_CHARTS` is enabled, a Vision Language Model (VLM) is used during ingestion to caption embedded images and extract structured data from charts (axis labels, data points, chart type). This makes visual content in PDFs searchable and retrievable alongside text. The VLM is only invoked at ingestion time, not at query time.

All options below can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| **Embedding** | | |
| `DEEP_RESEARCHER_EMBED_MODEL` | `nvidia/nemotron-3-embed-1b` | NVIDIA embedding model |
| `DEEP_RESEARCHER_EMBED_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Embedding API base URL — override for local NIM |
| `OPENSEARCH_EMBEDDING_DIM` | `2048` | OpenSearch vector dimension; must equal the selected embedding model's output length before index creation |
| **Extraction Flags** | | |
| `DEEP_RESEARCHER_EXTRACT_TABLES` | `false` | Extract tables from PDFs as markdown |
| `DEEP_RESEARCHER_EXTRACT_IMAGES` | `false` | Extract and caption images with VLM |
| `DEEP_RESEARCHER_EXTRACT_CHARTS` | `false` | Classify images as charts and extract structured data |
| **Vision Model** | | |
| `DEEP_RESEARCHER_VLM_MODEL` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | VLM for image captioning |
| `DEEP_RESEARCHER_VLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | VLM API base URL — override for local NIM |

When enabled, the startup log shows the active mode:

```
LlamaIndexIngestor initialized: persist_dir=/app/data/chroma_data, mode=text + tables + images
```

> **Note:** `DEEP_RESEARCHER_EXTRACT_IMAGES` and `DEEP_RESEARCHER_EXTRACT_CHARTS` work together. If both are enabled, each image is classified by the VLM as either a chart or a regular image. Foundational RAG handles multimodal extraction server-side. OpenSearch performs text extraction only, so these flags apply only to the LlamaIndex backend.

#### Document Summaries

Document summaries help research agents understand what files are available before making tool calls. When enabled, the knowledge layer generates a one-sentence summary during ingestion and injects it into agent system prompts.

```yaml
llms:
  summary_llm:
    _type: nim
    model_name: google/gemma-4-31b-it
    base_url: "https://integrate.api.nvidia.com/v1"
    temperature: 0.3
    max_tokens: 150

functions:
  knowledge_search:
    _type: knowledge_retrieval
    generate_summary: true
    summary_model: summary_llm     # Required: LLM reference from llms: section
    summary_db: ${DEEP_RESEARCHER_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

When `generate_summary: true`, you **must** configure `summary_model` to reference an LLM from the `llms:` section. For production deployments, use PostgreSQL for `summary_db` instead of SQLite.

For details on how summaries are stored, how agents consume them, and how to implement summaries in custom backends, refer to the [SDK Reference - Document Summaries](../reference/knowledge-layer-sdk.md#document-summaries).

#### Supported File Types

File type support depends on the configured backend:

| Backend | Supported Types |
|---------|----------------|
| **LlamaIndex** | PDF, DOCX, TXT, MD, HTML, JSON, CSV |
| **Foundational RAG** | PDF, DOCX, PPTX, TXT, MD, HTML, images (PNG, JPG) |
| **OpenSearch** | PDF, DOCX, PPTX, TXT, MD, CSV, JSON, YAML, YML, LOG |
| **Azure AI Search** | PDF, DOCX, TXT, MD |
| **NeMo Retriever service** | Determined by the deployed service image and extraction configuration |
| **NeMo Retriever local** | NRL-supported inputs; DOCX/PPTX conversion requires LibreOffice on `PATH` |

For custom backends, supported types are determined by the backend implementation.

> **Note:** The backends support more types than the default upload allowlist. The frontend and backend API default to
> `.pdf,.docx,.txt,.md` (the common subset across all backends). Types like HTML, JSON, CSV, and images are supported by
> some backends but must be explicitly enabled and supported by the selected backend.

The frontend and backend API use the same upload controls:

| Variable | Effect |
|----------|--------|
| `FILE_UPLOAD_ACCEPTED_TYPES` | Comma-separated extension allowlist; the API also validates declared and actual content |
| `FILE_UPLOAD_MAX_SIZE_MB` | Maximum size of each file and of all files combined in one request |
| `FILE_UPLOAD_MAX_FILE_COUNT` | Maximum number of files in one request |

Set identical values for both application components:

| Deployment | Where to set |
|-----------|-------------|
| **CLI** (`start_e2e.sh`) | `deploy/.env` |
| **Docker Compose** | `deploy/.env` (passed to the frontend and backend containers) |
| **Helm** | `deploy/helm/deployment-k8s/values.yaml` under both the backend and frontend apps' `env` sections |

For Foundational RAG or either NeMo Retriever backend, add `.pptx` to include PowerPoint support:
`FILE_UPLOAD_ACCEPTED_TYPES=.pdf,.docx,.pptx,.txt,.md`. Set it in the shared process environment, normally
`deploy/.env`, so the UI and backend receive the same value. Pinned NRL routes `.pptx` through its document/PDF branch
under both `auto` and `fast-text`.

Upload request validation is atomic: one disallowed or malformed file rejects the complete request with HTTP 415 and
no ingestion job is created. Failures after job acceptance remain visible per file and can produce partial success.

### Programmatic Usage

```python
# Import the adapter module to trigger registration
from knowledge_layer.llamaindex import LlamaIndexRetriever, LlamaIndexIngestor

# Use the factory to get instances
from deep_researcher_agent.knowledge import get_retriever, get_ingestor

# Ingest documents
ingestor = get_ingestor("llamaindex", config={"persist_dir": "/tmp/chroma"})
ingestor.create_collection("my_docs")
file_info = ingestor.upload_file("doc.pdf", "my_docs")

# Check ingestion status
status = ingestor.get_file_status(file_info.file_id, "my_docs")
print(f"Status: {status.status}")  # UPLOADING, INGESTING, SUCCESS, FAILED

# Retrieve
retriever = get_retriever("llamaindex", config={"persist_dir": "/tmp/chroma"})
result = await retriever.retrieve("query", "my_docs", top_k=5)
for chunk in result.chunks:
    print(f"{chunk.display_citation}: {chunk.content[:100]}")
```

---

## Web UI Mode

Run the backend API server and frontend UI together for document upload, collection management, and chat.

### Start Backend

```bash
# Foundational RAG example (requires deployed FRAG server)
# dotenv loads API keys (NVIDIA_API_KEY, etc.) from deploy/.env
# Additional env vars needed: RAG_SERVER_URL, RAG_INGEST_URL
dotenv -f deploy/.env run nat serve --config_file configs/config_web_frag.yml --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd frontends/ui
npm run dev
```

Open `http://localhost:3000` in your browser.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/collections` | Create collection |
| `GET` | `/v1/collections` | List collections |
| `GET` | `/v1/collections/{name}` | Get collection details |
| `DELETE` | `/v1/collections/{name}` | Delete collection |
| `POST` | `/v1/collections/{name}/documents` | Upload files |
| `GET` | `/v1/collections/{name}/documents` | List documents in collection |
| `DELETE` | `/v1/collections/{name}/documents` | Delete files |
| `GET` | `/v1/documents/{job_id}/status` | Poll ingestion status |
| `GET` | `/v1/knowledge/health` | Check knowledge backend health |

### Session Collections

All shipped knowledge backends support session-based collections (`s_<uuid>`) created by the UI. Each UI
conversation gets its own isolated logical collection; the physical storage mapping differs by backend as described in
[Collection Routing](#collection-routing).

The active session collection is used for both UI ingestion and retrieval and takes precedence over the configured
`collection_name` fallback.

### TTL Cleanup

Collections inactive for 24 hours are auto-deleted based on `updated_at` timestamp. Background thread runs hourly.

```python
COLLECTION_TTL_HOURS = 24
TTL_CLEANUP_INTERVAL_SECONDS = 3600
```

NeMo Retriever runs the same hourly thread, but that service owns collection lifetime: it deletes collections on the
absolute deadline it was given at creation (`nrl_collection_ttl_hours`), and the thread only expires the summaries and
cached state Deep Researcher Agent keeps for them rather than deleting anything itself.

---

## Architecture

### Core Library (`src/deep_researcher_agent/knowledge/`)

```
src/deep_researcher_agent/knowledge/
    __init__.py        # Exports: Chunk, get_retriever, get_ingestor, etc.
    base.py            # Abstract classes: BaseRetriever, BaseIngestor
    schema.py          # Data models: Chunk, RetrievalResult, FileInfo, CollectionInfo
    factory.py         # Registry + factory: register_retriever(), get_retriever()
    summary_store.py   # SQLAlchemy-backed document summary persistence
```

| File | Purpose |
|------|---------|
| `base.py` | Defines the interface all backends must implement |
| `schema.py` | Universal data models - backends convert native formats to these |
| `factory.py` | Registration decorators + factory functions for instantiation |
| `summary_store.py` | Persistent storage for document summaries (SQLite/PostgreSQL) |

### Backend Adapters (`sources/knowledge_layer/src/`)

```
sources/knowledge_layer/src/
    <backend_name>/
        __init__.py      # Imports adapter to trigger registration
        adapter.py       # @register_retriever/@register_ingestor decorated classes
        README.md        # Backend-specific documentation
        pyproject.toml   # Optional: isolated dependencies
```

### How Registration Works

Backends register themselves using decorators when their module is imported:

```python
# In adapter.py
from deep_researcher_agent.knowledge.factory import register_retriever, register_ingestor

@register_retriever("my_backend")  # Registration name used in config
class MyRetriever(BaseRetriever):
    ...

@register_ingestor("my_backend")
class MyIngestor(BaseIngestor):
    ...
```

The registration name (for example, `"my_backend"`) is what you use in:
- YAML config: `backend: my_backend`
- Factory calls: `get_retriever("my_backend")`

**Important:** The adapter module must be imported for registration to happen. This is why:
1. `__init__.py` imports the adapter classes
2. The NeMo Agent Toolkit function imports from `knowledge_layer.<backend>.adapter`

### NeMo Agent Toolkit Integration

```
sources/knowledge_layer/src/
    register.py      # @register_function exposes retrieval to agents
```

The `register.py` defines `KnowledgeRetrievalConfig` which maps YAML config to backend instantiation.

---

## Configuration

### Configuration Precedence

Configuration values are resolved in the following order (highest to lowest priority):

1. **Explicit parameter** - Values passed directly to factory functions (`get_retriever("llamaindex")`)
2. **YAML config file** - The `backend:` field and other options in your workflow config (recommended)
3. **Environment variables** - `KNOWLEDGE_RETRIEVER_BACKEND`, `RAG_SERVER_URL`, etc.
4. **Hardcoded defaults** - Built-in fallback values

**Recommendation:** Use YAML config as your single source of truth for workflow configuration. Environment variables are useful for:
- Container deployments (12-factor app pattern)
- CI/CD overrides
- Secrets management (API keys)

### Environment Variables

| Variable | Backend | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | All | Required for embeddings/VLM |
| `KNOWLEDGE_RETRIEVER_BACKEND` | All | Default retriever backend (fallback if not in YAML) |
| `KNOWLEDGE_INGESTOR_BACKEND` | All | Default ingestor backend (fallback if not in YAML) |
| `DEEP_RESEARCHER_CHROMA_DIR` | llamaindex | ChromaDB persistence path |
| `DEEP_RESEARCHER_COLLECTION_TTL_HOURS` | all local/managed backends | Hours before stale collections are deleted (default: 24) |
| `DEEP_RESEARCHER_TTL_CLEANUP_INTERVAL_SECONDS` | All | Collection cleanup interval (default: 3600) |
| `RAG_SERVER_URL` | foundational_rag | Query server URL (port 8081) |
| `RAG_INGEST_URL` | foundational_rag | Ingestion server URL (port 8082) |
| `OPENSEARCH_URL` | opensearch | OpenSearch endpoint URL |
| `OPENSEARCH_AUTH_TYPE` | opensearch | `none`, `basic`, or `sigv4` |
| `OPENSEARCH_USERNAME`, `OPENSEARCH_PASSWORD` | opensearch | Credentials for basic authentication |
| `AWS_REGION`, `OPENSEARCH_AWS_SERVICE` | opensearch | SigV4 region and service (`es` or `aoss`) |
| `OPENSEARCH_INDEX_PREFIX` | opensearch | Prefix for Deep Researcher Agent-managed indexes |
| `OPENSEARCH_INGESTION_MODE` | opensearch | `local`, `dask`, or `auto` |
| `OPENSEARCH_DASK_SCHEDULER_ADDRESS` | opensearch | Optional Dask scheduler for distributed ingestion |
| `DEEP_RESEARCHER_EMBED_MODEL`, `DEEP_RESEARCHER_EMBED_BASE_URL` | llamaindex, opensearch, azure_ai_search | Embedding model and endpoint |
| `NRL_BASE_URL` | nemo_retriever | Public NeMo Retriever gateway URL |
| `NRL_API_TOKEN`, `NRL_SCOPE` | nemo_retriever | Deployment bearer token and required workspace scope |
| `NRL_CONNECT_TIMEOUT_S`, `NRL_REQUEST_TIMEOUT_S` | nemo_retriever | Connection and request timeout seconds |
| `NRL_MAX_RETRIES`, `NRL_MAX_CONCURRENCY`, `NRL_MAX_QUEUED_UPLOADS` | nemo_retriever | Transient retry, active multipart, and queued-upload bounds |
| `NRL_VERIFY_SSL`, `NRL_CA_BUNDLE` | nemo_retriever | TLS verification and optional enterprise CA bundle |
| `NRL_LOCAL_DATA_DIR`, `NRL_LOCAL_PROFILE` | nemo_retriever_local | Embedded data directory and NRL `auto` or `fast-text` profile |
| `NRL_PAGE_ELEMENTS_INVOKE_URL`, `NRL_OCR_INVOKE_URL`, `NRL_TABLE_STRUCTURE_INVOKE_URL` | nemo_retriever_local | Optional extraction endpoint overrides |
| `NRL_EMBED_INVOKE_URL`, `NRL_EMBED_MODEL_NAME`, `NRL_EMBED_MODEL_PROVIDER_PREFIX` | nemo_retriever_local | Embedding endpoint and model overrides |
| `NRL_INFERENCE_API_KEY` | nemo_retriever_local | Optional explicit NVIDIA Build credential for extraction and document/query embedding; it may match `NVIDIA_API_KEY`, which is the first fallback when this variable is unset |
| `NRL_COLLECTION_TTL_HOURS` | nemo_retriever, nemo_retriever_local | Expiration applied to new NRL collections |
| `COLLECTION_NAME` | All | Default retrieval collection when no conversation or session context is present |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `Unknown backend: my_backend` | Adapter not imported/registered | Import the adapter module before calling factory |
| `ormsgpack` attribute error | Version conflict with [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | `uv pip install "ormsgpack>=1.5.0"` |
| Empty retrieval results | Collection empty | Run ingestion first, verify collection name matches |
| Job status 404 | Different process/instance | Factory uses singletons - ensure same process |
| `milvus-lite` required | Missing dependency | `uv pip install "pymilvus[milvus_lite]"` |
| `opensearchpy` import error | OpenSearch extra not installed | `uv pip install -e "sources/knowledge_layer[opensearch]"` |
| OpenSearch `401` or `403` | Auth mode, credentials, IAM, or AOSS data-access policy mismatch | Verify `opensearch_auth_type`; for AOSS follow the IAM and data-access steps in the deployment guide |
| NRL connection or health failure | Deep Researcher Agent cannot reach the public gateway | Verify `NRL_BASE_URL`, network policy, ingress, or the SSH tunnel |
| NRL `401` or `403` | Missing/invalid token or unauthorized scope | Verify `NRL_API_TOKEN` and its authorization for `NRL_SCOPE` |
| NRL job creation/upload `404` or `410` | Deep Researcher Agent and NRL use incompatible collection-management APIs | Upgrade the NRL chart/image to the validated API version; polling `404`/`410` instead means the job is missing or expired |
| NRL TXT/HTML failure | Service image predates the validated integration baseline | Deploy the documented compatible NRL revision or a released successor |
| Embedded NRL inference `401` | Hosted extraction or embedding rejected its credential | Set a valid NVIDIA Build key in `NRL_INFERENCE_API_KEY`; it may match `NVIDIA_API_KEY`, or use a distinct value when Retriever and agent endpoints require different credentials |
| Embedded NRL collection ownership mismatch | The data directory was created with a different scope, profile, embedding model, or provider prefix | Restore the original settings or select a new `NRL_LOCAL_DATA_DIR` and re-ingest |
| Embedded NRL data-directory lock | Another Deep Researcher Agent process already owns the directory | Stop the other process or select a different `NRL_LOCAL_DATA_DIR`; sharing one directory across processes is unsupported |
| Backend registered twice | Module imported multiple times | Normal - factory logs warning but works fine |

### Debug Registration

```python
# Check what's registered
from deep_researcher_agent.knowledge.factory import list_retrievers, list_ingestors, get_knowledge_layer_config

print("Retrievers:", list_retrievers())
print("Ingestors:", list_ingestors())
print("Full config:", get_knowledge_layer_config())
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [SDK Reference](../reference/knowledge-layer-sdk.md) | Build custom backend adapters - data schemas, interfaces, full implementation example |
| Foundational RAG Setup (`sources/knowledge_layer/src/foundational_rag/README.md`) | Production deployment with NVIDIA RAG Blueprint |
| [Amazon OpenSearch Serverless](../deployment/aws-opensearch-serverless.md) | Deploy the OpenSearch backend on EKS with AOSS and SigV4 |
| NeMo Retriever backends (`sources/knowledge_layer/src/nemo_retriever/README.md`) | Operate the deployed REST and experimental embedded modes |
