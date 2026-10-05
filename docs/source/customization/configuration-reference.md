<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configuration Reference

The AI-Q blueprint is configured through a single YAML file that defines LLMs, tools, agents, and the workflow. The NeMo Agent Toolkit reads this file at startup and wires everything together.

```{note}
The NVIDIA API Catalog serving profile for Nemotron 3.5 Lightning has a known shallow citation-output limitation.
AI-Q fails closed rather than publishing citation-incomplete drafts. See
[Troubleshooting](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog) before using this hosted
profile for shallow research.
```

## Config File Structure

Every config file has four top-level sections:

```yaml
general:     # Telemetry, logging, front-end settings
llms:        # LLM definitions (model, endpoint, parameters)
functions:   # Tools and agents (search tools, classifiers, research agents)
workflow:    # Top-level orchestrator configuration
```

## Environment Variable Substitution

You can reference environment variables anywhere in the YAML using shell-style syntax:

```yaml
# Required variable (fails if not set)
api_key: ${NVIDIA_API_KEY}

# Variable with a default value
checkpoint_db: ${AIQ_CHECKPOINT_DB:-./checkpoints.db}

# Nested in a URL
collection_name: ${COLLECTION_NAME:-test_collection}
```

The syntax `${VAR_NAME}` substitutes the value of the environment variable. The syntax `${VAR_NAME:-default}` provides a fallback value if the variable is not set. Environment variables are typically defined in `deploy/.env` or `.env` at the project root.

---

## `general` Section

Controls telemetry, logging, and the application front-end.

```yaml
general:
  use_uvloop: true          # Use uvloop for better async performance (web mode)
  telemetry:
    logging:
      console:
        _type: console
        level: INFO          # DEBUG, INFO, WARNING, ERROR
  front_end:                 # Only for web/API mode
    _type: aiq_api
    runner_class: aiq_api.plugin.AIQAPIWorker
    db_url: ${NAT_JOB_STORE_DB_URL:-sqlite+aiosqlite:///./jobs.db}
    expiry_seconds: 86400
    cors:
      allow_origin_regex: 'http://localhost(:\d+)?|http://127.0.0.1(:\d+)?'
      allow_methods: [GET, POST, DELETE, OPTIONS]
      allow_headers: ["*"]
      allow_credentials: true
      expose_headers: ["*"]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_uvloop` | `bool` | `false` | Enable uvloop for improved async I/O performance. Recommended for web mode. |
| `telemetry.logging.console._type` | `str` | `console` | Logging backend type. |
| `telemetry.logging.console.level` | `str` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `front_end._type` | `str` | -- | Front-end type. Use `aiq_api` for the web API server. Omit for CLI mode. |
| `front_end.db_url` | `str` | `sqlite+aiosqlite:///./jobs.db` | Database URL for async job persistence. |
| `front_end.expiry_seconds` | `int` | `86400` | How long completed jobs remain in the database (seconds). |
| `front_end.cors` | `object` | -- | CORS settings for the API server. |

Tracing is configured through `workflow.relay`, not `general.telemetry`.
For `aiq_api`, request tag enrichment for Relay-exported spans is configured via
environment variables rather than YAML fields. Refer to `frontends/aiq_api/README.md`
and the [Observability](../deployment/observability.md) guide for:

- `AIQ_TRACE_USER_IDENTITY_MODE`
- `AIQ_TRACE_USER_IDENTITY_HMAC_SECRET`
- `AIQ_TRACE_CLIENT_ID_MODE`
- `AIQ_TRACE_CLIENT_ID_HMAC_SECRET`
- `AIQ_TRACE_CLIENT_IP_HEADERS`

---

## `llms` Section

Defines named LLM instances. Each entry gets a user-chosen key (for example, `nemotron_ultra_llm`) that agents reference.

```yaml
llms:
  nemotron_ultra_llm:
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 16384
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_type` | `str` | **required** | LLM provider type. Use `nim` for NVIDIA NIM endpoints, `openai` for OpenAI-compatible endpoints. |
| `model_name` | `str` | **required** | Model identifier (for example, `nvidia/nemotron-3-ultra-550b-a55b`, `azure/openai/gpt-4.1-mini`). |
| `base_url` | `str` | `None` | API endpoint URL. Should always be set explicitly for NVIDIA NIM endpoints. |
| `api_key` | `str` | -- | API key. If omitted, uses `NVIDIA_API_KEY` from the environment. |
| `temperature` | `float` | `None` | Sampling temperature. Lower values produce more deterministic output. When `None`, the API uses its server-side default. |
| `top_p` | `float` | `None` | Nucleus sampling threshold. When `None`, the API uses its server-side default. |
| `max_tokens` | `int` | `300` | Maximum tokens in the response. Set higher values (for example, `16384` or `128000`) for research agents. |
| `num_retries` | `int` | `5` | Number of retry attempts on API failure. |
| `parallel_tool_calls` | `bool` | Provider default | Whether the provider can emit parallel tool calls. The default intent and shallow profiles set this to `false`. |
| `chat_template_kwargs` | `object` | -- | Extra arguments passed to the chat template. Use `enable_thinking: true` to activate the model's chain-of-thought reasoning. |

### Common LLM Configurations

Different agents benefit from different parameter profiles:

| Role | Temperature | Top-p | Max Tokens | Notes |
|------|------------|-------|------------|-------|
| Intent classifier (Nemotron 3.5 Lightning) | `0.1` | `0.9` | `1024` | Short deterministic classification; thinking disabled |
| Shallow researcher (Nemotron 3.5 Lightning) | `0.2` | `0.7` | `8192` | Tool-calling profile; parallel tool calls disabled and thinking enabled |
| Deep research roles (Ultra) | `0.2` | `0.7` | `16384` | Source routing, orchestration, planning, and research |
| Deep research writer (Ultra) | `0.2` | `0.7` | `32768` | Larger report-writing budget |
| Summary LLM (Gemma) | `0.1` | -- | `100` | Conservative, short document summaries |

---

## `functions` Section

Defines tools and agents. Each entry has a `_type` field that maps to a registered NeMo Agent Toolkit plugin. The key you assign (for example, `web_search_tool`) becomes the name used in `tools` lists.

### `tavily_web_search`

Web search powered by the [Tavily API](https://tavily.com/).

```yaml
functions:
  web_search_tool:
    _type: tavily_web_search
    max_results: 5
    max_content_length: 1000

  advanced_web_search_tool:
    _type: tavily_web_search
    max_results: 2
    advanced_search: true

  proxied_web_search_tool:
    _type: tavily_web_search
    api_base_url: https://search-proxy.example.com
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_results` | `int` | `3` | Maximum number of search results to return. |
| `include_answer` | `str` | `"advanced"` | Whether to include a synthesized answer alongside search results. Tavily returns a direct answer in addition to individual result documents. |
| `api_key` | `str` | `None` | Tavily API key. Falls back to `TAVILY_API_KEY` environment variable. |
| `max_retries` | `int` | `3` | Number of retry attempts on search failure. |
| `advanced_search` | `bool` | `false` | Use Tavily's advanced search mode for deeper, more thorough results. |
| `max_content_length` | `int` | `None` | Truncate each result's content to this many characters. Reduces token usage. |
| `api_base_url` | `str` | `None` | Optional custom or proxy-compatible Tavily API base URL. A non-empty value is passed to the Tavily client; `None` uses the client's default endpoint. |

### `exa_web_search`

Web search powered by the [Exa API](https://exa.ai/) via `langchain-exa`.

```yaml
functions:
  web_search_tool:
    _type: exa_web_search
    max_results: 5
    full_text: true
    max_content_length: 10000

  deep_web_search_tool:
    _type: exa_web_search
    max_results: 5
    search_type: deep
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_results` | `int` | `5` | Maximum number of search results to return. |
| `api_key` | `str` | `None` | Exa API key. Falls back to `EXA_API_KEY` environment variable. |
| `max_retries` | `int` | `3` | Number of retry attempts on search failure. |
| `search_type` | `str` | `"auto"` | Exa search type. See options below. |
| `full_text` | `bool` | `false` | Return full page text for each result. Off by default because full text is expensive in tokens; when false, results use `highlights` instead. |
| `highlights` | `bool` | `true` | Return highlighted snippets for each result. Highlights are token-efficient and are used as the result body when `full_text` is `false`. |
| `max_content_length` | `int` | `10000` | Only applied when `full_text` is `true`. Truncates each result's full page text to this many characters. Set to `None` to disable truncation. |

**`search_type` options:**

- **`auto`** (default) -- Let Exa pick the best strategy for the query. Balances latency and recall; a safe default for general research workloads.
- **`fast`** -- Optimized for low latency. Returns results quickly at the cost of recall and semantic depth. Use for interactive UIs, high-volume calls, or when the query is narrow and keyword-like.
- **`deep`** -- Optimized for thoroughness. Runs a more expensive semantic search with broader retrieval. Use for research-quality queries where completeness matters more than speed.

### `nimble_web_search`

Web search powered by the [Nimble Search API](https://docs.nimbleway.com/nimble-sdk/web-tools/search) via
`langchain-nimble`.

```yaml
functions:
  web_search_tool:
    _type: nimble_web_search
    max_results: 5
    max_content_length: 10000

  advanced_web_search_tool:
    _type: nimble_web_search
    max_results: 5
    search_depth: deep
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_results` | `int` | `5` | Maximum number of search results to return. |
| `api_key` | `str` | `None` | Nimble API key. Falls back to `NIMBLE_API_KEY` environment variable. |
| `max_retries` | `int` | `3` | Number of retry attempts on search failure. |
| `search_depth` | `str` | `"lite"` | Nimble search depth. See options below. |
| `focus` | `str` | `"general"` | Nimble focus mode. See options below. |
| `country` | `str` | `"US"` | ISO 3166 country code passed to Nimble (e.g. `US`, `GB`, `FR`). |
| `locale` | `str` | `"en"` | Language/locale passed to Nimble (e.g. `en`, `fr`, `es`). |
| `max_content_length` | `int \| None` | `10000` | Max characters per result's page content. Set to `None` to disable truncation. |

**`search_depth` options:**

- **`lite`** (default) -- Returns metadata only (title, URL, description). Fastest, lowest token cost, safe default for general lookups.
- **`fast`** -- Returns rich content at low latency. **Enterprise-tier only**; non-enterprise accounts receive a 403 with a clear entitlement message.
- **`deep`** -- Returns full page content for each result. Use for research workflows that need the body text, not just URLs.

**`focus` options:**

- **`general`** (default) -- Broad web/research queries. The right choice for almost all agent use.
- **`news`** -- Restricts results to news-publisher sources, ordered by recency. There is no recency threshold -- older articles still appear; it changes the source mix, not the time window. (Recency windowing is a separate Nimble `time_range` capability that also works with `focus=general`; not exposed in this initial integration.)
- **`location`**, **`shopping`**, **`geo`**, **`social`** -- Domain-specific routing; set only when the tool targets that domain.

`focus` is a workflow-config setting, not an agent-chosen parameter -- the model only passes a query, so general research queries cannot silently switch to `news`. Answer generation (`include_answer`) is **not exposed** in this initial integration.

### `paper_search`

Academic paper search through Google Scholar using [Serper](https://serper.dev/),
[SerpAPI](https://serpapi.com/), or [SearchAPI](https://www.searchapi.io/). All three providers are normalized to the
same agent-facing result shape. Serper is the default.

```yaml
functions:
  paper_search_tool:
    _type: paper_search
    provider: serper
    max_results: 5
    serper_api_key: ${SERPER_API_KEY}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `serper` | Google Scholar backend: `serper`, `serpapi`, or `searchapi`. |
| `max_results` | `int` | `10` | Maximum number of paper results. |
| `serper_api_key` | `str` | `None` | Serper key for `provider: serper`. The tool also reads `SERPER_API_KEY`. |
| `serpapi_api_key` | `str` | `None` | SerpAPI key for `provider: serpapi`. The tool also reads `SERPAPI_API_KEY`. |
| `searchapi_api_key` | `str` | `None` | SearchAPI key for `provider: searchapi`. The tool also reads `SEARCHAPI_API_KEY`. |
| `timeout` | `int` | `30` | Timeout in seconds for search requests. |

### `knowledge_retrieval`

Semantic search over ingested documents. Supports LlamaIndex (local ChromaDB), Foundational RAG
(hosted NVIDIA RAG Blueprint), OpenSearch (self-hosted OpenSearch or Amazon OpenSearch Serverless), and Azure AI Search.

```yaml
functions:
  # LlamaIndex backend
  knowledge_search:
    _type: knowledge_retrieval
    backend: llamaindex
    collection_name: ${COLLECTION_NAME:-test_collection}
    top_k: 5
    chroma_dir: ${AIQ_CHROMA_DIR:-/tmp/chroma_data}
    generate_summary: true
    summary_model: summary_llm
    summary_db: ${AIQ_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

```yaml
functions:
  # Foundational RAG backend
  knowledge_search:
    _type: knowledge_retrieval
    backend: foundational_rag
    collection_name: ${COLLECTION_NAME:-test_collection}
    top_k: 5
    rag_url: ${RAG_SERVER_URL:-http://localhost:8081/v1}
    ingest_url: ${RAG_INGEST_URL:-http://localhost:8082/v1}
    timeout: 300
    # verify_ssl: false # Only set to false for self-signed certs
```

```yaml
functions:
  # Azure AI Search backend
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: ${COLLECTION_NAME:-test_collection}
```

This example reads `AZURE_SEARCH_ENDPOINT` and `NVIDIA_API_KEY` from the
environment. `AZURE_SEARCH_API_KEY` is optional; when absent, the adapter uses
`DefaultAzureCredential`.

```yaml
functions:
  # OpenSearch backend
  knowledge_search:
    _type: knowledge_retrieval
    backend: opensearch
    collection_name: ${COLLECTION_NAME:-test_collection}
    top_k: 5
    opensearch_url: ${OPENSEARCH_URL:-http://localhost:9200}
    opensearch_auth_type: ${OPENSEARCH_AUTH_TYPE:-none}
    opensearch_aws_region: ${AWS_REGION:-us-east-1}
    opensearch_aws_service: ${OPENSEARCH_AWS_SERVICE:-aoss}
    opensearch_index_prefix: ${OPENSEARCH_INDEX_PREFIX:-aiq}
    opensearch_ingestion_mode: ${OPENSEARCH_INGESTION_MODE:-auto}
    embed_model: ${AIQ_EMBED_MODEL:-nvidia/nemotron-3-embed-1b}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `llamaindex` | Backend type: `llamaindex`, `opensearch`, `foundational_rag`, or `azure_ai_search`. |
| `collection_name` | `str` | `default` | Fallback retrieval collection when no conversation or session context is present. Request context takes precedence; ingestion selects its collection explicitly. |
| `top_k` | `int` | `5` | Number of results to return per query. |
| `generate_summary` | `bool` | `false` | Generate one-sentence summaries for ingested documents. |
| `summary_model` | `str` | `None` | LLM reference from `llms` section. Required when `generate_summary: true`. |
| `summary_db` | `str` | `sqlite+aiosqlite:///./summaries.db` | Database URL for document summaries (SQLite or PostgreSQL). |
| `chroma_dir` | `str` | `/tmp/chroma_data` | ChromaDB persistence directory. LlamaIndex backend only. |
| `rag_url` | `str` | `http://localhost:8081/v1` | RAG query server URL. Foundational RAG backend only. |
| `ingest_url` | `str` | `http://localhost:8082/v1` | RAG ingestion server URL. Foundational RAG backend only. |
| `timeout` | `int` | `120` | Request timeout in seconds. Foundational RAG backend only. |
| `verify_ssl` | `bool` | `true` | Verify SSL certificates. Set `false` for self-signed certs. Foundational RAG backend only. |
| `azure_search_endpoint` | `URL` | `AZURE_SEARCH_ENDPOINT` | Azure AI Search service endpoint. Required for Azure AI Search. |
| `azure_search_api_key` | `SecretStr` | `AZURE_SEARCH_API_KEY` | Optional admin API key. |
| `azure_search_index_prefix` | `str` | `AIQ_AZURE_SEARCH_INDEX_PREFIX` or `aiq` | Deployment-unique namespace for the shared AI-Q index. |
| `embed_dim` | `int` | `AIQ_EMBED_DIM` or `2048` | Embedding dimensions; must match the model and existing index schema. |
| `opensearch_url` | `str` | `http://localhost:9200` | OpenSearch endpoint. OpenSearch backend only. |
| `opensearch_auth_type` | `str` | `none` | Authentication mode: `none`, `basic`, or `sigv4`. |
| `opensearch_username` | `str` | `None` | Username for basic authentication. Also read from `OPENSEARCH_USERNAME`. |
| `opensearch_password` | `str` | `None` | Password for basic authentication. Also read from `OPENSEARCH_PASSWORD`. |
| `opensearch_verify_certs` | `bool` | `true` | Verify OpenSearch TLS certificates. Disable only for a trusted development cluster. |
| `opensearch_ca_certs` | `str` | `None` | Optional custom CA bundle path. |
| `opensearch_aws_region` | `str` | `us-east-1` | AWS region for SigV4 authentication. |
| `opensearch_aws_service` | `str` | `aoss` | SigV4 service: `aoss` for Serverless or `es` for managed OpenSearch Service. |
| `opensearch_index_prefix` | `str` | `aiq` | Prefix for the physical index created for each AI-Q collection. |
| `opensearch_embedding_dim` | `int` | `2048` | Vector dimension; must match the configured embedding model. |
| `opensearch_ingestion_mode` | `str` | `local` | Ingestion executor: `local`, `dask`, or `auto`. `auto` uses Dask only when a scheduler address is configured. |
| `opensearch_dask_scheduler_address` | `str` | `None` | Dask scheduler for distributed ingestion. Also reads `NAT_DASK_SCHEDULER_ADDRESS`. |
| `opensearch_dask_file_transfer` | `str` | `bytes` | Send uploads to Dask workers as `bytes` or shared filesystem `paths`. |
| `embed_model` | `str` | `nvidia/nemotron-3-embed-1b` | Embedding model for OpenSearch and Azure AI Search ingestion and retrieval. |
| `embed_base_url` | `str` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible embeddings endpoint for OpenSearch and Azure AI Search. |

Refer to [Knowledge Layer](./knowledge-layer.md) for backend selection and the
[Amazon OpenSearch Serverless](../deployment/aws-opensearch-serverless.md) guide for SigV4, IAM, and AOSS setup.

### `gsf` function group

The GSF function group exposes `gsf__catalog_search` for semantic discovery and
`gsf__text_to_sql` for validated, bounded structured-data queries. Declare it
under the top-level `function_groups` section:

```yaml
function_groups:
  gsf:
    _type: gsf
    base_url: ${GSF_BASE_URL}
    auth:
      mode: password
      email: ${GSF_EMAIL}
      password: GSF_PASSWORD
    include:
      - catalog_search
      - text_to_sql
```

Omit `auth` in an authenticated AI-Q deployment to forward the current user's
bearer token. Password mode is intended for local development and evaluation.
The `password` field names the environment variable containing the secret; it
does not contain or interpolate the secret itself.
Refer to `sources/gsf/README.md` for the complete contract and limits.

### `sandboxed_python`

Stateless scientific analysis inside one fresh OpenShell sandbox per Data
Science Agent request. Each call runs a self-contained script in a fresh Python
namespace. The tool is an unmapped utility rather than a data source; add its
function key directly to the agent's explicit `tools` list.

```yaml
functions:
  ds_python_sandbox:
    _type: deep_research_sandbox
    provider: openshell
    openshell_image: ${AIQ_DS_OPENSHELL_IMAGE:-aiq-openshell-demo:latest}
    policy: ${AIQ_DS_OPENSHELL_POLICY_FILE}
    workdir: /sandbox
    network: blocked
    delete_on_exit: true
    attest: true

  python:
    _type: sandboxed_python
    sandbox: ds_python_sandbox
    wall_timeout_seconds: 60
    max_code_chars: 50000
    max_output_chars: 50000
    max_evidence_bytes: 20000000
    max_memory_mb: 8192
    max_cpu_seconds: 600
    max_processes: 256
    max_open_files: 256
    max_file_bytes: 100000000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sandbox` | function ref | **required** | A `deep_research_sandbox` function configured for a fresh OpenShell sandbox with `network: blocked`. Shared-sandbox attachment and other providers are rejected. |
| `wall_timeout_seconds` | `float` | `30` | Maximum wall time for one Python script. A timeout terminates the complete request sandbox. |
| `max_code_chars` | `int` | `50000` | Maximum characters accepted in one Python script. |
| `max_output_chars` | `int` | `50000` | Maximum captured script output and displayed-result characters. |
| `max_evidence_bytes` | `int` | `20000000` | Maximum total bytes of request-owned GSF receipts and manifest synchronized into the sandbox. |
| `max_memory_mb` | `int` | `8192` | Hard address-space limit for each Python process. |
| `max_cpu_seconds` | `int` | `600` | Hard CPU-time limit for each Python process. |
| `max_processes` | `int` | `256` | Hard per-user process/thread limit inside the fresh sandbox. |
| `max_open_files` | `int` | `256` | Hard open-file-descriptor limit for the worker. |
| `max_file_bytes` | `int` | `100000000` | Hard maximum size of a file created by the worker process. |

The runner preloads pandas, NumPy, SciPy, scikit-learn, and statsmodels for every
call. It has
no GSF client, SQL connection, host-process fallback, or network access. AI-Q
uploads only the version-matched runner, model code request, and validated request-local
GSF receipt JSON; application environment variables and credentials are not
included in the OpenShell sandbox specification. OpenShell owns the physical
sandbox boundary, while each process additionally receives hard Unix resource
limits before model-authored code starts.

### `intent_classifier`

Classifies user queries as meta (conversational) or research, and determines research depth (shallow vs. deep).

```yaml
functions:
  intent_classifier:
    _type: intent_classifier
    llm: nemotron_lightning_intent_llm
    tools:
      - web_search_tool
      - paper_search_tool
    llm_timeout: 90
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `str` | **required** | Reference to an LLM defined in `llms` section. |
| `tools` | `list[str]` | `[]` | Tool references passed to the intent prompt for tool-awareness. |
| `llm_timeout` | `float` | `90` | Timeout in seconds for the intent classification LLM call. |

### `clarifier_agent`

Interactive clarification dialog for deep research queries. Asks follow-up questions to refine scope before research begins.

```yaml
functions:
  clarifier_agent:
    _type: clarifier_agent
    llm: nemotron_llm
    tools:
      - web_search_tool
    max_turns: 3
    log_response_max_chars: 2000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `str` | **required** | LLM for generating clarification questions. |
| `tools` | `list[str]` | `[]` | Tools available for gathering context during clarification. |
| `exclude_tools` | `list[str]` | `[]` | Tool names to exclude when inheriting from the data source registry. |
| `max_turns` | `int` | `3` | Maximum number of clarification Q&A turns before auto-completing. |
| `log_response_max_chars` | `int` | `2000` | Maximum characters to log from LLM responses. |

### `shallow_research_agent`

Fast, single-pass research agent that attempts to produce citation-backed answers in one tool-calling loop.

```yaml
functions:
  shallow_research_agent:
    _type: shallow_research_agent
    llm: nemotron_llm
    tools:
      - web_search_tool
      - knowledge_search
    max_llm_turns: 10
    max_tool_iterations: 5
    enforce_citations: false
    verbose: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `str` | **required** | LLM for research and synthesis. |
| `tools` | `list[str]` | `[]` | Search tools available to the agent. |
| `max_llm_turns` | `int` | `10` | Maximum number of LLM turns (includes both reasoning and tool-calling steps). |
| `max_tool_iterations` | `int` | `5` | Maximum tool-calling iterations before forcing synthesis. |
| `enforce_citations` | `bool` | `false` | Fail the run when citation integrity cannot be preserved. When `false`, AI-Q returns the generated answer after sanitization instead of failing solely on the citation contract. |
| `verbose` | `bool` | `false` | Enable verbose logging. |

### `data_science_agent`

Adaptive ReAct agent for structured-data analysis, document retrieval, web
evidence, and final synthesis.

```yaml
functions:
  data_science_agent:
    _type: data_science_agent
    llm: data_science_llm
    # tools omitted -> inherit every tool in data_source_registry
    ontology_provider:
      provider: gsf
      catalog_tools: [gsf__catalog_search]
      analytical_tools: [gsf__text_to_sql]
    response_mode: standard
    structured_catalog_call_limit: 2
    structured_text_to_sql_call_limit: 6
    python_call_limit: 8
    finalization_model_call_limit: 18
    recursion_limit: 64
    verbose: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `str` | **required** | LLM used for tool selection, analysis, and synthesis. |
| `tools` | `list[str]` | `[]` | Explicit callable tools. An empty list inherits all tool and function-group references in `data_source_registry`. |
| `exclude_tools` | `list[str]` | `[]` | Exact runtime tool names removed after inherited or explicit tools are resolved. |
| `interaction_mode` | `interactive` or `headless` | `interactive` | In `headless` mode, never wait for clarification; resolve supported assumptions and perform one bounded synthesis retry if needed. |
| `response_mode` | `standard` or `fdabench_choice` | `standard` | In `fdabench_choice` mode, preserve explicitly supplied option labels and emit an `Answer:` marker; non-choice requests retain normal report behavior. |
| `ontology_provider` | object or `None` | `None` | Assign one provider identifier, non-empty `catalog_tools` and `analytical_tools` lists, and optional `predictive_tools`. Every referenced tool must be enabled and mapped to the same `data_source_registry` source. |
| `structured_catalog_call_limit` | `int` or `None` | `None` | Optional request-local hard limit on ontology catalog calls. Minimum `1`; exact cache hits do not count. |
| `structured_text_to_sql_call_limit` | `int` or `None` | `None` | Optional request-local hard limit on ontology-provider text-to-SQL calls. Predictive execution is not included. Minimum `1`; exact cache hits do not count. |
| `structured_cache_repeated_calls` | `bool` | `true` | Reuse exact repeated catalog and text-to-SQL calls within one agent request. Cache state never crosses requests. |
| `python_call_limit` | `int` or `None` | `None` | Optional request-local call ceiling for stateless scientific Python execution. |
| `finalization_model_call_limit` | `int` or `None` | derived | Model-call count at which tools are disabled and a no-tool synthesis turn is forced before recursion exhaustion. |
| `recursion_limit` | `int` | `64` | Hard LangGraph step limit for one adaptive run. Minimum `4`. |
| `verbose` | `bool` | `false` | Enable verbose tracing. |

### `data_science_hybrid_adapter`

Maps the catalog-aware Chat Researcher state into a configured
`data_science_agent`. Direct DS Agent workflows do not use this adapter.

```yaml
functions:
  data_science_hybrid_adapter:
    _type: data_science_hybrid_adapter
    agent: data_science_agent
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `str` | **required** | Function reference for the `data_science_agent` instance that handles Hybrid requests. |

### `deep_research_agent`

Multi-phase research agent with an orchestrator, optional advisory source router, planner, concurrent researcher workers,
and final writer. The planner records an answer strategy plus structured `ResearchQuery` objects; the orchestrator
batches those queries for researcher workers and delegates final synthesis to the writer.

```yaml
functions:
  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_ultra_llm
    source_router_llm: nemotron_ultra_llm
    researcher_llm: nemotron_ultra_llm
    planner_llm: nemotron_ultra_llm
    writer_llm: nemotron_ultra_writer_llm
    # tools omitted -> inherit every tool in data_source_registry
    exclude_tools:
      - web_search_tool
    enable_source_router: true
    domain_catalog_path: configs/domain_catalogs/deep_research_domain_catalog.yml
    enable_citation_verification: true
    # Optional config-function references; define these functions before enabling:
    # skills: deep_research_skills
    # sandbox: deep_research_sandbox
    max_research_concurrency: 6
    max_researcher_model_calls: 100
    max_concurrent_source_tool_calls: 5
    max_source_tool_batch_size: 4
    resource_limits:
      max_input_chars: 32768
      max_execution_seconds: 3600
      max_plan_bytes: 1048576
      max_source_routing_bytes: 1048576
      max_final_report_bytes: 2097152
      max_state_file_count: 64
      max_total_state_bytes: 25165824
      max_research_queries: 20
      max_total_query_chars: 10000
      max_research_note_bytes: 524288
      max_total_research_note_bytes: 10485760
      max_source_tool_calls: 100
      max_todo_items: 20
      max_todo_item_chars: 2048
      max_total_todo_chars: 10000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `orchestrator_llm` | `str` | **required** | LLM for the orchestrator that coordinates the research workflow. |
| `source_router_llm` | `str` | `None` | LLM for the source-router sub-agent. Falls back to `orchestrator_llm` if not specified. |
| `researcher_llm` | `str` | `None` | LLM for the researcher sub-agent. Falls back to `orchestrator_llm` if not specified. |
| `planner_llm` | `str` | `None` | LLM for the planner sub-agent. Falls back to `orchestrator_llm` if not specified. |
| `writer_llm` | `str` | `None` | LLM for the final writer/synthesis sub-agent. Falls back to `orchestrator_llm` if not specified. |
| `tools` | `list[str]` | `[]` | Explicit callable tools. An empty list inherits all tool and function-group references in `data_source_registry`; a non-empty list bypasses inheritance. |
| `exclude_tools` | `list[str]` | `[]` | Exact runtime tool names removed after inherited or explicit tools are resolved. |
| `domain_catalog_path` | `str` | `None` | Optional YAML or JSON domain catalog used by the source router. Without one, AI-Q generates a general route from available mapped sources. |
| `enable_source_router` | `bool` | `true` | Run the advisory source-router sub-agent before planning. It recommends available mapped sources but does not restrict worker tool bindings. |
| `enable_citation_verification` | `bool` | `true` | Verify final citations against sources captured from configured tool results. Set `false` only when the active source formats are not compatible with verification. |
| `skills` | object or function ref | `None` | Inline `deep_research_skills` config or a reference to a config-only function of that type. Skill assignments are keyed by `researcher-agent` and `writer-agent`. |
| `sandbox` | object or function ref | `None` | Inline `deep_research_sandbox` config or a reference to a config-only function of that type. Enables the DeepAgents execution backend. |
| `max_research_concurrency` | `int` | `6` | Maximum `ResearchQuery` objects accepted and run concurrently by one `run_research_batch` call. |
| `max_researcher_model_calls` | `int` | `100` | Maximum normal model turns per researcher worker before one tools-disabled finalization turn. |
| `max_concurrent_source_tool_calls` | `int` | `5` | Shared cap on concurrent source-tool calls across all researcher workers in the run. |
| `max_source_tool_batch_size` | `int` | `4` | Maximum concrete inputs accepted by a batch-capable source-tool wrapper in one call. |
| `resource_limits` | object | See below | Non-disableable per-job request, graph, state, and provider-call ceilings. Values may be reduced but cannot exceed the defaults. |

`resource_limits` is enforced in both synchronous and async-job construction:

| Parameter | Default and maximum | Enforcement boundary |
|-----------|---------------------|----------------------|
| `max_input_chars` | `32768` | Combined final user query and clarifier context, before state preparation or graph construction. |
| `max_execution_seconds` | `3600` | Top-level deep-research graph execution timeout. |
| `max_plan_bytes` | `1048576` | UTF-8 serialized `ResearchPlan`, before `/shared/plan.json` persistence. |
| `max_source_routing_bytes` | `1048576` | UTF-8 serialized `SourceRoutingPlan`, before `/shared/source_routing.json` persistence. |
| `max_final_report_bytes` | `2097152` | UTF-8 serialized writer output, before `/shared/output.md` persistence. |
| `max_state_file_count` | `64` | Files held in the job's StateBackend filesystem, including resumed state. Sandbox workspace files are outside this state budget. |
| `max_total_state_bytes` | `25165824` | Aggregate payload bytes held in the job's StateBackend filesystem, including resumed state. |
| `max_research_queries` | `20` | Accepted researcher queries across every batch in the job. Each query returns at most one persisted note, so this also caps the job at 20 research-note files. |
| `max_total_query_chars` | `10000` | Aggregate main-query and subquery characters across the job. |
| `max_research_note_bytes` | `524288` | UTF-8 serialized size of one `ResearchNotes` payload. |
| `max_total_research_note_bytes` | `10485760` | Aggregate serialized research-note bytes across the job. |
| `max_source_tool_calls` | `100` | AI-Q source-tool attempts and concrete batch items across workers in this job. Retries hidden inside a provider SDK are not observable to this counter. |
| `max_todo_items` | `20` | Top-level orchestrator todo items in one state replacement. Subagents cannot write todos. |
| `max_todo_item_chars` | `2048` | Characters in one top-level todo item. |
| `max_total_todo_chars` | `10000` | Aggregate todo content characters in one state replacement. |

`max_research_concurrency` cannot exceed `resource_limits.max_research_queries`.
When lowering the job-wide query budget below the default concurrency of `6`,
lower `max_research_concurrency` in the same configuration.

`data_sources` request filtering happens after this configured tool set is resolved. It removes tools mapped to
unselected registry sources but preserves configured tools with no source mapping. Router recommendations become
ordered `preferred_tools` and `fallback_tools` guidance on each `ResearchQuery`; workers still receive the full
request-filtered callable set. Refer to [Tools and Sources](./tools-and-sources.md#automatic-source-routing) and the
[`config_domain_routing_and_skills.yml`](../../../configs/config_domain_routing_and_skills.yml) reference profile.

```{note}
**Migration: `chart-generation` moved to the `visualization` collection.** The built-in
`chart-generation` skill previously lived in the `research` collection; it now lives in its own
`visualization` collection, and charts are no longer sandbox-gated. The `visualization` skill ships
enabled only in the skills and sandbox example configs (`config_domain_routing_and_skills.yml` and
`config_openshell.yml`); every other shipped config presents chart-worthy data as a Markdown table.
A writer that wants inline charts must be assigned the `visualization` collection in its
`deep_research_skills` assignment.
```

---

## `workflow` Section

Defines the top-level workflow. The normal product pipeline uses
`chat_deepresearcher_agent`; direct DS Agent development uses
`data_science_workflow`.

```yaml
workflow:
  _type: chat_deepresearcher_agent
  enable_escalation: true
  enable_clarifier: true
  use_async_deep_research: true
  max_history: 20
  checkpoint_db: ${AIQ_CHECKPOINT_DB:-./checkpoints.db}
  relay:
    logging: true
    observability:
      enable_full_payloads: true
      atof: {enabled: true, output_directory: ./relay, filename: aiq-relay.atof.jsonl, mode: append}
      opentelemetry:
        enabled: false
        endpoints:
          - type: openinference
            endpoint: "${RELAY_OTEL_ENDPOINT:-http://localhost:6006/v1/traces}"
            service_name: aiq-relay
            resource_attributes: {openinference.project.name: aiq-relay}
    redaction:
      enabled: true
      request_privacy_attributes: [data, category_profile]
```

The default pricing source list is empty, so default configs omit the pricing
block and do not load a catalog. The dedicated
`configs/nemo_relay/config_web_default_with_pricing.yml` example loads
deployment-specific rates from `configs/nemo_relay/relay_pricing_catalog.json`.
Its zero-dollar Nemotron entries describe the NVIDIA-hosted access path used by
the example; they are not estimates for self-hosted infrastructure. Review the
catalog when the provider offer or deployment changes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_type` | `str` | **required** | Use `chat_deepresearcher_agent` for the full pipeline or `data_science_workflow` to run the DS Agent directly. |
| `enable_escalation` | `bool` | `true` | Allow the intent classifier to route queries to deep research. When `false`, all research queries use shallow research only. |
| `enable_clarifier` | `bool` | `true` | Run the clarifier agent before deep research to gather user requirements. |
| `use_async_deep_research` | `bool` | `false` | Submit deep research as an async background job (requires [Dask](https://www.dask.org/) scheduler). |
| `hybrid_research_agent` | `str` or `None` | `None` | Optional function reference invoked when catalog-aware routing selects Hybrid research. |
| `max_history` | `int` | `20` | Maximum number of messages to keep in conversation history before trimming. |
| `checkpoint_db` | `str` | `./checkpoints.db` | SQLite path or PostgreSQL DSN for persistent conversation checkpoints. |
| `relay` | `object` | enabled defaults | NeMo Relay logging, Observability v3 ATOF/OTEL destinations, PII redaction, and pricing sources. Relay instrumentation itself has no workflow disable switch. See [Observability with NeMo Relay](../deployment/observability.md). |

Relay configuration is strict: unknown nested fields and invalid OTLP endpoint
URLs fail workflow validation instead of being silently ignored.

> **Note:** `interactive_auth` is a YAML-level field consumed by the CLI entry point (`start_cli.sh` / `aiq-research`), not a Pydantic field on `ChatDeepResearcherConfig`. It can be set in YAML config files but is not part of the workflow config class.

---

## Annotated Core Pipeline Example

Below is a self-contained CLI configuration with web search, paper search, and clarification enabled. It intentionally
does not combine the knowledge backends, MCP OAuth, guardrails, domain routing, or skills/sandbox examples; use the
provided profiles in the next section as focused starting points for those capabilities.

```yaml
# General settings
general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO                    # Set to DEBUG for troubleshooting

# LLM definitions
llms:
  lightning_intent_llm:                # Used by intent classifier
    _type: nim
    model_name: nvidia/nemotron-3.5-lightning-30b-a3b
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.1
    top_p: 0.9
    max_tokens: 1024
    num_retries: 5
    parallel_tool_calls: false
    chat_template_kwargs:
      enable_thinking: false

  lightning_agent_llm:                 # Used by shallow researcher
    _type: nim
    model_name: nvidia/nemotron-3.5-lightning-30b-a3b
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 8192
    num_retries: 5
    parallel_tool_calls: false
    chat_template_kwargs:
      enable_thinking: true

  ultra_llm:                           # Used by clarifier and deep research
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 16384
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: false

  ultra_writer_llm:                    # Used by deep research writer
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 32768
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: false

# Tools and agents
functions:
  web_search_tool:                     # Standard web search
    _type: tavily_web_search
    max_results: 5
    max_content_length: 1000

  advanced_web_search_tool:            # Deep search (fewer results, more depth)
    _type: tavily_web_search
    max_results: 2
    advanced_search: true

  paper_search_tool:                   # Academic paper search
    _type: paper_search
    max_results: 5
    serper_api_key: ${SERPER_API_KEY}

  intent_classifier:                   # Classifies queries, routes depth
    _type: intent_classifier
    llm: lightning_intent_llm
    tools:
      - web_search_tool
      - paper_search_tool

  clarifier_agent:                     # Asks clarifying questions for deep research
    _type: clarifier_agent
    llm: ultra_llm
    tools:
      - web_search_tool
    max_turns: 3

  shallow_research_agent:              # Fast single-pass research
    _type: shallow_research_agent
    llm: lightning_agent_llm
    tools:
      - web_search_tool
    max_llm_turns: 10
    max_tool_iterations: 5

  deep_research_agent:                 # Multi-phase deep research
    _type: deep_research_agent
    orchestrator_llm: ultra_llm
    researcher_llm: ultra_llm
    source_router_llm: ultra_llm
    planner_llm: ultra_llm
    writer_llm: ultra_writer_llm
    tools:
      - paper_search_tool
      - advanced_web_search_tool

# Top-level orchestrator
workflow:
  _type: chat_deepresearcher_agent
  enable_escalation: true              # Allow deep research routing
  enable_clarifier: true               # Ask clarifying questions first
  checkpoint_db: ${AIQ_CHECKPOINT_DB:-./checkpoints.db}
```

## Provided Config Files

The repository includes fourteen top-level workflow configurations. They are focused reference profiles, not cumulative
layers, and no single profile enables every capability. Start from the profile closest to the deployment and merge
only the additional sections you need.

| File | Mode | Enabled behavior and opt-ins |
|------|------|------------------------------|
| `configs/config_cli_default.yml` | CLI | Chat pipeline with Tavily web search and clarification. No knowledge backend. Paper search is present only as a commented opt-in. |
| `configs/config_cli_data_science.yml` | Direct DS Agent CLI | GSF catalog/text-to-SQL, Foundational RAG knowledge retrieval, and Tavily web search without the top-level router. |
| `configs/config_cli_data_science_fdabench_lite.yml` | Direct DS Agent evaluation | Headless FDABench-Lite DS ReAct profile with GSF, Foundational RAG, Tavily, choice-label output, and request-local GSF budgets. |
| `configs/config_cli_data_science_fdabench_lite_python.yml` | Direct DS Agent evaluation | Same FDABench-Lite profile plus blocked-network, stateless OpenShell scientific Python execution and an exact GSF-result bridge. |
| `configs/config_web_default_llamaindex.yml` | Web API | Default chat pipeline with LlamaIndex/ChromaDB knowledge retrieval and Tavily. Paper search is commented out. |
| `configs/config_web_azure_ai_search.yml` | Web API | Azure AI Search knowledge retrieval and web search |
| `configs/config_web_frag.yml` | Web API / Helm base | Foundational RAG plus Tavily. Requires separately deployed RAG query and ingestion services. Paper search is commented out. |
| `configs/config_web_opensearch.yml` | Web API | Built-in OpenSearch knowledge backend plus Tavily. Supports unauthenticated or basic self-hosted OpenSearch and SigV4 (`es` or `aoss`); infrastructure and credentials are deployment opt-ins. |
| `configs/config_frontier_models.yml` | Web API | Shipped LlamaIndex frontier profile: GPT-5.6 Luna for intent/shallow/source routing/research, GPT-5.6 Sol for clarification/orchestration/planning/writing, and Gemma 4 for summaries. Requires `NVIDIA_API_KEY`, `OPENAI_API_KEY`, and `TAVILY_API_KEY` for the enabled Tavily tools; the commented paper-search opt-in requires `SERPER_API_KEY` when enabled. Validate the complete workflow against the configured provider endpoints before deployment. |
| `configs/config_web_default_guardrails.yml` | Web API | LlamaIndex with workflow Guardrails attached explicitly, shallow-agent Guardrails dynamically attached through `workflow_functions`, and async deep-agent Guardrails applied by the AI-Q runner from the same target configuration. |
| `configs/config_web_frag_mcp_auth.yml` | Web API | Foundational RAG plus a protected per-user OAuth MCP source example. Requires a real protected MCP endpoint and shared token-store configuration; it is not a zero-config default. |
| `configs/config_domain_routing_and_skills.yml` | Direct deep-research workflow | Automatic domain routing, Tavily, DuckDuckGo news, Polymarket, LlamaIndex, enabled Serper paper search, built-in skills, and a Modal sandbox. Requires the corresponding service credentials and Modal setup. |
| `configs/config_openshell.yml` | Web API, experimental | Skills and artifact capture over one policy-bound OpenShell sandbox per deep-research job, with fail-closed policy attestation and terminal deletion. |
| `configs/config_mcp.yml` | Standalone MCP server | Public NIM and Tavily research over stateless submit, poll, and final-report tools with PostgreSQL-backed job state. Requires `NVIDIA_API_KEY`, `TAVILY_API_KEY`, and `AIQ_CHECKPOINT_DB`. |

You are not limited to these files -- you can supply your own config per
deployment method: with Docker Compose, via `BACKEND_CONFIG` and the
`configs/` bind mount; with Helm, via `aiq.apps.backend.config` (inline
`content` or an `existingConfigMap`). See
[Kubernetes → Configuration](../deployment/kubernetes.md#configuration) and
[Docker Compose](../deployment/docker-compose.md) for details.

## Related

- [Swapping Models](./swapping-models.md) -- Change LLMs without touching agent code
- [Tools and Sources](./tools-and-sources.md) -- Enable and disable search tools
- [Knowledge Layer](./knowledge-layer.md) -- Configure document retrieval backends
- [Prompts](./prompts.md) -- Customize agent behavior through prompt templates
- [Kubernetes Deployment](../deployment/kubernetes.md) -- Deploy with Helm, including custom config overlays
