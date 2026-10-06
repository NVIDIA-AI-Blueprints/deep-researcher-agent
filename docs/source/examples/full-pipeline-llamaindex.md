<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Full Pipeline (LlamaIndex)

The complete Deep Researcher Agent blueprint configuration using **LlamaIndex + ChromaDB** for knowledge retrieval. This is the recommended setup for local development -- zero external RAG infrastructure required.

This is based on `configs/config_web_default_llamaindex.yml`.

```{note}
This example preserves the shipped Lightning shallow profile. The NVIDIA API Catalog serving profile has a known
[shallow citation-output limitation](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog).
Deep Researcher Agent fails closed rather than publishing citation-incomplete drafts.
```

## Configuration

```yaml
# config_web_default_llamaindex.yml (annotated)
# Full pipeline: Web mode with LlamaIndex knowledge layer

# ===========================================================================
# General settings
# ===========================================================================
general:
  use_uvloop: true

  telemetry:
    logging:
      console:
        _type: console
        level: INFO

  front_end:
    _type: deep_researcher_api
    runner_class: deep_researcher_api.plugin.DeepResearcherAPIWorker
    db_url: ${NAT_JOB_STORE_DB_URL:-sqlite+aiosqlite:///./jobs.db}
    expiry_seconds: 86400
    cors:
      allow_origin_regex: 'http://localhost(:\d+)?|http://127.0.0.1(:\d+)?'
      allow_methods: [GET, POST, DELETE, OPTIONS]
      allow_headers: ["*"]
      allow_credentials: true
      expose_headers: ["*"]

# ===========================================================================
# LLMs
# ===========================================================================
llms:
  nemotron_lightning_intent_llm:
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

  nemotron_lightning_agent_llm:
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

  # LLM for clarification and deep-research roles
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

  nemotron_ultra_writer_llm:
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

  # LLM for document summaries (shown in the UI after upload)
  summary_llm:
    _type: nim
    model_name: google/gemma-4-31b-it
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.1
    max_tokens: 100

# ===========================================================================
# Functions (tools and agents)
# ===========================================================================
functions:
  web_search_tool:
    _type: tavily_web_search
    max_results: 5
    max_content_length: 1000

  advanced_web_search_tool:
    _type: tavily_web_search
    max_results: 2
    advanced_search: true

  # -------------------------------------------------------------------------
  # Knowledge retrieval (LlamaIndex + ChromaDB)
  # -------------------------------------------------------------------------
  # Stores embeddings locally in ChromaDB. No external RAG server needed.
  # Documents are uploaded through the Knowledge API (/v1/collections).
  knowledge_search:
    _type: knowledge_retrieval
    backend: llamaindex
    collection_name: ${COLLECTION_NAME:-test_collection}
    generate_summary: true                                   # Generate per-doc summaries
    summary_model: summary_llm                               # LLM for summaries
    summary_db: ${DEEP_RESEARCHER_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
    top_k: 5
    chroma_dir: ${DEEP_RESEARCHER_CHROMA_DIR:-/tmp/chroma_data}          # Local vector store

  # Paper Search (optional - requires SERPER_API_KEY)
  # Uncomment the block below and set SERPER_API_KEY to enable.
  # paper_search_tool:
  #   _type: paper_search
  #   max_results: 5
  #   serper_api_key: ${SERPER_API_KEY}

  intent_classifier:
    _type: intent_classifier
    llm: nemotron_lightning_intent_llm
    tools:
      - web_search_tool
      # - paper_search_tool  # Uncomment if SERPER_API_KEY is set
      - knowledge_search

  clarifier_agent:
    _type: clarifier_agent
    llm: nemotron_ultra_llm
    tools:
      - web_search_tool
      - knowledge_search
    max_turns: 3
    log_response_max_chars: 2000

  shallow_research_agent:
    _type: shallow_research_agent
    llm: nemotron_lightning_agent_llm
    tools:
      - web_search_tool
      - knowledge_search
    max_llm_turns: 10
    max_tool_iterations: 5

  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_ultra_llm
    source_router_llm: nemotron_ultra_llm
    planner_llm: nemotron_ultra_llm
    researcher_llm: nemotron_ultra_llm
    writer_llm: nemotron_ultra_writer_llm
    tools:
      # - paper_search_tool  # Uncomment if SERPER_API_KEY is set
      - advanced_web_search_tool
      - knowledge_search

workflow:
  _type: chat_deepresearcher_agent
  enable_escalation: true
  enable_clarifier: true
  use_async_deep_research: true
  checkpoint_db: ${DEEP_RESEARCHER_CHECKPOINT_DB:-./checkpoints.db}
```

## Required Environment Variables

```bash
# Core (required)
export NVIDIA_API_KEY="nvapi-..."    # pragma: allowlist secret
export TAVILY_API_KEY="tvly-..."     # pragma: allowlist secret
```

No RAG server URLs are needed -- LlamaIndex uses local ChromaDB storage.

## How to Run

### Backend

```bash
source .venv/bin/activate

dotenv -f deploy/.env run .venv/bin/nat serve \
  --config_file configs/config_web_default_llamaindex.yml
```

The server starts at `http://localhost:8000`.

### Frontend (optional)

```bash
cd frontends/ui && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Upload Documents

```bash
# Create a collection
curl -X POST http://localhost:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs", "description": "My document collection"}'

# Upload files
curl -X POST http://localhost:8000/v1/collections/my-docs/documents \
  -F "files=@report.pdf"
```

### Ask Questions

```bash
# Submit a query
curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -H "conversation-id: my-docs" \
  -d '{"agent_type": "shallow_researcher", "input": "What is CUDA?"}'

# Stream events
curl -N http://localhost:8000/v1/jobs/async/job/{job_id}/stream
```

The `conversation-id` header selects the collection used for retrieval. Keep it equal to the collection used in the
upload path (`my-docs` in this example); without the header, retrieval uses the config's `collection_name` fallback.

## Key Differences from Foundational RAG

| Aspect | LlamaIndex (this config) | Foundational RAG |
|--------|--------------------------|------------------|
| Vector store | Local ChromaDB | Hosted RAG server |
| External infra | None | RAG + ingest servers |
| Document summaries | Yes (`generate_summary: true`) | No |
| Best for | Local development | Production multi-user |

For production multi-user deployments, refer to [Full Pipeline -- Foundational RAG](./full-pipeline-web.md).
