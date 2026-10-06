<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Full Pipeline (Foundational RAG)

The complete Deep Researcher Agent blueprint configuration with all features enabled: intent classification, shallow and deep research agents, knowledge retrieval (Foundational RAG), paper search, web search, clarifier with human-in-the-loop clarification, and the async jobs API with SSE streaming.

This is based on `configs/config_web_frag.yml`, which is the default for Helm deployments.

```{note}
This example preserves the shipped Lightning shallow profile. The NVIDIA API Catalog serving profile has a known
[shallow citation-output limitation](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog).
Deep Researcher Agent fails closed rather than publishing citation-incomplete drafts.
```

## Configuration

```yaml
# config_web_frag.yml (annotated)
# Full pipeline: Web mode with Foundational RAG knowledge layer

# ===========================================================================
# General settings
# ===========================================================================
general:
  use_uvloop: true  # Use uvloop for better async performance (Linux/macOS)

  telemetry:
    logging:
      console:
        _type: console
        level: INFO

  # ---------------------------------------------------------------------------
  # Front-end: Deep Researcher Agent API plugin
  # ---------------------------------------------------------------------------
  # This enables the async jobs API, SSE streaming, and Knowledge API.
  # Without this section, `nat serve` uses NeMo Agent Toolkit's default WebSocket front-end.
  front_end:
    _type: deep_researcher_api
    runner_class: deep_researcher_api.plugin.DeepResearcherAPIWorker

    # Async job database (JobStore + EventStore)
    # SQLite for local dev, PostgreSQL for production
    db_url: ${NAT_JOB_STORE_DB_URL:-sqlite+aiosqlite:///./jobs.db}

    # Completed jobs are cleaned up after this duration
    expiry_seconds: 86400  # 24 hours (range: 600 to 604800)

    # CORS settings for the frontend UI
    cors:
      allow_origin_regex: 'http://localhost(:\d+)?|http://127.0.0.1(:\d+)?'
      allow_methods: [GET, POST, DELETE, OPTIONS]
      allow_headers: ["*"]
      allow_credentials: true
      expose_headers: ["*"]

# ===========================================================================
# LLMs
# ===========================================================================
# Role-specific LLM configurations:
# - Nemotron 3.5 Lightning for intent classification and shallow research
# - Ultra for clarification and every deep-research role
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

# ===========================================================================
# Functions (tools and agents)
# ===========================================================================
functions:
  # -------------------------------------------------------------------------
  # Search tools
  # -------------------------------------------------------------------------
  web_search_tool:
    _type: tavily_web_search
    max_results: 5
    max_content_length: 1000

  advanced_web_search_tool:
    _type: tavily_web_search
    max_results: 2
    advanced_search: true   # Full page content extraction

  paper_search_tool:
    _type: paper_search
    max_results: 5
    serper_api_key: ${SERPER_API_KEY}

  # -------------------------------------------------------------------------
  # Knowledge retrieval (Foundational RAG)
  # -------------------------------------------------------------------------
  # This enables the Knowledge API endpoints (/v1/collections, /v1/documents)
  # and gives agents access to uploaded document collections.
  knowledge_search:
    _type: knowledge_retrieval
    backend: foundational_rag
    collection_name: ${COLLECTION_NAME:-test_collection}
    top_k: 5
    rag_url: ${RAG_SERVER_URL:-http://localhost:8081}
    ingest_url: ${RAG_INGEST_URL:-http://localhost:8082}
    timeout: 300

  # -------------------------------------------------------------------------
  # Intent classifier
  # -------------------------------------------------------------------------
  # Routes queries to shallow or deep research based on complexity.
  # Has access to tools for context-aware routing decisions.
  intent_classifier:
    _type: intent_classifier
    llm: nemotron_lightning_intent_llm
    tools:
      - web_search_tool
      - paper_search_tool
      - knowledge_search

  # -------------------------------------------------------------------------
  # Clarifier agent (human-in-the-loop)
  # -------------------------------------------------------------------------
  # For deep research: asks clarifying questions before handing off to the
  # deep_research_agent.
  clarifier_agent:
    _type: clarifier_agent
    llm: nemotron_ultra_llm
    tools:
      - web_search_tool
      - knowledge_search
    max_turns: 3                  # Max clarification rounds
    log_response_max_chars: 2000

  # -------------------------------------------------------------------------
  # Shallow research agent
  # -------------------------------------------------------------------------
  # Single-turn ReAct agent for quick queries.
  shallow_research_agent:
    _type: shallow_research_agent
    llm: nemotron_lightning_agent_llm
    tools:
      - web_search_tool
      - knowledge_search
    max_llm_turns: 10
    max_tool_iterations: 5

  # -------------------------------------------------------------------------
  # Deep research agent
  # -------------------------------------------------------------------------
  # Multi-loop orchestrator that plans research, delegates to sub-agents,
  # and synthesizes comprehensive reports.
  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_ultra_llm
    source_router_llm: nemotron_ultra_llm
    planner_llm: nemotron_ultra_llm
    researcher_llm: nemotron_ultra_llm
    writer_llm: nemotron_ultra_writer_llm
    tools:
      - paper_search_tool
      - advanced_web_search_tool
      - knowledge_search

# ===========================================================================
# Workflow
# ===========================================================================
# The chat_deepresearcher_agent is the meta-routing workflow:
# 1. Intent classifier determines shallow vs deep
# 2. Shallow queries go directly to shallow_research_agent
# 3. Deep queries go through clarifier -> deep_research_agent
workflow:
  _type: chat_deepresearcher_agent
  enable_escalation: true          # Allow shallow -> deep escalation
  enable_clarifier: true           # Enable clarification flow for deep research
  use_async_deep_research: true    # Run deep research asynchronously
  checkpoint_db: ${DEEP_RESEARCHER_CHECKPOINT_DB:-./checkpoints.db}
```

## Required Environment Variables

```bash
# Core (required)
export NVIDIA_API_KEY="nvapi-..."    # pragma: allowlist secret
export TAVILY_API_KEY="tvly-..."     # pragma: allowlist secret
export SERPER_API_KEY="..."

# Knowledge layer (required if using Foundational RAG)
export RAG_SERVER_URL="http://localhost:8081"
export RAG_INGEST_URL="http://localhost:8082"

# Optional: production database
# export NAT_JOB_STORE_DB_URL="postgresql+asyncpg://user:pass@host:5432/deep_researcher_jobs"  # pragma: allowlist secret
```

## How to Run

### Local Development

```bash
dotenv -f deploy/.env run .venv/bin/nat serve \
  --config_file configs/config_web_frag.yml
```

The server starts at `http://localhost:8000`. The API docs are at `http://localhost:8000/docs`.

### Docker Compose

The FRAG workflow requires separately deployed RAG query and ingestion services.
Set both endpoints to addresses that are reachable from the `deep-researcher-agent`
container. Container-local `localhost` points back to the Deep Researcher Agent backend and is not
a valid cross-service address.

From the repository root:

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env with your API keys and these container-reachable values:
# BACKEND_CONFIG=/app/configs/config_web_frag.yml
# RAG_SERVER_URL=http://rag-server:8081/v1
# RAG_INGEST_URL=http://ingestor-server:8082/v1
docker compose --env-file deploy/.env \
  -f deploy/compose/docker-compose.yaml \
  up -d --build --wait
```

With the service-name endpoints shown above and both stacks running, connect
the Deep Researcher Agent backend to the RAG network:

```bash
docker network connect nvidia-rag deep-researcher-agent
```

Repeat this command whenever the `deep-researcher-agent` container is recreated.

### Test the Pipeline

```bash
# List available agents
curl http://localhost:8000/v1/jobs/async/agents

# Submit a shallow query
curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "shallow_researcher", "input": "What is CUDA?"}'

# Submit a deep research query
curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "deep_researcher", "input": "Compare transformer architectures for long-context inference"}'

# Stream events
curl -N http://localhost:8000/v1/jobs/async/job/{job_id}/stream

# Get the final report
curl http://localhost:8000/v1/jobs/async/job/{job_id}/report
```

## All Features Enabled

This configuration enables every major feature:

| Feature | Config Key | Status |
|---------|-----------|--------|
| Intent classification | `intent_classifier` | Enabled |
| Shallow research | `shallow_research_agent` | Enabled |
| Deep research | `deep_research_agent` | Enabled |
| Clarifier (HITL) | `clarifier_agent` + `enable_clarifier: true` | Enabled |
| Research escalation | `enable_escalation: true` | Enabled |
| Web search | `web_search_tool`, `advanced_web_search_tool` | Enabled |
| Paper search | `paper_search_tool` | Enabled |
| Knowledge layer | `knowledge_search` (Foundational RAG) | Enabled |
| Async jobs API | `front_end._type: deep_researcher_api` | Enabled |
| SSE streaming | Automatic with `deep_researcher_api` | Enabled |
| Knowledge API | Automatic when `knowledge_retrieval` configured | Enabled |
| Conversation persistence | `checkpoint_db` | Enabled |
