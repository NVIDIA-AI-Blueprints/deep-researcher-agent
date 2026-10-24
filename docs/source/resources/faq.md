<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FAQ

Frequently asked questions about the Deep Researcher Agent blueprint.

## General

**What is the Deep Researcher Agent blueprint?**

An NVIDIA blueprint for AI-powered deep research built on the NeMo Agent Toolkit. It combines intelligent query routing, multi-agent research pipelines, and pluggable knowledge retrieval to deliver comprehensive, citation-backed research answers.

**What models does it use?**

The default profiles use NVIDIA Nemotron models through the `integrate.api.nvidia.com` API. The shipped frontier
profile uses the exact GPT Sol/Luna role assignment in `configs/config_frontier_models.yml`. Deep Researcher Agent is configurable with
self-hosted NIMs and other providers, but a bring-your-own model is not automatically compatible and can require
provider-specific prompt, hyperparameter, tool-calling, and structured-output tuning. Run the complete workflow against
the exact provider configuration before deployment. Refer to [Swapping Models](../customization/swapping-models.md).

**Do I need a GPU?**

Not for running the blueprint itself — it calls cloud-hosted LLM APIs. You only need a GPU if you self-host NIM models locally. Refer to [Using Downloadable NIMs](../customization/swapping-models.md#using-downloadable-nims-self-hosted).

## Architecture

**What's the difference between shallow and deep research?**

- **Shallow research** is fast (30-60s), uses a single agent with bounded tool calls, and produces concise answers with citations. Best for simple factual queries.
- **Deep research** is thorough and commonly takes several minutes; complex jobs can exceed ten minutes. An orchestrator coordinates an optional advisory source router, a planner, concurrent researcher workers, and a writer that performs final synthesis. Best for complex multi-faceted topics and output shapes that need evidence from several focused queries.

The [Intent Classifier](../architecture/agents/intent-classifier.md) automatically routes queries to the appropriate depth.

**Can I disable the clarifier step?**

Yes. The clarifier gathers missing context or the requested output type and
research planning starts afterward inside the deep-research workflow. Refer
to [Human-in-the-Loop](../customization/hitl.md) for configuration options.
You can disable the clarifier entirely or limit how many clarification
questions it asks.

**What happens when shallow research escalates to deep?**

If `enable_escalation: true` in the workflow config, the orchestrator evaluates the shallow research result. If it detects insufficient coverage (response too short, "unable to find" keywords), it escalates to the clarifier and then deep research. The clarifier asks only for missing context; planning happens inside the deep-research workflow. Refer to [Architecture Overview](../architecture/overview.md).

**How does deep research choose data sources?**

The request's `data_sources` selection is a hard boundary for tools mapped in
`data_source_registry`. Unmapped configured or utility tools remain active
and do not appear in the router catalog. The optional source router recommends
mapped sources only from the allowed set and cannot restore a filtered-out
mapped source.

The planner records preferred and fallback tool names in each structured
`ResearchQuery` as guidance. `run_research_batch` sends those queries to
concurrent workers that are bound to the full request-filtered tool set and
prompted to follow the recorded order.

## Tools and Sources

**What search tools are available?**

- **Tavily Web Search** — General web search (requires `TAVILY_API_KEY`)
- **You.com APIs** — Web search, page contents, cited general research, and finance research (`YDC_API_KEY` is
  required for live API calls; without it, Deep Researcher Agent starts with diagnostic stubs for these tools)
- **Exa Web Search** — General web search via Exa (requires `EXA_API_KEY`)
- **Nimble Web Search** — General web search via Nimble (requires `NIMBLE_API_KEY`)
- **Firecrawl Web Search** — Web search with page Markdown via Firecrawl (`FIRECRAWL_API_KEY` is optional and raises rate limits)
- **DuckDuckGo News Search** — Recent news search (no API key)
- **Polymarket Prediction Markets** — Events and market-implied probabilities (no API key)
- **Google Scholar Paper Search** — Academic search through Serper, SerpAPI, or SearchAPI (requires the selected provider's key)
- **Knowledge Layer** — Document retrieval through LlamaIndex, Foundational RAG, OpenSearch, or Azure AI Search

Refer to [Tools and Sources](../customization/tools-and-sources.md).

**Can I add my own tools?**

Yes. Refer to [Adding a Tool](../extending/adding-a-tool.md) for an end-to-end guide on building and registering custom NeMo Agent Toolkit functions.

## Knowledge Layer

**Which knowledge layer backend should I use?**

- **LlamaIndex** (ChromaDB) for development and prototyping — runs locally, no external services needed
- **Foundational RAG** for production — connects to NVIDIA RAG Blueprint, supports multi-user with Milvus
- **OpenSearch** for an existing OpenSearch deployment or AWS-managed vector retrieval — supports self-hosted/basic auth,
  Amazon OpenSearch Service, and Amazon OpenSearch Serverless with SigV4
- **Azure AI Search** for managed hybrid retrieval — supports API-key authentication or Azure managed identity

Refer to [Knowledge Layer](../customization/knowledge-layer.md). For AOSS on EKS, use the
[Amazon OpenSearch Serverless guide](../deployment/aws-opensearch-serverless.md).

**How do I upload documents?**

Through the Web UI (drag and drop), the Knowledge API (`POST /v1/collections/{name}/documents`), or programmatically through the ingestor SDK. Refer to [Knowledge Layer](../customization/knowledge-layer.md#web-ui-mode).

## Deployment

**What are the deployment options?**

1. **Local development** — `nat run` or `nat serve` directly
2. **Docker Compose** — Full stack with backend, frontend, and database

Refer to [Deployment](../deployment/index.md).

**What database should I use in production?**

PostgreSQL. The default compose stack includes a PostgreSQL container. For production, use a managed database service. Refer to [Deployment — Production Considerations](../deployment/production.md).

## Evaluation

**How do I measure research quality?**

Use the built-in benchmark suites: FreshQA (factual accuracy), Deep Research Bench (report quality), DeepSearchQA (document QA), and Intent Classifier (routing accuracy). Refer to [Benchmarks](../evaluation/benchmarks/index.md).

**How do I run benchmarks?**

```bash
dotenv -f deploy/.env run .venv/bin/nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_research_only.yml
```

Refer to each benchmark's documentation for config options and methodology.
