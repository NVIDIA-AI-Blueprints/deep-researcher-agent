<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->
<h1>NVIDIA Deep Researcher Agent Blueprint</h1>

> **🏆 BENCHMARK NOTE 🏆**
>
> To obtain results consistent with the **nvidia-deep-researcher** [DeepResearch Bench](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard) leaderboard and [DeepResearch Bench II](https://github.com/imlrz/DeepResearch-Bench-II) benchmark repository results, please use the [`drb1`](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/tree/drb1) and [`drb2`](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/tree/drb2) branches, respectively.


## Table of Contents
- [Overview](#overview)
- [What's New](#whats-new)
- [Software Components](#software-components)
- [Target Audience](#target-audience)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Automated Setup](#automated-setup)
  - [Obtain API Keys](#obtain-api-keys)
  - [Set Up Environment Variables](#set-up-environment-variables)
- [Configuration Files](#configuration-files)
- [Ways to Run the Agents](#ways-to-run-the-agents)
  - [Command-line interface (CLI)](#command-line-interface-cli)
  - [Web UI](#web-ui)
  - [Async Deep Research Jobs](#async-deep-research-jobs)
  - [MCP Server](#mcp-server)
  - [Benchmarks](#benchmarks)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Evaluating the Workflow](#evaluating-the-workflow)
  - [Available Benchmarks](#available-benchmarks)
  - [Running Evaluations](#running-evaluations)
- [Development](#development)
- [Roadmap](#roadmap)
- [Security Considerations](#security-considerations)
- [License](#license)

## Overview

The NVIDIA Deep Researcher Agent Blueprint is a deployable research backend built on the [NVIDIA NeMo Agent Toolkit](https://docs.nvidia.com/nemo/agent-toolkit/latest/) and [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview). Teams can self-host the application boundary and connect deployment-owned models, data sources, authentication, policy controls, storage, and observability. It provides both **quick, cited answers** and **in-depth, report-style research**, plus benchmarks and evaluation harnesses for measuring quality. Deep Researcher Agent is focused on governed research workflows; it is not a general-purpose coding-agent harness.

<p align="center">
<img src="./docs/assets/DeepResearcher-arch-light.png" alt="Deep Researcher Agent Architecture" width="800">
</p>

**Key features:**

- **Orchestration node** — One node classifies intent (meta vs. research), produces meta responses (for example, greetings, capabilities), and sets research depth (shallow vs. deep).
- **Shallow research** — Bounded, faster researcher with tool-calling and source citation.
- **Structured deep research** — Advisory source routing, structured planning, concurrent researcher workers, bounded source-tool batching, and a dedicated writer produce citation-backed reports and other requested output shapes.
- **Report follow-up** — Ask questions about a completed report, create a child-job cosmetic rewrite, or run delta research with the parent report as context.
- **Workflow configuration** — YAML configs define agents, tools, LLMs, and routing behavior so you can tune workflows without code changes.
- **Modular workflows** — All agents (orchestration node, shallow researcher, deep researcher, clarifier) are composable; each can run standalone or as part of the full pipeline.
- **Skills, sandbox execution, and durable outputs** — Built-in research/synthesis skills are exposed as host-side, read-only definitions. Skills that invoke code use a provider-neutral sandbox contract; Modal and OpenShell create one physical sandbox per deep-research job. Opt-in rich-file capture checkpoints manifest-declared files after successful sandbox commands, finalizes on success/failure, stores bytes in SQL or S3-compatible storage, and delivers metadata to the Files tab live and on replay.
- **Portable Agent Skills** — `deep-researcher-deploy` selects, starts, and validates an Deep Researcher Agent deployment; `deep-researcher-research` calls routed chat and async research from compatible coding harnesses.
- **Data source registry** — UI toggles and request payloads can select web, paper, enterprise, collaboration, and knowledge-layer sources per message.
- **Expanded sources** — Paper search supports Serper, SerpAPI, and SearchAPI; You.com adds web, contents, general-research, and finance-research tools; Nimble adds configurable web search; focused profiles demonstrate DuckDuckGo news, Polymarket, OpenSearch, and Azure AI Search knowledge retrieval.
- **Production API and auth** — REST endpoints, async job ownership, per-user OAuth-protected MCP sources, token validator entry points, and provider lifecycle hooks support authenticated deployments; a separate public MCP server exposes stateless research tools for trusted networks.
- **Opt-in policy controls** — NeMo Guardrails middleware covers selected workflow and agent boundaries, and narrow application-level encryption can protect final async output plus selected artifact-event content.
- **Observability, profiling, and cost analysis** — NeMo Relay preserves task, named-agent, LLM, and tool hierarchy across interactive turns and async researchers. ATOF feeds local debugging and tokenomics reports; OTEL exports traces to external observability backends.
- **Evaluation harnesses** — Built-in benchmarks (for example, FreshQA, DeepResearch) and evaluation scripts to measure quality and iterate on prompts and agent architecture.
- **Frontend options** — Run through CLI, web UI, or async jobs. Refer to [Getting started](#getting-started) and [Ways to run the agents](#ways-to-run-the-agents).
- **Deployment options** - Deployment assets for [Docker Compose](deploy/compose/) and [Helm](deploy/helm/deployment-k8s/); the repository source chart honors the Helm release namespace for every namespaced resource.

## What's New

Recent changes include:

- **Structured, concurrent deep research** — Advisory source routing, structured planning,
  concurrent researcher workers, bounded source-tool batching, and a dedicated writer replace
  the earlier three-role flow and move plan ownership out of the clarifier.
- **Work that continues from a completed report** — Users can ask questions against an existing
  report, create child-job rewrites, or run delta research with the parent report as context.
- **Portable skills, sandboxes, and durable files** — Provider-neutral sandbox execution,
  the new `deep-researcher-deploy` skill, expanded `deep-researcher-research` workflows, opt-in artifact capture, SQL or
  S3-compatible storage, and live or replayed Files-tab access turn generated files into durable
  outputs.
- **Sources, integrations, and policy controls** — OpenSearch and Azure AI Search join the knowledge backends;
  You.com adds four search and research tools, Nimble adds configurable web search, and the standalone public MCP server exposes submit/poll/report operations; per-user MCP
  OAuth, opt-in NeMo Guardrails middleware, and narrowly scoped async-content encryption add
  deployment controls without making them universal defaults.
- **Operations and user experience** — Async traces preserve the agent hierarchy, the source Helm
  chart honors the selected release namespace, and the UI improves concurrent-research activity,
  session recovery, and WebSocket reliability.

Deep Researcher Agent v2.2.0 is published on NVIDIA NGC as the
[`deep-researcher-agent` backend container](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/containers/aiq-agent/2.2.0),
[`deep-researcher-frontend` web container](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/containers/aiq-frontend/2.2.0),
and [`deep-researcher-web` Helm chart](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/helm-charts/aiq2-web/2.2.0).
Each artifact uses version `2.2.0`.

See the [changelog](CHANGELOG.md) for detailed release history; the linked feature docs describe
configuration and current limitations.


## Software Components

The checked-in default CLI and web profiles use these core components:

- [NVIDIA NeMo Agent Toolkit 1.8.0](https://docs.nvidia.com/nemo/agent-toolkit/latest/)
- [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) 0.6.5 or newer
- NVIDIA Nemotron 3.5 Lightning for intent classification and shallow research in the default profiles
- [NVIDIA Nemotron 3 Ultra](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b) for clarification and every deep-research role
- [Google Gemma 4 31B IT](https://build.nvidia.com/google/gemma-4-31b-it) (document summary, if used)
- [NVIDIA Nemotron 3 Embed 1B](https://build.nvidia.com/nvidia/nemotron-3-embed-1b) (embedding model for the knowledge layer, if used)
- [NVIDIA Nemotron 3 Nano Omni 30B A3B Reasoning](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) (vision-language model for the LlamaIndex knowledge layer, if used)
- [Tavily Search API](https://tavily.com/) for web search
- Serper, SerpAPI, or SearchAPI for Google Scholar paper search

> **Known hosted-serving limitation:** Nemotron 3.5 Lightning can intermittently produce citation-incomplete or
> malformed shallow drafts when served through NVIDIA API Catalog. Deep Researcher Agent fails closed instead of publishing those
> drafts. The Brev getting-started launchable uses Nemotron Ultra for shallow research; the general-purpose shipped
> profiles retain Lightning. See [Troubleshooting](docs/source/resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog)
> for details and the self-hosted Lightning option.

The shipped frontier profile, `configs/config_frontier_models.yml`, uses GPT-5.6 Luna for intent classification,
shallow research, source routing, and deep-research execution, with GPT-5.6 Sol for clarification, orchestration,
planning, and writing. Treat any bring-your-own model or modified profile as an experimental customization until the
complete workflow is evaluated with that exact model, prompt, hyperparameter, tool-calling, and structured-output
configuration. Refer to
[Configuration Files](#configuration-files); there is no single all-features profile.

## Target Audience

This project is for:

- **AI researchers and developers**: People building or extending agentic research workflows
- **Enterprise teams**: Organizations needing tool-augmented research with citation-backed research
- **NeMo Agent Toolkit users**: Developers looking to understand advanced multi-agent patterns

## Prerequisites

- Python 3.11–3.13
- [uv](https://github.com/astral-sh/uv) package manager
- NVIDIA API key from [NVIDIA AI](https://build.nvidia.com) (for NIM models)
- Node.js 22+ and npm (optional, for web UI mode)


**Optional requirements:**
- A web-search API key for the configured provider: Tavily, Exa, Nimble, or You.com
- A paper search API key for one of the supported providers: Serper (`SERPER_API_KEY`), SerpAPI (`SERPAPI_API_KEY`), or SearchAPI (`SEARCHAPI_API_KEY`)

> **Note:** Configure at least one data source (web search, paper search, or knowledge layer) to enable research functionality.

If these optional API keys are not provided, the agent continues to operate without the corresponding search capabilities. Refer to [Obtain API Keys](#obtain-api-keys) for details.

## Hardware Requirements

When using [NVIDIA API Catalog](https://build.nvidia.com/) (the default), inference runs on NVIDIA-hosted infrastructure and there are no local GPU requirements. The hardware references below apply only when self-hosting models via [NVIDIA NIM](https://docs.nvidia.com/nim/).

| Component | Default Model | Self-Hosted Hardware Reference |
|-----------|---------------|-------------------------------|
| LLM (intent classifier, shallow researcher) | `nvidia/nemotron-3.5-lightning-30b-a3b` | [Nemotron 3.5 Lightning](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard) |
| LLM (clarifier and all deep-research roles) | `nvidia/nemotron-3-ultra-550b-a55b` | [Nemotron 3 Ultra](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b) |
| Document summary (optional) | `google/gemma-4-31b-it` | [Gemma 4 31B IT](https://build.nvidia.com/google/gemma-4-31b-it) |
| Text embedding | `nvidia/nemotron-3-embed-1b` | [NeMo Retriever embedding support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html) |
| VLM (image/chart extraction, optional) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | [Nemotron 3 Nano Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) |
| Knowledge layer (Foundational RAG, optional) | -- | [RAG Blueprint support matrix](https://docs.nvidia.com/rag/latest/support-matrix.html) |

For detailed installation instructions, refer to [Installation -- Hardware Requirements](docs/source/get-started/installation.md#hardware-requirements).

## Architecture

Deep Researcher Agent uses a [LangGraph](https://www.langchain.com/langgraph)-based state machine with the following key components:

- **Orchestration node**: Classifies intent (meta vs. research), produces meta responses when needed, and sets depth (shallow vs. deep) in one step
- **Shallow research agent**: Bounded tool-augmented research optimized for speed
- **Deep research agent**: Multi-phase research with planning, iteration, and citation management

Each agent can be run individually or as part of the orchestrated workflow. For detailed architecture documentation, refer to [Architecture](docs/source/architecture/overview.md).

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent.git deep-researcher && cd deep-researcher
```

### Automated Setup

Run the setup script to initialize the environment:

```bash
./scripts/setup.sh
```

This script:
- Creates a Python virtual environment with uv
- Installs all Python dependencies (core, frontends, benchmarks, data sources)
- Installs UI dependencies (if Node.js is available)

### Manual Installation

For selective installation, install packages individually:

```bash
# Create and activate virtual environment
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install core with development dependencies
uv pip install -e ".[dev]"

# Install frontends (pick what you need)
uv pip install -e ./frontends/cli          # CLI frontend
uv pip install -e ./frontends/debug        # Debug console
uv pip install -e ./frontends/deep_researcher_api      # Unified API (includes debug)

# Install benchmarks (pick what you need)
uv pip install -e ./frontends/benchmarks/freshqa

# Install data sources (pick what you need)
uv pip install -e ./sources/tavily_web_search
uv pip install -e ./sources/exa_web_search
uv pip install -e ./sources/google_scholar_paper_search
uv pip install -e ./sources/nimble_web_search
uv pip install -e ./sources/you_com
uv pip install -e "./sources/knowledge_layer[llamaindex,foundational_rag]"
```

### Obtain API Keys


| API        | Environment Variable | Purpose                   | Required                                                    |
| ---------- | -------------------- | ------------------------- | ----------------------------------------------------------- |
| NVIDIA API | `NVIDIA_API_KEY`     | LLM inference through NIM | Yes                                                         |
| Tavily     | `TAVILY_API_KEY`     | Web search                | No (if not specified, agent continues without web search)   |
| Exa        | `EXA_API_KEY`        | Web search                | No (required only when Exa search is configured)            |
| Nimble     | `NIMBLE_API_KEY`     | Configurable web search   | No (required only when Nimble search is configured)         |
| You.com    | `YDC_API_KEY`        | Web, contents, and research APIs | No (required only when You.com tools are configured)   |
| Paper search | `SERPER_API_KEY`, `SERPAPI_API_KEY`, or `SEARCHAPI_API_KEY` | Academic paper search | No (choose one matching the configured provider) |


#### Obtain an NVIDIA API Key

1. Sign in to [NVIDIA Build](https://build.nvidia.com/)
2. Click on any model, then select "Deploy" > "Get API Key" > "Generate Key"

#### Obtain a Tavily API Key

1. Sign in to [Tavily](https://tavily.com/)
2. Navigate to your dashboard
3. Generate an API key

#### Obtain an Exa API Key

1. Sign in to [Exa](https://exa.ai/)
2. Create an API key from the dashboard
3. Add it to `deploy/.env` as `EXA_API_KEY`

Refer to the `exa_web_search` section in the
[Configuration Reference](docs/source/customization/configuration-reference.md) for workflow usage.

#### Obtain a You.com API Key

Follow the [You.com quickstart](https://you.com/docs/quickstart) to create an API key and add it to `deploy/.env` as
`YDC_API_KEY`. Refer to [You.com API Suite](docs/source/customization/you-com.md) for tool configuration.

#### Obtain a Paper Search API Key

Paper search supports three interchangeable providers. Set the `provider` field on the `paper_search` function in your workflow config (defaults to `serper`):

| Provider | Environment Variable | Sign-up |
|----------|----------------------|---------|
| Serper (default) | `SERPER_API_KEY` | [serper.dev](https://serper.dev/) |
| SerpAPI | `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com/) |
| SearchAPI | `SEARCHAPI_API_KEY` | [searchapi.io](https://www.searchapi.io/) |

Refer to [sources/google_scholar_paper_search/README.md](sources/google_scholar_paper_search/README.md) for configuration details.

### Set Up Environment Variables

Create a `.env` file in `deploy/` directory:

```bash
cp deploy/.env.example deploy/.env
```

Replace your API keys.

> **Note:** Depending on your usecase, deep research report quality can be enhanced by enabling searching across academic research papers. We use Serper for this. If you want to use paper search, follow the steps in the [Customization guide](docs/source/customization/tools-and-sources.md#disabling-a-tool) to enable it.

## Configuration Files

The `configs/` directory holds YAML workflow configs that define agents, tools, LLMs, and routing. Use the one that matches your run mode and data sources:

| Config | Models | Description |
|--------|--------|-------------|
| `config_cli_default.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra | CLI chat pipeline with Tavily and clarification; no knowledge backend. Paper search is a commented opt-in. |
| `config_web_default_llamaindex.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra; Gemma 4 summary | Default web/API chat pipeline with LlamaIndex/ChromaDB and Tavily. Paper search is commented out. |
| `config_web_frag.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra | Web/API and Helm base with Foundational RAG plus Tavily. Requires separately deployed RAG query and ingestion services. |
| `config_web_opensearch.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra; Nemotron 3 Embed | Web/API with built-in OpenSearch knowledge retrieval plus Tavily; supports self-hosted, `es`, and `aoss` authentication modes. |
| `config_web_azure_ai_search.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra; Nemotron 3 Embed | Web/API with Azure AI Search knowledge retrieval plus Tavily; supports API-key and Azure identity authentication. |
| `config_frontier_models.yml` | GPT Sol/Luna; Gemma 4 summary | LlamaIndex frontier profile using GPT Luna for intent, shallow research, source routing, and research, with GPT Sol for clarification, orchestration, planning, and writing. Requires `OPENAI_API_KEY`, `NVIDIA_API_KEY`, and `TAVILY_API_KEY` for the enabled Tavily tools. |
| `config_web_default_guardrails.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra; Gemma 4 summary | LlamaIndex profile with workflow Guardrails explicitly attached, shallow-agent Guardrails dynamically attached through `workflow_functions`, and async deep-agent Guardrails applied by the Deep Researcher Agent runner from the same target configuration. |
| `config_web_frag_mcp_auth.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra | Foundational RAG plus an opt-in protected per-user OAuth MCP source example. Requires a real MCP endpoint and shared token store. |
| `config_domain_routing_and_skills.yml` | Nemotron 3 Ultra; Gemma 4 summary | Direct deep-research profile with domain routing, DuckDuckGo news, Polymarket, enabled Serper paper search, LlamaIndex, built-in skills, and a fresh per-job Modal sandbox. |
| `config_openshell.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra; Gemma 4 summary | Experimental web/API skills profile with artifact capture, fail-closed policy attestation, and one OpenShell sandbox per deep-research job. |
| `config_mcp.yml` | Nemotron 3.5 Lightning; Nemotron 3 Ultra | Standalone MCP server. Public NIM + Tavily research with PostgreSQL-backed stateless submit/poll/report. Requires `NVIDIA_API_KEY`, `TAVILY_API_KEY`, and `DEEP_RESEARCHER_CHECKPOINT_DB`. |

## Ways to Run the Agents

The `frontends/` directory contains different interfaces for interacting with the agents. You can also run agents directly through the NeMo Agent Toolkit CLI.

### Command-line interface (CLI)

The CLI provides an interactive research assistant in your terminal:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run with the convenience script
./scripts/start_cli.sh

# Verbose logging
./scripts/start_cli.sh --verbose

# Or run directly with the NeMo Agent Toolkit CLI (dotenv loads deploy/.env into the environment)
dotenv -f deploy/.env run nat run --config_file configs/config_cli_default.yml --input "How do I install CUDA?"
```

The CLI frontend source is in `frontends/cli/`.

### Web UI

For a full web-based experience:

```bash
./scripts/start_e2e.sh
```

This starts:
- Backend API server at `http://localhost:8000`
- Frontend UI at `http://localhost:3000`

The web UI source is in `frontends/ui/`. Refer to [frontends/ui/README.md](frontends/ui/README.md) for more details.

#### Web UI with Docker Compose

You can also run the backend and UI with Docker Compose:

```bash
cd deploy/compose

# No-auth local setup (LlamaIndex default)
docker compose --env-file ../.env -f docker-compose.yaml up -d --build

# To select a different backend config, set BACKEND_CONFIG in deploy/.env, for example:
# BACKEND_CONFIG=/app/configs/config_web_frag.yml
```

For more details, refer to:
- `deploy/compose/README.md`

### Async Deep Research Jobs

For public endpoints, SSE replay, report follow-up, and durable artifact access, refer to the
[REST API documentation](docs/source/integration/rest-api.md).

### MCP Server

Expose Deep Researcher Agent to MCP clients through the standalone, stateless Streamable HTTP server:

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY}"
: "${TAVILY_API_KEY:?Set TAVILY_API_KEY}"
uv sync --project mcp --frozen
DEEP_RESEARCHER_CHECKPOINT_DB=postgresql://localhost/deep_researcher_jobs \
  uv run --project mcp --frozen deep-researcher-mcp-server
```

The endpoint defaults to `http://localhost:9001/mcp` and advertises exactly `submit_query`, `poll_query`, and
`get_final_report`. This public server intentionally has no authentication; job UUIDs are bearer capabilities and
the endpoint must not be exposed directly to an untrusted network. See [Expose Deep Researcher Agent as an MCP Server](docs/source/integration/mcp-server.md)
for the exact JSON protocol, health contracts, security model, and container deployment.

MCP is an independent uv project with its own `mcp/uv.lock`. The root lock remains compatible with NAT's
`cryptography<47` constraint, while the frozen MCP release/container profile pins `cryptography==50.0.0` as a
security hardening measure. The release-supported platform is Linux x86_64 with CPython 3.13, through either the
frozen source project or the release container. Other 64-bit source hosts are development-only; x86_64 macOS and
32-bit Windows are unsupported because `cryptography` 50 does not publish those wheels. Run the release container
on a supported 64-bit Linux/container host. The MCP package's local path dependency closure is not published as a
standalone wheel. See the
[MCP security policy](mcp/SECURITY.md#platform-compatibility) for the full platform contract.

### Benchmarks

To run agents in evaluation mode, refer to the [Evaluating the Workflow](#evaluating-the-workflow) section.

### Jupyter Notebooks

The `docs/notebooks/` directory contains a three-part series that walks through the blueprint from first run to full customization. Run them in order:

| # | Notebook | What it covers | Prerequisites |
|---|----------|----------------|---------------|
| 0 | [Getting Started with Deep Researcher Agent](docs/notebooks/0_Getting_Started_with_Deep_Researcher_Agent.ipynb) | Full blueprint overview — environment setup, orchestrated workflow (intent routing, shallow and deep research), and Docker Compose deployment | `NVIDIA_API_KEY`; optionally `TAVILY_API_KEY`, `SERPER_API_KEY` |
| 1 | [Deep Researcher — Web Search](docs/notebooks/1_Deep_Researcher_Web_Search.ipynb) | Deep researcher in depth — Python API, `nat run`, and end-to-end evaluation against the DeepResearch Bench with `nat eval` | Notebook 0 completed; `NVIDIA_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`; OpenAI or Gemini key for the judge model |
| 2 | [Deep Researcher — Customization](docs/notebooks/2_Deep_Researcher_Customization.ipynb) | Extending the deep researcher — adding paper search, assigning different LLMs per agent role, editing prompts, and enabling the knowledge layer | Notebooks 0 and 1 completed; `NVIDIA_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY` |


## Evaluating the Workflow

The `frontends/benchmarks/` directory contains evaluation pipelines for assessing agent performance.

### Available Benchmarks

| Benchmark | Description | Location |
|-----------|-------------|----------|
| Deep Research Bench | RACE and FACT evaluation for research quality | `frontends/benchmarks/deepresearch_bench/` |
| FreshQA | Factuality evaluation on time-sensitive questions | `frontends/benchmarks/freshqa/` |

### Running Evaluations

### Step 1: Install the dataset

The dataset files are not included in the repository. We have included a script to retrieve them from the [Deep Research Bench Github Repository](https://github.com/Ayanami0730/deep_research_bench/tree/main) and format them for the NeMo Agent Toolkit evaluator.

To download the dataset files, run the following script:

```bash
python frontends/benchmarks/deepresearch_bench/scripts/download_drb_dataset.py
```

### Step 2: Generate reports using NAT evaluation harness

```bash
dotenv -f deploy/.env run nat eval --config_file frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml
```

### Step 3: Convert the output into a compatible format
```bash
python frontends/benchmarks/deepresearch_bench/scripts/export_drb_jsonl.py --input <path to your workflow_output.json> --output <path to the output file you want to create with .jsonl extension>
```

### Step 4: Run evaluation
Follow instructions in the [Deep Research Bench Github Repository](https://github.com/Ayanami0730/deep_research_bench/tree/main) to run evaluation and obtain scores.


### Optional: Phoenix Tracing

If your config enables Phoenix tracing, start the Phoenix server before running `nat eval`.

Start server (separate terminal):

```bash
uvx --from arize-phoenix phoenix serve
```

For detailed benchmark documentation, refer to:
- [Deep Research Bench README](frontends/benchmarks/deepresearch_bench/README.md)
- [FreshQA README](frontends/benchmarks/freshqa/README.md)

## Development

For development, contribution, and documentation, refer to:

- **[Development and Contributing](docs/source/contributing/index.md)**: Setup, testing, PR workflow, sign-off/DCO
- **[Tutorial Notebooks](docs/notebooks/)**: Getting started overview, deeper dive, and customization notebooks
- **[Architecture](docs/source/architecture/overview.md)**: Component details and data flow
- **[Customization](docs/source/customization/index.md)**: Configuration and customization options
- **[Knowledge Layer Setup](sources/knowledge_layer/KNOWLEDGE-LAYER-SETUP.md)**: RAG backends and document ingestion
- **[Agent Skills](docs/source/integration/agent-skills.md)**: Install the portable Deep Researcher Agent research skill in compatible coding harnesses
- **[Skills and Sandbox Example](docs/source/examples/skills-sandbox/index.md)**: Run deep research with built-in skills and Modal sandbox execution
- **[Profiling and Cost Analysis](docs/source/profiling/index.md)**: Generate tokenomics and latency reports from Relay ATOF traces
- **[Docs index](docs/README.md)**: Full documentation list and component docs
- **[Changelog](docs/source/resources/changelog.md)**: Version history and changes

## Roadmap

The checkboxes below track implementation in the current branch; a checked item does not by itself
indicate availability in a published release.

- [x] **[NeMo Guardrails](docs/source/customization/guardrails.md) Integration:** Opt-in middleware for selected workflow, shallow-researcher, and deep-researcher boundaries.
- [ ] **[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) Integration:** Reduce latency via priority scheduling at scale.
- [x] **Per-user MCP OAuth:** Connect each signed-in user to protected MCP data sources through the UI.
- [x] **Skills & Sandboxing:** Support host-side, role-assigned deep-research skills and provider-neutral execution. Modal and OpenShell create one physical sandbox per deep-research job.
- [x] **Report Follow-up and Rewriting:** Answer against a completed report, submit cosmetic child rewrites, and run delta research with parent-report context.
- [x] **Durable Sandbox Artifacts:** Opt-in manifest checkpoints after successful sandbox commands, terminal harvesting on success/failure, SQL or S3-compatible byte storage, and live/replayed Files-tab access.
- [ ] **Custom Skill Management:** Add UI and lifecycle controls for user-provided skill bundles.
- [ ] **Dynamic Model Routing:** Allow sub-agents to automatically select the optimal model per task.
- [ ] **Resource Management:** Implement configurable token caps and tool-call budgets.
- [x] **Expanded Web Search:** Tavily, Exa, You.com, and Nimble integrations with configurable source selection.
- [ ] **Additional Search Providers:** Add integration examples including Perplexity.
- [ ] **Multimedia Output:** Embed audio, video, and images directly into reports.
- [ ] **Voice-to-Text Input:** Integrate [NVIDIA Riva](https://developer.nvidia.com/riva) for hands-free accessibility.

## Security Considerations

- The Deep Researcher Agent Blueprint is shared as a reference and is provided "as is". The security in the production environment is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats; define the trust boundaries, implement logging and monitoring capabilities, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment up to date, ensure the containers/source code are secure and free of known vulnerabilities.
- A robust frontend that handles AuthN & AuthZ is highly recommended. Missing AuthN & AuthZ will result in ungated access to customer models if directly exposed e.g. the internet, resulting in either cost to the customer, resource exhaustion, or denial of service.
- Deep Researcher Agent includes opt-in [NeMo Guardrails middleware](docs/source/customization/guardrails.md) for selected workflow and agent boundaries. Guardrails are not enabled universally; operators must attach and test the policies required for their deployment.
- Optional [async job content encryption](docs/source/deployment/content-encryption.md) protects final job output and selected artifact-event content only. It is off by default and is not full database-level job-content encryption.
- The Deep Researcher Agent Blueprint doesn't require any privileged access to the system.
- Deep research skills can invoke sandboxed code execution for analysis workflows. Keep sandbox credentials, quotas, lifecycle cleanup, attestation, and network policy aligned with your deployment's trust boundaries. Shared OpenShell attachment is an explicit debug-only mode and is not job-isolated.
- End users are responsible for ensuring the availability of their deployment.
- End users are responsible for building, and patching, the container images to keep them up to date.
- The end users are responsible for ensuring that OSS packages used by the developer blueprint are current.
- The built-in workflow and agent logger paths redact raw prompts, tool payloads, and model completions. Enabled source adapters, external middleware, model and tool providers, tracing exporters, and reverse proxies can have separate request-logging and retention behavior; audit and configure every enabled component independently, and keep production logs access-controlled.


## License

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use, found in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).

GOVERNING TERMS: Deep Researcher Agent blueprint software and materials are governed by the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
