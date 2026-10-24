<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Installation

This guide walks through setting up the Deep Researcher Agent blueprint for local development. For containerized or production deployments, refer to [Deployment](../deployment/index.md).

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 -- 3.13 | 3.13 recommended |
| [uv](https://github.com/astral-sh/uv) | 0.11.25+ | Python package manager (installed automatically by the setup script if missing; CI uses 0.11.26) |
| Git | 2.x+ | |
| Node.js | 22+ | Optional -- only needed for the web UI |

You also need at least one LLM API key. Refer to [API key setup](#api-key-setup) below.


### Hardware Requirements

When using [NVIDIA API Catalog](https://build.nvidia.com/) (the default), inference runs on NVIDIA-hosted infrastructure and there are no local GPU requirements. The hardware requirements below apply only when self-hosting models via [NVIDIA NIM](https://docs.nvidia.com/nim/).

| Component | Default Model | Self-Hosted Hardware Reference |
|-----------|---------------|-------------------------------|
| LLM (intent classifier, shallow researcher) | `nvidia/nemotron-3.5-lightning-30b-a3b` | [Nemotron 3.5 Lightning](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard) |
| LLM (clarifier and all deep-research roles) | `nvidia/nemotron-3-ultra-550b-a55b` | [Nemotron 3 Ultra](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b) |
| Document summary (optional) | `google/gemma-4-31b-it` | [Gemma 4 31B IT](https://build.nvidia.com/google/gemma-4-31b-it) |
| Text embedding | `nvidia/nemotron-3-embed-1b` | [NeMo Retriever embedding support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html) |
| VLM (image/chart extraction, optional) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | [Nemotron 3 Nano Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) |
| Knowledge layer (Foundational RAG, optional) | -- | [RAG Blueprint support matrix](https://docs.nvidia.com/rag/latest/support-matrix.html) |

```{warning}
The NVIDIA API Catalog serving profile for Nemotron 3.5 Lightning has a known shallow citation-output limitation.
Deep Researcher Agent fails closed rather than publishing a citation-incomplete draft. The Brev getting-started launchable therefore
uses Nemotron Ultra for shallow research while retaining Lightning for intent classification. See
[Troubleshooting](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog) for details and the
self-hosted Lightning option.
```

## Automated Setup (Recommended)

The setup script handles everything -- virtual environment, Python dependencies, and UI dependencies:

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent.git deep-researcher
cd deep-researcher

./scripts/setup.sh
```

The script performs the following steps:

1. Installs `uv` if not already present and rejects versions older than 0.11.25
2. Creates a Python 3.13 virtual environment at `.venv/`
3. Installs the core package with dev dependencies
4. Installs all frontends (CLI, debug console, API server)
5. Installs benchmark packages (freshqa, deepsearch_qa)
6. Installs the data source plugins (Tavily, Exa, Nimble, Firecrawl, You.com, Google Scholar) and the LlamaIndex and Foundational RAG knowledge extras
7. Sets up pre-commit hooks
8. Copies `deploy/.env.example` to `deploy/.env` if no `.env` file exists
9. Installs UI npm dependencies (if Node.js is available)

After the script completes, activate the virtual environment:

```bash
source .venv/bin/activate
```

## Manual Setup

If you prefer to install components selectively, follow these steps.

### 1. Clone the Repository

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent.git deep-researcher
cd deep-researcher
```

### 2. Create the Virtual Environment

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the core package and only the frontends, benchmarks, and data sources you need:

```bash
# Core with development dependencies
uv pip install -e ".[dev]"

# Frontends (pick what you need)
uv pip install -e ./frontends/cli          # CLI interface
uv pip install -e ./frontends/debug        # Debug console
uv pip install -e ./frontends/deep_researcher_api      # Unified API server (includes debug)

# Data sources (pick what you need)
uv pip install -e ./sources/tavily_web_search
uv pip install -e ./sources/exa_web_search
uv pip install -e ./sources/nimble_web_search
uv pip install -e ./sources/firecrawl_web_search
uv pip install -e ./sources/you_com
uv pip install -e ./sources/duckduckgo_news_search
uv pip install -e ./sources/polymarket_prediction_market
uv pip install -e ./sources/google_scholar_paper_search
uv pip install -e "./sources/knowledge_layer[llamaindex,foundational_rag]"
# Or include the OpenSearch backend as well:
uv pip install -e "./sources/knowledge_layer[llamaindex,foundational_rag,opensearch]"

# Benchmarks (optional)
uv pip install -e ./frontends/benchmarks/freshqa
uv pip install -e ./frontends/benchmarks/deepsearch_qa
```

### 4. Set Up Pre-Commit Hooks (Development)

```bash
pre-commit install
```

## API Key Setup

Deep Researcher Agent needs API keys to access LLMs and search providers. Create an environment file from the provided template:

```bash
cp deploy/.env.example deploy/.env
```

Then edit `deploy/.env` and fill in your keys.

### Required Keys

| Variable | Provider | How to obtain |
|----------|----------|---------------|
| `NVIDIA_API_KEY` | [NVIDIA Build](https://build.nvidia.com/) | Sign in, click any model, select Deploy > Get API Key > Generate Key |

### Optional Keys

| Variable | Provider | Purpose |
|----------|----------|---------|
| `TAVILY_API_KEY` | [Tavily](https://tavily.com/) | Web search (Tavily provider) |
| `EXA_API_KEY` | [Exa](https://exa.ai/) | Web search (Exa provider) |
| `NIMBLE_API_KEY` | [Nimble](https://nimbleway.com/) | Web search (Nimble provider) |
| `FIRECRAWL_API_KEY` | [Firecrawl](https://www.firecrawl.dev/search?utm_source=aiq&utm_medium=integration) | Web search with page Markdown (Firecrawl provider); optional, raises rate limits (keyless use has daily per-IP limits) |
| `SERPER_API_KEY` | [Serper](https://serper.dev/) | Google Scholar paper search with `provider: serper` (the default) |
| `SERPAPI_API_KEY` | [SerpAPI](https://serpapi.com/) | Google Scholar paper search with `provider: serpapi` |
| `SEARCHAPI_API_KEY` | [SearchAPI](https://www.searchapi.io/) | Google Scholar paper search with `provider: searchapi` |

At minimum, you need `NVIDIA_API_KEY` for LLM inference and a credential for the web provider selected by your config.
Paper search requires one provider-specific key. It is commented out in the standard CLI and web profiles, while
`configs/config_domain_routing_and_skills.yml` enables the default Serper provider. DuckDuckGo News and Polymarket use
public endpoints and do not require API keys.

OpenSearch uses endpoint-specific authentication rather than one universal API key. Install the `opensearch` extra,
start from `configs/config_web_opensearch.yml`, and configure `none`, `basic`, or SigV4 authentication. For Amazon
OpenSearch Serverless, follow the [AOSS deployment guide](../deployment/aws-opensearch-serverless.md).

## Verify Installation

Confirm that the NeMo Agent Toolkit CLI is available and can find the project plugins:

```bash
# Must use the project venv, not the system nat
.venv/bin/nat --help
```

You should observe the `nat` CLI help output with available commands (`run`, `serve`, `eval`, etc.).

To verify plugins are registered:

```bash
.venv/bin/nat run --help
```

This should list available workflow configurations.

## Building the Documentation

The project documentation is built with [Sphinx](https://www.sphinx-doc.org/) and uses MyST-Parser for Markdown support. To build the HTML docs locally:

```bash
# Install docs dependencies and build in one step
uv run --extra docs sphinx-build -M html docs/source docs/build
```

The generated site is written to `docs/build/html/`. Open `docs/build/html/index.html` in a browser to view it.

If you already have the virtual environment activated with docs extras installed, you can also run:

```bash
sphinx-build -M html docs/source docs/build
```

## Next Steps

- **[Quick Start](./quick-start.md)** -- Run your first research query in 5 minutes
- **[Developer Guide](./developer-guide.md)** -- Recommended reading path through the documentation
- **[Deployment](../deployment/index.md)** -- Docker Compose deployment
