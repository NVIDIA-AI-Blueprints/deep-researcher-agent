<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Quick Start

Run your first research query in 5 minutes. This guide assumes you have already completed [Installation](./installation.md).

## Step 1: Set API Keys

If you have not already, create the environment file and add your keys:

```bash
cp deploy/.env.example deploy/.env
```

Edit `deploy/.env` and set at minimum:

```bash
NVIDIA_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...
# Or, to use Exa instead of Tavily for web search:
# EXA_API_KEY=...
# Or, to use Nimble instead of Tavily for web search:
# NIMBLE_API_KEY=...
# Or, to use Firecrawl for web search:
# FIRECRAWL_API_KEY=...
```

## Step 2: Choose a Mode

The Deep Researcher Agent blueprint supports two primary modes for interactive use: a terminal-based CLI and a browser-based web UI.

### Option A: CLI Mode

The CLI provides an interactive research assistant in your terminal.

```bash
source .venv/bin/activate

# Using the convenience script
./scripts/start_cli.sh

# Or run directly with the custom CLI entry point
dotenv -f deploy/.env run .venv/bin/deep-researcher-research --config_file configs/config_cli_default.yml
```

> **Note:** `start_cli.sh` runs `.venv/bin/deep-researcher-research` (a custom CLI entry point registered by this project), not `nat run`. The custom entry point adds interactive features like conversation history that are not part of the standard NeMo Agent Toolkit CLI.

You should observe the agent start up and present an input prompt where you can type questions.

### Option B: Web UI Mode

For a browser-based experience with a chat interface:

```bash
source .venv/bin/activate

./scripts/start_e2e.sh
```

This starts:

- **Backend API** at `http://localhost:8000`
- **Frontend UI** at `http://localhost:3000`

Open [http://localhost:3000](http://localhost:3000) in your browser.

```{note}
The web UI requires Node.js 22+ and npm. If these were available during `./scripts/setup.sh`, UI dependencies are already installed. Otherwise, run `cd frontends/ui && npm ci` first.
```

```{tip}
**Running on a remote VM?** If you access the VM via SSH, you need to forward ports 3000 and 8000 to your local machine: `ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@your-vm-host`. Refer to [Troubleshooting — VM / Remote Development](../resources/troubleshooting.md#vm--remote-development) for details.
```

## Step 3: Ask a Question

Try one of these example queries to observe the system in action:

**Shallow research** (fast, concise answers):

> What is CUDA and how does it relate to GPU programming?

> What are the key differences between TCP and UDP?

**Deep research** (detailed, report-style output):

> Compare the transformer architectures used in GPT-4 and Gemini, including their training approaches, parameter counts, and benchmark performance.

> What are the current approaches to solving the protein folding problem, and how do AlphaFold and RoseTTAFold compare?

### What to Expect

- **Shallow queries** typically produce a concise answer with inline citations and source links in under a minute.
- **Deep queries** trigger a multi-phase research process (planning, research, synthesis) that produces a structured report with a table of contents, inline citations, and a references section. This commonly takes several minutes and can exceed ten minutes depending on the query, models, providers, and data sources.

The system automatically routes queries to the appropriate depth based on complexity. You do not need to specify shallow vs. deep -- the orchestration node decides for you.

## Example CLI Session

```
$ ./scripts/start_cli.sh
============================================
  Deep Researcher Agent Blueprint - CLI Mode
============================================

Config: config_cli_default.yml
Verbose: OFF (use -v to enable)

Type 'exit' or 'quit' to exit
--------------------------------------------

   NVIDIA Deep Researcher Agent Blueprint
   Research Assistant powered by NVIDIA NeMo Agent Toolkit

Deep Researcher Agent initialized!
Type 'exit', 'quit', or 'q' to quit.

You: What is the NVIDIA NeMo Agent Toolkit?

  Answer
  The NVIDIA NeMo Agent Toolkit (NAT) is a framework for building and deploying
  AI agents and multi-agent systems...

  Sources:
  [1] https://docs.nvidia.com/nemo/agent-toolkit/latest/ - NeMo Agent Toolkit Documentation
  [2] https://developer.nvidia.com/nemo - NVIDIA NeMo Overview
```

## Verbose Logging

To view detailed agent execution logs (tool calls, routing decisions, LLM interactions):

```bash
./scripts/start_cli.sh --verbose
```

## What's Next

Now that you have the system running, explore these topics:

- **[Review recent changes](../resources/changelog.md)** -- See what has changed and choose among the focused [configuration profiles](../customization/configuration-reference.md#provided-config-files)
- **[Agent Skills](../integration/agent-skills.md)** -- Deploy and call Deep Researcher Agent from a compatible coding harness
- **[Architecture Overview](../architecture/overview.md)** -- Understand how the orchestrator, shallow researcher, and deep researcher work together
- **[Customization](../customization/index.md)** -- Swap models, configure tools, adjust prompts, and tune agent behavior
- **[Deployment](../deployment/index.md)** -- Run with Docker Compose
- **[Evaluation](../evaluation/index.md)** -- Measure research quality with built-in benchmarks
