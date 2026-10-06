<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: CLI with Local NIMs

Run the full research pipeline interactively from the command line using self-hosted NIM models. This is useful for air-gapped environments, custom fine-tuned models, or reducing latency by running inference locally.

This example is based on `configs/config_cli_default.yml` with modifications to point LLMs at locally-hosted NIM containers. To use local NIMs, copy the default config to `configs/config_cli_local_nims.yml` and update the `base_url` fields as shown below.

## Prerequisites

You need Docker and NVIDIA GPUs with sufficient VRAM to run the NIM containers. Check the downloadable Nemotron 3 Ultra model card and support matrix for current self-hosted hardware requirements.

## Running NIM Containers

Authenticate to NGC, then start the NIM model server locally using Docker:

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Pull and run the Nemotron NIM container
# Adjust --gpus and CUDA_VISIBLE_DEVICES for your hardware
docker run -d \
  --name nemotron-nim \
  --gpus all \
  -p 8001:8000 \
  -e NGC_API_KEY="${NGC_API_KEY}" \
  nvcr.io/nim/nvidia/nemotron-3-ultra-550b-a55b:latest
```

Verify the model is ready:

```bash
curl http://localhost:8001/v1/models
```

## Configuration

```yaml
# configs/config_cli_local_nims.yml
# Copy of config_cli_default.yml with base_url pointing to local NIM containers

general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO

# ===========================================================================
# LLMs - pointing to local NIM containers
# ===========================================================================
# The key difference from cloud configs: base_url points to localhost
# instead of integrate.api.nvidia.com. No NVIDIA_API_KEY is needed for
# inference (only for pulling the container image).
llms:
  local_ultra_llm:
    _type: nim
    # Use the identifier returned by this local NIM's /v1/models endpoint.
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "http://localhost:8001/v1"   # <-- Local NIM
    temperature: 0.2
    top_p: 0.7
    max_tokens: 16384
    num_retries: 3
    chat_template_kwargs:
      enable_thinking: false

  local_ultra_writer_llm:
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "http://localhost:8001/v1"   # <-- Local NIM
    temperature: 0.2
    top_p: 0.7
    max_tokens: 32768
    num_retries: 3
    chat_template_kwargs:
      enable_thinking: false

# ===========================================================================
# Functions
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

  paper_search_tool:
    _type: paper_search
    max_results: 5
    serper_api_key: ${SERPER_API_KEY}

  intent_classifier:
    _type: intent_classifier
    llm: local_ultra_llm
    tools:
      - web_search_tool
      - paper_search_tool

  clarifier_agent:
    _type: clarifier_agent
    llm: local_ultra_llm
    tools:
      - web_search_tool
    max_turns: 3
    log_response_max_chars: 2000

  shallow_research_agent:
    _type: shallow_research_agent
    llm: local_ultra_llm
    tools:
      - web_search_tool
    max_llm_turns: 10
    max_tool_iterations: 5

  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: local_ultra_llm
    source_router_llm: local_ultra_llm
    planner_llm: local_ultra_llm
    researcher_llm: local_ultra_llm
    writer_llm: local_ultra_writer_llm
    tools:
      - paper_search_tool
      - advanced_web_search_tool

# ===========================================================================
# Workflow
# ===========================================================================
workflow:
  _type: chat_deepresearcher_agent
  enable_escalation: true
  enable_clarifier: true
  checkpoint_db: ${DEEP_RESEARCHER_CHECKPOINT_DB:-./checkpoints.db}
```

## Required Environment Variables

```bash
# Required to authenticate to NGC, pull the image, and download model artifacts
export NGC_API_KEY="nvapi-..."  # pragma: allowlist secret

# Web search still requires API keys (runs externally)
export TAVILY_API_KEY="tvly-..."   # pragma: allowlist secret
export SERPER_API_KEY="..."
```

## How to Run

```bash
# Interactive CLI mode (recommended)
./scripts/start_cli.sh --config_file configs/config_cli_local_nims.yml

# Single query mode
dotenv -f deploy/.env run .venv/bin/nat run \
  --config_file configs/config_cli_local_nims.yml \
  --input "What are the latest advances in quantum error correction?"
```

The CLI script starts an interactive session. Type your research query and the system will:

1. Classify the intent (shallow vs deep)
2. Ask a focused clarification only when the request is genuinely ambiguous; the clarifier does not ask you to approve a plan
3. For deep queries, build an internal structured plan and run independent research queries concurrently
4. Show research tool activity in real time
5. Have the writer synthesize the captured evidence into the requested output shape

### Example Session

```
> What are the latest advances in quantum error correction?

[Intent: shallow]
[Tool: web_search] Searching: "quantum error correction advances 2026"
[Tool: web_search] Searching: "quantum error correction codes recent breakthroughs"
[Tool: web_search] Found 5 results

# Quantum Error Correction: Recent Advances

...
```

## Tips for Local NIMs

- **GPU memory**: Monitor with `nvidia-smi`. Size GPUs for the
  `nvidia/nemotron-3-ultra-550b-a55b` NIM using the model card and support matrix.
- **Startup time**: NIM containers take 2--5 minutes to load the model on first start. Wait until `/v1/models` returns a response.
- **Multiple GPUs**: Use `--gpus '"device=0,1"'` to spread across GPUs, or run separate containers per GPU for different model roles.
- **Networking**: If running inside Docker Compose, use container names instead of `localhost` for `base_url`.
- **num_retries**: Lower retry counts (3 vs 5) are appropriate for local NIMs since failures are less likely to be transient.
