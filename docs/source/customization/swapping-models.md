<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Swapping Models

LLMs are defined in the `llms` section and referenced by agents and tools. You can swap NIM models, change parameters, or add alternative providers.

## Shipped Profiles and Validation Boundary

AI-Q 2.2 ships these exact model assignments and parameters:

| Configuration | Intent and shallow roles | Clarification and deep-research roles | Optional summary role |
| --- | --- | --- | --- |
| `configs/config_cli_default.yml`, `configs/config_web_default_llamaindex.yml` | `nvidia/nemotron-3.5-lightning-30b-a3b` | `nvidia/nemotron-3-ultra-550b-a55b` for clarification, orchestration, source routing, research, planning, and writing | `google/gemma-4-31b-it` in the web profile |
| `configs/config_frontier_models.yml` | `gpt-5.6-luna` | `gpt-5.6-sol` for clarification, orchestration, planning, and writing; `gpt-5.6-luna` for source routing and research | `google/gemma-4-31b-it` |

The checked-in files define the documented compatibility boundary; they are not a substitute for an end-to-end run
against your provider endpoints and credentials. Changing a model, endpoint, role assignment, prompt, or inference
parameter creates a custom profile outside that boundary. OpenAI-compatible transport alone does not establish
workflow compatibility. Other bring-your-own models can require provider-specific prompt, hyperparameter,
tool-calling, and structured-output tuning, and should be treated as experimental until the complete workflow is
evaluated in that exact configuration.

```{warning}
Nemotron 3.5 Lightning can intermittently produce citation-incomplete or malformed shallow drafts when served through
the NVIDIA API Catalog endpoint. AI-Q fails closed rather than publishing those drafts. See
[Nemotron 3.5 Lightning on NVIDIA API Catalog](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog)
for the validated mitigation choices. Model weights alone do not define the compatibility boundary; the serving
profile is part of the deployment contract.
```

**Example: NIM model (default)**

```yaml
llms:
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
```

**Example: NIM with thinking (for example, for deep research)**

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
    chat_template_kwargs:
      enable_thinking: true
```

**Model roles:** The workflow maps LLMs to roles (orchestrator, researcher, planner, etc.) through the `LLMProvider`. In YAML you assign which named LLM each agent uses (for example, `orchestrator_llm: nemotron_ultra_llm`, `llm: nemotron_lightning_agent_llm`). Use different keys in `llms` and point agents at them to swap models per role.

## Using Downloadable NIMs (Self-Hosted)

By default, configs use NVIDIA's hosted NIM API (`integrate.api.nvidia.com`). You can also run NIMs locally or on your own infrastructure for lower latency, data privacy, or offline use.

### 1. Find Downloadable NIMs

Browse available NIMs at [build.nvidia.com](https://build.nvidia.com/explore/discover). Each model page includes a "Self-Host" tab with Docker pull commands and setup instructions.

### 2. Run a NIM Locally

```bash
# Example: run Nemotron on port 8080
docker run --gpus all -p 8080:8000 \
  nvcr.io/nim/nvidia/nemotron-3-ultra-550b-a55b:latest
```

Refer to the [NIM documentation](https://docs.nvidia.com/nim/) for GPU requirements, environment variables, and multi-GPU setup.

### 3. Update Your Config

Change `base_url` to point to your local NIM instance instead of the hosted API. Use the model identifier returned by the local NIM's `/v1/models` endpoint; it can differ from the hosted API identifier. You can remove `api_key` since local NIMs typically don't require one.

```yaml
llms:
  local_ultra_llm:
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "http://localhost:8080/v1"    # local NIM
    temperature: 0.2
    max_tokens: 16384
    num_retries: 5
```

```{note}
**Hosted Endpoint Availability:** The default profiles use Nemotron 3.5 Lightning for intent and shallow research, and Nemotron 3 Ultra for clarification and every deep-research role. Shared hosted endpoints can have limited availability during high demand (HTTP 429 or 503 responses), and the API Catalog Lightning serving profile has a separate [shallow citation-output limitation](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog). For production deployments requiring consistent throughput, refer to the [self-hosting guidance](../resources/troubleshooting.md#nemotron-hosted-endpoint-availability).
```

**Apply the config change:** how you get the edited config in front of the
running backend depends on the deployment method:

| Method | How to apply |
|--------|-------------|
| CLI | Edit the `configs/*.yml` file passed to `--config_file`. |
| Docker Compose | Edit the `configs/*.yml` file (bind-mounted read-only into the container); select it with `BACKEND_CONFIG`. |
| Helm / Kubernetes | Supply the edited config as a custom ConfigMap or inline via `aiq.apps.backend.config`; see [Kubernetes → Configuration](../deployment/kubernetes.md#configuration). |

```{note}
In Kubernetes, `base_url` must be an in-cluster address (for example,
`http://<nim-service>.<namespace>.svc.cluster.local:8000/v1`), not
`http://localhost:8080/v1` -- the backend pod cannot reach a NIM running
outside the cluster network via `localhost`.
```

You can mix hosted and local NIMs in the same config -- for example, use a hosted endpoint for shallow research and a local downloadable Ultra NIM for deep research:

```yaml
llms:
  hosted_shallow_llm:
    _type: nim
    model_name: nvidia/nemotron-3.5-lightning-30b-a3b
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 8192
    parallel_tool_calls: false
    chat_template_kwargs:
      enable_thinking: true

  local_ultra_llm:
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "http://localhost:8080/v1"
    temperature: 0.2
    max_tokens: 16384

functions:
  shallow_research_agent:
    _type: shallow_research_agent
    llm: hosted_shallow_llm
    # ...

  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: local_ultra_llm
    # ...
```

## NVIDIA Hosted API Considerations

The default configs use `https://integrate.api.nvidia.com/v1`, NVIDIA API Catalog's OpenAI-compatible endpoint. This service is convenient for getting started but has limitations that matter for production use and long-running evaluations.

### Known Limitations

**Model availability**

Models served through `integrate.api.nvidia.com` are subject to change:

- Model versions may be updated or deprecated without notice
- Preview model identifiers can change when a model reaches general availability
- Check [build.nvidia.com](https://build.nvidia.com/explore/discover) for current model availability and changelogs

### Mitigation Strategies

| Issue | Recommended action |
|-------|-------------------|
| Rate limiting during benchmarks | Lower `max_concurrency` in eval config; add delays between requests |
| Frequent `504` timeouts | Increase `timeout` parameter; switch to a self-hosted NIM for that role |
| Model removed or deprecated | Update `model_name` to the replacement model; pin a tested model version when possible |
| Production reliability requirements | Use self-hosted NIMs or an NVIDIA Enterprise API agreement with SLA guarantees |

For self-hosted NIM setup, refer to [Using Downloadable NIMs](#using-downloadable-nims-self-hosted) above.
