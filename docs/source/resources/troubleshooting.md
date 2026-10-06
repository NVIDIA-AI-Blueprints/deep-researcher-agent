<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

Common issues and solutions for the Deep Researcher Agent blueprint.

## Installation Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: deep_researcher_agent` | Package not installed in editable mode | `uv pip install -e .` |
| `nat` command not found | Using system `nat` instead of venv | Use `.venv/bin/nat` or activate the venv |
| NeMo Agent Toolkit plugins not found | Plugins not installed | `uv pip install -e .` to register entry points |
| Pre-commit hook failures | Missing pre-commit setup | `pre-commit install && pre-commit run --all-files` |
| `ormsgpack` attribute error | Version conflict with [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | `uv pip install "ormsgpack>=1.5.0"` |

## API Key Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `[404] Not found for account` | Invalid or expired NVIDIA API key | Regenerate key at [build.nvidia.com](https://build.nvidia.com) |
| `Gateway timeout (504)` | Model endpoint overloaded or unavailable | Retry, or switch to a different model in config |
| Tavily search returns empty | Invalid `TAVILY_API_KEY` | Verify key at [tavily.com](https://tavily.com) |
| You.com tools return an unavailable or 401 error | Missing or invalid `YDC_API_KEY` | Create or verify the key using the [You.com quickstart](https://you.com/docs/quickstart) and restart Deep Researcher Agent |
| Exa search returns empty or 401 | Invalid or missing `EXA_API_KEY` | Verify key at [exa.ai](https://exa.ai) |
| Nimble search returns empty or 401 | Invalid or missing `NIMBLE_API_KEY` | Verify the key through [Nimble](https://nimbleway.com/) |
| Nimble search returns 403 with "enterprise" | `search_depth: fast` requires an Enterprise plan | Switch to `search_depth: lite` (default) or `deep`, or upgrade your Nimble plan |
| Serper search fails | Missing `SERPER_API_KEY` | Set key or remove `paper_search_tool` from config |

## Runtime Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Agent hangs on deep research | LLM timeout or rate limit | Inspect Relay logs/traces and check LLM API availability and rate limits |
| HTTP 429 or 503 on deep research | Nemotron hosted endpoint availability | Retry after a short delay, reduce concurrency, or follow the [self-hosting guidance](#nemotron-hosted-endpoint-availability) for consistent throughput |
| Intermittent shallow-research failure with Nemotron 3.5 Lightning on NVIDIA API Catalog | The hosted serving profile can produce citation-incomplete or malformed final drafts | Use Nemotron Ultra for the shallow role, or use a validated self-hosted Lightning serving profile; see [Nemotron 3.5 Lightning on NVIDIA API Catalog](#nemotron-35-lightning-on-nvidia-api-catalog) |
| Shallow research returns generic answers | Insufficient tool calls | Increase `max_tool_iterations` (default: 5) |
| Clarifier keeps asking questions | Too many clarification turns | Reduce `max_turns`, or set `enable_clarifier: false` in the workflow to disable clarification |
| SSE stream disconnects | Network timeout | Client auto-reconnects using `last_event_id`; refer to [Data Flow](../architecture/data-flow.md) |
| Job status stuck on RUNNING | Dask worker crashed | Check Dask logs; the ghost job reaper will eventually mark it FAILURE |
| OpenShell setup, attestation, readiness, or deletion fails | Gateway, version, policy/config, image, or service-owner mismatch | Follow the canonical [OpenShell inspection and troubleshooting guide](../deployment/openshell.md#inspection-and-troubleshooting) |

## Nemotron Hosted Endpoint Availability

Nemotron 3.5 Lightning (`nvidia/nemotron-3.5-lightning-30b-a3b`) and Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`) are compatible and tested with Deep Researcher Agent, but their NVIDIA-hosted endpoints can have limited availability during high demand. During peak periods you may observe:

- Elevated latency or timeouts on LLM inference calls
- HTTP 429 (rate-limited) or 503 (service unavailable) responses from the Build API
- Degraded agent workflow performance due to upstream model availability

**Default Configuration:** The default configs use Nemotron 3.5 Lightning for intent classification and shallow research, and Nemotron 3 Ultra for clarification and all deep-research roles. If a hosted endpoint is saturated, retry after a short delay, reduce concurrency, or self-host a downloadable model for consistent throughput.

### Recommended Mitigation: Self-Host the Affected Model

For production and staging deployments that require consistent throughput and low-latency inference, self-host a downloadable NVIDIA NIM rather than relying on shared endpoints. Preview endpoint availability and downloadable NIM availability do not necessarily move in lockstep; verify the current model card before choosing an image.

- [Self-host Nemotron 3.5 Lightning 30B A3B](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b?nim=self-hosted) for the default intent and shallow-research roles
- [Self-host Nemotron 3 Ultra 550B A55B](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b?nim=self-hosted) for the default clarification and deep-research roles

Once your self-hosted endpoint is running, update the corresponding `base_url` in your config to point at it. Deep Researcher Agent's configuration validator currently requires `NVIDIA_API_KEY` for every `_type: nim` profile, even when a local NIM does not enforce client authentication. Set a non-secret placeholder for the local deployment before starting Deep Researcher Agent:

```bash
export NVIDIA_API_KEY=local-nim
```

Then reference that variable in the local profile:

```yaml
llms:
  local_ultra_llm:
    _type: nim
    # Use the identifier returned by the local NIM's /v1/models endpoint.
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "https://<your-ultra-endpoint>/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    top_p: 0.7
    max_tokens: 16384
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: false
```

### Nemotron 3.5 Lightning on NVIDIA API Catalog

The default profiles retain Nemotron 3.5 Lightning for intent classification and shallow research. When the shallow
role uses Lightning through the NVIDIA API Catalog endpoint (`integrate.api.nvidia.com`), the hosted serving profile
can intermittently return citation-incomplete or malformed final drafts. Deep Researcher Agent verifies the draft against the captured
source registry and fails closed instead of publishing an unsupported answer, so an affected request ends with a
failed workflow outcome even when its search completed successfully.

This behavior depends on the serving profile, not only the model weights. It did not reproduce at the same rate in
validation with the tested self-hosted NVFP4 vLLM profile. For a deployment that prioritizes shallow-answer
reliability, use one of these configurations:

- Assign Nemotron Ultra to `shallow_research_agent.llm`, while keeping Lightning for intent classification.
- Serve Lightning through a self-hosted profile that you validate end to end with Deep Researcher Agent's citation and tool-calling
  workflow.

The Brev getting-started launchable uses the first option. This keeps the launchable reliable without changing the
model assignment in the general-purpose shipped profiles.

## Knowledge Layer Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `Unknown backend` | Adapter module not imported | Ensure backend package is installed: `uv pip install -e "sources/knowledge_layer[llamaindex]"` |
| Empty retrieval results | Ingestion and retrieval resolved different collections | Verify the upload-path collection and active `conversation-id`; without session context, verify the configured `collection_name` fallback |
| Foundational RAG connection refused | RAG Blueprint not running | Start the RAG Blueprint server; verify `rag_url` and `ingest_url` |
| `milvus-lite` required | Missing dependency | `uv pip install "pymilvus[milvus_lite]"` |

## Docker / Deployment Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Container fails to start | Missing environment variables | Check `deploy/.env` has all required keys |
| Port already in use | Another service on port 3000/8000 | Set `PORT=8100` or `FRONTEND_PORT=3100` in `.env` |
| UI shows "Backend unavailable" | Backend not healthy | `curl http://localhost:8000/health`; check backend container logs |

## VM / Remote Development

If you are running the Deep Researcher Agent blueprint on a remote VM (cloud instance, WSL, SSH server) and accessing it from your local browser, `localhost:3000` and `localhost:8000` will not resolve because the services are listening on the VM — not your local machine.

### SSH Port Forwarding

Forward the required ports through your SSH connection:

```bash
# Forward both the frontend and backend ports
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@your-vm-host
```

Then open [http://localhost:3000](http://localhost:3000) on your local machine as usual.

To forward ports to an already-active SSH session, you can also use `~C` (SSH escape sequence) to open the SSH command line and type the following on a single line (press Enter at the end):

```text
-L 3000:localhost:3000 -L 8000:localhost:8000
```

### VS Code Remote SSH

If you are using VS Code Remote-SSH, ports are typically forwarded automatically when the server starts listening. If not, open the **Ports** panel (`Ctrl+Shift+P` → "Ports: Focus on Ports View") and add ports `3000` and `8000` manually.

### Common Symptoms

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| "This site can't be reached" on `localhost:3000` or `localhost:8000` | Ports not forwarded from VM to local machine | Use SSH port forwarding (see above) |
| Connection refused after forwarding | Service not running on the VM | SSH into the VM and verify with `curl http://localhost:8000/health` |
| Port forwarding conflicts | Local port already in use | Use alternate local ports: `ssh -L 3001:localhost:3000 -L 8001:localhost:8000 user@vm` |

```{note}
Docker Compose deployments on the VM handle container-to-host port mapping automatically. The SSH forwarding described here is for making the VM's ports accessible on your local machine.
```

## Debugging Tips

### Inspect Relay Logging

```yaml
# In your config YAML
workflow:
  _type: chat_deepresearcher_agent
  relay:
    logging: true
```

### Phoenix Tracing Through Relay

For full setup and trace-reading instructions, see [Observability with NeMo Relay](../deployment/observability.md).

Start a Phoenix server and enable tracing in config:

```yaml
workflow:
  relay:
    observability:
      opentelemetry:
        enabled: true
        endpoints:
          - type: openinference
            endpoint: ${RELAY_OTEL_ENDPOINT:-http://localhost:6006/v1/traces}
            resource_attributes:
              openinference.project.name: deep-researcher-relay
```

Then open [http://localhost:6006](http://localhost:6006) to inspect traces, token usage, and latency.
Set `RELAY_OTEL_ENDPOINT` to use a remote Phoenix or collector endpoint; local Phoenix is the default.
If the trace is missing, also inspect the project configured in
`~/.config/nemo-relay/plugins.toml`; Relay can discover an existing user-level
Phoenix destination.

### Check Registered Components

```bash
# List registered NeMo Agent Toolkit plugins
.venv/bin/nat info components
```
