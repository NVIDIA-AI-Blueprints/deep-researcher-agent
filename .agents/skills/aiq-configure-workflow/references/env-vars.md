# Environment variables for config authors

Use this as a quick map while composing `configs/config_*.yml`. Keep secret
values in `deploy/.env`; never paste values into config files or PR text.

Canonical references:

- `deploy/.env.example`
- `frontends/aiq_api/README.md`
- `docs/source/deployment/observability.md`
- profile comments in `configs/config_*.yml`

## Core runtime

| Variable | When needed | Notes |
|----------|-------------|-------|
| `NVIDIA_API_KEY` | Default NIM-backed model profiles | Required for model calls unless the config uses a different model provider. |
| `OPENAI_API_KEY` | Frontier/OpenAI model profiles | Required by `config_frontier_models.yml`. |
| `AIQ_DATA_SCIENCE_MODEL` | Direct data-science and FDABench-Lite profiles | Required model name for the configured inference endpoint. |
| `NAT_JOB_STORE_DB_URL` | Web/API mode | Job, event, and artifact metadata database. Defaults to local SQLite. Use PostgreSQL for production. |
| `NAT_DASK_SCHEDULER_ADDRESS` | Distributed execution | Optional. A local Dask cluster is created when unset. |
| `AIQ_CHECKPOINT_DB` | `workflow.checkpoint_db` | Optional. Defaults to local SQLite file `./checkpoints.db`. |

## Retrieval and tools

| Variable | Enables |
|----------|---------|
| `TAVILY_API_KEY` | Tavily web search |
| `EXA_API_KEY` | Exa web search |
| `SERPER_API_KEY` | Serper paper search |
| `SERPAPI_API_KEY` | SerpAPI paper search |
| `SEARCHAPI_API_KEY` | SearchAPI paper search |
| `RAG_SERVER_URL`, `RAG_INGEST_URL` | Foundational RAG profiles |
| `GSF_BASE_URL`, `GSF_EMAIL`, `GSF_PASSWORD` | Direct DS Agent profile with a local GSF password session |
| `GSF_READ_TIMEOUT_SECONDS` | Optional GSF read timeout override for long analytical calls |
| `AIQ_DS_INTERACTION_MODE` | `interactive` by default; set `headless` for non-interactive DS evaluation |

## Web API, auth, and Relay correlation

| Variable | When needed |
|----------|-------------|
| `REQUIRE_AUTH` | Enforce API authentication. Requires validator registration. |
| `AIQ_TRACE_USER_IDENTITY_MODE`, `AIQ_TRACE_USER_IDENTITY_HMAC_SECRET` | User identity tagging for Relay-exported spans. |
| `AIQ_TRACE_CLIENT_ID_MODE`, `AIQ_TRACE_CLIENT_ID_HMAC_SECRET`, `AIQ_TRACE_CLIENT_IP_HEADERS` | Client tagging for Relay-exported spans. |

## Sandbox and artifact storage

| Variable | When needed |
|----------|-------------|
| `AIQ_OPENSHELL_GATEWAY_NAME`, `AIQ_OPENSHELL_IMAGE`, `AIQ_OPENSHELL_POLICY_FILE` | OpenShell sandbox profile. |
| `AIQ_DS_OPENSHELL_IMAGE`, `AIQ_DS_OPENSHELL_POLICY_FILE`, `AIQ_DS_PYTHON_MAX_MEMORY_MB`, `AIQ_DS_PYTHON_MAX_CPU_SECONDS` | OpenShell-only `sandboxed_python` image, offline policy, and optional hard worker-limit overrides. |
| `AIQ_ARTIFACT_BLOB_PROVIDER`, `AIQ_ARTIFACT_S3_BUCKET`, `AIQ_ARTIFACT_S3_ENDPOINT_URL`, `AIQ_ARTIFACT_S3_REGION`, `AIQ_ARTIFACT_S3_PREFIX` | Optional object storage for sandbox artifacts. |

## Validate without leaking

```bash
uv run python .agents/skills/aiq-configure-workflow/scripts/validate_config.py configs/config_<name>.yml
```

The validator prints variable names and whether they are set in the current
shell; it does not print secret values.
