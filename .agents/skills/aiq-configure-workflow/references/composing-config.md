# Composing a workflow config

Holistic map for `configs/config_*.yml`. **Canonical field tables and defaults:**
`docs/source/customization/configuration-reference.md`. **Worked examples:**
`configs/config_web_default_llamaindex.yml` (web) and `configs/config_cli_default.yml`
(CLI).

## How to edit

1. Copy the nearest shipped profile ([config-profiles.md](config-profiles.md)).
2. Merge feature blocks from other `configs/*.yml` files (knowledge, guardrails,
   sandbox, MCP auth) — do not invent fields.
3. Tune sections below; cross-check names against the canonical reference.
4. Run `validate_config.py` ([config-schema.md](config-schema.md)).

Sibling skills for depth (do not duplicate here):

- Registry / UI → `.agents/skills/aiq-add-data-source/references/registry-and-ui.md`
- LLM roles → `.agents/skills/aiq-customize-prompts-models/references/model-selection.md`
- New NAT packages → `aiq-add-tool` / `aiq-add-data-source`
- Secrets → `skills/aiq-deploy/references/env-and-secrets.md`

## Top-level sections

```yaml
general:      # use_uvloop, telemetry, front_end (web only)
llms:         # named aliases → referenced by agents
functions:    # data_source_registry, tools, agents (+ optional feature functions)
workflow:     # chat_deepresearcher_agent orchestrator flags
```

`${VAR}` / `${VAR:-default}` anywhere; secrets in `deploy/.env`.

---

## `general:`

| Area | CLI | Web (`nat serve`) | Where defaults live |
|------|-----|-------------------|---------------------|
| `use_uvloop` | optional | recommended `true` | `config_web_default_llamaindex.yml` |
| `telemetry` | yes | yes | both shipped profiles |
| `front_end` | **omit** | **required** (`aiq_api`) | `config_web_default_llamaindex.yml` |

### Telemetry

**Console** (always available):

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO   # DEBUG | INFO | WARNING | ERROR
```

**Observability** — configure NeMo Relay under `workflow.relay`. Relay logging,
ATOF, and redaction are enabled by default. OTEL is opt-in; uncomment the Relay
OpenInference endpoint in a default config to send traces to Phoenix. Omit
`workflow.relay.pricing` unless the workflow intentionally loads an audited
catalog. See `docs/source/deployment/observability.md`.

`workflow.relay.logging` controls the console subscriber; agent and workflow
configs do not have separate verbose switches.

### `front_end` (`aiq_api`)

Required for UI, REST, WebSocket chat, async jobs, and `aiq-research`. Full
block with comments: `config_web_default_llamaindex.yml` lines 18–47.

| Field | Default (web profile) | Tune when |
|-------|----------------------|-----------|
| `_type` | `aiq_api` | always |
| `runner_class` | `aiq_api.plugin.AIQAPIWorker` | rarely |
| `db_url` | `${NAT_JOB_STORE_DB_URL:-sqlite+aiosqlite:///./jobs.db}` | Postgres prod |
| `expiry_seconds` | `86400` (600–604800) | job retention policy |
| `cors` | localhost regex | production UI origin |

**Env (not YAML):** `NAT_JOB_STORE_DB_URL`, `NAT_DASK_SCHEDULER_ADDRESS`,
`REQUIRE_AUTH`, `AIQ_TRACE_*` — `frontends/aiq_api/README.md`.

`knowledge_retrieval` under `functions:` auto-enables `/v1/collections` and
`/v1/documents`.

---

## `llms:`

Define **aliases** under `llms:`; agents reference them by key (`llm`,
`orchestrator_llm`, `planner_llm`, etc.). Swap models by editing aliases or
repointing role fields — no Python changes.

| Parameter | Typical use | Notes |
|-----------|-------------|-------|
| `_type` | `nim` or `openai` | Provider plugin |
| `model_name` | required | e.g. `nvidia/nemotron-3-ultra-550b-a55b` |
| `base_url` | NIM / compatible endpoint | Set explicitly for hosted NIM |
| `api_key` | optional | Falls back to `NVIDIA_API_KEY` for NIM |
| `temperature`, `top_p`, `max_tokens` | per role | See role table in configuration-reference |
| `num_retries` | resilience | default `5` |
| `chat_template_kwargs` | e.g. `enable_thinking: true` | chain-of-thought models |

**Role-specific starting points** (temperature / max_tokens): intent classifier
(moderate), shallow researcher (low), deep orchestrator/writer (high + thinking),
summary LLM (short output). Full table: `configuration-reference.md` §
`llms` → "Common LLM Configurations".

**Deeper guidance:** `docs/source/customization/swapping-models.md` and
`.agents/skills/aiq-customize-prompts-models/references/model-selection.md`.

---

## `functions:` — tools, registry, agents

Each entry is a named function under `functions:`. The YAML `_type` selects the
NAT plugin; the **key you choose** (e.g. `web_search_tool`) is what registry
`tools:` lists and agents reference.

### Option index: retrieval / search tools

Full parameter tables: `configuration-reference.md` § `functions`. Examples in
shipped configs under `configs/`.

| `_type` | Purpose | Key options to tune | Env / profile |
|---------|---------|---------------------|---------------|
| `tavily_web_search` | Web search | `max_results`, `advanced_search`, `max_content_length`, `api_base_url` | `TAVILY_API_KEY`; all web profiles |
| `exa_web_search` | Web search (Exa) | `max_results`, `search_type` (`auto`/`fast`/`deep`), `full_text`, `highlights` | `EXA_API_KEY` |
| `paper_search` | Academic papers | `provider` (`serper`/`serpapi`/`searchapi`), `max_results` | `SERPER_API_KEY`, etc.; commented in most profiles |
| `knowledge_retrieval` | Document RAG | `backend` (`llamaindex`/`foundational_rag`/`opensearch`), `top_k`, `collection_name`, backend-specific URLs/auth | `config_web_default_llamaindex.yml`, `config_web_frag.yml`, `config_web_opensearch.yml` |

Enable/disable for agents: register in `data_source_registry`, then inherit or
`exclude_tools` (below). Knowledge-layer backends: `docs/source/customization/knowledge-layer.md`.

### `data_source_registry`

Single source of truth for tools → UI toggles → agent inherit. One block with
`_type: data_source_registry` (often `data_sources`):

```yaml
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        tools: [web_search_tool, advanced_web_search_tool]
  web_search_tool:
    _type: tavily_web_search
    max_results: 5
```

- Omit agent `tools:` → inherit all registry tools; use `exclude_tools` to specialize
  (`config_web_default_llamaindex.yml` shallow vs deep).
- Enable a tool: declare under `functions:`, add to a source's `tools:`, set env vars.

Optional **feature** function blocks (guardrails middleware, sandbox, skills,
domain catalog) are not search tools — copy YAML from the profile listed in
[config-profiles.md](config-profiles.md); field docs in `configuration-reference.md`
and feature guides under `docs/source/customization/`.

### Option index: agents

| `_type` | Key options to tune | Doc anchor |
|---------|---------------------|------------|
| `intent_classifier` | `llm`, `tools`, `llm_timeout` | `configuration-reference.md` § `intent_classifier` |
| `clarifier_agent` | `llm`, `max_turns`, `exclude_tools` | § `clarifier_agent` |
| `shallow_research_agent` | `llm`, `max_llm_turns`, `max_tool_iterations`, `exclude_tools` | § `shallow_research_agent` |
| `deep_research_agent` | role LLMs, `exclude_tools`, `enable_source_router`, `domain_catalog_path`, `enable_citation_verification`, `skills`, `sandbox`, concurrency caps | § `deep_research_agent` |

### Agents (required by workflow)

| Function | When | LLM fields |
|----------|------|------------|
| `intent_classifier` | always | `llm` |
| `shallow_research_agent` | always | `llm` |
| `deep_research_agent` | always | `orchestrator_llm`, `planner_llm`, `researcher_llm`, `writer_llm`, `source_router_llm` |
| `clarifier_agent` | `workflow.enable_clarifier: true` | `llm` |
| `data_science_agent` | direct `data_science_workflow` | `llm` |
| `data_science_hybrid_adapter` | `workflow.hybrid_research_agent` routes catalog-supported requests to DS | configured `agent` function reference |

---

## `workflow:`

```yaml
workflow:
  _type: chat_deepresearcher_agent
  hybrid_research_agent: data_science_hybrid_adapter
  enable_escalation: true       # false → shallow only
  enable_clarifier: true
  use_async_deep_research: true   # needs general.front_end
  max_history: 20
  checkpoint_db: ${AIQ_CHECKPOINT_DB:-./checkpoints.db}
```

Full defaults table: `configuration-reference.md` § `workflow`.

For direct DS Agent development, use `_type: data_science_workflow` and define
`data_science_agent`; the chat-only intent, shallow, and deep functions are not
required in that profile.

For product Hybrid routing, configure `intent_classifier` with
`_type: context_aware_intent_router`, define `data_science_hybrid_adapter` with
an `agent` reference to the DS function, and set
`workflow.hybrid_research_agent` to the adapter. The adapter consumes the
router's validated catalog context; direct DS evaluation continues to bypass
both the router and adapter.

---

## Validate

```bash
uv run python .agents/skills/aiq-configure-workflow/scripts/validate_config.py configs/config_<name>.yml
```

What it checks: [config-schema.md](config-schema.md). Fix all errors before deploy.
