# What `validate_config.py` checks

Run on every new or edited `configs/config_*.yml` before deploy/serve.

```bash
uv run python .agents/skills/aiq-configure-workflow/scripts/validate_config.py <config.yml>
```

## Errors (exit 1)

| Check | Meaning |
|-------|---------|
| LLM aliases | Every `llm`, `orchestrator_llm`, `planner_llm`, `researcher_llm`, `writer_llm`, `source_router_llm`, `summary_llm`, `intent_llm`, `summary_model` value must exist under `llms:` |
| Registry tools | Each tool in a `data_source_registry` source's `tools:` must be a key under `functions:` |
| Workflow | `workflow:` must exist and `_type` must be `chat_deepresearcher_agent` or `data_science_workflow` |
| Required agents | The chat workflow requires `intent_classifier`, `shallow_research_agent`, and `deep_research_agent`; the direct DS workflow requires `data_science_agent` |
| Clarifier | When `workflow.enable_clarifier` is true, `clarifier_agent` must exist under `functions:` |
| `front_end` type | When `general.front_end` is set, `_type` must be `aiq_api` |
| `aiq_api` settings | `expiry_seconds`, `db_url`, and `cors` shape are checked |
| Telemetry | Console logging and tracing exporter `_type` values are checked |
| Registry shape | `data_source_registry.sources` entries need `id`, `name`, and declared `tools` |

## Warnings (exit 0)

- No `llms:` block
- No `data_source_registry`
- `requires_auth: true` on a source (confirm MCP/OAuth wiring)
- `use_async_deep_research: true` without `general.front_end`

## Env checklist

Lists every `${VAR}` in the file and whether it is set in the current shell (not
values). Operator should align with `skills/aiq-deploy/references/env-and-secrets.md`.

## Top-level shape

```yaml
general:      # telemetry; front_end (web)
llms:
functions:    # registry, tools, agents
workflow:     # chat_deepresearcher_agent
```

Field reference: `docs/source/customization/configuration-reference.md`.
