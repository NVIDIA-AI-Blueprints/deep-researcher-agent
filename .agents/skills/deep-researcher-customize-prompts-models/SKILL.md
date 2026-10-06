---
name: deep-researcher-customize-prompts-models
description: Use when customizing Deep Researcher Agent behavior through Jinja2 prompt templates or per-agent model selection — editing prompts under src/deep_researcher_agent/agents/*/prompts/, adding template variables, or assigning/swapping LLMs per agent role via config (the llms section plus per-agent fields like orchestrator_llm, planner_llm, researcher_llm, writer_llm, source_router_llm).
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags: "deep-researcher nemo-agent-toolkit prompts models jinja2 customization"
allowed-tools: Read Bash Edit
---

# Customize Deep Researcher Agent Prompts and Models

Use this skill when a developer wants to change *how* an Deep Researcher Agent reasons or
*which* model it uses — by editing a Jinja2 prompt template or by assigning a
different LLM to an agent role — usually without touching agent code. Deep Researcher Agent
behavior is driven by prompts and config, so most tuning is a template or YAML
change. The one exception is adding a brand-new template, which needs a one-line
`load_prompt` wiring in the agent (see the prompt-templates reference).

## Start Here

- Confirm the change is prompt or model customization, not new tool/agent logic.
  For a new retrieval source use `deep-researcher-add-data-source`; for a new tool use
  `deep-researcher-add-tool`.
- Read the authoritative docs and the existing templates/config below first.
- Prefer editing an existing template or config field over adding new machinery.
- Keep the prompt's STRICT citation rules intact, and never hard-code a model
  name where an `llms:` ref belongs.
- Keep templates general-purpose: don't hard-code specific queries, domains, or
  source/tool names — those come from the user's request and the
  `data_source_registry` at runtime.

## Authoritative References

- `docs/source/customization/prompts.md`: prompt guide — template inventory,
  `load_prompt(path, name)`, `render_prompt_template(template, ...)`, the
  documented template variables, the STRICT citation rules, and "Creating a New
  Template". Note it does not document every template's variables (e.g.
  `source_router.j2`, `writer.j2`, `source_registry.j2`) — the `.j2` files are
  authoritative for the variables they actually use.
- `docs/source/customization/swapping-models.md`: choosing hosted vs. self-hosted
  NIMs and pointing config at them.
- `docs/source/customization/configuration-reference.md`: the `llms` section and
  each agent's config fields (`deep_research_agent`, `clarifier_agent`, …).
- `src/deep_researcher_agent/common/prompt_utils.py`: `load_prompt` and
  `render_prompt_template`.
- `src/deep_researcher_agent/common/llm_provider.py`: `LLMRole` and `LLMProvider.configure`,
  which bind a resolved LLM to an agent role (used by the deep research agent).
- Templates to model on: `src/deep_researcher_agent/agents/deep_researcher/prompts/*.j2`
  (orchestrator, planner, researcher, source_router, writer) and
  `src/deep_researcher_agent/agents/clarifier/prompts/*.j2`. Other agents have prompts too
  (e.g. `shallow_researcher`, `chat_researcher`) — check
  `src/deep_researcher_agent/agents/*/prompts/`.

Longer procedures live in this bundle:

- [references/prompt-templates.md](references/prompt-templates.md): where templates
  live, how they load and render, template variables, citation rules, and how to
  edit or add one safely.
- [references/model-selection.md](references/model-selection.md): the `llms`
  section, per-agent LLM fields, role binding via `LLMProvider`, and swapping models.

## Workflow

1. Identify the target agent and whether the change is a prompt or a model.
2. For a prompt: edit the relevant `src/deep_researcher_agent/agents/<agent>/prompts/*.j2`
   template; keep its variables and citation rules intact (see the references).
3. For a model: add or point an `llms:` entry in the config and set the agent's
   role field (e.g. `orchestrator_llm`, `planner_llm`, `researcher_llm`,
   `writer_llm`, `source_router_llm`) to that ref — do not edit Python to swap a
   model.
4. Keep token cost in mind: prefer reordering static instructions before dynamic
   content (KV-cache reuse) and a cheaper model for low-stakes roles.
5. Validate (below): lint any changed Python, run the agent's tests, and
   smoke-run the CLI against the config you changed.
6. Summarize changed files and paste the validation evidence.

## Validation

Run the narrowest checks first; broaden only if you touched shared code.

```bash
uv run ruff check src/deep_researcher_agent                      # only if you changed Python
uv run pytest tests/deep_researcher_agent/agents/<agent>         # the agent's tests (a prompt-only edit may have none)
./scripts/start_cli.sh --config_file <your config>   # smoke against the config you edited
```

Expected: the agent loads its templates without a Jinja2 error and runs with the
configured model. A bare `./scripts/start_cli.sh` uses the fixed default
(`configs/config_cli_default.yml`), so pass `--config_file` to exercise your
change. For a prompt-only edit (which often has no dedicated unit test), the
smoke run is the real check; a config/prompt-only change needs no Python lint.

## Common Mistakes

- Breaking a template variable or the STRICT citation rules in
  `docs/source/customization/prompts.md`, which degrades report grounding.
- Hard-coding specific queries, domains, or source/tool names into a template,
  which biases the agent toward one task and breaks generalization. Source/domain
  selection is data-driven (`data_source_registry`, `source_router.j2`); keep
  prompts task-agnostic.
- Hard-coding a model name in Python instead of using an `llms:` ref and the
  agent's role field, so the model can no longer be swapped from config.
- Changing an agent's default model when you meant a single sub-role. The deep
  research agent's default is `orchestrator_llm` (there is no generic `llm`
  field); the clarifier's default is `llm`. Editing the default shifts every
  unset role.
- Introducing a large dynamic prefix that defeats KV-cache reuse and raises cost.
- Pointing an agent's role at a model whose entry is not defined in `llms:`.

## Related Skills

- `deep-researcher-configure-workflow`
- `deep-researcher-add-tool`
- `deep-researcher-add-data-source`
- `deep-researcher-release-qa`
- `deep-researcher-prepare-pr`
