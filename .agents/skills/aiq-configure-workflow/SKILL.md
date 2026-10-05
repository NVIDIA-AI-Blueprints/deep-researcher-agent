---
name: aiq-configure-workflow
description: Use when composing, adapting, or validating an AI-Q workflow YAML under configs/ — selecting a shipped profile, enabling tools and data_source_registry sources, wiring chat or direct data-science workflows, configuring NeMo Relay and general.telemetry observability plus general.front_end aiq_api settings, and pre-flighting cross-references before deploy or serve. Hand off deploy to aiq-deploy, live research to aiq-research, prompt/model edits to aiq-customize-prompts-models, and new source code to aiq-add-tool or aiq-add-data-source.
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/aiq"
  tags: "aiq nemo-agent-toolkit config yaml workflow data-source-registry telemetry aiq_api"
allowed-tools: Read Bash Edit
---

# Configure AI-Q Workflows

Use this skill when a developer or operator needs a **new `configs/config_*.yml`**
file.

## Start Here

- Confirm this is **config composition** — not deploy (`aiq-deploy`), live research
  (`aiq-research`), prompt edits (`aiq-customize-prompts-models`), or new NAT
  packages (`aiq-add-tool` / `aiq-add-data-source`).
- Copy the closest shipped `configs/*.yml` profile; merge feature blocks from others.
- **Every produced config must pass `validate_config.py` before hand-off.**

## Authoritative References

- `docs/source/customization/configuration-reference.md` — all fields and defaults
- `docs/source/customization/tools-and-sources.md`
- `docs/source/deployment/observability.md` — tracing setup detail
- `frontends/aiq_api/README.md`
- `configs/config_web_default_llamaindex.yml` / `configs/config_cli_default.yml`

Bundle:

- [references/config-profiles.md](references/config-profiles.md) — pick a starting profile.
- [references/composing-config.md](references/composing-config.md) — holistic config map
  (`general`, `llms`, `functions`, `workflow`, telemetry, `aiq_api`) and how to tune them.
- [references/env-vars.md](references/env-vars.md) — environment variables by config feature.
- [references/config-schema.md](references/config-schema.md) — validator checks only.
- [assets/config-scaffold.yml](assets/config-scaffold.yml) — fallback scaffold.

## Workflow

1. **Scaffold** — `cp configs/<profile>.yml configs/config_<name>.yml` (or
   [assets/config-scaffold.yml](assets/config-scaffold.yml) + merge blocks).
2. **Compose** — [references/composing-config.md](references/composing-config.md):
   adjust registry, tools, agents, LLMs, telemetry, `aiq_api`, workflow flags.
   Use `config_web_default_llamaindex.yml` as the live default for web `general:`
   blocks; `configuration-reference.md` for every option. Use
   [references/env-vars.md](references/env-vars.md) for feature-specific env vars.
3. **Validate (required)** —

```bash
uv run python .agents/skills/aiq-configure-workflow/scripts/validate_config.py configs/config_<name>.yml
```

Fix every `ERROR:`; re-run until exit code 0. Then hand off to `aiq-deploy` or:

```bash
dotenv -f deploy/.env run nat serve --config_file configs/config_<name>.yml --port 8000
```

## Validation

```bash
uv run python .agents/skills/aiq-configure-workflow/scripts/validate_config.py <config.yml>
```

See [references/config-schema.md](references/config-schema.md). Expected: exit 0.

## Common Mistakes

- Skipping `validate_config.py` on a new config.
- Undefined `llms:` alias or registry tool not declared under `functions:`.
- Missing functions required by the selected workflow: the chat workflow needs
  `intent_classifier`, `shallow_research_agent`, and `deep_research_agent`; the
  direct DS workflow needs `data_science_agent`.
- `use_async_deep_research: true` without `general.front_end` (`aiq_api`).
- Inventing feature YAML — copy from a shipped profile.

## Related Skills

- `aiq-deploy`
- `aiq-research`
- `aiq-customize-prompts-models`
- `aiq-add-tool`
- `aiq-add-data-source`
- `aiq-release-qa`
- `aiq-prepare-pr`
