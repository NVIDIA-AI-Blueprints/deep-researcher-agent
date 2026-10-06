<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompt templates

Authoritative sources: the `.j2` files themselves (for the variables a template
actually uses) and `docs/source/customization/prompts.md` (for the documented
workflow and many template variables). Note `prompts.md` does not cover every
template — e.g. `source_router.j2`, `writer.j2`, and `source_registry.j2` have no
variable section there yet — so read the `.j2` file when in doubt.

## Where templates live

Each agent owns its Jinja2 templates under
`src/deep_researcher_agent/agents/<agent>/prompts/*.j2`. For example, the deep researcher has
`orchestrator.j2`, `planner.j2`, `researcher.j2`, `source_router.j2`, `writer.j2`,
and `source_registry.j2`; the clarifier has `research_clarification.j2`;
`shallow_researcher` and `chat_researcher` have their own as well.

## How templates load and render

- `load_prompt(path, name)` in `src/deep_researcher_agent/common/prompt_utils.py` reads a
  template file as a string.
- `render_prompt_template(template, **kwargs)` (same module) renders it with
  Jinja2, injecting the variables. `prompts.md` has a "Template Variables" section
  for many agents; for a template it doesn't list, read the `.j2` to see which
  variables it references.

## Editing a template safely

1. Keep every `{{ variable }}` the agent passes in; removing one breaks rendering
   or silently drops context.
2. Preserve the **Citation Rules (STRICT)** section in `prompts.md` — report
   grounding depends on the model emitting citations exactly as instructed.
3. Prefer putting static instructions before dynamic content so the KV cache is
   reused across calls (lower latency and token cost).
4. Keep the template task-agnostic — don't hard-code specific queries, domains, or
   source/tool names; source/domain routing is data-driven (`data_source_registry`,
   `source_router.j2`), and hard-coding it bypasses that routing.
5. Editing an existing `.j2` needs **no** code change.

## Adding a new template

This is the one prompt change that touches Python. Follow "Creating a New
Template" in `prompts.md`:

1. Add the `.j2` under the agent's `prompts/` directory.
2. Write the template (keep variable names consistent with what the agent passes).
3. **Wire it in the agent's Python** — load it with `load_prompt` and render it
   with `render_prompt_template` (this is `prompts.md` Step 3). Without this step
   the new template is never used.

## Validation

```bash
./scripts/start_cli.sh --config_file <a config using that agent>   # template loads (no Jinja2 error)
uv run pytest tests/deep_researcher_agent/agents/<agent>                        # if the agent has tests
```

Expected: the agent starts and renders the template without error. A bare
`start_cli.sh` runs the fixed default config, so pass the config that exercises
the agent whose template you changed.
