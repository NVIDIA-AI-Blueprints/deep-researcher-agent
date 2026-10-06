---
name: deep-researcher-add-tool
description: Use when adding or changing a general-purpose Deep Researcher Agent tool (a NeMo Agent Toolkit function) under sources/, defining its FunctionBaseConfig schema, registering it with @register_function, wiring it into an agent's tools list, or testing it.
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags: "deep-researcher nemo-agent-toolkit tool function sources"
allowed-tools: Read Bash Edit
---

# Add an Deep Researcher Agent Tool

Use this skill when a developer wants to add a general-purpose tool to Deep Researcher Agent — a
NeMo Agent Toolkit (NAT) function such as a web search, calculator, or code
helper. The tool is a package under `sources/`, registered with
`@register_function` and referenced directly in an agent's `tools` list.

## Start Here

- Confirm this is a **general utility tool**. If it is domain-specific retrieval
  that should appear as a toggleable source in the UI, use `deep-researcher-add-data-source`
  instead — a data source is the same NAT function plus a `data_source_registry`
  entry. This skill stops at wiring the tool into an agent.
- Read the authoritative files below before editing.
- Copy the closest existing tool package rather than inventing a new shape.
- Never print or commit API keys; resolve secrets at runtime via `SecretStr`.

## Authoritative References

- `docs/source/extending/adding-a-tool.md`: canonical 8-step walkthrough; the
  workflow below mirrors it.
- `sources/tavily_web_search/`: minimal tool package.
- `sources/google_scholar_paper_search/`: tool package with a separate client,
  a graceful missing-secret stub, and tests.
- `docs/source/extending/adding-a-data-source.md`: "Data Source vs. Tool" — a
  data source is architecturally identical to a tool; only the registry wiring
  differs.

Existing tools to model on: `tavily_web_search`, `exa_web_search`,
`paper_search` (Google Scholar), `knowledge_retrieval`.

Longer procedures live in this bundle:

- [references/nat-function-pattern.md](references/nat-function-pattern.md):
  package layout, config class, registration, the missing-secret stub, the
  `pyproject.toml` entry point, and wiring the tool into an agent in YAML.
- [references/testing.md](references/testing.md): unit-test pattern and the
  install/test/lint validation commands.

## Workflow

1. Pick the closest existing package under `sources/` and inspect its layout.
2. Create `sources/<my_tool>/` with `src/register.py`, a client module,
   `pyproject.toml`, and `tests/`.
3. Define a `FunctionBaseConfig` subclass with a stable `name=` (this becomes the
   YAML `_type`); resolve any API key via `SecretStr`.
4. Register an async `@register_function` that yields a `FunctionInfo`; yield a
   graceful stub when a required secret is missing.
5. Add the `[project.entry-points."nat.plugins"]` entry and install the package
   editable.
6. Reference the tool in a config under `configs/` (under `functions:`, then in
   an agent's `tools:` list).
7. Add focused tests; run the validation commands below.
8. Summarize changed files and paste the test/lint evidence.

## Validation

Run the narrowest commands first; broaden only if the change touches shared code.

```bash
uv pip install -e ./sources/my_tool
uv run pytest sources/my_tool/tests
uv run ruff check sources/my_tool
uv run ruff format --check sources/my_tool
```

Expected: the package installs, its tests pass, and Ruff reports no lint or
format failures for the new tool package.

## Common Mistakes

- Omitting the `[project.entry-points."nat.plugins"]` entry in `pyproject.toml`,
  so NAT never discovers the registration at import time.
- Crashing on a missing API key instead of yielding a stub that returns a clear
  error string.
- Raising exceptions from the tool function; tools must return error messages as
  strings so they never crash the agent.
- Weak docstrings: the LLM uses the function docstring as the tool description to
  decide when to call it — state what it does, when to use it, and what it returns.
- Printing API keys or embedding secrets in YAML instead of using environment
  variables or `SecretStr`.

## Related Skills

- `deep-researcher-configure-workflow`
- `deep-researcher-add-data-source`
- `deep-researcher-release-qa`
- `deep-researcher-prepare-pr`
