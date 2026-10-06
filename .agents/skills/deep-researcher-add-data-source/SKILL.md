---
name: deep-researcher-add-data-source
description: Use when adding or changing an Deep Researcher Agent data source under sources/, registering it as a NeMo Agent Toolkit function, wiring it into the data_source_registry for UI toggles, or validating retrieval behavior with tests.
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags: "deep-researcher nemo-agent-toolkit data-source sources registry"
allowed-tools: Read Bash Edit
---

# Add an Deep Researcher Agent Data Source

Use this skill when a developer wants to add a retrieval or search source to
Deep Researcher Agent and expose it as a toggleable source in the UI. A data source is a NeMo
Agent Toolkit (NAT) function package under `sources/`, registered in the
`data_source_registry`.

## Start Here

- Confirm this is a new retrieval/search source (not a UI, auth, or prompt
  change). For a general utility function, use `deep-researcher-add-tool` instead.
- Read the authoritative files below before editing.
- Copy the closest existing source package rather than inventing a new shape.
- Never print or commit API keys; resolve secrets at runtime via `SecretStr`.

## Authoritative References

- `docs/source/extending/adding-a-data-source.md`: canonical package and
  registration walkthrough (the steps below mirror it).
- `sources/google_scholar_paper_search/`: complete example package with a
  client, a config + registration, a graceful missing-secret stub, and tests.
- `sources/tavily_web_search/`: minimal source package for comparison.
- `src/deep_researcher_agent/common/data_source_registry.py`: the `data_source_registry`
  config (`name="data_source_registry"`) that drives `GET /v1/data_sources`.
- `docs/source/customization/tools-and-sources.md`: how the registry maps to UI
  toggles and per-request filtering.
- `frontends/ui/src/features/layout/data-sources.ts`: the UI `DataSource` type;
  sources are fetched dynamically, so usually no UI code change is needed.

Longer procedures live in this bundle:

- [references/package-layout.md](references/package-layout.md): package files,
  `pyproject.toml`, config class, `@register_function`, and the missing-secret stub.
- [references/registry-and-ui.md](references/registry-and-ui.md): registering in
  YAML and how the registry surfaces toggles and filtering.
- [references/validation.md](references/validation.md): install, test, and lint
  commands with expected results.

## Workflow

1. Pick the closest existing package under `sources/` and inspect its layout.
2. Create `sources/<my_data_source>/` with `src/register.py`, the client
   module, `pyproject.toml`, and `tests/` (see package-layout reference).
3. Define a `FunctionBaseConfig` subclass with a stable `name=` and resolve any
   API key via `SecretStr`; register it with `@register_function`.
4. Yield a graceful stub when the required secret is missing.
5. Install the package editable and add it to the `data_source_registry` in the
   relevant config under `configs/`.
6. Add focused tests; run the validation commands below.
7. Summarize changed files and paste the test/lint evidence.

## Validation

Run the narrowest commands first; broaden only if the change touches shared code.

```bash
uv pip install -e ./sources/my_data_source
uv run pytest sources/my_data_source/tests
uv run ruff check sources/my_data_source
uv run ruff format --check sources/my_data_source
```

Expected: the package installs, its tests pass, and Ruff reports no lint or
format failures for the new source package.

## Common Mistakes

- Forgetting to add the source to the `data_source_registry`, so the UI cannot
  toggle it and agents do not inherit the tool.
- Omitting the `[project.entry-points."nat.plugins"]` entry in `pyproject.toml`,
  so NAT never discovers the registration.
- Crashing on a missing API key instead of yielding a stub that returns a clear
  error message.
- Returning unstructured or citation-poor output, which weakens report grounding.
- Printing API keys or embedding secrets in YAML instead of using environment
  variables or `SecretStr`.

## Related Skills

- `deep-researcher-configure-workflow`
- `deep-researcher-add-tool`
- `deep-researcher-release-qa`
- `deep-researcher-prepare-pr`
