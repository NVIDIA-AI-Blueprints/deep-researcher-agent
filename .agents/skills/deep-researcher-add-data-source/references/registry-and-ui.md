<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registry and UI wiring

Authoritative sources: `src/deep_researcher_agent/common/data_source_registry.py` and
`docs/source/customization/tools-and-sources.md`.

## Register the source in YAML

After installing the package editable, add the source to the
`data_source_registry` in the relevant config under `configs/`. This is normally
the **only** config change needed — agents inherit registry tools automatically:

```yaml
functions:
  data_sources:                  # function instance name; _type is the registry
    _type: data_source_registry
    sources:
      - id: my_data_source
        name: "My Data Source"
        description: "What this source retrieves."
        default_enabled: true
        tools:
          - my_data_source_tool  # function instance name(s), as plain strings

  my_data_source_tool:
    _type: my_data_source        # _type matches the config class name= / entry-point key
```

Confirm the per-source field names against `DataSourceEntry` (and the parent
`DataSourceRegistryConfig`) in `src/deep_researcher_agent/common/data_source_registry.py`;
do not guess the schema. A source entry takes `id`, `name`, `description`,
`default_enabled`, `requires_auth`, and a `tools` list of plain function-instance
names — there is no `category` field on the config model.

## How the registry surfaces the source

- `GET /v1/data_sources` returns the registered sources; the UI renders them as
  toggles. No UI code change is normally required.
- Per-request filtering: the WebSocket chat payload and
  `POST /v1/jobs/async/submit` accept a `data_sources` list to scope which
  sources are active for that request.
- Auto-inheritance: every agent gets every registered tool by default; use an
  agent's `exclude_tools` for per-agent specialization.
- Tools not listed in any registry source entry are always included (e.g. utility
  tools like "think"). An explicit empty list (`data_sources: []`) disables
  registry tools while leaving unmapped utility tools available.

## UI type (reference only)

`frontends/ui/src/features/layout/data-sources.ts` defines the `DataSource`
TypeScript interface (`id`, `name`, `description`, `category`, `defaultEnabled`,
`requiresAuth`). Because sources are fetched dynamically from
`GET /v1/data_sources`, adding a source does not require editing this file. Touch
the UI only for a genuinely new presentation need — that is a separate frontend
change under `frontends/ui/`, outside this skill's scope.

## Authenticated sources

If the source requires per-user auth, set the auth flags rather than hard-coding
credentials, and follow the auth integration path. Protected-source auth is out
of scope for this skill; coordinate with the auth integration workflow.
