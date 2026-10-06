<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GSF as a data source

This package connects Deep Researcher Agent to NVIDIA Generative Semantic Fabric (GSF).

## Tools

- `gsf__catalog_search`: finds relevant semantic objects.
- `gsf__text_to_sql`: answers analytical questions with SQL and bounded rows.
- `gsf__text_to_pql`: runs predictive questions through the GSF/Kumo path.

## Configuration

Set `GSF_BASE_URL` to GSF's auth-aware API origin and add the function group to the data-source registry:

```yaml
function_groups:
  gsf:
    _type: gsf
    base_url: ${GSF_BASE_URL}
    include:
      - catalog_search
      - text_to_sql
      - text_to_pql

functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: gsf
        name: "Enterprise Structured Data"
        description: >-
          Build authorized semantic context and execute bounded structured-data
          queries through GSF.
        default_enabled: true
        requires_auth: true
        tools:
          - gsf
```

When `auth` is omitted, each tool invocation obtains the current Deep Researcher Agent user token and forwards it to GSF without
storing it on the shared client.

For local development or automated evaluation without an incoming Deep Researcher Agent user token, explicitly configure password
authentication using environment variables:

```yaml
function_groups:
  gsf:
    _type: gsf
    base_url: ${GSF_BASE_URL}
    auth:
      mode: password
      email: ${GSF_EMAIL}
      password: GSF_PASSWORD
    include:
      - catalog_search
      - text_to_sql
```

When `auth` is omitted, the existing request-scoped Deep Researcher Agent user-token flow is
used. Password mode carries only the `password` variable name through NAT
configuration and distributed-worker serialization. The worker reads that
variable directly from its process environment immediately before creating the
GSF client. It creates one GSF session, reuses its cookie for local development
or evaluation calls, and signs out when the group closes. Every worker must
receive the named environment variable. The client does not fall back between
authentication methods.

## API mapping

- Catalog search calls `POST /api/question-entity-coverage`.
- Text-to-SQL calls `POST /api/chat/completions` with `prediction: false`.
- Text-to-PQL calls `POST /api/chat/completions` with `prediction: true`.
- Normal Deep Researcher Agent calls rely on GSF's routing and omit database selection.
- Automated benchmarks may explicitly set optional `database_name`; Deep Researcher Agent forwards it to GSF as `target_db`.
