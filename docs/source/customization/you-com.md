<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# You.com API Suite

Deep Researcher Agent includes four NeMo Agent Toolkit functions backed by the You.com API. They can be enabled independently or
grouped into one entry in the [data source registry](./tools-and-sources.md#data-source-registry).

| Function type | Purpose |
|---|---|
| `you_web_search` | Return LLM-ready web results with optional live-crawled content, freshness filters, and news results. |
| `you_contents` | Extract Markdown, HTML, or metadata from up to 10 URLs. |
| `you_research` | Run multi-search, cited open-domain research with a configurable effort level. |
| `you_finance_research` | Run cited research over finance-focused sources such as filings, earnings, and market data. |

## Prerequisites

The standard `./scripts/setup.sh` flow installs the `sources/you_com` plugin. For an existing environment, install it
directly:

```bash
uv pip install -e ./sources/you_com
```

Create a key using the [You.com API quickstart](https://you.com/docs/quickstart), then add it to `deploy/.env`:

```text
YDC_API_KEY=<you-com-api-key>
```

Do not commit `deploy/.env`. Each function also accepts an `api_key` field, but the environment variable keeps the
secret out of workflow YAML.

## Configure the Tools

Add only the functions your workflow needs. The following example registers all four under one user-selectable data
source:

```yaml
functions:
  you_web_search:
    _type: you_web_search
    max_results: 10
    safesearch: moderate
    livecrawl_mode: web
    livecrawl_format: markdown
    freshness: off
    include_news_results: false

  you_contents:
    _type: you_contents
    formats: [markdown, metadata]
    crawl_timeout: 30

  you_research:
    _type: you_research
    research_effort: standard

  you_finance_research:
    _type: you_finance_research
    research_effort: deep

  data_sources:
    _type: data_source_registry
    sources:
      - id: you_com
        name: You.com
        description: Web search, content extraction, and cited research.
        tools:
          - you_web_search
          - you_contents
          - you_research
          - you_finance_research
```

Agents with no explicit `tools` list inherit these functions from the registry. Request clients can then select the
source with `data_sources: ["you_com"]`. Refer to [Tools and Sources](./tools-and-sources.md) for filtering and
per-agent specialization.

## Configuration Reference

All four functions accept these shared fields:

| Field | Default | Description |
|---|---|---|
| `api_key` | `null` | Optional inline API key. Prefer `YDC_API_KEY`. |
| `max_retries` | `3` | Maximum attempts for a failed request. |
| `timeout` | `null` | Per-attempt timeout in seconds. `null` disables the timeout. |

`you_web_search` also accepts:

| Field | Default | Allowed values or behavior |
|---|---|---|
| `max_results` | `10` | 1–100 results. |
| `safesearch` | `moderate` | `off`, `moderate`, or `strict`. |
| `livecrawl_mode` | `web` | `off`, `web`, `news`, or `all`. |
| `livecrawl_format` | `markdown` | `off`, `markdown`, or `html`. |
| `freshness` | `off` | `off`, `day`, `week`, `month`, or `year`. |
| `max_content_length` | `50000` | Maximum live-crawled characters per result; `null` is unbounded. |
| `include_news_results` | `false` | Include results classified as news. |

Research and contents functions accept these fields:

| Function | Field | Default | Allowed values or behavior |
|---|---|---|---|
| `you_research` | `research_effort` | `standard` | `lite`, `standard`, `deep`, or `exhaustive`. |
| `you_finance_research` | `research_effort` | `deep` | `deep` or `exhaustive`. |
| `you_contents` | `formats` | `[markdown, metadata]` | Any combination of `markdown`, `html`, and `metadata`. |
| `you_contents` | `crawl_timeout` | `null` | Per-URL crawl timeout from 1 to 60 seconds. |

When `YDC_API_KEY` and `api_key` are both absent, Deep Researcher Agent still starts and registers diagnostic stubs for the configured
functions. Calls return an actionable missing-key error instead of failing workflow initialization.
