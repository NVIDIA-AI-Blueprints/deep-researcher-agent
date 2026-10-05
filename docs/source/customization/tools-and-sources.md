<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Tools and Sources

AI-Q ships provider integrations for Tavily, Google Scholar search providers, Exa, DuckDuckGo News, Polymarket, and
the [You.com API Suite](./you-com.md). Knowledge retrieval is configured separately through the
[Knowledge Layer](./knowledge-layer.md).

Nimble provides configurable web search with lite and deep modes, plus an Enterprise-only fast mode. Refer to the
[configuration reference](./configuration-reference.md) for its focus, country, and locale controls.

## Data Source Registry

The `data_source_registry` function is the **single source of truth** for which tools exist and which data source they belong to. It controls the UI toggles, per-message filtering, and -- by default -- which tools each agent receives.

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        description: "Search the web for real-time information."
        tools:
          - web_search_tool
          - advanced_web_search_tool
      - id: knowledge_layer
        name: "Knowledge Base"
        description: "Search uploaded documents and files."
        tools:
          - knowledge_search
      - id: enterprise_search
        name: "Enterprise Search"
        description: "Search Confluence, Google Drive, and more."
        requires_auth: true
        tools:
          - eci
```

The `GET /v1/data_sources` API endpoint returns these entries, which the UI renders as toggles. Clients can scope a request to a subset by passing `data_sources: ["web_search"]` in the WebSocket chat payload or in the body of `POST /v1/jobs/async/submit`; only tools belonging to selected sources are active for that request.

Tools not listed in any data source entry (e.g., utility tools like "think") are always included regardless of filtering. Passing an explicit empty list (`data_sources: []`) -- in the WebSocket chat payload or in a `POST /v1/jobs/async/submit` body -- disables data-source tools while leaving those unmapped utility tools available.

The opt-in `sandboxed_python` function is the full scientific-analysis utility.
It runs every call as a self-contained script with a fresh namespace in one
request-owned OpenShell sandbox, preloads pandas, NumPy, SciPy, scikit-learn,
and statsmodels, and exposes exact GSF-result helpers every time. Configure its
function key directly in an agent's `tools` list and do not add it to
`data_source_registry`. It derives calculations from retrieved evidence; it is
not a source and has no configured GSF or SQL connection. There is no
host-process backend: `sandbox` must reference a blocked-network, per-request
`deep_research_sandbox` using OpenShell.

### Source Entry Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | string | *required* | Unique key used in API payloads and filtering (e.g., `web_search`) |
| `name` | string | *required* | Display name shown in the UI |
| `description` | string | `""` | Human-readable description shown in the UI |
| `tools` | list[string] | `[]` | NAT function names or function group names belonging to this source |
| `requires_auth` | bool | `false` | If `true`, the UI greys out this source until the user signs in. Use for sources that need user-level OAuth tokens (e.g., enterprise SSO). Sources that use backend API keys (Tavily, Serper) should leave this `false`. |
| `default_enabled` | bool | `true` | Whether the source is enabled by default when a user first loads the UI |

## Automatic Source Routing

Deep research can run an optional source-router subagent before planning. Request filtering and automatic routing have
different responsibilities:

1. A request's `data_sources` value is the hard boundary for tools mapped in `data_source_registry`. A mapped tool is
   callable only when its source ID is selected. Configured tools that are not mapped to a registry source remain
   callable, including when `data_sources: []` is passed.
2. The router catalog contains only mapped sources that still have an available runtime tool after request filtering.
   Unmapped tools are reported separately for diagnostics and are never source recommendations.
3. The router writes advisory preferred and fallback source guidance. The planner uses that guidance to populate the
   ordered `preferred_tools` and `fallback_tools` fields of each structured `ResearchQuery`.
4. Researcher workers receive the complete request-filtered callable tool set. The preferred and fallback fields guide
   tool order; they are not per-worker tool allowlists.

Set `enable_source_router: false` on `deep_research_agent` to skip the routing step. When enabled, the router uses
`source_router_llm`, or falls back to `orchestrator_llm` when no router-specific LLM is configured. Routing remains
advisory in either case: the planner owns the final query plan.

### Domain Catalog

Set `domain_catalog_path` to a YAML or JSON file with this schema:

```yaml
default_domain_id: general_research

domains:
  - domain_id: general_research
    domain_name: General Research
    description: Broad factual and mixed-domain research.
    preferred_source_ids:
      - knowledge_layer
      - web_search
    fallback_source_ids:
      - web_search
    is_default: true
```

| Field | Type | Default | Behavior |
|---|---|---|---|
| `default_domain_id` | string or null | `null` | Root-level fallback domain. If omitted, the first entry with `is_default: true` is used, then the first domain entry. |
| `domains` | list | `[]` | Domain routes loaded from an explicitly configured catalog. An explicitly configured empty catalog exposes no domain entries, so the router may use `unconfigured` with runtime fallback sources. When `domain_catalog_path` is omitted, the runtime synthesizes `general_research` instead. |
| `domain_id` | string | *required* | Stable route identifier returned in the source-routing plan. |
| `domain_name` | string | *required* | Human-readable route name. |
| `description` | string | `""` | Guidance for deciding whether the request belongs to this domain. |
| `preferred_source_ids` | list[string] | `[]` | Ordered primary sources for the domain. IDs unavailable in the active runtime are removed before the router sees the route. |
| `fallback_source_ids` | list[string] | `[]` | Ordered alternatives. Unavailable IDs are removed before routing. |
| `is_default` | bool | `false` | Marks a fallback domain when the root `default_domain_id` is not set. |

For each domain, the runtime also computes `unavailable_source_ids` from configured preferred and fallback IDs that do
not exist in the active mapped source set. The router cannot recommend those sources; it uses an available domain
fallback instead. If `domain_catalog_path` is omitted, AI-Q creates a `general_research` route whose preferred sources
are all active mapped sources. Its fallback is `web_search` when available, otherwise the first active mapped source.
An explicitly configured empty catalog remains empty: the router may return `unconfigured` and use that same runtime
fallback order. A nonempty configured catalog also uses the runtime fallback when no domain fits.

Refer to the runnable
[`config_domain_routing_and_skills.yml`](../../../configs/config_domain_routing_and_skills.yml) example and its
[`deep_research_domain_catalog.yml`](../../../configs/domain_catalogs/deep_research_domain_catalog.yml) catalog.

## Auto-Inherit: Agents Get All Registry Tools by Default

When an agent's `tools` list is **empty** (the default), it automatically inherits every tool registered in `data_source_registry`. This means adding a new tool or data source requires only **one config change** -- adding it to the registry.

```yaml
functions:
  # Add a tool to the registry -- all agents get it automatically
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        tools:
          - web_search_tool
          - advanced_web_search_tool
      - id: knowledge_layer
        name: "Knowledge Base"
        tools:
          - knowledge_search

  # Agents with no tools list inherit all registry tools
  intent_classifier:
    _type: intent_classifier
    llm: nemotron_lightning_intent_llm

  clarifier_agent:
    _type: clarifier_agent
    llm: nemotron_llm

  # Use exclude_tools for per-agent specialization
  shallow_research_agent:
    _type: shallow_research_agent
    llm: nemotron_llm
    exclude_tools:
      - advanced_web_search_tool    # shallow uses regular web search

  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_llm_deep
    exclude_tools:
      - web_search_tool             # deep uses advanced web search
```

### Per-Agent Specialization with `exclude_tools`

Use `exclude_tools` to remove specific tools from the inherited set. This is useful when different agents need different variants of a tool (e.g., shallow research uses `web_search_tool` while deep research uses `advanced_web_search_tool`).

### Explicit Override (Backward Compatible)

If an agent specifies an explicit `tools` list, it uses exactly those tools and ignores the registry. This preserves backward compatibility with existing configs:

```yaml
  # Explicit tools list -- registry is NOT used for this agent
  shallow_research_agent:
    _type: shallow_research_agent
    llm: nemotron_llm
    tools:
      - web_search_tool
      - knowledge_search
```

## Adding MCP Tools as Data Sources

MCP tools (via `mcp_client` function groups) work with the registry the same way as any other tool. Add the group name to a registry source entry and all agents get it automatically:

```yaml
# Connect to an external MCP server
function_groups:
  mcp_financial_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${MCP_SERVER_URL:-http://localhost:9901/mcp}

functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        tools:
          - web_search_tool
          - advanced_web_search_tool
      - id: knowledge_layer
        name: "Knowledge Base"
        tools:
          - knowledge_search
      - id: financial_data
        name: "Financial Data"
        description: "Query financial reports and market data via MCP."
        tools:
          - mcp_financial_tools         # function group name
```

That's it -- one registry entry. Every agent automatically gets the MCP tools. The UI shows a "Financial Data" toggle. Per-request `data_sources` filtering works.

The registry auto-detects that `mcp_financial_tools` is a function group and uses NAT's group separator (`__`) for prefix matching. All tools exposed by the MCP server (e.g., `mcp_financial_tools__get_stock_quote`, `mcp_financial_tools__get_earnings`) map to the `financial_data` data source.

For details on MCP server setup, transport options, tool overrides, and prompt tuning, refer to [MCP Tools](./mcp-tools.md).

## Disabling a Tool

To disable a tool (for example, to avoid API usage or restrict agents to specific sources), remove it from the `data_source_registry`:

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        tools:
          - web_search_tool
          - advanced_web_search_tool
      # paper_search removed -- no agent will receive it
```

Since agents inherit from the registry, removing a tool from the registry removes it from all agents. No per-agent config changes needed.

Optionally comment out or remove the tool's function definition in `functions` so the config is clearer.

## Adding New Tools or Data Sources

For guidance on implementing and registering new tools or data sources, refer to:

- [Adding a Tool](../extending/adding-a-tool.md) -- How to create and register a new tool with the NeMo Agent Toolkit.
- [Adding a Data Source](../extending/adding-a-data-source.md) -- How to add a new data source, register it with the data source registry, and use MCP tools as data sources.
