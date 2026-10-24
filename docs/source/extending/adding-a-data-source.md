<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adding a Data Source

This guide walks through creating a new data source plugin. Data sources are specialized tools that provide domain-specific search capabilities -- academic papers, patent databases, financial data, internal knowledge bases, and similar services.

The pattern follows the existing Google Scholar paper search at `sources/google_scholar_paper_search/`.

---

## Data Source vs. Tool

A data source is architecturally identical to a [tool](./adding-a-tool.md). The distinction is conceptual:

| Aspect | Tool | Data Source |
|---|---|---|
| **Purpose** | General utility (web search, code execution) | Domain-specific information retrieval |
| **Location** | `sources/` | `sources/` |
| **Registration** | `@register_function` | `@register_function` |
| **Agent interaction** | Agent calls it for actions | Agent calls it for knowledge |

Both use the same NeMo Agent Toolkit plugin pattern. This guide focuses on data-source-specific patterns like structured result formatting, pagination, and search parameter handling.

---

## Prerequisites

- The repository virtual environment is active (`.venv`)
- You understand NeMo Agent Toolkit's `@register_function` decorator
- You have access to the data source's API or SDK

---

## Step 1: Create the Package

```
sources/my_data_source/
    pyproject.toml
    README.md
    src/
        __init__.py
        register.py       # Config + NAT registration
        search.py          # Search implementation
    tests/
        test_search.py
```

```bash
mkdir -p sources/my_data_source/src sources/my_data_source/tests
touch sources/my_data_source/src/__init__.py
```

---

## Step 2: Implement the Search Client

Separate the API client logic from the NeMo Agent Toolkit registration. This makes it testable independently.

```python
# sources/my_data_source/src/search.py

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class PatentSearchClient:
    """Client for searching a patent database API."""

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_results: int = 10,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results

    async def search(
        self,
        query: str,
        year: str | None = None,
        jurisdiction: str | None = None,
    ) -> str:
        """Search for patents matching the query.

        Args:
            query: The search query describing the invention or technology.
            year: Optional year filter (e.g., "2024").
            jurisdiction: Optional jurisdiction filter (e.g., "US", "EP").

        Returns:
            Formatted patent search results as a string.
        """
        params: dict[str, Any] = {
            "q": query,
            "limit": self.max_results,
        }
        if year:
            params["year"] = year
        if jurisdiction:
            params["jurisdiction"] = jurisdiction

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.patents.example.com/v2/search",
                params=params,
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            data = response.json()

        patents = data.get("patents", [])
        if not patents:
            return f"No patents found for query: {query}"

        results = []
        for patent in patents:
            patent_id = patent.get("id", "")
            title = patent.get("title", "")
            abstract = patent.get("abstract", "")
            filing_date = patent.get("filing_date", "")
            url = patent.get("url", "")

            results.append(
                f"**{title}** ({patent_id})\n"
                f"Filed: {filing_date}\n"
                f"Abstract: {abstract}\n"
                f"Link: {url}"
            )

        return "\n\n---\n\n".join(results)
```

---

## Step 3: Define Config and Register

```python
# sources/my_data_source/src/register.py

import logging
import os

from pydantic import Field, SecretStr

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .search import PatentSearchClient

logger = logging.getLogger(__name__)

_missing_key_warned = False


class PatentSearchConfig(FunctionBaseConfig, name="patent_search"):
    """
    Configuration for the patent search data source.

    Searches a patent database API for relevant patents.
    Requires a PATENT_API_KEY environment variable or config.
    """

    timeout: int = Field(
        default=30, description="Timeout in seconds for search requests"
    )
    max_results: int = Field(
        default=10, description="Maximum number of results to return"
    )
    api_key: SecretStr | None = Field(
        default=None, description="API key for the patent search service"
    )


@register_function(config_type=PatentSearchConfig)
async def patent_search(tool_config: PatentSearchConfig, builder: Builder):
    """Register patent search data source."""

    # Resolve API key
    if not os.environ.get("PATENT_API_KEY") and tool_config.api_key:
        os.environ["PATENT_API_KEY"] = tool_config.api_key.get_secret_value()

    api_key = os.environ.get("PATENT_API_KEY")

    if not api_key:
        global _missing_key_warned
        if not _missing_key_warned:
            logger.warning(
                "PATENT_API_KEY not found. The patent search tool will be "
                "registered but will return an error when called."
            )
            _missing_key_warned = True

        async def _stub(query: str, year: str | None = None) -> str:
            """Patent search (unavailable - missing PATENT_API_KEY)."""
            return (
                "Error: Patent search is unavailable because PATENT_API_KEY is not set.\n"
                "Set the API key in your environment or .env file and restart."
            )

        yield FunctionInfo.from_fn(_stub, description=_stub.__doc__)
        return

    client = PatentSearchClient(
        api_key=api_key,
        timeout=tool_config.timeout,
        max_results=tool_config.max_results,
    )

    async def _patent_search(
        query: str,
        year: str | None = None,
        jurisdiction: str | None = None,
    ) -> str:
        """Searches for patents and intellectual property filings.

        This tool returns patents from a patent database with titles,
        abstracts, filing dates, and links. Use for queries about
        inventions, prior art, technology patents, and IP research.
        """
        return await client.search(query, year, jurisdiction)

    yield FunctionInfo.from_fn(
        _patent_search,
        description=_patent_search.__doc__,
    )
```

---

## Step 4: Create pyproject.toml

```toml
# sources/my_data_source/pyproject.toml

[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools >= 64", "setuptools-scm>=8"]

[tool.setuptools]
packages = ["my_data_source"]
package-dir = {"my_data_source" = "src"}

[project]
name = "my-data-source"
version = "1.0.0"
description = "NAT-based patent search data source"
requires-python = ">=3.11,<3.14"
dependencies = [
    "nvidia-nat-core==1.8.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

[project.entry-points."nat.plugins"]
patent_search = "my_data_source.register"
```

Install:

```bash
uv pip install -e ./sources/my_data_source
```

---

## Step 5: Register as a Data Source in YAML

To make your data source appear in the UI as a toggleable source with per-message filtering, add it to the `data_source_registry` in your config. Agents automatically inherit all registry tools, so **this is the only config change needed**:

```yaml
functions:
  # Register data sources for UI toggles and filtering
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        description: "Search the web for real-time information."
        tools:
          - web_search_tool
      - id: patent_search
        name: "Patent Search"
        description: "Search patent databases for intellectual property filings."
        tools:
          - patent_tool

  # Define the tool instances
  patent_tool:
    _type: patent_search
    max_results: 5
    timeout: 15

  web_search_tool:
    _type: tavily_web_search
    max_results: 3

  # Agents inherit all registry tools automatically --
  # no need to list tools per-agent unless you want to exclude specific ones
  shallow_research_agent:
    _type: shallow_research_agent
    llm: research_llm
```

The `data_source_registry` provides:

- **`GET /v1/data_sources`** API endpoint returns the registered sources (the UI renders these as toggles)
- **Per-message filtering** via `data_sources: ["web_search"]` in the WebSocket chat payload, or as a `data_sources` field on `POST /v1/jobs/async/submit` -- only tools belonging to selected sources are active
- **Display metadata** (name, description) shown in the UI
- **Auth gating** -- set `requires_auth: true` on a source to grey it out in the UI until the user signs in (e.g., enterprise sources that need user-level OAuth tokens). Sources using backend API keys (Tavily, Serper) should leave this `false` (the default).
- **Auto-inheritance** -- all agents get every registered tool by default (use `exclude_tools` on an agent for per-agent specialization)

If a tool isn't listed in any `data_source_registry` source entry, it is always included regardless of filtering (e.g., utility tools like "think" or "calculator"). Passing an explicit empty list (`data_sources: []`) -- in the WebSocket chat payload or in a `POST /v1/jobs/async/submit` body -- disables data-source tools while leaving those unmapped utility tools available.

### Source Entry Field Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | string | *required* | Unique key used in API payloads and filtering |
| `name` | string | *required* | Display name shown in the UI |
| `description` | string | `""` | Human-readable description for the UI |
| `tools` | list | `[]` | NAT function or function group names belonging to this source |
| `requires_auth` | bool | `false` | Grey out in UI until user signs in (for user-token sources) |
| `default_enabled` | bool | `true` | Whether enabled by default in the UI |

---

## Using MCP Tools as Data Sources

NAT MCP client tools are function groups. They work with the data source registry the same way as any other function group -- just reference the group name in `tools`:

```yaml
function_groups:
  mcp_docs_search:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"

functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        description: "Search the web for real-time information."
        tools:
          - web_search_tool
      - id: docs_search
        name: "Internal Docs"
        description: "Search internal documentation via MCP."
        tools:
          - mcp_docs_search

  web_search_tool:
    _type: tavily_web_search

  deep_research_agent:
    _type: deep_research_agent
    tools:
      - web_search_tool
      - mcp_docs_search
```

The registry auto-detects that `mcp_docs_search` is a function group and uses NAT's group separator (`__`) for prefix matching. All tools exposed by the MCP server (e.g., `mcp_docs_search__find_pages`, `mcp_docs_search__get_page`) will map to the `docs_search` data source.

---

## Data Source Design Patterns

### Multi-Parameter Search Functions

Data sources often accept optional filters beyond the query string. Expose these as optional parameters with clear type annotations:

```python
async def _search(
    query: str,
    year: str | None = None,
    jurisdiction: str | None = None,
    category: str | None = None,
) -> str:
    """Search patents with optional filters.

    Args:
        query: Natural language description of the technology.
        year: Filter by filing year (e.g., "2024").
        jurisdiction: Filter by jurisdiction ("US", "EP", "CN").
        category: Filter by patent category.
    """
```

The LLM will use the parameter descriptions to decide which filters to apply.

### Structured Result Formatting

Format results so the agent can extract citations. Use a consistent pattern:

For pseudo-XML output, use the
[fixed-shape renderer pattern](./adding-a-tool.md#output-formatting), which
escapes every provider-controlled field before interpolation. Do not insert raw
provider output into markup.

For a plain structured-text alternative:

```python
f"**{title}** ({id})\nAbstract: {abstract}\nLink: {url}"
```

### Pagination

If the source API supports pagination, handle it internally rather than exposing it to the agent:

```python
async def search(self, query: str) -> str:
    all_results = []
    offset = 0

    while len(all_results) < self.max_results:
        batch = await self._fetch_page(query, offset, batch_size=20)
        if not batch:
            break
        all_results.extend(batch)
        offset += len(batch)

    return self._format_results(all_results[:self.max_results])
```

### Rate Limiting and Retries

```python
import asyncio

async def search(self, query: str) -> str:
    for attempt in range(self.max_retries):
        try:
            return await self._do_search(query)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited, retrying in {wait}s")
                await asyncio.sleep(wait)
            elif attempt == self.max_retries - 1:
                return f"Error: Search failed after {self.max_retries} attempts - {e}"
            else:
                await asyncio.sleep(1)
```

---

## Existing Data Source Reference

| Data Source | `_type` | Package | Description |
|---|---|---|---|
| Tavily Web Search | `tavily_web_search` | `sources/tavily_web_search` | General web search through Tavily API |
| Exa Web Search | `exa_web_search` | `sources/exa_web_search` | General web search through Exa API |
| Nimble Web Search | `nimble_web_search` | `sources/nimble_web_search` | General web search through Nimble API (`langchain-nimble`) |
| Firecrawl Web Search | `firecrawl_web_search` | `sources/firecrawl_web_search` | General web search with page Markdown through the Firecrawl API |
| Google Scholar | `paper_search` | `sources/google_scholar_paper_search` | Academic papers through Serper/Google Scholar |
| Knowledge Layer | `knowledge_retrieval` | `sources/knowledge_layer` | Document retrieval through pluggable backends |

---

## Checklist

- [ ] Package created under `sources/<name>/` with `pyproject.toml`
- [ ] Search client implemented separately from registration
- [ ] Config class with `FunctionBaseConfig` and unique `name`
- [ ] `@register_function` with stub fallback for missing API key
- [ ] Clear docstring explaining what the data source searches
- [ ] Optional search parameters with descriptions for the LLM
- [ ] Proper error handling and retries
- [ ] Entry point in `pyproject.toml`
- [ ] Added to `data_source_registry` in YAML config for UI integration
- [ ] Installed and tested with `nat run`

---

## Related

- [Adding a Tool](./adding-a-tool.md) -- General tool creation guide
- [Knowledge Layer SDK](../reference/knowledge-layer-sdk.md) -- Build custom retrieval backends
- [Prompts](../customization/prompts.md) -- Modify agent behavior
