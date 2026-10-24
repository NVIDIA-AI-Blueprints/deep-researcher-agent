<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adding a Tool

This guide walks through creating a new tool (NeMo Agent Toolkit function) end-to-end. Tools are the primary way agents interact with external services, APIs, and data sources. Each tool is a standalone package that registers itself with NeMo Agent Toolkit's plugin system.

The pattern follows the existing Tavily web search tool at `sources/tavily_web_search/`.

```{note}
The model block later in this guide demonstrates wiring only. The NVIDIA API Catalog serving profile for Nemotron 3.5
Lightning has a known [shallow citation-output limitation](../resources/troubleshooting.md#nemotron-35-lightning-on-nvidia-api-catalog).
Deep Researcher Agent fails closed rather than publishing citation-incomplete drafts.
```

---

## Prerequisites

- The repository virtual environment is active (`.venv`)
- You understand NeMo Agent Toolkit's `@register_function` decorator and YAML configuration

---

## Step 1: Create the Package

Tools live under `sources/` as independent Python packages with their own `pyproject.toml`. This keeps dependencies isolated and makes the tool reusable across projects.

```
sources/my_search_tool/
    pyproject.toml
    README.md
    src/
        __init__.py
        register.py      # Config + NAT registration
        my_client.py     # Tool implementation (API client, etc.)
    tests/
        test_my_tool.py
```

```bash
mkdir -p sources/my_search_tool/src sources/my_search_tool/tests
touch sources/my_search_tool/src/__init__.py
```

---

## Step 2: Define the Configuration Class

The config class extends `FunctionBaseConfig` and declares the `name` that YAML configs reference with `_type`. Place this in `register.py`.

```python
# sources/my_search_tool/src/register.py

import logging
import os

from pydantic import Field, SecretStr

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MySearchToolConfig(FunctionBaseConfig, name="my_search_tool"):
    """
    Tool that searches a custom API for relevant information.
    Requires a MY_SEARCH_API_KEY environment variable or api_key config.
    """

    max_results: int = Field(
        default=5, description="Maximum number of search results to return"
    )
    api_key: SecretStr | None = Field(
        default=None, description="API key for the search service"
    )
    timeout: int = Field(
        default=30, description="Timeout in seconds for requests"
    )
```

Key points:

- The `name="my_search_tool"` becomes the `_type:` value in YAML.
- Use `SecretStr` for API keys to prevent accidental logging.
- Document the required environment variables in field descriptions.

---

## Step 3: Implement the Tool Function

The tool function is what the LLM invokes. It must have clear type annotations and a docstring -- the LLM uses the docstring to decide when to call the tool.

```python
# sources/my_search_tool/src/my_client.py

import html
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_INVALID_XML_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")


def _xml_text(value: object) -> str:
    """Normalize provider text and remove characters forbidden by XML 1.0."""
    return _INVALID_XML_CHARACTERS.sub("", "" if value is None else str(value))


def _render_document(url: object, title: object, content: object) -> str:
    """Render provider-controlled fields inside the trusted document structure."""
    url_text = _xml_text(url)
    title_text = _xml_text(title)
    content_text = _xml_text(content)
    return (
        f'<Document href="{html.escape(url_text, quote=True)}">\n'
        f"<title>\n{html.escape(title_text, quote=True)}\n</title>\n"
        f"{html.escape(content_text, quote=True)}\n</Document>"
    )


class MySearchClient:
    """Client for the custom search API."""

    def __init__(self, api_key: str, timeout: int = 30, max_results: int = 5):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results

    async def search(self, query: str) -> str:
        """Execute a search query and return formatted results."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.example.com/search",
                params={"q": query, "limit": self.max_results},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        if not results:
            return "No results found for this query."

        formatted = []
        for doc in results:
            formatted.append(
                _render_document(
                    doc.get("url"),
                    doc.get("title"),
                    doc.get("content"),
                )
            )

        return "\n\n---\n\n".join(formatted)
```

---

## Step 4: Register the Tool

Add the `@register_function` decorated async generator to `register.py`. This wires the config to the implementation.

```python
# Continuing in sources/my_search_tool/src/register.py

from .my_client import MySearchClient

# Track if we've already warned about missing API key
_missing_key_warned = False


@register_function(config_type=MySearchToolConfig)
async def my_search_tool(tool_config: MySearchToolConfig, builder: Builder):
    """Register my custom search tool."""

    # Resolve API key from config or environment
    if not os.environ.get("MY_SEARCH_API_KEY") and tool_config.api_key:
        os.environ["MY_SEARCH_API_KEY"] = tool_config.api_key.get_secret_value()

    api_key = os.environ.get("MY_SEARCH_API_KEY")

    if not api_key:
        global _missing_key_warned
        if not _missing_key_warned:
            logger.warning(
                "MY_SEARCH_API_KEY not found. The tool will be registered "
                "but will return an error when called."
            )
            _missing_key_warned = True

        # Yield a stub that returns a friendly error
        async def _stub(query: str) -> str:
            """Search tool (unavailable - missing MY_SEARCH_API_KEY)."""
            return (
                "Error: Search is unavailable because MY_SEARCH_API_KEY is not set.\n"
                "Set the API key in your environment or .env file and restart."
            )

        yield FunctionInfo.from_fn(_stub, description=_stub.__doc__)
        return

    # Create the real client
    client = MySearchClient(
        api_key=api_key,
        timeout=tool_config.timeout,
        max_results=tool_config.max_results,
    )

    async def _search(query: str) -> str:
        """Searches for information using the custom search API.

        Args:
            query: The search query string.

        Returns:
            Formatted search results with source URLs.
        """
        return await client.search(query)

    yield FunctionInfo.from_fn(
        _search,
        description=_search.__doc__,
    )
```

Important patterns from the existing codebase:

- **Graceful degradation**: When the API key is missing, register a stub that returns an error message instead of crashing at startup.
- **Environment variable resolution**: Check the environment first, then fall back to the config value.
- **Docstring as description**: The inner function's docstring is passed as the tool description. The LLM reads this to decide when to call the tool, so make it clear and specific.

---

## Step 5: Create pyproject.toml

```toml
# sources/my_search_tool/pyproject.toml

[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools >= 64", "setuptools-scm>=8"]

[tool.setuptools]
packages = ["my_search_tool"]
package-dir = {"my_search_tool" = "src"}

[project]
name = "my-search-tool"
version = "1.0.0"
description = "NAT-based custom search tool"
requires-python = ">=3.11,<3.14"
dependencies = [
    "nvidia-nat-core==1.8.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

[project.entry-points."nat.plugins"]
my_search_tool = "my_search_tool.register"
```

Key points:

- The `package-dir` maps the package name to `src/` so Python can find your module.
- The entry point key (`my_search_tool`) maps to the `register` module, which triggers `@register_function` at import time.
- Pin `nvidia-nat-core` to the same version used by the main project.

---

## Step 6: Add to the Workspace

Add your package to the uv workspace in the root `pyproject.toml` if it follows the `sources/*` pattern (it should be auto-discovered):

```toml
[tool.uv.workspace]
members = [
    "sources/*",         # <-- Auto-discovers your package
    "frontends/deep_researcher_api",
    "frontends/cli",
    "frontends/debug",
]
```

Install the new package:

```bash
uv pip install -e ./sources/my_search_tool
```

---

## Step 7: Use in a YAML Config

Reference your tool in any workflow configuration:

```yaml
llms:
  research_llm:
    _type: nim
    model_name: nvidia/nemotron-3.5-lightning-30b-a3b

functions:
  my_search:
    _type: my_search_tool
    max_results: 10
    timeout: 15

  shallow_research_agent:
    _type: shallow_research_agent
    llm: research_llm
    tools:
      - my_search

workflow:
  _type: shallow_research_workflow
```

Run it:

```bash
dotenv -f deploy/.env run .venv/bin/nat run \
  --config_file configs/my_config.yml \
  --input "What is quantum computing?"
```

---

## Step 8: Test Your Tool

### Unit Tests

```python
# sources/my_search_tool/tests/test_my_tool.py

import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from my_search_tool.my_client import MySearchClient


@pytest.mark.asyncio
async def test_search_returns_results():
    """Test that the search client returns formatted results."""
    url = 'https://example.com/pa\x00th?q="quoted"&close=</Document>'
    title = 'Example\x00 & "title" </title>'
    content = "Example\x00 result & </Document>"
    mock_response = {
        "results": [
            {"url": url, "title": title, "content": content},
        ]
    }

    client = MySearchClient(api_key="test-key", max_results=5)

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: mock_response,
            raise_for_status=lambda: None,
        )
        result = await client.search("test query")

    assert result.count("<Document ") == 1
    assert result.count("</Document>") == 1
    document = ET.fromstring(result)
    title_element = document.find("title")
    assert title_element is not None
    assert document.attrib["href"] == url.replace("\x00", "")
    assert (title_element.text or "").strip("\n") == title.replace("\x00", "")
    assert (title_element.tail or "").strip("\n") == content.replace("\x00", "")


@pytest.mark.asyncio
async def test_search_no_results():
    """Test graceful handling of empty results."""
    client = MySearchClient(api_key="test-key")

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"results": []},
            raise_for_status=lambda: None,
        )
        result = await client.search("nonexistent topic")

    assert "No results found" in result
```

### Integration Test

```bash
# Ensure API key is available
export MY_SEARCH_API_KEY="your-key-here"  # pragma: allowlist secret

.venv/bin/nat run --config_file configs/my_config.yml --input "test query"
```

---

## Tool Design Best Practices

### Docstrings Matter

The LLM reads the tool's docstring (passed as `description`) to decide when to call it. Write docstrings that clearly describe:

- **What** the tool does
- **When** to use it (what kinds of queries it handles)
- **What** it returns

```python
async def _search(query: str) -> str:
    """Searches for peer-reviewed academic papers and scientific publications.

    This tool returns papers from Google Scholar with citations, abstracts,
    and links for research queries requiring authoritative, scholarly sources.
    """
```

### Error Handling

Tools should never raise exceptions that crash the agent. Return error messages as strings:

```python
async def _search(query: str) -> str:
    for attempt in range(max_retries):
        try:
            return await client.search(query)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: Search failed - {str(e)}"
            await asyncio.sleep(2 ** attempt)
```

### Output Formatting

Use the XML `<Document>` format for results that include URLs. This allows the agent's prompt to extract and cite sources.
Reuse the private `_render_document()` helper from Step 3:

```python
result = _render_document(url, title, content)
```

Keep this renderer private to the independently installable plugin and preserve
the fixed document shape. Removing XML 1.0-invalid code points and escaping
provider-controlled values prevents them from breaking the fragment, closing
trusted tags, or creating additional document elements.

---

## Existing Tool Reference

| Tool | `_type` | Package | API Key |
|---|---|---|---|
| Tavily Web Search | `tavily_web_search` | `sources/tavily_web_search` | `TAVILY_API_KEY` |
| Exa Web Search | `exa_web_search` | `sources/exa_web_search` | `EXA_API_KEY` |
| Nimble Web Search | `nimble_web_search` | `sources/nimble_web_search` | `NIMBLE_API_KEY` |
| Firecrawl Web Search | `firecrawl_web_search` | `sources/firecrawl_web_search` | `FIRECRAWL_API_KEY` (optional) |
| Google Scholar | `paper_search` | `sources/google_scholar_paper_search` | `SERPER_API_KEY` |
| Knowledge Layer | `knowledge_retrieval` | `sources/knowledge_layer` | (varies by backend) |

---

## Checklist

- [ ] Package created under `sources/<name>/` with `pyproject.toml`
- [ ] Config class extends `FunctionBaseConfig` with a unique `name`
- [ ] Tool function registered with `@register_function`
- [ ] Graceful degradation when API key is missing (stub function)
- [ ] Clear docstring for LLM tool selection
- [ ] Entry point in `pyproject.toml` `[project.entry-points."nat.plugins"]`
- [ ] Installed with `uv pip install -e ./sources/<name>`
- [ ] YAML config references the tool correctly
- [ ] Unit tests written and passing

---

## Related

- [Adding a Data Source](./adding-a-data-source.md) -- Data source plugin pattern
