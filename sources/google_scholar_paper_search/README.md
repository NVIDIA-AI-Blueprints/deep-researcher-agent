# Google Scholar Paper Search

A NeMo Agent Toolkit function that searches for academic papers using Google Scholar. You can choose between four backend providers:

| Provider | `provider` value | Env var | Sign-up |
|----------|---------------|---------|---------|
| **Serper** (default) | `serper` | `SERPER_API_KEY` | [serper.dev](https://serper.dev/) |
| **SerpAPI** | `serpapi` | `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com/) |
| **SearchAPI** | `searchapi` | `SEARCHAPI_API_KEY` | [searchapi.io](https://www.searchapi.io/) |
| **Serply** | `serply` | `SERPLY_API_KEY` | [serply.io](https://serply.io/) |

All four query Google Scholar and return the same normalized result shape (title, year, snippet, link, publication info, citations), so the agent-facing tool behavior is identical regardless of provider. Serply's Scholar results do not include an abstract snippet, so `snippet` is empty for that provider; the publication info and citation count are still filled in.

## Prerequisites

You need an API key for **one** of the supported providers. The default is Serper.

1. Create an account at your chosen provider (see table above)
2. Generate an API key from the dashboard
3. Add the key to your `deploy/.env` file in the project root, for example:

```bash
SERPER_API_KEY="your-serper-api-key"
# OR
SERPAPI_API_KEY="your-serpapi-api-key"  # pragma: allowlist secret
# OR
SEARCHAPI_API_KEY="your-searchapi-api-key"  # pragma: allowlist secret
# OR
SERPLY_API_KEY="your-serply-api-key"  # pragma: allowlist secret
```

Alternatively, you can provide the API key directly in the configuration file (see below).

## Installation

Install the package using `uv` from the project root:

```bash
uv pip install -e sources/google_scholar_paper_search
```

After installation, verify the plugin is registered:

```bash
nat info components -t function | grep paper_search
```

## Configuration

### Adding the Function

Add the `paper_search` function to the `functions` section of your workflow configuration file. Use the `provider` field to select the backend (defaults to `serper`):

```yaml
functions:
  paper_search_tool:
    _type: paper_search
    provider: serper          # 'serper' (default), 'serpapi', 'searchapi', or 'serply'
    max_results: 10
    timeout: 30
    serper_api_key: ${SERPER_API_KEY}
```

To use SerpAPI instead:

```yaml
functions:
  paper_search_tool:
    _type: paper_search
    provider: serpapi
    serpapi_api_key: ${SERPAPI_API_KEY}
```

To use SearchAPI:

```yaml
functions:
  paper_search_tool:
    _type: paper_search
    provider: searchapi
    searchapi_api_key: ${SEARCHAPI_API_KEY}
```

To use Serply:

```yaml
functions:
  paper_search_tool:
    _type: paper_search
    provider: serply
    serply_api_key: ${SERPLY_API_KEY}
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | string | `serper` | Backend provider: `serper`, `serpapi`, `searchapi`, or `serply` |
| `max_results` | integer | 10 | Maximum number of search results to return (capped at 50) |
| `timeout` | integer | 30 | Timeout in seconds for search requests |
| `serper_api_key` | string | None | Serper API key (required when `provider: serper`; also read from `SERPER_API_KEY` env var) |
| `serpapi_api_key` | string | None | SerpAPI key (required when `provider: serpapi`; also read from `SERPAPI_API_KEY` env var) |
| `searchapi_api_key` | string | None | SearchAPI key (required when `provider: searchapi`; also read from `SEARCHAPI_API_KEY` env var) |
| `serply_api_key` | string | None | Serply key (required when `provider: serply`; also read from `SERPLY_API_KEY` env var) |

### Adding as a Tool to an Agent

After defining the function, add it to the `tools` list of any agent that should use paper search capabilities:

```yaml
functions:
  paper_search_tool:
    _type: paper_search
    max_results: 5
    serper_api_key: ${SERPER_API_KEY}

  my_research_agent:
    _type: shallow_research_agent
    llm: my_llm
    tools:
      - paper_search_tool
    max_llm_turns: 10
```

### Complete Example

Here is a complete configuration example showing how to integrate the paper search tool with SerpAPI as the provider:

```yaml
llms:
  nemotron_ultra_llm:
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "https://integrate.api.nvidia.com/v1"
    temperature: 0.2
    max_tokens: 16384

  nemotron_ultra_writer_llm:
    _type: nim
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    base_url: "https://integrate.api.nvidia.com/v1"
    temperature: 0.2
    max_tokens: 32768

functions:
  paper_search_tool:
    _type: paper_search
    provider: serpapi
    max_results: 10
    serpapi_api_key: ${SERPAPI_API_KEY}

  web_search_tool:
    _type: tavily_internet_search
    max_results: 5

  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_ultra_llm
    source_router_llm: nemotron_ultra_llm
    researcher_llm: nemotron_ultra_llm
    planner_llm: nemotron_ultra_llm
    writer_llm: nemotron_ultra_writer_llm
    tools:
      - paper_search_tool
      - web_search_tool

workflow:
  _type: chat_deepresearcher_agent
```

## Usage

The paper search function accepts the following arguments when called by an agent:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | Yes | The search query for finding academic papers |
| `year` | string | No | Year or year range filter (for example, "2023" or "2020-2023") |

## Troubleshooting

### Common Issues

**API key not found**

If you see an error about the API key not being found:

- Verify the correct environment variable is set for your chosen `provider` (`SERPER_API_KEY`, `SERPAPI_API_KEY`, `SEARCHAPI_API_KEY`, or `SERPLY_API_KEY`)
- Alternatively, ensure the matching key field is specified in the configuration file
- Make sure the `provider` field matches the key you provided

**Request timeout**

If searches are timing out:

- Increase the `timeout` value in the configuration
- Check your network connection

**No results returned**

If no papers are found:

- Try a broader search query
- Remove year filters to expand the search range
- Verify your provider API key has available quota

## Disabling Paper Search

If you don't have a provider API key or don't need paper search functionality, you can disable it by removing the tool from your configuration:

### Remove from Configuration

Edit your configuration file (for example, `configs/config_cli_default.yml`) and remove or comment out the `paper_search_tool` definition:

```yaml
functions:
  # Remove or comment out this section
  # paper_search_tool:
  #   _type: paper_search
  #   provider: serper
  #   max_results: 5
  #   serper_api_key: ${SERPER_API_KEY}
```

Also remove it from any agents that use it:

```yaml
functions:
  deep_research_agent:
    _type: deep_research_agent
    orchestrator_llm: nemotron_llm
    max_loops: 2
    tools:
      # Remove paper_search_tool from the tools list
      - advanced_web_search_tool
```

After making these changes, the agent will function without paper search capabilities.
