# Firecrawl Web Search

NAT-based web search tool backed by the [Firecrawl Search API](https://www.firecrawl.dev/search?utm_source=aiq&utm_medium=integration). Each result can include the page content as Markdown, so the agent reads the source page instead of a snippet. It follows the same package pattern as the other web search sources in this repository.

Works without an API key under daily per-IP request and credit limits (HTTP 429 when exceeded). The keyless budget is small and meant for trying it out; for regular use, set the `FIRECRAWL_API_KEY` environment variable or the `api_key` config option. A key is not needed when `api_url` points at a self-hosted Firecrawl instance.

```yaml
functions:
  web_search_tool:
    _type: firecrawl_web_search
    max_results: 5
    max_content_length: 10000
```

Queries (up to 400 characters) are sent to `api.firecrawl.dev` unless `api_url` points at a self-hosted instance. Without a key they are unauthenticated and rate-limited per IP.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_results` | `5` | Number of results to return (1-100). Total output is about `max_results` x `max_content_length` characters. |
| `api_key` | `None` | Firecrawl API key (optional, raises rate limits). Falls back to `FIRECRAWL_API_KEY`. |
| `api_url` | `None` | API base URL. Falls back to `FIRECRAWL_API_URL`, then `https://api.firecrawl.dev`. |
| `max_retries` | `3` | Maximum attempts (1-10) for rate limits, timeouts, and server errors. |
| `scrape_results` | `true` | Return each result's page as Markdown. When `false`, results use the search description only. |
| `max_content_length` | `10000` | Max characters per result (minimum 1). `null` disables truncation. |
| `tbs` | `None` | Time filter, for example `qdr:w` for the past week. |
| `country` | `None` | ISO 3166-1 alpha-2 country code, for example `US`. |
| `timeout_ms` | `45000` | Search timeout in milliseconds (1000-300000). |

## Credits

Search costs 2 credits per 10 results, and each scraped result adds 1 credit, so the default of 5 results with scraping on uses about 7 credits per search. Set `scrape_results: false` to pay only for the search.

Get a key at [firecrawl.dev](https://www.firecrawl.dev/app/api-keys?utm_source=aiq&utm_medium=integration). See the [Search API reference](https://docs.firecrawl.dev/api-reference/endpoint/search) for the full request options.
