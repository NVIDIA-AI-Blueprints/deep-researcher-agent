# Context.dev Search

NAT data sources backed by the [Context.dev Search API](https://docs.context.dev/search/overview):

| Function | Purpose | Output |
|---|---|---|
| `context_dev_web_search` | General web search with the passages of each page that answer the question | `<Document href="...">` blocks, the same shape as the other web search sources |
| `context_dev_paper_search` | Research paper search over the same API, which includes the Context.dev Research Index of research papers | Numbered `N. **Title**` entries, the same layout as `paper_search` |

Both tools use one API key and plain `httpx`; there is no SDK dependency. Each result carries
[Highlights](https://docs.context.dev/scrape/highlights), the passages of the page that are relevant to
the query, so the researcher reads evidence instead of whole pages.

## Setup

1. Get an API key at [context.dev](https://context.dev/signup).
2. Set it in `deploy/.env`:

   ```bash
   CONTEXT_DEV_API_KEY=your-key
   ```

   Or export it in the shell that runs the agent:

   ```bash
   export CONTEXT_DEV_API_KEY=your-key
   ```

3. Select the tools in a workflow config. They are drop-in replacements for the existing web and
   paper search functions. Keep the existing function names (`web_search_tool`,
   `advanced_web_search_tool`, `paper_search_tool`) so the agents, prompts, and the
   `data_source_registry` entries that list those names need no changes:

   ```yaml
   functions:
     web_search_tool:
       _type: context_dev_web_search
       max_results: 5
     advanced_web_search_tool:
       _type: context_dev_web_search
       max_results: 10
     paper_search_tool:
       _type: context_dev_paper_search
       max_results: 10
   ```

If no key is configured, each tool registers a stub that returns setup instructions instead of
failing workflow startup.

## Configuration

Shared by both functions:

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `None` | API key. Falls back to `CONTEXT_DEV_API_KEY`. |
| `max_results` | `int` | `5` web, `10` paper | Results to return (1-100). |
| `max_content_length` | `int` | `4000` web, `1500` paper | Max characters of passages (or snippet) per result (100-20000). |
| `include_highlights` | `bool` | `true` | Return query-relevant passages. When `false`, each result carries only the search snippet. |
| `include_domains` | `list[str]` | `None` | Only return results from these domains (up to 100). Entries are normalized to host names (`https://www.arxiv.org/abs` becomes `arxiv.org`) and match subdomains. Paper search applies it after the search. |
| `exclude_domains` | `list[str]` | `None` | Drop results from these domains (up to 100), normalized the same way. Paper search applies it after the search. |
| `country` | `str` | `None` | ISO 3166-1 alpha-2 country code used to localize results. |
| `freshness` | `str` | `None` | `last_24_hours`, `last_week`, `last_month`, or `last_year`. |
| `zero_data_retention` | `bool` | `false` | Ask Context.dev not to retain the request or results. Requires an organization with zero data retention enabled. |
| `max_retries` | `int` | `3` | Total attempts for transient failures (network, 408, 429, 5xx), 1-10. |
| `timeout` | `float` | `30.0` | Timeout for each request, end to end, in seconds (2-300). |
| `api_base_url` | `str` | `https://api.context.dev/v1` | API base URL. Must use https (http is accepted only for localhost) so the key never travels in cleartext. |

Paper search only:

| Field | Type | Default | Description |
|---|---|---|---|
| `scholarly_sources_only` | `bool` | `true` | Keep only results hosted by publishers, preprint servers, proceedings, and scholarly indexes (`SCHOLARLY_DOMAINS` in `register.py`). Ignored when `include_domains` is non-empty. |

### Paper search and `scholarly_sources_only`

The Search API returns papers from the Context.dev Research Index alongside web pages: the same
unfiltered endpoint the [NEEDLE benchmark](https://keenableai.github.io/needle/) scores. With the
default `scholarly_sources_only: true`, the tool requests three times `max_results` (at most 100) and keeps
the results hosted by publishers, preprint servers (arXiv, bioRxiv, SSRN), proceedings (ACL Anthology,
NeurIPS, PMLR, OpenReview), PubMed Central, DOI links, and scholarly indexes, dropping news and blog
coverage of a paper. If none of the results is on a scholarly host, the unfiltered results are
returned rather than an empty answer. Set it to `false` to always return the unfiltered results.

The filter runs on the tool's side on purpose: the API leaves its indexed papers out of any request
that sets `includeDomains` or `excludeDomains`, or whose query uses an operator such as `site:`,
`OR`, a quoted phrase, or a `-` exclusion. For the same reason, paper search applies
`include_domains` (which replaces the scholarly list, with no fallback) and `exclude_domains` to the
results itself instead of sending them. Filtering by an allow-list means fetching up to three times as
many results, so such a paper search costs up to three times the per-result credits of an unfiltered
one; set `scholarly_sources_only: false` to avoid that.

The Search API has no publication-year filter, so unlike `paper_search` this tool does not take a
`year` argument. Use `freshness` for recency.

## Behavior

- **Retries:** transient failures (network errors, unusable response bodies, 408, 429, 5xx) are
  retried with exponential backoff capped at 30 seconds. A `Retry-After` within that cap is honored; a
  longer one ends the retries, since retrying sooner would hit the same limit. Invalid requests (400),
  invalid keys (401), exhausted credits (402), and missing entitlements (403) return immediately with
  an actionable message.
- **Deadlines:** `timeout` bounds the whole request. The API is given a deadline one second shorter,
  and when that deadline is at least five seconds it returns the results it already has instead of
  failing the search because one page was slow to read (logged at INFO).
- **Errors:** every failure is returned as a string starting with `Error:`, which citation verification
  treats as a status message rather than evidence. Messages include the API `request_id` and never the
  API key.
- **Output safety:** provider text is html-escaped inside `<Document>` blocks, characters that XML 1.0
  forbids are removed, whitespace entity artifacts from page conversion (`&#x20;`) become spaces, and
  results without an absolute http(s) URL are dropped because they cannot be cited. Redirects are
  reported, not followed, so the API key is never replayed to another host.

## Testing

Three layers, from cheapest to most realistic:

| Layer | Files | Network / key | What it covers |
|---|---|---|---|
| Unit | `test_context_dev_client.py`, `test_context_dev_web_search.py`, `test_context_dev_paper_search.py`, `test_context_dev_citation_compatibility.py` | No | Request building, error handling, retries, rendering, escaping, and compatibility with `citation_verification` |
| Replay | `test_context_dev_recorded_replay.py` + `fixtures/` | No | Real recorded API responses through the full pipeline (see `fixtures/README.md`) |
| Live | `test_context_dev_live_integration.py` | Yes, opt-in | Two real calls (one per tool) against the shared output contracts |

```bash
# Unit and replay tests (no key needed)
uv run pytest sources/context_dev_search/tests

# Live integration tests: two API calls, each bounded at 120 seconds
DEEP_RESEARCHER_CONTEXT_DEV_LIVE_TESTS=1 CONTEXT_DEV_API_KEY=<key> \
    uv run pytest sources/context_dev_search/tests -m integration -v
```

The live tests assert structure only: a non-error response, between 1 and `max_results` results,
every result with a title and an http(s) URL, every `<Document>` block XML-parseable, and at least one
non-empty body or snippet.
