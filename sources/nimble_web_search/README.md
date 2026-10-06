# Nimble Web Search

NAT-based [Nimble](https://nimbleway.com/) web search tool for agentic search workflows that need live web context. Requires a `NIMBLE_API_KEY` environment variable or `api_key` config.

## When to use

Use `nimble_web_search` when your Deep Researcher Agent needs fresh web context from Nimble's real-time web intelligence infrastructure. The provider is designed for agentic search workflows that benefit from live web discovery, structured results, and reliable retrieval through Nimble's search layer.

Choose `nimble_web_search` when Tavily or Exa are not the right fit for your workflow, or when you want to standardize web-search retrieval through Nimble. The provider follows the same integration pattern as `exa_web_search` and `tavily_web_search`, making it straightforward to configure as an alternative search backend in Deep Researcher Agent.

## Install

This package is installed automatically by `scripts/setup.sh` alongside the other data-source plugins. Manual install:

```bash
uv pip install -e ./sources/nimble_web_search
```

## Configure

Add a `nimble_web_search` function to your workflow YAML:

```yaml
functions:
  web_search_tool:
    _type: nimble_web_search
    max_results: 5
    search_depth: lite
    country: US
    locale: en
```

See [`docs/source/customization/configuration-reference.md`](../../docs/source/customization/configuration-reference.md) for the full parameter table.

## Environment

```bash
NIMBLE_API_KEY=...
```

Visit [Nimble](https://nimbleway.com/) to create an account, obtain an API key, and access the provider's setup guides. Store the key securely.

You can alternatively set `api_key` directly in the YAML (as a string). Both paths use the standard `nat.data_models.function.FunctionBaseConfig` `SecretStr` handling — the key is not logged.

## Test

The package ships three test layers; the first two are credential-free and run everywhere.

```bash
# 1. Mocked unit tests + recorded-response replay (credential-free; the default run)
uv run pytest sources/nimble_web_search -v
# → credential-free tests pass; the opt-in live test is skipped

# 2. Live integration test (opt-in; exactly one API call, bounded at 120 s)
DEEP_RESEARCHER_NIMBLE_LIVE_TESTS=1 NIMBLE_API_KEY=<key> \
    uv run pytest sources/nimble_web_search/tests -m integration -v

# Lint
uv run ruff check sources/nimble_web_search
uv run ruff format --check sources/nimble_web_search
```

- **Mocked unit tests** (`test_nimble_register.py`) cover: config defaults / all fields / invalid `search_depth` rejection, missing-key stub + warn-once, direct config-key passthrough to the SDK, successful render + description fallback, deep depth passthrough, query/content truncation, empty/error handling, retry-then-success, final-retry failure, 401, 403 enterprise-tier, non-default country/locale passthrough, and renderer behavior on titles containing special characters.
- **Recorded-response replay** (`test_nimble_recorded_replay.py`) replays real, redacted `NimbleSearchRetriever` responses ([`tests/fixtures/README.md`](tests/fixtures/README.md)) through the full provider pipeline — a deterministic test mode that needs no network and no key.
- **Live integration** (`test_nimble_live_integration.py`) runs the canned query `NVIDIA CUDA Toolkit documentation` once against the real API with the shipped defaults and asserts the structural output contract: a non-error response containing 1..`max_results` `<Document>` blocks, every block with an http(s) `href` and a title, every block XML-parseable, and at least one non-empty body. Assertions are structural — never content-exact — so ordinary result variation cannot flake the run. To run it in a dedicated opt-in CI job, add `NIMBLE_API_KEY` as a repository secret and set `DEEP_RESEARCHER_NIMBLE_LIVE_TESTS=1` in the job.

## Verification

Beyond the mocked unit tests above, verify the provider is fully integrated with Deep Researcher Agent by walking the [adding-a-data-source checklist](../../docs/source/extending/adding-a-data-source.md):

```bash
# 1. Mocked unit tests + recorded replay pass (CI-safe, no credentials)
uv run pytest sources/nimble_web_search -q
# → credential-free tests pass; the opt-in live test is skipped

# 2. NAT discovers the registered function
nat info components --types function | grep nimble_web_search
# → nimble_web_search 1.0.0 function

# 3. Lint, format, and dependency lock all clean
uv run ruff check sources/nimble_web_search
uv run ruff format --check sources/nimble_web_search
uv lock --check

# 4. Live smoke through any workflow that names `_type: nimble_web_search` (requires NIMBLE_API_KEY)
export NIMBLE_API_KEY=...
nat run --config_file <your-workflow.yml> --input "your test query"
```

Step 4 satisfies the checklist's "Installed and tested with `nat run`" item. Any of the existing Deep Researcher Agent web search configs (`configs/config_cli_default.yml`, `configs/config_web_default_llamaindex.yml`, etc.) becomes a Nimble-backed test by swapping `_type: tavily_web_search` → `_type: nimble_web_search` and translating `advanced_search: true` → `search_depth: deep`.

## Native capabilities

The provider exposes the following Nimble-specific surface. Defaults are tuned for the common Deep Researcher Agent research workflow (lite-mode SERP for a few results, US/English regional bias):

| Capability | Field | Default | Notes |
|---|---|---|---|
| Result count | `max_results` | `5` | Range `1-100` (Nimble's documented cap). Soft cap (Nimble may return up to N+2; see Known limitations). |
| Search depth | `search_depth` | `lite` | See the dedicated [Search depth](#search-depth) section below. |
| Search focus | `focus` | `general` | Nimble focus mode: `general` (default, broad web/research), `news` (news-publisher sources ordered by recency — not a recency filter; older articles still appear), or domain-specific `location` / `shopping` / `geo` / `social`. Leave `general` for normal research; the LLM never selects focus, so general queries can't drift to `news`. |
| Localization — country | `country` | `US` | Two-letter ISO 3166 country code (e.g. `FR`, `JP`, `GB`). Reaches the SDK constructor verbatim. |
| Localization — language | `locale` | `en` | ISO 639-1 language code (e.g. `fr`, `ja`). |
| Per-result content size | `max_content_length` | `10000` chars | Truncates each result's body to N chars (3-char ellipsis included). Minimum `1`; set to `null` to disable truncation; omit to use default. |
| Retries | `max_retries` | `3` | Exponential backoff on transient errors. Final failure surfaces a friendly per-status message (401, 403, generic). |
| Auth | `api_key` / `NIMBLE_API_KEY` | env or config | `pydantic.SecretStr`; never logged. A config-side `api_key` is passed directly to the SDK without modifying the process environment. |

Each rendered `<Document>` block contains the result URL, title, and body. Other provider metadata, including `entity_type`, is not included. The `include_answer=True` capability is intentionally **not exposed** in this initial integration. See [Known limitations](#known-limitations).

## Search depth

| Value | Behavior | Account requirement |
|---|---|---|
| `lite` (default) | Metadata only — URL, title, description. Token-efficient. | Any |
| `fast` | Enterprise tier. Lower latency, richer content. | **Enterprise account required.** Returns a 403 ToolException with `"search_depth='fast' is not enabled for this account. Contact sales for access."` on non-enterprise accounts. |
| `deep` | Higher token cost. May return short page-content snippets in addition to metadata. | Any (in this account; behavior may vary by tier) |

If you do not know your tier, leave `search_depth: lite` and let the description field carry the content.

## Known limitations

- `max_results` is a **soft cap**. The Nimble API may return up to N+2 documents when asked for N. The provider returns them all; Deep Researcher Agent's downstream consumers can slice if they need a hard cap.
- `lite` mode returns `page_content == ""` per result. The provider falls back to `description` (~150 chars per result, organic-result quality).
- `include_answer` is **not exposed** in this initial integration. It can be added in a follow-up.

## Security

- API key handling follows the existing `EXA_API_KEY` / `TAVILY_API_KEY` pattern: env var or `SecretStr` config; never logged.
- Untrusted API fields (`url`, `title`, body) are HTML-escaped before being rendered into the `<Document>` markup, so a result can't break the block or inject into downstream parsers.
- Default CI is credential-free and network-free. An explicitly configured live-test job may access Nimble when `NIMBLE_API_KEY` and `DEEP_RESEARCHER_NIMBLE_LIVE_TESTS=1` are set.
- The optional live smoke is documented in the PR description and uses a redacted output pattern.
