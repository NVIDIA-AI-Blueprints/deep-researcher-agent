# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NAT registration for the Context.dev web search and paper search tools.

Two functions share one HTTP client implementation:

- ``context_dev_web_search``: general web search. Each result is rendered as a
  ``<Document href="...">`` block, the same shape as the other web search
  sources, so citation verification and the researcher prompts work unchanged.
- ``context_dev_paper_search``: research paper search over the same Search
  API, which includes the Context.dev Research Index of research papers.
  Results use the numbered ``N. **Title**`` layout of ``paper_search`` so the
  citation parser recognizes titles.
"""

import html
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import Annotated
from urllib.parse import urlparse

import httpx
from pydantic import AfterValidator
from pydantic import Field
from pydantic import SecretStr

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .client import CONTEXT_DEV_API_URL
from .client import ContextDevAPIError
from .client import ContextDevSearchClient
from .client import Freshness
from .client import SearchResult
from .client import search_with_retries

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "CONTEXT_DEV_API_KEY"  # pragma: allowlist secret
SIGNUP_URL = "https://context.dev/signup"
NO_RESULTS = "Search returned no results"

# Publishers, preprint servers, proceedings, and scholarly indexes. With
# ``scholarly_sources_only`` the paper search tool keeps only results hosted here,
# which drops news and blog coverage of a paper. The filter runs on our side: the
# API leaves its indexed research papers out of any request that sets
# ``includeDomains``. Subdomains match (``nature.com`` covers ``www.nature.com``).
SCHOLARLY_DOMAINS: tuple[str, ...] = (
    # Preprints and open repositories
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "chemrxiv.org",
    "ssrn.com",
    "osf.io",
    "zenodo.org",
    "hal.science",
    # Conference proceedings and review platforms
    "openreview.net",
    "aclanthology.org",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "jmlr.org",
    "ojs.aaai.org",
    "ijcai.org",
    "openaccess.thecvf.com",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    # Publishers and journals
    "link.springer.com",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "cell.com",
    "pnas.org",
    "plos.org",
    "onlinelibrary.wiley.com",
    "tandfonline.com",
    "frontiersin.org",
    "mdpi.com",
    "academic.oup.com",
    "cambridge.org",
    "journals.sagepub.com",
    "annualreviews.org",
    "iopscience.iop.org",
    "journals.aps.org",
    "pubs.acs.org",
    "royalsocietypublishing.org",
    "elifesciences.org",
    # Medicine
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "europepmc.org",
    "bmj.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
    # Identifiers and scholarly indexes
    "doi.org",
    "semanticscholar.org",
    "researchgate.net",
    "jstor.org",
)

# With scholarly_sources_only, ask for this many times max_results so filtering
# still leaves enough papers.
SCHOLARLY_OVERFETCH_FACTOR = 3

# Warn about a missing key once per tool type, not once per configured instance.
_missing_key_warned: set[str] = set()

_INVALID_XML_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")
_WHITESPACE = re.compile(r"\s+")
# Whitespace references that page-to-Markdown conversion leaves in passages
# (for example a trailing "&#x20;"). Other entities are kept verbatim, because a
# page may legitimately show text such as "&lt;" that must be quoted as written.
_WHITESPACE_ENTITY = re.compile(r"&(?:#x0*(?:20|a0)|#0*(?:32|160)|nbsp);", re.IGNORECASE)


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _new_http_client() -> httpx.AsyncClient:
    """Create the HTTP client one registered tool reuses for all of its calls.

    Environment proxies stay in effect, except for plain-http loopback
    endpoints (local test servers): those connect directly, so the key is never
    handed to a proxy in cleartext.
    """
    direct = {f"http://[{host}]" if ":" in host else f"http://{host}": None for host in _LOOPBACK_HOSTS}
    return httpx.AsyncClient(mounts=direct)


_DOMAIN_CHARACTERS = re.compile(r"[a-z0-9.-]+")


def _normalize_domains(value: list[str] | None) -> list[str] | None:
    """Reduce entries such as ``https://www.arxiv.org/abs`` or ``.edu`` to a bare host.

    Both tools then interpret a domain list the same way, whether it is sent to
    the API (web search) or applied to results (paper search).
    """
    if value is None:
        return None
    normalized: list[str] = []
    for entry in value:
        text = entry.strip().lower()
        host = urlparse(text if "://" in text else f"//{text}").hostname or ""
        host = host.removeprefix("www.").strip(".")
        if len(host) < 3 or not _DOMAIN_CHARACTERS.fullmatch(host):
            raise ValueError(f"invalid domain {entry!r}; use a host name such as 'arxiv.org'")
        if host not in normalized:
            normalized.append(host)
    return normalized


def _validate_base_url(value: str) -> str:
    """Require https so the API key never travels in cleartext (http only for loopback)."""
    parsed = urlparse(value)
    if not parsed.hostname or parsed.scheme not in {"http", "https"}:
        raise ValueError("api_base_url must be an absolute https URL")
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("api_base_url must use https (http is allowed only for localhost)")
    return value


# ── Shared configuration fields ──
ApiKeyField = Annotated[
    SecretStr | None,
    Field(default=None, description=f"Context.dev API key. Falls back to the {API_KEY_ENV_VAR} environment variable."),
]
ApiBaseUrlField = Annotated[
    str,
    AfterValidator(_validate_base_url),
    Field(default=CONTEXT_DEV_API_URL, description="Context.dev API base URL (https; http only for localhost)."),
]
MaxRetriesField = Annotated[
    int,
    Field(default=3, ge=1, le=10, description="Total attempts for transient failures (network, 408, 429, 5xx)."),
]
TimeoutField = Annotated[
    float,
    Field(default=30.0, ge=2.0, le=300.0, description="Per-request timeout in seconds."),
]
IncludeHighlightsField = Annotated[
    bool,
    Field(
        default=True,
        description=(
            "Return the passages of each result page that are relevant to the query. "
            "When false, each result carries only the search snippet."
        ),
    ),
]
IncludeDomainsField = Annotated[
    list[str] | None,
    AfterValidator(_normalize_domains),
    Field(default=None, max_length=100, description="Only return results from these domains."),
]
ExcludeDomainsField = Annotated[
    list[str] | None,
    AfterValidator(_normalize_domains),
    Field(default=None, max_length=100, description="Drop results from these domains."),
]
CountryField = Annotated[
    str | None,
    Field(
        default=None,
        pattern=r"^[A-Za-z]{2}$",
        description="ISO 3166-1 alpha-2 country code used to localize results (e.g. 'us', 'gb').",
    ),
]
FreshnessField = Annotated[
    Freshness | None,
    Field(default=None, description="Only return content published within this window."),
]
ZeroDataRetentionField = Annotated[
    bool,
    Field(
        default=False,
        description=(
            "Ask Context.dev not to retain the request or its results. "
            "Requires an organization with zero data retention enabled."
        ),
    ),
]


class ContextDevWebSearchToolConfig(FunctionBaseConfig, name="context_dev_web_search"):
    """
    Tool that retrieves relevant contexts from web search (using Context.dev) for the given question.
    Requires a CONTEXT_DEV_API_KEY environment variable or api_key config.
    """

    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of search results to return.")
    max_content_length: int = Field(
        default=4000,
        ge=100,
        le=20000,
        description="Max characters of page content per result (relevant passages, or the snippet).",
    )
    api_key: ApiKeyField
    api_base_url: ApiBaseUrlField
    max_retries: MaxRetriesField
    timeout: TimeoutField
    include_highlights: IncludeHighlightsField
    include_domains: IncludeDomainsField
    exclude_domains: ExcludeDomainsField
    country: CountryField
    freshness: FreshnessField
    zero_data_retention: ZeroDataRetentionField


class ContextDevPaperSearchToolConfig(FunctionBaseConfig, name="context_dev_paper_search"):
    """
    Tool that searches for research papers (using Context.dev) for the given query.
    Requires a CONTEXT_DEV_API_KEY environment variable or api_key config.
    """

    scholarly_sources_only: bool = Field(
        default=True,
        description=(
            "Keep only results hosted by publishers, preprint servers, proceedings, and scholarly "
            "indexes (SCHOLARLY_DOMAINS), dropping news and blog coverage of papers. Filtering runs "
            "after the search, so the API's research paper index stays included. If no result is "
            "on a scholarly host, the unfiltered results are returned. Ignored when include_domains "
            "is non-empty; include_domains and exclude_domains are also applied after the search."
        ),
    )

    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of papers to return.")
    max_content_length: int = Field(
        default=1500,
        ge=100,
        le=20000,
        description="Max characters of relevant text per paper (passages, or the snippet).",
    )
    api_key: ApiKeyField
    api_base_url: ApiBaseUrlField
    max_retries: MaxRetriesField
    timeout: TimeoutField
    include_highlights: IncludeHighlightsField
    include_domains: IncludeDomainsField
    exclude_domains: ExcludeDomainsField
    country: CountryField
    freshness: FreshnessField
    zero_data_retention: ZeroDataRetentionField


ContextDevToolConfig = ContextDevWebSearchToolConfig | ContextDevPaperSearchToolConfig


# ── Shared helpers ──
def _resolve_api_key(tool_config: ContextDevToolConfig) -> tuple[str | None, str]:
    """Return ``(key, where it came from)``: the config ``api_key``, else ``CONTEXT_DEV_API_KEY``."""
    if tool_config.api_key is not None:
        configured = tool_config.api_key.get_secret_value().strip()
        if configured:
            return configured, "the api_key in your workflow config"
    return os.environ.get(API_KEY_ENV_VAR, "").strip() or None, API_KEY_ENV_VAR


def _warn_missing_key(tool_type: str) -> None:
    if tool_type in _missing_key_warned:
        return
    logger.warning(
        "%s not found. The %s tool will be registered but will return an error when called. "
        "To enable: set %s in your environment, .env file, or specify api_key in your workflow config.",
        API_KEY_ENV_VAR,
        tool_type,
        API_KEY_ENV_VAR,
    )
    _missing_key_warned.add(tool_type)


def _missing_key_message(tool_label: str) -> str:
    return (
        f"Error: {tool_label} is unavailable because {API_KEY_ENV_VAR} is not set.\n"
        "To enable this tool:\n"
        f"1. Get an API key from {SIGNUP_URL}\n"
        f"2. Set {API_KEY_ENV_VAR} in your environment or in your .env file\n"
        "   (alternatively, specify api_key in your workflow config)\n"
        "3. Restart the application"
    )


def _error_message(exc: ContextDevAPIError, tool_label: str, key_source: str = API_KEY_ENV_VAR) -> str:
    """Translate an API failure into an ``Error:`` string for the agent.

    The ``Error:`` prefix marks the output as a status message, so citation
    verification never treats it as evidence.
    """
    suffix = f" (request_id: {exc.request_id})" if exc.request_id else ""
    status = exc.status_code
    if status == 401:
        return (
            f"Error: {tool_label} failed due to an invalid API key (401 Unauthorized).\n"
            f"Please check {key_source} and ensure it is valid.{suffix}\n"
        )
    if status == 402:
        return (
            f"Error: {tool_label} failed because the Context.dev account has no credits left "
            f"(402 USAGE_EXCEEDED). Add credits or upgrade the plan at https://context.dev.{suffix}\n"
        )
    if status == 403 and exc.error_code == "ZDR_NOT_ENABLED":
        return (
            f"Error: {tool_label} failed because zero data retention is not enabled for this "
            f"Context.dev organization (403 ZDR_NOT_ENABLED). Set `zero_data_retention: false` "
            f"or ask Context.dev to enable it.{suffix}\n"
        )
    if status == 403:
        return (
            f"Error: {tool_label} was refused by Context.dev (403 {exc.error_code or 'Forbidden'}): "
            f"{exc.message}{suffix}\n"
        )
    if status == 400:
        return (
            f"Error: {tool_label} request was rejected (400 {exc.error_code or 'Bad Request'}): {exc.message}{suffix}"
        )
    if not exc.retryable:
        return f"Error: {tool_label} failed - HTTP {status}: {exc.message}{suffix}"
    detail = exc.message if status is None else f"HTTP {status}: {exc.message}"
    return f"Error: {tool_label} failed after {exc.attempts} attempt(s) - {detail}{suffix}"


def _is_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def _xml_text(value: str) -> str:
    """Remove characters that XML 1.0 forbids."""
    return _INVALID_XML_CHARACTERS.sub("", value)


def _clean_text(value: str) -> str:
    """Turn whitespace entity artifacts into spaces and drop XML-invalid characters.

    Passages are Markdown converted from HTML and can end lines with ``&#x20;``.
    Rendering that as a space avoids double-escaped noise (``&amp;#x20;``) in the
    output. Never apply this to URLs.
    """
    return _xml_text(_WHITESPACE_ENTITY.sub(" ", value))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _result_body(result: SearchResult, limit: int) -> str:
    """Prefer query-relevant passages; fall back to the search snippet."""
    if result.highlights:
        body = "\n\n".join(_clean_text(passage).strip() for passage in result.highlights)
    else:
        body = _clean_text(result.description).strip()
    return _truncate(body, limit)


def _usable(results: list[SearchResult]) -> list[SearchResult]:
    """Keep only results with an absolute http(s) URL, which citations require."""
    return [result for result in results if _is_http_url(result.url)]


def _render_document(result: SearchResult, limit: int) -> str:
    """Render one result as an escaped ``<Document>`` block (Tavily-compatible)."""
    url = _xml_text(result.url)
    title = _clean_text(result.title).strip() or url
    return (
        f'<Document href="{html.escape(url, quote=True)}">\n'
        f"<title>\n{html.escape(title, quote=True)}\n</title>\n"
        f"{html.escape(_result_body(result, limit), quote=True)}\n</Document>"
    )


def _one_line(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _render_paper(index: int, result: SearchResult, limit: int) -> str:
    """Render one result in the numbered ``paper_search`` layout.

    Fields are flattened to single lines so the layout cannot be broken by
    newlines or blank lines inside provider text.
    """
    url = _one_line(_xml_text(result.url))
    # A literal "*" next to the bold markers would shift the citation title match.
    title = _one_line(_clean_text(result.title)).replace("*", "\u2217") or url
    source = urlparse(url).hostname or ""
    if source.startswith("www."):
        source = source[4:]
    snippet = _one_line(_result_body(result, limit))
    return f"{index}. **{title}**\n   - **Source**: {source}\n   - **Snippet**: {snippet}\n   - **Link**: {url}"


def _host_matches(url: str, domains: list[str] | tuple[str, ...]) -> bool:
    """Whether the URL's host is one of ``domains`` or a subdomain of one."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _is_scholarly(url: str) -> bool:
    return _host_matches(url, SCHOLARLY_DOMAINS)


def _search_kwargs(
    tool_config: ContextDevToolConfig,
    *,
    num_results: int | None = None,
    send_domain_filters: bool = True,
) -> dict:
    return {
        "num_results": num_results or tool_config.max_results,
        "include_highlights": tool_config.include_highlights,
        "max_content_length": tool_config.max_content_length,
        "include_domains": tool_config.include_domains if send_domain_filters else None,
        "exclude_domains": tool_config.exclude_domains if send_domain_filters else None,
        "country": tool_config.country,
        "freshness": tool_config.freshness,
        "zero_data_retention": tool_config.zero_data_retention,
    }


# ── Web search ──
@register_function(config_type=ContextDevWebSearchToolConfig)
async def context_dev_web_search(
    tool_config: ContextDevWebSearchToolConfig,
    builder: Builder,
) -> AsyncGenerator[FunctionInfo, None]:
    """Register the Context.dev web search tool with NAT.

    If no API key is available (``api_key`` config or ``CONTEXT_DEV_API_KEY``),
    a stub is registered that returns setup instructions instead of failing at
    workflow load time.

    Args:
        tool_config: Result count, content budget, retries, and search filters.
        builder: NAT builder handle (unused; accepted for interface parity).

    Yields:
        A ``FunctionInfo`` wrapping either the live search callable or the stub.
    """
    tool_label = "Context.dev web search"
    search_kwargs = _search_kwargs(tool_config)
    api_key, key_source = _resolve_api_key(tool_config)
    if api_key is None:
        _warn_missing_key("context_dev_web_search")

        async def _context_dev_web_search_stub(question: str) -> str:
            """Web search tool (unavailable - missing CONTEXT_DEV_API_KEY)."""
            return _missing_key_message(tool_label)

        yield FunctionInfo.from_fn(
            _context_dev_web_search_stub,
            description=_context_dev_web_search_stub.__doc__,
        )
        return

    async with _new_http_client() as http_client:
        client = ContextDevSearchClient(
            api_key,
            http_client=http_client,
            base_url=tool_config.api_base_url,
            timeout=tool_config.timeout,
        )

        async def _context_dev_web_search(question: str) -> str:
            """Search the web with Context.dev and return relevant sources for a question.

            General-purpose web search: pass a natural-language question and get
            back the most relevant pages with their URLs and the passages of each
            page that answer the question.

            Args:
                question (str): The question to answer. Truncated to 500 characters if longer.

            Returns:
                str: Relevant documents and their URLs, rendered as XML <Document> blocks.
            """
            if not question or not question.strip():
                return "Error: 'question' argument is required"
            try:
                results = await search_with_retries(
                    client, question, max_retries=tool_config.max_retries, **search_kwargs
                )
            except ContextDevAPIError as exc:
                logger.warning("Context.dev web search failed (status=%s)", exc.status_code)
                return _error_message(exc, tool_label, key_source)

            documents = [_render_document(result, tool_config.max_content_length) for result in _usable(results)]
            return "\n\n---\n\n".join(documents) if documents else NO_RESULTS

        yield FunctionInfo.from_fn(
            _context_dev_web_search,
            description=_context_dev_web_search.__doc__,
        )


# ── Paper search ──
@register_function(config_type=ContextDevPaperSearchToolConfig)
async def context_dev_paper_search(
    tool_config: ContextDevPaperSearchToolConfig,
    builder: Builder,
) -> AsyncGenerator[FunctionInfo, None]:
    """Register the Context.dev research paper search tool with NAT.

    Queries the Context.dev Search API, whose index includes the Context.dev
    Research Index of research papers. If no API key is available, a stub is
    registered that returns setup instructions.

    Args:
        tool_config: Result count, content budget, retries, and search filters.
        builder: NAT builder handle (unused; accepted for interface parity).

    Yields:
        A ``FunctionInfo`` wrapping either the live search callable or the stub.
    """
    tool_label = "Context.dev paper search"
    # Domain filters are applied to the results here, never sent to the API: the
    # API leaves its indexed research papers out of domain-filtered requests.
    allowed_domains = tool_config.include_domains or (SCHOLARLY_DOMAINS if tool_config.scholarly_sources_only else None)
    blocked_domains = tool_config.exclude_domains or None
    # Only the default scholarly list falls back to unfiltered results; an explicit
    # include_domains list is a hard requirement.
    fallback_when_empty = not tool_config.include_domains
    num_results = tool_config.max_results
    if allowed_domains:
        num_results = min(100, tool_config.max_results * SCHOLARLY_OVERFETCH_FACTOR)
    search_kwargs = _search_kwargs(tool_config, num_results=num_results, send_domain_filters=False)
    api_key, key_source = _resolve_api_key(tool_config)
    if api_key is None:
        _warn_missing_key("context_dev_paper_search")

        async def _context_dev_paper_search_stub(query: str) -> str:
            """Research paper search tool (unavailable - missing CONTEXT_DEV_API_KEY)."""
            return _missing_key_message(tool_label)

        yield FunctionInfo.from_fn(
            _context_dev_paper_search_stub,
            description=_context_dev_paper_search_stub.__doc__,
        )
        return

    async with _new_http_client() as http_client:
        client = ContextDevSearchClient(
            api_key,
            http_client=http_client,
            base_url=tool_config.api_base_url,
            timeout=tool_config.timeout,
        )

        async def _context_dev_paper_search(query: str) -> str:
            """Searches for research papers and scientific publications.

            Use for research queries that need authoritative, scholarly sources:
            scientific concepts, algorithms, methodologies, empirical studies, and
            specific papers described by title, finding, or a half-remembered
            description. Returns papers with links and the passages that match
            the query. Write plain natural-language queries: search operators
            such as quotes, site:, OR, or a leading - exclude the research paper
            index.

            Args:
                query (str): The search query string. Truncated to 500 characters if longer.

            Returns:
                str: Numbered list of papers with source, relevant passages, and link.
            """
            if not query or not query.strip():
                return "Error: 'query' argument is required"
            try:
                results = await search_with_retries(client, query, max_retries=tool_config.max_retries, **search_kwargs)
            except ContextDevAPIError as exc:
                logger.warning("Context.dev paper search failed (status=%s)", exc.status_code)
                return _error_message(exc, tool_label, key_source)

            usable = _usable(results)
            if blocked_domains:
                usable = [result for result in usable if not _host_matches(result.url, blocked_domains)]
            if allowed_domains:
                allowed = [result for result in usable if _host_matches(result.url, allowed_domains)]
                if allowed or not fallback_when_empty:
                    usable = allowed
                else:
                    # Better to return the papers' web coverage than to tell the
                    # researcher nothing exists.
                    logger.info("No results on scholarly hosts; returning unfiltered results")
            papers = [
                _render_paper(index, result, tool_config.max_content_length)
                for index, result in enumerate(usable[: tool_config.max_results], 1)
            ]
            return "\n\n".join(papers) if papers else NO_RESULTS

        yield FunctionInfo.from_fn(
            _context_dev_paper_search,
            description=_context_dev_paper_search.__doc__,
        )
