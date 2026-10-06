# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Deterministic citation and report post-processing for research reports.

This module provides:
- SourceRegistry: captures URLs/citation keys from tool call results
- verify_citations(): validates cited source identities against the registry
- sanitize_report(): normalizes final report display and URL hygiene
- Extensible parser registry for adding new source types

Usage:
    registry = SourceRegistry()
    # ... populate via SourceRegistryMiddleware or manually ...
    result = verify_citations(report_text, registry)
    clean_report = result.verified_report
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from html import escape
from html import unescape
from urllib.parse import parse_qs
from urllib.parse import unquote
from urllib.parse import urlparse
from urllib.parse import urlunparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceEntry:
    """A single source captured from a tool call result."""

    url: str | None = None
    title: str | None = None
    citation_key: str | None = None
    source_type: str = ""
    tool_name: str = ""


def source_entries_from_parent_context(parent_context: str) -> list[SourceEntry]:
    """Parse durable parent-report source metadata into verification entries."""
    try:
        payload = json.loads(parent_context)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        return []

    entries: list[SourceEntry] = []
    for source in payload["sources"]:
        if not isinstance(source, dict):
            continue
        url = source.get("url")
        citation_key = source.get("citation_key")
        url = url.strip() if isinstance(url, str) and url.strip() else None
        citation_key = citation_key.strip() if isinstance(citation_key, str) and citation_key.strip() else None
        if not (url or citation_key):
            continue
        title = source.get("title")
        entries.append(
            SourceEntry(
                url=url,
                citation_key=citation_key,
                title=title.strip() if isinstance(title, str) and title.strip() else None,
                source_type=str(source.get("source_type") or "parent_report"),
                tool_name=str(source.get("tool_name") or "parent_report"),
            )
        )
    return entries


@dataclass
class CitationVerificationResult:
    """Result of running verify_citations()."""

    verified_report: str
    removed_citations: list[dict] = field(default_factory=list)
    valid_citations: list[dict] = field(default_factory=list)


class CitationIntegrityError(RuntimeError):
    """Raised when a verified report would be published without any verified citations."""

    def __init__(self) -> None:
        super().__init__("citation_integrity_lost")


class EmptySourceRegistryReason(StrEnum):
    """Stable classifications for research completing without sources."""

    NO_SOURCES_SELECTED = "no_sources_selected"
    NO_SOURCE_RESULTS = "no_source_results"
    SOURCE_TOOLS_UNAVAILABLE = "source_tools_unavailable"

    @property
    def public_message(self) -> str:
        """Return safe remediation suitable for direct display to users."""
        if self is EmptySourceRegistryReason.NO_SOURCES_SELECTED:
            return "No data sources are selected. Select at least one data source and run the research again."
        if self is EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE:
            return (
                "The selected data source tools are currently unavailable. "
                "Check their configuration or select different data sources and run the research again."
            )
        return (
            "The selected data sources returned no results. "
            "Try rephrasing the question or selecting different data sources."
        )


def classify_empty_source_registry_reason(
    data_sources: list[str] | None,
    available_count: int,
    unavailable_tools: list[str],
) -> EmptySourceRegistryReason:
    """Classify an empty source registry without conflating selection and availability."""
    if data_sources == []:
        return EmptySourceRegistryReason.NO_SOURCES_SELECTED
    if available_count == 0 and unavailable_tools:
        return EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE
    return EmptySourceRegistryReason.NO_SOURCE_RESULTS


class EmptySourceRegistryError(Exception):
    """Raised when no sources were captured during research."""

    def __init__(
        self,
        agent_type: str = "research",
        unavailable_tools: list[str] | None = None,
        available_count: int = 0,
        reason: EmptySourceRegistryReason = EmptySourceRegistryReason.NO_SOURCE_RESULTS,
        generated_answer: str | None = None,
    ) -> None:
        """Build the empty-registry error with agent type and tool-availability context."""
        self.agent_type = agent_type
        self.unavailable_tools = unavailable_tools or []
        self.available_count = available_count
        self.reason = reason
        self.public_message = reason.public_message
        self.generated_answer = generated_answer
        super().__init__(
            f"Research failed: no sources were captured during {agent_type}. "
            "All tool calls may have failed or returned no results."
        )

    @property
    def public_response(self) -> str:
        """Return the generated answer, when present, with remediation."""
        if self.generated_answer and self.generated_answer.strip():
            return f"{self.generated_answer.rstrip()}\n\n{self.public_message}"
        return self.public_message


_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "source",
    }
)


# Match permissive HTTP(S) candidates, then remove only punctuation and closing
# delimiters that are not part of the URL. In particular, a trailing ``)`` is
# valid when it balances an earlier ``(``, as in Wikipedia article URLs.
_HTTP_URL_CANDIDATE_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_HTTP_MARKDOWN_LINK_CANDIDATE_RE = re.compile(
    r"^https?://[^\s<>\"']+?\]\((?P<target>https?://[^\s<>\"']+)\)[.,;]*$",
    re.IGNORECASE,
)
_URL_TRAILING_PUNCTUATION = ".,;"
_URL_CLOSING_DELIMITERS = {")": "(", "]": "[", "}": "{"}


def clean_extracted_url(candidate: str) -> str:
    """Remove prose wrappers from a URL without corrupting balanced delimiters."""
    # The generic matcher starts after a Markdown link's opening bracket, so a
    # URL label and target are captured as one candidate. Select the target only
    # when both URLs and the closing Markdown delimiter are present; otherwise
    # ``](`` may be literal URL content and must remain intact.
    if markdown_link := _HTTP_MARKDOWN_LINK_CANDIDATE_RE.fullmatch(candidate):
        candidate = markdown_link.group("target")
    cleaned = unescape(candidate).strip().rstrip(_URL_TRAILING_PUNCTUATION)
    while cleaned and (opener := _URL_CLOSING_DELIMITERS.get(cleaned[-1])):
        closer = cleaned[-1]
        if cleaned.count(closer) <= cleaned.count(opener):
            break
        cleaned = cleaned[:-1].rstrip(_URL_TRAILING_PUNCTUATION)
    return cleaned


def extract_http_urls(text: str) -> list[str]:
    """Extract unique HTTP(S) URLs while preserving balanced URL delimiters."""
    if not text:
        return []
    urls = (clean_extracted_url(match.group(0)) for match in _HTTP_URL_CANDIDATE_RE.finditer(str(text)))
    return list(dict.fromkeys(url for url in urls if url))


def _first_http_url(text: str) -> str | None:
    """Return the first cleaned HTTP(S) URL in text, if present."""
    urls = extract_http_urls(text)
    return urls[0] if urls else None


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison.

    Lowercases scheme/host, strips trailing slash, removes fragments
    and common tracking parameters.
    """
    url = unescape(url).strip()
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = unquote(parsed.path).rstrip("/") or "/"
    # Remove tracking params
    qs = parse_qs(parsed.query, keep_blank_values=True)
    filtered_qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
    query_str = "&".join(f"{k}={v[0]}" for k, v in sorted(filtered_qs.items()) if v)
    return urlunparse((scheme, netloc, path, "", query_str, ""))


# ---------------------------------------------------------------------------
# Knowledge-layer fuzzy matching helpers
# ---------------------------------------------------------------------------

_PAGE_RE = re.compile(r"(?:p\.?|page)\s*(\d+)", re.IGNORECASE)


def _parse_citation_key(key: str) -> tuple[str, int | None]:
    """Extract (filename, page_number) from a citation key.

    Handles: "report.pdf, p.15", "report.pdf, page 15", "report.pdf"
    """
    page_match = _PAGE_RE.search(key)
    page = int(page_match.group(1)) if page_match else None
    # Filename is everything before the page reference (or the whole key)
    if page_match:
        filename = key[: page_match.start()].rstrip(", ").strip()
    else:
        filename = key.strip()
    return filename, page


@dataclass
class _ParsedURL:
    """Pre-parsed URL components cached at registration time."""

    host: str
    path: str
    path_segments: list[str]
    query: dict[str, list[str]]
    entry: SourceEntry


# ---------------------------------------------------------------------------
# SourceRegistry
# ---------------------------------------------------------------------------


class SourceRegistry:
    """Registry of sources captured from tool call results.

    Not thread-safe — this is intentional.  All access happens on a single
    asyncio event loop (cooperative concurrency), and Python's GIL already
    protects the underlying dict/list/set operations from corruption.
    The module-level ``_session_registries`` dict *is* lock-protected because
    it is accessed across event loops when creating/looking up sessions.
    """

    def __init__(self) -> None:
        """Initialize empty URL, parsed-URL, and citation-key indexes."""
        self._urls: dict[str, SourceEntry] = {}
        self._parsed_urls: dict[str, _ParsedURL] = {}
        self._citation_keys: list[SourceEntry] = []
        self._citation_key_files: set[str] = set()
        self._all: list[SourceEntry] = []

    def add(self, entry: SourceEntry) -> None:
        """Register a source entry. One entry per logical URL (dedup by normalized form).

        Raw URL = entry.url (exactly as the tool returned it); that is what we
        retain in the report. Normalized URL is used only as a key for dedup
        and matching. Both raw and normalized are stored as keys to the same
        entry so we never have duplicate entries and lookups find the tool URL.
        """
        added = False
        if entry.url:
            raw = entry.url
            normalized = _normalize_url(raw)
            if normalized not in self._urls:
                self._urls[normalized] = entry
                parsed = urlparse(normalized)
                self._parsed_urls[normalized] = _ParsedURL(
                    host=parsed.netloc,
                    path=parsed.path,
                    path_segments=[s for s in parsed.path.split("/") if s],
                    query=parse_qs(parsed.query, keep_blank_values=True),
                    entry=entry,
                )
                added = True
            if raw != normalized:
                self._urls[raw] = self._urls[normalized]
        if entry.citation_key:
            filename, _ = _parse_citation_key(entry.citation_key)
            key_lower = filename.lower()
            if key_lower not in self._citation_key_files:
                self._citation_key_files.add(key_lower)
                self._citation_keys.append(entry)
                if not added:
                    added = True
        if added:
            self._all.append(entry)

    def has_url(self, url: str) -> bool:
        """Check if a URL (after normalization) is in the registry."""
        return _normalize_url(url) in self._urls

    @staticmethod
    def _pick_unique(candidates: list[SourceEntry], strategy: str, url: str) -> str | None:
        """Return the registry URL when exactly one candidate matches.

        The source section can only show one URL per citation. If multiple
        registry URLs match (e.g. same path, different query), we cannot know
        which one the author meant, so we reject.
        """
        if len(candidates) == 1:
            logger.debug("[CitationVerify] Unique %s URL match", strategy)
            return candidates[0].url
        if len(candidates) > 1:
            logger.debug(
                "[CitationVerify] Ambiguous %s URL match — %d candidates, rejecting",
                strategy,
                len(candidates),
            )
        return None

    def resolve_url(self, url: str) -> str | None:
        """Return the registry URL (full, as returned by the tool) when the report URL matches.

        Matching strategy (first unambiguous match wins):
        1. Exact match — raw or normalized
        2. Truncation — report URL is a prefix of exactly one registry URL (raw)
        3. Prefix — report normalized is prefix of registry normalized
        4. Child-path — report path is a subpath of exactly one registry URL
        5. Query-subset — same host+path, report params subset of one registry URL

        Always returns the tool's URL (with query params etc.). If multiple
        registry URLs match, we reject (ambiguous).
        """
        # 1. Exact match — raw or normalized; retain the tool's URL
        if url in self._urls:
            return self._urls[url].url
        normalized = _normalize_url(url)
        if normalized in self._urls:
            return self._urls[normalized].url

        # 2. Truncation — report URL is a prefix of exactly one registry URL (raw).
        #    Normalized match fails when the report is cut mid-query (param order differs).
        truncation_entries = [e for e in self._urls.values() if e.url and e.url.startswith(url)]
        result = self._pick_unique(list({e.url: e for e in truncation_entries}.values()), "truncation", url)
        if result:
            return result

        # 3. Prefix match — report normalized is prefix of registry normalized
        #    Deduplicate by url to avoid raw+normalized keys for the same entry
        #    being counted as ambiguous.
        prefix_entries = [e for n, e in self._urls.items() if n.startswith(normalized)]
        result = self._pick_unique(
            list({e.url: e for e in prefix_entries}.values()),
            "prefix",
            url,
        )
        if result:
            return result

        parsed = urlparse(normalized)
        host, path = parsed.netloc, parsed.path
        same_host = [p for p in self._parsed_urls.values() if p.host == host]

        # 4. Child-path match — report path extends a registry path (subpage)
        #    Use rstrip("/") + "/" to enforce segment boundaries (prevents
        #    /us/benefits matching /us/benefitsOther).
        result = self._pick_unique(
            [
                p.entry
                for p in same_host
                if len(p.path_segments) >= 2 and path != p.path and path.startswith(p.path.rstrip("/") + "/")
            ],
            "child-path",
            url,
        )
        if result:
            return result

        # 5. Query-subset match — same host+path, report params are a subset of registry params
        report_qs = parse_qs(parsed.query, keep_blank_values=True)
        if report_qs:
            result = self._pick_unique(
                [
                    p.entry
                    for p in same_host
                    if p.path == path and p.query and all(p.query.get(k) == v for k, v in report_qs.items())
                ],
                "query-subset",
                url,
            )
            if result:
                return result

        return None

    def has_citation_key(self, key: str) -> bool:
        """Lenient match of a citation key against registry entries.

        Matches if filename (case-insensitive) matches ANY registry entry.
        Page numbers are not required to match — the LLM may cite a different
        page than what the knowledge layer returned, and that's acceptable
        since the document itself was verified as a real source.
        """
        target_file, _ = _parse_citation_key(key)
        target_lower = target_file.lower()
        for entry in self._citation_keys:
            entry_file, _ = _parse_citation_key(entry.citation_key)
            if entry_file.lower() == target_lower:
                return True
        return False

    def all_sources(self) -> list[SourceEntry]:
        """Return all registered sources."""
        return list(self._all)

    def clear(self) -> None:
        """Reset the registry."""
        self._urls.clear()
        self._parsed_urls.clear()
        self._citation_keys.clear()
        self._citation_key_files.clear()
        self._all.clear()


# ---------------------------------------------------------------------------
# Session-scoped registry (ContextVar)
# ---------------------------------------------------------------------------

_session_source_registry: contextvars.ContextVar[SourceRegistry | None] = contextvars.ContextVar(
    "_session_source_registry", default=None
)

_MAX_SESSION_REGISTRIES = 1000
_session_registries: OrderedDict[str, SourceRegistry] = OrderedDict()
_session_registries_lock = threading.Lock()


def get_or_create_session_registry(session_id: str | None) -> SourceRegistry:
    """Get or create a session-scoped SourceRegistry (LRU, max 1000 sessions).

    When session_id is None (e.g. CLI or batch modes with no conversation context),
    a fresh isolated SourceRegistry is returned on every call to prevent anonymous
    sessions from sharing state and leaking citations across concurrent requests.
    """
    if session_id is None:
        return SourceRegistry()
    with _session_registries_lock:
        if session_id in _session_registries:
            _session_registries.move_to_end(session_id)
            return _session_registries[session_id]
        registry = SourceRegistry()
        _session_registries[session_id] = registry
        while len(_session_registries) > _MAX_SESSION_REGISTRIES:
            _session_registries.popitem(last=False)
        return registry


def set_session_registry(registry: SourceRegistry | None) -> contextvars.Token:
    """Set the session-scoped SourceRegistry for the current async context."""
    return _session_source_registry.set(registry)


def reset_session_registry(token: contextvars.Token) -> None:
    """Restore the session-scoped SourceRegistry to its previous value."""
    _session_source_registry.reset(token)


def get_session_registry() -> SourceRegistry | None:
    """Get the session-scoped SourceRegistry for the current async context."""
    return _session_source_registry.get()


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

SourceParser = Callable[[str, str], list[SourceEntry]]

_PARSER_REGISTRY: list[tuple[Callable[[str], bool], SourceParser]] = []


def register_source_parser(
    match_fn: Callable[[str], bool],
    parser_fn: SourceParser,
) -> None:
    """Register a parser for a tool name pattern.

    Args:
        match_fn: Predicate on lowercase tool name.
        parser_fn: (content, tool_name) -> list[SourceEntry]
    """
    _PARSER_REGISTRY.append((match_fn, parser_fn))


def extract_sources_from_tool_result(
    tool_name: str,
    content: str,
    source_id: str | None = None,
    result_status: str | None = None,
) -> list[SourceEntry]:
    """Extract sources from a tool's output.

    Strategy:
    1. If a registered parser matches the tool name, use it (for special
       formats like knowledge layer citation keys).
    2. Otherwise, fall back to the generic URL extractor which finds all
       URLs in any tool output regardless of format.
    3. If neither produces entries, register the tool result itself as a
       non-URL citation source.

    This means new sources (Bing, Perplexity, etc.) work automatically
    without any parser registration — as long as their output contains URLs.

    A typed ``error`` result is never citable, even when its diagnostic body
    contains URLs. The non-URL fallback is permissive on purpose: callers (the shallow and
    deep researchers) are responsible for deciding which tool calls are
    eligible to contribute sources, typically by limiting capture to the
    agent's loaded tool set. The optional ``source_id`` is stored on the
    returned entries when callers have resolved this tool to a configured
    data source via
    :func:`deep_researcher_agent.common.data_source_registry.get_source_id_for_tool`,
    but it does not gate the fallback.
    """
    # Reject provider/status payloads before running source parsers. Error
    # responses can contain documentation or request URLs, but those URLs are
    # diagnostics rather than evidence and must not satisfy citation checks.
    if result_status == "error" or is_non_citable_status_output(content):
        return []

    name_lower = tool_name.lower()
    for match_fn, parser_fn in _PARSER_REGISTRY:
        if match_fn(name_lower):
            try:
                return parser_fn(content, tool_name)
            except Exception:
                logger.warning("Parser failed for tool %s, falling back to generic", tool_name, exc_info=True)
                break
    # Generic fallback: extract all URLs from content
    entries = _parse_generic_urls(content, tool_name)
    if entries:
        return entries

    # Non-URL fallback: register the tool result itself as a source whenever
    # the tool produced non-empty output. The caller has already decided
    # this tool is eligible to contribute sources (typically by limiting
    # capture to the agent's loaded tool set).
    if content.strip():
        return [SourceEntry(citation_key=tool_name, source_type="tool_result", tool_name=tool_name)]

    return []


def is_non_citable_status_output(content: str) -> bool:
    """Return whether content is a tool status/error message, not evidence."""
    normalized = re.sub(r"\s+", " ", content.strip()).rstrip(".").lower()
    if not normalized:
        return False

    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, (dict, list)) and not payload:
        return True
    if isinstance(payload, dict):
        error = payload.get("error")
        if error not in (None, False, "", [], {}):
            return True
        for key in ("status", "status_code", "statusCode", "http_status"):
            status = payload.get(key)
            if isinstance(status, str) and status.strip().lower() == "error":
                return True
            try:
                status_code = int(status)
            except (TypeError, ValueError):
                continue
            if 400 <= status_code < 600:
                return True

    if re.match(r"^error(?:\s*[:=]|\s+[45]\d{2}\b)", normalized):
        return True
    if re.match(
        r"^(?:search|request|tool)\b.{0,80}\b(?:error|failed|failure)\b.{0,40}\b(?:status\s*)?[45]\d{2}\b",
        normalized,
    ):
        return True
    if normalized.startswith(("{", "[")) and re.search(
        r"[\"']?(?:status|status[_-]?code|http[_-]?status)[\"']?\s*[:=]\s*[\"']?[45]\d{2}\b",
        normalized,
    ):
        return True
    return normalized == "search returned no results" or normalized.endswith(" search returned no results")


# ---------------------------------------------------------------------------
# Built-in parsers
# ---------------------------------------------------------------------------

# Generic URL extraction uses the shared balanced-delimiter-aware helper above.
# Commas are valid URL path characters (RFC 3986 sub-delim), so extraction is
# permissive and only trailing sentence punctuation is removed.
# Patterns for extracting titles near URLs in common tool output formats
_TITLE_NEAR_URL_PATTERNS = [
    # Tavily: <title>\nSome Title\n</title>
    re.compile(r"<title>\s*\n?(.*?)\n?\s*</title>", re.DOTALL | re.IGNORECASE),
    # Paper search: N. **Title** (Year)
    re.compile(r"^\d+\.\s+\*\*(.+?)\*\*", re.MULTILINE),
    # Additional title patterns: --- Title ---
    re.compile(r"^---\s+(.+?)\s+---$", re.MULTILINE),
    # Key-value: Title: Some Title
    re.compile(r"^Title:\s*(.+)$", re.MULTILINE),
]


def _extract_title_for_url(content: str, url: str) -> str | None:
    """Try to extract a title associated with a URL from the surrounding content.

    Finds the title pattern **closest to** (and preceding) the URL within its
    text block.  This prevents a single block containing multiple search
    results from assigning the first result's title to every URL.
    """
    # Connector renderers escape URL attributes. Match that exact spelling in
    # the trusted structure while retaining the decoded URL as source identity.
    escaped_url = escape(url, quote=True)

    # Find the block of text containing this URL (split by --- or double newlines)
    blocks = re.split(r"\n\n---\n\n|\n\n\n", content)
    for block in blocks:
        block_url = escaped_url if escaped_url in block else url
        if block_url not in block:
            continue
        url_pos = block.index(block_url)
        best_title: str | None = None
        best_distance = float("inf")
        for pattern_index, pattern in enumerate(_TITLE_NEAR_URL_PATTERNS):
            for title_match in pattern.finditer(block):
                title = title_match.group(1).strip()
                if pattern_index == 0:
                    title = unescape(title)
                if not title or title == url:
                    continue
                # Prefer titles that appear before (and closest to) the URL
                distance = url_pos - title_match.end()
                if distance < 0:
                    # Title appears after the URL — use large penalty
                    distance = abs(distance) + 10000
                if distance < best_distance:
                    best_distance = distance
                    best_title = title
        if best_title:
            return best_title
    return None


def _parse_generic_urls(content: str, tool_name: str) -> list[SourceEntry]:
    """Extract all URLs from any tool output, regardless of format.

    This is the universal fallback. It finds every URL in the content
    and registers it. Works for Tavily XML, paper search markdown,
    plain text with links, or any future source format. Also attempts to
    extract titles from common patterns near each URL.
    """
    seen: set[str] = set()
    entries: list[SourceEntry] = []
    for url in extract_http_urls(content):
        normalized = _normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            title = _extract_title_for_url(content, url)
            entries.append(SourceEntry(url=url, title=title, source_type="generic", tool_name=tool_name))
    return entries


# Knowledge layer is the only source that needs a specific parser because
# it uses citation keys (e.g., "report.pdf, p.15") instead of URLs.
_KL_CITATION_RE = re.compile(r"^Citation:\s*(.+)$", re.MULTILINE)
_KL_SOURCE_RE = re.compile(r"^Source:\s*(.+)$", re.MULTILINE)


def _parse_knowledge_layer(content: str, tool_name: str) -> list[SourceEntry]:
    """Parse knowledge layer retrieval output.

    Extracts citation keys (filename + page) AND any URLs present.
    Falls back to generic URL extraction if no Citation: fields found.
    """
    entries: list[SourceEntry] = []
    citations = _KL_CITATION_RE.findall(content)
    sources = _KL_SOURCE_RE.findall(content)
    for i, citation_key in enumerate(citations):
        title = sources[i].strip() if i < len(sources) else None
        entries.append(
            SourceEntry(
                citation_key=citation_key.strip(), title=title, source_type="knowledge_layer", tool_name=tool_name
            )
        )
    if not entries:
        return _parse_generic_urls(content, tool_name)
    return entries


# Register knowledge layer as the only special-case parser.
# All other tools (Tavily, paper search, etc.) use the generic URL fallback.
register_source_parser(lambda name: "knowledge" in name, _parse_knowledge_layer)

# ---------------------------------------------------------------------------
# Citation parsing and source-section layout normalization
# ---------------------------------------------------------------------------

_REFERENCE_HEADING_LABEL_PATTERN = r"(?:Sources|References|Reference[^\S\n]+List)"
_REFERENCE_HEADING_PATTERN = (
    r"^[^\S\n]*(?:"
    rf"#{{1,3}}[^\S\n]+{_REFERENCE_HEADING_LABEL_PATTERN}[^\S\n]*:?"
    rf"|\*\*{_REFERENCE_HEADING_LABEL_PATTERN}:?\*\*"
    rf"|{_REFERENCE_HEADING_LABEL_PATTERN}[^\S\n]*:?"
    r")[^\S\n]*$"
)
_REFERENCE_ENTRY_START_PATTERN = r"(?:[-*][^\S\n]*)?(?:\[\d+\]|\[\^\d+\]:?|\d+[.)])[^\S\n]+"
_REFERENCE_SECTION_RE = re.compile(
    rf"{_REFERENCE_HEADING_PATTERN}(?=\n[^\S\n]*(?:\n[^\S\n]*)*{_REFERENCE_ENTRY_START_PATTERN})",
    re.MULTILINE | re.IGNORECASE,
)
_REFERENCE_HEADING_LINE_RE = re.compile(_REFERENCE_HEADING_PATTERN, re.IGNORECASE)

_CITATION_LINE_RE = re.compile(r"^\s*[-*]?\s*\[(\d+)\]\s*(.+)$", re.MULTILINE)
_ORDERED_REFERENCE_LINE_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.+)$", re.MULTILINE)
_COLLAPSED_SOURCE_BOUNDARY_RE = re.compile(r"\s+(?=\[(\d+)\]\s+)")
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_FOOTNOTE_REFERENCE_LINE_RE = re.compile(r"^\s*\[\^(\d+)\]:?\s*", re.MULTILINE)
_FOOTNOTE_INLINE_CITATION_RE = re.compile(r"\[\^(\d+)\]")
_SOURCE_LOCATION_CITATION_RE = re.compile(r"\[(\d+)\s*†[^\]]+\]")

# Knowledge-layer citation pattern: "filename.ext" optionally followed by ", p.N" or ", page N"
_KL_CITATION_PATTERN_RE = re.compile(r"^(.+\.\w{2,5})(?:,\s*(?:p\.?|page)\s*\d+)?$", re.IGNORECASE)


def _is_knowledge_citation(ref_text: str, registry: SourceRegistry | None = None) -> tuple[bool, str | None]:
    """Check if reference text looks like a knowledge-layer citation.

    Uses a lenient matching strategy:
    1. Try exact pattern match (filename.ext, p.N) after stripping markdown
    2. If a registry is provided, check if ANY registered citation key's
       filename appears anywhere in the reference text (very lenient —
       handles all formatting variations the LLM might produce)

    Returns (is_kl, citation_key_or_none).
    """
    # Strip trailing "(Internal)" or similar parenthetical
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", ref_text).strip()
    # Strip markdown bold/italic markers only (*, **) — preserve underscores in filenames
    cleaned = re.sub(r"\*+", "", cleaned).strip()
    # Remove leading "Title - " or "Title: " prefix by taking last segment
    # if it contains a filename pattern
    for segment in [cleaned, cleaned.split(" - ")[-1].strip(), cleaned.split(": ")[-1].strip()]:
        if _KL_CITATION_PATTERN_RE.match(segment):
            return True, segment

    # Lenient fallback: check if any registered knowledge-layer filename
    # appears in the reference text (handles arbitrary LLM formatting)
    if registry is not None:
        ref_lower = cleaned.lower()
        for entry in registry._citation_keys:
            entry_file, _ = _parse_citation_key(entry.citation_key)
            if entry_file.lower() in ref_lower:
                return True, entry.citation_key

    return False, None


def _format_registry_reference(num: int, entry: SourceEntry) -> str | None:
    """Render a registered source as a source-section line."""
    title = entry.title or entry.tool_name or entry.source_type or "Source"
    if entry.url:
        return f"[{num}] {title}: {entry.url}"
    if entry.citation_key:
        return f"[{num}] {entry.citation_key}"
    return None


def _normalize_reference_title(text: str) -> str:
    """Normalize a source-line title for exact-match backfill comparison.

    Strips any trailing URL, trailing parenthetical (e.g. ``(Internal)``),
    markdown emphasis, and a trailing ``:`` separator, then lowercases and
    collapses whitespace so a writer line and its registry entry compare equal.
    """
    cleaned = _HTTP_URL_CANDIDATE_RE.sub("", text)
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = cleaned.strip().rstrip(":").strip()
    return re.sub(r"\s+", " ", cleaned).lower()


def _build_reference_title_index(
    reference_sources: Sequence[SourceEntry] | None,
    registry: SourceRegistry,
) -> dict[str, tuple[str | None, str | None]]:
    """Index writer-facing source titles to their registry-validated target.

    Used to repair URL-less ``[N] Title`` lines (the writer dropped the ``: url``
    suffix) from the same captured-source list the writer was shown. Each title
    is validated against the registry exactly as an inline target would be, and
    titles shared by more than one distinct source are dropped so an ambiguous
    title is never auto-resolved.

    Args:
        reference_sources: Writer-facing source list, or None.
        registry: SourceRegistry the targets must resolve against.

    Returns:
        Mapping of normalized title -> ``(canonical_url, citation_key)`` with
        exactly one tuple value set.
    """
    if not reference_sources:
        return {}
    index: dict[str, tuple[str | None, str | None]] = {}
    ambiguous: set[str] = set()
    for entry in reference_sources:
        if not entry.title:
            continue
        key = _normalize_reference_title(entry.title)
        if not key:
            continue
        target: tuple[str | None, str | None] | None = None
        if entry.url:
            canonical = registry.resolve_url(entry.url)
            if canonical:
                target = (canonical, None)
        elif entry.citation_key and registry.has_citation_key(entry.citation_key):
            target = (None, entry.citation_key)
        if target is None:
            continue
        if key in index and index[key] != target:
            ambiguous.add(key)
            continue
        index[key] = target
    for key in ambiguous:
        index.pop(key, None)
    return index


def _backfill_reference_target(
    ref_text: str,
    reference_index: dict[str, tuple[str | None, str | None]],
) -> tuple[str | None, str | None] | None:
    """Resolve a URL-less source line's title to a registry-backed target.

    Exact normalized-title match only. Aggregate-style labels (multiple sources
    joined with ``;``) are never matched, so they keep stripping as before.

    Returns:
        ``(canonical_url, citation_key)`` when the title uniquely maps to a
        registry source, otherwise None.
    """
    if ";" in ref_text:
        return None
    return reference_index.get(_normalize_reference_title(ref_text))


def _normalize_citation_syntax(report_text: str) -> str:
    """Normalize citation bracket variants before verification/sanitization."""
    report_text = report_text.replace("【", "[").replace("】", "]")
    report_text = _SOURCE_LOCATION_CITATION_RE.sub(r"[\1]", report_text)
    report_text = _FOOTNOTE_REFERENCE_LINE_RE.sub(r"[\1] ", report_text)
    return _FOOTNOTE_INLINE_CITATION_RE.sub(r"[\1]", report_text)


def _normalize_ordered_reference_lines(ref_section: str) -> str:
    """Convert ordered-list source lines to canonical [N] form."""
    return _ORDERED_REFERENCE_LINE_RE.sub(r"\1[\2] \3", ref_section)


def _reference_segment_has_target(segment: str) -> bool:
    """Return true when a source segment already contains a verifiable target."""
    line_match = _CITATION_LINE_RE.match(segment)
    if line_match is None:
        return False

    ref_text = line_match.group(2).strip()
    if _first_http_url(ref_text):
        return True

    cleaned = re.sub(r"\*+", "", ref_text).strip()
    return bool(_KL_CITATION_PATTERN_RE.match(cleaned))


def _split_collapsed_source_line(line: str) -> str:
    """Split only truly collapsed source entries on one line.

    Avoid splitting bracketed numbers that are part of a title, such as
    ``[1] Semiconductor outlook [2024] update: https://...``.
    """
    first_line_match = _CITATION_LINE_RE.match(line)
    if first_line_match is None:
        return line

    current_num = int(first_line_match.group(1))
    segment_start = 0
    segments: list[str] = []
    for boundary_match in _COLLAPSED_SOURCE_BOUNDARY_RE.finditer(line):
        next_num = int(boundary_match.group(1))
        current_segment = line[segment_start : boundary_match.start()]
        if next_num <= current_num or next_num >= 1000 or not _reference_segment_has_target(current_segment):
            continue

        segments.append(current_segment.rstrip())
        segment_start = boundary_match.end()
        current_num = next_num

    if not segments:
        return line

    segments.append(line[segment_start:].lstrip())
    return "\n".join(segments)


def _normalize_source_section_layout(ref_section: str) -> str:
    """Normalize source-section presentation before parsing or final cleanup.

    This is report hygiene, not source-identity verification. ``verify_citations``
    calls it only so common writer variants are parseable; ``sanitize_report``
    owns the final display normalization.
    """
    ref_section = _normalize_ordered_reference_lines(ref_section)
    lines = ref_section.split("\n")
    if lines and _REFERENCE_HEADING_LINE_RE.match(lines[0]):
        lines[0] = "## Sources"
        ref_section = "\n".join(lines)
    return "\n".join(_split_collapsed_source_line(line) for line in ref_section.split("\n"))


def extract_source_entries_from_report(report_text: str) -> list[SourceEntry]:
    """Extract verifiable source identities from a report's source section.

    A report source can anchor verification only when it has a URL or a
    recognizable knowledge-layer citation key.
    """
    report_text = _normalize_citation_syntax(report_text)
    ref_match = _REFERENCE_SECTION_RE.search(report_text)
    if ref_match is None:
        return []

    entries: list[SourceEntry] = []
    ref_section = _normalize_source_section_layout(report_text[ref_match.start() :])
    for line_match in _CITATION_LINE_RE.finditer(ref_section):
        ref_text = line_match.group(2).strip()
        url = _first_http_url(ref_text)
        if url:
            entries.append(
                SourceEntry(
                    url=url,
                    source_type="parent_report",
                    tool_name="parent_report",
                )
            )
            continue

        is_knowledge, citation_key = _is_knowledge_citation(ref_text)
        if is_knowledge and citation_key:
            entries.append(
                SourceEntry(
                    citation_key=citation_key,
                    source_type="parent_report",
                    tool_name="parent_report",
                )
            )
    return entries


def report_has_citations(report_text: str) -> bool:
    """Return whether a report claims to contain numbered citations."""
    report_text = _normalize_citation_syntax(report_text)
    return bool(_REFERENCE_SECTION_RE.search(report_text) or _INLINE_CITATION_RE.search(report_text))


def _inline_citation_numbers(text: str) -> set[int]:
    """Return numeric inline citation labels present in text."""
    return {int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(text)}


def _strip_inline_citations_not_in(text: str, valid_numbers: set[int]) -> str:
    """Remove inline citations whose labels are not in valid_numbers."""
    stripped = _INLINE_CITATION_RE.sub(
        lambda match: match.group(0) if int(match.group(1)) in valid_numbers else "",
        text,
    )
    return re.sub(r"[ \t]+(?=[,.;:!?])", "", stripped)


def _renumber_citations(body: str, ref_section: str) -> tuple[str, str, dict[int, int]]:
    """Renumber [N] citations sequentially, closing any gaps.

    Scans the source section for citation numbers, builds a mapping
    from old to new sequential numbers, and applies it to both body and
    source lines via collision-safe placeholders.

    Returns:
        (body, ref_section, renumber_map) where renumber_map maps every
        old citation number to its new sequential number.
    """
    remaining = sorted(int(m.group(1)) for m in _CITATION_LINE_RE.finditer(ref_section))
    renumber_map: dict[int, int] = {old: new for new, old in enumerate(remaining, 1)}

    # Nothing to do if already sequential
    if all(old == new for old, new in renumber_map.items()):
        return body, ref_section, renumber_map

    # Apply renumbering via placeholders (descending order avoids [1] matching inside [10])
    for old_num in sorted(renumber_map, reverse=True):
        new_num = renumber_map[old_num]
        if old_num != new_num:
            placeholder = f"__CITE_{new_num}__"
            body = body.replace(f"[{old_num}]", placeholder)
            ref_section = ref_section.replace(f"[{old_num}]", placeholder)

    for new_num in sorted(renumber_map.values()):
        placeholder = f"__CITE_{new_num}__"
        body = body.replace(placeholder, f"[{new_num}]")
        ref_section = ref_section.replace(placeholder, f"[{new_num}]")

    return body, ref_section, renumber_map


def verify_citations(
    report_text: str,
    registry: SourceRegistry,
    *,
    reference_sources: Sequence[SourceEntry] | None = None,
) -> CitationVerificationResult:
    """Verify cited source identities against the captured source registry.

    This function decides whether each cited source line maps to a real captured
    URL or citation key. It may normalize source-section layout enough to parse
    common writer variants, but final Markdown hygiene belongs to
    ``sanitize_report``.

    Algorithm:
    1. Find the source section
    2. Parse each [N] source line
    3. Validate URL or citation_key against registry
    4. Remove invalid source lines and orphaned inline citations

    Renumbering is NOT done here — it is deferred to sanitize_report()
    which always runs after this function and handles it in a single pass.

    Args:
        report_text: The full report text with citations.
        registry: SourceRegistry populated from tool call results.
        reference_sources: Optional writer-facing source list, in the same
            numbering order the writer saw. Used only to synthesize a missing
            source section.

    Returns:
        CitationVerificationResult with cleaned report and audit trail.
    """
    # Normalize citation syntax variants before validation.
    report_text = _normalize_citation_syntax(report_text)

    # Early exit: nothing to validate against
    all_sources = registry.all_sources()
    if not all_sources:
        logger.debug("[CitationVerify] Skipping — registry is empty (no tool calls captured)")
        return CitationVerificationResult(verified_report=report_text)

    logger.info(
        "[CitationVerify] Starting verification against %d registered source(s)",
        len(all_sources),
    )
    logger.debug(
        "[CitationVerify] Registry contains %d URL source(s)",
        sum(source.url is not None for source in all_sources),
    )

    # Find source section
    ref_match = _REFERENCE_SECTION_RE.search(report_text)
    if not ref_match:
        if not _INLINE_CITATION_RE.search(report_text):
            logger.warning("[CitationVerify] No source section found in report; skipping")
            return CitationVerificationResult(verified_report=report_text)

        if reference_sources is None:
            logger.warning(
                "[CitationVerify] No source section found; cannot safely synthesize sources "
                "without the writer-facing source list"
            )
            return CitationVerificationResult(verified_report=report_text)

        writer_sources = list(reference_sources)
        cited_numbers = sorted(_inline_citation_numbers(report_text))
        reference_lines = [
            line
            for i in cited_numbers
            if 1 <= i <= len(writer_sources)
            if (line := _format_registry_reference(i, writer_sources[i - 1]))
        ]
        if not reference_lines:
            logger.warning("[CitationVerify] No source section found and no renderable writer-facing sources")
            return CitationVerificationResult(verified_report=_strip_inline_citations_not_in(report_text, set()))

        logger.warning(
            "[CitationVerify] No source section found; appending %d inline-cited registered source(s)",
            len(reference_lines),
        )
        report_text = report_text.rstrip() + "\n\n## Sources\n" + "\n".join(reference_lines)
        ref_match = _REFERENCE_SECTION_RE.search(report_text)
        if ref_match is None:
            return CitationVerificationResult(verified_report=report_text)

    ref_start = ref_match.start()
    body = report_text[:ref_start]
    original_ref_section = report_text[ref_start:]
    ref_section = _normalize_source_section_layout(original_ref_section)

    # Parse citation lines in the source section
    valid_citations: list[dict] = []
    removed_citations: list[dict] = []
    url_replacements: dict[str, str] = {}  # garbled_url -> canonical_url
    line_replacements: dict[str, str] = {}  # url-less line -> backfilled canonical line

    # Exact-title index over the writer-facing source list, so a "[N] Title"
    # line whose ": url" suffix the writer dropped can be repaired from the same
    # captured registry instead of stripped (recall fix, precision unchanged).
    reference_index = _build_reference_title_index(reference_sources, registry)

    for line_match in _CITATION_LINE_RE.finditer(ref_section):
        num = int(line_match.group(1))
        ref_text = line_match.group(2).strip()
        full_line = line_match.group(0)

        # Try URL match first
        url = _first_http_url(ref_text)
        if url:
            canonical = registry.resolve_url(url)
            if canonical:
                if canonical != url:
                    logger.debug("[CitationVerify]   [%d] VALID  — repaired URL", num)
                    url_replacements[url] = canonical
                else:
                    logger.debug("[CitationVerify]   [%d] VALID  — URL", num)
                valid_citations.append({"number": num, "url": canonical, "citation_key": None, "line": full_line})
            else:
                logger.info("[CitationVerify]   [%d] REMOVE — url_not_in_registry", num)
                removed_citations.append({"number": num, "line": full_line, "reason": "url_not_in_registry"})
            continue

        # Try knowledge-layer citation key (lenient — passes registry for fuzzy filename match)
        is_kl, citation_key = _is_knowledge_citation(ref_text, registry)
        if is_kl and citation_key:
            if registry.has_citation_key(citation_key):
                logger.debug("[CitationVerify]   [%d] VALID  — citation key", num)
                valid_citations.append({"number": num, "url": None, "citation_key": citation_key, "line": full_line})
            else:
                logger.debug("[CitationVerify]   [%d] REMOVE — citation_key_not_in_registry", num)
                removed_citations.append({"number": num, "line": full_line, "reason": "citation_key_not_in_registry"})
            continue

        # Backfill: the writer emitted "[N] Title" but dropped the verified URL.
        # Recover the target from the writer-facing source list (exact-title,
        # unique match) and rewrite the line to canonical "[N] Title: url" form.
        backfilled = _backfill_reference_target(ref_text, reference_index)
        if backfilled:
            canonical_url, citation_key = backfilled
            if canonical_url:
                rewritten = f"[{num}] {ref_text}: {canonical_url}"
                line_replacements[full_line] = rewritten
                logger.info("[CitationVerify]   [%d] BACKFILL — URL", num)
                valid_citations.append({"number": num, "url": canonical_url, "citation_key": None, "line": rewritten})
            else:
                logger.info("[CitationVerify]   [%d] BACKFILL — citation key", num)
                valid_citations.append({"number": num, "url": None, "citation_key": citation_key, "line": full_line})
            continue

        # Neither URL nor recognizable citation key
        logger.debug("[CitationVerify]   [%d] REMOVE — unverifiable", num)
        removed_citations.append({"number": num, "line": full_line, "reason": "unverifiable"})

    # Dedup: collapse multiple [N] source lines that resolve to the same
    # registry source. The model often makes the same tool call twice (e.g.
    # ``mcp_time__get_current_time`` for two timezones) and emits a separate
    # ``[N] tool_name`` line for each call; without this pass both lines
    # survive verification because each is independently valid. We keep the
    # lowest-numbered occurrence and rewrite later inline citations to that
    # number so the prose still cites the source.
    seen_keys: dict[str, int] = {}  # canonical_key -> kept citation number
    duplicate_rewrites: dict[int, int] = {}  # duplicate_num -> canonical_num
    deduped_valid: list[dict] = []
    for c in valid_citations:
        key = c["url"] or c["citation_key"]
        if key is None:
            # Defensive: a valid citation must have one of url/citation_key.
            # If neither is set we cannot dedup, so keep the entry.
            deduped_valid.append(c)
            continue
        canonical_num = seen_keys.get(key)
        if canonical_num is None:
            seen_keys[key] = c["number"]
            deduped_valid.append(c)
            continue
        duplicate_rewrites[c["number"]] = canonical_num
        removed_citations.append(
            {
                "number": c["number"],
                "line": c["line"],
                "reason": f"duplicate_of_citation_{canonical_num}",
            }
        )
        logger.debug(
            "[CitationVerify]   [%d] REMOVE — duplicate of [%d]",
            c["number"],
            canonical_num,
        )
    valid_citations = deduped_valid

    # Apply URL replacements (garbled -> canonical) in the source section
    if url_replacements:
        for garbled, canonical in url_replacements.items():
            ref_section = ref_section.replace(garbled, canonical)

    # Apply backfilled lines (url-less "[N] Title" -> "[N] Title: url").
    if line_replacements:
        for original, rewritten in line_replacements.items():
            ref_section = ref_section.replace(original, rewritten)

    removed_numbers = {c["number"] for c in removed_citations}

    # Remove invalid (and duplicate) source lines from the source section.
    cleaned_ref_lines = [
        line
        for line in ref_section.split("\n")
        if not ((line_match := _CITATION_LINE_RE.match(line)) and int(line_match.group(1)) in removed_numbers)
    ]
    cleaned_ref_section = "\n".join(cleaned_ref_lines)

    # Body fixups:
    #  * Duplicate citations get rewritten to the canonical number — the
    #    cited source is real, only the [N] label is wrong.
    #  * Genuinely invalid citations get stripped — the source is fabricated
    #    or unverifiable.
    cleaned_body = body
    for old_num, canonical_num in duplicate_rewrites.items():
        cleaned_body = re.sub(rf"\[{old_num}\]", f"[{canonical_num}]", cleaned_body)
    valid_numbers = {c["number"] for c in valid_citations}
    cleaned_body = _strip_inline_citations_not_in(cleaned_body, valid_numbers)

    # Note: renumbering is deferred to sanitize_report() which always runs after
    # this function and handles renumbering in a single pass.
    verified_report = cleaned_body + cleaned_ref_section

    logger.debug(
        "[CitationVerify] Result: kept %d, removed %d",
        len(valid_citations),
        len(removed_citations),
    )

    return CitationVerificationResult(
        verified_report=verified_report,
        removed_citations=removed_citations,
        valid_citations=valid_citations,
    )


# ---------------------------------------------------------------------------
# Report sanitization (deterministic post-processing)
# ---------------------------------------------------------------------------

# Known URL shortener domains
_SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "short.io",
        "rb.gy",
        "cutt.ly",
        "lnkd.in",
        "soo.gd",
        "s.coop",
        "cli.gs",
        "budurl.com",
        "yourls.org",
    }
)

# Patterns indicating a truncated/garbled URL
_TRUNCATED_URL_RE = re.compile(r"\.\.\.$|…$")  # ends in ... or ellipsis

# Suspicious URL patterns
_IP_ADDRESS_RE = re.compile(r"^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_SUSPICIOUS_SCHEMES_RE = re.compile(r"^(?:javascript|data|vbscript|file):", re.IGNORECASE)
# Body URL patterns (used by sanitize_report)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*(\w+://[^\s)]+)\)")
_BODY_URL_RE = re.compile(r"\w+://[^\s<>\"']+")


@dataclass
class ReportSanitizationResult:
    """Result of running sanitize_report()."""

    sanitized_report: str
    body_urls_removed: int
    body_urls_replaced: int
    shortened_urls_removed: list[str]
    truncated_urls_removed: list[str]
    unsafe_urls_removed: list[str]


def sanitize_report(report_text: str) -> ReportSanitizationResult:
    """Deterministic report hygiene and final display normalization.

    Checks:
    1. Normalize the source-section heading and one-source-per-line layout.
    2. Strip body URLs — collapse markdown links to display text, replace
       bare URLs that match a reference with ``[N]``, remove the rest.
    3. Remove shortened/obfuscated URLs from Sources — all URLs must
       be fully expanded (no bit.ly, t.co, etc.).
    4. Remove truncated/garbled URLs — URLs ending in '...' or with no
       path (domain-only like 'https://arxiv.org') are incomplete.
    5. Block unsafe URLs — no IP-address URLs, no non-http schemes.
    6. Renumber citations after verification/sanitization and trim any
       trailing source-section meta-commentary.

    Args:
        report_text: Report text (ideally after verify_citations()).

    Returns:
        ReportSanitizationResult with cleaned report and audit trail.
    """
    report_text = _normalize_citation_syntax(report_text)

    body_urls_removed = 0
    body_urls_replaced = 0
    shortened_urls_removed: list[str] = []
    truncated_urls_removed: list[str] = []
    unsafe_urls_removed: list[str] = []

    # Split into body and source section
    ref_match = _REFERENCE_SECTION_RE.search(report_text)
    if ref_match:
        body = report_text[: ref_match.start()]
        ref_section = _normalize_source_section_layout(report_text[ref_match.start() :])
    else:
        body = report_text
        ref_section = ""

    # --- Check 1: Strip body URLs ---
    # Build URL → citation number map from references so matching body
    # URLs are replaced with [N] instead of being deleted entirely.
    url_to_citation: dict[str, int] = {}
    if ref_section:
        for m in _CITATION_LINE_RE.finditer(ref_section):
            num = int(m.group(1))
            url = _first_http_url(m.group(2))
            if url:
                url_to_citation[_normalize_url(url)] = num

    def _replace_body_url(match: re.Match) -> str:
        """Replace a bare body URL with its citation number, or strip it; keep artifact refs."""
        nonlocal body_urls_removed, body_urls_replaced
        url = clean_extracted_url(match.group(0))
        # Preserve internal artifact references (e.g. embedded chart images). They use the
        # non-http ``artifact://`` scheme and are validated/rewritten downstream by
        # ArtifactManager.resolve_report_references, so they must survive sanitization.
        if url.startswith("artifact://"):
            return match.group(0)
        normalized = _normalize_url(url)
        if normalized in url_to_citation:
            body_urls_replaced += 1
            return f"[{url_to_citation[normalized]}]"
        body_urls_removed += 1
        return ""

    # Collapse markdown links to display text, but preserve internal artifact:// image
    # references verbatim so embedded charts survive into the final report.
    def _collapse_md_link(match: re.Match) -> str:
        """Collapse a markdown link to its display text, preserving ``artifact://`` targets."""
        # Preserve only when the link destination is an artifact ref; a label that merely
        # contains "artifact://" must still collapse so stray URLs don't leak into the body.
        if match.group(2).startswith("artifact://"):
            return match.group(0)
        return match.group(1)

    cleaned_body = _MD_LINK_RE.sub(_collapse_md_link, body)
    # Replace matching bare URLs with [N], strip the rest
    cleaned_body = _BODY_URL_RE.sub(_replace_body_url, cleaned_body)
    # Clean up leftover empty parentheses and extra spaces
    cleaned_body = re.sub(r"\(\s*\)", "", cleaned_body)
    cleaned_body = re.sub(r"  +", " ", cleaned_body)

    if body_urls_replaced:
        logger.debug("[ReportSanitize] Replaced %d body URL(s) with citation numbers", body_urls_replaced)
    if body_urls_removed:
        logger.debug("[ReportSanitize] Removed %d unmatched URL(s) from report body", body_urls_removed)

    # --- Checks 3 & 4: Validate URLs in source section ---
    if ref_section:
        lines_to_remove: set[int] = set()
        ref_lines = ref_section.split("\n")

        for i, line in enumerate(ref_lines):
            url = _first_http_url(line)
            if not url:
                continue

            # Check for non-http schemes embedded in text
            if _SUSPICIOUS_SCHEMES_RE.search(line):
                unsafe_urls_removed.append(url)
                lines_to_remove.add(i)
                continue

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check 2: shortened URLs
            # Strip www. and port for comparison
            bare_domain = domain.split(":")[0]
            if bare_domain.startswith("www."):
                bare_domain = bare_domain[4:]
            if bare_domain in _SHORTENER_DOMAINS:
                shortened_urls_removed.append(url)
                lines_to_remove.add(i)
                continue

            # Check 3: truncated/garbled URLs — only catch obvious truncation markers
            raw_url = next(match.group(0) for match in _HTTP_URL_CANDIDATE_RE.finditer(line))
            if _TRUNCATED_URL_RE.search(raw_url) or "…" in raw_url:
                truncated_urls_removed.append(raw_url)
                lines_to_remove.add(i)
                continue

            # Check 4: IP address URLs
            if _IP_ADDRESS_RE.match(url):
                unsafe_urls_removed.append(url)
                lines_to_remove.add(i)
                continue

            # Check 4: non-http schemes
            if parsed.scheme not in ("http", "https"):
                unsafe_urls_removed.append(url)
                lines_to_remove.add(i)
                continue

        if lines_to_remove:
            # Collect which [N] numbers were removed
            removed_numbers: set[int] = set()
            for i in lines_to_remove:
                line_m = _CITATION_LINE_RE.match(ref_lines[i])
                if line_m:
                    removed_numbers.add(int(line_m.group(1)))

            cleaned_ref_lines = [line for i, line in enumerate(ref_lines) if i not in lines_to_remove]
            ref_section = "\n".join(cleaned_ref_lines)

            # Strip orphaned inline [N] from body
            if removed_numbers:
                for num in removed_numbers:
                    cleaned_body = re.sub(rf"\[{num}\]", "", cleaned_body)

        if shortened_urls_removed:
            logger.debug(
                "[ReportSanitize] Removed %d shortened URL(s) from references",
                len(shortened_urls_removed),
            )
        if truncated_urls_removed:
            logger.debug(
                "[ReportSanitize] Removed %d truncated/incomplete URL(s) from references",
                len(truncated_urls_removed),
            )
        if unsafe_urls_removed:
            logger.debug(
                "[ReportSanitize] Removed %d unsafe URL(s) from references",
                len(unsafe_urls_removed),
            )

    # Renumber citations to close any gaps (from verify_citations and/or sanitize removals)
    if ref_section:
        cleaned_body, ref_section, _ = _renumber_citations(cleaned_body, ref_section)

    sanitized_report = cleaned_body + ref_section

    # --- Strip leaked tool-call XML fragments ---
    # LLMs sometimes output raw tool-call syntax as text
    sanitized_report = re.sub(
        r"</?(parameter|function|tool_call|tool_use|invoke|antml:[\w]+)[\s>].*",
        "",
        sanitized_report,
        flags=re.DOTALL,
    )

    # --- Trim everything after the last citation in the Sources section ---
    # The LLM often appends meta-commentary after the references (e.g.,
    # "All citations refer to...", "This report meets..."). Rather than
    # pattern-matching specific phrases, just cut after the last [N] line.
    if ref_section:
        last_citation_end = None
        for m in _CITATION_LINE_RE.finditer(sanitized_report):
            last_citation_end = m.end()
        if last_citation_end is not None:
            sanitized_report = sanitized_report[:last_citation_end].rstrip() + "\n"

    return ReportSanitizationResult(
        sanitized_report=sanitized_report,
        body_urls_removed=body_urls_removed,
        body_urls_replaced=body_urls_replaced,
        shortened_urls_removed=shortened_urls_removed,
        truncated_urls_removed=truncated_urls_removed,
        unsafe_urls_removed=unsafe_urls_removed,
    )
