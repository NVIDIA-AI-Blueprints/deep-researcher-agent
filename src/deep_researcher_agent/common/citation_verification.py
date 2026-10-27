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
    all_citations: list[dict] = field(default_factory=list)
    removed_citations: list[dict] = field(default_factory=list)
    valid_citations: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


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
    r"^https?://[^\s<>\"]+?\]\((?P<target>https?://[^\s<>\"]+)\)[.,;]*$",
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


def calculate_confidence(match_kind: str) -> float:
    """Map match type to confidence score."""
    return {
        "exact": 1.0,
        "normalized": 0.9, # Used for truncation match previously
        "prefix": 0.8,
        "child-path": 0.7,
        "query-subset": 0.6,
        "host-only": 0.5,
        "unmatched": 0.0
    }.get(match_kind, 0.0)

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

    def resolve_with_kind(self, url: str) -> tuple[str | None, str]:
        """Resolve URL and return (resolved_url, match_kind)."""
        # 1. Exact match — raw or normalized; retain the tool's URL
        if url in self._urls:
            return self._urls[url].url, "exact"
        normalized = _normalize_url(url)
        if normalized in self._urls:
            return self._urls[normalized].url, "exact"

        # 2. Truncation — report URL is a prefix of exactly one registry URL (raw).
        #    Normalized match fails when the report is cut mid-query (param order differs).
        truncation_entries = [e for e in self._urls.values() if e.url and e.url.startswith(url)]
        result = self._pick_unique(list({e.url: e for e in truncation_entries}.values()), "truncation", url)
        if result:
            return result, "normalized"

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
            return result, "prefix"

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
            return result, "child-path"

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
                return result, "query-subset"
        
        # 6. Host-only match (if none of the above matched)
        host_only_entries = [p.entry for p in self._parsed_urls.values() if p.host == host]
        result = self._pick_unique(list({e.url: e for e in host_only_entries}.values()), "host-only", url)
        if result:
            return result, "host-only"

        return None, "unmatched"

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


def get_session_source_registry(session_id: str) -> SourceRegistry:
    """Return the SourceRegistry for the given session ID, creating if needed."""
    with _session_registries_lock:
        if session_id not in _session_registries:
            if len(_session_registries) >= _MAX_SESSION_REGISTRIES:
                _session_registries.popitem(last=False)
            _session_registries[session_id] = SourceRegistry()
            logger.debug("[SourceRegistry] Created new session registry %s", session_id)
        _session_registries.move_to_end(session_id)
        return _session_registries[session_id]


class SourceRegistryMiddleware:
    """Assigns a new or existing SourceRegistry to the current ContextVar."""

    def __init__(self, session_id: str) -> None:
        """Initialize with the session ID."""
        self._session_id = session_id
        self._old_registry_token: contextvars.Token | None = None

    async def __aenter__(self) -> SourceRegistry:
        """Enter the async context, binding the registry to the ContextVar."""
        registry = get_session_source_registry(self._session_id)
        self._old_registry_token = _session_source_registry.set(registry)
        return registry

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, restoring the previous ContextVar value."""
        if self._old_registry_token:
            _session_source_registry.reset(self._old_registry_token)


def current_source_registry() -> SourceRegistry | None:
    """Return the SourceRegistry for the current context, or None if not set."""
    return _session_source_registry.get()


# ---------------------------------------------------------------------------
# Citation syntax parsing
# ---------------------------------------------------------------------------

# Match standalone ``[N]`` and ``[N]:`` (the former allows parsing inline
# citations that have no trailing ``:`` separator).
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\](?::)?")

# Match citation lines in a source section, e.g. ``[1] Title: URL``.
# We capture the number and the rest of the line (title, URL) separately.
_CITATION_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)", re.M)

# Match variations of the knowledge-layer citation pattern, e.g. ``(Filename)``.
_KL_CITATION_PATTERN_RE = re.compile(r"\(.+\)")

# Match footnoted inline citations like ``[1, p.15]``, which the writer sometimes
# produces when the knowledge-layer tool returns page numbers.
_SOURCE_LOCATION_CITATION_RE = re.compile(r"\[(\d+),\s*(?:p\.?|page)\s*\d+\]")

# Match bare markdown footnote references that the LLM may produce from
# hallucinated output, e.g. ``[^1]``, then convert to inline ``[1]``.
_FOOTNOTE_INLINE_CITATION_RE = re.compile(r"\^\[(\d+)\]")
# Match bare markdown footnote reference sections at the end of the report
# e.g. ``[^1]: https://...``, then convert to inline ``[1]``
_FOOTNOTE_REFERENCE_LINE_RE = re.compile(r"\^\[(\d+)\]:")

# Match ``### References``, ``## References``, etc. or a bare ``References`` line.
_REFERENCE_HEADING_LINE_RE = re.compile(r"^#+\s*References\s*$", re.M | re.IGNORECASE)
# Match the beginning of the References section (``# References`` or ``## Sources``)
_REFERENCE_SECTION_RE = re.compile(
    r"^(?:#+\s*(?:References|Sources)|References)\s*$\n", re.M | re.IGNORECASE
)

# Match shortened URL domains, excluding common top-level domains that
# aren't URL shorteners (e.g. co.uk)
_SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly",
        "goo.gl",
        "t.co",
        "tinyurl.com",
        "ow.ly",
        "buff.ly",
        "cutt.ly",
        "rebrand.ly",
        "rb.gy",
    }
)

_TRUNCATED_URL_RE = re.compile(r"(,\s*|\.{2,}|\s*\[\.{2,})$")
_IP_ADDRESS_RE = re.compile(r"^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_SUSPICIOUS_SCHEMES_RE = re.compile(r"(?:file|ftp|sftp|gopher|telnet|ssh):", re.IGNORECASE)

_MD_LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
_BODY_URL_RE = re.compile(r"https?://[^\s<>\"]+")

# Match an ordered list entry of the form `N. ` or `N) ` or `[N] `
_ORDERED_REFERENCE_LINE_RE = re.compile(r"^\s*(\d+)[). ]\s*(.+?)$", re.M)

_COLLAPSED_SOURCE_BOUNDARY_RE = re.compile(r"\s*\[(\d+)\]\s*")


@dataclass
class ReportSanitizationResult:
    """Result of running sanitize_report()."""

    sanitized_report: str
    body_urls_removed: int
    body_urls_replaced: int
    shortened_urls_removed: list[str]
    truncated_urls_removed: list[str]
    unsafe_urls_removed: list[str]


def _is_knowledge_citation(ref_text: str) -> tuple[bool, str | None]:
    """Return true when the source line is a knowledge-layer citation key.

    Matches if a phrase like ``(Filename)`` or a full knowledge-layer
    citation key (e.g. ``(report.pdf, p.15)``) is present in the line.
    """
    # 1. Exact match
    m = _KL_CITATION_PATTERN_RE.match(ref_text)
    if m:
        # Strip parentheses
        citation_key = ref_text.strip("()").strip()
        if citation_key:
            return True, citation_key

    # 2. Leaky agent pattern (when the LLM leaks an internal ID)
    m = _KL_LEAKY_CITATION_PATTERN_RE.match(ref_text)
    if m:
        return True, m.group(1)

    # 3. Fuzzy match: case-insensitive match of any registered knowledge-layer filename
    # appears in the reference text (handles arbitrary LLM formatting)
    if registry := current_source_registry():
        ref_lower = ref_text.lower()
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
    return re.sub(r"[ 	]+(?=[,.;:!?])", "", stripped)


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
    reference_sources: Sequence[SourceEntry] | None = None,
) -> CitationVerificationResult:
    """Verify citations with confidence scoring."""
    report_text = _normalize_citation_syntax(report_text)
    
    all_citations = []
    diagnostics: list[str] = []

    # Identify citation lines and resolve with kind
    ref_match = _REFERENCE_SECTION_RE.search(report_text)
    if ref_match:
        ref_section = _normalize_source_section_layout(report_text[ref_match.start():])
        for line_match in _CITATION_LINE_RE.finditer(ref_section):
            num = int(line_match.group(1))
            ref_text = line_match.group(2).strip()
            
            url = _first_http_url(ref_text)
            if url:
                resolved_url, kind = registry.resolve_with_kind(url)
                confidence = calculate_confidence(kind)
                all_citations.append({
                    "number": num,
                    "url": resolved_url or url, 
                    "confidence": confidence,
                    "match_kind": kind
                })
                if kind == "unmatched":
                    diagnostics.append(f"Citation {num} ({url}) could not be matched to any source.")
                elif confidence < 1.0:
                    diagnostics.append(f"Citation {num} ({url}) partially matched (kind: {kind}, confidence: {confidence}).")
            else:
                # Handle knowledge-layer citations if applicable
                is_knowledge, citation_key = _is_knowledge_citation(ref_text)
                if is_knowledge and citation_key:
                    # For now, knowledge citations are binary; no confidence scoring specified.
                    # Assuming exact match for present citation keys.
                    if registry.has_citation_key(citation_key):
                        all_citations.append({
                            "number": num,
                            "citation_key": citation_key,
                            "confidence": 1.0, # Treat as exact for now
                            "match_kind": "exact_knowledge_key"
                        })
                    else:
                        all_citations.append({
                            "number": num,
                            "citation_key": citation_key,
                            "confidence": 0.0,
                            "match_kind": "unmatched_knowledge_key"
                        })
                        diagnostics.append(f"Knowledge citation {num} ({citation_key}) not found in registry.")
                else:
                    all_citations.append({
                        "number": num,
                        "url": None, # No URL found
                        "confidence": 0.0,
                        "match_kind": "malformed_citation"
                    })
                    diagnostics.append(f"Citation {num} ('{ref_text}') is malformed or has no verifiable target.")
    
    # The issue specifies that citations are not removed at this stage;
    # the UI decides how to render based on confidence. Thus, verified_report
    # remains the original report_text.
    return CitationVerificationResult(verified_report=report_text, all_citations=all_citations, diagnostics=diagnostics)


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
