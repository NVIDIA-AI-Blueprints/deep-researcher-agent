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
"""Paper search tool using Google Scholar via Serper, SerpAPI, SearchAPI, or Serply.

This module contains the NAT-independent PaperSearchTool class. The provider
is selected at construction time; each provider's raw response is normalized
into a common shape consumed by ``format_results``.
"""

import asyncio
import logging
import math
import re
from enum import StrEnum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

SERPER_API_URL = "https://google.serper.dev/scholar"
SERPAPI_API_URL = "https://serpapi.com/search"
SEARCHAPI_API_URL = "https://www.searchapi.io/api/v1/search"
SERPLY_API_URL = "https://api.serply.io/v1/scholar"

# Matches a 4-digit publication year (1500-2099). The window is wide enough to
# cover pre-1900 scholarly works while excluding arXiv-style identifiers, and
# the negative lookahead prevents matching the leading digits of an id such as
# ``1910.12345``.
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b(?!\.\d)")


class PaperSearchProvider(StrEnum):
    """Supported Google Scholar search providers."""

    SERPER = "serper"
    SERPAPI = "serpapi"
    SEARCHAPI = "searchapi"
    SERPLY = "serply"


class PaperSearchTool:
    """Paper search tool for academic papers using Google Scholar.

    The ``provider`` argument selects which backend API to use. All providers
    return results normalized to the same shape so ``format_results`` and the
    rest of the pipeline are provider-agnostic.

    This class is NAT-independent and receives all dependencies via constructor.

    Example:
        >>> # Serper (default, backward compatible)
        >>> tool = PaperSearchTool(serper_api_key="your-key")  # pragma: allowlist secret
        >>>
        >>> # SerpAPI
        >>> tool = PaperSearchTool(
        ...     provider="serpapi",
        ...     serpapi_api_key="your-key",  # pragma: allowlist secret
        ... )
        >>>
        >>> # SearchAPI
        >>> tool = PaperSearchTool(
        ...     provider="searchapi",
        ...     searchapi_api_key="your-key",  # pragma: allowlist secret
        ... )
        >>>
        >>> # Serply
        >>> tool = PaperSearchTool(
        ...     provider="serply",
        ...     serply_api_key="your-key",  # pragma: allowlist secret
        ... )
        >>> result = await tool.search("machine learning transformers")
    """

    def __init__(
        self,
        serper_api_key: str | None = None,
        *,
        provider: str | PaperSearchProvider = PaperSearchProvider.SERPER,
        serpapi_api_key: str | None = None,
        searchapi_api_key: str | None = None,
        serply_api_key: str | None = None,
        timeout: int = 30,
        max_results: int = 10,
    ) -> None:
        """Initialize the paper search tool.

        Args:
            serper_api_key: API key for Serper. Kept as the first positional
                argument for backward compatibility; required when
                ``provider="serper"``.
            provider: Which backend to use — ``"serper"``, ``"serpapi"``,
                ``"searchapi"``, or ``"serply"``. Defaults to ``"serper"``.
            serpapi_api_key: API key for SerpAPI (required when provider is serpapi).
            searchapi_api_key: API key for SearchAPI (required when provider is searchapi).
            serply_api_key: API key for Serply (required when provider is serply).
            timeout: Timeout in seconds for search requests (default 30).
            max_results: Maximum number of search results to return (default 10).
        """
        self.provider = PaperSearchProvider(provider)
        self.serper_api_key = serper_api_key
        self.serpapi_api_key = serpapi_api_key
        self.searchapi_api_key = searchapi_api_key
        self.serply_api_key = serply_api_key
        self.timeout = timeout
        self.max_results = max_results

    def _selected_api_key(self) -> str | None:
        """Return the API key for the active provider, or ``None`` if unset."""
        if self.provider is PaperSearchProvider.SERPER:
            return self.serper_api_key
        if self.provider is PaperSearchProvider.SERPAPI:
            return self.serpapi_api_key
        if self.provider is PaperSearchProvider.SEARCHAPI:
            return self.searchapi_api_key
        if self.provider is PaperSearchProvider.SERPLY:
            return self.serply_api_key
        return None  # pragma: no cover - exhausted by enum

    # ── Public API ──
    async def search(
        self,
        query: str,
        year: str | None = None,
    ) -> str:
        """Search for peer-reviewed academic papers and scientific publications.

        This method returns papers from Google Scholar with citations, abstracts,
        and links for research queries requiring authoritative, scholarly sources
        including: scientific concepts, algorithms, methodologies, technical
        foundations, theoretical frameworks, empirical studies, and peer-reviewed
        evidence.

        Args:
            query: The search query string.
            year: Optional year or year range (e.g., "2023" or "2020-2023").

        Returns:
            Formatted string with search results.
        """
        if not query:
            return "Error: 'query' argument is required"

        if year is not None and not isinstance(year, str):
            year = str(year)

        if not self._selected_api_key():
            logger.warning(
                "Paper search unavailable: no API key configured for provider '%s'",
                self.provider.value,
            )
            return (
                f"Error: Paper search is unavailable because no API key is configured "
                f"for provider '{self.provider.value}'."
            )

        logger.info(f"Paper search ({self.provider.value}) for: {query}")

        try:
            if self.provider is PaperSearchProvider.SERPER:
                results = await self._search_serper(query, year, self.max_results)
            elif self.provider is PaperSearchProvider.SERPAPI:
                results = await self._search_serpapi(query, year, self.max_results)
            elif self.provider is PaperSearchProvider.SEARCHAPI:
                results = await self._search_searchapi(query, year, self.max_results)
            elif self.provider is PaperSearchProvider.SERPLY:
                results = await self._search_serply(query, year, self.max_results)
            else:  # pragma: no cover - exhausted by enum
                raise ValueError(f"Unsupported provider: {self.provider}")
            return self.format_results(results)

        except TimeoutError:
            logger.error("Paper search timed out after %ss for provider '%s'", self.timeout, self.provider.value)
            return f"Paper search timed out after {self.timeout}s. Try again or narrow the query."
        except Exception:
            logger.error("Paper search failed for provider '%s'", self.provider.value)
            return f"Paper search failed: unable to fetch results from {self.provider.value}."

    @staticmethod
    def format_results(results: list[dict[str, Any]]) -> str:
        """Format normalized Google Scholar results.

        Expects each dict to have the keys: ``title``, ``year``, ``snippet``,
        ``link``, ``publicationInfo``, ``citedBy``. All provider responses are
        normalized to this shape before reaching here.
        """
        if not results:
            return "No papers found via Google Scholar."

        formatted_papers = []
        for i, paper in enumerate(results, 1):
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "Unknown Year")
            snippet = paper.get("snippet", "")
            link = paper.get("link", "")
            pub_info = paper.get("publicationInfo", "")
            citations = paper.get("citedBy", 0)

            paper_str = (
                f"{i}. **{title}** ({year})\n"
                f"   - **Publication**: {pub_info}\n"
                f"   - **Citations**: {citations}\n"
                f"   - **Snippet**: {snippet}\n"
                f"   - **Link**: {link}"
            )
            formatted_papers.append(paper_str)

        return "\n\n".join(formatted_papers)

    # ── Shared helpers ──
    @staticmethod
    def _parse_year_range(year: str | None) -> tuple[str | None, str | None]:
        """Parse a year argument into a ``(start_year, end_year)`` tuple.

        Accepts a single year (``"2023"``) or a range (``"2020-2023"``,
        ``"-2023"``, ``"2020-"``).
        """
        if not year:
            return None, None
        if "-" in year:
            parts = year.split("-")
            if len(parts) == 2:
                start_year = parts[0] if parts[0] else None
                end_year = parts[1] if parts[1] else None
                return start_year, end_year
        return year, year

    @staticmethod
    def _extract_year(publication_info: Any) -> str:
        """Extract the publication year from a publication summary string.

        SerpAPI, SearchAPI, and Serply embed the year inside the publication summary
        (e.g. ``"JL Harper - ..., 1977 - cabdirect.org"``). The publication
        year is the final year token before the trailing source suffix, not the
        first one: ``"... (1919-1933 …, 1926 - JSTOR"`` yields ``1926``, not
        ``1919``. Serper returns a clean ``year`` field, so this is only used
        for the other providers.
        """
        if not isinstance(publication_info, str):
            return "Unknown Year"
        # Drop the trailing source/host suffix (" - cabdirect.org") so its
        # tokens (e.g. arXiv identifiers) can't be mistaken for a year, then
        # take the final year-shaped token in the remaining summary.
        summary = publication_info.rsplit(" - ", 1)[0]
        matches = _YEAR_RE.findall(summary)
        return matches[-1] if matches else "Unknown Year"

    # ── Serper (POST, X-API-KEY header) ──
    async def _fetch_serper_page(
        self,
        query: str,
        num: int,
        offset: int,
        start_year: str | None,
        end_year: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "q": query,
            "num": min(num, 20),
            "start": offset,
        }

        if start_year:
            payload["as_ylo"] = start_year
        if end_year:
            payload["as_yhi"] = end_year

        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                SERPER_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Serper API error: HTTP {response.status}")
                return await response.json()

    async def _search_serper(
        self,
        query: str,
        year: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform search using Serper (Google Scholar).

        Serper already returns the normalized shape (``title``, ``year``,
        ``snippet``, ``link``, ``publicationInfo``, ``citedBy``), so results
        are returned as-is.
        """
        start_year, end_year = self._parse_year_range(year)

        limit = min(limit, 50)

        page_size = 10  # Serper default/typical
        total_pages = math.ceil(limit / page_size)

        tasks = []
        for page in range(total_pages):
            current_limit = min(page_size, limit - (page * page_size))
            if current_limit <= 0:
                break

            tasks.append(
                self._fetch_serper_page(
                    query,
                    current_limit,
                    page * page_size,
                    start_year,
                    end_year,
                )
            )

        page_results = await asyncio.gather(*tasks)

        all_papers = []
        for result in page_results:
            if result.get("organic"):
                all_papers.extend(result["organic"])

        return all_papers[:limit]

    # ── SerpAPI (GET, api_key query param, start offset) ──
    async def _fetch_serpapi_page(
        self,
        query: str,
        num: int,
        offset: int,
        start_year: str | None,
        end_year: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_scholar",
            "q": query,
            "num": min(num, 20),
            "start": offset,
            "api_key": self.serpapi_api_key or "",
        }

        if start_year:
            params["as_ylo"] = start_year
        if end_year:
            params["as_yhi"] = end_year

        async with aiohttp.ClientSession() as session:
            async with session.get(
                SERPAPI_API_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"SerpAPI error: HTTP {response.status}")
                return await response.json()

    async def _search_serpapi(
        self,
        query: str,
        year: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        start_year, end_year = self._parse_year_range(year)
        limit = min(limit, 50)

        page_size = 10
        total_pages = math.ceil(limit / page_size)

        tasks = []
        for page in range(total_pages):
            current_limit = min(page_size, limit - (page * page_size))
            if current_limit <= 0:
                break
            tasks.append(
                self._fetch_serpapi_page(
                    query,
                    current_limit,
                    page * page_size,
                    start_year,
                    end_year,
                )
            )

        page_results = await asyncio.gather(*tasks)
        raw = []
        for result in page_results:
            raw.extend(result.get("organic_results", []))
        return self._normalize_serpapi(raw)[:limit]

    @staticmethod
    def _normalize_serpapi(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for paper in results:
            pub_info = paper.get("publication_info", {}) or {}
            summary = pub_info.get("summary", "")
            inline_links = paper.get("inline_links", {}) or {}
            cited_by = inline_links.get("cited_by", {}) or {}
            normalized.append(
                {
                    "title": paper.get("title", "Unknown Title"),
                    "year": PaperSearchTool._extract_year(summary),
                    "snippet": paper.get("snippet", ""),
                    "link": paper.get("link", ""),
                    "publicationInfo": summary,
                    "citedBy": cited_by.get("total", 0),
                }
            )
        return normalized

    # ── SearchAPI (GET, api_key query param, 1-based page) ──
    async def _fetch_searchapi_page(
        self,
        query: str,
        num: int,
        page: int,
        start_year: str | None,
        end_year: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_scholar",
            "q": query,
            "num": min(num, 20),
            "page": page,
            "api_key": self.searchapi_api_key or "",
        }

        if start_year:
            params["as_ylo"] = start_year
        if end_year:
            params["as_yhi"] = end_year

        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARCHAPI_API_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"SearchAPI error: HTTP {response.status}")
                return await response.json()

    async def _search_searchapi(
        self,
        query: str,
        year: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        start_year, end_year = self._parse_year_range(year)
        limit = min(limit, 50)

        page_size = 10
        total_pages = math.ceil(limit / page_size)

        tasks = []
        for page_idx in range(total_pages):
            current_limit = min(page_size, limit - (page_idx * page_size))
            if current_limit <= 0:
                break
            # SearchAPI pages are 1-based
            tasks.append(
                self._fetch_searchapi_page(
                    query,
                    current_limit,
                    page_idx + 1,
                    start_year,
                    end_year,
                )
            )

        page_results = await asyncio.gather(*tasks)
        raw = []
        for result in page_results:
            raw.extend(result.get("organic_results", []))
        return self._normalize_searchapi(raw)[:limit]

    @staticmethod
    def _normalize_searchapi(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for paper in results:
            pub_info = paper.get("publication", "") or ""
            inline_links = paper.get("inline_links", {}) or {}
            cited_by = inline_links.get("cited_by", {}) or {}
            normalized.append(
                {
                    "title": paper.get("title", "Unknown Title"),
                    "year": PaperSearchTool._extract_year(pub_info),
                    "snippet": paper.get("snippet", ""),
                    "link": paper.get("link", ""),
                    "publicationInfo": pub_info,
                    "citedBy": cited_by.get("total", 0),
                }
            )
        return normalized

    # ── Serply (GET, X-Api-Key header, start offset) ──
    async def _fetch_serply_page(
        self,
        query: str,
        num: int,
        offset: int,
        start_year: str | None,
        end_year: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": query,
            "num": min(num, 20),
            "start": offset,
        }

        if start_year:
            params["as_ylo"] = start_year
        if end_year:
            params["as_yhi"] = end_year

        headers = {
            "X-Api-Key": self.serply_api_key or "",
            "User-Agent": "aiq",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                SERPLY_API_URL,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Serply API error: HTTP {response.status}")
                return await response.json()

    async def _search_serply(
        self,
        query: str,
        year: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        start_year, end_year = self._parse_year_range(year)
        limit = min(limit, 50)

        page_size = 10
        total_pages = math.ceil(limit / page_size)

        tasks = []
        for page in range(total_pages):
            current_limit = min(page_size, limit - (page * page_size))
            if current_limit <= 0:
                break
            tasks.append(
                self._fetch_serply_page(
                    query,
                    current_limit,
                    page * page_size,
                    start_year,
                    end_year,
                )
            )

        page_results = await asyncio.gather(*tasks)
        raw = []
        for result in page_results:
            raw.extend(result.get("articles", []))
        return self._normalize_serply(raw)[:limit]

    @staticmethod
    def _normalize_serply(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize Serply ``articles`` into the shared result shape.

        Serply puts the publication summary (authors, venue, year, host) in
        ``author.names`` and reports citations as a ``"Cited by N"`` string
        under ``extras.citations.count``. Its Scholar results carry no abstract
        snippet: ``description`` repeats the publication summary, so it is
        only used as the snippet when it adds something new.
        """
        normalized = []
        for paper in results:
            author = paper.get("author", {}) or {}
            summary = (author.get("names", "") or "").replace("\xa0", " ")
            description = (paper.get("description", "") or "").replace("\xa0", " ")
            extras = paper.get("extras", {}) or {}
            citations = extras.get("citations", {}) or {}
            cited_match = re.search(r"\d+", str(citations.get("count", "")).replace(",", ""))
            normalized.append(
                {
                    "title": paper.get("title", "Unknown Title"),
                    "year": PaperSearchTool._extract_year(summary),
                    "snippet": "" if description == summary else description,
                    "link": paper.get("link", ""),
                    "publicationInfo": summary,
                    "citedBy": int(cited_match.group()) if cited_match else 0,
                }
            )
        return normalized
