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

"""NAT registration for DuckDuckGo news search."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from html import escape as html_escape
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

NEWS_BACKENDS = "bing,duckduckgo,yahoo"


class DuckDuckGoNewsSearchToolConfig(FunctionBaseConfig, name="duckduckgo_news_search"):
    """DuckDuckGo news search using the `ddgs` package."""

    max_results: int = Field(default=5, ge=1, le=25, description="Maximum number of news results to return")
    region: str = Field(default="us-en", description="DDGS region code, such as us-en")
    safesearch: Literal["on", "moderate", "off"] = Field(default="moderate", description="Safe search setting")
    timelimit: Literal["d", "w", "m", "y"] | None = Field(
        default="w",
        description="Optional recency window: d=day, w=week, m=month, y=year",
    )
    timeout: float = Field(default=20.0, gt=0, description="Maximum seconds to wait for one search attempt")
    max_retries: int = Field(default=2, ge=1, description="Maximum number of search attempts")


def _result_value(result: dict, *keys: str) -> str:
    """Return the first non-empty string-like result value."""
    for key in keys:
        value = result.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _format_news_result(result: dict) -> str:
    """Render one DDGS news item as a document block."""
    url = html_escape(_result_value(result, "url", "href", "link"), quote=True)
    title = html_escape(_result_value(result, "title"), quote=True)
    body = html_escape(_result_value(result, "body", "snippet", "description"), quote=True)
    source = html_escape(_result_value(result, "source"), quote=True)
    date = html_escape(_result_value(result, "date", "published", "published_date"), quote=True)
    metadata_lines = []
    if source:
        metadata_lines.append(f"<source>{source}</source>")
    if date:
        metadata_lines.append(f"<date>{date}</date>")
    metadata = "\n".join(metadata_lines)
    if metadata:
        metadata = f"\n{metadata}"
    return f'<Document href="{url}">\n<title>\n{title}\n</title>{metadata}\n{body}\n</Document>'


@register_function(config_type=DuckDuckGoNewsSearchToolConfig)
async def duckduckgo_news_search(
    tool_config: DuckDuckGoNewsSearchToolConfig,
    builder: Builder,
) -> AsyncGenerator[FunctionInfo, None]:
    """Register the DuckDuckGo news search tool with NAT."""
    del builder

    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError:

        async def _duckduckgo_news_search_stub(query: str) -> str:
            """News search tool (unavailable - missing optional `ddgs` package)."""
            return (
                "Error: DuckDuckGo news search is unavailable because the `ddgs` package is not installed. "
                "Install the duckduckgo-news-search workspace package dependencies and restart AIQ."
            )

        yield FunctionInfo.from_fn(
            _duckduckgo_news_search_stub,
            description=_duckduckgo_news_search_stub.__doc__,
        )
        return

    async def _duckduckgo_news_search(query: str) -> str:
        """Search recent news using DuckDuckGo News.

        Args:
            query: News search query.

        Returns:
            News search results as citable document blocks with URLs.
        """
        query = query.strip()
        if not query:
            return "Error: query must be a non-empty string"
        if len(query) > 400:
            query = query[:397] + "..."

        def _search() -> list[dict]:
            search_kwargs = {
                "region": tool_config.region,
                "safesearch": tool_config.safesearch,
                "max_results": tool_config.max_results,
                "backend": NEWS_BACKENDS,
            }
            if tool_config.timelimit is not None:
                search_kwargs["timelimit"] = tool_config.timelimit
            with DDGS() as ddgs:
                try:
                    return list(ddgs.news(query, **search_kwargs))
                except DDGSException as exc:
                    if str(exc).strip().lower() == "no results found.":
                        return []
                    raise

        for attempt in range(tool_config.max_retries):
            try:
                results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=tool_config.timeout)
                if results:
                    return "\n\n---\n\n".join(_format_news_result(result) for result in results)
                return "News search returned no results"
            except Exception as exc:  # noqa: BLE001 - source APIs can raise transport-specific exceptions
                if attempt == tool_config.max_retries - 1:
                    logger.exception(
                        "DuckDuckGo news search failed after %s attempts: %s",
                        tool_config.max_retries,
                        exc,
                    )
                    return "Error: News search failed"
                await asyncio.sleep(2**attempt)

        return "Error: News search failed after all retries"

    yield FunctionInfo.from_fn(
        _duckduckgo_news_search,
        description=_duckduckgo_news_search.__doc__,
    )
