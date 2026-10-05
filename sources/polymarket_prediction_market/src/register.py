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

"""NAT registration for Polymarket prediction market search."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from collections.abc import Sequence
from html import escape as html_escape
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"
POLYMARKET_WEB_BASE_URL = "https://polymarket.com"
MAX_RETRY_BACKOFF_SECONDS = 30


class PolymarketSearchToolConfig(FunctionBaseConfig, name="polymarket_search"):
    """Search Polymarket events and markets using the public Gamma API."""

    max_results: int = Field(default=5, ge=1, le=20, description="Maximum document blocks to return")
    active: bool = Field(default=True, description="Restrict searches to active/open markets when supported")
    event_scan_limit: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of high-volume active events to scan for keyword matches",
    )
    include_markets_per_event: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum nested markets to render for each matching event",
    )
    timeout: float = Field(default=15.0, gt=0, description="Maximum seconds to wait for one API attempt")
    max_retries: int = Field(default=2, ge=1, description="Maximum number of API attempts")
    gamma_api_base_url: str = Field(
        default=GAMMA_API_BASE_URL,
        description="Polymarket Gamma API base URL",
    )


def _as_text(value: Any) -> str:
    """Return a stripped string for scalar-ish API values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    """Coerce API values to float when possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    """Coerce API values to bool when possible."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _as_list(value: Any) -> list[Any]:
    """Coerce Polymarket JSON-string/list fields to lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _trim(text: str, limit: int = 900) -> str:
    """Trim long API text fields while preserving useful snippets."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _escape(text: str) -> str:
    """Escape text for XML-like document output."""
    return html_escape(text, quote=True)


def _market_title(market: dict[str, Any]) -> str:
    """Return the best display title for a market."""
    return _as_text(market.get("question")) or _as_text(market.get("title")) or _as_text(market.get("slug"))


def _event_title(event: dict[str, Any]) -> str:
    """Return the best display title for an event."""
    return _as_text(event.get("title")) or _as_text(event.get("ticker")) or _as_text(event.get("slug"))


def _query_terms(query: str) -> list[str]:
    """Build simple lexical terms used only to rank public API results."""
    return [part.lower() for part in query.replace("-", " ").split() if len(part) > 2]


def _score_text(text: str, terms: Sequence[str]) -> int:
    """Score text by query-term coverage."""
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _event_score(event: dict[str, Any], terms: Sequence[str]) -> int:
    """Score an event and nested markets against the query."""
    parts = [
        _event_title(event),
        _as_text(event.get("description")),
        _as_text(event.get("category")),
    ]
    for market in _as_list(event.get("markets")):
        if isinstance(market, dict):
            parts.extend([_market_title(market), _as_text(market.get("description"))])
    return _score_text(" ".join(parts), terms)


def _market_score(market: dict[str, Any], terms: Sequence[str]) -> int:
    """Score a market against the query."""
    parts = [
        _market_title(market),
        _as_text(market.get("description")),
        _as_text(market.get("groupItemTitle")),
    ]
    return _score_text(" ".join(parts), terms)


def _format_number(value: Any) -> str:
    """Format volume/liquidity-like numbers compactly."""
    number = _as_float(value)
    if number is None:
        return ""
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:.0f}"


def _format_probability(value: Any) -> str:
    """Format outcome price/probability values."""
    number = _as_float(value)
    if number is None:
        return ""
    if 0 <= number <= 1:
        return f"{number * 100:.1f}%"
    return f"{number:.3g}"


def _polymarket_event_url(slug: str) -> str:
    """Build a public Polymarket event URL."""
    if not slug:
        return POLYMARKET_WEB_BASE_URL
    return f"{POLYMARKET_WEB_BASE_URL}/event/{slug}"


def _market_url(market: dict[str, Any]) -> str:
    """Build the best public URL for a market."""
    direct_url = _as_text(market.get("url"))
    if direct_url:
        return direct_url
    event_slug = _as_text(market.get("eventSlug")) or _as_text(market.get("event_slug"))
    if event_slug:
        return _polymarket_event_url(event_slug)
    return _polymarket_event_url(_as_text(market.get("slug")))


def _format_market_line(market: dict[str, Any]) -> str:
    """Render a market as one compact bullet."""
    title = _escape(_market_title(market) or "Untitled market")
    outcomes = [_as_text(outcome) for outcome in _as_list(market.get("outcomes"))]
    prices = _as_list(market.get("outcomePrices"))
    if not prices:
        prices = _as_list(market.get("outcome_prices"))

    pairs = []
    for index, outcome in enumerate(outcomes):
        price = _format_probability(prices[index]) if index < len(prices) else ""
        if outcome and price:
            pairs.append(f"{_escape(outcome)}: {price}")
        elif outcome:
            pairs.append(_escape(outcome))

    metadata = []
    volume = _format_number(market.get("volume") or market.get("volume24hr"))
    liquidity = _format_number(market.get("liquidity"))
    end_date = _as_text(market.get("endDate") or market.get("end_date"))
    if pairs:
        metadata.append("; ".join(pairs))
    if volume:
        metadata.append(f"volume {volume}")
    if liquidity:
        metadata.append(f"liquidity {liquidity}")
    if end_date:
        metadata.append(f"ends {_escape(end_date)}")

    detail = f" ({'; '.join(metadata)})" if metadata else ""
    return f"- {title}{detail}"


def _format_market_document(market: dict[str, Any]) -> str:
    """Render one market as a citable document block."""
    url = _escape(_market_url(market))
    title = _escape(_market_title(market) or "Polymarket market")
    description = _escape(_trim(_as_text(market.get("description"))))
    metadata_lines = [
        "<source>Polymarket</source>",
        "<source_type>prediction_market</source_type>",
    ]
    active = _as_bool(market.get("active"))
    if active is not None:
        metadata_lines.append(f"<active>{str(active).lower()}</active>")
    end_date = _as_text(market.get("endDate") or market.get("end_date"))
    if end_date:
        metadata_lines.append(f"<end_date>{_escape(end_date)}</end_date>")
    volume = _format_number(market.get("volume") or market.get("volume24hr"))
    if volume:
        metadata_lines.append(f"<volume>{volume}</volume>")
    body = "\n".join([_format_market_line(market), description]).strip()
    return (
        f'<Document href="{url}">\n<title>\n{title}\n</title>\n' + "\n".join(metadata_lines) + f"\n{body}\n</Document>"
    )


def _format_event_document(event: dict[str, Any], max_markets: int) -> str:
    """Render one event with nested markets as a citable document block."""
    slug = _as_text(event.get("slug"))
    url = _escape(_polymarket_event_url(slug))
    title = _escape(_event_title(event) or "Polymarket event")
    description = _escape(_trim(_as_text(event.get("description"))))
    metadata_lines = [
        "<source>Polymarket</source>",
        "<source_type>prediction_market</source_type>",
    ]
    active = _as_bool(event.get("active"))
    if active is not None:
        metadata_lines.append(f"<active>{str(active).lower()}</active>")
    end_date = _as_text(event.get("endDate") or event.get("end_date"))
    if end_date:
        metadata_lines.append(f"<end_date>{_escape(end_date)}</end_date>")
    volume = _format_number(event.get("volume") or event.get("volume24hr"))
    if volume:
        metadata_lines.append(f"<volume>{volume}</volume>")
    markets = [
        _format_market_line(market)
        for market in _as_list(event.get("markets"))[:max_markets]
        if isinstance(market, dict)
    ]
    body_parts = [description] if description else []
    if markets:
        body_parts.append("<markets>\n" + "\n".join(markets) + "\n</markets>")
    body = "\n".join(body_parts).strip()
    return (
        f'<Document href="{url}">\n<title>\n{title}\n</title>\n' + "\n".join(metadata_lines) + f"\n{body}\n</Document>"
    )


def _dedupe_documents(documents: Sequence[tuple[str, str]]) -> list[str]:
    """Return document text deduped by URL/title key while preserving order."""
    seen: set[str] = set()
    output: list[str] = []
    for key, document in documents:
        if key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output


async def _fetch_json(client: Any, base_url: str, path: str, params: dict[str, Any]) -> Any:
    """Fetch JSON from the Gamma API."""
    response = await client.get(f"{base_url.rstrip('/')}/{path.lstrip('/')}", params=params)
    response.raise_for_status()
    return response.json()


def _list_payload(data: Any, key: str) -> list[dict[str, Any]]:
    """Normalize Gamma API list or keyed-list responses."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        value = data.get(key) or data.get("data") or data.get("results")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


async def _search_events(client: Any, config: PolymarketSearchToolConfig, query: str) -> list[dict[str, Any]]:
    """Search active high-volume events and rank them against the query."""
    params: dict[str, Any] = {
        "limit": config.event_scan_limit,
        "order": "volume24hr",
        "ascending": "false",
    }
    if config.active:
        params["active"] = "true"
        params["closed"] = "false"
    data = await _fetch_json(client, config.gamma_api_base_url, "/events", params)
    terms = _query_terms(query)
    scored = [
        (score, index, event)
        for index, event in enumerate(_list_payload(data, "events"))
        if (score := _event_score(event, terms)) > 0
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [event for _, _, event in scored[: config.max_results]]


async def _search_markets(client: Any, config: PolymarketSearchToolConfig, query: str) -> list[dict[str, Any]]:
    """Search markets by keyword and rank returned markets against the query."""
    params: dict[str, Any] = {
        "limit": config.max_results,
        "keyword": query,
        "order": "volume24hr",
        "ascending": "false",
    }
    if config.active:
        params["active"] = "true"
        params["closed"] = "false"
    data = await _fetch_json(client, config.gamma_api_base_url, "/markets", params)
    terms = _query_terms(query)
    markets = _list_payload(data, "markets")
    markets.sort(key=lambda market: _market_score(market, terms), reverse=True)
    return markets[: config.max_results]


@register_function(config_type=PolymarketSearchToolConfig)
async def polymarket_search(
    tool_config: PolymarketSearchToolConfig,
    builder: Builder,
) -> AsyncGenerator[FunctionInfo, None]:
    """Register the Polymarket prediction market search tool with NAT."""
    del builder

    try:
        import httpx
    except ImportError:

        async def _polymarket_search_stub(query: str) -> str:
            """Prediction market search (unavailable - missing optional `httpx` package)."""
            return (
                "Error: Polymarket search is unavailable because the `httpx` package is not installed. "
                "Install the polymarket-prediction-market workspace package dependencies and restart AIQ."
            )

        yield FunctionInfo.from_fn(
            _polymarket_search_stub,
            description=_polymarket_search_stub.__doc__,
        )
        return

    async def _polymarket_search(query: str) -> str:
        """Search Polymarket prediction markets and events.

        Args:
            query: Search query describing the event, market, entity, or outcome.

        Returns:
            Polymarket event and market results as citable document blocks.
        """
        query = query.strip()
        if not query:
            return "Error: query must be a non-empty string"
        if len(query) > 300:
            query = query[:297] + "..."

        last_error: Exception | None = None
        for attempt in range(tool_config.max_retries):
            try:
                # The client timeout caps each individual HTTP request.
                async with httpx.AsyncClient(timeout=tool_config.timeout) as client:
                    # wait_for caps the combined event+market search attempt.
                    events, markets = await asyncio.wait_for(
                        asyncio.gather(
                            _search_events(client, tool_config, query),
                            _search_markets(client, tool_config, query),
                        ),
                        timeout=tool_config.timeout,
                    )
                documents: list[tuple[str, str]] = []
                for event in events:
                    key = _as_text(event.get("slug")) or _event_title(event)
                    documents.append(
                        (
                            f"event:{key}",
                            _format_event_document(event, tool_config.include_markets_per_event),
                        )
                    )
                for market in markets:
                    key = _market_url(market) or _market_title(market)
                    documents.append((f"market:{key}", _format_market_document(market)))
                output = _dedupe_documents(documents)[: tool_config.max_results]
                if output:
                    return "\n\n---\n\n".join(output)
                return "Polymarket search returned no results"
            except Exception as exc:  # noqa: BLE001 - source APIs can raise transport-specific exceptions
                last_error = exc
                if attempt == tool_config.max_retries - 1:
                    logger.warning("Polymarket search failed for query %r: %s", query, exc)
                    return f"Error: Polymarket search failed - {exc}"
                await asyncio.sleep(min(2**attempt, MAX_RETRY_BACKOFF_SECONDS))

        return f"Error: Polymarket search failed - {last_error}"

    yield FunctionInfo.from_fn(
        _polymarket_search,
        description=_polymarket_search.__doc__,
    )
