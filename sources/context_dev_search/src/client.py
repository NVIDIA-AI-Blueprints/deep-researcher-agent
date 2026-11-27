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

"""NAT-independent async client for the Context.dev Search API.

The client sends one ``POST /web/search`` request per call and returns typed
results. Retry policy lives with the caller (see ``search_with_retries``), so
the HTTP layer stays easy to test with ``httpx.MockTransport``.

API reference: https://docs.context.dev/search/overview
"""

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

CONTEXT_DEV_API_URL = "https://api.context.dev/v1"
SEARCH_PATH = "/web/search"

# The Search API rejects queries longer than this many characters.
MAX_QUERY_LENGTH = 500

# 408 is the API's own request deadline (REQUEST_TIMEOUT); 429 is rate limiting;
# 5xx are server-side failures. Every other 4xx is a caller problem that a retry
# cannot fix (bad request, bad key, no credits, missing entitlement).
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Upper bound on any single backoff sleep, including a server-sent Retry-After.
MAX_BACKOFF_SECONDS = 30.0

# Server-side deadline bounds from the API's ``timeoutOpts`` contract.
_MIN_SERVER_DEADLINE_MS = 1000
_MIN_PARTIAL_DEADLINE_MS = 5000
_RESPONSE_HEADROOM_MS = 1000

Freshness = Literal["last_24_hours", "last_week", "last_month", "last_year"]


@dataclass(frozen=True)
class SearchResult:
    """One ranked result from the Context.dev Search API."""

    url: str
    title: str
    description: str = ""
    highlights: list[str] = field(default_factory=list)


class ContextDevAPIError(Exception):
    """A failed Search API call.

    The message never contains the API key: the key travels only in the
    ``Authorization`` header and is not part of any URL or error payload.

    Attributes:
        status_code: HTTP status, or ``None`` for network-level failures.
        error_code: Machine-readable ``error_code`` from the response body, if any.
        request_id: The API ``request_id`` to quote when contacting support, if any.
        retry_after: Seconds the server asked us to wait (``Retry-After``), if any.
        attempts: How many requests were made before giving up (set by
            ``search_with_retries``).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.retry_after = retry_after
        self.attempts = 1

    @property
    def retryable(self) -> bool:
        """Whether retrying the same request can succeed."""
        return self.status_code is None or self.status_code in RETRYABLE_STATUS_CODES


def _utf16_length(text: str) -> int:
    """Length in UTF-16 code units, the unit the API uses for its length limits."""
    return len(text.encode("utf-16-le")) // 2


def _replace_lone_surrogates(text: str) -> str:
    """Replace unpaired surrogates (from a broken escape in model output) with U+FFFD.

    They cannot be encoded as UTF-8 or UTF-16, so they would otherwise make
    request serialization raise.
    """
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def truncate_query(query: str) -> str:
    """Shorten a query to the API's maximum length, marking the cut with an ellipsis.

    The limit is measured in UTF-16 code units, so characters outside the Basic
    Multilingual Plane (emoji, rare CJK) count twice. Unpaired surrogates are
    replaced with U+FFFD first.
    """
    query = _replace_lone_surrogates(query)
    if _utf16_length(query) <= MAX_QUERY_LENGTH:
        return query
    budget = MAX_QUERY_LENGTH - 3
    kept: list[str] = []
    used = 0
    for char in query:
        width = 2 if ord(char) > 0xFFFF else 1
        if used + width > budget:
            break
        kept.append(char)
        used += width
    return "".join(kept) + "..."


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header given in seconds; ignore HTTP-date forms."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_from_response(response: httpx.Response) -> ContextDevAPIError:
    """Build a ``ContextDevAPIError`` from a non-2xx response."""
    message = ""
    error_code = None
    request_id = response.headers.get("x-request-id")
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        message = str(body.get("message") or "")
        error_code = body.get("error_code") or None
        request_id = body.get("request_id") or request_id
    if not message:
        message = response.reason_phrase or "request failed"
    if response.is_redirect:
        # Redirects are not followed: the Authorization header must not be replayed
        # to another host. A redirect almost always means api_base_url is wrong.
        message = (
            f"unexpected redirect to {response.headers.get('location', 'an unknown location')}; check api_base_url"
        )
    return ContextDevAPIError(
        message,
        status_code=response.status_code,
        error_code=str(error_code) if error_code else None,
        request_id=str(request_id) if request_id else None,
        retry_after=_parse_retry_after(response.headers.get("retry-after")),
    )


def _server_deadline(client_timeout: float) -> dict[str, Any]:
    """Build ``timeoutOpts`` so the API answers before the client gives up.

    The server deadline is one second shorter than the client timeout, leaving
    room for the response to arrive. When that deadline allows it (the API
    requires at least 5 seconds), ``return-partial`` makes the API return the
    results it already has instead of failing the whole search because one
    result page was slow to read.
    """
    milliseconds = max(_MIN_SERVER_DEADLINE_MS, int(client_timeout * 1000) - _RESPONSE_HEADROOM_MS)
    behavior = "return-partial" if milliseconds >= _MIN_PARTIAL_DEADLINE_MS else "fail"
    return {"milliseconds": milliseconds, "behavior": behavior}


def _parse_result(item: Any) -> SearchResult | None:
    """Convert one raw result object into a ``SearchResult``; skip malformed items."""
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    if not isinstance(url, str) or not url:
        return None
    highlights_block = item.get("highlights")
    passages = highlights_block.get("highlights") if isinstance(highlights_block, dict) else None
    return SearchResult(
        url=url,
        title=str(item.get("title") or ""),
        description=str(item.get("description") or ""),
        highlights=[p for p in passages if isinstance(p, str) and p.strip()] if isinstance(passages, list) else [],
    )


class ContextDevSearchClient:
    """Minimal async client for ``POST /web/search``.

    Example:
        >>> async with httpx.AsyncClient() as http:
        ...     client = ContextDevSearchClient(api_key="your-key", http_client=http)  # pragma: allowlist secret
        ...     results = await client.search("attention is all you need", num_results=5)
    """

    def __init__(
        self,
        api_key: str,
        *,
        http_client: httpx.AsyncClient,
        base_url: str = CONTEXT_DEV_API_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Context.dev API key, sent as a Bearer token.
            http_client: Shared ``httpx.AsyncClient``; the caller owns its lifecycle.
            base_url: API base URL, without a trailing ``/web/search``.
            timeout: Per-request timeout in seconds. The same deadline is passed
                to the API so it can stop work it could not return in time.
        """
        self._api_key = api_key
        self._http = http_client
        self._url = base_url.rstrip("/") + SEARCH_PATH
        self._timeout = timeout

    async def search(
        self,
        query: str,
        *,
        num_results: int,
        include_highlights: bool = True,
        max_content_length: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        country: str | None = None,
        freshness: Freshness | None = None,
        zero_data_retention: bool = False,
    ) -> list[SearchResult]:
        """Run one search request.

        Args:
            query: Search query; longer queries are truncated to ``MAX_QUERY_LENGTH``.
            num_results: Number of results to request (1-100).
            include_highlights: Return query-relevant passages from each result page.
            max_content_length: Cap, in characters, for each result's passages and snippet.
            include_domains: Only return results from these domains.
            exclude_domains: Drop results from these domains.
            country: ISO 3166-1 alpha-2 country code used to localize results.
            freshness: Only return content published within this window.
            zero_data_retention: Ask the API not to retain the request or results.
                Requires an organization with zero data retention enabled.

        Returns:
            Results in relevance order. Malformed items are skipped.

        Raises:
            ContextDevAPIError: On a non-2xx response, a network failure, or an
                unexpected response body.
        """
        payload: dict[str, Any] = {"query": truncate_query(query), "numResults": num_results}
        if include_highlights:
            highlights_options: dict[str, Any] = {"enabled": True}
            if max_content_length is not None:
                highlights_options["maxCharacters"] = max_content_length
            payload["highlightsOptions"] = highlights_options
        if max_content_length is not None:
            payload["descriptionMaxCharacters"] = max_content_length
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if country:
            payload["country"] = country.lower()
        if freshness:
            payload["freshness"] = freshness
        if zero_data_retention:
            payload["zdr"] = "enabled"
        payload["timeoutOpts"] = _server_deadline(self._timeout)

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            # httpx timeouts apply per phase (connect, write, read); bound the whole request too.
            async with asyncio.timeout(self._timeout):
                response = await self._http.post(self._url, json=payload, headers=headers, timeout=self._timeout)
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ContextDevAPIError(f"request timed out after {self._timeout:g}s") from exc
        except httpx.HTTPError as exc:
            raise ContextDevAPIError(f"network error ({type(exc).__name__})") from exc

        if response.status_code >= 300:
            raise _error_from_response(response)

        # A 2xx with an unusable body (for example an HTML page from a proxy) is
        # reported without a status code so it is treated like a transient failure.
        try:
            body = response.json()
        except ValueError as exc:
            raise ContextDevAPIError("response was not valid JSON") from exc
        raw_results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(raw_results, list):
            raise ContextDevAPIError("response did not include a results list")
        if body.get("partial"):
            logger.info(
                "Context.dev returned partial results at its deadline (request_id=%s)",
                body.get("request_id"),
            )

        results = [result for result in (_parse_result(item) for item in raw_results) if result is not None]
        logger.debug(
            "Context.dev search returned %d results (request_id=%s)",
            len(results),
            body.get("request_id"),
        )
        return results


async def search_with_retries(
    client: ContextDevSearchClient,
    query: str,
    *,
    max_retries: int,
    **search_kwargs: Any,
) -> list[SearchResult]:
    """Call ``client.search`` with bounded exponential backoff on transient errors.

    Non-retryable errors (400, 401, 402, 403, ...) are raised immediately.
    Transient errors (network failures, unusable response bodies, 408, 429,
    5xx) are retried up to ``max_retries`` total attempts with exponential
    backoff capped at ``MAX_BACKOFF_SECONDS``. A longer ``Retry-After`` within
    that cap replaces the backoff; one beyond the cap ends the retries, since
    retrying sooner would hit the same limit.

    Raises:
        ContextDevAPIError: The last error, once retries are exhausted or the
            error is not retryable.
    """
    for attempt in range(max_retries):
        try:
            return await client.search(query, **search_kwargs)
        except ContextDevAPIError as exc:
            exc.attempts = attempt + 1
            if not exc.retryable or attempt == max_retries - 1:
                raise
            # A server asking for a longer wait than we are willing to sleep would
            # only rate-limit the next attempt too; give up now instead.
            if exc.retry_after is not None and exc.retry_after > MAX_BACKOFF_SECONDS:
                raise
            backoff = min(float(2**attempt), MAX_BACKOFF_SECONDS)
            delay = backoff if exc.retry_after is None else max(exc.retry_after, backoff)
            logger.warning(
                "Context.dev search attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc.status_code or exc.message,
                delay,
            )
            await asyncio.sleep(delay)
    raise ValueError("max_retries must be at least 1")  # pragma: no cover - guarded by config validation
