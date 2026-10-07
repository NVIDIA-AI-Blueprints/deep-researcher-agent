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

import asyncio
import html
import logging
import os
import re
from collections.abc import AsyncGenerator

import httpx
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

_missing_key_warned = False
_INVALID_XML_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF￾￿]")

_DEFAULT_API_URL = "https://api.firecrawl.dev"
_ORIGIN = "aiq"
_RETRYABLE_STATUS_CODES = frozenset({408, 429})
_MAX_RETRY_DELAY_SECONDS = 30


def _xml_text(value: object) -> str:
    """Normalize provider text and remove characters forbidden by XML 1.0."""
    return _INVALID_XML_CHARACTERS.sub("", "" if value is None else str(value))


def _render_document(url: object, title: object, content: object) -> str:
    """Render provider-controlled fields inside the trusted document structure."""
    url_text = _xml_text(url)
    title_text = _xml_text(title)
    content_text = _xml_text(content)
    return (
        f'<Document href="{html.escape(url_text, quote=True)}">\n'
        f"<title>\n{html.escape(title_text, quote=True)}\n</title>\n"
        f"{html.escape(content_text, quote=True)}\n</Document>"
    )


class _RetryableSearchError(Exception):
    """Raised for responses that are worth retrying (408, 429, 5xx)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the wait the API asks for: the `Retry-After` header, else `retry_after_seconds` in the body."""
    candidates: list[object] = [response.headers.get("Retry-After")]
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        candidates.append(payload.get("retry_after_seconds"))
    for value in candidates:
        try:
            return max(0.0, float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _error_message(response: httpx.Response) -> str:
    """Return the API's `error` string when present, else a short HTTP description."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return f"HTTP {response.status_code}"


def _failure_message(detail: str, has_key: bool) -> str:
    message = f"Error: Web search failed - {detail}"
    if not has_key:
        message += "\nSet FIRECRAWL_API_KEY for higher rate limits."
    return message


class FirecrawlWebSearchToolConfig(FunctionBaseConfig, name="firecrawl_web_search"):
    """
    Tool that retrieves relevant contexts from web search (using Firecrawl) for the given question.
    Works without an API key under daily per-IP request and credit limits (HTTP 429 when exceeded). Set a
    FIRECRAWL_API_KEY environment variable or the api_key config for higher limits.
    """

    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of search results to return (1-100)")
    api_key: SecretStr | None = Field(default=None, description="The API key for the Firecrawl service")
    api_url: str | None = Field(
        default=None,
        description=(
            "Base URL of the Firecrawl API. Falls back to the FIRECRAWL_API_URL environment variable, then to "
            "https://api.firecrawl.dev. Set this to use a self-hosted Firecrawl instance."
        ),
    )
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of attempts for the search request")
    scrape_results: bool = Field(
        default=True,
        description=(
            "Whether to scrape each result and return its page content as Markdown. When False, results use "
            "the search description only."
        ),
    )
    max_content_length: int | None = Field(
        default=10000,
        ge=1,
        description=(
            "Max characters per result's text. Truncates each result to reduce token usage. "
            "Set to None to disable truncation."
        ),
    )
    tbs: str | None = Field(default=None, description="Time-based search filter, for example 'qdr:w' for past week")
    country: str | None = Field(
        default=None, description="ISO 3166-1 alpha-2 country code used to localize results, for example 'US'"
    )
    timeout_ms: int = Field(
        default=45000, ge=1000, le=300000, description="Search timeout in milliseconds, sent to Firecrawl"
    )

    @field_validator("api_url")
    @classmethod
    def _api_url_has_scheme(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("api_url must start with http:// or https://")
        return value


@register_function(config_type=FirecrawlWebSearchToolConfig)
async def firecrawl_web_search(
    tool_config: FirecrawlWebSearchToolConfig,
    builder: Builder,
) -> AsyncGenerator[FunctionInfo, None]:
    """Register the Firecrawl web search tool with NAT.

    Calls the Firecrawl Search API (`POST /v2/search`) so agents can search the
    web and read each result's page as Markdown. An API key (via
    `tool_config.api_key` or `FIRECRAWL_API_KEY`) is optional: without one the
    request is sent unauthenticated under a small daily per-IP budget.

    Args:
        tool_config: Configuration controlling result count, retries, page
            scraping, time and country filters, and content truncation.
        builder: NAT builder handle (unused; accepted for interface parity).

    Yields:
        A `FunctionInfo` wrapping the Firecrawl search callable.
    """
    api_key = (
        tool_config.api_key.get_secret_value() if tool_config.api_key else os.environ.get("FIRECRAWL_API_KEY")
    ) or None
    api_key = api_key.strip() if api_key else None
    base_url = (tool_config.api_url or os.environ.get("FIRECRAWL_API_URL") or _DEFAULT_API_URL).rstrip("/")

    global _missing_key_warned
    if not api_key and base_url == _DEFAULT_API_URL and not _missing_key_warned:
        logger.warning(
            "FIRECRAWL_API_KEY not found. Firecrawl web search will run without an API key, which has a small "
            "daily per-IP budget and is meant for trying it out. For regular use, set FIRECRAWL_API_KEY in your "
            "environment or .env file, or specify api_key in your workflow config. Free keys: "
            "https://www.firecrawl.dev/app/api-keys?utm_source=aiq&utm_medium=integration"
        )
        _missing_key_warned = True

    search_url = f"{base_url}/v2/search"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _firecrawl_web_search(question: str) -> str:
        """Retrieves relevant contexts from web search (using Firecrawl) for the given question.
        Each result includes the page content as Markdown.

        Args:
            question (str): The question to be answered. Will be truncated to 400 characters if longer.

        Returns:
            str: The web search results containing relevant documents and their URLs.
        """
        if len(question) > 400:
            question = question[:397] + "..."

        def _truncate_content(content: object) -> str:
            content = "" if content is None else str(content)
            limit = tool_config.max_content_length
            if limit is None or len(content) <= limit:
                return content
            if limit <= 3:
                return content[:limit]
            return content[: limit - 3] + "..."

        def _render(item: dict) -> str:
            markdown = _truncate_content(item.get("markdown"))
            body_text = markdown if markdown else _truncate_content(item.get("description"))
            return _render_document(item.get("url"), item.get("title"), body_text)

        body: dict[str, object] = {
            "query": question,
            "limit": tool_config.max_results,
            "timeout": tool_config.timeout_ms,
            "origin": _ORIGIN,
        }
        if tool_config.scrape_results:
            body["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
        if tool_config.tbs:
            body["tbs"] = tool_config.tbs
        if tool_config.country:
            body["country"] = tool_config.country

        request_timeout = tool_config.timeout_ms / 1000 + 10

        for attempt in range(tool_config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    response = await client.post(search_url, json=body, headers=headers)

                status = response.status_code
                if status == 401:
                    return (
                        "Error: Web search failed due to invalid API key (401 Unauthorized).\n"
                        "Please check your FIRECRAWL_API_KEY and ensure it is valid.\n"
                    )
                if status == 403:
                    return (
                        "Error: Web search failed due to invalid API key or insufficient permissions "
                        "(403 Forbidden).\n"
                        "Please check your FIRECRAWL_API_KEY and ensure it is valid.\n"
                    )
                if status == 402:
                    return (
                        "Error: Firecrawl web search failed because the account is out of credits "
                        "(402 Payment Required)."
                    )
                if status in _RETRYABLE_STATUS_CODES or status >= 500:
                    delay = _retry_after_seconds(response)
                    if delay is not None and delay > _MAX_RETRY_DELAY_SECONDS:
                        # A long wait (for example a daily keyless cap) will not clear within our retries.
                        return _failure_message(_error_message(response), has_key=bool(api_key))
                    raise _RetryableSearchError(_error_message(response), delay)
                if status >= 400:
                    return f"Error: Web search failed - {_error_message(response)}"

                try:
                    payload = response.json()
                except ValueError:
                    return "Error: Web search failed - Firecrawl returned an invalid response"
                data = payload.get("data") if isinstance(payload, dict) else None
                results = data.get("web") if isinstance(data, dict) else None
                if not results:
                    return "Search returned no results"

                documents = [_render(item) for item in results if isinstance(item, dict)]
                return "\n\n---\n\n".join(documents) if documents else "Search returned no results"

            except (_RetryableSearchError, httpx.TransportError) as e:
                if attempt == tool_config.max_retries - 1:
                    detail = str(e) or type(e).__name__
                    if isinstance(e, _RetryableSearchError):
                        return _failure_message(detail, has_key=bool(api_key))
                    return f"Error: Web search failed - {detail}"
                delay = e.retry_after if isinstance(e, _RetryableSearchError) else None
                await asyncio.sleep(min(max(delay or 0, 2**attempt), _MAX_RETRY_DELAY_SECONDS))
            except httpx.InvalidURL as e:
                return f"Error: Web search failed - invalid Firecrawl API URL ({e})"

        return "Error: Search failed after all retries"

    yield FunctionInfo.from_fn(
        _firecrawl_web_search,
        description=_firecrawl_web_search.__doc__,
    )
