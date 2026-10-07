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

"""Tests for the firecrawl_web_search NAT registration."""

import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import httpx
import pytest
from firecrawl_web_search.register import FirecrawlWebSearchToolConfig
from firecrawl_web_search.register import firecrawl_web_search
from pydantic import SecretStr

ADVERSARIAL_URL = 'https://example.com/pa\x00th?q="quoted"&next=<unsafe>&close=</Document>'
ADVERSARIAL_TITLE = 'Research\x00 & "Roadmap" <2026> </title>'
ADVERSARIAL_CONTENT = 'Evidence\x00 & "claims" <external> </Document> </title>'
SANITIZED_URL = 'https://example.com/path?q="quoted"&next=<unsafe>&close=</Document>'
SANITIZED_TITLE = 'Research & "Roadmap" <2026> </title>'
SANITIZED_CONTENT = 'Evidence & "claims" <external> </Document> </title>'

HOSTED_URL = "https://api.firecrawl.dev/v2/search"


def _parse_document(output: str) -> tuple[ET.Element, ET.Element]:
    assert output.count("<Document ") == 1
    assert output.count("</Document>") == 1
    assert output.count("<title>") == 1
    assert output.count("</title>") == 1
    root = ET.fromstring(output)
    title = root.find("title")
    assert title is not None
    return root, title


def _web(*items: dict) -> dict:
    return {"success": True, "data": {"web": list(items)}}


def _item(url="https://a.example", title="Title A", description="Desc A", markdown=None) -> dict:
    item = {"url": url, "title": title, "description": description}
    if markdown is not None:
        item["markdown"] = markdown
    return item


class _Transport:
    """Queue of canned responses served through `httpx.MockTransport`; records every request."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


@pytest.fixture
def transport(monkeypatch):
    """Route every `httpx.AsyncClient` the tool creates through a mock transport (no network)."""
    holder = _Transport(httpx.Response(200, json=_web(_item(markdown="Body A"))))
    real_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(holder.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("firecrawl_web_search.register.httpx.AsyncClient", _client)
    return holder


@pytest.fixture(autouse=True)
def _reset_warn_flag():
    import firecrawl_web_search.register as reg

    reg._missing_key_warned = False
    yield
    reg._missing_key_warned = False


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)


@pytest.fixture
def sleeps(monkeypatch):
    delays: list[float] = []

    async def _sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("firecrawl_web_search.register.asyncio.sleep", _sleep)
    return delays


async def _search(config: FirecrawlWebSearchToolConfig, question: str = "query") -> str:
    async with firecrawl_web_search(config, MagicMock()) as info:
        return await info.single_fn(question)


class TestFirecrawlWebSearchToolConfig:
    def test_defaults(self):
        config = FirecrawlWebSearchToolConfig()
        assert config.max_results == 5
        assert config.api_key is None
        assert config.api_url is None
        assert config.max_retries == 3
        assert config.scrape_results is True
        assert config.max_content_length == 10000
        assert config.tbs is None
        assert config.country is None
        assert config.timeout_ms == 45000

    def test_all_fields(self):
        config = FirecrawlWebSearchToolConfig(
            max_results=10,
            api_key=SecretStr("fc-test"),
            api_url="http://localhost:3002/",
            max_retries=1,
            scrape_results=False,
            max_content_length=50,
            tbs="qdr:w",
            country="US",
            timeout_ms=1000,
        )
        assert config.max_results == 10
        assert config.api_key.get_secret_value() == "fc-test"
        assert config.api_url == "http://localhost:3002/"
        assert config.max_retries == 1
        assert config.scrape_results is False
        assert config.max_content_length == 50
        assert config.tbs == "qdr:w"
        assert config.country == "US"
        assert config.timeout_ms == 1000

    def test_inherits_from_function_base_config(self):
        from nat.data_models.function import FunctionBaseConfig

        assert issubclass(FirecrawlWebSearchToolConfig, FunctionBaseConfig)


class TestFirecrawlWebSearchAuth:
    async def test_works_without_api_key(self, transport):
        out = await _search(FirecrawlWebSearchToolConfig())

        assert len(transport.requests) == 1
        assert str(transport.requests[-1].url) == HOSTED_URL
        assert "authorization" not in transport.requests[-1].headers
        assert "Body A" in out

    async def test_missing_key_is_logged_once(self, transport, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="firecrawl_web_search.register"):
            await _search(FirecrawlWebSearchToolConfig())
            await _search(FirecrawlWebSearchToolConfig())

        messages = [r.getMessage() for r in caplog.records if "FIRECRAWL_API_KEY" in r.getMessage()]
        assert len(messages) == 1
        assert "per-IP" in messages[0]
        assert all(r.levelno == logging.WARNING for r in caplog.records if "FIRECRAWL_API_KEY" in r.getMessage())

    async def test_api_key_from_config_is_used_when_env_absent(self, transport):
        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-from-config")))

        assert transport.requests[-1].headers["authorization"] == "Bearer fc-from-config"
        assert "Body A" in out

    async def test_api_key_whitespace_is_stripped(self, transport, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env\n")
        await _search(FirecrawlWebSearchToolConfig())

        assert transport.requests[-1].headers["authorization"] == "Bearer fc-env"

    async def test_api_key_from_env(self, transport, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env")
        await _search(FirecrawlWebSearchToolConfig())

        assert transport.requests[-1].headers["authorization"] == "Bearer fc-env"

    async def test_custom_api_url_needs_no_key(self, transport):
        out = await _search(FirecrawlWebSearchToolConfig(api_url="http://localhost:3002/"))

        assert str(transport.requests[-1].url) == "http://localhost:3002/v2/search"
        assert "authorization" not in transport.requests[-1].headers
        assert "Body A" in out

    async def test_api_url_from_env(self, transport, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_URL", "https://fc.internal.example")
        await _search(FirecrawlWebSearchToolConfig())

        assert str(transport.requests[-1].url) == "https://fc.internal.example/v2/search"

    async def test_custom_api_url_still_sends_key_when_set(self, transport):
        await _search(FirecrawlWebSearchToolConfig(api_url="http://localhost:3002", api_key=SecretStr("fc-x")))

        assert transport.requests[-1].headers["authorization"] == "Bearer fc-x"


class TestFirecrawlWebSearchRequest:
    async def test_default_request_body(self, transport):
        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_results=3))

        request = transport.requests[-1]
        assert request.method == "POST"
        assert str(request.url) == HOSTED_URL
        assert request.headers["content-type"] == "application/json"
        assert transport.body == {
            "query": "query",
            "limit": 3,
            "timeout": 45000,
            "origin": "aiq",
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }

    async def test_scrape_results_false_omits_scrape_options(self, transport):
        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), scrape_results=False))

        assert "scrapeOptions" not in transport.body

    async def test_tbs_and_country_only_when_set(self, transport):
        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))
        assert "tbs" not in transport.body
        assert "country" not in transport.body

        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), tbs="qdr:w", country="US"))
        assert transport.body["tbs"] == "qdr:w"
        assert transport.body["country"] == "US"

    async def test_no_unsupported_keys(self, transport):
        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), tbs="qdr:d", country="GB"))

        for key in ("integration", "highlights", "sources", "categories"):
            assert key not in transport.body

    @pytest.mark.parametrize("limit", [1, 25, 100])
    async def test_limit_is_sent(self, transport, limit):
        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_results=limit))

        assert transport.body["limit"] == limit

    async def test_truncates_long_query(self, transport):
        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")), "x" * 500)

        assert len(transport.body["query"]) == 400
        assert transport.body["query"].endswith("...")


class TestFirecrawlWebSearchResults:
    async def test_structurally_escapes_provider_fields(self, transport):
        transport.responses = [
            httpx.Response(
                200,
                json=_web(_item(ADVERSARIAL_URL, ADVERSARIAL_TITLE, "unused", markdown=ADVERSARIAL_CONTENT)),
            )
        ]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        document, title = _parse_document(out)
        assert document.attrib["href"] == SANITIZED_URL
        assert (title.text or "").strip("\n") == SANITIZED_TITLE
        assert (title.tail or "").strip("\n") == SANITIZED_CONTENT

    async def test_multiple_results_are_separated(self, transport):
        transport.responses = [
            httpx.Response(
                200,
                json=_web(
                    _item("https://a.example", "Title A", markdown="Body A"),
                    _item("https://b.example", "Title B", markdown="Body B"),
                ),
            )
        ]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_results=2))

        assert out.count("<Document ") == 2
        assert "\n\n---\n\n" in out
        assert "Title A" in out and "Title B" in out

    async def test_markdown_preferred_over_description(self, transport):
        transport.responses = [httpx.Response(200, json=_web(_item(description="Short desc", markdown="Full page")))]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert "Full page" in out
        assert "Short desc" not in out

    async def test_description_is_fallback(self, transport):
        transport.responses = [httpx.Response(200, json=_web(_item(description="Short desc", markdown="")))]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert "Short desc" in out

    async def test_description_used_when_not_scraping(self, transport):
        transport.responses = [httpx.Response(200, json=_web(_item(description="Short desc")))]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), scrape_results=False))

        assert "Short desc" in out

    async def test_truncates_content(self, transport):
        transport.responses = [httpx.Response(200, json=_web(_item(markdown="abcdefghijklmnop")))]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_content_length=8))

        _, title = _parse_document(out)
        assert (title.tail or "").strip("\n") == "abcde..."
        assert "abcdefghi" not in out

    async def test_none_disables_truncation(self, transport):
        text = "z" * 20000
        transport.responses = [httpx.Response(200, json=_web(_item(markdown=text)))]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_content_length=None))

        assert text in out

    @pytest.mark.parametrize(("limit", "expected"), [(1, "a"), (2, "ab"), (3, "abc"), (4, "a...")])
    async def test_tiny_content_limits_never_exceed_cap(self, transport, limit, expected):
        transport.responses = [httpx.Response(200, json=_web(_item(markdown="abcdefghij")))]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_content_length=limit))

        _, title = _parse_document(out)
        assert (title.tail or "").strip("\n") == expected

    def test_config_bounds(self):
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(max_retries=0)
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(max_content_length=0)
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(timeout_ms=999)
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(timeout_ms=300001)
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(max_results=0)
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(max_results=101)
        with pytest.raises(ValueError):
            FirecrawlWebSearchToolConfig(api_url="localhost:3002")

    async def test_invalid_json_is_not_retried(self, transport, sleeps):
        transport.responses = [httpx.Response(200, content=b"not json")]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert out.startswith("Error: Web search failed")
        assert len(transport.requests) == 1

    @pytest.mark.parametrize("payload", [_web(), _web("not-a-dict"), {"success": True, "data": {}}, {"success": True}])
    async def test_empty_results(self, transport, payload):
        transport.responses = [httpx.Response(200, json=payload)]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert out == "Search returned no results"


class TestFirecrawlWebSearchErrors:
    @pytest.mark.parametrize("status", [401, 403])
    async def test_invalid_key_is_not_retried(self, transport, sleeps, status):
        transport.responses = [httpx.Response(status, json={"success": False, "error": "Unauthorized"})]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert str(status) in out
        assert "invalid API key" in out
        assert "FIRECRAWL_API_KEY" in out
        assert len(transport.requests) == 1
        assert sleeps == []

    async def test_out_of_credits_is_not_retried(self, transport, sleeps):
        transport.responses = [httpx.Response(402, json={"success": False, "error": "Payment required"})]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert "out of credits" in out
        assert "402" in out
        assert len(transport.requests) == 1
        assert sleeps == []

    async def test_bad_request_returns_api_error_without_retry(self, transport, sleeps):
        transport.responses = [httpx.Response(400, json={"success": False, "error": "Invalid tbs value"})]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert out == "Error: Web search failed - Invalid tbs value"
        assert len(transport.requests) == 1
        assert sleeps == []

    async def test_rate_limit_then_success(self, transport, sleeps):
        transport.responses = [
            httpx.Response(429, json={"success": False, "error": "Rate limit exceeded"}),
            httpx.Response(200, json=_web(_item(markdown="ok body"))),
        ]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert "ok body" in out
        assert len(transport.requests) == 2
        assert sleeps == [1]

    async def test_retry_after_header_is_honored(self, transport, sleeps):
        transport.responses = [
            httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "Rate limit exceeded"}),
            httpx.Response(200, json=_web(_item(markdown="ok body"))),
        ]

        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert sleeps == [7]

    async def test_keyless_rate_limit_hints_at_api_key(self, transport, sleeps):
        transport.responses = [httpx.Response(429, json={"error": "Rate limit exceeded"})]

        out = await _search(FirecrawlWebSearchToolConfig(max_retries=1))

        assert "Rate limit exceeded" in out
        assert "FIRECRAWL_API_KEY" in out

    async def test_daily_cap_is_not_retried(self, transport, sleeps):
        body = {
            "success": False,
            "error": "You've hit Firecrawl's keyless free tier rate limit.",
            "reason": "credits",
            "retry_after_seconds": 84289,
        }
        transport.responses = [httpx.Response(429, json=body)]

        out = await _search(FirecrawlWebSearchToolConfig())

        assert "keyless free tier rate limit" in out
        assert "FIRECRAWL_API_KEY" in out
        assert len(transport.requests) == 1
        assert sleeps == []

    async def test_short_body_retry_after_is_honored(self, transport, sleeps):
        transport.responses = [
            httpx.Response(429, json={"error": "slow down", "retry_after_seconds": 5}),
            httpx.Response(200, json=_web(_item(markdown="ok body"))),
        ]

        await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert sleeps == [5]

    async def test_keyed_rate_limit_has_no_key_hint(self, transport, sleeps):
        transport.responses = [httpx.Response(429, json={"error": "Rate limit exceeded"})]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_retries=1))

        assert out == "Error: Web search failed - Rate limit exceeded"

    async def test_server_error_exhausts_retries(self, transport, sleeps):
        transport.responses = [httpx.Response(503, json={"success": False, "error": "Unavailable"})]

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_retries=3))

        assert out == "Error: Web search failed - Unavailable"
        assert len(transport.requests) == 3
        assert sleeps == [1, 2]

    async def test_timeout_is_retried(self, monkeypatch, sleeps):
        calls = []

        def _handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(200, json=_web(_item(markdown="after timeout")))

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            "firecrawl_web_search.register.httpx.AsyncClient",
            lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(_handler)}),
        )

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test")))

        assert "after timeout" in out
        assert len(calls) == 2

    async def test_connection_error_returns_message(self, monkeypatch, sleeps):
        def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            "firecrawl_web_search.register.httpx.AsyncClient",
            lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(_handler)}),
        )

        out = await _search(FirecrawlWebSearchToolConfig(api_key=SecretStr("fc-test"), max_retries=2))

        assert out == "Error: Web search failed - connection refused"
        assert sleeps == [1]
