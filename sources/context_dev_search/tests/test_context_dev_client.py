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

"""Tests for the NAT-independent Context.dev Search API client."""

import asyncio
import json
import logging

import httpx
import pytest
from context_dev_search.client import CONTEXT_DEV_API_URL
from context_dev_search.client import MAX_BACKOFF_SECONDS
from context_dev_search.client import MAX_QUERY_LENGTH
from context_dev_search.client import ContextDevAPIError
from context_dev_search.client import ContextDevSearchClient
from context_dev_search.client import SearchResult
from context_dev_search.client import search_with_retries
from context_dev_search.client import truncate_query

KEY = "ctx-test-key-not-real"  # pragma: allowlist secret


@pytest.fixture
async def make_client(mock_api):
    """Build a client wired to the scripted API; closes its HTTP client afterwards."""
    clients: list[httpx.AsyncClient] = []

    def _make(**kwargs) -> ContextDevSearchClient:
        http = httpx.AsyncClient(transport=httpx.MockTransport(mock_api.handler))
        clients.append(http)
        return ContextDevSearchClient(KEY, http_client=http, **kwargs)

    yield _make
    for http in clients:
        await http.aclose()


class TestRequest:
    async def test_minimal_request(self, make_client, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await make_client().search("cuda graphs", num_results=5, include_highlights=False)

        request = mock_api.requests[0]
        assert request.method == "POST"
        assert str(request.url) == f"{CONTEXT_DEV_API_URL}/web/search"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.headers["content-type"] == "application/json"
        assert mock_api.bodies[0] == {
            "query": "cuda graphs",
            "numResults": 5,
            "timeoutOpts": {"milliseconds": 29000, "behavior": "return-partial"},
        }

    async def test_all_options_map_to_api_fields(self, make_client, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await make_client().search(
            "cuda graphs",
            num_results=7,
            include_highlights=True,
            max_content_length=1200,
            include_domains=["nvidia.com"],
            exclude_domains=["pinterest.com"],
            country="GB",
            freshness="last_month",
            zero_data_retention=True,
        )

        body = mock_api.bodies[0]
        assert body["numResults"] == 7
        assert body["highlightsOptions"] == {"enabled": True, "maxCharacters": 1200}
        assert body["descriptionMaxCharacters"] == 1200
        assert body["includeDomains"] == ["nvidia.com"]
        assert body["excludeDomains"] == ["pinterest.com"]
        assert body["country"] == "gb"
        assert body["freshness"] == "last_month"
        assert body["zdr"] == "enabled"

    async def test_highlights_without_length_cap(self, make_client, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await make_client().search("q", num_results=1)
        assert mock_api.bodies[0]["highlightsOptions"] == {"enabled": True}
        assert "descriptionMaxCharacters" not in mock_api.bodies[0]

    async def test_base_url_trailing_slash(self, make_client, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await make_client(base_url="https://proxy.example.com/v1/").search("q", num_results=1)
        assert str(mock_api.requests[0].url) == "https://proxy.example.com/v1/web/search"

    async def test_long_query_is_truncated(self, make_client, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await make_client().search("x" * 2000, num_results=1)
        sent = mock_api.bodies[0]["query"]
        assert len(sent) == MAX_QUERY_LENGTH
        assert sent.endswith("...")

    @pytest.mark.parametrize(
        ("timeout", "expected"),
        [
            (30.0, {"milliseconds": 29000, "behavior": "return-partial"}),
            (6.0, {"milliseconds": 5000, "behavior": "return-partial"}),
            (5.0, {"milliseconds": 4000, "behavior": "fail"}),
            (2.0, {"milliseconds": 1000, "behavior": "fail"}),
        ],
    )
    async def test_server_deadline_leaves_headroom(self, make_client, mock_api, result_factory, timeout, expected):
        mock_api.add_results([result_factory()])
        await make_client(timeout=timeout).search("q", num_results=1)
        assert mock_api.bodies[0]["timeoutOpts"] == expected


class TestTruncateQuery:
    def test_short_query_unchanged(self):
        assert truncate_query("hello") == "hello"

    def test_boundary_length_unchanged(self):
        query = "y" * MAX_QUERY_LENGTH
        assert truncate_query(query) == query

    def test_counts_astral_characters_as_two_utf16_units(self):
        query = "\U0001f600" * 260  # 260 code points, 520 UTF-16 code units
        truncated = truncate_query(query)
        assert truncated.endswith("...")
        assert len(truncated.encode("utf-16-le")) // 2 <= MAX_QUERY_LENGTH
        assert truncated[:-3] == "\U0001f600" * 248

    def test_replaces_lone_surrogates(self):
        truncated = truncate_query("broken \ud83d escape")
        assert truncated == "broken \ufffd escape"
        truncated.encode("utf-8")

    async def test_lone_surrogate_query_is_sent(self, make_client, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await make_client().search("abc\ud83d", num_results=1)
        assert mock_api.bodies[0]["query"] == "abc\ufffd"

    def test_never_splits_a_surrogate_pair(self):
        query = "a" + "\U0001f600" * 300
        truncated = truncate_query(query)
        assert len(truncated.encode("utf-16-le")) // 2 <= MAX_QUERY_LENGTH
        truncated.encode("utf-8")  # raises if a lone surrogate were left behind


class TestResponseParsing:
    async def test_parses_results_and_highlights(self, make_client, mock_api, result_factory):
        mock_api.add_results(
            [
                result_factory(url="https://a.example/1", title="A", description="snip", highlights=["p1", "  ", "p2"]),
                result_factory(url="https://b.example/2", title="B", highlights=None),
            ]
        )
        results = await make_client().search("q", num_results=2)
        assert results == [
            SearchResult(url="https://a.example/1", title="A", description="snip", highlights=["p1", "p2"]),
            SearchResult(url="https://b.example/2", title="B", description="An example snippet.", highlights=[]),
        ]

    async def test_skips_malformed_items(self, make_client, mock_api, result_factory):
        mock_api.add_results(
            [
                "not-an-object",
                {"title": "missing url"},
                {"url": "", "title": "empty url"},
                {"url": "https://ok.example", "title": None, "description": None, "highlights": "bad"},
            ]
        )
        results = await make_client().search("q", num_results=4)
        assert results == [SearchResult(url="https://ok.example", title="", description="", highlights=[])]

    async def test_empty_results(self, make_client, mock_api):
        mock_api.add_results([])
        assert await make_client().search("q", num_results=3) == []

    async def test_invalid_json_body_is_transient(self, make_client, mock_api):
        mock_api.add(httpx.Response(200, content=b"<html>gateway</html>"))
        with pytest.raises(ContextDevAPIError, match="not valid JSON") as excinfo:
            await make_client().search("q", num_results=1)
        assert excinfo.value.status_code is None
        assert excinfo.value.retryable is True

    async def test_missing_results_list(self, make_client, mock_api):
        mock_api.add(httpx.Response(200, json={"query": "q"}))
        with pytest.raises(ContextDevAPIError, match="results list"):
            await make_client().search("q", num_results=1)

    async def test_partial_results_are_returned_and_logged(self, make_client, mock_api, result_factory, caplog):
        mock_api.add_results([result_factory()], partial=True)
        with caplog.at_level(logging.INFO, logger="context_dev_search.client"):
            results = await make_client().search("q", num_results=5)
        assert len(results) == 1
        assert any("partial results" in record.message for record in caplog.records)


class TestErrors:
    async def test_error_body_is_parsed(self, make_client, mock_api):
        mock_api.add_error(402, "USAGE_EXCEEDED", "Out of credits")
        with pytest.raises(ContextDevAPIError) as excinfo:
            await make_client().search("q", num_results=1)
        error = excinfo.value
        assert (error.status_code, error.error_code, error.message, error.request_id) == (
            402,
            "USAGE_EXCEEDED",
            "Out of credits",
            "req-402",
        )
        assert error.retryable is False

    async def test_non_json_error_uses_reason_phrase_and_header_request_id(self, make_client, mock_api):
        mock_api.add(httpx.Response(502, content=b"bad gateway", headers={"x-request-id": "hdr-1"}))
        with pytest.raises(ContextDevAPIError) as excinfo:
            await make_client().search("q", num_results=1)
        assert excinfo.value.status_code == 502
        assert excinfo.value.message == "Bad Gateway"
        assert excinfo.value.request_id == "hdr-1"
        assert excinfo.value.retryable is True

    @pytest.mark.parametrize(("header", "expected"), [("7", 7.0), ("0", 0.0), ("-1", None), ("soon", None)])
    async def test_retry_after_header(self, make_client, mock_api, header, expected):
        mock_api.add_error(429, "RATE_LIMITED", **{"retry-after": header})
        with pytest.raises(ContextDevAPIError) as excinfo:
            await make_client().search("q", num_results=1)
        assert excinfo.value.retry_after == expected

    async def test_redirects_are_reported_not_followed(self, make_client, mock_api):
        mock_api.add(httpx.Response(301, headers={"location": "https://api.context.dev/v1/web/search/"}))
        with pytest.raises(ContextDevAPIError) as excinfo:
            await make_client().search("q", num_results=1)
        assert excinfo.value.status_code == 301
        assert excinfo.value.retryable is False
        assert "unexpected redirect to https://api.context.dev/v1/web/search/" in excinfo.value.message
        assert len(mock_api.requests) == 1

    async def test_network_error_is_wrapped(self, make_client, mock_api):
        mock_api.add(httpx.ConnectError("connection refused"))
        with pytest.raises(ContextDevAPIError, match="network error \\(ConnectError\\)") as excinfo:
            await make_client().search("q", num_results=1)
        assert excinfo.value.status_code is None
        assert excinfo.value.retryable is True

    async def test_timeout_is_wrapped(self, make_client, mock_api):
        mock_api.add(httpx.ReadTimeout("slow"))
        with pytest.raises(ContextDevAPIError, match="timed out after 30s"):
            await make_client().search("q", num_results=1)

    async def test_whole_request_is_bounded_by_timeout(self):
        async def _slow_handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(_slow_handler)) as http:
            client = ContextDevSearchClient(KEY, http_client=http, timeout=0.05)
            with pytest.raises(ContextDevAPIError, match="timed out") as excinfo:
                await client.search("q", num_results=1)
        assert excinfo.value.retryable is True

    @pytest.mark.parametrize(("status", "retryable"), [(400, False), (401, False), (403, False), (408, True)])
    async def test_retryable_classification(self, make_client, mock_api, status, retryable):
        mock_api.add_error(status)
        with pytest.raises(ContextDevAPIError) as excinfo:
            await make_client().search("q", num_results=1)
        assert excinfo.value.retryable is retryable

    async def test_api_key_never_appears_in_errors(self, make_client, mock_api):
        mock_api.add_error(401, message="Unauthorized")
        with pytest.raises(ContextDevAPIError) as excinfo:
            await make_client().search("q", num_results=1)
        assert KEY not in str(excinfo.value)
        assert KEY not in json.dumps(vars(excinfo.value), default=str)


class TestSearchWithRetries:
    async def test_retries_transient_errors_then_succeeds(self, make_client, mock_api, no_sleep, result_factory):
        mock_api.add_error(503).add(httpx.ConnectError("reset")).add_results([result_factory()])
        results = await search_with_retries(make_client(), "q", max_retries=3, num_results=1)
        assert len(results) == 1
        assert len(mock_api.requests) == 3
        assert [call.args[0] for call in no_sleep.await_args_list] == [1.0, 2.0]

    async def test_honors_retry_after(self, make_client, mock_api, no_sleep, result_factory):
        mock_api.add_error(429, **{"retry-after": "3"}).add_error(429, **{"retry-after": "0"})
        mock_api.add_results([result_factory()])
        await search_with_retries(make_client(), "q", max_retries=3, num_results=1)
        # A longer Retry-After replaces the backoff; a shorter one never undercuts it.
        assert [call.args[0] for call in no_sleep.await_args_list] == [3.0, 2.0]

    async def test_retry_after_beyond_cap_fails_fast(self, make_client, mock_api, no_sleep):
        mock_api.add_error(429, "RATE_LIMITED", **{"retry-after": "600"})
        with pytest.raises(ContextDevAPIError) as excinfo:
            await search_with_retries(make_client(), "q", max_retries=3, num_results=1)
        assert excinfo.value.status_code == 429
        assert excinfo.value.attempts == 1
        assert len(mock_api.requests) == 1
        no_sleep.assert_not_awaited()

    async def test_backoff_is_capped(self, make_client, mock_api, no_sleep):
        mock_api.add_error(500)
        with pytest.raises(ContextDevAPIError):
            await search_with_retries(make_client(), "q", max_retries=8, num_results=1)
        delays = [call.args[0] for call in no_sleep.await_args_list]
        assert len(delays) == 7
        assert max(delays) == MAX_BACKOFF_SECONDS

    @pytest.mark.parametrize("status", [400, 401, 402, 403])
    async def test_non_retryable_errors_fail_fast(self, make_client, mock_api, no_sleep, status):
        mock_api.add_error(status)
        with pytest.raises(ContextDevAPIError) as excinfo:
            await search_with_retries(make_client(), "q", max_retries=3, num_results=1)
        assert excinfo.value.status_code == status
        assert len(mock_api.requests) == 1
        no_sleep.assert_not_awaited()

    async def test_raises_last_error_when_exhausted(self, make_client, mock_api, no_sleep):
        mock_api.add_error(503)
        with pytest.raises(ContextDevAPIError) as excinfo:
            await search_with_retries(make_client(), "q", max_retries=2, num_results=1)
        assert excinfo.value.status_code == 503
        assert excinfo.value.attempts == 2
        assert len(mock_api.requests) == 2

    async def test_single_attempt_never_sleeps(self, make_client, mock_api, no_sleep):
        mock_api.add_error(503)
        with pytest.raises(ContextDevAPIError):
            await search_with_retries(make_client(), "q", max_retries=1, num_results=1)
        no_sleep.assert_not_awaited()
