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

"""Shared fixtures and output-contract helpers for context_dev_search tests.

The HTTP layer is faked with ``httpx.MockTransport`` at the tools' client
factory, so every test exercises real request building, response parsing,
retries, and rendering without network access. The output-contract helpers
define the tools' structural success criteria once, so the recorded-response
replay tests and the key-gated live integration test certify exactly the same
criteria. Everything is exposed as pytest fixtures (not module imports) so the
test modules work under any pytest import mode.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
API_KEY_ENV_VAR = "CONTEXT_DEV_API_KEY"  # pragma: allowlist secret
PLACEHOLDER_KEY = "ctx-test-key-not-real"  # pragma: allowlist secret

# Whole-block extraction rather than splitting on the "\n\n---\n\n" joiner:
# passages are Markdown and can contain that byte sequence (horizontal rules).
# Extracting <Document ...>...</Document> spans is unambiguous because the
# renderer html-escapes all interior fields, so no raw '<' or '>' can appear
# inside a block.
_DOCUMENT_BLOCK_RE = re.compile(r'<Document href="[^"]*">.*?</Document>', re.DOTALL)
_DOCUMENT_HEAD_RE = re.compile(r'<Document href="(?P<href>[^"]*)">\n<title>\n(?P<title>.*?)\n</title>\n', re.DOTALL)
_PAPER_ENTRY_RE = re.compile(
    r"^(?P<index>\d+)\. \*\*(?P<title>[^\n]+?)\*\*\n"
    r"   - \*\*Source\*\*: (?P<source>[^\n]*)\n"
    r"   - \*\*Snippet\*\*: (?P<snippet>[^\n]*)\n"
    r"   - \*\*Link\*\*: (?P<link>[^\n]+)$",
    re.MULTILINE,
)


class MockSearchAPI:
    """A scripted stand-in for ``POST /web/search``.

    Queue responses with ``add``; each request consumes the next one (the last
    response repeats once the queue is exhausted). Every request is recorded in
    ``requests`` with its decoded JSON body.
    """

    def __init__(self) -> None:
        self._responses: list[httpx.Response | Exception] = []
        self.requests: list[httpx.Request] = []

    def add(self, response: httpx.Response | Exception) -> "MockSearchAPI":
        self._responses.append(response)
        return self

    def add_results(self, results: list[dict], **extra) -> "MockSearchAPI":
        return self.add(httpx.Response(200, json={"results": results, "request_id": "req-ok", **extra}))

    def add_error(self, status: int, error_code: str | None = None, message: str = "", **headers) -> "MockSearchAPI":
        body = {"message": message or f"HTTP {status}", "request_id": f"req-{status}"}
        if error_code:
            body["error_code"] = error_code
        return self.add(httpx.Response(status, json=body, headers=headers))

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("MockSearchAPI received a request with no scripted response")
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def make_result(
    url: str = "https://example.com/page",
    title: str = "Example page",
    description: str = "An example snippet.",
    highlights: list[str] | None = None,
) -> dict:
    """Build one raw API result in the documented response shape."""
    return {
        "url": url,
        "title": title,
        "description": description,
        "relevance": "high",
        "markdown": {"markdown": None, "code": "NOT_REQUESTED"},
        "highlights": {
            "highlights": highlights,
            "code": "SUCCESS" if highlights is not None else "NOT_REQUESTED",
        },
    }


def _extract_documents(result: str) -> list[str]:
    return _DOCUMENT_BLOCK_RE.findall(result)


def _assert_web_contract(result: str, max_results: int) -> list[str]:
    """Assert the web search tool's structural output contract; return the blocks.

    1. ``result`` is a non-empty string that is neither an ``Error:`` message
       nor the ``Search returned no results`` sentinel.
    2. It contains between 1 and ``max_results`` ``<Document>`` blocks.
    3. Every block has a non-empty http(s) ``href`` and a non-empty title.
    4. Every block parses as XML (all interior fields are html-escaped).
    5. At least one block has non-empty body content.
    """
    assert isinstance(result, str) and result.strip(), "empty result"
    assert not result.startswith("Error:"), f"provider returned an error: {result[:200]}"
    assert result != "Search returned no results"

    blocks = _extract_documents(result)
    assert 1 <= len(blocks) <= max_results, f"expected 1..{max_results} <Document> blocks, got {len(blocks)}"

    bodies = []
    for block in blocks:
        head = _DOCUMENT_HEAD_RE.match(block)
        assert head is not None, f"malformed <Document> head: {block[:120]!r}"
        assert head.group("href").startswith(("http://", "https://")), f"non-http(s) href: {head.group('href')!r}"
        assert head.group("title").strip(), "empty title"
        element = ET.fromstring(block)
        body = (element.text or "") + "".join(child.tail or "" for child in element)
        bodies.append(body.strip())
    assert any(bodies), "every block had an empty body"
    return blocks


def _assert_paper_contract(result: str, max_results: int) -> list[re.Match]:
    """Assert the paper search tool's structural output contract; return the entries.

    1. ``result`` is a non-empty string that is neither an ``Error:`` message
       nor the ``Search returned no results`` sentinel.
    2. It contains between 1 and ``max_results`` numbered entries, numbered 1..N.
    3. Every entry has a non-empty title and an http(s) link.
    4. Entries account for the whole output (no stray lines that could confuse
       the citation parser).
    5. At least one entry has a non-empty snippet.
    """
    assert isinstance(result, str) and result.strip(), "empty result"
    assert not result.startswith("Error:"), f"provider returned an error: {result[:200]}"
    assert result != "Search returned no results"

    entries = list(_PAPER_ENTRY_RE.finditer(result))
    assert 1 <= len(entries) <= max_results, f"expected 1..{max_results} papers, got {len(entries)}"
    assert [int(entry.group("index")) for entry in entries] == list(range(1, len(entries) + 1))
    for entry in entries:
        assert entry.group("title").strip(), "empty title"
        assert entry.group("link").startswith(("http://", "https://")), f"non-http(s) link: {entry.group('link')!r}"
    assert "\n\n".join(entry.group(0) for entry in entries) == result, "output has lines outside paper entries"
    assert any(entry.group("snippet").strip() for entry in entries), "every paper had an empty snippet"
    return entries


@pytest.fixture
def mock_api(monkeypatch) -> MockSearchAPI:
    """Route the tools' HTTP client to a scripted ``MockSearchAPI``."""
    api = MockSearchAPI()

    def _factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(api.handler))

    monkeypatch.setattr("context_dev_search.register._new_http_client", _factory)
    return api


@pytest.fixture
def api_key_env(monkeypatch) -> str:
    """Provide a placeholder key through the environment."""
    monkeypatch.setenv(API_KEY_ENV_VAR, PLACEHOLDER_KEY)
    return PLACEHOLDER_KEY


@pytest.fixture
def no_api_key_env(monkeypatch) -> None:
    """Guarantee no key is visible through the environment."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)


@pytest.fixture
def no_sleep(monkeypatch) -> AsyncMock:
    """Make retry backoff instant and record the requested delays."""
    sleep = AsyncMock()
    monkeypatch.setattr("context_dev_search.client.asyncio.sleep", sleep)
    return sleep


@pytest.fixture
def result_factory():
    """Factory for raw API result objects."""
    return make_result


@pytest.fixture
def assert_web_contract():
    """Structural output contract for ``context_dev_web_search``."""
    return _assert_web_contract


@pytest.fixture
def assert_paper_contract():
    """Structural output contract for ``context_dev_paper_search``."""
    return _assert_paper_contract


@pytest.fixture
def extract_documents():
    """Whole-block ``<Document>`` extraction helper."""
    return _extract_documents


@pytest.fixture
def load_recorded_fixture():
    """Load a recorded API response fixture by file name."""

    def _load(name: str) -> dict:
        with (FIXTURES_DIR / name).open(encoding="utf-8") as fh:
            return json.load(fh)

    return _load
