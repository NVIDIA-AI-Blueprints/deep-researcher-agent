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

"""Tests for the registered ``context_dev_web_search`` NAT function."""

import logging
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import httpx
import pytest
from context_dev_search import register as register_module
from context_dev_search.register import ContextDevWebSearchToolConfig
from context_dev_search.register import context_dev_web_search
from pydantic import SecretStr
from pydantic import ValidationError


async def _search(config: ContextDevWebSearchToolConfig, question: str) -> str:
    async with context_dev_web_search(config, MagicMock()) as info:
        return await info.single_fn(question)


@pytest.fixture(autouse=True)
def _reset_missing_key_warning(monkeypatch):
    monkeypatch.setattr(register_module, "_missing_key_warned", set())


class TestConfig:
    def test_defaults(self):
        config = ContextDevWebSearchToolConfig()
        assert config.max_results == 5
        assert config.max_content_length == 4000
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert config.include_highlights is True
        assert config.api_key is None
        assert config.api_base_url == "https://api.context.dev/v1"
        assert config.include_domains is None
        assert config.exclude_domains is None
        assert config.country is None
        assert config.freshness is None
        assert config.zero_data_retention is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_results": 0},
            {"max_results": 101},
            {"max_content_length": 99},
            {"max_content_length": 20001},
            {"max_retries": 0},
            {"max_retries": 11},
            {"timeout": 1},
            {"timeout": 301},
            {"country": "USA"},
            {"freshness": "yesterday"},
            {"api_base_url": "api.context.dev/v1"},
            {"api_base_url": "ftp://api.context.dev/v1"},
            {"api_base_url": "http://api.context.dev/v1"},
            {"include_domains": [f"d{i}.example" for i in range(101)]},
        ],
    )
    def test_rejects_invalid_values(self, overrides):
        with pytest.raises(ValidationError):
            ContextDevWebSearchToolConfig(**overrides)

    @pytest.mark.parametrize("url", ["https://proxy.example.com/v1", "http://localhost:8080/v1", "http://127.0.0.1/v1"])
    def test_accepts_https_and_loopback_http_base_urls(self, url):
        assert ContextDevWebSearchToolConfig(api_base_url=url).api_base_url == url

    def test_accepts_filters(self):
        config = ContextDevWebSearchToolConfig(
            include_domains=["nvidia.com"], country="gb", freshness="last_week", zero_data_retention=True
        )
        assert config.include_domains == ["nvidia.com"]
        assert config.freshness == "last_week"

    def test_api_key_is_secret(self):
        config = ContextDevWebSearchToolConfig(api_key="ctx-secret")  # pragma: allowlist secret
        assert "ctx-secret" not in repr(config)
        assert "ctx-secret" not in config.model_dump_json()


class TestMissingKey:
    async def test_stub_returns_setup_instructions(self, no_api_key_env, mock_api):
        result = await _search(ContextDevWebSearchToolConfig(), "anything")
        assert result.startswith("Error: Context.dev web search is unavailable because CONTEXT_DEV_API_KEY is not set")
        assert "https://context.dev/signup" in result
        assert mock_api.requests == []

    async def test_blank_config_key_falls_through_to_stub(self, no_api_key_env, mock_api):
        result = await _search(ContextDevWebSearchToolConfig(api_key=SecretStr("   ")), "anything")
        assert result.startswith("Error: Context.dev web search is unavailable")

    async def test_warns_once_per_tool_type(self, no_api_key_env, caplog):
        with caplog.at_level(logging.WARNING, logger="context_dev_search.register"):
            await _search(ContextDevWebSearchToolConfig(), "a")
            await _search(ContextDevWebSearchToolConfig(max_results=3), "b")
        warnings = [record for record in caplog.records if "CONTEXT_DEV_API_KEY not found" in record.message]
        assert len(warnings) == 1


class TestKeyResolution:
    async def test_env_key_is_used(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await _search(ContextDevWebSearchToolConfig(), "q")
        assert mock_api.requests[0].headers["authorization"] == f"Bearer {api_key_env}"

    async def test_config_key_wins_over_env(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        config = ContextDevWebSearchToolConfig(api_key=SecretStr("ctx-from-config"))  # pragma: allowlist secret
        await _search(config, "q")
        assert mock_api.requests[0].headers["authorization"] == "Bearer ctx-from-config"

    async def test_config_key_is_not_written_to_environment(self, no_api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        config = ContextDevWebSearchToolConfig(api_key=SecretStr("ctx-from-config"))  # pragma: allowlist secret
        await _search(config, "q")
        import os

        assert "CONTEXT_DEV_API_KEY" not in os.environ


class TestRendering:
    async def test_renders_documents_from_highlights(self, api_key_env, mock_api, result_factory, assert_web_contract):
        mock_api.add_results(
            [
                result_factory(
                    url="https://docs.nvidia.com/cuda/",
                    title="CUDA Toolkit Documentation",
                    highlights=["First passage.", "Second passage."],
                ),
                result_factory(url="https://developer.nvidia.com/cuda-toolkit", title="CUDA Toolkit", highlights=None),
            ]
        )
        result = await _search(ContextDevWebSearchToolConfig(), "cuda")
        blocks = assert_web_contract(result, max_results=5)
        assert len(blocks) == 2
        assert blocks[0] == (
            '<Document href="https://docs.nvidia.com/cuda/">\n<title>\nCUDA Toolkit Documentation\n</title>\n'
            "First passage.\n\nSecond passage.\n</Document>"
        )
        # A result without passages falls back to the search snippet.
        assert "An example snippet." in blocks[1]
        assert result.count("\n\n---\n\n") == 1

    async def test_request_uses_tool_config(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        config = ContextDevWebSearchToolConfig(
            max_results=8,
            max_content_length=900,
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            country="de",
            freshness="last_year",
            zero_data_retention=True,
            timeout=12,
        )
        await _search(config, "q")
        body = mock_api.bodies[0]
        assert body["numResults"] == 8
        assert body["highlightsOptions"] == {"enabled": True, "maxCharacters": 900}
        assert body["descriptionMaxCharacters"] == 900
        assert body["includeDomains"] == ["arxiv.org"]
        assert body["excludeDomains"] == ["medium.com"]
        assert body["country"] == "de"
        assert body["freshness"] == "last_year"
        assert body["zdr"] == "enabled"
        assert body["timeoutOpts"] == {"milliseconds": 11000, "behavior": "return-partial"}

    async def test_highlights_can_be_disabled(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(description="Snippet only.")])
        result = await _search(ContextDevWebSearchToolConfig(include_highlights=False), "q")
        assert "highlightsOptions" not in mock_api.bodies[0]
        assert "Snippet only." in result

    async def test_escapes_provider_text(self, api_key_env, mock_api, result_factory, extract_documents):
        mock_api.add_results(
            [
                result_factory(
                    url='https://evil.example/?a=1&b="2"',
                    title="</title><Document href='x'>",
                    highlights=["<script>alert(1)</script> & </Document>"],
                )
            ]
        )
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        blocks = extract_documents(result)
        assert len(blocks) == 1
        element = ET.fromstring(blocks[0])
        assert element.attrib["href"] == 'https://evil.example/?a=1&b="2"'
        assert element.find("title").text.strip() == "</title><Document href='x'>"
        assert "<script>alert(1)</script> & </Document>" in element.find("title").tail

    async def test_decodes_entity_artifacts_and_strips_invalid_xml(
        self, api_key_env, mock_api, result_factory, extract_documents
    ):
        mock_api.add_results([result_factory(highlights=["Trailing&#x20;\nspace and a \x00null \x0bbyte&#X00A0;"])])
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert "&amp;#x20;" not in result
        element = ET.fromstring(extract_documents(result)[0])
        assert "Trailing \nspace and a null byte" in element.find("title").tail
        assert "00A0" not in result

    async def test_keeps_non_whitespace_entities_verbatim(
        self, api_key_env, mock_api, result_factory, extract_documents
    ):
        mock_api.add_results([result_factory(highlights=["Write &lt; to escape < and &amp; for &."])])
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        body = ET.fromstring(extract_documents(result)[0]).find("title").tail
        assert "Write &lt; to escape < and &amp; for &." in body

    async def test_truncates_long_content(self, api_key_env, mock_api, result_factory, extract_documents):
        mock_api.add_results([result_factory(highlights=["word " * 400])])
        result = await _search(ContextDevWebSearchToolConfig(max_content_length=100), "q")
        body = ET.fromstring(extract_documents(result)[0]).find("title").tail.strip()
        assert len(body) <= 100
        assert body.endswith("...")

    async def test_missing_title_falls_back_to_url(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url="https://example.org/untitled", title="")])
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert "<title>\nhttps://example.org/untitled\n</title>" in result

    async def test_drops_results_without_http_urls(self, api_key_env, mock_api, result_factory, extract_documents):
        mock_api.add_results(
            [
                result_factory(url="/goto?url=x"),
                result_factory(url="javascript:alert(1)"),
                result_factory(url="https://kept.example/"),
            ]
        )
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        blocks = extract_documents(result)
        assert len(blocks) == 1
        assert 'href="https://kept.example/"' in blocks[0]

    async def test_no_results_sentinel(self, api_key_env, mock_api):
        mock_api.add_results([])
        assert await _search(ContextDevWebSearchToolConfig(), "q") == "Search returned no results"

    @pytest.mark.parametrize("question", ["", "   "])
    async def test_blank_question_is_rejected_without_a_request(self, api_key_env, mock_api, question):
        result = await _search(ContextDevWebSearchToolConfig(), question)
        assert result == "Error: 'question' argument is required"
        assert mock_api.requests == []


class TestErrorMessages:
    async def test_invalid_key(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(401, message="Unauthorized")
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert result.startswith("Error: Context.dev web search failed due to an invalid API key (401 Unauthorized)")
        assert "request_id: req-401" in result
        assert api_key_env not in result
        assert len(mock_api.requests) == 1

    async def test_invalid_config_key_points_at_the_config(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(401, message="Unauthorized")
        config = ContextDevWebSearchToolConfig(api_key=SecretStr("ctx-bad-config-key"))  # pragma: allowlist secret
        result = await _search(config, "q")
        assert "Please check the api_key in your workflow config" in result
        assert "ctx-bad-config-key" not in result

    async def test_domains_are_normalized_before_sending(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory()])
        await _search(ContextDevWebSearchToolConfig(include_domains=["https://www.NVIDIA.com/en-us/"]), "q")
        assert mock_api.bodies[0]["includeDomains"] == ["nvidia.com"]

    async def test_out_of_credits(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(402, "USAGE_EXCEEDED")
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert "402 USAGE_EXCEEDED" in result
        no_sleep.assert_not_awaited()

    async def test_zero_data_retention_not_enabled(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(403, "ZDR_NOT_ENABLED")
        result = await _search(ContextDevWebSearchToolConfig(zero_data_retention=True), "q")
        assert "403 ZDR_NOT_ENABLED" in result
        assert "zero_data_retention: false" in result

    async def test_other_forbidden(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(403, "FORBIDDEN", "Key lacks search permission")
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert result.startswith("Error: Context.dev web search was refused by Context.dev (403 FORBIDDEN)")
        assert "Key lacks search permission" in result

    async def test_bad_request(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(400, "INPUT_VALIDATION_ERROR", "country is invalid")
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert result.startswith("Error: Context.dev web search request was rejected (400 INPUT_VALIDATION_ERROR)")

    async def test_other_client_errors_do_not_claim_retries(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(404, message="Not Found")
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert result == "Error: Context.dev web search failed - HTTP 404: Not Found (request_id: req-404)"
        assert len(mock_api.requests) == 1

    async def test_unusable_body_reports_actual_attempts(self, api_key_env, mock_api, no_sleep):
        mock_api.add(httpx.Response(200, json={"unexpected": True}))
        result = await _search(ContextDevWebSearchToolConfig(max_retries=2), "q")
        assert (
            result
            == "Error: Context.dev web search failed after 2 attempt(s) - response did not include a results list"
        )

    async def test_long_retry_after_stops_retrying(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(429, "RATE_LIMITED", "Too many requests", **{"retry-after": "120"})
        result = await _search(ContextDevWebSearchToolConfig(max_retries=3), "q")
        assert result.startswith(
            "Error: Context.dev web search failed after 1 attempt(s) - HTTP 429: Too many requests"
        )
        no_sleep.assert_not_awaited()

    async def test_transient_errors_exhaust_retries(self, api_key_env, mock_api, no_sleep):
        mock_api.add_error(503, message="Service Unavailable")
        result = await _search(ContextDevWebSearchToolConfig(max_retries=3), "q")
        assert result.startswith("Error: Context.dev web search failed after 3 attempt(s) - HTTP 503")
        assert len(mock_api.requests) == 3

    async def test_network_errors_exhaust_retries(self, api_key_env, mock_api, no_sleep):
        mock_api.add(httpx.ConnectError("refused"))
        result = await _search(ContextDevWebSearchToolConfig(max_retries=2), "q")
        assert result == "Error: Context.dev web search failed after 2 attempt(s) - network error (ConnectError)"

    async def test_recovers_after_transient_error(self, api_key_env, mock_api, no_sleep, result_factory):
        mock_api.add_error(500).add_results([result_factory()])
        result = await _search(ContextDevWebSearchToolConfig(), "q")
        assert result.startswith("<Document")


class TestLifecycle:
    async def test_http_client_is_reused_and_closed(self, api_key_env, monkeypatch, result_factory):
        created: list[httpx.AsyncClient] = []

        def _factory():
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": [result_factory()]}))
            )
            created.append(client)
            return client

        monkeypatch.setattr(register_module, "_new_http_client", _factory)
        async with context_dev_web_search(ContextDevWebSearchToolConfig(), MagicMock()) as info:
            await info.single_fn("one")
            await info.single_fn("two")
            assert len(created) == 1
            assert not created[0].is_closed
        assert created[0].is_closed

    @pytest.mark.parametrize("url", ["http://localhost:8080/v1", "http://127.0.0.1/v1", "http://[::1]:9000/v1"])
    async def test_loopback_http_bypasses_environment_proxy(self, monkeypatch, url):
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        async with register_module._new_http_client() as client:
            assert client._transport_for_url(httpx.URL(url)) is client._transport

    @pytest.mark.parametrize("url", ["https://api.context.dev/v1", "https://localhost/v1", "http://example.com/v1"])
    async def test_other_urls_keep_environment_proxy(self, monkeypatch, url):
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        async with register_module._new_http_client() as client:
            assert client._transport_for_url(httpx.URL(url)) is not client._transport

    def test_client_module_imports_without_nat(self):
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['nat'] = None; "
            "import context_dev_search.client; "
            "import context_dev_search; "
            "assert 'context_dev_search.register' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    async def test_tool_description_comes_from_docstring(self, api_key_env, mock_api):
        async with context_dev_web_search(ContextDevWebSearchToolConfig(), MagicMock()) as info:
            assert info.description.startswith("Search the web with Context.dev")
