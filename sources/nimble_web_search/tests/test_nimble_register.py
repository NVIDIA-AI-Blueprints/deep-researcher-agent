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

"""Tests for the nimble_web_search NAT registration."""

import os
import sys
import types
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nimble_web_search.register import NimbleWebSearchToolConfig
from nimble_web_search.register import nimble_web_search
from pydantic import SecretStr


class _FakeDoc:
    """Mimic langchain Document just enough for the renderer."""

    def __init__(self, url="", title="", page_content="", description=""):
        self.page_content = page_content
        self.metadata = {
            "url": url,
            "title": title,
            "description": description,
            "position": 1,
            "entity_type": "OrganicResult",
        }


@pytest.fixture
def fake_langchain_nimble(monkeypatch):
    """Install a fake `langchain_nimble` module so tests never hit the network.

    Returns the shared NimbleSearchRetriever instance the registration will create.
    """

    module = types.ModuleType("langchain_nimble")
    instance = MagicMock()
    instance.ainvoke = AsyncMock()

    module.NimbleSearchRetriever = MagicMock(return_value=instance)
    monkeypatch.setitem(sys.modules, "langchain_nimble", module)
    return instance


@pytest.fixture(autouse=True)
def _reset_warn_flag():
    import nimble_web_search.register as reg

    reg._missing_key_warned = False
    yield
    reg._missing_key_warned = False


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)


async def _no_sleep(_):
    return None


class TestNimbleWebSearchToolConfig:
    def test_defaults(self):
        config = NimbleWebSearchToolConfig()
        assert config.max_results == 5
        assert config.api_key is None
        assert config.max_retries == 3
        assert config.search_depth == "lite"
        assert config.focus == "general"
        assert config.country == "US"
        assert config.locale == "en"
        assert config.max_content_length == 10000

    def test_all_fields(self):
        config = NimbleWebSearchToolConfig(
            max_results=10,
            api_key=SecretStr("sk-test"),
            max_retries=1,
            search_depth="deep",
            focus="news",
            country="UK",
            locale="fr",
            max_content_length=50,
        )
        assert config.max_results == 10
        assert config.api_key.get_secret_value() == "sk-test"
        assert config.max_retries == 1
        assert config.search_depth == "deep"
        assert config.focus == "news"
        assert config.country == "UK"
        assert config.locale == "fr"
        assert config.max_content_length == 50

    def test_invalid_search_depth_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NimbleWebSearchToolConfig(search_depth="ultra")

    def test_invalid_focus_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NimbleWebSearchToolConfig(focus="newsy")

    def test_no_include_answer_field(self):
        # include_answer is intentionally not exposed as Deep Researcher Agent-facing config in this
        # initial integration (see register.py / README).
        assert "include_answer" not in NimbleWebSearchToolConfig.model_fields
        assert "include_answers" not in NimbleWebSearchToolConfig.model_fields

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_results", 0),  # below ge=1
            ("max_results", 101),  # above le=100 (Nimble's documented cap)
            ("max_retries", 0),  # below ge=1
            ("max_retries", 11),  # above le=10
            ("max_content_length", 0),  # below ge=1 (use None to disable truncation)
        ],
    )
    def test_out_of_range_numeric_fields_rejected(self, field, value):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NimbleWebSearchToolConfig(**{field: value})

    def test_max_content_length_none_allowed(self):
        config = NimbleWebSearchToolConfig(max_content_length=None)
        assert config.max_content_length is None

    def test_inherits_from_function_base_config(self):
        from nat.data_models.function import FunctionBaseConfig

        assert issubclass(NimbleWebSearchToolConfig, FunctionBaseConfig)


class TestNimbleWebSearchStub:
    async def test_stub_when_no_api_key(self):
        config = NimbleWebSearchToolConfig()
        builder = MagicMock()

        async with nimble_web_search(config, builder) as info:
            result = await info.single_fn("anything")

        assert "NIMBLE_API_KEY" in result
        assert "unavailable" in result.lower()

    async def test_warn_once_when_key_missing(self, caplog):
        import logging

        config = NimbleWebSearchToolConfig()
        builder = MagicMock()
        with caplog.at_level(logging.WARNING, logger="nimble_web_search.register"):
            async with nimble_web_search(config, builder):
                pass
            async with nimble_web_search(config, builder):
                pass

        warnings = [r for r in caplog.records if "NIMBLE_API_KEY not found" in r.message]
        assert len(warnings) == 1


class TestNimbleWebSearchLive:
    async def test_api_key_from_config_passed_to_retriever(self, fake_langchain_nimble):
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://a.example", title="A", page_content="body a")
        ]
        config = NimbleWebSearchToolConfig(api_key=SecretStr("sk-from-config"))
        builder = MagicMock()

        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("question")

        # The configured key is passed straight to the retriever, never written
        # to the process-global environment.
        retriever_cls = sys.modules["langchain_nimble"].NimbleSearchRetriever
        assert retriever_cls.call_args.kwargs["api_key"].get_secret_value() == "sk-from-config"
        assert os.environ.get("NIMBLE_API_KEY") is None
        assert "https://a.example" in out
        assert "body a" in out

    async def test_successful_search_formats_documents(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://a.example", title="Title A", page_content="Body A"),
            _FakeDoc(url="https://b.example", title="Title B", page_content="Body B"),
        ]
        config = NimbleWebSearchToolConfig(max_results=2)
        builder = MagicMock()

        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("query")

        assert "Title A" in out
        assert "Title B" in out
        assert "Body A" in out
        assert "Body B" in out
        assert "---" in out
        assert "https://a.example" in out

    async def test_description_used_when_page_content_empty(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://a.example", title="A", page_content="", description="metadata-only description"),
        ]
        config = NimbleWebSearchToolConfig()
        builder = MagicMock()

        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "metadata-only description" in out

    async def test_search_depth_deep_passes_through(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://example.com", title="t", page_content="body")
        ]

        config = NimbleWebSearchToolConfig(search_depth="deep")
        builder = MagicMock()
        async with nimble_web_search(config, builder):
            pass

        # Verify the constructor received the expected kwargs via the module mock
        ctor = sys.modules["langchain_nimble"].NimbleSearchRetriever
        ctor.assert_called()
        kwargs = ctor.call_args.kwargs
        assert kwargs["search_depth"] == "deep"
        assert kwargs["max_results"] == 5
        assert kwargs["focus"] == "general"  # default is general, passed explicitly
        assert kwargs["country"] == "US"
        assert kwargs["locale"] == "en"
        # include_answer is intentionally omitted in v1 — the upstream retriever
        # surfaces it as a 403 enterprise gate for non-enterprise accounts.
        assert "include_answer" not in kwargs

    async def test_focus_defaults_to_general(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: with no focus configured, the retriever is built with focus='general' —
        so general research queries never silently use a news/other focus.
        """
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [_FakeDoc(url="https://example.com", title="t", page_content="b")]

        config = NimbleWebSearchToolConfig()
        builder = MagicMock()
        async with nimble_web_search(config, builder):
            pass

        kwargs = sys.modules["langchain_nimble"].NimbleSearchRetriever.call_args.kwargs
        assert kwargs["focus"] == "general"

    async def test_focus_news_passes_through_when_configured(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: an operator can deliberately configure focus='news'; it reaches the SDK.
        (The LLM never chooses focus — it's not a tool parameter — so general queries can't
        drift to news on their own.)
        """
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [_FakeDoc(url="https://example.com", title="t", page_content="b")]

        config = NimbleWebSearchToolConfig(focus="news")
        builder = MagicMock()
        async with nimble_web_search(config, builder):
            pass

        kwargs = sys.modules["langchain_nimble"].NimbleSearchRetriever.call_args.kwargs
        assert kwargs["focus"] == "news"

    async def test_truncates_long_query(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://example.com", title="t", page_content="body")
        ]

        config = NimbleWebSearchToolConfig()
        builder = MagicMock()
        long_q = "x" * 500
        async with nimble_web_search(config, builder) as info:
            await info.single_fn(long_q)

        (passed_q,), _ = fake_langchain_nimble.ainvoke.call_args
        assert len(passed_q) == 400
        assert passed_q.endswith("...")

    async def test_truncates_content(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://example.com", title="t", page_content="abcdefghijklmnop")
        ]

        config = NimbleWebSearchToolConfig(max_content_length=8)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "abcde..." in out
        assert "abcdefghi" not in out

    async def test_truncates_content_small_limit_no_negative_slice(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: a max_content_length below 4 hard-cuts without a negative slice, and the
        result never exceeds the configured budget (the ellipsis needs 3 chars of headroom).
        """
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(url="https://example.com", title="t", page_content="abcdefghij")
        ]

        config = NimbleWebSearchToolConfig(max_content_length=2)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        # body hard-cut to exactly 2 chars ("ab"), no "..." appended, no over-run
        assert "ab\n</Document>" in out
        assert "abc" not in out

    async def test_empty_results_returns_error(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = []

        config = NimbleWebSearchToolConfig(max_retries=1)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "no results" in out.lower()

    async def test_retries_then_succeeds(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        monkeypatch.setattr("nimble_web_search.register.asyncio.sleep", _no_sleep)

        fake_langchain_nimble.ainvoke.side_effect = [
            RuntimeError("transient"),
            [_FakeDoc(url="https://example.com", title="t", page_content="ok")],
        ]

        config = NimbleWebSearchToolConfig(max_retries=3)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "ok" in out
        assert fake_langchain_nimble.ainvoke.call_count == 2

    async def test_final_retry_failure_returns_error(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        monkeypatch.setattr("nimble_web_search.register.asyncio.sleep", _no_sleep)
        fake_langchain_nimble.ainvoke.side_effect = RuntimeError("upstream broken")

        config = NimbleWebSearchToolConfig(max_retries=2)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "Web search failed" in out
        assert "upstream broken" in out

    async def test_401_returns_friendly_message(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        monkeypatch.setattr("nimble_web_search.register.asyncio.sleep", _no_sleep)
        fake_langchain_nimble.ainvoke.side_effect = RuntimeError("401 Unauthorized")

        config = NimbleWebSearchToolConfig(max_retries=2)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "401" in out
        assert "NIMBLE_API_KEY" in out

    async def test_403_returns_friendly_entitlement_message(self, fake_langchain_nimble, monkeypatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        monkeypatch.setattr("nimble_web_search.register.asyncio.sleep", _no_sleep)
        fake_langchain_nimble.ainvoke.side_effect = RuntimeError(
            "403 - {'detail': 'This feature is only available for enterprise accounts.'}"
        )

        config = NimbleWebSearchToolConfig(max_retries=1, search_depth="fast")
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "403" in out
        assert "enterprise" in out.lower()

    async def test_non_transient_errors_short_circuit_without_retry(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: 401/403 (and other non-transient) errors return immediately after a single
        attempt — no retry, no backoff sleep — so the caller isn't made to wait through retries
        for an error that can't be resolved by retrying.
        """
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        slept = []
        monkeypatch.setattr(
            "nimble_web_search.register.asyncio.sleep",
            lambda d: slept.append(d) or _no_sleep(d),
        )
        fake_langchain_nimble.ainvoke.side_effect = RuntimeError("401 Unauthorized")

        config = NimbleWebSearchToolConfig(max_retries=3)
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        assert "401" in out
        # short-circuited: exactly one upstream call, and no backoff sleep happened
        assert fake_langchain_nimble.ainvoke.call_count == 1
        assert slept == []

    # --- Field-passthrough tests (defaults are covered by test_search_depth_deep_passes_through;
    # the four tests below verify that *non-default* values flow into the SDK constructor and
    # that the rendered output stays well-formed for unusual title content) -------------------

    async def test_non_default_country_passthrough(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: NimbleWebSearchToolConfig(country="FR") forwards country='FR' to the SDK."""
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [_FakeDoc(url="https://example.com", title="t", page_content="b")]

        config = NimbleWebSearchToolConfig(country="FR")
        builder = MagicMock()
        async with nimble_web_search(config, builder):
            pass

        kwargs = sys.modules["langchain_nimble"].NimbleSearchRetriever.call_args.kwargs
        assert kwargs["country"] == "FR"

    async def test_non_default_locale_passthrough(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: NimbleWebSearchToolConfig(locale="fr") forwards locale='fr' to the SDK."""
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [_FakeDoc(url="https://example.com", title="t", page_content="b")]

        config = NimbleWebSearchToolConfig(locale="fr")
        builder = MagicMock()
        async with nimble_web_search(config, builder):
            pass

        kwargs = sys.modules["langchain_nimble"].NimbleSearchRetriever.call_args.kwargs
        assert kwargs["locale"] == "fr"

    async def test_country_locale_combined_passthrough(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: Both country and locale non-defaults are forwarded together to the SDK."""
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        fake_langchain_nimble.ainvoke.return_value = [_FakeDoc(url="https://example.com", title="t", page_content="b")]

        config = NimbleWebSearchToolConfig(country="JP", locale="ja")
        builder = MagicMock()
        async with nimble_web_search(config, builder):
            pass

        kwargs = sys.modules["langchain_nimble"].NimbleSearchRetriever.call_args.kwargs
        assert kwargs["country"] == "JP"
        assert kwargs["locale"] == "ja"

    async def test_renderer_escapes_special_characters(self, fake_langchain_nimble, monkeypatch):
        """VERIFIES: untrusted API fields containing <, >, &, " are HTML-escaped in the
        rendered <Document> block, so they can't break the markup or inject into downstream
        renderers/parsers.
        """
        monkeypatch.setenv("NIMBLE_API_KEY", "sk-env")
        weird_title = "Apple Q4 < Microsoft Q4 > Forecast & Analysis"
        fake_langchain_nimble.ainvoke.return_value = [
            _FakeDoc(
                url="https://example.com/q4?a=1&b=2",
                title=weird_title,
                page_content='snippet with </Document> and "quotes"',
            )
        ]

        config = NimbleWebSearchToolConfig()
        builder = MagicMock()
        async with nimble_web_search(config, builder) as info:
            out = await info.single_fn("q")

        # Title special chars are escaped (raw < > & no longer present in the title text)
        assert "Apple Q4 &lt; Microsoft Q4 &gt; Forecast &amp; Analysis" in out
        assert weird_title not in out
        # The href attribute is escaped (& → &amp;) so it can't break the attribute
        assert 'href="https://example.com/q4?a=1&amp;b=2"' in out
        # A body that contains a literal </Document> can't terminate the block early
        assert "&lt;/Document&gt;" in out
        # The block is still well-formed and terminates correctly
        assert out.endswith("</Document>")
