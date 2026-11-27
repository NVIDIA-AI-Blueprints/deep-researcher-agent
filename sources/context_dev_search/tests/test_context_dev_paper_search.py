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

"""Tests for the registered ``context_dev_paper_search`` NAT function."""

from unittest.mock import MagicMock

import pytest
from context_dev_search import register as register_module
from context_dev_search.register import SCHOLARLY_DOMAINS
from context_dev_search.register import ContextDevPaperSearchToolConfig
from context_dev_search.register import context_dev_paper_search
from pydantic import ValidationError

ATTENTION_URL = "https://arxiv.org/abs/1706.03762"


async def _search(config: ContextDevPaperSearchToolConfig, query: str) -> str:
    async with context_dev_paper_search(config, MagicMock()) as info:
        return await info.single_fn(query)


@pytest.fixture(autouse=True)
def _reset_missing_key_warning(monkeypatch):
    monkeypatch.setattr(register_module, "_missing_key_warned", set())


class TestConfig:
    def test_defaults(self):
        config = ContextDevPaperSearchToolConfig()
        assert config.max_results == 10
        assert config.max_content_length == 1500
        assert config.include_highlights is True
        assert config.max_retries == 3
        assert config.scholarly_sources_only is True

    @pytest.mark.parametrize("overrides", [{"max_results": 0}, {"max_content_length": 50}, {"country": "1x"}])
    def test_rejects_invalid_values(self, overrides):
        with pytest.raises(ValidationError):
            ContextDevPaperSearchToolConfig(**overrides)


class TestMissingKey:
    async def test_stub_returns_setup_instructions(self, no_api_key_env, mock_api):
        result = await _search(ContextDevPaperSearchToolConfig(), "transformers")
        assert result.startswith(
            "Error: Context.dev paper search is unavailable because CONTEXT_DEV_API_KEY is not set"
        )
        assert mock_api.requests == []

    async def test_stub_keeps_the_live_tool_signature(self, no_api_key_env):
        async with context_dev_paper_search(ContextDevPaperSearchToolConfig(), MagicMock()) as info:
            assert list(info.input_schema.model_fields) == ["query"]


class TestRendering:
    async def test_renders_numbered_papers(self, api_key_env, mock_api, result_factory, assert_paper_contract):
        mock_api.add_results(
            [
                result_factory(
                    url=ATTENTION_URL,
                    title="Attention Is All You Need",
                    highlights=["The Transformer is based solely on attention mechanisms."],
                ),
                result_factory(
                    url="https://www.nature.com/articles/nature14539",
                    title="Deep learning",
                    description="Deep learning allows computational models...",
                    highlights=None,
                ),
            ]
        )
        result = await _search(ContextDevPaperSearchToolConfig(), "self-attention replaces recurrence")
        entries = assert_paper_contract(result, max_results=10)
        assert entries[0].group(0) == (
            "1. **Attention Is All You Need**\n"
            "   - **Source**: arxiv.org\n"
            "   - **Snippet**: The Transformer is based solely on attention mechanisms.\n"
            f"   - **Link**: {ATTENTION_URL}"
        )
        assert entries[1].group("source") == "nature.com"
        assert entries[1].group("snippet") == "Deep learning allows computational models..."

    async def test_flattens_multiline_provider_text(self, api_key_env, mock_api, result_factory, assert_paper_contract):
        mock_api.add_results(
            [
                result_factory(
                    url="https://arxiv.org/abs/2310.00927",
                    title="Understanding\nZero-shot   Transfer",
                    highlights=["[Section > 5]\n\n\nZero-shot transfer is done by\n---\nissuing prompts.", "| a | b |"],
                )
            ]
        )
        result = await _search(ContextDevPaperSearchToolConfig(), "clip zero-shot")
        entry = assert_paper_contract(result, max_results=10)[0]
        assert entry.group("title") == "Understanding Zero-shot Transfer"
        assert entry.group("snippet") == "[Section > 5] Zero-shot transfer is done by --- issuing prompts. | a | b |"

    @pytest.mark.parametrize(
        ("title", "rendered"),
        [("**Bold** claims", "\u2217\u2217Bold\u2217\u2217 claims"), ("Learning A*", "Learning A\u2217")],
    )
    async def test_asterisks_in_titles_cannot_break_layout(
        self, api_key_env, mock_api, result_factory, assert_paper_contract, title, rendered
    ):
        mock_api.add_results([result_factory(url=ATTENTION_URL, title=title)])
        result = await _search(ContextDevPaperSearchToolConfig(), "q")
        assert assert_paper_contract(result, max_results=10)[0].group("title") == rendered

    async def test_keeps_ampersands_that_are_not_entity_references(self, api_key_env, mock_api, result_factory):
        mock_api.add_results(
            [result_factory(url="https://example.org/p?a=1&copy=2", title="R&D at scale", highlights=["x &#x20;y"])]
        )
        result = await _search(ContextDevPaperSearchToolConfig(scholarly_sources_only=False), "q")
        assert "**R&D at scale**" in result
        assert "**Link**: https://example.org/p?a=1&copy=2" in result
        assert "**Snippet**: x y" in result

    async def test_truncates_snippets(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url=ATTENTION_URL, highlights=["token " * 500])])
        result = await _search(ContextDevPaperSearchToolConfig(max_content_length=200), "q")
        snippet = result.split("**Snippet**: ", 1)[1].split("\n", 1)[0]
        assert len(snippet) <= 200
        assert snippet.endswith("...")

    async def test_drops_results_without_http_urls_and_renumbers(
        self, api_key_env, mock_api, result_factory, assert_paper_contract
    ):
        mock_api.add_results([result_factory(url="doi:10.1000/182"), result_factory(url=ATTENTION_URL)])
        result = await _search(ContextDevPaperSearchToolConfig(), "q")
        entries = assert_paper_contract(result, max_results=10)
        assert len(entries) == 1
        assert entries[0].group("index") == "1"

    async def test_sends_query_as_is(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url=ATTENTION_URL)])
        await _search(ContextDevPaperSearchToolConfig(max_results=4), "tip of the tongue: paper about attention")
        body = mock_api.bodies[0]
        assert body["query"] == "tip of the tongue: paper about attention"
        assert body["numResults"] == 12  # over-fetch so scholarly filtering still fills max_results
        assert body["highlightsOptions"] == {"enabled": True, "maxCharacters": 1500}


class TestScholarlySources:
    async def test_never_sends_domain_filters_by_default(self, api_key_env, mock_api, result_factory):
        # The API leaves its indexed research papers out of domain-filtered requests.
        mock_api.add_results([result_factory(url=ATTENTION_URL)])
        await _search(ContextDevPaperSearchToolConfig(), "q")
        body = mock_api.bodies[0]
        assert "includeDomains" not in body
        assert "excludeDomains" not in body
        assert body["numResults"] == 30

    async def test_keeps_only_scholarly_hosts_and_trims_to_max_results(
        self, api_key_env, mock_api, result_factory, assert_paper_contract
    ):
        mock_api.add_results(
            [
                result_factory(url="https://economictimes.indiatimes.com/news/attention", title="News"),
                result_factory(url=ATTENTION_URL, title="Attention Is All You Need"),
                result_factory(url="https://medium.com/@someone/attention", title="Blog"),
                result_factory(url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7177153", title="PMC"),
                result_factory(url="https://www.nature.com/articles/nature14539", title="Nature"),
                result_factory(url="https://doi.org/10.1000/182", title="DOI"),
            ]
        )
        result = await _search(ContextDevPaperSearchToolConfig(max_results=3), "q")
        entries = assert_paper_contract(result, max_results=3)
        assert [entry.group("title") for entry in entries] == ["Attention Is All You Need", "PMC", "Nature"]

    async def test_overfetch_is_capped_by_the_api_limit(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url=ATTENTION_URL)])
        await _search(ContextDevPaperSearchToolConfig(max_results=40), "q")
        assert mock_api.bodies[0]["numResults"] == 100

    async def test_include_domains_replace_the_scholarly_list_and_are_applied_locally(
        self, api_key_env, mock_api, result_factory
    ):
        mock_api.add_results(
            [result_factory(url="https://example.org/paper"), result_factory(url=ATTENTION_URL, title="arXiv")]
        )
        result = await _search(ContextDevPaperSearchToolConfig(include_domains=["Example.org"]), "q")
        body = mock_api.bodies[0]
        assert "includeDomains" not in body
        assert body["numResults"] == 30
        assert "https://example.org/paper" in result
        assert ATTENTION_URL not in result

    async def test_exclude_domains_are_applied_locally(self, api_key_env, mock_api, result_factory):
        mock_api.add_results(
            [
                result_factory(url="https://www.researchgate.net/publication/1", title="ResearchGate"),
                result_factory(url=ATTENTION_URL, title="arXiv"),
            ]
        )
        result = await _search(ContextDevPaperSearchToolConfig(exclude_domains=["researchgate.net"]), "q")
        assert "excludeDomains" not in mock_api.bodies[0]
        assert mock_api.bodies[0]["numResults"] == 30  # the scholarly list is still active
        assert "researchgate" not in result
        assert ATTENTION_URL in result

    async def test_can_search_the_whole_index(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url="https://example.org/explainer")])
        result = await _search(ContextDevPaperSearchToolConfig(scholarly_sources_only=False), "q")
        assert mock_api.bodies[0]["numResults"] == 10
        assert "https://example.org/explainer" in result

    async def test_exclude_only_does_not_overfetch(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url=ATTENTION_URL)])
        config = ContextDevPaperSearchToolConfig(scholarly_sources_only=False, exclude_domains=["medium.com"])
        await _search(config, "q")
        assert mock_api.bodies[0]["numResults"] == 10

    async def test_falls_back_to_unfiltered_results_when_none_are_scholarly(
        self, api_key_env, mock_api, result_factory, assert_paper_contract
    ):
        mock_api.add_results([result_factory(url="https://lab.example.edu/paper.pdf", title="Lab preprint")])
        result = await _search(ContextDevPaperSearchToolConfig(), "q")
        assert assert_paper_contract(result, max_results=10)[0].group("title") == "Lab preprint"

    async def test_explicit_include_domains_never_fall_back(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url="https://medium.com/@someone/attention")])
        result = await _search(ContextDevPaperSearchToolConfig(include_domains=["arxiv.org"]), "q")
        assert result == "Search returned no results"

    @pytest.mark.parametrize(
        ("entries", "normalized"),
        [
            (["https://www.arXiv.org/abs/1706.03762"], ["arxiv.org"]),
            ([".edu", "Nature.com", "nature.com"], ["edu", "nature.com"]),
        ],
    )
    def test_domain_entries_are_normalized(self, entries, normalized):
        assert ContextDevPaperSearchToolConfig(include_domains=entries).include_domains == normalized

    @pytest.mark.parametrize("entry", ["", "ab", "https://", "bad domain.com"])
    def test_invalid_domain_entries_are_rejected(self, entry):
        with pytest.raises(ValidationError):
            ContextDevPaperSearchToolConfig(exclude_domains=[entry])

    async def test_tld_entries_match_subdomains(self, api_key_env, mock_api, result_factory):
        mock_api.add_results([result_factory(url="https://cs.stanford.edu/paper.pdf", title="Stanford")])
        result = await _search(ContextDevPaperSearchToolConfig(include_domains=[".edu"]), "q")
        assert "**Stanford**" in result

    def test_domain_list_is_well_formed(self):
        assert len(SCHOLARLY_DOMAINS) == len(set(SCHOLARLY_DOMAINS))
        assert all("/" not in domain and domain == domain.lower() and "." in domain for domain in SCHOLARLY_DOMAINS)
