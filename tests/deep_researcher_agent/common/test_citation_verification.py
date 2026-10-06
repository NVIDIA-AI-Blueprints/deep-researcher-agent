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

"""Tests for citation verification module."""

import logging
from html import escape

import pytest

from deep_researcher_agent.common.citation_verification import _PARSER_REGISTRY
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryError
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryReason
from deep_researcher_agent.common.citation_verification import SourceEntry
from deep_researcher_agent.common.citation_verification import SourceRegistry
from deep_researcher_agent.common.citation_verification import _normalize_url
from deep_researcher_agent.common.citation_verification import _parse_citation_key
from deep_researcher_agent.common.citation_verification import classify_empty_source_registry_reason
from deep_researcher_agent.common.citation_verification import clean_extracted_url
from deep_researcher_agent.common.citation_verification import extract_http_urls
from deep_researcher_agent.common.citation_verification import extract_sources_from_tool_result
from deep_researcher_agent.common.citation_verification import register_source_parser
from deep_researcher_agent.common.citation_verification import sanitize_report
from deep_researcher_agent.common.citation_verification import verify_citations


@pytest.fixture(autouse=True)
def fixture_restore_parser_registry():
    """Restore the parser registry after each test to prevent leaks."""
    original = list(_PARSER_REGISTRY)
    yield
    _PARSER_REGISTRY.clear()
    _PARSER_REGISTRY.extend(original)


class TestEmptySourceRegistryContract:
    """Tests for the public empty-source failure contract."""

    @pytest.mark.parametrize(
        ("reason", "expected_message"),
        [
            (
                EmptySourceRegistryReason.NO_SOURCES_SELECTED,
                "No data sources are selected. Select at least one data source and run the research again.",
            ),
            (
                EmptySourceRegistryReason.NO_SOURCE_RESULTS,
                "The selected data sources returned no results. "
                "Try rephrasing the question or selecting different data sources.",
            ),
            (
                EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE,
                "The selected data source tools are currently unavailable. "
                "Check their configuration or select different data sources and run the research again.",
            ),
        ],
    )
    def test_reason_has_safe_actionable_public_message(self, reason, expected_message):
        error = EmptySourceRegistryError(reason=reason)

        assert error.reason is reason
        assert error.public_message == expected_message
        assert "Please try again" not in error.public_message

    def test_reason_values_are_stable(self):
        assert [reason.value for reason in EmptySourceRegistryReason] == [
            "no_sources_selected",
            "no_source_results",
            "source_tools_unavailable",
        ]

    @pytest.mark.parametrize(
        ("data_sources", "available_count", "unavailable_tools", "expected"),
        [
            ([], 0, ["web_search (missing key)"], EmptySourceRegistryReason.NO_SOURCES_SELECTED),
            (["web"], 0, ["web_search (missing key)"], EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE),
            (None, 0, ["web_search (missing key)"], EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE),
            (["web"], 1, ["scholar_search (missing key)"], EmptySourceRegistryReason.NO_SOURCE_RESULTS),
            (["web"], 1, [], EmptySourceRegistryReason.NO_SOURCE_RESULTS),
            (None, 0, [], EmptySourceRegistryReason.NO_SOURCE_RESULTS),
        ],
    )
    def test_classification_distinguishes_selection_availability_and_empty_results(
        self, data_sources, available_count, unavailable_tools, expected
    ):
        assert classify_empty_source_registry_reason(data_sources, available_count, unavailable_tools) is expected

    def test_preserves_compatibility_fields_and_optional_generated_answer(self):
        error = EmptySourceRegistryError(
            "deep research",
            unavailable_tools=["web_search"],
            available_count=2,
            generated_answer="Sanitized draft",
        )

        assert error.agent_type == "deep research"
        assert error.unavailable_tools == ["web_search"]
        assert error.available_count == 2
        assert error.generated_answer == "Sanitized draft"
        assert error.reason is EmptySourceRegistryReason.NO_SOURCE_RESULTS
        assert error.public_response == (
            "Sanitized draft\n\n"
            "The selected data sources returned no results. "
            "Try rephrasing the question or selecting different data sources."
        )


# ---------------------------------------------------------------------------
# URL normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    """Tests for URL normalization."""

    def test_lowercase_scheme_and_host(self):
        assert _normalize_url("HTTPS://Example.COM/path") == "https://example.com/path"

    def test_strip_trailing_slash(self):
        assert _normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_strip_fragment(self):
        assert _normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_remove_utm_params(self):
        result = _normalize_url("https://example.com/page?utm_source=twitter&key=val")
        assert "utm_source" not in result
        assert "key=val" in result

    def test_unescape_html_entities(self):
        assert _normalize_url("https://example.com/page?a=1&amp;b=2") == "https://example.com/page?a=1&b=2"

    def test_root_path_preserved(self):
        assert _normalize_url("https://example.com") == "https://example.com/"

    def test_identical_urls_match(self):
        url1 = "https://example.com/article?id=42"
        url2 = "https://example.com/article?id=42"
        assert _normalize_url(url1) == _normalize_url(url2)


# ---------------------------------------------------------------------------
# Citation key parsing tests
# ---------------------------------------------------------------------------


class TestParseCitationKey:
    """Tests for knowledge-layer citation key parsing."""

    def test_filename_with_page(self):
        filename, page = _parse_citation_key("report.pdf, p.15")
        assert filename == "report.pdf"
        assert page == 15

    def test_filename_with_page_word(self):
        filename, page = _parse_citation_key("report.pdf, page 15")
        assert filename == "report.pdf"
        assert page == 15

    def test_filename_only(self):
        filename, page = _parse_citation_key("report.pdf")
        assert filename == "report.pdf"
        assert page is None

    def test_filename_with_spaces(self):
        filename, page = _parse_citation_key("my report.pdf, p.3")
        assert filename == "my report.pdf"
        assert page == 3


# ---------------------------------------------------------------------------
# SourceRegistry tests
# ---------------------------------------------------------------------------


class TestSourceRegistry:
    """Tests for SourceRegistry."""

    @pytest.fixture(name="registry")
    def fixture_registry(self):
        return SourceRegistry()

    def test_add_and_has_url(self, registry):
        registry.add(SourceEntry(url="https://example.com/article", source_type="tavily"))
        assert registry.has_url("https://example.com/article")

    def test_url_normalization_on_lookup(self, registry):
        registry.add(SourceEntry(url="https://Example.COM/path/"))
        assert registry.has_url("https://example.com/path")

    def test_missing_url_returns_false(self, registry):
        registry.add(SourceEntry(url="https://example.com/a"))
        assert not registry.has_url("https://example.com/b")

    def test_has_citation_key_exact(self, registry):
        registry.add(SourceEntry(citation_key="report.pdf, p.15", source_type="knowledge_layer"))
        assert registry.has_citation_key("report.pdf, p.15")

    def test_has_citation_key_fuzzy_page_format(self, registry):
        registry.add(SourceEntry(citation_key="report.pdf, p.15"))
        assert registry.has_citation_key("report.pdf, page 15")

    def test_has_citation_key_case_insensitive(self, registry):
        registry.add(SourceEntry(citation_key="Report.PDF, p.15"))
        assert registry.has_citation_key("report.pdf, p.15")

    def test_has_citation_key_no_page(self, registry):
        registry.add(SourceEntry(citation_key="report.pdf"))
        assert registry.has_citation_key("report.pdf")

    def test_has_citation_key_different_page_matches(self, registry):
        """Same file, different page — still matches (lenient)."""
        registry.add(SourceEntry(citation_key="report.pdf, p.15"))
        assert registry.has_citation_key("report.pdf, p.5")

    def test_has_citation_key_different_file_no_match(self, registry):
        registry.add(SourceEntry(citation_key="report.pdf, p.15"))
        assert not registry.has_citation_key("other.pdf, p.15")

    def test_all_sources(self, registry):
        e1 = SourceEntry(url="https://a.com")
        e2 = SourceEntry(citation_key="doc.pdf")
        registry.add(e1)
        registry.add(e2)
        assert len(registry.all_sources()) == 2

    def test_clear(self, registry):
        registry.add(SourceEntry(url="https://a.com"))
        registry.clear()
        assert not registry.has_url("https://a.com")
        assert len(registry.all_sources()) == 0

    def test_deduplicates_urls(self, registry):
        registry.add(SourceEntry(url="https://example.com/page"))
        registry.add(SourceEntry(url="https://example.com/page"))
        assert registry.has_url("https://example.com/page")
        assert len(registry.all_sources()) == 1  # deduplicated by normalized URL

    def test_deduplicates_citation_keys_by_filename(self, registry):
        registry.add(SourceEntry(citation_key="report.pdf, p.5"))
        registry.add(SourceEntry(citation_key="report.pdf, p.10"))
        # Same file, different pages — deduplicated by filename
        assert len(registry.all_sources()) == 1
        assert registry.has_citation_key("report.pdf")

    def test_different_citation_key_files_not_deduped(self, registry):
        registry.add(SourceEntry(citation_key="report.pdf, p.5"))
        registry.add(SourceEntry(citation_key="other.pdf, p.5"))
        assert len(registry.all_sources()) == 2

    def test_resolve_url_exact_match(self, registry):
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.03762"))
        assert registry.resolve_url("https://arxiv.org/abs/1706.03762") == "https://arxiv.org/abs/1706.03762"

    def test_resolve_url_truncated_path(self, registry):
        """LLM truncated the URL path — resolve to full canonical URL."""
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.03762"))
        assert registry.resolve_url("https://arxiv.org/abs/1706") == "https://arxiv.org/abs/1706.03762"

    def test_resolve_url_domain_only(self, registry):
        """LLM kept domain but dropped entire path."""
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.03762"))
        assert registry.resolve_url("https://arxiv.org") == "https://arxiv.org/abs/1706.03762"

    def test_resolve_url_no_match(self, registry):
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.03762"))
        assert registry.resolve_url("https://totally-different.com") is None

    def test_resolve_url_with_ellipsis_truncation(self, registry):
        """LLM wrote URL with trailing '...' which gets stripped."""
        registry.add(SourceEntry(url="https://example.com/very/long/path/article"))
        # After stripping "...", the truncated URL should match
        resolved = registry.resolve_url("https://example.com/very/long")
        assert resolved == "https://example.com/very/long/path/article"

    def test_resolve_url_truncated_mid_query(self, registry):
        """Report URL cut mid-query (e.g. copy-paste); match by raw prefix."""
        full = "https://example.sharepoint.com/sites/foo/_layouts/15/Doc.aspx?sourcedoc=%7BGUID%7D&file=US%20Benefits%20Open%20Enrollment.pptx&action=edit"
        registry.add(SourceEntry(url=full))
        truncated = (
            "https://example.sharepoint.com/sites/foo/_layouts/15/Doc.aspx?sourcedoc=%7BGUID%7D&file=US%20Benefit"
        )
        resolved = registry.resolve_url(truncated)
        assert resolved == full

    def test_resolve_url_ambiguous_returns_none(self, registry):
        """Shallow ambiguous prefix (e.g. arxiv abs/1706) — reject."""
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.03762"))
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.08500"))
        # path "abs/1706" has only 2 segments — too shallow to treat as parent
        assert registry.resolve_url("https://arxiv.org/abs/1706") is None

    def test_resolve_url_prefix_ambiguous_rejected(self, registry):
        """Report URL is prefix of multiple registry URLs — ambiguous, reject."""
        registry.add(SourceEntry(url="https://example.sharepoint.com/sites/hr/Pages/New-Employees.aspx"))
        registry.add(SourceEntry(url="https://example.sharepoint.com/sites/hr/Pages/ourculture.aspx"))
        assert registry.resolve_url("https://example.sharepoint.com/sites/hr/") is None

    def test_resolve_url_unique_prefix_succeeds(self, registry):
        """Single registry URL matches the prefix — unambiguous, return it."""
        registry.add(SourceEntry(url="https://arxiv.org/abs/1706.03762"))
        registry.add(SourceEntry(url="https://arxiv.org/abs/2301.00001"))
        # "https://arxiv.org/abs/1706" only matches the first
        assert registry.resolve_url("https://arxiv.org/abs/1706") == "https://arxiv.org/abs/1706.03762"

    def test_resolve_url_child_path_single_match(self, registry):
        """LLM expanded a registry URL to a subpage — child-path match."""
        registry.add(SourceEntry(url="https://www.example.com/us/benefits/"))
        resolved = registry.resolve_url("https://www.example.com/us/benefits/healthcare/")
        assert resolved == "https://www.example.com/us/benefits/"

    def test_resolve_url_child_path_requires_depth(self, registry):
        """Domain-only registry URLs (< 2 path segments) should NOT child-match."""
        registry.add(SourceEntry(url="https://example.com/"))
        assert registry.resolve_url("https://example.com/us/benefits/") is None

    def test_resolve_url_child_path_single_parent(self, registry):
        """Report URL is a child of only one registry path — match succeeds."""
        registry.add(SourceEntry(url="https://example.com/us/benefits/"))
        registry.add(SourceEntry(url="https://example.com/us/benefits/time-off/"))
        # "healthcare/" is under "benefits/" but not under "time-off/"
        # Only one match, so this should succeed
        resolved = registry.resolve_url("https://example.com/us/benefits/healthcare/")
        assert resolved == "https://example.com/us/benefits/"

    def test_resolve_url_child_path_different_domain(self, registry):
        """Child-path match requires same domain."""
        registry.add(SourceEntry(url="https://example.com/us/benefits/"))
        assert registry.resolve_url("https://other.com/us/benefits/healthcare/") is None

    def test_resolve_url_query_subset_match(self, registry):
        """LLM dropped some query params — query-subset match recovers it."""
        full_url = (
            "https://example.sharepoint.com/:p:/r/sites/benefits/_layouts/15/Doc.aspx"
            "?sourcedoc=%7BGUID-AAA%7D&file=Benefits.pptx&action=edit&mobileredirect=true"
        )
        registry.add(SourceEntry(url=full_url))
        partial_url = (
            "https://example.sharepoint.com/:p:/r/sites/benefits/_layouts/15/Doc.aspx?sourcedoc=%7BGUID-AAA%7D"
        )
        assert registry.resolve_url(partial_url) == full_url

    def test_resolve_url_query_subset_disambiguates_by_params(self, registry):
        """Multiple SharePoint docs with same path — query-subset uses params to pick the right one."""
        url_a = (
            "https://example.sharepoint.com/:p:/r/sites/benefits/_layouts/15/Doc.aspx"
            "?sourcedoc=%7BGUID-AAA%7D&file=Benefits.pptx&action=edit"
        )
        url_b = (
            "https://example.sharepoint.com/:p:/r/sites/benefits/_layouts/15/Doc.aspx"
            "?sourcedoc=%7BGUID-BBB%7D&file=HSA.pptx&action=edit"
        )
        registry.add(SourceEntry(url=url_a))
        registry.add(SourceEntry(url=url_b))
        partial = "https://example.sharepoint.com/:p:/r/sites/benefits/_layouts/15/Doc.aspx?sourcedoc=%7BGUID-BBB%7D"
        assert registry.resolve_url(partial) == url_b

    def test_resolve_url_query_subset_no_match_wrong_value(self, registry):
        """Query-subset rejects when param values differ."""
        registry.add(SourceEntry(url="https://example.com/doc?id=123&mode=view"))
        assert registry.resolve_url("https://example.com/doc?id=999") is None

    def test_resolve_url_query_subset_reordered_params(self, registry):
        """Report URL has a subset of params in different order — only Strategy 5 can match."""
        full_url = (
            "https://example.sharepoint.com/sites/hr/_layouts/15/Doc.aspx"
            "?sourcedoc=%7BGUID-X%7D&file=Handbook.pptx&action=edit&mobileredirect=true"
        )
        registry.add(SourceEntry(url=full_url))
        # Reordered param (action before sourcedoc) — not a raw prefix, so Strategy 2 won't match.
        # Normalization sorts params, so if the subset params match, Strategy 5 picks it up.
        reordered = "https://example.sharepoint.com/sites/hr/_layouts/15/Doc.aspx?action=edit&sourcedoc=%7BGUID-X%7D"
        assert registry.resolve_url(reordered) == full_url

    def test_resolve_url_no_query_params_matched_by_prefix(self, registry):
        """Dropping ALL query params is handled by prefix match (step 2), not query-subset."""
        registry.add(SourceEntry(url="https://example.com/doc?id=123&mode=view"))
        # The path-only URL is a prefix of the full normalized URL → prefix match succeeds
        assert registry.resolve_url("https://example.com/doc") == "https://example.com/doc?id=123&mode=view"


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestGenericUrlExtractor:
    """Tests for the generic URL extractor (works for all tool output formats)."""

    @pytest.mark.parametrize(
        "candidate",
        [
            "https://literal.example/path](not-a-markdown-target)",
            "https://literal.example/path](https://target.example/incomplete",
        ],
    )
    def test_literal_or_incomplete_markdown_delimiter_is_preserved(self, candidate):
        assert clean_extracted_url(candidate) == candidate

    def test_complete_http_label_and_target_selects_markdown_target(self):
        candidate = "https://label.example/path](https://target.example/article)"

        assert clean_extracted_url(candidate) == "https://target.example/article"

    @pytest.mark.parametrize("punctuation", [".", ",", ";"])
    def test_complete_markdown_link_followed_by_punctuation_selects_target(self, punctuation):
        content = f"[https://label.example/path](https://target.example/article){punctuation}"

        assert extract_http_urls(content) == ["https://target.example/article"]

    def test_markdown_target_preserves_balanced_parentheses(self):
        content = "[https://label.example/path](https://en.wikipedia.org/wiki/Function_(mathematics))."

        assert extract_http_urls(content) == ["https://en.wikipedia.org/wiki/Function_(mathematics)"]

    def test_extract_http_urls_ignores_non_http_schemes(self):
        content = "ftp://files.example/archive mailto:user@example.com https://secure.example http://plain.example"

        assert extract_http_urls(content) == ["https://secure.example", "http://plain.example"]

    def test_tavily_xml_format(self):
        content = (
            '<Document href="https://example.com/article">\n'
            "<title>\nTest Article\n</title>\n"
            "Some content here.\n</Document>"
        )
        entries = extract_sources_from_tool_result("tavily_web_search", content)
        assert len(entries) == 1
        assert entries[0].url == "https://example.com/article"

    def test_escaped_document_url_and_title_round_trip(self):
        url = 'https://example.com/search?q="quoted"&next=<unsafe>&close=</Document>'
        title = 'Research & "Roadmap" <2026> </title>'
        content = (
            f'<Document href="{escape(url, quote=True)}">\n'
            f"<title>\n{escape(title, quote=True)}\n</title>\n"
            "Evidence\n</Document>"
        )

        entries = extract_sources_from_tool_result("tavily_web_search", content)

        assert len(entries) == 1
        assert entries[0].url == url
        assert entries[0].title == title

    def test_multiple_urls_deduplicated(self):
        content = (
            '<Document href="https://a.com">\nContent A\n</Document>'
            "\n\n---\n\n"
            '<Document href="https://b.com">\nContent B\n</Document>'
        )
        entries = extract_sources_from_tool_result("any_tool", content)
        assert len(entries) == 2
        assert entries[0].url == "https://a.com"
        assert entries[1].url == "https://b.com"

    def test_paper_search_markdown_format(self):
        content = (
            "1. **Attention Is All You Need** (2017)\n"
            "   - **Publication**: NeurIPS\n"
            "   - **Link**: https://arxiv.org/abs/1706.03762"
        )
        entries = extract_sources_from_tool_result("paper_search_tool", content)
        assert len(entries) == 1
        assert entries[0].url == "https://arxiv.org/abs/1706.03762"

    def test_plain_text_with_urls(self):
        """Generic extractor works even for unknown tool formats."""
        content = "Check out https://example.com/page and also https://other.com/doc for details."
        entries = extract_sources_from_tool_result("totally_new_tool", content)
        assert len(entries) == 2

    def test_config_tool_names_work(self):
        """Tools named 'web_search_tool' or 'advanced_web_search_tool' work."""
        content = '<Document href="https://example.com">\nContent\n</Document>'
        for name in ["web_search_tool", "advanced_web_search_tool", "my_custom_search"]:
            entries = extract_sources_from_tool_result(name, content)
            assert len(entries) == 1, f"Failed for tool name: {name}"

    def test_no_results_status_without_source_id_is_not_citable(self):
        """No-results status text is not evidence and should not produce a source."""
        entries = extract_sources_from_tool_result("any_tool", "Search returned no results")
        assert entries == []

    def test_no_results_status_with_source_id_is_not_citable(self):
        """source_id does not make source status text citable evidence."""
        entries = extract_sources_from_tool_result(
            "duckduckgo_news_search_tool",
            "News search returned no results",
            source_id="news_search",
        )
        assert entries == []

    def test_error_status_is_not_citable(self):
        entries = extract_sources_from_tool_result(
            "duckduckgo_news_search_tool",
            "Error: News search failed",
            source_id="news_search",
        )
        assert entries == []

    @pytest.mark.parametrize(
        "content",
        [
            "Error 432: search request rejected",
            "Search Error 432",
            "Search failed with status 429",
            '{"status_code": 429, "error": "rate limited"}',
        ],
    )
    def test_provider_failure_statuses_are_not_citable(self, content):
        entries = extract_sources_from_tool_result("web_search_tool", content, source_id="web_search")

        assert entries == []

    def test_error_payload_urls_are_not_citable(self):
        content = '{"status": 432, "error": "see https://provider.example/errors/432"}'

        entries = extract_sources_from_tool_result("web_search_tool", content, source_id="web_search")

        assert entries == []

    def test_typed_error_result_urls_are_not_citable(self):
        entries = extract_sources_from_tool_result(
            "web_search_tool",
            "See https://provider.example/errors/unknown",
            source_id="web_search",
            result_status="error",
        )

        assert entries == []

    def test_valid_http_status_explanation_remains_citable(self):
        content = "An HTTP 404 status code means not found. See https://developer.mozilla.org/en-US/docs/Web/HTTP/404"

        entries = extract_sources_from_tool_result("web_search_tool", content, source_id="web_search")

        assert len(entries) == 1
        assert entries[0].url == "https://developer.mozilla.org/en-US/docs/Web/HTTP/404"

    def test_duplicate_urls_deduplicated(self):
        content = "See https://example.com/page and also https://example.com/page for reference."
        entries = extract_sources_from_tool_result("any_tool", content)
        assert len(entries) == 1

    def test_multiple_urls_in_same_block_get_correct_titles(self):
        """Each URL should get the title closest to it, not the first title in the block."""
        content = (
            "1. **Spider 2.0: Enterprise Text-to-SQL** (2024)\n"
            "   - Link: https://arxiv.org/abs/2411.07763\n"
            "\n"
            "2. **Spider: A Large-Scale Dataset** (2018)\n"
            "   - Link: https://arxiv.org/abs/2308.15363"
        )
        entries = extract_sources_from_tool_result("paper_search_tool", content)
        assert len(entries) == 2
        # Each URL should have its own title, not both sharing the first title
        titles = {e.url: e.title for e in entries}
        assert titles["https://arxiv.org/abs/2411.07763"] == "Spider 2.0: Enterprise Text-to-SQL"
        assert titles["https://arxiv.org/abs/2308.15363"] == "Spider: A Large-Scale Dataset"

    def test_title_extraction_prefers_preceding_title(self):
        """Title that appears before the URL is preferred over one after."""
        content = "<title>Correct Title</title>\nhttps://example.com/page\n<title>Wrong Title</title>"
        entries = extract_sources_from_tool_result("web_search", content)
        assert len(entries) == 1
        assert entries[0].title == "Correct Title"

    def test_url_with_commas_in_path(self):
        """Commas inside URL paths (e.g., lat/lon coordinates) must not truncate the URL.

        Regression: the generic URL regex previously excluded ``,`` from the
        character class, so an FAA cam URL like
        ``https://weathercams.faa.gov/map/-122.31167,47.22287,10/...`` was
        registered as ``https://weathercams.faa.gov/map/-122.31167``. The LLM
        would then cite the full URL, the verifier would compare full vs.
        truncated, and the citation would be silently removed as
        ``url_not_in_registry``.
        """
        full_url = "https://weathercams.faa.gov/map/-122.31167,47.22287,10/airport/SEA/details/weather"
        content = f'<Document href="{full_url}">\n<title>\nFAA Cam\n</title>\nObservation.\n</Document>'
        entries = extract_sources_from_tool_result("tavily_web_search", content)
        assert len(entries) == 1
        assert entries[0].url == full_url

    def test_trailing_comma_still_trimmed(self):
        """A comma immediately followed by whitespace is still treated as
        sentence punctuation and stripped from the captured URL."""
        content = "See https://example.com/page, then continue reading."
        entries = extract_sources_from_tool_result("any_tool", content)
        assert len(entries) == 1
        assert entries[0].url == "https://example.com/page"

    def test_markdown_bracket_still_terminates(self):
        """``]`` continues to terminate a URL match so markdown links don't
        leak the closing bracket into the captured URL."""
        content = "[See here](https://example.com/page) for details."
        entries = extract_sources_from_tool_result("any_tool", content)
        assert len(entries) == 1
        assert entries[0].url == "https://example.com/page"


class TestKnowledgeLayerParser:
    """Tests for knowledge layer output parser."""

    def test_parse_single_result(self):
        content = (
            "Found 1 relevant document(s):\n\n"
            "--- Result 1 ---\n"
            "Source: report.pdf\n"
            "Page: 15\n"
            "Citation: report.pdf, p.15\n"
            "Content Type: text\n"
            "Relevance Score: 0.85\n\n"
            "Some content from the document."
        )
        entries = extract_sources_from_tool_result("knowledge_retrieval", content)
        assert len(entries) == 1
        assert entries[0].citation_key == "report.pdf, p.15"
        assert entries[0].title == "report.pdf"
        assert entries[0].url is None
        assert entries[0].source_type == "knowledge_layer"

    def test_parse_multiple_results(self):
        content = (
            "Found 2 relevant document(s):\n\n"
            "--- Result 1 ---\n"
            "Source: doc1.pdf\n"
            "Citation: doc1.pdf\n\n"
            "Content 1.\n\n"
            "--- Result 2 ---\n"
            "Source: doc2.pdf\n"
            "Page: 3\n"
            "Citation: doc2.pdf, p.3\n\n"
            "Content 2."
        )
        entries = extract_sources_from_tool_result("knowledge_retrieval", content)
        assert len(entries) == 2
        assert entries[0].citation_key == "doc1.pdf"
        assert entries[1].citation_key == "doc2.pdf, p.3"


class TestParserDispatcher:
    """Tests for parser dispatcher and fallback behavior."""

    def test_tool_without_content_returns_empty(self):
        entries = extract_sources_from_tool_result("weather_observation_tool", "   ", source_id="weather")

        assert entries == []

    def test_data_source_tool_without_urls_registers_tool_result_source(self):
        entries = extract_sources_from_tool_result(
            "weather_observation_tool", "Visibility: 10 miles", source_id="weather"
        )

        assert len(entries) == 1
        assert entries[0].url is None
        assert entries[0].citation_key == "weather_observation_tool"
        assert entries[0].source_type == "tool_result"
        assert entries[0].tool_name == "weather_observation_tool"

    def test_unknown_tool_with_urls_extracts_them(self):
        """Generic fallback extracts URLs from any unknown tool."""
        entries = extract_sources_from_tool_result("future_tool", "See https://example.com for details")
        assert len(entries) == 1
        assert entries[0].url == "https://example.com"

    def test_custom_parser_takes_priority(self):
        """Registered parsers take priority over generic fallback."""

        def _parse_custom(content: str, tool_name: str) -> list[SourceEntry]:
            return [SourceEntry(url="https://custom.com", source_type="custom", tool_name=tool_name)]

        register_source_parser(lambda name: "my_custom" in name, _parse_custom)
        entries = extract_sources_from_tool_result("my_custom_tool", "anything")
        assert len(entries) == 1
        assert entries[0].url == "https://custom.com"
        assert entries[0].source_type == "custom"


# ---------------------------------------------------------------------------
# verify_citations tests
# ---------------------------------------------------------------------------


class TestVerifyCitations:
    """Tests for verify_citations()."""

    @pytest.fixture(name="registry")
    def fixture_registry(self):
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://valid.com/article1", title="Article 1", source_type="tavily"))
        reg.add(SourceEntry(url="https://valid.com/article2", title="Article 2", source_type="tavily"))
        reg.add(SourceEntry(citation_key="report.pdf, p.15", title="report.pdf", source_type="knowledge_layer"))
        return reg

    def test_verification_and_sanitization_logs_do_not_expose_source_locators(self, caplog):
        registry = SourceRegistry()
        private_url = "https://private.example/document?access_token=secret-value"
        rejected_url = "https://fabricated.example/private-path"
        citation_key = "confidential-plan.pdf, p.7"
        shortened_url = "https://bit.ly/private-report"
        registry.add(SourceEntry(url=private_url, title="Private", source_type="web"))
        registry.add(SourceEntry(citation_key=citation_key, source_type="knowledge_layer"))
        report = (
            "Supported [1] [2], unsupported [3].\n\n"
            "## Sources\n"
            f"[1] Private: {private_url}\n"
            f"[2] {citation_key}\n"
            f"[3] Fabricated: {rejected_url}"
        )

        with caplog.at_level(logging.DEBUG, logger="deep_researcher_agent.common.citation_verification"):
            verify_citations(report, registry)
            sanitize_report(f"Short link [1].\n\n## Sources\n[1] Short: {shortened_url}")

        assert private_url not in caplog.text
        assert rejected_url not in caplog.text
        assert citation_key not in caplog.text
        assert shortened_url not in caplog.text

    def test_empty_registry_returns_unchanged(self):
        registry = SourceRegistry()
        report = "Some report with [1] citations.\n\n## Sources\n[1] Fake: https://fake.com"
        result = verify_citations(report, registry)
        assert result.verified_report == report
        assert len(result.removed_citations) == 0

    def test_no_references_section_returns_unchanged(self, registry):
        report = "A report without any references section."
        result = verify_citations(report, registry)
        assert result.verified_report == report

    def test_missing_references_section_with_inline_citations_does_not_guess_from_registry_order(self, registry):
        report = "Finding one [1]. Finding two [2]."
        result = verify_citations(report, registry)

        assert result.verified_report == report
        assert not result.valid_citations
        assert not result.removed_citations

    def test_missing_references_section_with_reference_sources_appends_sources(self, registry):
        report = "Finding one [1]. Finding two [2]."
        result = verify_citations(report, registry, reference_sources=registry.all_sources())

        assert result.verified_report.startswith(report)
        assert "## Sources" in result.verified_report
        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_missing_references_section_strips_unresolved_inline_citations(self, registry):
        report = "Finding one [1]. Missing source [3]. Finding two [2]."
        result = verify_citations(report, registry, reference_sources=registry.all_sources()[:2])

        assert "Missing source." in result.verified_report
        assert "[3]" not in result.verified_report
        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_missing_references_section_uses_reference_sources_not_registry_order(self):
        reg = SourceRegistry()
        unused = SourceEntry(url="https://valid.com/unused", title="Unused", source_type="tavily")
        compact_one = SourceEntry(url="https://valid.com/compact-one", title="Compact One", source_type="tavily")
        compact_two = SourceEntry(url="https://valid.com/compact-two", title="Compact Two", source_type="tavily")
        reg.add(unused)
        reg.add(compact_one)
        reg.add(compact_two)

        report = "Finding one [1]. Finding two [2]."
        result = verify_citations(report, reg, reference_sources=[compact_one, compact_two])

        assert "[1] Compact One: https://valid.com/compact-one" in result.verified_report
        assert "[2] Compact Two: https://valid.com/compact-two" in result.verified_report
        assert "Unused" not in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_footnote_inline_citations_are_normalized_before_appending_sources(self, registry):
        report = "Finding one [^1]. Finding two [^2]."
        result = verify_citations(report, registry, reference_sources=registry.all_sources())

        assert "Finding one [1]. Finding two [2]." in result.verified_report
        assert "[^1]" not in result.verified_report
        assert "## Sources" in result.verified_report
        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_footnote_reference_lines_are_normalized(self, registry):
        report = (
            "Finding one [^1]. Finding two [^2].\n\n"
            "## Sources\n"
            "[^1]: Article 1: https://valid.com/article1\n"
            "[^2]: Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)

        assert "Finding one [1]. Finding two [2]." in result.verified_report
        assert "[^" not in result.verified_report
        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_source_location_citations_are_normalized_before_appending_sources(self, registry):
        report = "Finding one \u30101\u2020L2-L4\u3011. Finding two \u30102\u2020L56-L60\u3011."
        result = verify_citations(report, registry, reference_sources=registry.all_sources())

        assert "Finding one [1]. Finding two [2]." in result.verified_report
        assert "\u2020" not in result.verified_report
        assert "## Sources" in result.verified_report
        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_source_location_reference_lines_are_normalized(self, registry):
        report = (
            "Finding one \u30101\u2020L2-L4\u3011. Finding two \u30102\u2020L56-L60\u3011.\n\n"
            "## Sources\n"
            "\u30101\u2020L2-L4\u3011 Article 1: https://valid.com/article1\n"
            "\u30102\u2020L56-L60\u3011 Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)

        assert "Finding one [1]. Finding two [2]." in result.verified_report
        assert "\u2020" not in result.verified_report
        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_valid_citations_preserved(self, registry):
        report = (
            "Finding one [1]. Finding two [2].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)
        assert "[1]" in result.verified_report
        assert "[2]" in result.verified_report
        assert len(result.valid_citations) == 2
        assert len(result.removed_citations) == 0

    def test_plain_sources_heading_and_collapsed_lines_are_normalized(self, registry):
        report = (
            "Finding one [1]. Finding two [2].\n\n"
            "Sources\n"
            "[1] Article 1: https://valid.com/article1 [2] Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)

        assert "## Sources\n[1] Article 1: https://valid.com/article1\n" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_bracketed_year_in_source_title_is_not_split(self, registry):
        report = "Finding [1].\n\n## Sources\n[1] Semiconductor outlook [2024] update: https://valid.com/article1"
        result = verify_citations(report, registry)

        assert "[1] Semiconductor outlook [2024] update: https://valid.com/article1" in result.verified_report
        assert len(result.valid_citations) == 1
        assert not result.removed_citations

    def test_bare_sources_line_does_not_terminate_body_before_marked_sources(self, registry):
        report = (
            "Summary.\n"
            "Sources\n"
            "This sentence is part of the report body and should stay there [1].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1"
        )
        result = verify_citations(report, registry)

        assert result.verified_report.startswith(
            "Summary.\nSources\nThis sentence is part of the report body and should stay there [1]."
        )
        assert len(result.valid_citations) == 1
        assert not result.removed_citations

    def test_ordered_list_references_are_normalized_and_verified(self, registry):
        report = (
            "Finding one [1]. Finding two [2].\n\n"
            "## Sources\n"
            "1. Article 1: https://valid.com/article1\n"
            "2. Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)

        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert "[2] Article 2: https://valid.com/article2" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_invalid_ordered_list_reference_is_removed(self, registry):
        report = (
            "Good finding [1]. Bad finding [2].\n\n"
            "## Sources\n"
            "1. Article 1: https://valid.com/article1\n"
            "2. Fake Source: https://fake.com/nonexistent"
        )
        result = verify_citations(report, registry)

        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["number"] == 2
        assert result.removed_citations[0]["reason"] == "url_not_in_registry"
        assert "Good finding [1]. Bad finding." in result.verified_report
        assert "Fake Source" not in result.verified_report

    def test_ordered_list_parenthesis_references_are_normalized_and_verified(self, registry):
        report = "Finding [1].\n\n## Sources\n1) Article 1: https://valid.com/article1"
        result = verify_citations(report, registry)

        assert "[1] Article 1: https://valid.com/article1" in result.verified_report
        assert len(result.valid_citations) == 1
        assert not result.removed_citations

    def test_url_in_markdown_brackets_still_verifies(self, registry):
        """Regression: when the LLM wraps a citation URL in markdown brackets
        (``[https://valid.com/article1]``), the verifier captured the trailing
        ``]`` as part of the URL and then failed to resolve it against the
        registry, silently removing an otherwise-valid citation."""
        report = "Finding [1].\n\n## Sources\n[1] Article 1: [https://valid.com/article1]"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 0

    def test_url_in_angle_brackets_still_verifies(self, registry):
        """Same idea as above for ``<https://...>`` Markdown-style autolinks."""
        report = "Finding [1].\n\n## Sources\n[1] Article 1: <https://valid.com/article1>"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 0

    def test_invalid_citation_removed(self, registry):
        report = (
            "Good finding [1]. Bad finding [2].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Fake Source: https://fake.com/nonexistent"
        )
        result = verify_citations(report, registry)
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["number"] == 2
        assert result.removed_citations[0]["reason"] == "url_not_in_registry"
        # Body should have [2] removed
        assert "[2]" not in result.verified_report
        # [1] stays as [1]
        assert "[1]" in result.verified_report

    def test_unreferenced_inline_citation_removed(self, registry):
        report = "Good finding [1]. Missing reference [3].\n\n## Sources\n[1] Article 1: https://valid.com/article1"
        result = verify_citations(report, registry)

        assert "Missing reference." in result.verified_report
        assert "[3]" not in result.verified_report
        assert len(result.valid_citations) == 1
        assert not result.removed_citations

    def test_unreferenced_inline_citation_removed_with_invalid_reference(self, registry):
        report = (
            "Good finding [1]. Bad finding [2]. Missing reference [3].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Fake Source: https://fake.com/nonexistent"
        )
        result = verify_citations(report, registry)

        assert "Bad finding." in result.verified_report
        assert "Missing reference." in result.verified_report
        assert "[2]" not in result.verified_report
        assert "[3]" not in result.verified_report
        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 1

    def test_removal_leaves_gaps_for_sanitize(self, registry):
        """verify_citations removes invalid refs but does NOT renumber — gaps are left for sanitize_report."""
        report = (
            "A [1]. B [2]. C [3].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Fake: https://fake.com\n"
            "[3] Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)
        assert "A [1]" in result.verified_report
        assert "[2]" not in result.verified_report
        # [3] is NOT renumbered — gaps are closed by sanitize_report()
        assert "C [3]" in result.verified_report

    def test_knowledge_layer_citation_validated(self, registry):
        report = "Internal finding [1].\n\n## Sources\n[1] report.pdf, p.15"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 0

    def test_knowledge_layer_citation_fuzzy_match(self, registry):
        report = "Finding [1].\n\n## Sources\n[1] report.pdf, page 15"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1

    def test_all_citations_removed(self, registry):
        report = "Bad finding [1].\n\n## Sources\n[1] Totally Fake: https://fake.com/nothing"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 0
        assert len(result.removed_citations) == 1
        assert "[1]" not in result.verified_report

    def test_grouped_inline_citations(self, registry):
        report = (
            "Finding [1][2][3].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Fake: https://fake.com\n"
            "[3] Article 2: https://valid.com/article2"
        )
        result = verify_citations(report, registry)
        # [2] removed, [3] stays (renumbering deferred to sanitize_report)
        assert "Finding [1][3]." in result.verified_report

    def test_references_with_dashes(self, registry):
        """Shallow researcher uses '- [N] Title - URL' format."""
        report = "Finding [1].\n\n**References:**\n- [1] Article 1 - https://valid.com/article1"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1

    def test_references_with_markdown_link_urls(self, registry):
        report = (
            "Finding [1][2].\n\n"
            "**References:**\n"
            "- [1] Article 1 - [https://valid.com/article1](https://valid.com/article1)\n"
            "- [2] Article 2 - [https://valid.com/article2](https://valid.com/article2)"
        )

        result = verify_citations(report, registry)

        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_references_bold_without_colon(self, registry):
        """Model sometimes outputs **References** without the colon."""
        report = "Finding [1].\n\n**References**\n- [1] Article 1 - https://valid.com/article1"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1

    def test_references_bold_with_trailing_spaces(self, registry):
        """Model outputs **References** followed by trailing spaces."""
        report = "Finding [1].\n\n**References**  \n- [1] Article 1 - https://valid.com/article1"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1

    def test_references_with_hash_header_variants(self, registry):
        """Test ### Sources header variant."""
        report = "Finding [1].\n\n### Sources\n[1] Article 1: https://valid.com/article1"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1

    def test_unverifiable_citation_removed(self, registry):
        """Citation with no URL and no recognizable citation key is removed."""
        report = (
            "Finding [1]. Other [2].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Some vague reference with no URL"
        )
        result = verify_citations(report, registry)
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["reason"] == "unverifiable"

    def test_knowledge_citation_with_internal_label(self, registry):
        """Knowledge citation with '(Internal)' suffix."""
        report = "Finding [1].\n\n**References:**\n- [1] report.pdf, p.15 (Internal)"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1

    def test_knowledge_citation_with_markdown_italics(self, registry):
        """LLM wraps citation in markdown italics *filename.pdf*."""
        report = "Finding [1].\n\n**References**\n- [1] *report.pdf*, p.15"
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 0

    def test_garbled_url_repaired_to_canonical(self):
        """LLM truncated a URL — verify_citations repairs it to the full canonical URL."""
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://example.com/papers/deep-learning-2024", source_type="generic"))
        report = "Finding [1].\n\n## Sources\n[1] Paper: https://example.com/papers/deep-learning"
        result = verify_citations(report, reg)
        assert len(result.valid_citations) == 1
        # The truncated URL should be repaired to the full canonical URL
        assert "https://example.com/papers/deep-learning-2024" in result.verified_report

    def test_garbled_url_domain_only_repaired(self):
        """LLM dropped entire path — repair to full URL."""
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://arxiv.org/abs/1706.03762", source_type="generic"))
        report = "Finding [1].\n\n## Sources\n[1] Paper: https://arxiv.org"
        result = verify_citations(report, reg)
        assert len(result.valid_citations) == 1
        assert "https://arxiv.org/abs/1706.03762" in result.verified_report

    def test_duplicate_tool_result_refs_collapse_to_single_citation(self):
        """Two [N] lines that resolve to the same non-URL tool_result source are merged.

        The model often makes the same tool call twice (e.g. for two
        timezones) and emits two separate ``[N] mcp_time__get_current_time``
        reference lines. Both lines resolve to the single registered
        SourceEntry for that tool, so verify_citations should keep one and
        rewrite the body's ``[2]`` to ``[1]`` so the prose still cites the
        source.
        """
        reg = SourceRegistry()
        reg.add(
            SourceEntry(
                citation_key="mcp_time__get_current_time",
                source_type="tool_result",
                tool_name="mcp_time__get_current_time",
            )
        )
        report = (
            "Time in Mumbai [1].\n"
            "Time in Tokyo [2].\n\n"
            "**References**\n"
            "- [1] mcp_time__get_current_time\n"
            "- [2] mcp_time__get_current_time"
        )

        result = verify_citations(report, reg)

        assert len(result.valid_citations) == 1
        # The duplicate is recorded as removed for audit, with a clear reason.
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["reason"].startswith("duplicate_of_citation_")
        # Body keeps a citation for both sentences — the second [2] is
        # rewritten to [1] rather than stripped, since the source is real.
        assert "Time in Mumbai [1]." in result.verified_report
        assert "Time in Tokyo [1]." in result.verified_report
        # Reference section keeps exactly one entry for the source.
        ref_section = result.verified_report.split("## Sources", 1)[1]
        assert ref_section.count("mcp_time__get_current_time") == 1
        assert "[2]" not in ref_section

    def test_duplicate_url_refs_collapse_to_single_citation(self):
        """Two [N] lines pointing at the same URL are merged.

        Latent variant of the tool_result case: even URL-based citations
        should collapse when the model emits the same source twice.
        """
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://valid.com/article1", title="Article 1", source_type="tavily"))
        report = (
            "Finding A [1]. Finding B [2].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Article 1: https://valid.com/article1"
        )

        result = verify_citations(report, reg)

        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["reason"].startswith("duplicate_of_citation_")
        assert "Finding A [1]." in result.verified_report
        assert "Finding B [1]." in result.verified_report
        ref_section = result.verified_report.split("## Sources", 1)[1]
        assert ref_section.count("[1]") == 1
        assert "[2]" not in ref_section

    def test_dedup_keeps_lowest_number_when_duplicates_appear_after_unique(self):
        """Mixed: [1] valid URL A, [2] dup of [1], [3] valid URL B.

        After verify_citations, [2] should be merged into [1] and [3] should
        survive. Renumbering happens later in sanitize_report.
        """
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://valid.com/article1", title="Article 1", source_type="tavily"))
        reg.add(SourceEntry(url="https://valid.com/article2", title="Article 2", source_type="tavily"))
        report = (
            "A [1]. B [2]. C [3].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Article 1: https://valid.com/article1\n"
            "[3] Article 2: https://valid.com/article2"
        )

        result = verify_citations(report, reg)

        assert len(result.valid_citations) == 2
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["number"] == 2
        # Body: B's [2] becomes [1]; A and C unchanged.
        assert "A [1]. B [1]. C [3]." in result.verified_report
        ref_section = result.verified_report.split("## Sources", 1)[1]
        assert ref_section.count("[1]") == 1
        assert "[2]" not in ref_section
        assert ref_section.count("[3]") == 1


class TestVerifyCitationsBackfill:
    """Backfill of URL-less ``[N] Title`` lines from the writer-facing list.

    The writer sometimes emits ``[N] Title`` and drops the ``: url`` suffix.
    These tests cover recovering the URL from ``reference_sources`` (same
    captured registry) without weakening precision.
    """

    @pytest.fixture(name="registry")
    def fixture_registry(self):
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://amd.com/q1-2024", title="AMD Q1 2024 Financial Results", source_type="tavily"))
        reg.add(SourceEntry(url="https://meta.com/q1-2024", title="Meta Q1 2024 Results", source_type="tavily"))
        reg.add(SourceEntry(citation_key="report.pdf, p.15", title="Internal Report", source_type="knowledge_layer"))
        return reg

    def test_url_less_lines_backfilled_from_reference_sources(self, registry):
        report = (
            "AMD spent more [1]. Meta followed [2].\n\n"
            "## Sources\n"
            "[1] AMD Q1 2024 Financial Results\n"
            "[2] Meta Q1 2024 Results"
        )
        result = verify_citations(report, registry, reference_sources=registry.all_sources())

        assert "[1] AMD Q1 2024 Financial Results: https://amd.com/q1-2024" in result.verified_report
        assert "[2] Meta Q1 2024 Results: https://meta.com/q1-2024" in result.verified_report
        assert len(result.valid_citations) == 2
        assert not result.removed_citations

    def test_backfill_recovers_url_less_citation_key_source(self, registry):
        report = "Per the internal report [1].\n\n## Sources\n[1] Internal Report"
        result = verify_citations(report, registry, reference_sources=registry.all_sources())

        assert len(result.valid_citations) == 1
        assert result.valid_citations[0]["citation_key"] == "report.pdf, p.15"
        assert not result.removed_citations
        assert "[1]" in result.verified_report.split("## Sources", 1)[1]

    def test_url_less_line_without_matching_reference_still_removed(self, registry):
        report = "Claim [1].\n\n## Sources\n[1] Some Source The Writer Invented"
        result = verify_citations(report, registry, reference_sources=registry.all_sources())

        assert not result.valid_citations
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["reason"] == "unverifiable"

    def test_ambiguous_title_not_backfilled(self):
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://a.com/one", title="Quarterly Update", source_type="tavily"))
        reg.add(SourceEntry(url="https://b.com/two", title="Quarterly Update", source_type="tavily"))
        report = "Claim [1].\n\n## Sources\n[1] Quarterly Update"
        result = verify_citations(report, reg, reference_sources=reg.all_sources())

        assert not result.valid_citations
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["reason"] == "unverifiable"

    def test_aggregate_label_not_resurrected(self):
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://cnn.com/biz", title="CNN", source_type="tavily"))
        report = "Markets moved [1].\n\n## Sources\n[1] CNN; Yahoo Finance; Barchart"
        result = verify_citations(report, reg, reference_sources=reg.all_sources())

        assert not result.valid_citations
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["reason"] == "unverifiable"

    def test_existing_url_and_key_paths_unchanged_without_reference_sources(self, registry):
        report = (
            "AMD [1]. Doc [2].\n\n"
            "## Sources\n"
            "[1] AMD Q1 2024 Financial Results: https://amd.com/q1-2024\n"
            "[2] report.pdf, p.15"
        )
        result = verify_citations(report, registry)

        assert len(result.valid_citations) == 2
        assert not result.removed_citations
        assert "https://amd.com/q1-2024" in result.verified_report


# ---------------------------------------------------------------------------
# sanitize_report tests
# ---------------------------------------------------------------------------


class TestSanitizeReport:
    """Tests for deterministic report sanitization."""

    def test_body_url_not_in_refs_stripped(self):
        """Bare body URL with no matching reference is removed."""
        report = (
            "NVIDIA is great (see https://nvidia.com/gpus for details) [1].\n\n"
            "## Sources\n"
            "[1] NVIDIA: https://nvidia.com/article"
        )
        result = sanitize_report(report)
        assert "https://nvidia.com/gpus" not in result.sanitized_report
        # URL in references preserved
        assert "https://nvidia.com/article" in result.sanitized_report
        assert result.body_urls_removed == 1
        assert result.body_urls_replaced == 0

    def test_body_url_with_commas_matched_to_reference(self):
        """Regression: ``_BODY_URL_RE`` previously stopped at the first comma,
        so a bare body URL with commas in its path was truncated, only the
        prefix got replaced with ``[N]``, and the rest of the URL was left as
        dangling text in the sanitized report."""
        full_url = "https://weathercams.faa.gov/map/-122.31167,47.22287,10/airport/SEA/details/weather"
        report = f"Live cam at {full_url} confirms it [1].\n\n## Sources\n[1] FAA cam: {full_url}"
        result = sanitize_report(report)
        assert "Live cam at [1] confirms it [1]" in result.sanitized_report
        # No fragment of the URL should be left dangling in the body
        assert ",47.22287" not in result.sanitized_report.split("## Sources", 1)[0]
        assert result.body_urls_replaced == 1

    def test_body_url_matching_ref_replaced_with_citation(self):
        """Bare body URL matching a reference is replaced with [N]."""
        report = (
            "Visit https://arxiv.org/abs/1706.03762 for the paper [1].\n\n"
            "## Sources\n"
            "[1] Paper: https://arxiv.org/abs/1706.03762"
        )
        result = sanitize_report(report)
        # Body URL replaced with citation number
        assert "Visit [1] for the paper [1]" in result.sanitized_report
        # Reference section URL preserved
        assert "[1] Paper: https://arxiv.org/abs/1706.03762" in result.sanitized_report
        assert result.body_urls_replaced == 1
        assert result.body_urls_removed == 0

    def test_markdown_links_collapsed_to_text_in_body(self):
        """Markdown hyperlinks [text](url) should be collapsed to display text."""
        report = (
            "Read the [NVIDIA docs](https://nvidia.com/docs/guide) for details [1].\n\n"
            "## Sources\n"
            "[1] Article: https://example.com/article"
        )
        result = sanitize_report(report)
        assert "NVIDIA docs" in result.sanitized_report
        assert "https://nvidia.com/docs/guide" not in result.sanitized_report

    def test_body_without_urls_unchanged(self):
        report = (
            "Finding [1]. Another finding [2].\n\n"
            "## Sources\n"
            "[1] Title: https://example.com/article\n"
            "[2] Title: https://other.com/page"
        )
        result = sanitize_report(report)
        assert result.body_urls_removed == 0
        assert "Finding [1]" in result.sanitized_report

    def test_plain_sources_heading_and_collapsed_lines_are_sanitized(self):
        report = (
            "Finding [1]. Another finding [2].\n\n"
            "Sources\n"
            "[1] Title: https://example.com/article [2] Title: https://other.com/page"
        )
        result = sanitize_report(report)

        assert "## Sources\n[1] Title: https://example.com/article\n" in result.sanitized_report
        assert "[2] Title: https://other.com/page" in result.sanitized_report

    def test_bracketed_year_in_sanitized_source_title_is_not_split(self):
        report = "Finding [1].\n\n## Sources\n[1] Semiconductor outlook [2024] update: https://example.com/article"
        result = sanitize_report(report)

        assert "[1] Semiconductor outlook [2024] update: https://example.com/article" in result.sanitized_report

    def test_bare_sources_line_does_not_terminate_sanitized_body_before_marked_sources(self):
        report = (
            "Summary.\n"
            "Sources\n"
            "This sentence is part of the report body and should stay there [1].\n\n"
            "## Sources\n"
            "[1] Title: https://example.com/article"
        )
        result = sanitize_report(report)

        assert result.sanitized_report.startswith(
            "Summary.\nSources\nThis sentence is part of the report body and should stay there [1]."
        )
        assert "## Sources\n[1] Title: https://example.com/article" in result.sanitized_report

    def test_shortened_url_removed_from_references(self):
        report = "Finding [1].\n\n## Sources\n[1] Article: https://bit.ly/abc123"
        result = sanitize_report(report)
        assert "bit.ly" not in result.sanitized_report
        assert len(result.shortened_urls_removed) == 1
        assert "bit.ly" in result.shortened_urls_removed[0]

    def test_tco_shortened_url_removed(self):
        report = "Finding [1].\n\n## Sources\n[1] Tweet: https://t.co/xyz789"
        result = sanitize_report(report)
        assert "t.co" not in result.sanitized_report
        assert len(result.shortened_urls_removed) == 1

    def test_ip_address_url_removed(self):
        report = "Finding [1].\n\n## Sources\n[1] Suspicious: https://192.168.1.1/malware"
        result = sanitize_report(report)
        assert "192.168.1.1" not in result.sanitized_report
        assert len(result.unsafe_urls_removed) == 1

    def test_legitimate_urls_preserved(self):
        report = (
            "Finding [1] and [2].\n\n"
            "## Sources\n"
            "[1] Paper: https://arxiv.org/abs/1706.03762\n"
            "[2] News: https://www.reuters.com/technology/nvidia-2026"
        )
        result = sanitize_report(report)
        assert "arxiv.org" in result.sanitized_report
        assert "reuters.com" in result.sanitized_report
        assert result.body_urls_removed == 0
        assert len(result.shortened_urls_removed) == 0
        assert len(result.unsafe_urls_removed) == 0

    def test_multiple_shorteners_all_removed(self):
        report = (
            "Finding [1][2][3].\n\n"
            "## Sources\n"
            "[1] A: https://bit.ly/abc\n"
            "[2] B: https://tinyurl.com/def\n"
            "[3] C: https://arxiv.org/real-paper"
        )
        result = sanitize_report(report)
        assert "bit.ly" not in result.sanitized_report
        assert "tinyurl.com" not in result.sanitized_report
        assert "arxiv.org" in result.sanitized_report
        assert len(result.shortened_urls_removed) == 2

    def test_no_references_section(self):
        """Report without references — only body URL stripping applies."""
        report = "Check https://example.com for more info."
        result = sanitize_report(report)
        assert "https://example.com" not in result.sanitized_report
        assert result.body_urls_removed == 1

    def test_truncated_url_with_ellipsis_removed(self):
        """URL ending in ... is truncated/garbled — entire reference line removed."""
        report = "Finding [1].\n\n## Sources\n[1] Paper: https://arxiv.org/abs/1706.037..."
        result = sanitize_report(report)
        # The entire reference line with the truncated URL should be gone
        assert "[1] Paper:" not in result.sanitized_report
        assert len(result.truncated_urls_removed) == 1

    def test_domain_only_url_preserved(self):
        """Domain-only URLs are legitimate if they came from tool results."""
        report = "Finding [1].\n\n## Sources\n[1] Weather API: https://www.weatherapi.com/"
        result = sanitize_report(report)
        assert "weatherapi.com" in result.sanitized_report
        assert len(result.truncated_urls_removed) == 0

    def test_full_url_not_flagged_as_truncated(self):
        """URLs with actual paths are fine."""
        report = "Finding [1].\n\n## Sources\n[1] Paper: https://arxiv.org/abs/1706.03762"
        result = sanitize_report(report)
        assert "arxiv.org/abs/1706.03762" in result.sanitized_report
        assert len(result.truncated_urls_removed) == 0

    def test_renumbering_closes_gaps_from_verify(self):
        """sanitize_report renumbers to close gaps left by verify_citations."""
        # Simulate output of verify_citations that removed [2]: gaps [1], [3]
        report = (
            "A [1]. C [3].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[3] Article 2: https://valid.com/article2"
        )
        result = sanitize_report(report)
        assert "A [1]" in result.sanitized_report
        assert "C [2]" in result.sanitized_report
        assert "[3]" not in result.sanitized_report

    def test_full_pipeline_verify_then_sanitize(self):
        """End-to-end: verify removes invalid, sanitize renumbers and cleans."""
        registry = SourceRegistry()
        registry.add(SourceEntry(url="https://valid.com/article1", source_type="tavily"))
        registry.add(SourceEntry(url="https://valid.com/article2", source_type="tavily"))
        report = (
            "A [1]. B [2]. C [3].\n\n"
            "## Sources\n"
            "[1] Article 1: https://valid.com/article1\n"
            "[2] Fake: https://fake.com\n"
            "[3] Article 2: https://valid.com/article2"
        )
        verified = verify_citations(report, registry).verified_report
        sanitized = sanitize_report(verified).sanitized_report
        assert "A [1]" in sanitized
        assert "C [2]" in sanitized
        assert "[3]" not in sanitized

    def test_mixed_issues(self):
        """Body URL + shortened reference + valid references."""
        report = (
            "See https://nvidia.com/gpus inline [1][2].\n\n"
            "## Sources\n"
            "[1] Good: https://arxiv.org/abs/paper\n"
            "[2] Short: https://bit.ly/short"
        )
        result = sanitize_report(report)
        assert "https://nvidia.com" not in result.sanitized_report
        assert "arxiv.org/abs/paper" in result.sanitized_report
        assert "bit.ly" not in result.sanitized_report
        # nvidia.com/gpus doesn't match any reference → removed
        assert result.body_urls_removed == 1
        assert len(result.shortened_urls_removed) == 1

    def test_mixed_body_urls_some_match_some_not(self):
        """Body has two URLs: one matches a reference, one doesn't."""
        report = (
            "See https://arxiv.org/abs/paper and https://unknown.com for details [1].\n\n"
            "## Sources\n"
            "[1] Paper: https://arxiv.org/abs/paper"
        )
        result = sanitize_report(report)
        # Matching URL replaced with [1], unknown URL stripped
        assert "See [1] and for details [1]" in result.sanitized_report
        assert "https://unknown.com" not in result.sanitized_report
        assert result.body_urls_replaced == 1
        assert result.body_urls_removed == 1


# ---------------------------------------------------------------------------
# EmptySourceRegistryError
# ---------------------------------------------------------------------------


class TestEmptySourceRegistryError:
    """Tests for EmptySourceRegistryError."""

    def test_default_message(self):
        err = EmptySourceRegistryError()
        assert "no sources were captured" in str(err)
        assert "research" in str(err)

    def test_custom_agent_type(self):
        err = EmptySourceRegistryError("deep research")
        assert "deep research" in str(err)
        assert err.agent_type == "deep research"

    def test_is_exception(self):
        with pytest.raises(EmptySourceRegistryError):
            raise EmptySourceRegistryError("test")


# ---------------------------------------------------------------------------
# Session Registry
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    """Tests for session-scoped registry management."""

    def setup_method(self):
        """Clear session registries before each test."""
        from deep_researcher_agent.common.citation_verification import _session_registries
        from deep_researcher_agent.common.citation_verification import _session_registries_lock

        with _session_registries_lock:
            _session_registries.clear()

    def test_get_or_create_returns_same_instance(self):
        from deep_researcher_agent.common.citation_verification import get_or_create_session_registry

        r1 = get_or_create_session_registry("session-1")
        r2 = get_or_create_session_registry("session-1")
        assert r1 is r2

    def test_different_sessions_different_registries(self):
        from deep_researcher_agent.common.citation_verification import get_or_create_session_registry

        r1 = get_or_create_session_registry("session-a")
        r2 = get_or_create_session_registry("session-b")
        assert r1 is not r2

    def test_contextvar_set_and_get(self):
        from deep_researcher_agent.common.citation_verification import get_session_registry
        from deep_researcher_agent.common.citation_verification import set_session_registry

        assert get_session_registry() is None
        reg = SourceRegistry()
        set_session_registry(reg)
        assert get_session_registry() is reg
        set_session_registry(None)
        assert get_session_registry() is None

    def test_lru_eviction(self):
        from deep_researcher_agent.common.citation_verification import _MAX_SESSION_REGISTRIES
        from deep_researcher_agent.common.citation_verification import _session_registries
        from deep_researcher_agent.common.citation_verification import _session_registries_lock
        from deep_researcher_agent.common.citation_verification import get_or_create_session_registry

        # Fill to max + 10
        for i in range(_MAX_SESSION_REGISTRIES + 10):
            get_or_create_session_registry(f"evict-{i}")
        with _session_registries_lock:
            assert len(_session_registries) == _MAX_SESSION_REGISTRIES

    def test_sources_persist_across_calls(self):
        """Sources added to a session registry persist when retrieved again."""
        from deep_researcher_agent.common.citation_verification import get_or_create_session_registry

        reg = get_or_create_session_registry("persist-test")
        reg.add(SourceEntry(url="https://example.com/first"))

        reg2 = get_or_create_session_registry("persist-test")
        assert reg2.has_url("https://example.com/first")
        assert len(reg2.all_sources()) == 1
