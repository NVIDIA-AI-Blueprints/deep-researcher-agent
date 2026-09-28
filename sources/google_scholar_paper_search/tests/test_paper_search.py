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

"""Tests for PaperSearchTool class."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import google_scholar_paper_search.register as register_module
import pytest
from google_scholar_paper_search.paper_search import PaperSearchProvider
from google_scholar_paper_search.paper_search import PaperSearchTool


class TestPaperSearchToolInit:
    """Tests for PaperSearchTool initialization."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        tool = PaperSearchTool(serper_api_key="test-key")

        assert tool.serper_api_key == "test-key"
        assert tool.timeout == 30  # default
        assert tool.max_results == 10  # default

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        tool = PaperSearchTool(
            serper_api_key="test-key",
            timeout=60,
            max_results=20,
        )

        assert tool.serper_api_key == "test-key"
        assert tool.timeout == 60
        assert tool.max_results == 20

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        tool = PaperSearchTool(serper_api_key="test-key", timeout=120)

        assert tool.timeout == 120

    def test_init_with_custom_max_results(self):
        """Test initialization with custom max_results."""
        tool = PaperSearchTool(serper_api_key="test-key", max_results=50)

        assert tool.max_results == 50

    def test_init_defaults_to_serper_provider(self):
        """Test that provider defaults to serper."""
        tool = PaperSearchTool(serper_api_key="test-key")

        assert tool.provider is PaperSearchProvider.SERPER

    def test_init_with_serpapi_provider(self):
        """Test initialization with serpapi provider."""
        tool = PaperSearchTool(provider="serpapi", serpapi_api_key="serpapi-key")

        assert tool.provider is PaperSearchProvider.SERPAPI
        assert tool.serpapi_api_key == "serpapi-key"  # pragma: allowlist secret

    def test_init_with_searchapi_provider(self):
        """Test initialization with searchapi provider."""
        tool = PaperSearchTool(provider="searchapi", searchapi_api_key="searchapi-key")

        assert tool.provider is PaperSearchProvider.SEARCHAPI
        assert tool.searchapi_api_key == "searchapi-key"  # pragma: allowlist secret

    def test_init_with_serply_provider(self):
        """Test initialization with serply provider."""
        tool = PaperSearchTool(provider="serply", serply_api_key="serply-key")

        assert tool.provider is PaperSearchProvider.SERPLY
        assert tool.serply_api_key == "serply-key"  # pragma: allowlist secret

    def test_init_provider_from_enum(self):
        """Test that provider accepts the enum directly."""
        tool = PaperSearchTool(provider=PaperSearchProvider.SERPAPI, serpapi_api_key="key")

        assert tool.provider is PaperSearchProvider.SERPAPI


class TestFormatResults:
    """Tests for format_results static method."""

    def test_format_results_empty_list(self):
        """Test formatting empty results returns appropriate message."""
        result = PaperSearchTool.format_results([])

        assert result == "No papers found via Google Scholar."

    def test_format_results_single_paper(self, sample_papers):
        """Test formatting a single paper."""
        result = PaperSearchTool.format_results([sample_papers[0]])

        assert "1. **Test Paper 1** (2023)" in result
        assert "**Publication**: Test Journal" in result
        assert "**Citations**: 100" in result
        assert "**Snippet**: This is a test snippet." in result
        assert "**Link**: https://example.com/paper1" in result

    def test_format_results_multiple_papers(self, sample_papers):
        """Test formatting multiple papers."""
        result = PaperSearchTool.format_results(sample_papers)

        assert "1. **Test Paper 1** (2023)" in result
        assert "2. **Test Paper 2** (2024)" in result
        assert "\n\n" in result  # Papers should be separated

    def test_format_results_missing_fields(self):
        """Test formatting papers with missing fields uses defaults."""
        papers = [{"title": "Only Title"}]
        result = PaperSearchTool.format_results(papers)

        assert "1. **Only Title** (Unknown Year)" in result
        assert "**Publication**: " in result
        assert "**Citations**: 0" in result

    def test_format_results_all_fields_missing(self):
        """Test formatting papers with all fields missing."""
        papers = [{}]
        result = PaperSearchTool.format_results(papers)

        assert "1. **Unknown Title** (Unknown Year)" in result


class TestSearch:
    """Tests for search method."""

    @pytest.mark.asyncio
    async def test_search_empty_query(self, paper_search_tool):
        """Test search with empty query returns error."""
        result = await paper_search_tool.search("")

        assert result == "Error: 'query' argument is required"

    @pytest.mark.asyncio
    async def test_search_success(self, paper_search_tool, sample_serper_response):
        """Test successful search with mocked API response."""
        with patch.object(
            paper_search_tool,
            "_search_serper",
            new_callable=AsyncMock,
            return_value=sample_serper_response["organic"],
        ):
            result = await paper_search_tool.search("transformers")

        assert "Attention Is All You Need" in result
        assert "BERT" in result

    @pytest.mark.asyncio
    async def test_search_with_year(self, paper_search_tool, sample_serper_response):
        """Test search with year filter."""
        mock_search = AsyncMock(return_value=sample_serper_response["organic"])
        with patch.object(paper_search_tool, "_search_serper", mock_search):
            await paper_search_tool.search("transformers", year="2023")

        mock_search.assert_called_once_with("transformers", "2023", 10)

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, paper_search_tool):
        """Test search handles timeout error gracefully."""
        with patch.object(
            paper_search_tool,
            "_search_serper",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Request timed out"),
        ):
            result = await paper_search_tool.search("test query")

        assert "Paper search timed out" in result
        assert "30s" in result
        assert "test query" not in result

    @pytest.mark.asyncio
    async def test_search_general_exception(self, paper_search_tool):
        """Test search handles general exceptions gracefully."""
        with patch.object(
            paper_search_tool,
            "_search_serper",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            result = await paper_search_tool.search("test query")

        assert "Paper search failed" in result
        assert "API Error" not in result

    @pytest.mark.asyncio
    async def test_search_with_integer_year(self, paper_search_tool, sample_serper_response):
        """Test search handles integer year by converting to string."""
        mock_search = AsyncMock(return_value=sample_serper_response["organic"])
        with patch.object(paper_search_tool, "_search_serper", mock_search):
            await paper_search_tool.search("transformers", year=2023)

        mock_search.assert_called_once_with("transformers", "2023", 10)

    @pytest.mark.parametrize(
        "provider",
        [
            PaperSearchProvider.SERPER,
            PaperSearchProvider.SERPAPI,
            PaperSearchProvider.SEARCHAPI,
            PaperSearchProvider.SERPLY,
        ],
    )
    @pytest.mark.asyncio
    async def test_search_short_circuits_when_provider_key_missing(self, provider):
        """Direct construction with the selected provider's key missing short-circuits."""
        tool = PaperSearchTool(provider=provider)
        result = await tool.search("transformers")

        assert "unavailable" in result.lower()
        assert provider.value in result

    @pytest.mark.asyncio
    async def test_search_missing_key_does_not_dispatch(self, paper_search_tool):
        """A missing provider key short-circuits before any provider call."""
        paper_search_tool.serper_api_key = None
        with patch.object(
            paper_search_tool,
            "_search_serper",
            new_callable=AsyncMock,
        ) as mock_search:
            result = await paper_search_tool.search("transformers")

        assert "unavailable" in result.lower()
        mock_search.assert_not_called()


class TestSearchSerper:
    """Tests for _search_serper internal method."""

    @pytest.mark.asyncio
    async def test_year_parsing_single_year(self, paper_search_tool):
        """Test year parsing for single year."""
        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            return_value={"organic": []},
        ) as mock_fetch:
            await paper_search_tool._search_serper(  # noqa: SLF001
                "query", year="2023", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        # start_year and end_year should both be "2023"
        assert call_args[0][3] == "2023"  # start_year
        assert call_args[0][4] == "2023"  # end_year

    @pytest.mark.asyncio
    async def test_year_parsing_range(self, paper_search_tool):
        """Test year parsing for year range."""
        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            return_value={"organic": []},
        ) as mock_fetch:
            await paper_search_tool._search_serper(  # noqa: SLF001
                "query", year="2020-2023", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][3] == "2020"  # start_year
        assert call_args[0][4] == "2023"  # end_year

    @pytest.mark.asyncio
    async def test_year_parsing_open_start(self, paper_search_tool):
        """Test year parsing for open start range."""
        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            return_value={"organic": []},
        ) as mock_fetch:
            await paper_search_tool._search_serper(  # noqa: SLF001
                "query", year="-2023", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][3] is None  # start_year
        assert call_args[0][4] == "2023"  # end_year

    @pytest.mark.asyncio
    async def test_year_parsing_open_end(self, paper_search_tool):
        """Test year parsing for open end range."""
        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            return_value={"organic": []},
        ) as mock_fetch:
            await paper_search_tool._search_serper(  # noqa: SLF001
                "query", year="2020-", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][3] == "2020"  # start_year
        assert call_args[0][4] is None  # end_year

    @pytest.mark.asyncio
    async def test_limit_capped_at_50(self, paper_search_tool):
        """Test that limit is capped at 50."""
        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            return_value={"organic": []},
        ):
            result = await paper_search_tool._search_serper(  # noqa: SLF001
                "query", limit=100
            )

        assert result == []  # Empty since mocked

    @pytest.mark.asyncio
    async def test_pagination_multiple_pages(self, paper_search_tool):
        """Test pagination for results requiring multiple pages."""
        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            return_value={"organic": [{"title": "Paper"}]},
        ) as mock_fetch:
            await paper_search_tool._search_serper(  # noqa: SLF001
                "query", limit=25
            )

        # 25 results / 10 per page = 3 pages
        assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_aggregates_results_from_pages(self, paper_search_tool):
        """Test that results from multiple pages are aggregated."""
        page1 = {"organic": [{"title": "Paper 1"}, {"title": "Paper 2"}]}
        page2 = {"organic": [{"title": "Paper 3"}]}

        with patch.object(
            paper_search_tool,
            "_fetch_serper_page",
            new_callable=AsyncMock,
            side_effect=[page1, page2],
        ):
            result = await paper_search_tool._search_serper(  # noqa: SLF001
                "query", limit=20
            )

        assert len(result) == 3
        assert result[0]["title"] == "Paper 1"
        assert result[2]["title"] == "Paper 3"


class TestFetchSerperPage:
    """Tests for _fetch_serper_page internal method."""

    @pytest.mark.asyncio
    async def test_fetch_builds_correct_payload(self, paper_search_tool):
        """Test that fetch builds correct API payload."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"organic": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.post = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await paper_search_tool._fetch_serper_page(  # noqa: SLF001
                query="test query",
                num=10,
                offset=0,
                start_year="2020",
                end_year="2023",
            )

        # Verify post was called with correct arguments
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args[1]
        payload = call_kwargs["json"]

        assert payload["q"] == "test query"
        assert payload["num"] == 10
        assert payload["start"] == 0
        assert payload["as_ylo"] == "2020"
        assert payload["as_yhi"] == "2023"

    @pytest.mark.asyncio
    async def test_fetch_num_capped_at_20(self, paper_search_tool):
        """Test that num parameter is capped at 20 (API limit)."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"organic": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.post = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await paper_search_tool._fetch_serper_page(  # noqa: SLF001
                query="test",
                num=50,  # More than limit
                offset=0,
                start_year=None,
                end_year=None,
            )

        call_kwargs = mock_session.post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["num"] == 20  # Should be capped

    @pytest.mark.asyncio
    async def test_fetch_no_year_params_when_none(self, paper_search_tool):
        """Test that year params are not included when None."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"organic": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.post = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await paper_search_tool._fetch_serper_page(  # noqa: SLF001
                query="test",
                num=10,
                offset=0,
                start_year=None,
                end_year=None,
            )

        call_kwargs = mock_session.post.call_args[1]
        payload = call_kwargs["json"]
        assert "as_ylo" not in payload
        assert "as_yhi" not in payload


class TestExtractYear:
    """Tests for _extract_year static method."""

    def test_extract_year_from_summary(self):
        """Test extracting year from a publication summary string."""
        summary = "JL Harper - Population biology of plants., 1977 - cabdirect.org"
        assert PaperSearchTool._extract_year(summary) == "1977"  # noqa: SLF001

    def test_extract_year_multiple_numbers(self):
        """Test that the year before the source suffix is extracted."""
        summary = "A Author - Journal published 2020, vol 123 - host.com"
        assert PaperSearchTool._extract_year(summary) == "2020"  # noqa: SLF001

    def test_extract_year_no_year(self):
        """Test returns Unknown Year when no year is present."""
        assert PaperSearchTool._extract_year("no year here") == "Unknown Year"  # noqa: SLF001

    def test_extract_year_empty_string(self):
        """Test returns Unknown Year for empty string."""
        assert PaperSearchTool._extract_year("") == "Unknown Year"  # noqa: SLF001

    def test_extract_year_non_string(self):
        """Test returns Unknown Year for non-string input."""
        assert PaperSearchTool._extract_year(None) == "Unknown Year"  # noqa: SLF001
        assert PaperSearchTool._extract_year(123) == "Unknown Year"  # noqa: SLF001

    def test_extract_year_supports_pre_1900(self):
        """Pre-1900 publication years are extracted, not forced to Unknown."""
        summary = "J Smith - An early treatise on chemistry, 1859 - jstor.org"
        assert PaperSearchTool._extract_year(summary) == "1859"  # noqa: SLF001

    def test_extract_year_date_range_picks_publication_year(self):
        """Final year before the source suffix wins, not a range start."""
        summary = "Science Progress in the Twentieth Century (1919-1933 \u2026, 1926 - JSTOR"
        assert PaperSearchTool._extract_year(summary) == "1926"  # noqa: SLF001

    def test_extract_year_ignores_arxiv_identifier(self):
        """An arXiv id prefix is not mistaken for the publication year."""
        summary = "Vaswani et al., Attention Is All You Need, arXiv:1706.03762, 2017 - arxiv.org"
        assert PaperSearchTool._extract_year(summary) == "2017"  # noqa: SLF001

    def test_extract_year_arxiv_id_only_is_unknown(self):
        """A summary carrying only an arXiv id (no year) yields Unknown Year."""
        summary = "Some Author - A preprint, arXiv:1910.12345 - arxiv.org"
        assert PaperSearchTool._extract_year(summary) == "Unknown Year"  # noqa: SLF001

    def test_extract_year_21st_century(self):
        """Test that 21st century years are matched."""
        assert PaperSearchTool._extract_year("2023 - arxiv.org") == "2023"  # noqa: SLF001


class TestNormalizeSerpapi:
    """Tests for _normalize_serpapi static method."""

    def test_normalize_full_result(self, sample_serpapi_response):
        """Test normalizing a complete SerpAPI result."""
        raw = sample_serpapi_response["organic_results"]
        normalized = PaperSearchTool._normalize_serpapi(raw)  # noqa: SLF001

        assert len(normalized) == 2
        first = normalized[0]
        assert first["title"] == "Attention Is All You Need"
        assert first["year"] == "2017"
        assert first["link"] == "https://arxiv.org/abs/1706.03762"
        assert first["snippet"] == "The dominant sequence transduction models..."
        assert "2017" in first["publicationInfo"]
        assert first["citedBy"] == 50000

    def test_normalize_missing_fields(self):
        """Test normalizing results with missing fields uses defaults."""
        raw = [{"title": "Only Title"}]
        normalized = PaperSearchTool._normalize_serpapi(raw)  # noqa: SLF001

        assert normalized[0]["title"] == "Only Title"
        assert normalized[0]["year"] == "Unknown Year"
        assert normalized[0]["snippet"] == ""
        assert normalized[0]["link"] == ""
        assert normalized[0]["publicationInfo"] == ""
        assert normalized[0]["citedBy"] == 0

    def test_normalize_empty_list(self):
        """Test normalizing an empty list."""
        assert PaperSearchTool._normalize_serpapi([]) == []  # noqa: SLF001


class TestNormalizeSearchapi:
    """Tests for _normalize_searchapi static method."""

    def test_normalize_full_result(self, sample_searchapi_response):
        """Test normalizing a complete SearchAPI result."""
        raw = sample_searchapi_response["organic_results"]
        normalized = PaperSearchTool._normalize_searchapi(raw)  # noqa: SLF001

        assert len(normalized) == 2
        first = normalized[0]
        assert first["title"] == "Attention Is All You Need"
        assert first["year"] == "2017"
        assert first["link"] == "https://arxiv.org/abs/1706.03762"
        assert first["snippet"] == "The dominant sequence transduction models..."
        assert "2017" in first["publicationInfo"]
        assert first["citedBy"] == 50000

    def test_normalize_missing_fields(self):
        """Test normalizing results with missing fields uses defaults."""
        raw = [{"title": "Only Title"}]
        normalized = PaperSearchTool._normalize_searchapi(raw)  # noqa: SLF001

        assert normalized[0]["title"] == "Only Title"
        assert normalized[0]["year"] == "Unknown Year"
        assert normalized[0]["citedBy"] == 0

    def test_normalize_empty_list(self):
        """Test normalizing an empty list."""
        assert PaperSearchTool._normalize_searchapi([]) == []  # noqa: SLF001


class TestNormalizeSerply:
    """Tests for _normalize_serply static method."""

    def test_normalize_full_result(self, sample_serply_response):
        """Test normalizing a complete Serply result."""
        raw = sample_serply_response["articles"]
        normalized = PaperSearchTool._normalize_serply(raw)  # noqa: SLF001

        assert len(normalized) == 2
        first = normalized[0]
        assert first["title"] == "Attention Is All You Need"
        assert first["year"] == "2017"
        assert first["link"] == "https://arxiv.org/abs/1706.03762"
        assert "2017" in first["publicationInfo"]
        assert "\u00a0" not in first["publicationInfo"]
        assert first["citedBy"] == 50000

    def test_normalize_drops_description_that_repeats_summary(self, sample_serply_response):
        """A description identical to the publication summary is not used as a snippet."""
        normalized = PaperSearchTool._normalize_serply(  # noqa: SLF001
            sample_serply_response["articles"]
        )

        assert normalized[0]["snippet"] == ""
        assert normalized[1]["snippet"] == "We introduce a new language model..."

    def test_normalize_parses_citation_count_with_separators(self, sample_serply_response):
        """Citation counts like 'Cited by 40,000' parse to an int."""
        normalized = PaperSearchTool._normalize_serply(  # noqa: SLF001
            sample_serply_response["articles"]
        )

        assert normalized[1]["citedBy"] == 40000
        assert normalized[1]["year"] == "2018"

    def test_normalize_missing_fields(self):
        """Test normalizing results with missing fields uses defaults."""
        raw = [{"title": "Only Title"}]
        normalized = PaperSearchTool._normalize_serply(raw)  # noqa: SLF001

        assert normalized[0]["title"] == "Only Title"
        assert normalized[0]["year"] == "Unknown Year"
        assert normalized[0]["snippet"] == ""
        assert normalized[0]["citedBy"] == 0

    def test_normalize_empty_list(self):
        """Test normalizing an empty list."""
        assert PaperSearchTool._normalize_serply([]) == []  # noqa: SLF001


class TestSearchSerpapi:
    """Tests for _search_serpapi internal method."""

    @pytest.mark.asyncio
    async def test_year_parsing_single_year(self, serpapi_tool):
        """Test year parsing passes start/end to fetch."""
        with patch.object(
            serpapi_tool,
            "_fetch_serpapi_page",
            new_callable=AsyncMock,
            return_value={"organic_results": []},
        ) as mock_fetch:
            await serpapi_tool._search_serpapi(  # noqa: SLF001
                "query", year="2023", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][3] == "2023"  # start_year
        assert call_args[0][4] == "2023"  # end_year

    @pytest.mark.asyncio
    async def test_pagination_multiple_pages(self, serpapi_tool):
        """Test pagination for results requiring multiple pages."""
        with patch.object(
            serpapi_tool,
            "_fetch_serpapi_page",
            new_callable=AsyncMock,
            return_value={"organic_results": [{"title": "Paper"}]},
        ) as mock_fetch:
            await serpapi_tool._search_serpapi(  # noqa: SLF001
                "query", limit=25
            )

        assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_normalizes_results(self, serpapi_tool, sample_serpapi_response):
        """Test that raw SerpAPI results are normalized."""
        with patch.object(
            serpapi_tool,
            "_fetch_serpapi_page",
            new_callable=AsyncMock,
            return_value=sample_serpapi_response,
        ):
            result = await serpapi_tool._search_serpapi(  # noqa: SLF001
                "query", limit=10
            )

        assert len(result) == 2
        assert result[0]["title"] == "Attention Is All You Need"
        assert result[0]["citedBy"] == 50000


class TestSearchSearchapi:
    """Tests for _search_searchapi internal method."""

    @pytest.mark.asyncio
    async def test_year_parsing_single_year(self, searchapi_tool):
        """Test year parsing passes start/end to fetch."""
        with patch.object(
            searchapi_tool,
            "_fetch_searchapi_page",
            new_callable=AsyncMock,
            return_value={"organic_results": []},
        ) as mock_fetch:
            await searchapi_tool._search_searchapi(  # noqa: SLF001
                "query", year="2023", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][3] == "2023"  # start_year
        assert call_args[0][4] == "2023"  # end_year

    @pytest.mark.asyncio
    async def test_pagination_multiple_pages(self, searchapi_tool):
        """Test pagination for results requiring multiple pages."""
        with patch.object(
            searchapi_tool,
            "_fetch_searchapi_page",
            new_callable=AsyncMock,
            return_value={"organic_results": [{"title": "Paper"}]},
        ) as mock_fetch:
            await searchapi_tool._search_searchapi(  # noqa: SLF001
                "query", limit=25
            )

        assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_page_is_one_based(self, searchapi_tool):
        """Test that SearchAPI page numbers start at 1, not 0."""
        with patch.object(
            searchapi_tool,
            "_fetch_searchapi_page",
            new_callable=AsyncMock,
            return_value={"organic_results": []},
        ) as mock_fetch:
            await searchapi_tool._search_searchapi(  # noqa: SLF001
                "query", limit=25
            )

        page_args = [call.args[2] for call in mock_fetch.call_args_list]
        assert page_args == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_normalizes_results(self, searchapi_tool, sample_searchapi_response):
        """Test that raw SearchAPI results are normalized."""
        with patch.object(
            searchapi_tool,
            "_fetch_searchapi_page",
            new_callable=AsyncMock,
            return_value=sample_searchapi_response,
        ):
            result = await searchapi_tool._search_searchapi(  # noqa: SLF001
                "query", limit=10
            )

        assert len(result) == 2
        assert result[0]["title"] == "Attention Is All You Need"
        assert result[0]["citedBy"] == 50000


class TestSearchSerply:
    """Tests for _search_serply internal method."""

    @pytest.mark.asyncio
    async def test_year_parsing_single_year(self, serply_tool):
        """Test year parsing passes start/end to fetch."""
        with patch.object(
            serply_tool,
            "_fetch_serply_page",
            new_callable=AsyncMock,
            return_value={"articles": []},
        ) as mock_fetch:
            await serply_tool._search_serply(  # noqa: SLF001
                "query", year="2023", limit=10
            )

        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][3] == "2023"  # start_year
        assert call_args[0][4] == "2023"  # end_year

    @pytest.mark.asyncio
    async def test_pagination_uses_zero_based_offsets(self, serply_tool):
        """Test that Serply pages by result offset, like SerpAPI."""
        with patch.object(
            serply_tool,
            "_fetch_serply_page",
            new_callable=AsyncMock,
            return_value={"articles": [{"title": "Paper"}]},
        ) as mock_fetch:
            await serply_tool._search_serply(  # noqa: SLF001
                "query", limit=25
            )

        offsets = [call.args[2] for call in mock_fetch.call_args_list]
        assert offsets == [0, 10, 20]

    @pytest.mark.asyncio
    async def test_normalizes_results(self, serply_tool, sample_serply_response):
        """Test that raw Serply results are normalized."""
        with patch.object(
            serply_tool,
            "_fetch_serply_page",
            new_callable=AsyncMock,
            return_value=sample_serply_response,
        ):
            result = await serply_tool._search_serply(  # noqa: SLF001
                "query", limit=10
            )

        assert len(result) == 2
        assert result[0]["title"] == "Attention Is All You Need"
        assert result[0]["citedBy"] == 50000


class TestFetchSerpapiPage:
    """Tests for _fetch_serpapi_page internal method."""

    @pytest.mark.asyncio
    async def test_fetch_builds_correct_params(self, serpapi_tool):
        """Test that fetch builds correct GET params."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"organic_results": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.get = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await serpapi_tool._fetch_serpapi_page(  # noqa: SLF001
                query="test query",
                num=10,
                offset=0,
                start_year="2020",
                end_year="2023",
            )

        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]

        assert params["engine"] == "google_scholar"
        assert params["q"] == "test query"
        assert params["num"] == 10
        assert params["start"] == 0
        assert params["as_ylo"] == "2020"
        assert params["as_yhi"] == "2023"
        assert params["api_key"] == "test-serpapi-key"  # pragma: allowlist secret

    @pytest.mark.asyncio
    async def test_fetch_no_year_params_when_none(self, serpapi_tool):
        """Test that year params are not included when None."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"organic_results": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.get = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await serpapi_tool._fetch_serpapi_page(  # noqa: SLF001
                query="test",
                num=10,
                offset=0,
                start_year=None,
                end_year=None,
            )

        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]
        assert "as_ylo" not in params
        assert "as_yhi" not in params


class TestFetchSearchapiPage:
    """Tests for _fetch_searchapi_page internal method."""

    @pytest.mark.asyncio
    async def test_fetch_builds_correct_params(self, searchapi_tool):
        """Test that fetch builds correct GET params with 1-based page."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"organic_results": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.get = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await searchapi_tool._fetch_searchapi_page(  # noqa: SLF001
                query="test query",
                num=10,
                page=2,
                start_year="2020",
                end_year="2023",
            )

        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]

        assert params["engine"] == "google_scholar"
        assert params["q"] == "test query"
        assert params["num"] == 10
        assert params["page"] == 2
        assert params["as_ylo"] == "2020"
        assert params["as_yhi"] == "2023"
        assert params["api_key"] == "test-searchapi-key"  # pragma: allowlist secret


class TestFetchSerplyPage:
    """Tests for _fetch_serply_page internal method."""

    @pytest.mark.asyncio
    async def test_fetch_builds_correct_request(self, serply_tool):
        """Test that fetch sends the key as a header, never as a query param."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"articles": []})

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        )
        mock_session.get = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            await serply_tool._fetch_serply_page(  # noqa: SLF001
                query="test query",
                num=30,
                offset=10,
                start_year="2020",
                end_year="2023",
            )

        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args[0][0] == "https://api.serply.io/v1/scholar"
        params = call_args[1]["params"]
        headers = call_args[1]["headers"]

        assert params["q"] == "test query"
        assert params["num"] == 20
        assert params["start"] == 10
        assert params["as_ylo"] == "2020"
        assert params["as_yhi"] == "2023"
        assert "api_key" not in params
        assert headers["X-Api-Key"] == "test-serply-key"  # pragma: allowlist secret

    @pytest.mark.asyncio
    async def test_fetch_raises_on_http_error(self, serply_tool):
        """Test that a non-200 response raises."""
        mock_response = MagicMock()
        mock_response.status = 401

        mock_session = MagicMock()
        mock_context = MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        )
        mock_session.get = MagicMock(return_value=mock_context)

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="HTTP 401"):
                await serply_tool._fetch_serply_page(  # noqa: SLF001
                    query="q",
                    num=10,
                    offset=0,
                    start_year=None,
                    end_year=None,
                )


class TestProviderDispatch:
    """Tests for provider-based dispatch in search()."""

    @pytest.mark.asyncio
    async def test_search_dispatches_to_serper(self, paper_search_tool):
        """Test that search() dispatches to _search_serper by default."""
        with patch.object(
            paper_search_tool,
            "_search_serper",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_serper:
            await paper_search_tool.search("test query")

        mock_serper.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_dispatches_to_serpapi(self, serpapi_tool):
        """Test that search() dispatches to _search_serpapi."""
        with (
            patch.object(
                serpapi_tool,
                "_search_serpapi",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_serpapi,
            patch.object(
                serpapi_tool,
                "_search_serper",
                new_callable=AsyncMock,
            ) as mock_serper,
        ):
            await serpapi_tool.search("test query")

        mock_serpapi.assert_called_once()
        mock_serper.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_dispatches_to_searchapi(self, searchapi_tool):
        """Test that search() dispatches to _search_searchapi."""
        with (
            patch.object(
                searchapi_tool,
                "_search_searchapi",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_searchapi,
            patch.object(
                searchapi_tool,
                "_search_serper",
                new_callable=AsyncMock,
            ) as mock_serper,
            patch.object(
                searchapi_tool,
                "_search_serpapi",
                new_callable=AsyncMock,
            ) as mock_serpapi,
        ):
            await searchapi_tool.search("test query")

        mock_searchapi.assert_called_once()
        mock_serper.assert_not_called()
        mock_serpapi.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_dispatches_to_serply(self, serply_tool):
        """Test that search() dispatches to _search_serply."""
        with (
            patch.object(
                serply_tool,
                "_search_serply",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_serply,
            patch.object(
                serply_tool,
                "_search_serper",
                new_callable=AsyncMock,
            ) as mock_serper,
        ):
            await serply_tool.search("test query")

        mock_serply.assert_called_once()
        mock_serper.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_serply_success(self, serply_tool, sample_serply_response):
        """Test successful search via Serply produces formatted output."""
        with patch.object(
            serply_tool,
            "_search_serply",
            new_callable=AsyncMock,
            return_value=PaperSearchTool._normalize_serply(  # noqa: SLF001
                sample_serply_response["articles"]
            ),
        ):
            result = await serply_tool.search("transformers")

        assert "Attention Is All You Need" in result
        assert "BERT" in result

    @pytest.mark.asyncio
    async def test_search_serpapi_success(self, serpapi_tool, sample_serpapi_response):
        """Test successful search via SerpAPI produces formatted output."""
        with patch.object(
            serpapi_tool,
            "_search_serpapi",
            new_callable=AsyncMock,
            return_value=PaperSearchTool._normalize_serpapi(  # noqa: SLF001
                sample_serpapi_response["organic_results"]
            ),
        ):
            result = await serpapi_tool.search("transformers")

        assert "Attention Is All You Need" in result
        assert "BERT" in result

    @pytest.mark.asyncio
    async def test_search_searchapi_success(self, searchapi_tool, sample_searchapi_response):
        """Test successful search via SearchAPI produces formatted output."""
        with patch.object(
            searchapi_tool,
            "_search_searchapi",
            new_callable=AsyncMock,
            return_value=PaperSearchTool._normalize_searchapi(  # noqa: SLF001
                sample_searchapi_response["organic_results"]
            ),
        ):
            result = await searchapi_tool.search("transformers")

        assert "Attention Is All You Need" in result
        assert "BERT" in result

    @pytest.mark.asyncio
    async def test_search_serpapi_handles_error(self, serpapi_tool):
        """Test that search() handles errors for SerpAPI provider."""
        with patch.object(
            serpapi_tool,
            "_search_serpapi",
            new_callable=AsyncMock,
            side_effect=Exception("SerpAPI Error"),
        ):
            result = await serpapi_tool.search("test query")

        assert "Paper search failed" in result
        assert "SerpAPI Error" not in result


class TestRegisterMissingKeyStub:
    """Registration-level tests for the missing-API-key stub function."""

    @pytest.mark.parametrize(
        ("provider", "env_var"),
        [
            (PaperSearchProvider.SERPER, "SERPER_API_KEY"),
            (PaperSearchProvider.SERPAPI, "SERPAPI_API_KEY"),
            (PaperSearchProvider.SEARCHAPI, "SEARCHAPI_API_KEY"),
            (PaperSearchProvider.SERPLY, "SERPLY_API_KEY"),
        ],
    )
    @pytest.mark.asyncio
    async def test_stub_description_is_meaningful(self, provider, env_var, monkeypatch):
        """Missing-key FunctionInfo exposes a non-None, provider-aware description."""
        monkeypatch.delenv(env_var, raising=False)
        # Reset the module-level warn guard so prior runs cannot alter behavior.
        monkeypatch.setattr(register_module, "_missing_key_warned", False)

        tool_config = register_module.PaperSearchToolConfig(provider=provider)

        async with register_module.paper_search(tool_config, MagicMock()) as fn:
            description = fn.description

        assert description is not None
        assert provider.value in description
        assert env_var in description
