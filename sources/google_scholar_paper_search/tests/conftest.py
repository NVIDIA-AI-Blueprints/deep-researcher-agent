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

"""Pytest fixtures for paper search tests."""

import pytest
from google_scholar_paper_search.paper_search import PaperSearchTool


@pytest.fixture
def paper_search_tool():
    """Create a PaperSearchTool instance (Serper default) for testing."""
    return PaperSearchTool(
        serper_api_key="test-api-key",
        timeout=30,
        max_results=10,
    )


@pytest.fixture
def serpapi_tool():
    """Create a PaperSearchTool instance configured for SerpAPI."""
    return PaperSearchTool(
        provider="serpapi",
        serpapi_api_key="test-serpapi-key",  # pragma: allowlist secret
        timeout=30,
        max_results=10,
    )


@pytest.fixture
def searchapi_tool():
    """Create a PaperSearchTool instance configured for SearchAPI."""
    return PaperSearchTool(
        provider="searchapi",
        searchapi_api_key="test-searchapi-key",  # pragma: allowlist secret
        timeout=30,
        max_results=10,
    )


@pytest.fixture
def serply_tool():
    """Create a PaperSearchTool instance configured for Serply."""
    return PaperSearchTool(
        provider="serply",
        serply_api_key="test-serply-key",  # pragma: allowlist secret
        timeout=30,
        max_results=10,
    )


@pytest.fixture
def sample_serper_response():
    """Sample Serper API response for testing."""
    return {
        "organic": [
            {
                "title": "Attention Is All You Need",
                "year": "2017",
                "snippet": "The dominant sequence transduction models...",
                "link": "https://arxiv.org/abs/1706.03762",
                "publicationInfo": "Advances in neural information...",
                "citedBy": 50000,
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional...",
                "year": "2018",
                "snippet": "We introduce a new language model...",
                "link": "https://arxiv.org/abs/1810.04805",
                "publicationInfo": "arXiv preprint",
                "citedBy": 40000,
            },
        ]
    }


@pytest.fixture
def sample_serpapi_response():
    """Sample SerpAPI response (shape matches serpapi.com docs)."""
    return {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {
                "position": 0,
                "title": "Attention Is All You Need",
                "result_id": "abc123",
                "link": "https://arxiv.org/abs/1706.03762",
                "snippet": "The dominant sequence transduction models...",
                "publication_info": {
                    "summary": "A Vaswani, N Shazeer, N Parmar... - Advances in neural information..., 2017 - arxiv.org"
                },
                "inline_links": {
                    "cited_by": {
                        "total": 50000,
                        "link": "https://scholar.google.com/scholar?cites=123",
                    },
                },
            },
            {
                "position": 1,
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "link": "https://arxiv.org/abs/1810.04805",
                "snippet": "We introduce a new language model...",
                "publication_info": {
                    "summary": "J Devlin, MW Chang, K Lee, K Toutanova - arXiv preprint, 2018 - arxiv.org"
                },
                "inline_links": {
                    "cited_by": {"total": 40000},
                },
            },
        ],
    }


@pytest.fixture
def sample_searchapi_response():
    """Sample SearchAPI response (shape matches searchapi.io docs)."""
    return {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {
                "position": 1,
                "title": "Attention Is All You Need",
                "data_cid": "abc123",
                "link": "https://arxiv.org/abs/1706.03762",
                "publication": (
                    "A Vaswani, N Shazeer, N Parmar... - Advances in neural information..., 2017 - arxiv.org"
                ),
                "snippet": "The dominant sequence transduction models...",
                "inline_links": {
                    "cited_by": {
                        "cites_id": "123",
                        "total": 50000,
                        "link": "https://scholar.google.com/scholar?cites=123",
                    }
                },
            },
            {
                "position": 2,
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "link": "https://arxiv.org/abs/1810.04805",
                "publication": "J Devlin, MW Chang, K Lee, K Toutanova - 2018 - arxiv.org",
                "snippet": "We introduce a new language model...",
                "inline_links": {
                    "cited_by": {"total": 40000},
                },
            },
        ],
    }


@pytest.fixture
def sample_papers():
    """Sample normalized paper data for format testing."""
    return [
        {
            "title": "Test Paper 1",
            "year": "2023",
            "snippet": "This is a test snippet.",
            "link": "https://example.com/paper1",
            "publicationInfo": "Test Journal",
            "citedBy": 100,
        },
        {
            "title": "Test Paper 2",
            "year": "2024",
            "snippet": "Another test snippet.",
            "link": "https://example.com/paper2",
            "publicationInfo": "Another Journal",
            "citedBy": 50,
        },
    ]


@pytest.fixture
def sample_serply_response():
    """Sample Serply response (shape matches api.serply.io/v1/scholar)."""
    return {
        "articles": [
            {
                "title": "Attention Is All You Need",
                "link": "https://arxiv.org/abs/1706.03762",
                "id": "abc123",
                "author": {
                    "names": (
                        "A Vaswani, N Shazeer, N Parmar\u2026\u00a0- Advances in neural\u00a0\u2026, 2017 - arxiv.org"
                    ),
                    "authors": [{"name": "A Vaswani"}],
                },
                "description": (
                    "A Vaswani, N Shazeer, N Parmar\u2026\u00a0- Advances in neural\u00a0\u2026, 2017 - arxiv.org"
                ),
                "extras": {
                    "citations": {
                        "count": "Cited by 50000",
                        "link": "https://scholar.google.com/scholar?cites=123",
                    },
                },
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "link": "https://arxiv.org/abs/1810.04805",
                "author": {"names": "J Devlin, MW Chang, K Lee, K Toutanova - arXiv preprint, 2018 - arxiv.org"},
                "description": "We introduce a new language model...",
                "extras": {"citations": {"count": "Cited by 40,000"}},
            },
        ],
    }
