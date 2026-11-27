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

"""Both tools' outputs must work with the agent's citation verification unchanged.

``deep_researcher_agent.common.citation_verification`` registers sources from
tool output: titles come from ``<title>`` blocks (web search) or ``N. **Title**``
lines (paper search), and status messages must never be cited. These tests run
the real parser over real tool output; they skip when the agent package is not
installed, so the source package stays independently testable.
"""

from unittest.mock import MagicMock

import pytest
from context_dev_search.register import ContextDevPaperSearchToolConfig
from context_dev_search.register import ContextDevWebSearchToolConfig
from context_dev_search.register import context_dev_paper_search
from context_dev_search.register import context_dev_web_search

citation_verification = pytest.importorskip("deep_researcher_agent.common.citation_verification")

ATTENTION_URL = "https://arxiv.org/abs/1706.03762"
NATURE_URL = "https://www.nature.com/articles/nature14539"


async def _web(question: str) -> str:
    async with context_dev_web_search(ContextDevWebSearchToolConfig(), MagicMock()) as info:
        return await info.single_fn(question)


async def _paper(query: str) -> str:
    async with context_dev_paper_search(ContextDevPaperSearchToolConfig(), MagicMock()) as info:
        return await info.single_fn(query)


def _two_results(result_factory) -> list[dict]:
    return [
        result_factory(url=ATTENTION_URL, title="Attention Is All You Need", highlights=["Transformer & attention."]),
        result_factory(url=NATURE_URL, title="Deep learning", highlights=["Representation learning."]),
    ]


@pytest.mark.parametrize("search", [_web, _paper], ids=["web", "paper"])
async def test_sources_register_with_urls_and_titles(api_key_env, mock_api, result_factory, search):
    mock_api.add_results(_two_results(result_factory))
    output = await search("q")

    entries = citation_verification._parse_generic_urls(output, "search_tool")

    assert [(entry.url, entry.title) for entry in entries] == [
        (ATTENTION_URL, "Attention Is All You Need"),
        (NATURE_URL, "Deep learning"),
    ]


@pytest.mark.parametrize("search", [_web, _paper], ids=["web", "paper"])
@pytest.mark.parametrize(
    "scripted",
    [
        pytest.param({"status": 401}, id="invalid-key"),
        pytest.param({"status": 402, "error_code": "USAGE_EXCEEDED"}, id="no-credits"),
        pytest.param({"status": 503}, id="exhausted-retries"),
        pytest.param({"results": []}, id="no-results"),
    ],
)
async def test_status_messages_are_not_citable(api_key_env, mock_api, no_sleep, search, scripted):
    if "results" in scripted:
        mock_api.add_results(scripted["results"])
    else:
        mock_api.add_error(scripted["status"], scripted.get("error_code"))

    output = await search("q")

    assert citation_verification.is_non_citable_status_output(output)


@pytest.mark.parametrize("search", [_web, _paper], ids=["web", "paper"])
async def test_missing_key_stub_is_not_citable(no_api_key_env, search):
    assert citation_verification.is_non_citable_status_output(await search("q"))
