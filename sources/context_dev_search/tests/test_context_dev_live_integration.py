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

"""Live Context.dev integration tests: the minimal repeatable CI check.

Opt-in (mirrors ``tests/knowledge_layer_tests/test_opensearch_live.py``): set
``DEEP_RESEARCHER_CONTEXT_DEV_LIVE_TESTS=1`` and provide a real
``CONTEXT_DEV_API_KEY``. Cost per run: two API calls (one per tool), each bounded
at 120 seconds.

The tests perform no retries of their own. The tools' built-in retry loop
(``max_retries=3`` with backoff) is part of the code under test, so a failure here
means three consecutive attempts failed, which is the reliability signal CI
should surface.

Run:

    DEEP_RESEARCHER_CONTEXT_DEV_LIVE_TESTS=1 CONTEXT_DEV_API_KEY=<key> \
        uv run pytest sources/context_dev_search/tests -m integration -v
"""

import asyncio
import os
from unittest.mock import MagicMock

import pytest
from context_dev_search.register import ContextDevPaperSearchToolConfig
from context_dev_search.register import ContextDevWebSearchToolConfig
from context_dev_search.register import context_dev_paper_search
from context_dev_search.register import context_dev_web_search

# Time-invariant queries that stay on-topic across time and regions. Keep in sync
# with fixtures/README.md (the recorded fixtures were captured with these queries).
WEB_QUERY = "NVIDIA CUDA Toolkit documentation"
PAPER_QUERY = "paper that replaced recurrence with self-attention for machine translation"


def _env_bool(name: str, default: bool = False) -> bool:
    """env bool."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _env_bool("DEEP_RESEARCHER_CONTEXT_DEV_LIVE_TESTS"),
        reason="Set DEEP_RESEARCHER_CONTEXT_DEV_LIVE_TESTS=1 to run live Context.dev integration tests.",
    ),
    pytest.mark.skipif(
        not os.environ.get("CONTEXT_DEV_API_KEY"),
        reason="CONTEXT_DEV_API_KEY not set; live Context.dev tests require a real key.",
    ),
]


class TestLiveContextDevAPI:
    async def test_live_web_search_returns_parseable_documents(self, assert_web_contract):
        """One live web search with shipped defaults must satisfy the output contract.

        Assertions are structural, never content-exact, so ordinary result
        variation cannot flake the test.
        """
        config = ContextDevWebSearchToolConfig()

        async with context_dev_web_search(config, MagicMock()) as info:
            result = await asyncio.wait_for(info.single_fn(WEB_QUERY), timeout=120)

        assert_web_contract(result, config.max_results)

    async def test_live_paper_search_returns_numbered_papers(self, assert_paper_contract):
        """One live paper search with shipped defaults must satisfy the output contract."""
        config = ContextDevPaperSearchToolConfig()

        async with context_dev_paper_search(config, MagicMock()) as info:
            result = await asyncio.wait_for(info.single_fn(PAPER_QUERY), timeout=120)

        assert_paper_contract(result, config.max_results)
