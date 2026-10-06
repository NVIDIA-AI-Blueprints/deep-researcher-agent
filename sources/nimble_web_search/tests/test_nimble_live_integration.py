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

"""Live Nimble integration test — the minimal repeatable CI check.

Opt-in (mirrors ``tests/knowledge_layer_tests/test_opensearch_live.py``): set
``DEEP_RESEARCHER_NIMBLE_LIVE_TESTS=1`` and provide a real ``NIMBLE_API_KEY``. Cost per
run: exactly one API call, bounded at 120 seconds.

The test performs no retries of its own — the provider's built-in retry loop
(``max_retries=3`` with backoff) is part of the code under test, so a failure
here means three consecutive provider attempts failed, which is precisely the
reliability signal CI should surface.

Run:

    DEEP_RESEARCHER_NIMBLE_LIVE_TESTS=1 NIMBLE_API_KEY=<key> \
        uv run pytest sources/nimble_web_search/tests -m integration -v
"""

import asyncio
import os
from unittest.mock import MagicMock

import pytest
from nimble_web_search.register import NimbleWebSearchToolConfig
from nimble_web_search.register import nimble_web_search

# A concrete, time-invariant entity query that stays on-topic across time and
# regions — certified for repeated CI use. Keep in sync with
# fixtures/README.md (the recorded fixtures were captured with this query).
CANNED_QUERY = "NVIDIA CUDA Toolkit documentation"


def _env_bool(name: str, default: bool = False) -> bool:
    """env bool."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _env_bool("DEEP_RESEARCHER_NIMBLE_LIVE_TESTS"),
        reason="Set DEEP_RESEARCHER_NIMBLE_LIVE_TESTS=1 to run live Nimble integration tests.",
    ),
    pytest.mark.skipif(
        not os.environ.get("NIMBLE_API_KEY"),
        reason="NIMBLE_API_KEY not set; live Nimble tests require a real key.",
    ),
]


class TestLiveNimbleAPI:
    async def test_live_lite_search_returns_parseable_documents(self, assert_search_contract):
        """One live call with shipped defaults must satisfy the output contract.

        Success criteria (shared with the replay tests via
        ``assert_search_contract``): a non-error, non-empty response containing
        1..max_results ``<Document>`` blocks, every block with an http(s) href
        and a title, every block XML-parseable, and at least one non-empty body.
        Assertions are structural, never content-exact, so ordinary result
        variation cannot flake the test.
        """
        config = NimbleWebSearchToolConfig()  # shipped defaults: lite / 5 / general / US / en

        async with nimble_web_search(config, MagicMock()) as info:
            result = await asyncio.wait_for(info.single_fn(CANNED_QUERY), timeout=120)

        assert_search_contract(result, config.max_results)
