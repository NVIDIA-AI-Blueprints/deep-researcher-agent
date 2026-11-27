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

"""Recorded-response replay tests: the credential-free test-mode path.

Each file in ``fixtures/`` is a real Context.dev Search API response, redacted by
construction (see ``fixtures/README.md``). Replaying it through the HTTP transport
exercises the full tool pipeline (config, request building, retry wrapper,
parsing, rendering, escaping) deterministically, with no network and no key.

These tests certify the same structural output contracts as the key-gated live
integration test (``test_context_dev_live_integration.py``).
"""

from unittest.mock import MagicMock

import httpx
from context_dev_search.register import SCHOLARLY_OVERFETCH_FACTOR
from context_dev_search.register import ContextDevPaperSearchToolConfig
from context_dev_search.register import ContextDevWebSearchToolConfig
from context_dev_search.register import _is_scholarly
from context_dev_search.register import context_dev_paper_search
from context_dev_search.register import context_dev_web_search

WEB_FIXTURE = "recorded_web_search_response.json"
PAPER_FIXTURE = "recorded_paper_search_response.json"


def _replay(mock_api, payload: dict) -> None:
    mock_api.add(httpx.Response(200, json={"results": payload["results"], "request_id": payload["_request_id"]}))


class TestRecordedWebSearch:
    async def test_meets_output_contract(self, api_key_env, mock_api, load_recorded_fixture, assert_web_contract):
        payload = load_recorded_fixture(WEB_FIXTURE)
        _replay(mock_api, payload)
        config = ContextDevWebSearchToolConfig(**payload["_config"])

        async with context_dev_web_search(config, MagicMock()) as info:
            result = await info.single_fn(payload["_query"])

        blocks = assert_web_contract(result, config.max_results)
        assert len(blocks) == len(payload["results"])
        assert mock_api.bodies[0]["query"] == payload["_query"]

    async def test_entity_artifacts_are_decoded(self, api_key_env, mock_api, load_recorded_fixture):
        payload = load_recorded_fixture(WEB_FIXTURE)
        assert any("&#x20;" in p for r in payload["results"] for p in r["highlights"]["highlights"])
        _replay(mock_api, payload)

        async with context_dev_web_search(ContextDevWebSearchToolConfig(), MagicMock()) as info:
            result = await info.single_fn(payload["_query"])

        assert "#x20;" not in result


class TestRecordedPaperSearch:
    async def test_meets_output_contract(self, api_key_env, mock_api, load_recorded_fixture, assert_paper_contract):
        payload = load_recorded_fixture(PAPER_FIXTURE)
        _replay(mock_api, payload)
        config = ContextDevPaperSearchToolConfig(**payload["_config"])

        async with context_dev_paper_search(config, MagicMock()) as info:
            result = await info.single_fn(payload["_query"])

        entries = assert_paper_contract(result, config.max_results)
        assert len(entries) == config.max_results
        # The indexed paper itself ranks first once news and blog coverage is dropped.
        assert entries[0].group("link") == "https://arxiv.org/abs/1706.03762v7"
        assert all(_is_scholarly(entry.group("link")) for entry in entries)
        assert "includeDomains" not in mock_api.bodies[0]
        assert mock_api.bodies[0]["numResults"] == SCHOLARLY_OVERFETCH_FACTOR * config.max_results

    async def test_recording_contains_non_scholarly_results_to_filter(self, load_recorded_fixture):
        payload = load_recorded_fixture(PAPER_FIXTURE)
        hosts = [result["url"] for result in payload["results"]]
        assert any(not _is_scholarly(url) for url in hosts), "fixture no longer exercises the scholarly filter"
