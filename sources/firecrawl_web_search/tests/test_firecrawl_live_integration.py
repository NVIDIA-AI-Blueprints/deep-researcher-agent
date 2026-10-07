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

"""Live smoke test for firecrawl_web_search.

Skipped unless AIQ_FIRECRAWL_LIVE_TESTS=1. Uses FIRECRAWL_API_KEY when set, otherwise runs keyless.
"""

import os
from unittest.mock import MagicMock

import pytest
from firecrawl_web_search.register import FirecrawlWebSearchToolConfig
from firecrawl_web_search.register import firecrawl_web_search

pytestmark = pytest.mark.skipif(
    os.environ.get("AIQ_FIRECRAWL_LIVE_TESTS") != "1", reason="set AIQ_FIRECRAWL_LIVE_TESTS=1 to run live tests"
)


async def test_live_search_returns_documents():
    config = FirecrawlWebSearchToolConfig(max_results=2)

    async with firecrawl_web_search(config, MagicMock()) as info:
        out = await info.single_fn("What is the NVIDIA AI-Q research blueprint?")

    assert "<Document href=" in out
    body = out.split("</title>", 1)[1].split("</Document>", 1)[0].strip()
    assert body
