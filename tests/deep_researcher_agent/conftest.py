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

"""Pytest fixtures for tests."""

from unittest.mock import MagicMock

import pytest

from deep_researcher_agent.common.llm_provider import LLMProvider


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = MagicMock()
    return llm


@pytest.fixture
def llm_provider(mock_llm):
    """Create an LLMProvider with a mock default LLM."""
    provider = LLMProvider()
    provider.set_default(mock_llm)
    return provider
