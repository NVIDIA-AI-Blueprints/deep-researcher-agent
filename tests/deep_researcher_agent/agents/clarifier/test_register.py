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

"""Tests for the clarifier agent NAT registration."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from deep_researcher_agent.agents.clarifier.register import ClarifierConfig


class TestClarifierConfig:
    """Tests for the ClarifierConfig model."""

    def test_config_with_required_fields(self):
        """Test config with required fields only."""
        config = ClarifierConfig(llm="test_llm")

        assert config.llm == "test_llm"
        assert config.tools == []
        assert config.max_turns == 3
        assert config.log_response_max_chars == 2000

    def test_config_with_all_fields(self):
        """Test config with all fields specified."""
        config = ClarifierConfig(
            llm="test_llm",
            tools=["tool1", "tool2"],
            max_turns=5,
            log_response_max_chars=1000,
        )

        assert config.llm == "test_llm"
        assert config.tools == ["tool1", "tool2"]
        assert config.max_turns == 5
        assert config.log_response_max_chars == 1000

    def test_config_tools_default_factory(self):
        """Test tools default to empty list."""
        config = ClarifierConfig(llm="llm")
        assert config.tools == []
        # Verify it's a new list each time
        config2 = ClarifierConfig(llm="llm")
        assert config.tools is not config2.tools

    def test_config_inherits_from_function_base_config(self):
        """Test config inherits from FunctionBaseConfig."""
        from nat.data_models.function import FunctionBaseConfig

        assert issubclass(ClarifierConfig, FunctionBaseConfig)

    def test_config_field_descriptions(self):
        """Test config fields have descriptions."""
        fields = ClarifierConfig.model_fields
        assert fields["llm"].description is not None
        assert fields["tools"].description is not None
        assert fields["max_turns"].description is not None


class TestUserPromptCallback:
    """Tests for the user prompt callback creation."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock NAT context."""
        mock_user_manager = MagicMock()
        mock_user_manager.prompt_user_input = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.user_interaction_manager = mock_user_manager

        return mock_ctx

    @pytest.mark.asyncio
    async def test_user_prompt_callback_extracts_response(self, mock_context):
        """Test user prompt callback extracts text from response."""
        from deep_researcher_agent.agents.clarifier.utils import extract_user_response

        # Test the extract function directly
        class MockContent:
            text = "User's choice"

        class MockResponse:
            content = MockContent()

        result = extract_user_response(MockResponse())
        assert result == "User's choice"

    @pytest.mark.asyncio
    async def test_user_prompt_callback_with_string_response(self):
        """Test user prompt callback with string response."""
        from deep_researcher_agent.agents.clarifier.utils import extract_user_response

        result = extract_user_response("Direct string response")
        assert result == "Direct string response"
