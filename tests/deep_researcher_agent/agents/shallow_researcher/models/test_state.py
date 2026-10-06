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

"""Tests for ShallowResearchAgentState model."""

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState


class TestShallowResearchAgentState:
    """Tests for the ShallowResearchAgentState model."""

    def test_create_state_with_messages(self):
        """Test creating state with messages."""
        messages = [HumanMessage(content="Test query")]
        state = ShallowResearchAgentState(messages=messages)

        assert len(state.messages) == 1
        assert state.messages[0].content == "Test query"

    def test_create_state_empty_messages(self):
        """Test creating state with empty messages list."""
        state = ShallowResearchAgentState(messages=[])

        assert state.messages == []

    def test_state_with_user_info(self):
        """Test state with user info."""
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test")],
            user_info={"name": "John", "role": "developer"},
        )

        assert state.user_info == {"name": "John", "role": "developer"}

    def test_state_with_tools_info(self):
        """Test state with tools info."""
        tools_info = [
            {"name": "web_search", "description": "Search the web"},
            {"name": "doc_search", "description": "Search documents"},
        ]
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test")],
            tools_info=tools_info,
        )

        assert state.tools_info == tools_info
        assert len(state.tools_info) == 2

    def test_state_defaults(self):
        """Test state with default values."""
        state = ShallowResearchAgentState(messages=[])

        assert state.user_info is None
        assert state.tools_info is None

    def test_state_message_accumulation(self):
        """Test that messages properly accumulate."""
        state = ShallowResearchAgentState(
            messages=[
                HumanMessage(content="First"),
                AIMessage(content="Response"),
                HumanMessage(content="Second"),
            ]
        )

        assert len(state.messages) == 3

    def test_state_full_workflow(self):
        """Test state with all fields populated."""
        state = ShallowResearchAgentState(
            messages=[
                HumanMessage(content="What is CUDA?"),
                AIMessage(content="CUDA is a parallel computing platform."),
            ],
            user_info={"role": "developer"},
            tools_info=[{"name": "web_search", "description": "Search"}],
        )

        assert state.user_info["role"] == "developer"
        assert len(state.tools_info) == 1
