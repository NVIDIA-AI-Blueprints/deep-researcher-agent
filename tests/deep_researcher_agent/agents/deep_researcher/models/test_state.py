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

"""Tests for DeepResearchAgentState model."""

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from deep_researcher_agent.agents.deep_researcher.models import DeepResearchAgentState


class TestDeepResearchAgentState:
    """Tests for the DeepResearchAgentState model."""

    def test_create_state_with_messages(self):
        """Test creating state with messages."""
        messages = [HumanMessage(content="Test query")]
        state = DeepResearchAgentState(messages=messages)

        assert len(state.messages) == 1
        assert state.messages[0].content == "Test query"

    def test_create_state_empty_messages(self):
        """Test creating state with empty messages list."""
        state = DeepResearchAgentState(messages=[])

        assert state.messages == []

    def test_state_with_user_info(self):
        """Test state with user info."""
        state = DeepResearchAgentState(
            messages=[HumanMessage(content="Test")],
            user_info={"name": "John", "role": "developer"},
        )

        assert state.user_info == {"name": "John", "role": "developer"}

    def test_state_with_tools_info(self):
        """Test state with tools info."""
        tools_info = [
            {"name": "web_search", "description": "Search the web"},
        ]
        state = DeepResearchAgentState(
            messages=[HumanMessage(content="Test")],
            tools_info=tools_info,
        )

        assert state.tools_info == tools_info

    def test_state_with_todos(self):
        """Test state with todos."""
        todos = [
            {"id": "1", "task": "Research topic A", "status": "pending"},
            {"id": "2", "task": "Research topic B", "status": "done"},
        ]
        state = DeepResearchAgentState(
            messages=[HumanMessage(content="Test")],
            todos=todos,
        )

        assert state.todos == todos

    def test_state_with_files(self):
        """Test state with files."""
        files = {
            "report.md": "# Research Report\n\nContent here...",
            "notes.txt": "Some notes",
        }
        state = DeepResearchAgentState(
            messages=[HumanMessage(content="Test")],
            files=files,
        )

        assert state.files == files

    def test_state_defaults(self):
        """Test state with default values."""
        state = DeepResearchAgentState(messages=[])

        assert state.user_info is None
        assert state.tools_info is None
        assert state.todos == []
        assert state.files == {}
        assert state.subagents == []

    def test_state_message_accumulation(self):
        """Test that messages use add_messages reducer behavior."""
        state = DeepResearchAgentState(
            messages=[
                HumanMessage(content="First message"),
                AIMessage(content="Response"),
                HumanMessage(content="Second message"),
            ]
        )

        assert len(state.messages) == 3

    def test_state_model_validate(self):
        """Test state creation via model_validate."""
        data = {
            "messages": [{"type": "human", "content": "Test"}],
            "user_info": {"name": "Test User"},
        }

        state = DeepResearchAgentState.model_validate(data)

        assert state.user_info == {"name": "Test User"}
