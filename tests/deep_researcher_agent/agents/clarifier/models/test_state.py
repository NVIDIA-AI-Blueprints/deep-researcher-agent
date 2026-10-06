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

"""Tests for ClarifierAgentState and ClarifierResult models."""

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from deep_researcher_agent.agents.clarifier.models import ClarifierAgentState
from deep_researcher_agent.agents.clarifier.models import ClarifierResult


class TestClarifierAgentState:
    """Tests for the ClarifierAgentState model."""

    def test_create_with_messages(self):
        """Test creating state with messages."""
        messages = [HumanMessage(content="Research AI agents")]
        state = ClarifierAgentState(messages=messages)

        assert len(state.messages) == 1
        assert state.messages[0].content == "Research AI agents"

    def test_default_max_turns(self):
        """Test default max_turns is 3."""
        state = ClarifierAgentState(messages=[])
        assert state.max_turns == 3

    def test_default_data_sources(self):
        """Test default data_sources is None."""
        state = ClarifierAgentState(messages=[])
        assert state.data_sources is None

    def test_custom_data_sources(self):
        """Test custom data_sources."""
        state = ClarifierAgentState(messages=[], data_sources=["web_search"])
        assert state.data_sources == ["web_search"]

    def test_custom_max_turns(self):
        """Test custom max_turns."""
        state = ClarifierAgentState(messages=[], max_turns=5)
        assert state.max_turns == 5

    def test_default_clarifier_log(self):
        """Test default clarifier_log is empty string."""
        state = ClarifierAgentState(messages=[])
        assert state.clarifier_log == ""

    def test_custom_clarifier_log(self):
        """Test custom clarifier_log."""
        state = ClarifierAgentState(messages=[], clarifier_log="Turn 1: Hello")
        assert state.clarifier_log == "Turn 1: Hello"

    def test_default_iteration(self):
        """Test default iteration is 0."""
        state = ClarifierAgentState(messages=[])
        assert state.iteration == 0

    def test_custom_iteration(self):
        """Test custom iteration."""
        state = ClarifierAgentState(messages=[], iteration=2)
        assert state.iteration == 2

    def test_remaining_questions_full(self):
        """Test remaining_questions when iteration is 0."""
        state = ClarifierAgentState(messages=[], max_turns=3, iteration=0)
        assert state.remaining_questions == 3

    def test_remaining_questions_partial(self):
        """Test remaining_questions after some iterations."""
        state = ClarifierAgentState(messages=[], max_turns=3, iteration=1)
        assert state.remaining_questions == 2

    def test_remaining_questions_zero(self):
        """Test remaining_questions when max turns reached."""
        state = ClarifierAgentState(messages=[], max_turns=3, iteration=3)
        assert state.remaining_questions == 0

    def test_remaining_questions_negative(self):
        """Test remaining_questions when iteration exceeds max_turns."""
        state = ClarifierAgentState(messages=[], max_turns=3, iteration=5)
        assert state.remaining_questions == -2

    def test_multiple_messages(self):
        """Test state with multiple messages."""
        messages = [
            HumanMessage(content="Research AI"),
            AIMessage(content="What aspect?"),
            HumanMessage(content="Technical"),
        ]
        state = ClarifierAgentState(messages=messages)
        assert len(state.messages) == 3

    def test_model_validate(self):
        """Test validation from dict."""
        data = {
            "messages": [{"type": "human", "content": "Hello"}],
            "max_turns": 5,
            "iteration": 1,
            "clarifier_log": "Log",
            "data_sources": ["confluence", "sharepoint"],
        }
        state = ClarifierAgentState.model_validate(data)
        assert state.max_turns == 5
        assert state.iteration == 1
        assert state.clarifier_log == "Log"
        assert state.data_sources == ["confluence", "sharepoint"]


class TestClarifierResult:
    """Tests for the ClarifierResult model."""

    def test_default_values(self):
        """Test default values for ClarifierResult."""
        result = ClarifierResult()
        assert result.clarifier_log == ""

    def test_custom_values(self):
        """Test custom values for ClarifierResult."""
        result = ClarifierResult(clarifier_log="Test log")
        assert result.clarifier_log == "Test log"
