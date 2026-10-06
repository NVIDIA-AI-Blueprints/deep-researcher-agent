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

"""Tests for the chat researcher models __init__ module."""


class TestModelsInit:
    """Tests for the models module initialization."""

    def test_import_chat_researcher_state(self):
        """Test that ChatResearcherState can be imported from models."""
        from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState

        assert ChatResearcherState is not None

    def test_import_depth_decision(self):
        """Test that DepthDecision can be imported from models."""
        from deep_researcher_agent.agents.chat_researcher.models import DepthDecision

        assert DepthDecision is not None

    def test_import_intent_result(self):
        """Test that IntentResult can be imported from models."""
        from deep_researcher_agent.agents.chat_researcher.models import IntentResult

        assert IntentResult is not None

    def test_import_shallow_result(self):
        """Test that ShallowResult can be imported from models."""
        from deep_researcher_agent.agents.chat_researcher.models import ShallowResult

        assert ShallowResult is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from deep_researcher_agent.agents.chat_researcher import models

        assert "ChatResearcherState" in models.__all__
        assert "DepthDecision" in models.__all__
        assert "IntentResult" in models.__all__
        assert "ShallowResult" in models.__all__
