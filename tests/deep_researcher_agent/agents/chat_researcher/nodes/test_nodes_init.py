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

"""Tests for the chat researcher nodes __init__ module."""


class TestNodesInit:
    """Tests for the nodes module initialization."""

    def test_import_intent_classifier(self):
        """Test that IntentClassifier can be imported from nodes."""
        from deep_researcher_agent.agents.chat_researcher.nodes import IntentClassifier

        assert IntentClassifier is not None

    def test_all_exports(self):
        """Test that __all__ contains the entry-routing nodes."""
        from deep_researcher_agent.agents.chat_researcher import nodes

        assert "IntentClassifier" in nodes.__all__
        assert nodes.__all__ == ["ContextAwareIntentRouter", "IntentClassifier"]
