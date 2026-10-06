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

"""Tests for DepthDecision model."""

import pytest
from pydantic import ValidationError

from deep_researcher_agent.agents.chat_researcher.models.depth import DepthDecision


class TestDepthDecision:
    """Tests for the DepthDecision Pydantic model."""

    def test_depth_decision_shallow(self):
        """Test creating DepthDecision with shallow decision."""
        decision = DepthDecision(decision="shallow")
        assert decision.decision == "shallow"
        assert decision.raw_reasoning is None

    def test_depth_decision_deep(self):
        """Test creating DepthDecision with deep decision."""
        decision = DepthDecision(decision="deep")
        assert decision.decision == "deep"
        assert decision.raw_reasoning is None

    def test_depth_decision_with_reasoning(self):
        """Test creating DepthDecision with reasoning."""
        reasoning = "Query requires comprehensive analysis of multiple sources"
        decision = DepthDecision(decision="deep", raw_reasoning=reasoning)
        assert decision.decision == "deep"
        assert decision.raw_reasoning == reasoning

    def test_depth_decision_invalid_decision(self):
        """Test that invalid decision values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DepthDecision(decision="medium")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "decision" in str(errors[0]["loc"])

    def test_depth_decision_missing_decision(self):
        """Test that missing decision raises ValidationError."""
        with pytest.raises(ValidationError):
            DepthDecision()

    def test_depth_decision_dict_conversion(self):
        """Test DepthDecision can be converted to dict."""
        decision = DepthDecision(decision="shallow", raw_reasoning="Simple lookup")
        decision_dict = decision.model_dump()

        assert decision_dict["decision"] == "shallow"
        assert decision_dict["raw_reasoning"] == "Simple lookup"

    def test_depth_decision_json_serialization(self):
        """Test DepthDecision can be serialized to JSON."""
        decision = DepthDecision(decision="deep", raw_reasoning="Complex query")
        json_str = decision.model_dump_json()

        assert '"decision":"deep"' in json_str
        assert '"raw_reasoning":"Complex query"' in json_str

    def test_depth_decision_from_dict(self):
        """Test creating DepthDecision from a dictionary."""
        data = {"decision": "shallow", "raw_reasoning": None}
        decision = DepthDecision.model_validate(data)

        assert decision.decision == "shallow"
        assert decision.raw_reasoning is None

    def test_depth_decision_empty_reasoning(self):
        """Test DepthDecision with empty string reasoning."""
        decision = DepthDecision(decision="deep", raw_reasoning="")
        assert decision.decision == "deep"
        assert decision.raw_reasoning == ""
