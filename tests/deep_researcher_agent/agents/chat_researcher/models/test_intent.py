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

"""Tests for IntentResult model."""

import pytest
from pydantic import ValidationError

from deep_researcher_agent.agents.chat_researcher.models.intent import IntentResult


class TestIntentResult:
    """Tests for the IntentResult Pydantic model."""

    def test_intent_result_meta(self):
        """Test creating IntentResult with meta intent."""
        result = IntentResult(intent="meta")
        assert result.intent == "meta"
        assert result.raw is None

    def test_intent_result_research(self):
        """Test creating IntentResult with research intent."""
        result = IntentResult(intent="research")
        assert result.intent == "research"
        assert result.target == "new_research"
        assert result.report_action is None
        assert result.use_parent_report_context is False
        assert result.raw is None

    def test_intent_result_report_route(self):
        """Test creating IntentResult with report-aware route fields."""
        result = IntentResult(
            intent="research",
            target="report",
            report_action="edit",
            use_parent_report_context=False,
        )

        assert result.target == "report"
        assert result.report_action == "edit"
        assert result.use_parent_report_context is False

    def test_intent_result_hybrid_route(self):
        result = IntentResult(intent="research", target="hybrid_research")

        assert result.target == "hybrid_research"

    def test_intent_result_invalid_report_action(self):
        """Test invalid report actions are rejected."""
        with pytest.raises(ValidationError):
            IntentResult(intent="research", report_action="delete")

    def test_intent_result_with_raw_data(self):
        """Test creating IntentResult with raw classification data."""
        raw_data = {"confidence": 0.95, "reasoning": "Query asks for information"}
        result = IntentResult(intent="research", raw=raw_data)
        assert result.intent == "research"
        assert result.raw == raw_data

    def test_intent_result_invalid_intent(self):
        """Test that invalid intent values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntentResult(intent="invalid")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "intent" in str(errors[0]["loc"])

    def test_intent_result_missing_intent(self):
        """Test that missing intent raises ValidationError."""
        with pytest.raises(ValidationError):
            IntentResult()

    def test_intent_result_dict_conversion(self):
        """Test IntentResult can be converted to dict."""
        result = IntentResult(intent="meta", raw={"key": "value"})
        result_dict = result.model_dump()

        assert result_dict["intent"] == "meta"
        assert result_dict["raw"] == {"key": "value"}

    def test_intent_result_json_serialization(self):
        """Test IntentResult can be serialized to JSON."""
        result = IntentResult(intent="research", raw={"score": 0.9})
        json_str = result.model_dump_json()

        assert '"intent":"research"' in json_str
        assert '"raw":' in json_str

    def test_intent_result_from_dict(self):
        """Test creating IntentResult from a dictionary."""
        data = {"intent": "meta", "raw": None}
        result = IntentResult.model_validate(data)

        assert result.intent == "meta"
        assert result.raw is None
