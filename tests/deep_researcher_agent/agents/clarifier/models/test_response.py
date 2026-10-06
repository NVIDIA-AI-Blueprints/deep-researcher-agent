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

"""Tests for ClarificationResponse model."""

from deep_researcher_agent.agents.clarifier.models import ClarificationResponse


class TestClarificationResponse:
    """Tests for the ClarificationResponse model."""

    def test_create_needs_clarification_true(self):
        """Test creating response when clarification is needed."""
        response = ClarificationResponse(needs_clarification=True, clarification_question="What aspect interests you?")
        assert response.needs_clarification is True
        assert response.clarification_question == "What aspect interests you?"

    def test_create_needs_clarification_false(self):
        """Test creating response when clarification is complete."""
        response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        assert response.needs_clarification is False
        assert response.clarification_question is None

    def test_is_complete_when_not_needed(self):
        """Test is_complete returns True when clarification not needed."""
        response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        assert response.is_complete() is True

    def test_is_complete_when_needed(self):
        """Test is_complete returns False when clarification is needed."""
        response = ClarificationResponse(needs_clarification=True, clarification_question="What scope?")
        assert response.is_complete() is False

    def test_is_valid_with_question_mark(self):
        """Test is_valid returns True when question contains '?'."""
        response = ClarificationResponse(needs_clarification=True, clarification_question="What aspect interests you?")
        assert response.is_valid() is True

    def test_is_valid_without_question_mark(self):
        """Test is_valid returns True even without '?' - only checks question exists."""
        response = ClarificationResponse(needs_clarification=True, clarification_question="Please provide more details")
        assert response.is_valid() is True

    def test_is_valid_with_empty_question(self):
        """Test is_valid returns False when question is empty."""
        response = ClarificationResponse(needs_clarification=True, clarification_question="")
        assert response.is_valid() is False

    def test_is_valid_with_none_question_when_needed(self):
        """Test is_valid returns False when question is None but needed."""
        response = ClarificationResponse(needs_clarification=True, clarification_question=None)
        assert response.is_valid() is False

    def test_is_valid_when_not_needed(self):
        """Test is_valid returns True when clarification not needed."""
        response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        assert response.is_valid() is True

    def test_model_dump_json(self):
        """Test JSON serialization."""
        response = ClarificationResponse(needs_clarification=True, clarification_question="What scope?")
        json_str = response.model_dump_json()
        assert "needs_clarification" in json_str
        assert "clarification_question" in json_str

    def test_model_validate(self):
        """Test validation from dict."""
        data = {"needs_clarification": True, "clarification_question": "What focus area?"}
        response = ClarificationResponse.model_validate(data)
        assert response.needs_clarification is True
        assert response.clarification_question == "What focus area?"

    def test_default_clarification_question_is_none(self):
        """Test clarification_question defaults to None."""
        response = ClarificationResponse(needs_clarification=False)
        assert response.clarification_question is None
