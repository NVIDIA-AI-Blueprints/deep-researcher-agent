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

"""Tests for clarifier agent utility functions."""

from deep_researcher_agent.agents.clarifier.utils import extract_user_response


class TestExtractUserResponse:
    """Tests for the extract_user_response function."""

    def test_string_response(self):
        """Test extracting from string response."""
        result = extract_user_response("User's answer")
        assert result == "User's answer"

    def test_empty_string(self):
        """Test with empty string."""
        result = extract_user_response("")
        assert result == ""

    def test_object_with_content_text(self):
        """Test extracting from object with content.text attribute."""

        class MockContent:
            text = "The user's text"

        class MockResponse:
            content = MockContent()

        result = extract_user_response(MockResponse())
        assert result == "The user's text"

    def test_object_with_text_attribute(self):
        """Test extracting from object with direct text attribute."""

        class MockResponse:
            text = "Direct text"

        result = extract_user_response(MockResponse())
        assert result == "Direct text"

    def test_fallback_to_str(self):
        """Test fallback to str() conversion."""

        class MockResponse:
            def __str__(self):
                return "String representation"

        result = extract_user_response(MockResponse())
        assert result == "String representation"

    def test_nested_content_priority(self):
        """Test that content.text takes priority over text."""

        class MockContent:
            text = "From content"

        class MockResponse:
            content = MockContent()
            text = "Direct text"

        result = extract_user_response(MockResponse())
        assert result == "From content"

    def test_integer_input(self):
        """Test with integer input (should convert to string)."""
        result = extract_user_response(42)
        assert result == "42"

    def test_none_content_text(self):
        """Test when content exists but no text attribute."""

        class MockContent:
            pass

        class MockResponse:
            content = MockContent()
            text = "Fallback"

        result = extract_user_response(MockResponse())
        assert result == "Fallback"
