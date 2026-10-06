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

"""Tests for JSON utility functions."""

from deep_researcher_agent.common.json_utils import extract_json


class TestExtractJson:
    """Tests for the extract_json function."""

    def test_extract_json_direct_parse(self):
        """Test parsing a raw JSON string directly."""
        json_text = '{"key": "value", "number": 42}'
        result = extract_json(json_text)
        assert result == {"key": "value", "number": 42}

    def test_extract_json_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        json_text = '   {"key": "value"}   '
        result = extract_json(json_text)
        assert result == {"key": "value"}

    def test_extract_json_from_markdown_code_block(self):
        """Test extracting JSON from a markdown code block."""
        text = """Here is some JSON:
```json
{"intent": "research", "confidence": 0.95}
```
That was the JSON."""
        result = extract_json(text)
        assert result == {"intent": "research", "confidence": 0.95}

    def test_extract_json_from_markdown_with_newlines(self):
        """Test extracting JSON from markdown with extra newlines."""
        text = """```json

{
    "steps": [
        {"tool": "search", "query": "test"}
    ]
}

```"""
        result = extract_json(text)
        assert result == {"steps": [{"tool": "search", "query": "test"}]}

    def test_extract_json_embedded_in_text(self):
        """Test extracting JSON embedded in surrounding text."""
        text = 'The result is {"status": "success", "data": [1, 2, 3]} as expected.'
        result = extract_json(text)
        assert result == {"status": "success", "data": [1, 2, 3]}

    def test_extract_json_nested_objects(self):
        """Test extracting nested JSON objects."""
        text = """Output: {"outer": {"inner": {"deep": "value"}}}"""
        result = extract_json(text)
        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_extract_json_empty_string(self):
        """Test that empty string returns None."""
        result = extract_json("")
        assert result is None

    def test_extract_json_none_like_empty(self):
        """Test that whitespace-only string returns None."""
        result = extract_json("   ")
        assert result is None

    def test_extract_json_no_json_present(self):
        """Test that text without JSON returns None."""
        result = extract_json("This is just plain text without any JSON.")
        assert result is None

    def test_extract_json_invalid_json(self):
        """Test that invalid JSON returns None."""
        result = extract_json('{"key": "missing closing brace"')
        assert result is None

    def test_extract_json_array_not_object(self):
        """Test parsing a JSON array (top-level)."""
        json_text = '[1, 2, 3, "four"]'
        result = extract_json(json_text)
        # Direct parse should work for arrays
        assert result == [1, 2, 3, "four"]

    def test_extract_json_complex_nested_structure(self):
        """Test extracting complex nested JSON with various types."""
        text = """Result:
{"decision": "deep", "reasons": ["complexity", "multiple_sources"], "metadata": \
{"score": 0.87, "tags": ["ai", "research"]}}
End."""
        result = extract_json(text)
        assert result == {
            "decision": "deep",
            "reasons": ["complexity", "multiple_sources"],
            "metadata": {"score": 0.87, "tags": ["ai", "research"]},
        }

    def test_extract_json_with_escaped_characters(self):
        """Test JSON with escaped characters."""
        json_text = '{"message": "Hello\\nWorld", "path": "C:\\\\Users"}'
        result = extract_json(json_text)
        assert result == {"message": "Hello\nWorld", "path": "C:\\Users"}

    def test_extract_json_boolean_and_null_values(self):
        """Test JSON with boolean and null values."""
        json_text = '{"active": true, "disabled": false, "optional": null}'
        result = extract_json(json_text)
        assert result == {"active": True, "disabled": False, "optional": None}

    def test_extract_json_from_malformed_markdown(self):
        """Test extracting from malformed markdown (missing closing backticks)."""
        text = """```json
{"key": "value"}"""
        # Should fall back to finding the JSON object in text
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_multiple_objects_returns_first(self):
        """Test that when multiple JSON objects exist, the first valid one is returned."""
        text = 'First: {"a": 1} Second: {"b": 2}'
        result = extract_json(text)
        assert result == {"a": 1}
