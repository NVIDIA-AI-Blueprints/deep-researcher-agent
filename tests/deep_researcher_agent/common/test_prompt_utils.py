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

"""Tests for prompt utility functions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deep_researcher_agent.common.prompt_utils import PromptError
from deep_researcher_agent.common.prompt_utils import load_prompt
from deep_researcher_agent.common.prompt_utils import render_prompt_template


class TestLoadPrompt:
    """Tests for the load_prompt function."""

    def test_load_prompt_exact_name(self):
        """Test loading a prompt file with exact filename."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)
            (prompt_path / "system.j2").write_text("You are a helpful assistant.")

            result = load_prompt(prompt_path, "system.j2")
            assert result == "You are a helpful assistant."

    def test_load_prompt_without_extension(self):
        """Test loading a prompt file without specifying .j2 extension."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)
            (prompt_path / "researcher.j2").write_text("Research the topic: {{ topic }}")

            result = load_prompt(prompt_path, "researcher")
            assert result == "Research the topic: {{ topic }}"

    def test_load_prompt_exact_name_priority(self):
        """Test that exact name takes priority over adding .j2."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)
            (prompt_path / "custom").write_text("Exact match content")
            (prompt_path / "custom.j2").write_text("With extension content")

            result = load_prompt(prompt_path, "custom")
            assert result == "Exact match content"

    def test_load_prompt_file_not_found(self):
        """Test that PromptError is raised for missing prompt files."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)

            with pytest.raises(PromptError) as exc_info:
                load_prompt(prompt_path, "nonexistent")

            assert "Prompt file not found" in str(exc_info.value)
            assert "nonexistent" in str(exc_info.value)

    def test_load_prompt_multiline_content(self):
        """Test loading a multiline prompt template."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)
            multiline_prompt = """System: You are a research assistant.
Context: {{ context }}
Question: {{ question }}
Please provide a detailed answer."""
            (prompt_path / "qa.j2").write_text(multiline_prompt)

            result = load_prompt(prompt_path, "qa")
            assert result == multiline_prompt

    def test_load_prompt_empty_file(self):
        """Test loading an empty prompt file."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)
            (prompt_path / "empty.j2").write_text("")

            result = load_prompt(prompt_path, "empty")
            assert result == ""

    def test_load_prompt_with_unicode(self):
        """Test loading a prompt with unicode characters."""
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir)
            unicode_content = "Analyze: 日本語 テスト 🔍 émojis"
            (prompt_path / "unicode.j2").write_text(unicode_content)

            result = load_prompt(prompt_path, "unicode")
            assert result == unicode_content


class TestRenderPromptTemplate:
    """Tests for the render_prompt_template function."""

    def test_render_simple_variable(self):
        """Test rendering a template with a simple variable."""
        template = "Hello, {{ name }}!"
        result = render_prompt_template(template, name="World")
        assert result == "Hello, World!"

    def test_render_multiple_variables(self):
        """Test rendering a template with multiple variables."""
        template = "{{ greeting }}, {{ name }}! You have {{ count }} messages."
        result = render_prompt_template(template, greeting="Hi", name="Alice", count=5)
        assert result == "Hi, Alice! You have 5 messages."

    def test_render_no_variables(self):
        """Test rendering a template without variables."""
        template = "This is static content."
        result = render_prompt_template(template)
        assert result == "This is static content."

    def test_render_with_conditionals(self):
        """Test rendering a template with Jinja2 conditionals."""
        template = """{% if include_context %}Context: {{ context }}
{% endif %}Question: {{ question }}"""

        result_with = render_prompt_template(template, include_context=True, context="Some info", question="What?")
        assert "Context: Some info" in result_with
        assert "Question: What?" in result_with

        result_without = render_prompt_template(template, include_context=False, context="Some info", question="What?")
        assert "Context:" not in result_without
        assert "Question: What?" in result_without

    def test_render_with_loops(self):
        """Test rendering a template with Jinja2 loops."""
        template = """Sources:
{% for source in sources %}- {{ source }}
{% endfor %}"""
        result = render_prompt_template(template, sources=["Source A", "Source B", "Source C"])
        assert "- Source A" in result
        assert "- Source B" in result
        assert "- Source C" in result

    def test_render_with_filters(self):
        """Test rendering a template with Jinja2 filters."""
        template = "Name: {{ name | upper }}, Length: {{ items | length }}"
        result = render_prompt_template(template, name="test", items=[1, 2, 3])
        assert result == "Name: TEST, Length: 3"

    def test_render_undefined_variable_error(self):
        """Test that undefined variables raise PromptError."""
        template = "Hello, {{ undefined_var }}!"

        with pytest.raises(PromptError) as exc_info:
            render_prompt_template(template)

        assert "Failed to render template" in str(exc_info.value)

    def test_render_syntax_error(self):
        """Test that template syntax errors raise PromptError."""
        template = "{% if unclosed"

        with pytest.raises(PromptError) as exc_info:
            render_prompt_template(template)

        assert "Failed to render template" in str(exc_info.value)

    def test_render_nested_objects(self):
        """Test rendering a template with nested object access."""
        template = "User: {{ user.name }}, Role: {{ user.role }}"
        result = render_prompt_template(template, user={"name": "John", "role": "admin"})
        assert result == "User: John, Role: admin"

    def test_render_multiline_output(self):
        """Test rendering a complex multiline template."""
        template = """# Research Plan
Topic: {{ topic }}

## Steps
{% for step in steps %}
### Step {{ loop.index }}: {{ step.name }}
Description: {{ step.description }}
{% endfor %}

## Notes
{{ notes }}"""

        result = render_prompt_template(
            template,
            topic="AI Research",
            steps=[
                {"name": "Literature Review", "description": "Review existing papers"},
                {"name": "Data Collection", "description": "Gather relevant data"},
            ],
            notes="Handle with care",
        )

        assert "Topic: AI Research" in result
        assert "Step 1: Literature Review" in result
        assert "Step 2: Data Collection" in result
        assert "Handle with care" in result
