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

"""Agent-local prompt loading utilities.

This module provides utilities for loading prompts co-located with agents.
Each agent package has a prompts/ directory containing its Jinja2 templates.
"""

import logging
from pathlib import Path
from typing import Any

import jinja2

logger = logging.getLogger(__name__)


class PromptError(Exception):
    """Error loading or rendering prompts."""

    pass


def load_prompt(path: Path, name: str) -> str:
    """
    Load a prompt template from an agent's prompts/ directory.

    Args:
        path: Path to the prompts directory.
        name: Name of the prompt file (e.g., 'system' or 'system.j2').

    Returns:
        The prompt template as a string.

    Raises:
        PromptError: If the prompt file cannot be found.
    """
    # Try exact name first, then with .j2 extension
    prompt_path = path / name
    if not prompt_path.exists() and not name.endswith(".j2"):
        prompt_path = path / f"{name}.j2"

    if not prompt_path.exists():
        raise PromptError(f"Prompt file not found: {name} in {path}")

    try:
        return prompt_path.read_text()
    except Exception as e:
        raise PromptError(f"Failed to load prompt {name}: {e}") from e


def render_prompt_template(template: str, **kwargs: Any) -> str:
    """
    Render a Jinja2 template with the given variables.

    Args:
        template: The template string.
        **kwargs: Variables to substitute in the template.

    Returns:
        The rendered template.

    Raises:
        PromptError: If template rendering fails.
    """
    try:
        jinja_template = jinja2.Template(template, undefined=jinja2.StrictUndefined)
        return jinja_template.render(**kwargs)
    except jinja2.TemplateError as e:
        raise PromptError(f"Failed to render template: {e}") from e
