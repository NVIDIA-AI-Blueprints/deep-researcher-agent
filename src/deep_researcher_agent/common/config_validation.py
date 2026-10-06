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

"""Configuration validation utilities for checking required API keys."""

import os
import re
from typing import Any

# Mapping of LLM _type to required API key environment variable names
# This can be extended as new providers are added
LLM_API_KEY_MAP = {
    "nim": ["NVIDIA_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    "gemini": ["GOOGLE_API_KEY"],
    # Add more providers as needed
}


def _extract_env_var(value: str) -> str | None:
    """Extract environment variable name from ${VAR_NAME} syntax."""
    if isinstance(value, str):
        match = re.match(r"\$\{([^}]+)\}", value)
        if match:
            return match.group(1)
    return None


def _get_llm_api_key_requirements(llm_config: dict[str, Any]) -> list[str]:
    """
    Determine required API keys for an LLM configuration.

    Args:
        llm_config: LLM configuration dictionary with _type and optional api_key

    Returns:
        List of required API key environment variable names
    """
    llm_type = llm_config.get("_type", "").lower()
    required_keys = LLM_API_KEY_MAP.get(llm_type, [])

    # If api_key is explicitly set in config, check if it references an env var
    api_key_config = llm_config.get("api_key")
    if api_key_config:
        env_var = _extract_env_var(api_key_config)
        if env_var:
            # If config specifies an env var, that's the required key
            return [env_var]
        # If api_key is a literal value, no env var needed
        return []

    return required_keys


def validate_llm_configs(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that all required API keys are set for LLMs in the configuration.

    Args:
        config: Full workflow configuration dictionary

    Returns:
        Tuple of (is_valid, missing_keys) where:
        - is_valid: True if all required keys are present
        - missing_keys: List of missing API key names
    """
    llms_config = config.get("llms", {})
    if not llms_config:
        return True, []

    missing_keys = []
    checked_keys = set()  # Track which keys we've already checked

    for llm_name, llm_config in llms_config.items():
        if not isinstance(llm_config, dict):
            continue

        required_keys = _get_llm_api_key_requirements(llm_config)

        for key in required_keys:
            if key not in checked_keys:
                checked_keys.add(key)
                if not os.getenv(key):
                    missing_keys.append(key)

    return len(missing_keys) == 0, missing_keys


def get_llm_provider_info(llm_config: dict[str, Any]) -> str:
    """
    Get human-readable information about an LLM provider.

    Args:
        llm_config: LLM configuration dictionary

    Returns:
        Provider name and API key source information
    """
    llm_type = llm_config.get("_type", "unknown")
    api_key_config = llm_config.get("api_key")

    if api_key_config:
        env_var = _extract_env_var(api_key_config)
        if env_var:
            return f"{llm_type} (requires {env_var})"
        return f"{llm_type} (api_key set in config)"

    required_keys = LLM_API_KEY_MAP.get(llm_type.lower(), [])
    if required_keys:
        return f"{llm_type} (requires {', '.join(required_keys)})"

    return f"{llm_type} (no API key required)"
