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

"""JSON utility functions."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict[str, Any] | None:
    """
    Extract JSON from text, handling markdown code blocks and other formats.

    Tries multiple strategies to extract JSON:
    1. Parse as raw JSON
    2. Extract from ```json code blocks
    3. Find first { to last } substring

    Args:
        text: Text that may contain JSON.

    Returns:
        Parsed JSON dict, or None if extraction fails.
    """
    if not text:
        return None

    text = text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    start = text.find("{")
    if start != -1:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : start + i + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning("Failed to extract JSON from text")
    return None
