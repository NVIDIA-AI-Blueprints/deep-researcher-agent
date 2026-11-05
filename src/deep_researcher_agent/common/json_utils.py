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
    3. Decode the JSON object that starts at the first {

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
        # Let the JSON decoder find the end of the object so braces inside
        # string values are not counted as structural braces.
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to extract JSON from text")
    return None
