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

"""Utility functions for clarifier agent."""

from typing import Any


def extract_user_response(response: Any) -> str:
    """
    Extract user response text from a NAT HumanResponse object.

    Args:
        response: HumanResponse object from NAT user interaction.

    Returns:
        The user's response text.
    """
    # Handle string responses directly
    if isinstance(response, str):
        return response

    # Handle NAT HumanResponse with nested content structure
    # response.content is HumanResponseText with a text attribute
    if hasattr(response, "content"):
        content = response.content
        if hasattr(content, "text"):
            return str(content.text)

    # Fallback: check for direct text attribute
    if hasattr(response, "text"):
        return str(response.text)

    return str(response)
