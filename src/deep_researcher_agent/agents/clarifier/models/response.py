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

"""Response models for clarifier agent."""

from pydantic import BaseModel
from pydantic import Field


class ClarificationResponse(BaseModel):
    """
    Structured response from the clarifier agent.

    Attributes:
        needs_clarification: True if additional clarification is needed,
            False if the agent has enough information to proceed.
        clarification_question: The clarification question to ask the user.
            Required when needs_clarification is True, should be None otherwise.
    """

    needs_clarification: bool = Field(
        description="True if additional clarification is needed from the user, "
        "False if enough information has been gathered to proceed with research."
    )
    clarification_question: str | None = Field(
        default=None,
        description="The clarification question to ask the user. Required when needs_clarification is True.",
    )

    def is_complete(self) -> bool:
        """Check if clarification is complete."""
        return not self.needs_clarification

    def is_valid(self) -> bool:
        """Check if the response is valid (has question when needed)."""
        if self.needs_clarification:
            return bool(self.clarification_question)
        return True
