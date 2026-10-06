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

"""Result models for chat research agent."""

from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat.data_models.api_server import ChatResponse

RESEARCH_WORKFLOW_FAILURE_ERROR = "Research could not be completed. Please try again."


class WorkflowSuccess(BaseModel):
    """Successful terminal workflow outcome."""

    status: Literal["success"] = "success"
    result: str


class WorkflowFailure(BaseModel):
    """Failed terminal workflow outcome with a sanitized public error."""

    status: Literal["failed"] = "failed"
    error: str


WorkflowOutcome = Annotated[WorkflowSuccess | WorkflowFailure, Field(discriminator="status")]


class ChatResearcherResponse(ChatResponse):
    """Chat response carrying the workflow's explicit terminal outcome."""

    workflow_outcome: WorkflowOutcome


class ShallowResult(BaseModel):
    """
    Result from shallow research execution.

    Attributes:
        answer: The research answer or response text.
        confidence: Confidence level in the shallow research result.
        escalate_to_deep: Whether this query should be escalated to deep research.
        escalation_reason: Optional explanation for why escalation is needed.
    """

    answer: str
    confidence: Literal["low", "medium", "high"]
    escalate_to_deep: bool
    escalation_reason: str | None = None
