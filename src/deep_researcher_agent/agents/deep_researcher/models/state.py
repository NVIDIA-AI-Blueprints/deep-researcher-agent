# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa: E501
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

"""State models for deep research agent."""

from typing import Annotated
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic import Field

from deep_researcher_agent.knowledge import AvailableDocument


def _merge_dict_state(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    if not left:
        return right or {}
    if not right:
        return left
    merged = dict(left)
    merged.update(right)
    return merged


class DeepResearchAgentState(BaseModel):
    """
    State for deep research agent.

    The deepagents-based DeepResearcherAgent manages its own internal state
    through the deepagents library. This state primarily handles the interface
    with the orchestrator.

    Attributes:
        messages: Conversation history with LangGraph message reducer.
        data_sources: List of data sources selected by the user.
        user_info: Optional user information.
        tools_info: Information about available tools.
        todos: Todo list managed by TodoListMiddleware.
        files: Virtual filesystem managed by FilesystemMiddleware.
        subagents: Status of configured DeepAgents subagents.
        rubric: DeepAgents rubric used by RubricMiddleware when available.
        clarifier_result: Log from clarifier agent dialog.
        available_documents: User-uploaded documents with summaries for context.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    data_sources: list[str] | None = None
    user_info: dict[str, Any] | None = None
    tools_info: list[dict[str, Any]] | None = None
    todos: list[dict[str, Any]] = Field(default_factory=list)
    files: Annotated[dict[str, Any], _merge_dict_state] = Field(default_factory=dict)
    subagents: list[dict[str, Any]] = Field(default_factory=list)
    rubric: str | None = None
    clarifier_result: str | None = None
    available_documents: list[AvailableDocument] | None = None
