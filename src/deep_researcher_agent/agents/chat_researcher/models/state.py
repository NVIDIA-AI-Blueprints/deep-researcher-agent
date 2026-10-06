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

"""State models for chat researcher agent."""

from typing import Annotated
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from deep_researcher_agent.knowledge import AvailableDocument

from ..request_context import DatabaseName
from .catalog import CatalogRoutingResponse
from .depth import DepthDecision
from .intent import IntentResult
from .result import ShallowResult
from .result import WorkflowOutcome


def _keep_if_set(old: str | None, new: str | None) -> str | None:
    """Reducer that keeps the prior value unless a new non-empty one is provided.

    ``register.py`` builds a fresh state each turn, so a plain field would be clobbered to
    None and never persist across turns. This lets ``last_report_markdown`` carry the
    in-session report forward (and be overwritten when a newer report is produced).
    """
    return new if new else old


class ChatResearcherState(BaseModel):
    """
    State for the main chat researcher workflow graph.

    Attributes:
        messages: Conversation history with LangGraph message reducer.
        user_info: Optional user information for personalization.
        data_sources: Optional list of user-selected data source IDs.
        database_name: Optional validated GSF database scope for this request.
        user_intent: Result of intent classification.
        depth_decision: Result of depth routing.
        final_report: The final research report.
        shallow_result: Result from shallow research (if executed).
        clarifier_result: Log from clarifier agent dialog.
        original_query: The latest user query, preserved for deep research.
        available_documents: User-uploaded documents with summaries for context.
        skip_clarifier: When True the clarifier node is bypassed regardless of
            ``enable_clarifier``.  Set automatically for API-key and anonymous
            callers so headless workflows do not stall waiting for user input.
        active_report_job_id: Identifier of the active asynchronous report used
            for report follow-up turns.
        catalog_context: Validated catalog routing result supplied to hybrid
            research; reset at each turn boundary.
        catalog_request_id: Request identifier associated with
            ``catalog_context``; reset at each turn boundary.
        last_report_markdown: Most recent report produced inline, retained
            across turns for report follow-up when no report job is available.
        workflow_outcome: Explicit terminal workflow failure when a node
            degrades an exception to fallback text; reset at each turn boundary.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict[str, Any] | None = None
    data_sources: list[str] | None = None
    database_name: DatabaseName | None = None
    user_intent: IntentResult | None = None
    depth_decision: DepthDecision | None = None
    final_report: str | None = None
    shallow_result: ShallowResult | None = None
    clarifier_result: str | None = None
    original_query: str | None = None
    available_documents: list[AvailableDocument] | None = None
    skip_clarifier: bool = False
    active_report_job_id: str | None = None
    catalog_context: CatalogRoutingResponse | None = None
    catalog_request_id: str | None = None
    # The most recent report produced inline in this session (synchronous CLI mode), carried
    # across turns by the keep-if-set reducer so report follow-up works without an async job.
    last_report_markdown: Annotated[str | None, _keep_if_set] = None
    workflow_outcome: WorkflowOutcome | None = None
