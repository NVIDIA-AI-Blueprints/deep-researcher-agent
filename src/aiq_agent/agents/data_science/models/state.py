# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT boundary state and LangChain runtime context."""

from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.chat_researcher.request_context import DatabaseName

InteractionMode = Literal["interactive", "headless"]
ResponseMode = Literal["standard", "fdabench_choice"]


class DataScienceAgentState(BaseModel):
    """Conversation state for one adaptive analytical trajectory."""

    messages: Annotated[list[AnyMessage], add_messages]
    data_sources: list[str] | None = None
    user_info: dict[str, Any] | None = None
    database_name: DatabaseName | None = None
    catalog_context: CatalogRoutingResponse | None = None
    catalog_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class DataScienceAgentContext:
    """Request context available to dynamic agent middleware."""

    user_info: dict[str, Any] | None = None
    database_name: DatabaseName | None = None
    catalog_context: CatalogRoutingResponse | None = None
    catalog_request_id: str | None = None
