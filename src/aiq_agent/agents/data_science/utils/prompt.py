# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic prompt middleware for the data-science agent."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.tools import BaseTool

from aiq_agent.common import render_prompt_template

from ..models import DataScienceAgentContext
from ..models import InteractionMode
from ..models import ResponseMode


def build_prompt_middleware(
    template: str,
    tools: Sequence[BaseTool],
    *,
    interaction_mode: InteractionMode = "interactive",
    response_mode: ResponseMode = "standard",
    structured_catalog_call_limit: int | None = None,
    structured_text_to_sql_call_limit: int | None = None,
    python_call_limit: int | None = None,
) -> AgentMiddleware:
    """Render the data-science prompt with the request's exact tool surface."""
    tool_descriptions = tuple({"name": tool.name, "description": tool.description} for tool in tools)

    @dynamic_prompt
    def data_science_prompt(request: ModelRequest[DataScienceAgentContext]) -> str:
        context = request.runtime.context
        user_info: dict[str, Any] | None = context.user_info if context is not None else None
        return render_prompt_template(
            template,
            tools=tool_descriptions,
            user_info=user_info,
            has_catalog_context=context is not None and context.catalog_context is not None,
            interaction_mode=interaction_mode,
            response_mode=response_mode,
            structured_catalog_call_limit=structured_catalog_call_limit,
            structured_text_to_sql_call_limit=structured_text_to_sql_call_limit,
            python_call_limit=python_call_limit,
            current_datetime=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    return data_science_prompt


__all__ = ["build_prompt_middleware"]
