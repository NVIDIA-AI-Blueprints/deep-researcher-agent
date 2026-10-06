# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State models for the internal report rewriter agent."""

from __future__ import annotations

from typing import Annotated
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic import Field


def _merge_dict_state(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    if not left:
        return right or {}
    if not right:
        return left
    merged = dict(left)
    merged.update(right)
    return merged


class ReportRewriterAgentState(BaseModel):
    """State for report rewrite child jobs."""

    messages: Annotated[list[AnyMessage], add_messages]
    files: Annotated[dict[str, Any], _merge_dict_state] = Field(default_factory=dict)
