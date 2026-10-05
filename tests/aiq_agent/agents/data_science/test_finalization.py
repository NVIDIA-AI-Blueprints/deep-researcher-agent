# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reserved no-tool finalization turn."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage

from aiq_agent.agents.data_science.utils.analysis_runtime import begin_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import end_analysis_run
from aiq_agent.agents.data_science.utils.finalization import FinalizationReserveMiddleware


@pytest.mark.asyncio
async def test_finalization_removes_tools_at_configured_model_call() -> None:
    middleware = FinalizationReserveMiddleware(max_model_calls=2)
    configured_tools = [{"name": "gsf__text_to_sql"}]
    request = ModelRequest(
        model=MagicMock(),
        messages=[],
        system_message=SystemMessage(content="Base prompt"),
        tools=configured_tools,
        tool_choice="auto",
    )
    handler = AsyncMock(side_effect=lambda value: value)
    token = begin_analysis_run()
    try:
        first = await middleware.awrap_model_call(request, handler)
        second = await middleware.awrap_model_call(request, handler)
    finally:
        await end_analysis_run(token)

    assert first.tools == configured_tools
    assert first.tool_choice == "auto"
    assert first.system_message == SystemMessage(content="Base prompt")
    assert second.tools == []
    assert second.tool_choice is None
    assert "FINALIZATION TURN" in second.system_message.text
