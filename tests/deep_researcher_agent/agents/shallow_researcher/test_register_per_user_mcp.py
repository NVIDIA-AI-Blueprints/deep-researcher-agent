# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for shallow_researcher ``_run`` per-user-MCP handling (DeepResearcher-002).

The standalone MCP image intentionally omits ``deep_researcher_api`` (the per-user auth layer).
The per-user-MCP block must degrade gracefully when those imports fail — previously
its ``except`` clause referenced an exception type imported *inside* the failing
``try``, so a missing ``deep_researcher_api`` raised ``UnboundLocalError`` and every shallow query
failed.
"""

import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deep_researcher_agent.agents.shallow_researcher import register as shallow_register
from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryError
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryReason


@tool
def _dummy_search(query: str) -> str:
    """Dummy search tool used to build the agent."""
    return f"result for {query}"


async def _build_run():
    """Build the register's real ``_run`` closure with a mocked NAT builder."""
    config = shallow_register.ShallowResearchAgentConfig(llm="llm_ref", tools=["dummy"])
    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_tools = AsyncMock(return_value=[_dummy_search])
    agen = shallow_register.shallow_research_agent.__wrapped__(config, builder)
    fn_info = await agen.__anext__()
    return fn_info.single_fn, agen


# deep_researcher_api and every submodule ``_run`` imports; setting each to ``None`` in
# sys.modules makes ``from deep_researcher_api... import ...`` raise ImportError.
_DEEP_RESEARCHER_API_MODULES = [
    "deep_researcher_api",
    "deep_researcher_api.jobs",
    "deep_researcher_api.jobs.access",
    "deep_researcher_api.mcp_auth",
    "deep_researcher_api.mcp_auth.provider",
    "deep_researcher_api.mcp_auth.runtime_tools",
]


@pytest.mark.asyncio
async def test_shallow_run_skips_per_user_mcp_when_deep_researcher_api_absent():
    """Without deep_researcher_api (standalone MCP), ``_run`` skips per-user MCP and proceeds
    instead of raising UnboundLocalError."""
    run, agen = await _build_run()
    try:
        sentinel = ShallowResearchAgentState(messages=[HumanMessage(content="answered")])
        with (
            patch.dict(sys.modules, {m: None for m in _DEEP_RESEARCHER_API_MODULES}),
            patch.object(shallow_register, "ShallowResearcherAgent") as mock_agent_cls,
            patch("deep_researcher_agent.common.validate_tool_availability", return_value=(True, 1, [])),
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=sentinel)
            state = ShallowResearchAgentState(messages=[HumanMessage(content="What year was NVIDIA founded?")])
            result = await run(state)
        # Reached agent.run -> got past the per-user block without crashing.
        assert result is sentinel
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_shallow_run_surfaces_reconnect_when_source_unavailable():
    """With deep_researcher_api present, an unresolvable per-user source yields a reconnect
    message rather than silently answering without it."""
    pytest.importorskip("deep_researcher_api")
    from deep_researcher_api.mcp_auth.runtime_tools import PerUserMcpSourceUnavailableError

    run, agen = await _build_run()
    try:
        with (
            patch("deep_researcher_api.jobs.access.require_verified_principal", return_value=MagicMock()),
            patch("deep_researcher_api.mcp_auth.provider.principal_user_id", return_value="user-1"),
            patch(
                "deep_researcher_api.mcp_auth.runtime_tools.open_per_user_mcp_tools",
                AsyncMock(side_effect=PerUserMcpSourceUnavailableError(["google_drive"])),
            ),
        ):
            state = ShallowResearchAgentState(messages=[HumanMessage(content="summarize my drive doc")])
            result = await run(state)
        assert isinstance(result, ShallowResearchAgentState)
        content = result.messages[-1].content
        assert "google_drive" in content
        assert "Reconnect them in the data sources panel" in content
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_registered_shallow_run_raises_typed_no_sources_for_explicit_empty_selection():
    """Explicit empty selection fails before filtering or per-user setup."""
    run, agen = await _build_run()
    try:
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="research this")],
            data_sources=[],
        )
        with patch.object(shallow_register, "filter_tools_by_sources") as filter_tools:
            with pytest.raises(EmptySourceRegistryError) as exc_info:
                await run(state)

        assert exc_info.value.reason is EmptySourceRegistryReason.NO_SOURCES_SELECTED
        filter_tools.assert_not_called()
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_registered_shallow_run_classifies_selected_unavailable_tool():
    run, agen = await _build_run()
    try:
        _dummy_search.description = "Web search (unavailable - missing API key)"
        state = ShallowResearchAgentState(messages=[HumanMessage(content="research this")])
        with patch.dict(sys.modules, {m: None for m in _DEEP_RESEARCHER_API_MODULES}):
            with pytest.raises(EmptySourceRegistryError) as exc_info:
                await run(state)

        assert exc_info.value.reason is EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE
    finally:
        _dummy_search.description = "Dummy search tool used to build the agent."
        await agen.aclose()


@pytest.mark.asyncio
async def test_shallow_evaluation_wrapper_returns_public_response_for_empty_sources():
    generated_answer = "A source-free draft"
    error = EmptySourceRegistryError(generated_answer=generated_answer)
    agent_fn = MagicMock()
    agent_fn.ainvoke = AsyncMock(side_effect=error)
    builder = MagicMock()
    builder.get_function = AsyncMock(return_value=agent_fn)
    config = shallow_register.ShallowResearchWorkflowConfig()

    registration = shallow_register.shallow_research_workflow.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        response = await function_info.single_fn("research this")
    finally:
        await registration.aclose()

    assert response.choices[0].message.content == error.public_response
