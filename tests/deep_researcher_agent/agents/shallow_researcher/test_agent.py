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

"""Tests for the ShallowResearcherAgent."""

import asyncio
import re
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from deep_researcher_agent.agents.shallow_researcher.agent import ShallowResearcherAgent
from deep_researcher_agent.agents.shallow_researcher.agent import _append_minimal_citation
from deep_researcher_agent.agents.shallow_researcher.agent import _format_chat_references
from deep_researcher_agent.agents.shallow_researcher.agent import _has_citation_integrity
from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole
from deep_researcher_agent.common.callbacks import SUPPRESS_OUTPUT_ARTIFACT_TAG
from deep_researcher_agent.common.citation_verification import CitationIntegrityError
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryError
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryReason
from deep_researcher_agent.common.citation_verification import SourceEntry
from deep_researcher_agent.common.citation_verification import SourceRegistry
from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import reset_registry


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool
def empty_web_search_tool(query: str) -> str:
    """Search the web but return no usable evidence."""
    return "Search returned no results"


class TestShallowResearcherAgent:
    """Tests for the ShallowResearcherAgent class."""

    @pytest.fixture(autouse=True)
    def _bypass_citation_pipeline(self):
        """Bypass citation verification for tests that don't test it.

        These tests mock the LLM to return AIMessage directly (no tool calls),
        so no tools execute and the source registry stays empty. Patching the
        pipeline avoids EmptySourceRegistryError in run().
        """
        with (
            patch.object(SourceRegistry, "all_sources", return_value=[SourceEntry(url="https://example.com")]),
            patch("deep_researcher_agent.agents.shallow_researcher.agent.verify_citations") as mock_verify,
            patch("deep_researcher_agent.agents.shallow_researcher.agent.sanitize_report") as mock_sanitize,
            patch("deep_researcher_agent.agents.shallow_researcher.agent._has_citation_integrity", return_value=True),
        ):
            mock_verify.side_effect = lambda content, reg: MagicMock(verified_report=content, removed_citations=[])
            mock_sanitize.side_effect = lambda content: MagicMock(sanitized_report=content)
            yield

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.fixture
    def real_tool(self):
        """Create a real LangChain tool."""
        return web_search_tool

    def test_init_with_defaults(self, mock_llm_provider, real_tool):
        """Test ShallowResearcherAgent initialization with defaults."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
        )

        assert agent.llm_provider == mock_llm_provider
        assert len(agent.tools) == 1
        assert agent.max_llm_turns == 10
        assert agent.max_tool_iterations == 5
        assert agent.citation_repair_timeout == 60.0
        assert agent.enforce_citations is False
        assert agent.callbacks == []
        assert agent.system_prompt is not None

    def test_init_with_custom_prompt(self, mock_llm_provider, real_tool):
        """Test ShallowResearcherAgent initialization with custom system prompt."""
        custom_system = "Custom system prompt"
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
            system_prompt=custom_system,
        )
        assert agent.system_prompt == custom_system

    def test_init_with_custom_limits(self, mock_llm_provider, real_tool):
        """Test ShallowResearcherAgent initialization with custom limits."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
            max_llm_turns=5,
            max_tool_iterations=3,
            enforce_citations=True,
        )

        assert agent.max_llm_turns == 5
        assert agent.max_tool_iterations == 3
        assert agent.enforce_citations is True

    def test_init_with_callbacks(self, mock_llm_provider, real_tool):
        """Test ShallowResearcherAgent initialization with callbacks."""
        callbacks = [MagicMock()]
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
            callbacks=callbacks,
        )

        assert agent.callbacks == callbacks

    def test_init_with_empty_tools(self, mock_llm_provider):
        """Test ShallowResearcherAgent initialization with empty tools."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
        )

        assert agent.tools == []
        assert agent.tools_info == []

    @pytest.mark.asyncio
    async def test_empty_tools_invoke_unbound_llm(self, mock_llm_provider, mock_llm):
        """An empty tool selection must omit the provider's tools property."""
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Answer without tools"))
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[])

        result = await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="Answer directly")]))

        assert result.messages[-1].content == "Answer without tools"
        mock_llm.bind_tools.assert_not_called()
        mock_llm.ainvoke.assert_awaited()

    def test_build_tools_info(self, mock_llm_provider, real_tool):
        """Test _build_tools_info correctly extracts tool information."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
        )

        assert len(agent.tools_info) == 1
        assert agent.tools_info[0]["name"] == "web_search_tool"
        assert "Search the web" in agent.tools_info[0]["description"]

    def test_get_llm(self, mock_llm_provider, mock_llm, real_tool):
        """Test _get_llm returns LLM from provider."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
        )

        result = agent._get_llm()

        mock_llm_provider.get.assert_called_with(LLMRole.RESEARCHER)
        assert result == mock_llm

    def test_graph_property(self, mock_llm_provider, real_tool):
        """Test graph property returns compiled graph."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
        )

        assert agent.graph is not None
        assert agent.graph == agent._graph

    @pytest.mark.asyncio
    async def test_run_basic_query(self, mock_llm_provider, mock_llm, real_tool):
        """Test run() with a basic query."""
        # Create a proper AI response for the agent node
        agent_response = AIMessage(content="CUDA is a parallel computing platform.")
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")])

        result = await agent.run(state)

        assert result is not None
        assert result.messages is not None

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self, mock_llm_provider, mock_llm, real_tool):
        """Test run() passes callbacks to config."""
        agent_response = AIMessage(content="Answer")
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        mock_callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
            callbacks=[mock_callback],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="Test")])

        await agent.run(state)

        # Agent should complete without errors

    @pytest.mark.asyncio
    async def test_run_with_user_info(self, mock_llm_provider, mock_llm, real_tool):
        """Test run() with user info in state."""
        agent_response = AIMessage(content="Personalized answer")
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        # Use custom system_prompt that doesn't require email field
        custom_prompt = "You are an assistant. User: {{ user_info }}."
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
            system_prompt=custom_prompt,
        )

        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test query")],
            user_info={"name": "John", "role": "developer"},
        )

        result = await agent.run(state)

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_with_tools_info_in_state(self, mock_llm_provider, mock_llm, real_tool):
        """Test run() uses tools_info from state if provided."""
        agent_response = AIMessage(content="Answer")
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
        )

        custom_tools_info = [
            {"name": "custom_tool", "description": "A custom tool"},
        ]

        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test query")],
            tools_info=custom_tools_info,
        )

        result = await agent.run(state)

        assert result is not None

    def test_load_system_prompt_fallback(self, mock_llm_provider, real_tool):
        """Test _load_system_prompt returns fallback when file not found."""
        with patch(
            "deep_researcher_agent.agents.shallow_researcher.agent.load_prompt",
            side_effect=FileNotFoundError(),
        ):
            agent = ShallowResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            assert "research" in agent.system_prompt.lower()

    def test_default_prompt_has_structural_citation_contract(self, mock_llm_provider, real_tool):
        """Default prompt preserves the shared research and citation contract."""
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
        )

        assert re.search(r"\[\d+\]", agent.system_prompt)
        assert "**References:**" in agent.system_prompt
        assert "- [1] mcp_time__get_current_time" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_citation_repair_timeout_fails_closed(self, mock_llm_provider, mock_llm):
        """A stalled provider cannot extend a completed shallow run indefinitely."""

        async def stalled_repair(*_args, **_kwargs):
            await asyncio.sleep(1)

        mock_llm.ainvoke = AsyncMock(side_effect=stalled_repair)
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
            citation_repair_timeout=0.001,
        )

        async with asyncio.timeout(0.1):
            with pytest.raises(CitationIntegrityError, match="citation_integrity_lost"):
                await agent._repair_missing_citations(
                    [HumanMessage(content="Draft")],
                    [SourceEntry(url="https://example.com/source")],
                )

    @pytest.mark.asyncio
    async def test_initial_answer_without_tool_call_is_retried(self, mock_llm_provider, mock_llm, real_tool):
        """An initial memory-only answer is retried and replaced by a tool call."""
        initial_answer = AIMessage(content="Memory-only answer")
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "CUDA"}, "id": "retry-tool"}],
        )
        final_answer = AIMessage(content="Evidence-backed answer")
        mock_llm.ainvoke = AsyncMock(side_effect=[initial_answer, tool_call, final_answer])

        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
        result = await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

        assert result.messages[-1].content == "Evidence-backed answer"
        assert mock_llm.ainvoke.await_count == 3
        assert mock_llm.bind_tools.call_args_list[:2] == [
            call([real_tool]),
            call([real_tool], parallel_tool_calls=False),
        ]
        for invocation in mock_llm.ainvoke.await_args_list:
            assert invocation.kwargs["config"] == {"tags": [SUPPRESS_OUTPUT_ARTIFACT_TAG]}

    @pytest.mark.asyncio
    async def test_repeated_answer_without_tool_call_fails_closed(self, mock_llm_provider, mock_llm, real_tool):
        """A model that ignores the bounded tool-use retry cannot synthesize an answer."""
        mock_llm.ainvoke = AsyncMock(
            side_effect=[AIMessage(content="First answer"), AIMessage(content="Second answer")]
        )
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])

        with pytest.raises(RuntimeError, match="shallow_research_tool_required"):
            await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

    @pytest.mark.asyncio
    async def test_retry_with_multiple_tool_calls_fails_closed(self, mock_llm_provider, mock_llm, real_tool):
        """The bounded retry cannot schedule multiple tools or exceed its one-call contract."""
        multiple_tool_calls = AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search_tool", "args": {"query": "CUDA"}, "id": "retry-tool-1"},
                {"name": "web_search_tool", "args": {"query": "GPU"}, "id": "retry-tool-2"},
            ],
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[AIMessage(content="Memory-only answer"), multiple_tool_calls])
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])

        with pytest.raises(RuntimeError, match="exactly one allowed research tool"):
            await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

        assert mock_llm.ainvoke.await_count == 2
        assert mock_llm.bind_tools.call_args_list == [
            call([real_tool]),
            call([real_tool], parallel_tool_calls=False),
        ]

    @pytest.mark.asyncio
    async def test_retry_with_unknown_tool_fails_closed(self, mock_llm_provider, mock_llm, real_tool):
        """The bounded retry cannot schedule a tool outside the agent's allowlist."""
        unknown_tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "unavailable_tool", "args": {}, "id": "retry-tool"}],
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[AIMessage(content="Memory-only answer"), unknown_tool_call])
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])

        with pytest.raises(RuntimeError, match="exactly one allowed research tool"):
            await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

        assert mock_llm.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_tool_iterations_incremented_on_tool_calls(self, mock_llm_provider, mock_llm, real_tool):
        """Test tool_iterations counter increments when LLM makes tool calls."""
        # First call returns tool calls, second call returns final answer
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "test"}, "id": "1"}],
        )
        final_response = AIMessage(content="Final answer")
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
        )

        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test query")],
            tool_iterations=0,
        )

        result = await agent.run(state)

        # tool_iterations should have been incremented
        assert result.tool_iterations >= 1

    @pytest.mark.asyncio
    async def test_forced_synthesis_at_max_iterations(self, mock_llm_provider, mock_llm, real_tool):
        """Test that agent forces synthesis when max_tool_iterations is reached."""
        # Response would normally include tool calls, but should be overridden
        final_response = AIMessage(content="Forced synthesis response")
        mock_llm.ainvoke = AsyncMock(return_value=final_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
            max_tool_iterations=3,
        )

        # Start with iterations already at max
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test query")],
            tool_iterations=3,
        )

        result = await agent.run(state)

        assert result is not None
        # The unbounded LLM should have been called (without tools)
        mock_llm.ainvoke.assert_called()

    def test_state_has_tool_iterations_field(self):
        """Test that ShallowResearchAgentState has tool_iterations field."""
        state = ShallowResearchAgentState(messages=[HumanMessage(content="Test")])
        assert hasattr(state, "tool_iterations")
        assert state.tool_iterations == 0

    def test_state_tool_iterations_default_value(self):
        """Test tool_iterations defaults to 0."""
        state = ShallowResearchAgentState(messages=[HumanMessage(content="Test")])
        assert state.tool_iterations == 0

    def test_state_tool_iterations_can_be_set(self):
        """Test tool_iterations can be set to custom value."""
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test")],
            tool_iterations=5,
        )
        assert state.tool_iterations == 5

    @pytest.mark.asyncio
    async def test_run_returns_updated_tool_iterations(self, mock_llm_provider, mock_llm, real_tool):
        """Test that run() returns state with updated tool_iterations."""
        agent_response = AIMessage(content="Answer")
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
        )

        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test")],
            tool_iterations=0,
        )

        result = await agent.run(state)

        # Result should have tool_iterations field
        assert hasattr(result, "tool_iterations")

    @pytest.mark.asyncio
    async def test_forced_synthesis_adds_instruction_message(self, mock_llm_provider, mock_llm, real_tool):
        """Test that forced synthesis adds instruction to synthesize."""
        captured_messages = []
        captured_configs = []

        async def capture_messages(messages, *, config):
            captured_messages.append(messages)
            captured_configs.append(config)
            return AIMessage(content="Synthesized response")

        mock_llm.ainvoke = AsyncMock(side_effect=capture_messages)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[real_tool],
            max_tool_iterations=2,
        )

        # Start at max iterations to trigger forced synthesis
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test query")],
            tool_iterations=2,
        )

        await agent.run(state)

        # Check that synthesis instruction was added
        last_call_messages = captured_messages[0]
        synthesis_instruction_found = any(
            "synthesize" in str(msg.content).lower() for msg in last_call_messages if hasattr(msg, "content")
        )
        assert synthesis_instruction_found
        assert captured_configs == [{"tags": [SUPPRESS_OUTPUT_ARTIFACT_TAG]}]


# ---------------------------------------------------------------------------
# Integration tests — verify end-to-end source capture without bypasses
# ---------------------------------------------------------------------------


@tool
def web_search_with_urls(query: str) -> str:
    """Search the web and return results with URLs."""
    return (
        '<Document href="https://docs.nvidia.com/cuda/">\n'
        "<title>\nCUDA Toolkit Documentation\n</title>\n"
        "CUDA is a parallel computing platform.\n"
        "</Document>"
    )


@tool
def mcp_time__get_current_time(timezone: str = "UTC") -> str:
    """Get the current time for a timezone."""
    return "2026-05-11T14:30:00+09:00"


@tool
def weather_observation_tool(location: str) -> str:
    """Get current observed weather conditions."""
    return f"Current conditions for {location}: clear, 68F"


@tool
def knowledge_search(query: str) -> str:
    """Search uploaded documents and return file citation metadata."""
    return (
        "Found 2 relevant document(s):\n\n"
        "--- Result 1 ---\n"
        "Source: report.pdf\n"
        "Page: 15\n"
        "Citation: report.pdf, p.15\n\n"
        f"Relevant content for {query}.\n\n"
        "--- Result 2 ---\n"
        "Source: appendix.pdf\n"
        "Page: 4\n"
        "Citation: appendix.pdf, p.4\n\n"
        "Supporting appendix content."
    )


class TestShallowResearcherSourceRegistryGating:
    """Tests that shallow source capture is gated by data_source_registry."""

    @pytest.fixture(autouse=True)
    def _reset_data_source_registry(self):
        reset_registry()
        yield
        reset_registry()

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_explicit_tool_not_declared_as_data_source_is_not_captured(self, mock_llm_provider, mock_llm):
        """Loaded agent tools are not enough; the tool must be in data_source_registry."""
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "mcp_time__get_current_time", "args": {"timezone": "Asia/Tokyo"}, "id": "1"}],
        )
        final_response = AIMessage(
            content=("The current time was returned by the MCP tool.\n\n## Sources\n[1] mcp_time__get_current_time")
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[mcp_time__get_current_time],
            enforce_citations=True,
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What time is it in Tokyo?")])
        with pytest.raises(EmptySourceRegistryError):
            await agent.run(state)

        assert agent.source_registry.all_sources() == []

    @pytest.mark.asyncio
    async def test_registered_group_tool_without_urls_is_captured(self, mock_llm_provider, mock_llm):
        """Registered group child tools without URLs can be non-URL citation sources."""
        populate_from_config(
            [
                {
                    "id": "mcp_time",
                    "name": "MCP Time",
                    "description": "Get current time and timezone information through MCP.",
                    "tools": ["mcp_time"],
                }
            ],
            group_names={"mcp_time"},
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "mcp_time__get_current_time", "args": {"timezone": "Asia/Tokyo"}, "id": "1"}],
        )
        final_response = AIMessage(
            content=("The current time was returned by the MCP tool.\n\n## Sources\n[1] mcp_time__get_current_time")
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[mcp_time__get_current_time],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What time is it in Tokyo?")])
        result = await agent.run(state)

        sources = agent.source_registry.all_sources()
        assert len(sources) == 1
        assert sources[0].citation_key == "mcp_time__get_current_time"
        assert sources[0].source_type == "tool_result"
        assert result.messages[-1].content.rstrip().endswith("[1] mcp_time__get_current_time")

    @pytest.mark.asyncio
    async def test_missing_tool_result_citation_is_appended(self, mock_llm_provider, mock_llm):
        """Captured non-URL tool sources are appended when the model omits references."""
        populate_from_config(
            [
                {
                    "id": "mcp_time",
                    "name": "MCP Time",
                    "description": "Get current time and timezone information through MCP.",
                    "tools": ["mcp_time"],
                }
            ],
            group_names={"mcp_time"},
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "mcp_time__get_current_time", "args": {"timezone": "Asia/Tokyo"}, "id": "1"}],
        )
        final_response = AIMessage(content="It's currently 4:54 AM in Tokyo.")
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[mcp_time__get_current_time],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What time is it in Tokyo?")])
        result = await agent.run(state)

        assert result.messages[-1].content.rstrip() == (
            "It's currently 4:54 AM in Tokyo [1].\n\n**References:**\n- [1] mcp_time__get_current_time"
        )

    @pytest.mark.asyncio
    async def test_missing_uploaded_document_citation_is_repaired_before_publication(self, mock_llm_provider, mock_llm):
        """Knowledge-search summaries enforce file citation keys without requiring public URLs."""
        populate_from_config(
            [
                {
                    "id": "knowledge_layer",
                    "name": "Knowledge Layer",
                    "description": "Search uploaded documents.",
                    "tools": ["knowledge_search"],
                }
            ]
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "knowledge_search", "args": {"query": "revenue summary"}, "id": "1"}],
        )
        uncited_summary = AIMessage(content="The uploaded report says revenue increased.")
        repaired_summary = AIMessage(
            content=("The uploaded report says revenue increased [1].\n\n**References:**\n- [1] report.pdf, p.15")
        )
        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, uncited_summary])
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)
        mock_llm.ainvoke = AsyncMock(return_value=repaired_summary)
        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[knowledge_search],
            enforce_citations=True,
            callbacks=[callback],
        )

        result = await agent.run(
            ShallowResearchAgentState(messages=[HumanMessage(content="Summarize the uploaded revenue report.")])
        )

        sources = agent.source_registry.all_sources()
        assert len(sources) == 2
        assert sources[0].citation_key == "report.pdf, p.15"
        assert sources[0].source_type == "knowledge_layer"
        assert sources[1].citation_key == "appendix.pdf, p.4"
        assert "The uploaded report says revenue increased [1]." in result.messages[-1].content
        assert "[1] report.pdf, p.15" in result.messages[-1].content
        assert bound_llm.ainvoke.await_count == 2
        mock_llm.ainvoke.assert_awaited_once()
        callback.emit_final_report.assert_called_once_with(result.messages[-1].content, cited_urls=[])

    @pytest.mark.asyncio
    async def test_missing_url_citation_fallback_emits_authoritative_metadata(self, mock_llm_provider, mock_llm):
        """A single verified URL appended by fallback must be persisted as cited."""
        populate_from_config(
            [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web for real-time information.",
                    "tools": ["web_search_with_urls"],
                }
            ]
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "1"}],
        )
        mock_llm.ainvoke = AsyncMock(
            side_effect=[tool_call_response, AIMessage(content="CUDA is a parallel computing platform.")]
        )
        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_with_urls],
            callbacks=[callback],
        )

        result = await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

        assert "https://docs.nvidia.com/cuda/" in result.messages[-1].content
        callback.emit_final_report.assert_called_once_with(
            result.messages[-1].content,
            cited_urls=["https://docs.nvidia.com/cuda/"],
        )

    @pytest.mark.asyncio
    async def test_missing_citations_are_repaired_once_for_multiple_sources(self, mock_llm_provider, mock_llm):
        """Ambiguous multi-source drafts get one bounded repair instead of a guessed citation."""
        populate_from_config(
            [
                {
                    "id": "mcp_time",
                    "name": "MCP Time",
                    "description": "Get current time and timezone information through MCP.",
                    "tools": ["mcp_time"],
                },
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web for real-time information.",
                    "tools": ["web_search_with_urls"],
                },
            ],
            group_names={"mcp_time"},
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "mcp_time__get_current_time", "args": {"timezone": "Asia/Tokyo"}, "id": "1"},
                {"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "2"},
            ],
        )
        final_response = AIMessage(content="CUDA is a parallel computing platform.")
        repaired_response = AIMessage(
            content=(
                "CUDA is a parallel computing platform [2]. The current time came from the time tool [1].\n\n"
                "**References:**\n"
                "- [1] mcp_time__get_current_time\n"
                "- [2] Source 2 - https://docs.nvidia.com/cuda/"
            )
        )
        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)
        mock_llm.ainvoke = AsyncMock(return_value=repaired_response)

        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[mcp_time__get_current_time, web_search_with_urls],
            enforce_citations=True,
            callbacks=[callback],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA? Also note the time.")])
        result = await agent.run(state)

        sources = agent.source_registry.all_sources()
        assert len(sources) >= 2
        assert sources[0].citation_key == "mcp_time__get_current_time"
        assert any(source.url == "https://docs.nvidia.com/cuda/" for source in sources)
        assert "CUDA is a parallel computing platform [2]" in result.messages[-1].content
        assert "mcp_time__get_current_time" in result.messages[-1].content
        callback.emit_final_report.assert_called_once_with(
            result.messages[-1].content,
            cited_urls=["https://docs.nvidia.com/cuda/"],
        )
        assert bound_llm.ainvoke.await_count == 2
        mock_llm.ainvoke.assert_awaited_once()
        assert mock_llm.bind_tools.call_count == 2
        repair_call = mock_llm.ainvoke.await_args
        assert repair_call.kwargs["config"]["tags"] == [SUPPRESS_OUTPUT_ARTIFACT_TAG]
        assert isinstance(repair_call.args[0][0], SystemMessage)

    @pytest.mark.asyncio
    async def test_default_multi_source_citation_integrity_failure_returns_draft(self, mock_llm_provider, mock_llm):
        """Default citation mode returns the answer without running the strict repair pass."""
        populate_from_config(
            [
                {
                    "id": "mcp_time",
                    "name": "MCP Time",
                    "description": "Get current time and timezone information through MCP.",
                    "tools": ["mcp_time"],
                },
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web for real-time information.",
                    "tools": ["web_search_with_urls"],
                },
            ],
            group_names={"mcp_time"},
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "mcp_time__get_current_time", "args": {"timezone": "Asia/Tokyo"}, "id": "1"},
                {"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "2"},
            ],
        )
        source_only_draft = AIMessage(
            content=(
                "CUDA is a parallel computing platform.\n\n"
                "**References:**\n- [1] CUDA Toolkit Documentation - https://docs.nvidia.com/cuda/"
            )
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, source_only_draft])
        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[mcp_time__get_current_time, web_search_with_urls],
            callbacks=[callback],
        )

        result = await agent.run(
            ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA? Also note the time.")])
        )

        assert "CUDA is a parallel computing platform." in result.messages[-1].content
        assert mock_llm.ainvoke.await_count == 2
        callback.emit_final_report.assert_called_once_with(
            result.messages[-1].content,
            cited_urls=["https://docs.nvidia.com/cuda/"],
        )

    @pytest.mark.asyncio
    async def test_failed_multi_source_repair_is_not_published(self, mock_llm_provider, mock_llm):
        """A single unsuccessful repair fails closed without emitting an uncited report."""
        populate_from_config(
            [
                {
                    "id": "mcp_time",
                    "name": "MCP Time",
                    "description": "Get current time and timezone information through MCP.",
                    "tools": ["mcp_time"],
                },
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web for real-time information.",
                    "tools": ["web_search_with_urls"],
                },
            ],
            group_names={"mcp_time"},
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "mcp_time__get_current_time", "args": {"timezone": "Asia/Tokyo"}, "id": "1"},
                {"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "2"},
            ],
        )
        source_only_draft = AIMessage(
            content=(
                "CUDA is a parallel computing platform.\n\n"
                "**References:**\n- [1] CUDA Toolkit Documentation - https://docs.nvidia.com/cuda/"
            )
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, source_only_draft, source_only_draft])
        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[mcp_time__get_current_time, web_search_with_urls],
            enforce_citations=True,
            callbacks=[callback],
        )

        with pytest.raises(CitationIntegrityError, match="citation_integrity_lost"):
            await agent.run(
                ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA? Also note the time.")])
            )

        assert mock_llm.ainvoke.await_count == 3
        for invocation in mock_llm.ainvoke.await_args_list:
            assert invocation.kwargs["config"]["tags"] == [SUPPRESS_OUTPUT_ARTIFACT_TAG]
        callback.emit_final_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_registered_exact_data_source_tool_without_urls_is_captured(self, mock_llm_provider, mock_llm):
        """Any exact tool declared under data_sources can be a non-URL citation source."""
        populate_from_config(
            [
                {
                    "id": "weather_observations",
                    "name": "Weather Observations",
                    "description": "Current observed weather conditions.",
                    "tools": ["weather_observation_tool"],
                }
            ]
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "weather_observation_tool", "args": {"location": "San Francisco"}, "id": "1"}],
        )
        final_response = AIMessage(content="The weather is clear.\n\n## Sources\n[1] weather_observation_tool")
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[weather_observation_tool],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What is the weather in San Francisco?")])
        result = await agent.run(state)

        sources = agent.source_registry.all_sources()
        assert len(sources) == 1
        assert sources[0].citation_key == "weather_observation_tool"
        assert sources[0].source_type == "tool_result"
        assert result.messages[-1].content.rstrip().endswith("[1] weather_observation_tool")


class TestShallowResearcherSourceCaptureIntegration:
    """Integration tests verifying source capture through the full pipeline.

    These tests do NOT bypass the citation pipeline — they verify that
    tool_node_with_source_capture registers sources from real tool execution,
    and that verify_citations + sanitize_report run on the final output.
    """

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.fixture(autouse=True)
    def _register_web_search_source(self):
        reset_registry()
        populate_from_config(
            [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web for real-time information.",
                    "tools": ["web_search_with_urls"],
                }
            ]
        )
        yield
        reset_registry()

    @pytest.mark.asyncio
    async def test_source_registry_populated_from_tool_call(self, mock_llm_provider, mock_llm):
        """Tool execution populates the source registry with extracted URLs."""
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "1"}],
        )
        final_response = AIMessage(
            content=(
                "CUDA is a parallel computing platform.\n\n"
                "## Sources\n"
                "[1] CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/"
            )
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_with_urls],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")])
        result = await agent.run(state)

        # Source registry should have the URL from tool output
        sources = agent.source_registry.all_sources()
        assert len(sources) >= 1
        assert any(s.url == "https://docs.nvidia.com/cuda/" for s in sources)

        # Final output should exist and have been processed
        assert result.messages[-1].content

    @pytest.mark.asyncio
    async def test_final_verification_report_is_published_with_its_cited_urls(self, mock_llm_provider, mock_llm):
        """The final report body and authoritative URL set come from the same verification pass."""
        source_url = "https://docs.nvidia.com/cuda/"
        draft_report = f"CUDA is a parallel computing platform [1].\n\n## Sources\n[1] CUDA Docs: {source_url}"
        expected_report = (
            "CUDA is a parallel computing platform [1].\n\n"
            "**References:**\n"
            "[1] CUDA Docs: https://docs.nvidia.com/cuda/"
        )
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "1"}],
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, AIMessage(content=draft_report)])
        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_with_urls],
            callbacks=[callback],
        )
        first_verification = MagicMock(
            verified_report=draft_report,
            valid_citations=[{"number": 1, "url": source_url}],
            removed_citations=[],
        )
        final_verification = MagicMock(
            verified_report=draft_report,
            valid_citations=[{"number": 1, "url": source_url}],
            removed_citations=[],
        )

        with patch(
            "deep_researcher_agent.agents.shallow_researcher.agent.verify_citations",
            side_effect=[first_verification, final_verification],
        ):
            result = await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

        assert result.messages[-1].content == expected_report
        callback.emit_final_report.assert_called_once_with(expected_report, cited_urls=[source_url])

    @pytest.mark.asyncio
    async def test_finalization_cannot_publish_a_report_after_losing_inline_citations(
        self, mock_llm_provider, mock_llm
    ):
        """A later sanitization regression must fail closed instead of publishing source-only prose."""
        source_url = "https://docs.nvidia.com/cuda/"
        cited_report = f"CUDA is a parallel computing platform [1].\n\n## Sources\n[1] CUDA Docs: {source_url}"
        source_only_report = f"CUDA is a parallel computing platform.\n\n## Sources\n[1] CUDA Docs: {source_url}"
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "1"}],
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, AIMessage(content=cited_report)])
        callback = MagicMock()
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_with_urls],
            enforce_citations=True,
            callbacks=[callback],
        )

        with (
            patch(
                "deep_researcher_agent.agents.shallow_researcher.agent.sanitize_report",
                return_value=MagicMock(sanitized_report=source_only_report),
            ),
            pytest.raises(CitationIntegrityError, match="citation_integrity_lost"),
        ):
            await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")]))

        callback.emit_final_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_citation_removed_end_to_end(self, mock_llm_provider, mock_llm):
        """Citations not backed by registry sources are removed from output."""
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_with_urls", "args": {"query": "CUDA"}, "id": "1"}],
        )
        # LLM fabricates a citation [2] not in the registry
        final_response = AIMessage(
            content=(
                "CUDA is great [1]. Also see this [2].\n\n"
                "## Sources\n"
                "[1] CUDA Docs: https://docs.nvidia.com/cuda/\n"
                "[2] Fake Source: https://totally-fabricated.example.com/fake"
            )
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_with_urls],
        )

        state = ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")])
        result = await agent.run(state)

        output = result.messages[-1].content
        # The fabricated URL should have been removed by verify_citations
        assert "totally-fabricated.example.com" not in output
        # The valid citation should survive
        assert "docs.nvidia.com/cuda" in output


# ---------------------------------------------------------------------------
# Session registry integration tests
# ---------------------------------------------------------------------------


class TestShallowResearcherSessionRegistry:
    """Tests verifying session-scoped SourceRegistry integration.

    These tests do NOT use the _bypass_citation_pipeline fixture — they verify
    the actual ContextVar-based session registry behavior.
    """

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_run_uses_session_registry_when_set(self, mock_llm_provider, mock_llm):
        """When session registry is set via ContextVar, run() uses it and doesn't raise."""
        from deep_researcher_agent.common.citation_verification import set_session_registry

        # Pre-populate a session registry with a source from a "prior turn"
        session_reg = SourceRegistry()
        session_reg.add(SourceEntry(url="https://prior-turn.example.com/article"))

        # LLM answers from memory (no tool calls) citing the prior-turn URL
        agent_response = AIMessage(
            content=(
                "Answer based on prior context [1].\n\n"
                "## Sources\n"
                "[1] Prior Article: https://prior-turn.example.com/article"
            )
        )
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
        )

        set_session_registry(session_reg)
        try:
            state = ShallowResearchAgentState(messages=[HumanMessage(content="Follow-up question")])
            result = await agent.run(state)
            # Should NOT raise EmptySourceRegistryError because session registry has sources
            assert result is not None
            assert "prior-turn.example.com/article" in result.messages[-1].content
        finally:
            set_session_registry(None)

    @pytest.mark.asyncio
    async def test_run_clears_registry_in_standalone_mode(self, mock_llm_provider, mock_llm):
        """Without session registry ContextVar, run() clears instance registry and raises."""
        from deep_researcher_agent.common.citation_verification import EmptySourceRegistryError
        from deep_researcher_agent.common.citation_verification import set_session_registry

        set_session_registry(None)  # Ensure no session registry

        # LLM answers without calling tools
        agent_response = AIMessage(content="Answer without sources")
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
            enforce_citations=True,
        )
        # Pre-populate the instance registry (simulating stale data)
        agent.source_registry.add(SourceEntry(url="https://stale.example.com"))

        state = ShallowResearchAgentState(messages=[HumanMessage(content="Test")])

        with pytest.raises(EmptySourceRegistryError):
            await agent.run(state)

    @pytest.mark.asyncio
    async def test_default_empty_registry_returns_sanitized_answer(self, mock_llm_provider, mock_llm):
        """By default, citation-source failures return the generated answer instead of raising."""
        from deep_researcher_agent.common.citation_verification import set_session_registry

        set_session_registry(None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Draft answer with https://private.example/path"))
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[])
        state = ShallowResearchAgentState(messages=[HumanMessage(content="Test")])

        result = await agent.run(state)

        assert result.messages[-1].content == "Draft answer with "

    @pytest.mark.parametrize(
        ("data_sources", "expected_reason"),
        [
            ([], EmptySourceRegistryReason.NO_SOURCES_SELECTED),
            (None, EmptySourceRegistryReason.NO_SOURCE_RESULTS),
            (["web"], EmptySourceRegistryReason.NO_SOURCE_RESULTS),
        ],
    )
    @pytest.mark.asyncio
    async def test_empty_registry_classification_preserves_sanitized_answer(
        self,
        mock_llm_provider,
        mock_llm,
        data_sources,
        expected_reason,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Draft answer with https://private.example/path"))
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[], enforce_citations=True)
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Test")],
            data_sources=data_sources,
        )

        with pytest.raises(EmptySourceRegistryError) as exc_info:
            await agent.run(state)

        assert exc_info.value.reason is expected_reason
        assert exc_info.value.generated_answer == "Draft answer with "

    @pytest.mark.asyncio
    async def test_enabled_source_empty_result_preserves_generated_answer(self, mock_llm_provider, mock_llm):
        populate_from_config(
            [
                {
                    "id": "web",
                    "name": "Web Search",
                    "description": "Search the web.",
                    "tools": ["empty_web_search_tool"],
                }
            ]
        )
        tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "empty_web_search_tool",
                    "args": {"query": "quantum computing"},
                    "id": "empty-search-1",
                }
            ],
        )
        generated_answer = "No supporting sources were found, so I cannot provide a sourced answer."
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call, AIMessage(content=generated_answer)])
        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[empty_web_search_tool],
            enforce_citations=True,
        )
        state = ShallowResearchAgentState(
            messages=[HumanMessage(content="Summarize quantum computing.")],
            data_sources=["web"],
        )

        try:
            with pytest.raises(EmptySourceRegistryError) as exc_info:
                await agent.run(state)
        finally:
            reset_registry()

        assert exc_info.value.reason is EmptySourceRegistryReason.NO_SOURCE_RESULTS
        assert exc_info.value.generated_answer == generated_answer
        assert "Try rephrasing the question" in exc_info.value.public_response
        assert mock_llm.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_registry_without_final_message_raises_typed_failure(self, mock_llm_provider):
        agent = ShallowResearcherAgent(llm_provider=mock_llm_provider, tools=[web_search_tool])
        agent._graph = MagicMock()
        agent._graph.ainvoke = AsyncMock(return_value={"messages": []})

        with (
            patch.object(SourceRegistry, "all_sources", return_value=[]),
            pytest.raises(EmptySourceRegistryError) as exc_info,
        ):
            await agent.run(ShallowResearchAgentState(messages=[HumanMessage(content="Test")]))

        assert exc_info.value.reason is EmptySourceRegistryReason.NO_SOURCE_RESULTS
        assert exc_info.value.generated_answer is None

    @pytest.mark.asyncio
    async def test_session_registry_does_not_mutate_shared_instance(self, mock_llm_provider, mock_llm):
        """Setting a session registry must NOT overwrite self.source_registry on the agent."""
        from deep_researcher_agent.common.citation_verification import set_session_registry

        session_reg = SourceRegistry()
        session_reg.add(SourceEntry(url="https://session.example.com/doc"))

        agent_response = AIMessage(content=("Answer [1].\n\n## Sources\n[1] Doc: https://session.example.com/doc"))
        mock_llm.ainvoke = AsyncMock(return_value=agent_response)

        agent = ShallowResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[],
        )
        original_registry = agent.source_registry

        set_session_registry(session_reg)
        try:
            state = ShallowResearchAgentState(messages=[HumanMessage(content="Q")])
            await agent.run(state)
            # The instance attribute must remain unchanged
            assert agent.source_registry is original_registry
        finally:
            set_session_registry(None)


class TestAppendMinimalCitation:
    """Unit tests for the `_append_minimal_citation` fallback."""

    def _tool_source(self) -> SourceEntry:
        return SourceEntry(
            source_type="tool_result",
            citation_key="mcp_time__get_current_time",
            tool_name="mcp_time__get_current_time",
        )

    def test_strips_leftover_bold_references_header(self):
        # Simulates verify_citations stripping every fabricated citation under
        # a **References:** section but leaving the bare header behind.
        report = "Body sentence.\n\n**References:**\n"

        result = _append_minimal_citation(report, self._tool_source())

        assert result.count("**References:**") == 1
        assert result == "Body sentence [1].\n\n**References:**\n- [1] mcp_time__get_current_time"

    def test_strips_leftover_references_heading(self):
        report = "Body sentence.\n\n## References\n"

        result = _append_minimal_citation(report, self._tool_source())

        assert "## References" not in result
        assert result.count("**References:**") == 1

    def test_strips_leftover_sources_heading(self):
        report = "Body sentence.\n\n### Sources\n"

        result = _append_minimal_citation(report, self._tool_source())

        assert "### Sources" not in result
        assert result.count("**References:**") == 1

    def test_no_leftover_header_passes_through(self):
        report = "Body sentence."

        result = _append_minimal_citation(report, self._tool_source())

        assert result == "Body sentence [1].\n\n**References:**\n- [1] mcp_time__get_current_time"

    def test_replaces_source_only_section_and_adds_inline_marker(self):
        report = "Body sentence.\n\n## Sources\n[1] mcp_time__get_current_time"

        result = _append_minimal_citation(report, self._tool_source())

        assert result == "Body sentence [1].\n\n**References:**\n- [1] mcp_time__get_current_time"

    @pytest.mark.parametrize("heading", ["###### References", "Sources:", "**References:**"])
    def test_replaces_canonical_heading_variants(self, heading):
        report = f"Body sentence.\n\n{heading}\n[1] mcp_time__get_current_time"

        result = _append_minimal_citation(report, self._tool_source())

        assert result.count("**References:**") == 1
        assert result == "Body sentence [1].\n\n**References:**\n- [1] mcp_time__get_current_time"

    @pytest.mark.parametrize(
        "report",
        [
            "## Sources of renewable energy\nSolar is renewable.",
            "## References to previous work\nPrior work remains relevant.",
            "Sources: market revenue grew last year.",
            "Sources:\nMarket revenue grew last year.",
        ],
    )
    def test_preserves_source_like_answer_content(self, report):
        result = _append_minimal_citation(report, self._tool_source())

        assert report.removesuffix(".") in result
        assert result.endswith("**References:**\n- [1] mcp_time__get_current_time")

    def test_replaces_reference_definitions_without_discarding_trailing_answer(self):
        report = "Opening answer.\n\n## References\n- [1] Old source - https://example.com/old\n\nClosing answer."

        result = _append_minimal_citation(report, self._tool_source())

        assert "Opening answer." in result
        assert "Closing answer [1]." in result
        assert "Old source" not in result

    def test_preserves_marker_first_answer_immediately_after_reference_block(self):
        report = "## Sources\n[1] mcp_time__get_current_time\n[1] CUDA is a parallel computing platform."

        result = _append_minimal_citation(report, self._tool_source())

        assert "CUDA is a parallel computing platform" in result
        assert result.count("mcp_time__get_current_time") == 1


class TestCitationIntegrity:
    """Tests for the final inline-plus-source publication invariant."""

    @pytest.mark.parametrize("heading", ["###### References", "Sources:", "**References:**"])
    def test_accepts_inline_marker_outside_reference_definitions(self, heading):
        report = f"{heading}\n- [1] Source - https://example.com/source\n\nAnswer [1]."

        assert _has_citation_integrity(report, [{"number": 1, "url": "https://example.com/source"}])

    def test_does_not_treat_reference_definition_as_inline_citation(self):
        report = "Answer without a marker.\n\nSources:\n- [1] Source - https://example.com/source"

        assert not _has_citation_integrity(report, [{"number": 1, "url": "https://example.com/source"}])

    def test_preserves_answer_line_that_begins_with_an_inline_marker(self):
        report = "[1] CUDA is a parallel computing platform.\n\nSources:\n- [1] CUDA - https://example.com/cuda"

        assert _has_citation_integrity(report, [{"number": 1, "url": "https://example.com/cuda"}])

    def test_does_not_count_definitions_across_multiple_source_sections(self):
        report = (
            "Answer without a marker.\n\n"
            "## Sources\n"
            "- [1] First - https://example.com/first\n\n"
            "## References\n"
            "- [1] Second - https://example.com/second"
        )

        assert not _has_citation_integrity(report, [{"number": 1, "url": "https://example.com/first"}])

    def test_preserves_marker_first_answer_after_reference_definitions(self):
        report = "## Sources\n[1] mcp_time__get_current_time\n[1] CUDA is a parallel computing platform."

        assert _has_citation_integrity(report, [{"number": 1, "citation_key": "mcp_time__get_current_time"}])


class TestFormatChatReferences:
    def test_removes_empty_sources_heading(self):
        assert _format_chat_references("Body sentence.\n\n## Sources\n") == "Body sentence."

    def test_uses_compact_heading_for_populated_sources(self):
        report = "Body sentence [1].\n\n## Sources\n[1] Example: https://example.com/article"

        assert _format_chat_references(report) == (
            "Body sentence [1].\n\n**References:**\n[1] Example: https://example.com/article"
        )
