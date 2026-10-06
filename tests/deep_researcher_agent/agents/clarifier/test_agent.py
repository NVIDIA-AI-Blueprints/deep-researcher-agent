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

"""Tests for the ClarifierAgent."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from nemo_relay.integrations.langchain import NemoRelayMiddleware

from deep_researcher_agent.agents.clarifier.agent import DEFAULT_CLARIFICATION_PROMPT
from deep_researcher_agent.agents.clarifier.agent import FORCE_SEARCH_GUIDANCE
from deep_researcher_agent.agents.clarifier.agent import ClarifierAgent
from deep_researcher_agent.agents.clarifier.models import ClarificationResponse
from deep_researcher_agent.agents.clarifier.models import ClarifierAgentState
from deep_researcher_agent.agents.clarifier.models import ClarifierResult
from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


def _adjacent_assistant_pairs(messages: list[BaseMessage]) -> list[tuple[int, int]]:
    """Return index pairs of any two consecutive AIMessages (the invalid shape)."""
    return [
        (i, i + 1)
        for i in range(len(messages) - 1)
        if isinstance(messages[i], AIMessage) and isinstance(messages[i + 1], AIMessage)
    ]


class TestClarifierAgentInit:
    """Tests for ClarifierAgent initialization."""

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
    def mock_user_callback(self):
        """Create a mock user prompt callback."""
        return AsyncMock(return_value="User response")

    def test_init_with_defaults(self, mock_llm_provider, mock_user_callback):
        """Test initialization with default values."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        assert agent.llm_provider == mock_llm_provider
        assert agent.tools == []
        assert agent.user_prompt_callback == mock_user_callback
        assert agent.max_turns == 3
        assert agent.log_response_max_chars == 2000
        assert agent.callbacks == []
        assert agent.system_prompt is not None

    def test_init_with_tools(self, mock_llm_provider, mock_user_callback):
        """Test initialization with tools."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=mock_user_callback,
        )

        assert len(agent.tools) == 1

    def test_init_with_custom_max_turns(self, mock_llm_provider, mock_user_callback):
        """Test initialization with custom max_turns."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            max_turns=5,
        )

        assert agent.max_turns == 5

    def test_init_with_callbacks(self, mock_llm_provider, mock_user_callback):
        """Test initialization with callbacks."""
        mock_callback = MagicMock()
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            callbacks=[mock_callback],
        )

        assert agent.callbacks == [mock_callback]

    def test_graph_property(self, mock_llm_provider, mock_user_callback):
        """Test graph property returns compiled graph."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        assert agent.graph is not None
        assert agent.graph == agent._graph

    def test_get_llm(self, mock_llm_provider, mock_llm, mock_user_callback):
        """Test _get_llm returns LLM from provider."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        result = agent._get_llm()

        mock_llm_provider.get.assert_called_with(LLMRole.CLARIFIER)
        assert result == mock_llm


class TestClarifierAgentPromptLoading:
    """Tests for prompt loading functionality."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)
        return provider

    @pytest.fixture
    def mock_user_callback(self):
        """Create a mock user prompt callback."""
        return AsyncMock(return_value="Response")

    def test_load_prompt_fallback(self, mock_llm_provider, mock_user_callback):
        """Test fallback to default prompt when file not found."""
        with patch(
            "deep_researcher_agent.agents.clarifier.agent.load_prompt",
            side_effect=FileNotFoundError(),
        ):
            agent = ClarifierAgent(
                llm_provider=mock_llm_provider,
                user_prompt_callback=mock_user_callback,
            )
            assert agent.system_prompt == DEFAULT_CLARIFICATION_PROMPT

    def test_load_prompt_success(self, mock_llm_provider, mock_user_callback):
        """Test successful prompt loading."""
        custom_prompt = "Custom clarification prompt"
        with patch(
            "deep_researcher_agent.agents.clarifier.agent.load_prompt",
            return_value=custom_prompt,
        ):
            agent = ClarifierAgent(
                llm_provider=mock_llm_provider,
                user_prompt_callback=mock_user_callback,
            )
            assert agent.system_prompt == custom_prompt


class TestClarifierAgentParsing:
    """Tests for JSON response parsing."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing parsing methods."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_parse_response_valid_json(self, agent):
        """Test parsing valid JSON response."""
        text = '{"needs_clarification": true, "clarification_question": "What scope?"}'
        result = agent._parse_response(text)

        assert result is not None
        assert result.needs_clarification is True
        assert result.clarification_question == "What scope?"

    def test_parse_response_with_code_block(self, agent):
        """Test parsing JSON wrapped in code block."""
        text = '```json\n{"needs_clarification": false, "clarification_question": null}\n```'
        result = agent._parse_response(text)

        assert result is not None
        assert result.needs_clarification is False

    def test_parse_response_invalid_json(self, agent):
        """Test parsing invalid JSON returns None."""
        result = agent._parse_response("not valid json")
        assert result is None

    def test_parse_response_empty_string(self, agent):
        """Test parsing empty string returns None."""
        result = agent._parse_response("")
        assert result is None

    def test_parse_response_none(self, agent):
        """Test parsing None returns None."""
        result = agent._parse_response(None)
        assert result is None

    def test_is_needed_true(self, agent):
        """Test _is_needed returns True when needed."""
        text = '{"needs_clarification": true, "clarification_question": "What?"}'
        assert agent._is_needed(text) is True

    def test_is_needed_false(self, agent):
        """Test _is_needed returns False when not needed."""
        text = '{"needs_clarification": false, "clarification_question": null}'
        assert agent._is_needed(text) is False

    def test_is_needed_invalid_json(self, agent):
        """Test _is_needed returns True for invalid JSON (safe default)."""
        assert agent._is_needed("invalid") is True

    def test_is_complete_true(self, agent):
        """Test _is_complete returns True when complete."""
        text = '{"needs_clarification": false, "clarification_question": null}'
        assert agent._is_complete(text) is True

    def test_is_complete_false(self, agent):
        """Test _is_complete returns False when not complete."""
        text = '{"needs_clarification": true, "clarification_question": "What?"}'
        assert agent._is_complete(text) is False

    def test_is_complete_invalid_json(self, agent):
        """Test _is_complete returns False for invalid JSON."""
        assert agent._is_complete("invalid") is False

    def test_valid_needed_true(self, agent):
        """Test _valid_needed returns True for valid response."""
        text = '{"needs_clarification": true, "clarification_question": "What scope?"}'
        assert agent._valid_needed(text) is True

    def test_valid_needed_no_question_mark(self, agent):
        """Test _valid_needed returns True even without question mark."""
        text = '{"needs_clarification": true, "clarification_question": "Tell me more"}'
        assert agent._valid_needed(text) is True

    def test_valid_needed_invalid_json(self, agent):
        """Test _valid_needed returns False for invalid JSON."""
        assert agent._valid_needed("invalid") is False

    def test_get_clarification_question(self, agent):
        """Test extracting clarification question."""
        text = '{"needs_clarification": true, "clarification_question": "What aspect?"}'
        result = agent._get_clarification_question(text)
        assert result == "What aspect?"

    def test_get_clarification_question_fallback(self, agent):
        """Test fallback question for invalid response."""
        result = agent._get_clarification_question("invalid")
        assert "provide more details" in result.lower()


class TestClarifierAgentSkipCommands:
    """Tests for skip command detection."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    @pytest.mark.parametrize("command", ["skip", "done", "exit", "quit", "proceed", "continue", "no", "n", ""])
    def test_is_skip_command_recognized(self, agent, command):
        """Test all skip commands are recognized."""
        assert agent._is_skip_command(command) is True

    @pytest.mark.parametrize("command", ["SKIP", "Done", "EXIT", "  skip  ", "QUIT"])
    def test_is_skip_command_case_insensitive(self, agent, command):
        """Test skip commands are case insensitive."""
        assert agent._is_skip_command(command) is True

    def test_is_skip_command_not_recognized(self, agent):
        """Test non-skip responses are not recognized."""
        assert agent._is_skip_command("option 1") is False
        assert agent._is_skip_command("technical deep dive") is False

    def test_is_skip_command_whitespace_handling(self, agent):
        """Test whitespace is stripped."""
        assert agent._is_skip_command("  skip  ") is True
        # "\n\n" strips to "", which is a skip command (empty string)
        assert agent._is_skip_command("\n\n") is True
        assert agent._is_skip_command("some text") is False


class TestClarifierAgentFallback:
    """Tests for fallback clarification."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_get_fallback_clarification(self, agent):
        """Test fallback clarification returns valid JSON."""
        result = agent._get_fallback_clarification()

        # Should be valid JSON
        data = json.loads(result)
        assert data["needs_clarification"] is True
        assert "?" in data["clarification_question"]

    def test_fallback_is_valid_response(self, agent):
        """Test fallback response passes validation."""
        result = agent._get_fallback_clarification()
        response = ClarificationResponse.model_validate_json(result)

        assert response.needs_clarification is True
        assert response.is_valid() is True


class TestClarifierAgentRun:
    """Tests for the run method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_run_immediate_completion(self, mock_llm_provider, mock_llm):
        """Test run when LLM immediately returns complete."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert isinstance(result, ClarifierResult)

    @pytest.mark.asyncio
    async def test_run_with_skip_command(self, mock_llm_provider, mock_llm):
        """Test run when user skips clarification."""
        clarification_response = ClarificationResponse(needs_clarification=True, clarification_question="What scope?")
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=clarification_response.model_dump_json()))

        mock_user_callback = AsyncMock(return_value="skip")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        mock_user_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_max_turns_reached(self, mock_llm_provider, mock_llm):
        """Test run when max turns is 0."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
            max_turns=0,
        )

        state = ClarifierAgentState(
            messages=[HumanMessage(content="Research AI")],
            max_turns=0,
        )
        result = await agent.run(state)

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_logs_query(self, mock_llm_provider, mock_llm, caplog):
        """Test that run logs the query."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Test query")])

        with caplog.at_level("INFO"):
            await agent.run(state)

        assert "Clarifier: Starting" in caplog.text


class TestHasToolInvocations:
    """Tests for the _has_tool_invocations helper."""

    def test_empty_messages(self):
        """No messages -> no invocations."""
        assert ClarifierAgent._has_tool_invocations([]) is False

    def test_only_human_messages(self):
        """Human-only history has no invocations."""
        messages = [HumanMessage(content="hi"), HumanMessage(content="more")]
        assert ClarifierAgent._has_tool_invocations(messages) is False

    def test_ai_message_without_tool_calls(self):
        """An AIMessage without tool_calls counts as no invocation."""
        messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert ClarifierAgent._has_tool_invocations(messages) is False

    def test_ai_message_with_tool_calls(self):
        """An AIMessage with tool_calls counts as an invocation."""
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "x"}, "id": "call_1"}],
        )
        assert ClarifierAgent._has_tool_invocations([HumanMessage(content="hi"), ai]) is True

    def test_with_tool_message(self):
        """A ToolMessage by itself does not count - we look at the assistant turn."""
        msg = ToolMessage(content="result", tool_call_id="call_1")
        assert ClarifierAgent._has_tool_invocations([msg]) is False


class TestSearchedSinceLastUserTurn:
    """Tests for _searched_since_last_user_turn (current-request scoping)."""

    def test_no_messages(self):
        """An empty history has no search this turn."""
        assert ClarifierAgent._searched_since_last_user_turn([]) is False

    def test_only_current_query_no_search(self):
        """Just the user query, no tool calls yet -> not searched."""
        msgs = [HumanMessage(content="Research AI")]
        assert ClarifierAgent._searched_since_last_user_turn(msgs) is False

    def test_tool_call_after_query_counts(self):
        """A tool call after the latest user turn counts as searched."""
        ai = AIMessage(content="", tool_calls=[{"name": "web_search_tool", "args": {}, "id": "c1"}])
        msgs = [HumanMessage(content="Research AI"), ai]
        assert ClarifierAgent._searched_since_last_user_turn(msgs) is True

    def test_tool_call_before_latest_user_turn_is_ignored(self):
        """A tool call from an earlier turn must not suppress the nudge for a new query."""
        prior_tool = AIMessage(content="", tool_calls=[{"name": "web_search_tool", "args": {}, "id": "c0"}])
        msgs = [
            HumanMessage(content="earlier question"),
            prior_tool,
            ToolMessage(content="result", tool_call_id="c0"),
            HumanMessage(content="a fresh research query"),  # latest user turn, no search after it
        ]
        assert ClarifierAgent._searched_since_last_user_turn(msgs) is False


class TestClarifierForceSearch:
    """Tests for the search-before-clarify behavior (issue #234)."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_force_search_fires_when_llm_skips_tools(self, mock_llm_provider, mock_llm):
        """When tools are configured and the LLM tries to clarify without searching,
        the agent must first nudge the LLM with a force-search guidance message,
        then route to tools when the LLM complies."""

        # 1st LLM call: skip tools, ask for clarification.
        # 2nd LLM call (after force_search): produce a tool call.
        # 3rd LLM call (after tool result): return complete.
        clarif_msg = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "AI"}, "id": "call_1"}],
        )
        complete_msg = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif_msg, tool_call_msg, complete_msg])

        user_callback = AsyncMock()

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research Foo Project XYZ")])
        result = await agent.run(state)

        assert result is not None
        assert isinstance(result, ClarifierResult)
        # The LLM was invoked three times: clarify-attempt, tool-call, finalize.
        assert mock_llm.ainvoke.call_count == 3
        # The user was never prompted because the search-then-complete path was taken.
        user_callback.assert_not_called()
        # The 2nd LLM call (after force_search guidance) must carry the guidance
        # folded into a single LEADING SystemMessage — providers that only accept
        # a leading system message reject a trailing one.
        second_call_messages = mock_llm.ainvoke.call_args_list[1].args[0]
        system_messages = [m for m in second_call_messages if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1, "retry must contain exactly one system message"
        assert isinstance(second_call_messages[0], SystemMessage), "system message must lead the list"
        assert FORCE_SEARCH_GUIDANCE in second_call_messages[0].content

    @pytest.mark.asyncio
    async def test_force_search_skipped_when_no_tools(self, mock_llm_provider, mock_llm):
        """When no tools are configured, force_search must NOT fire; the agent
        should fall back to asking the user immediately."""
        clarif_msg = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        complete_msg = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif_msg, complete_msg])

        user_callback = AsyncMock(return_value="technical deep dive")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[],  # no tools available
            user_prompt_callback=user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        # User callback is called once for clarification (no force_search detour).
        user_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_search_fires_at_most_once(self, mock_llm_provider, mock_llm):
        """Even if the LLM stubbornly refuses to call a tool after the force_search
        nudge, the agent must not loop forever - it should proceed to asking the
        user after the single nudge attempt."""
        clarif_msg_1 = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        # After the force_search nudge, the model still refuses to call a tool
        # and returns another clarification request.
        clarif_msg_2 = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        # After the user replies, the model completes.
        complete_msg = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif_msg_1, clarif_msg_2, complete_msg])

        user_callback = AsyncMock(return_value="technical")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        # The LLM was invoked three times max - the nudge fired once, then we
        # fell through to ask_for_clarification, and the user reply produced
        # the final completion.
        assert mock_llm.ainvoke.call_count == 3
        # The user was prompted exactly once (no infinite loop of nudges).
        user_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_search_not_triggered_when_llm_searches_first(self, mock_llm_provider, mock_llm):
        """When the LLM voluntarily issues a tool call on the first turn, the
        force_search path is never entered - normal flow is preserved."""
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "AI"}, "id": "call_1"}],
        )
        complete_msg = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_msg, complete_msg])

        user_callback = AsyncMock()

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert mock_llm.ainvoke.call_count == 2
        user_callback.assert_not_called()
        # The guidance must not appear in ANY message of ANY call — the model
        # searched voluntarily, so the nudge path must never have fired.
        for call in mock_llm.ainvoke.call_args_list:
            assert not any(FORCE_SEARCH_GUIDANCE in str(m.content) for m in call.args[0])

    @pytest.mark.asyncio
    async def test_tool_node_uses_relay_middleware(self, mock_llm_provider, mock_llm, monkeypatch):
        """Application-owned ToolNode executions pass through Relay's maintained hook."""
        observed_tool_names: list[str] = []

        async def observe_tool_call(self, request, handler):
            observed_tool_names.append(request.tool_call["name"])
            return await handler(request)

        monkeypatch.setattr(NemoRelayMiddleware, "awrap_tool_call", observe_tool_call)
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "web_search_tool", "args": {"query": "AI"}, "id": "call_1"}],
                ),
                AIMessage(
                    content=ClarificationResponse(
                        needs_clarification=False,
                        clarification_question=None,
                    ).model_dump_json()
                ),
            ]
        )
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=AsyncMock(),
        )

        await agent.run(ClarifierAgentState(messages=[HumanMessage(content="Research AI")]))

        assert observed_tool_names == ["web_search_tool"]

    @pytest.mark.asyncio
    async def test_force_search_guidance_not_in_state_messages(self, mock_llm_provider, mock_llm):
        """The force_search guidance must be injected ephemerally only; it must
        never end up in state.messages, otherwise helpers like
        get_latest_user_query would surface internal scaffolding back to the
        user in fallback text."""
        # The LLM ignores the nudge and returns invalid JSON, triggering the
        # invalid-format fallback path inside ask_clarification. The user then
        # replies "skip", which forces completion - so only two LLM calls
        # actually happen in this run.
        clarif_msg_1 = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        clarif_invalid = AIMessage(content="not valid JSON at all")
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif_msg_1, clarif_invalid])

        # Capture what gets sent to the user.
        prompts_received: list[str] = []

        async def user_callback(question: str) -> str:
            """Return the canned user reply for this test."""
            prompts_received.append(question)
            return "skip"

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=user_callback,
        )

        original_query = "Research Project Foo at Acme"
        state = ClarifierAgentState(messages=[HumanMessage(content=original_query)])
        final = await agent.graph.ainvoke(state, config={"callbacks": []})

        # The guidance must never be persisted into state.messages.
        final_messages = final["messages"] if isinstance(final, dict) else final.messages
        assert not any(FORCE_SEARCH_GUIDANCE in str(m.content) for m in final_messages), (
            "force_search guidance leaked into persisted state"
        )
        # The user was prompted exactly once - with a fallback derived from
        # their actual query (the full topic survives), never from the
        # force-search guidance.
        assert len(prompts_received) == 1
        prompt_text = prompts_received[0]
        assert "Project Foo" in prompt_text and "Acme" in prompt_text
        assert FORCE_SEARCH_GUIDANCE not in prompt_text
        # The internal force-search guidance must never be visible in any
        # message the user-facing fallback would draw from.
        assert "You attempted to ask the user" not in prompt_text

    @pytest.mark.asyncio
    async def test_force_search_guidance_not_injected_after_user_reply(self, mock_llm_provider, mock_llm):
        """After the user has actually replied (iteration > 0), the agent must
        NOT re-inject the search-first nudge on the next LLM call. Otherwise
        the model would receive 'issue a tool call now' immediately after the
        user provided clarifying answer, causing a gratuitous search instead
        of synthesizing the answer."""
        clarif_msg_1 = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        # After the nudge, the model still refuses to call a tool.
        clarif_msg_2 = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="Which area?"
            ).model_dump_json()
        )
        # After the user replies, the model completes.
        complete_msg = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif_msg_1, clarif_msg_2, complete_msg])

        user_callback = AsyncMock(return_value="technical deep dive")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        await agent.run(state)

        # The 3rd LLM call happens AFTER the user reply (iteration moves from
        # 0 to 1), so the inline search-before-clarify guard (gated on
        # iteration == 0) must not fire again and the nudge must not appear in
        # that call's message list.
        assert mock_llm.ainvoke.call_count == 3
        third_call_messages = mock_llm.ainvoke.call_args_list[2].args[0]
        assert not any(FORCE_SEARCH_GUIDANCE in str(m.content) for m in third_call_messages), (
            "force_search guidance must not be re-injected after the user replies"
        )

    @pytest.mark.asyncio
    async def test_forced_retry_does_not_emit_consecutive_assistant_messages(self, mock_llm_provider, mock_llm):
        """After a forced search-retry whose retry produces a tool call, the
        message list fed to the LLM on the next turn must NOT contain two
        consecutive assistant (AIMessage) turns. Two adjacent assistant
        messages with no interleaved user/tool message are rejected with a 400
        by the OpenAI Chat Completions and Anthropic Messages APIs; mocked LLMs
        don't enforce this, so we assert the invariant explicitly."""
        clarif_msg = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "AI"}, "id": "call_1"}],
        )
        complete_msg = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif_msg, tool_call_msg, complete_msg])

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=AsyncMock(),
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research Foo Project XYZ")])
        await agent.run(state)

        # Inspect every message list that was actually sent to the LLM and assert
        # no two consecutive AIMessages appear (the API-invalid shape).
        for call_idx, call in enumerate(mock_llm.ainvoke.call_args_list):
            sent_messages = call.args[0]
            offenders = _adjacent_assistant_pairs(sent_messages)
            assert not offenders, (
                f"LLM call #{call_idx} contained consecutive assistant messages at {offenders}; "
                "this is rejected by OpenAI/Anthropic APIs"
            )

        # Specifically: the forced retry must not persist the skipped
        # clarification, so the post-tool history is [..., AIMessage(tool_call),
        # ToolMessage, ...] with no stale AIMessage before the tool call.
        third_call_messages = mock_llm.ainvoke.call_args_list[2].args[0]
        ai_then_tool = any(
            isinstance(third_call_messages[i], AIMessage)
            and getattr(third_call_messages[i], "tool_calls", None)
            and isinstance(third_call_messages[i + 1], ToolMessage)
            for i in range(len(third_call_messages) - 1)
        )
        assert ai_then_tool, "expected a tool-call AIMessage immediately followed by its ToolMessage in history"


class TestClarifierSkipMessageOrdering:
    """Skip-command branch must not produce consecutive assistant messages.

    Regression tests for the ordering bug surfaced during the PR #245 audit:
    the skip branch returned an AIMessage(complete) without persisting the
    user's reply, leaving it adjacent to the prior clarification AIMessage; the
    graph then re-entered agent_node and (with the budget exhausted) appended a
    third AIMessage. Two/three consecutive assistant turns are rejected by the
    OpenAI/Anthropic APIs and corrupt the planner call when plan approval is on.
    """

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider returning the mock LLM."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_skip_persists_reply_and_no_consecutive_assistants(self, mock_llm_provider, mock_llm):
        """User skips after one clarification: final history must interleave the
        skip reply (HumanMessage) and contain no consecutive AIMessages."""
        clarif = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        # Only one real LLM response is needed; after skip the graph completes
        # without another model call (the early-complete guard returns {}).
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif])

        async def user_callback(question: str) -> str:
            """Always skip the clarification."""
            return "skip"

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[],  # no tools → no forced search; isolate the skip path
            user_prompt_callback=user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])

        # Inspect persisted message ordering from the final graph state.
        final = await agent.graph.ainvoke(state, config={"callbacks": []})
        msgs = final["messages"] if isinstance(final, dict) else final.messages
        offenders = _adjacent_assistant_pairs(msgs)
        assert not offenders, f"persisted history has consecutive assistant messages at {offenders}: {msgs}"
        # The skip reply must be persisted as a HumanMessage.
        assert any(isinstance(m, HumanMessage) and m.content == "skip" for m in msgs), (
            "skip reply was not persisted as a HumanMessage"
        )
        # Exactly one *terminal completion* AIMessage should end the dialog: a
        # message that parses to needs_clarification=false. Matching any non-tool
        # AIMessage (e.g. the earlier clarification prompt) or allowing >= 1 would
        # not catch a missing or duplicated completion.
        completion_ais = [
            m
            for m in msgs
            if isinstance(m, AIMessage)
            and not ClarifierAgent._has_tool_invocations([m])
            and agent._is_complete(m.content)
        ]
        got = len(completion_ais)
        assert got == 1, f"expected exactly one terminal completion AIMessage, got {got}"


class TestClarifierReviewRegressions:
    """Regression tests for edge cases surfaced in deep review of #245."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider returning the mock LLM."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_force_search_fires_despite_prior_turn_tool_calls(self, mock_llm_provider, mock_llm):
        """A tool call from an earlier conversation turn must not suppress the
        search-before-clarify nudge for a fresh user query. The guard is scoped
        to tool activity since the latest user turn."""
        # First model call (for the new query): clarify without a tool call.
        # Second call (after the nudge): emit a tool call. Third: complete.
        clarif = AIMessage(
            content=ClarificationResponse(
                needs_clarification=True, clarification_question="What aspect?"
            ).model_dump_json()
        )
        tool_call = AIMessage(content="", tool_calls=[{"name": "web_search_tool", "args": {"query": "x"}, "id": "c1"}])
        complete = AIMessage(
            content=ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[clarif, tool_call, complete])

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=AsyncMock(),
        )

        # History carries a tool call from an EARLIER turn, then the fresh query.
        prior_tool = AIMessage(content="", tool_calls=[{"name": "web_search_tool", "args": {}, "id": "c0"}])
        state = ClarifierAgentState(
            messages=[
                HumanMessage(content="an earlier question"),
                prior_tool,
                ToolMessage(content="old result", tool_call_id="c0"),
                HumanMessage(content="a fresh research query"),
            ]
        )
        await agent.run(state)

        # The nudge must still have fired: 3 model calls, and the 2nd received
        # the FORCE_SEARCH_GUIDANCE (it is not suppressed by the prior tool call).
        assert mock_llm.ainvoke.call_count == 3
        second_call_messages = mock_llm.ainvoke.call_args_list[1].args[0]
        assert any(isinstance(m, SystemMessage) and FORCE_SEARCH_GUIDANCE in m.content for m in second_call_messages)

    @pytest.mark.asyncio
    async def test_exhausted_entry_with_pending_tool_call_is_not_completed_directly(self, mock_llm_provider, mock_llm):
        """At exhaustion, if the last message is an AIMessage with pending tool
        calls, agent_node must not stack a completion on top of it (which would
        leave the tool call unresolved / produce invalid history). It returns no
        new message so the graph can route the tool call to the tools node."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=AsyncMock(),
        )

        pending_tool = AIMessage(
            content="", tool_calls=[{"name": "web_search_tool", "args": {"query": "x"}, "id": "c1"}]
        )
        state = ClarifierAgentState(
            messages=[HumanMessage(content="Research AI"), pending_tool],
            max_turns=0,
        )

        # Run only the agent node against this state.
        result = await agent.graph.nodes["agent"].ainvoke(state)
        assert isinstance(result, dict), f"agent node must return a state-update dict, got {type(result)}"
        new_messages = result.get("messages", [])
        # No completion AIMessage may be appended on top of the pending tool call.
        assert not any(isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) for m in new_messages), (
            "must not append a completion on top of a pending tool call"
        )
