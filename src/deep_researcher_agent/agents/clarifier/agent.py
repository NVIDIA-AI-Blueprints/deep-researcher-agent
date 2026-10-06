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

"""
Clarifier agent for interactive clarification dialog.

This module provides the ClarifierAgent class which handles multi-turn
clarification dialogs with users before deep research begins. The agent
uses LangGraph for workflow orchestration and supports tool calling
for context gathering.

Example:
    >>> from deep_researcher_agent.agents.clarifier_agent import ClarifierAgent
    >>> from deep_researcher_agent.common import LLMProvider
    >>>
    >>> async def prompt_user(question: str) -> str:
    ...     return input(question)
    >>>
    >>> provider = LLMProvider()
    >>> provider.set_default(my_llm)
    >>> agent = ClarifierAgent(
    ...     llm_provider=provider,
    ...     user_prompt_callback=prompt_user,
    ... )
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from nemo_relay.integrations.langchain import NemoRelayMiddleware

from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole
from deep_researcher_agent.common import format_data_source_tools
from deep_researcher_agent.common import get_latest_user_query
from deep_researcher_agent.common import load_prompt
from deep_researcher_agent.common import render_prompt_template
from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.relay import ainvoke_with_relay
from deep_researcher_agent.relay import run_agent

from .models import ClarificationResponse
from .models import ClarifierAgentState
from .models import ClarifierResult

logger = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).parent
"""Path to the clarifier agent's directory, used for loading prompts."""

DEFAULT_CLARIFICATION_PROMPT = (
    "/no_think\n\n"
    "You are a helpful research clarification assistant. "
    "Ask focused questions to understand the user's needs. "
    'Respond with JSON: {"needs_clarification": true/false, "clarification_question": "your question?" or null}'
)
"""Fallback prompt used when the prompt file cannot be loaded."""

JSON_REMINDER_AFTER_TOOLS = (
    "Based on the search results above, now make your clarification decision. "
    "IMPORTANT: You must respond with ONLY a valid JSON object, nothing else. "
    "Do NOT write a report, summary, or analysis. "
    "Output exactly: "
    '{"needs_clarification": true, "clarification_question": "your question"} '
    "OR "
    '{"needs_clarification": false, "clarification_question": null}'
)
"""Reminder prompt added after tool results to reinforce JSON-only output."""

FORCE_SEARCH_GUIDANCE = (
    "You attempted to ask the user for clarification before gathering any context. "
    "Before asking the user a question, you MUST first use the available search tools "
    "to look up unfamiliar entities, acronyms, products, or terms in their request. "
    "Issue one focused tool call now with a query derived from the user's request. "
    "Only after reviewing the tool results should you decide whether clarification is still needed."
)
"""Guidance prompt injected when the LLM tries to clarify without having searched first."""

SKIPPED_CLARIFICATION_SENTINEL = "[skipped clarification]"
"""Stand-in user turn persisted when a skip reply is blank, so an empty
HumanMessage is never written to state (some chat APIs reject empty content)."""


class ClarifierAgent:
    """
    Clarifier agent for interactive clarification dialog.

    This agent handles interactive clarification dialogs for deep research queries.
    It asks follow-up questions to refine the research scope, constraints, and
    requirements before the actual research begins.

    The agent uses LangGraph for workflow orchestration with three main nodes:
    - agent: Generates clarification questions using the LLM
    - tools: Executes tool calls for context gathering (e.g., web search)
    - ask_for_clarification: Prompts the user and processes their response

    It gathers context and, when the request is vague, may clarify the scope or
    the type of output the user wants (e.g. report, table, comparison, prediction)
    before research begins. It does not produce or approve a research plan.

    Attributes:
        llm_provider: Provider for obtaining LLM instances.
        tools: List of tools available for context gathering.
        user_prompt_callback: Async callback for prompting user input.
        max_turns: Maximum number of Q&A turns before auto-completing.
        system_prompt: The loaded system prompt for the LLM.
        callbacks: LangChain callbacks for tracing/logging.

    Example:
        >>> async def user_prompt_fn(question: str) -> str:
        ...     return input(question)
        >>>
        >>> provider = LLMProvider()
        >>> provider.set_default(my_llm)
        >>> agent = ClarifierAgent(
        ...     llm_provider=provider,
        ...     tools=[search_tool],
        ...     user_prompt_callback=user_prompt_fn,
        ...     max_turns=3,
        ... )
        >>> state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        >>> result = await agent.run(state)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        user_prompt_callback: Callable[[str], Awaitable[str]],
        max_turns: int = 3,
        log_response_max_chars: int = 2000,
        callbacks: list[Any] | None = None,
    ) -> None:
        """
        Initialize the clarifier agent.

        Args:
            llm_provider: Provider for obtaining LLM instances by role.
            tools: Optional sequence of LangChain tools for context gathering
                (e.g., web search). Tools help the agent ask more informed questions.
            user_prompt_callback: Async callback function to prompt the user for input.
                Takes a question string and returns the user's response string.
            max_turns: Maximum number of clarification Q&A turns before
                automatically completing clarification. Defaults to 3.
            log_response_max_chars: Maximum characters to log from LLM responses.
                Used for debugging. Defaults to 2000.
            callbacks: Optional list of LangChain callback handlers for
                tracing and logging.
        """
        self.llm_provider: LLMProvider = llm_provider
        self.tools = list(tools) if tools else []
        self.user_prompt_callback = user_prompt_callback
        self.max_turns = max_turns
        self.log_response_max_chars = log_response_max_chars
        self.callbacks = callbacks or []

        self.system_prompt = self._load_default_prompt()

        self._graph = self._build_graph()

    def _load_default_prompt(self) -> str:
        """
        Load the research clarification prompt from file.

        Attempts to load the prompt from the prompts/research_clarification.j2
        file. Falls back to DEFAULT_CLARIFICATION_PROMPT if the file is not found.

        Returns:
            The loaded prompt string, or the default fallback prompt.
        """
        try:
            return load_prompt(AGENT_DIR / "prompts", "research_clarification")
        except Exception:
            logger.warning("Clarifier prompt not found, using inline default")
            return DEFAULT_CLARIFICATION_PROMPT

    def _parse_response(self, text: str) -> ClarificationResponse | None:
        """
        Parse JSON response from LLM into ClarificationResponse.

        Attempts multiple strategies to extract JSON:
        1. Parse the entire text as JSON
        2. Extract from markdown code blocks
        3. Find JSON object pattern anywhere in text

        Args:
            text: Raw text response from LLM.

        Returns:
            ClarificationResponse if parsing succeeds, None otherwise.
        """
        if not text:
            return None

        text = text.strip()

        # Strategy 1: Try parsing the entire text as JSON
        try:
            data = json.loads(text)
            return ClarificationResponse.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Strategy 2: Extract from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                return ClarificationResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Strategy 3: Find JSON object pattern anywhere in text
        # Look for {"needs_clarification": ...} pattern
        json_pattern = re.search(r'\{[^{}]*"needs_clarification"[^{}]*\}', text)
        if json_pattern:
            try:
                data = json.loads(json_pattern.group(0))
                return ClarificationResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Strategy 4: Find any JSON object (more permissive)
        brace_match = re.search(r"\{[\s\S]*?\}", text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return ClarificationResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        logger.warning("Failed to parse clarification response as JSON (response_%s)", log_content_metadata(text))
        return None

    def _is_needed(self, text: str) -> bool:
        """
        Check if clarification is needed based on JSON response.

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            True if clarification is needed or parsing failed, False otherwise.
        """
        response = self._parse_response(text)
        if response is None:
            logger.warning("Failed to parse response, assuming clarification needed")
            return True
        return response.needs_clarification

    def _is_complete(self, text: str) -> bool:
        """
        Check if clarification is complete based on JSON response.

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            True if clarification is complete (needs_clarification=false),
            False otherwise or if parsing failed.
        """
        response = self._parse_response(text)
        if response is None:
            return False
        return response.is_complete()

    def _valid_needed(self, text: str) -> bool:
        """
        Check if the clarification response is valid.

        A response is valid if:
        - It parses successfully as JSON
        - When needs_clarification is true, it contains a clarification question

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            True if the response is valid, False otherwise.
        """
        response = self._parse_response(text)
        if response is None:
            return False
        return response.is_valid()

    def _get_clarification_question(self, text: str) -> str:
        """
        Extract the clarification question from the response.

        Args:
            text: Raw text response from LLM.

        Returns:
            The clarification question text.
        """
        response = self._parse_response(text)
        if response is not None and response.clarification_question:
            return response.clarification_question
        logger.warning("No clarification question found in response")
        return "Could you provide more details about your research needs?"

    def _get_llm(self) -> BaseChatModel:
        """
        Get the LLM instance for the clarifier agent.

        Uses LLMRole.CLARIFIER to obtain the appropriate LLM from the provider.

        Returns:
            The LangChain LLM instance for generating clarification questions.
        """
        return self.llm_provider.get(LLMRole.CLARIFIER)

    def _get_fallback_clarification(self, query: str | None = None) -> str:
        """
        Get fallback clarification text when the LLM response is invalid.

        Returns a topic-aware clarification question when query is provided,
        otherwise falls back to a generic question.

        Args:
            query: Optional user query to make the fallback more relevant.

        Returns:
            JSON string representing a ClarificationResponse with a fallback question.
        """
        if query:
            # Create topic-aware fallback
            topic_snippet = query[:80].strip()
            if len(query) > 80:
                topic_snippet += "..."
            question = (
                f'To help with your research on: "{topic_snippet}"\n\n'
                "Could you specify:\n"
                "1. Which specific aspects are most important to you?\n"
                "2. What level of detail do you need?\n"
                "3. Or type 'skip' to proceed with a general approach."
            )
        else:
            question = (
                "I'd like to help with your research. Could you provide more details about:\n\n"
                "1. What specific aspects interest you most?\n"
                "2. Who is this report for?\n"
                "3. How detailed should it be?"
            )

        fallback = ClarificationResponse(
            needs_clarification=True,
            clarification_question=question,
        )
        return fallback.model_dump_json()

    SKIP_COMMANDS = {"skip", "done", "exit", "quit", "proceed", "continue", "no", "n", ""}
    """Set of commands that indicate the user wants to skip clarification."""

    @staticmethod
    def _has_tool_invocations(messages: Sequence[Any]) -> bool:
        """
        Check whether any prior assistant message in the conversation issued tool calls.

        Args:
            messages: The conversation message history.

        Returns:
            True if any AIMessage in the history carries non-empty tool_calls,
            False otherwise.
        """
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                return True
        return False

    @staticmethod
    def _searched_since_last_user_turn(messages: Sequence[Any]) -> bool:
        """Check whether a tool call has occurred since the latest user message.

        Scopes the search-before-clarify guard to the *current* request: tool
        calls from earlier conversation turns must not suppress the nudge for a
        fresh user query.

        Args:
            messages: The conversation message history.

        Returns:
            True if any message after the most recent HumanMessage carries tool
            calls, False otherwise.
        """
        last_user_idx = -1
        for idx, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_user_idx = idx
        return ClarifierAgent._has_tool_invocations(messages[last_user_idx + 1 :])

    def _is_skip_command(self, user_reply: str) -> bool:
        """
        Check if the user's reply indicates they want to skip clarification.

        Recognized skip commands: skip, done, exit, quit, proceed, continue, no, n,
        or empty string.

        Args:
            user_reply: The user's response text.

        Returns:
            True if the reply is a skip command, False otherwise.
        """
        return user_reply.strip().lower() in self.SKIP_COMMANDS

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph StateGraph for the clarification workflow.

        Creates a graph with the following nodes:
        - agent: Generates clarification questions using the LLM. On the first
          turn it also enforces search-before-clarify (issue #234): if the model
          asks for clarification without using its bound search tools, it nudges
          the model once and retries inline.
        - tools: Executes tool calls (e.g., web search) for context
        - ask_for_clarification: Prompts user and processes response

        The graph flow:
        1. agent generates a response (question, tool call, or completion);
           on turn 0 it may force one search-and-retry before yielding
        2. If tool call → tools node → back to agent
        3. If complete → end
        4. Otherwise → ask_for_clarification → back to agent

        Returns:
            Compiled LangGraph StateGraph ready for execution.
        """
        llm = self._get_llm()
        bound_llm = llm.bind_tools(self.tools, parallel_tool_calls=True) if self.tools else llm

        graph = StateGraph(ClarifierAgentState)

        async def agent_node(state: ClarifierAgentState):
            """Run the LLM for one turn and apply the search-before-clarify nudge.

            Emits a completion when the clarification budget is exhausted, and on
            the first turn forces one search-and-retry if the model tries to ask
            for clarification without using its bound tools.
            """
            if state.remaining_questions <= 0:
                # Clarification budget is exhausted — emit a completion signal,
                # but never create an invalid history (two adjacent assistant
                # messages, or a pending tool call left unresolved), since the
                # message list is replayed to the LLM on later turns.
                last_message = state.messages[-1] if state.messages else None
                if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                    # A pending tool call must be resolved before we complete;
                    # let decide_route route to the tools node, after which the
                    # tool result re-enters here as the last turn.
                    return {}
                if isinstance(last_message, AIMessage) and self._is_complete(getattr(last_message, "content", "")):
                    # A prior node (e.g. the skip-command branch) already emitted
                    # the completion; don't duplicate it. Let decide_route end.
                    return {}
                complete = AIMessage(
                    content=ClarificationResponse(
                        needs_clarification=False, clarification_question=None
                    ).model_dump_json()
                )
                if isinstance(last_message, AIMessage):
                    # The last turn is a non-complete assistant message (e.g. an
                    # unanswered clarification at exhaustion). Interleave a
                    # sentinel user turn so the completion is not adjacent to it.
                    return {"messages": [HumanMessage(content=SKIPPED_CLARIFICATION_SENTINEL), complete]}
                return {"messages": [complete]}
            tools_info = [
                {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")} for t in self.tools
            ]
            # Selected data sources (e.g. Google Drive) become tools in the research
            # phase but are NOT bound to the clarifier, so surface them explicitly —
            # otherwise the LLM refuses Drive/URL requests it thinks it can't access.
            connected_sources = format_data_source_tools(state.data_sources) if state.data_sources else []
            rendered_system_prompt = render_prompt_template(
                self.system_prompt,
                clarifier_result=state.clarifier_log,
                available_documents=state.available_documents or [],
                connected_sources=connected_sources,
                tools=tools_info,
                tool_names=[t["name"] for t in tools_info],
            )

            # Build message list
            messages = [SystemMessage(content=rendered_system_prompt)] + state.messages

            # If last message is a tool result, add JSON reminder to prevent report generation
            if state.messages and isinstance(state.messages[-1], ToolMessage):
                logger.info("Adding JSON reminder after tool results")
                messages.append(HumanMessage(content=JSON_REMINDER_AFTER_TOOLS))

            response = await ainvoke_with_relay(bound_llm, messages, callbacks=self.callbacks)

            # Search-before-clarify (issue #234): on the first turn, if the model
            # asks for clarification without searching, nudge it once (guidance as
            # ephemeral scaffolding, never persisted to state) and retry inline.
            # The guard is one-shot: iteration == 0 and no tool call yet for the
            # current request (scoped to the latest user turn). Return only the
            # retry so the skipped first response can't leave two adjacent
            # assistant messages in history. The guidance is folded into the
            # leading system prompt (a trailing SystemMessage is rejected by
            # providers that only accept a leading one).
            if (
                self.tools
                and state.iteration == 0
                and not self._searched_since_last_user_turn(state.messages)
                and not getattr(response, "tool_calls", None)
                and self._is_needed(response.content)
            ):
                logger.info("Clarifier: model skipped search before clarifying; injecting guidance and retrying once")
                retry_system = SystemMessage(content=f"{rendered_system_prompt}\n\n{FORCE_SEARCH_GUIDANCE}")
                retry_messages = [retry_system, *messages[1:], response]
                retry_response = await ainvoke_with_relay(bound_llm, retry_messages, callbacks=self.callbacks)
                return {"messages": [retry_response]}

            return {"messages": [response]}

        async def ask_clarification(state: ClarifierAgentState):
            """Prompt the user with the pending question and record their reply.

            Handles skip commands (substituting a non-empty sentinel for blank
            replies) and the max-turns cutoff, advancing the clarification log.
            """
            iteration = state.iteration
            max_turns = state.max_turns
            clarifier_log = state.clarifier_log
            if iteration >= max_turns:
                return {
                    "clarifier_log": f"Clarification complete: Met the maximum number of turns\n{clarifier_log}",
                }
            text = state.messages[-1].content if state.messages else ""
            if not self._is_needed(text):
                return {}

            if not self._valid_needed(text):
                logger.warning("Invalid clarification format, forcing fallback")
                # Extract latest query for topic-aware fallback
                original_query = get_latest_user_query(state.messages)
                text = self._get_fallback_clarification(query=original_query if original_query else None)

            question_text = self._get_clarification_question(text)
            clarifier_log = f"{clarifier_log}\n**Turn {iteration + 1} - Assistant:**\n{question_text}"
            user_reply = await self.user_prompt_callback(question_text)

            if self._is_skip_command(user_reply):
                logger.info("Clarifier: User requested to skip clarification")
                complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
                clarifier_log = f"{clarifier_log}\n**Turn {iteration + 1} - User:** [Skipped clarification]"
                # Persist the user's reply as a HumanMessage before the
                # completion AIMessage. The prior turn already left an
                # AIMessage(clarification) in history; without an interleaving
                # human message the two assistant turns would be adjacent, which
                # the OpenAI/Anthropic APIs reject. (The duplicate completion on
                # graph re-entry is suppressed by the guard in agent_node.)
                #
                # A blank/whitespace reply also counts as skip (see
                # SKIP_COMMANDS), but an empty HumanMessage must not be persisted
                # -- it would flow into plan generation, and some chat APIs reject
                # empty message content. Substitute a non-empty sentinel.
                skip_reply = user_reply if user_reply.strip() else SKIPPED_CLARIFICATION_SENTINEL
                return {
                    "messages": [
                        HumanMessage(content=skip_reply),
                        AIMessage(content=complete_response.model_dump_json()),
                    ],
                    "iteration": max_turns,  # Force end of clarification
                    "clarifier_log": clarifier_log,
                }

            clarifier_log = f"{clarifier_log}\n**Turn {iteration + 1} - User:**\n{user_reply}"
            return {
                "messages": [HumanMessage(content=user_reply)],
                "iteration": iteration + 1,
                "clarifier_log": clarifier_log,
            }

        def decide_route(state: ClarifierAgentState | dict):
            """Route after agent_node: to tools, plan preview, end, or the user."""
            if isinstance(state, dict):
                messages = state.get("messages", [])
            elif hasattr(state, "messages"):
                messages = state.messages
            else:
                msg = f"No messages found in input state to tool_edge: {state}"
                raise ValueError(msg)

            if not messages:
                msg = f"Empty messages list in state: {state}"
                raise ValueError(msg)

            ai_message = messages[-1]
            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                return "tools"

            if self._is_complete(ai_message.content):
                return "__end__"

            # The search-before-clarify nudge (issue #234) is handled inline in
            # agent_node, not here — see the retry block there. By the time a
            # clarification response reaches this router, any forced search has
            # already happened, so we route straight to the user.
            return "ask_for_clarification"

        graph.add_node("agent", agent_node)
        relay_middleware = NemoRelayMiddleware()
        graph.add_node("tools", ToolNode(self.tools, awrap_tool_call=relay_middleware.awrap_tool_call))
        graph.add_node("ask_for_clarification", ask_clarification)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            decide_route,
            {
                "tools": "tools",
                "ask_for_clarification": "ask_for_clarification",
                "__end__": "__end__",
            },
        )

        graph.add_edge("tools", "agent")
        graph.add_edge("ask_for_clarification", "agent")

        return graph.compile()

    async def run(self, state: ClarifierAgentState) -> ClarifierResult:
        """
        Execute the clarification dialog.

        Args:
            state: Initial state of the clarifier agent.

        Returns:
            ClarifierResult with the clarification log.
        """
        logger.info("Clarifier: Starting (max %d turns)", self.max_turns)
        query = get_latest_user_query(state.messages)
        logger.info("User query: %s", log_content_metadata(query or ""))

        async def _invoke_graph() -> dict[str, Any]:
            return await self._graph.ainvoke(state, config={"callbacks": self.callbacks})

        result = await run_agent("clarifier_agent", _invoke_graph, input_value=state)
        final_state = ClarifierAgentState.model_validate(result)
        return ClarifierResult(clarifier_log=final_state.clarifier_log)

    @property
    def graph(self) -> CompiledStateGraph:
        """Get the compiled LangGraph for direct access."""
        return self._graph
