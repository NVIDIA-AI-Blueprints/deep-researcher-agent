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

"""Shallow research agent for fast, bounded research with tool-calling."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from deep_researcher_agent.common import get_source_id_for_tool
from deep_researcher_agent.common import load_prompt
from deep_researcher_agent.common import render_prompt_template
from deep_researcher_agent.common.callbacks import SUPPRESS_OUTPUT_ARTIFACT_TAG
from deep_researcher_agent.common.citation_verification import CitationIntegrityError
from deep_researcher_agent.common.citation_verification import EmptySourceRegistryError
from deep_researcher_agent.common.citation_verification import SourceEntry
from deep_researcher_agent.common.citation_verification import SourceRegistry
from deep_researcher_agent.common.citation_verification import extract_sources_from_tool_result
from deep_researcher_agent.common.citation_verification import get_session_registry
from deep_researcher_agent.common.citation_verification import sanitize_report
from deep_researcher_agent.common.citation_verification import verify_citations
from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.relay import ainvoke_with_relay
from deep_researcher_agent.relay import run_agent
from deep_researcher_agent.relay.runtime import awrap_tool_call_with_relay

from ...common import LLMProvider
from ...common import LLMRole
from .models import ShallowResearchAgentState

logger = logging.getLogger(__name__)


# Path to this agent's directory (for loading prompts)
AGENT_DIR = Path(__file__).parent

_SOURCE_SECTION_HEADING_RE = re.compile(
    r"^[^\S\n]*(?:"
    r"#{1,6}[^\S\n]+(?:Sources|References):?"
    r"|\*\*(?:Sources|References):?\*\*:?"
    r"|(?:Sources|References):?"
    r")[^\S\n]*$",
    re.IGNORECASE,
)
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_REFERENCE_ENTRY_LINE_RE = re.compile(r"^[^\S\n]*(?P<bullet>[-*][^\S\n]*)?\[(?P<number>\d+)\][^\S\n]+(?P<target>.+)$")
_REFERENCE_TARGET_RE = re.compile(
    r"(?:https?://\S+|[^\s,]+\.\w{2,5}(?:,[^\n]+)?|[A-Za-z0-9]+(?:_+[A-Za-z0-9]+)+)$",
    re.IGNORECASE,
)


def _reference_entry_number(line: str, previous_number: int | None) -> int | None:
    """Return a reference number only for an unambiguous definition line."""
    match = _REFERENCE_ENTRY_LINE_RE.fullmatch(line)
    if match is None:
        return None

    number = int(match.group("number"))
    if previous_number is not None and number <= previous_number:
        # Reference definitions are unique and ordered. A repeated/reset
        # marker begins answer prose, even when it immediately follows them.
        return None

    target = match.group("target").strip()
    if match.group("bullet") is None and _REFERENCE_TARGET_RE.search(target) is None:
        # An unbulleted marker-first sentence is answer text. Verified
        # unbulleted definitions carry a URL, file/page key, or tool key.
        return None
    return number


def _source_section_spans(report_text: str) -> list[tuple[int, int]]:
    """Locate canonical source headings with definitions, or empty trailing headings."""
    lines = report_text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    line_index = 0
    while line_index < len(lines):
        heading = lines[line_index].rstrip("\r\n")
        if _SOURCE_SECTION_HEADING_RE.fullmatch(heading) is None:
            line_index += 1
            continue

        first_definition = line_index + 1
        while first_definition < len(lines) and not lines[first_definition].strip():
            first_definition += 1

        after_definitions = first_definition
        previous_number: int | None = None
        while after_definitions < len(lines):
            number = _reference_entry_number(lines[after_definitions].rstrip("\r\n"), previous_number)
            if number is None:
                break
            previous_number = number
            after_definitions += 1

        if after_definitions > first_definition:
            # Blank lines after a definition block are section separators, not
            # answer content. Consume them so preserved prose keeps its spacing.
            next_content = after_definitions
            while next_content < len(lines) and not lines[next_content].strip():
                next_content += 1
            end = offsets[next_content] if next_content < len(lines) else len(report_text)
            spans.append((offsets[line_index], end))
            line_index = next_content
        elif first_definition == len(lines):
            spans.append((offsets[line_index], len(report_text)))
            break
        else:
            # An exact heading followed by prose is answer content, not a
            # reference section, and must not be removed.
            line_index += 1

    return spans


def _remove_source_sections(report_text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Remove only recognized source-section spans, preserving surrounding prose."""
    if not spans:
        return report_text

    pieces: list[str] = []
    previous_end = 0
    for start, end in spans:
        pieces.append(report_text[previous_end:start])
        previous_end = end
    pieces.append(report_text[previous_end:])
    return "".join(pieces)


_TRAILING_REFERENCE_HEADING_RE = re.compile(
    r"\n{1,2}(?:#{1,3}\s+)?(?:References|Sources):?\s*$|\n{1,2}\*\*(?:References|Sources):?\*\*\s*$",
    re.IGNORECASE,
)
_SOURCE_HEADING_RE = re.compile(r"^## Sources\s*$", re.IGNORECASE | re.MULTILINE)


def _format_chat_references(report_text: str) -> str:
    content = _TRAILING_REFERENCE_HEADING_RE.sub("", report_text.rstrip()).rstrip()
    return _SOURCE_HEADING_RE.sub("**References:**", content, count=1)


def _append_minimal_citation(report_text: str, source: SourceEntry) -> str:
    """Append one verified citation when the model omitted references."""
    citation_target = source.url or source.citation_key
    if not citation_target:
        return report_text

    # Replace only canonical source sections. Similar-looking answer headings
    # and prose (for example, "Sources of renewable energy") are content.
    content = _remove_source_sections(report_text, _source_section_spans(report_text)).rstrip()
    if content.endswith((".", "!", "?")):
        content = f"{content[:-1]} [1]{content[-1]}"
    else:
        content = f"{content} [1]"

    if source.url:
        title = source.title or source.url
        reference = f"- [1] {title} - {source.url}"
    else:
        reference = f"- [1] {citation_target}"

    return f"{content}\n\n**References:**\n{reference}"


def _has_citation_integrity(report_text: str, valid_citations: Sequence[dict[str, Any]]) -> bool:
    """Return whether a verified report has both a source and an inline marker."""
    valid_numbers = {
        int(number)
        for citation in valid_citations
        if (number := citation.get("number")) is not None and str(number).isdigit()
    }
    if not valid_numbers:
        return False

    source_sections = _source_section_spans(report_text)
    if not source_sections:
        return False
    # Definition labels are not inline citations. Remove every recognized
    # source block so a later duplicate block cannot satisfy the invariant.
    prose = _remove_source_sections(report_text, source_sections)
    return any(int(number) in valid_numbers for number in _INLINE_CITATION_RE.findall(prose))


def _format_citation_repair_sources(sources: Sequence[SourceEntry]) -> str:
    """Render numbered source lines that a repair pass can copy verbatim."""
    lines: list[str] = []
    for number, source in enumerate(sources, 1):
        if source.url:
            lines.append(f"- [{number}] Source {number} - {source.url}")
        elif source.citation_key:
            lines.append(f"- [{number}] {source.citation_key}")
    return "\n".join(lines)


class ShallowResearcherAgent:
    """
    Shallow research agent for fast, bounded research with tool-calling.

    This agent performs quick lookups and straightforward queries using a
    LangGraph StateGraph with tool-calling capabilities. It generates optional
    mini-plans for multi-step queries and executes bounded tool-calling loops.

    The agent is NAT-independent and receives all dependencies via constructor.

    Example:
        >>> from deep_researcher_agent.common import LLMProvider, LLMRole
        >>> provider = LLMProvider()
        >>> provider.set_default(my_llm)
        >>>
        >>> from lib.models import ShallowResearchAgentState
        >>> agent = ShallowResearcherAgent(
        ...     llm_provider=provider,
        ...     tools=[web_search_tool, doc_search_tool],
        ...     max_tool_iterations=5,
        ... )
        >>> state = ShallowResearchAgentState(messages=[HumanMessage(content="What is CUDA?")])
        >>> result = await agent.run(state)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool],
        *,
        system_prompt: str | None = None,
        max_llm_turns: int = 10,
        max_tool_iterations: int = 5,
        citation_repair_timeout: float = 60.0,
        enforce_citations: bool = False,
        callbacks: list[Any] | None = None,
    ) -> None:
        """
        Initialize the shallow researcher agent.

        Args:
            llm_provider: LLMProvider for role-based LLM access.
            tools: Sequence of LangChain tools for research.
            system_prompt: Optional custom system prompt. If not provided,
                          loads system.j2 from prompts.
            max_llm_turns: Maximum LLM interaction turns (default 10).
            max_tool_iterations: Maximum tool-calling iterations before forcing
                                synthesis (default 5).
            citation_repair_timeout: Maximum seconds for the one-shot citation
                                     repair call (default 60).
            enforce_citations: Whether missing or invalid citation integrity
                               should fail the run instead of returning the
                               generated answer (default False).
            callbacks: Optional list of LangGraph callbacks.
        """
        self.llm_provider = llm_provider
        self.tools = list(tools)
        self.max_llm_turns = max_llm_turns
        self.max_tool_iterations = max_tool_iterations
        self.citation_repair_timeout = citation_repair_timeout
        self.enforce_citations = enforce_citations
        self.callbacks = callbacks or []

        # Load prompts
        self.system_prompt = system_prompt or self._load_system_prompt()

        # Build tools info for prompt rendering
        self.tools_info = self._build_tools_info()

        # Source registry for citation verification (standalone mode fallback)
        self.source_registry = SourceRegistry()

        # Build the LangGraph
        self._graph = self._build_graph()

    def _load_system_prompt(self) -> str:
        """Load the default system prompt."""
        try:
            return load_prompt(AGENT_DIR / "prompts", "researcher")
        except Exception:
            logger.warning("Shallow research prompt not found, using inline default")
            return (
                "You are a research assistant. Answer the user's question using the "
                "available tools. Be concise and cite sources when possible.\n\n"
                "{% if tools %}Available tools: "
                "{{ tools | map(attribute='name') | join(', ') }}{% endif %}"
            )

    def _build_tools_info(self) -> list[dict[str, str]]:
        """Build tools information for prompt rendering."""
        tools_info = []
        for tool in self.tools:
            tool_name = getattr(tool, "name", str(tool))
            tool_desc = getattr(tool, "description", "No description available")
            tools_info.append({"name": tool_name, "description": tool_desc})
        return tools_info

    def _get_llm(self) -> BaseChatModel:
        """Get the LLM for shallow research."""
        return self.llm_provider.get(LLMRole.RESEARCHER)

    async def _repair_missing_citations(
        self,
        messages: Sequence[Any],
        sources: Sequence[SourceEntry],
    ) -> str:
        """Run one bounded, tool-free repair against captured source identities."""
        source_catalog = _format_citation_repair_sources(sources)
        if not source_catalog:
            raise CitationIntegrityError()

        repair_system = SystemMessage(
            content=(
                "You are a deterministic citation-repair editor. Do not answer the original question again from "
                "memory and do not call tools. Rewrite only the immediately preceding draft. Keep only claims "
                "supported by prior tool results. Your response is invalid unless it contains at least one inline "
                "[N] marker and a final **References:** section copied from the allowed reference lines."
            )
        )
        repair_request = HumanMessage(
            content=(
                "The immediately preceding draft failed the citation contract. Rewrite it once using only claims "
                "supported by the prior tool results. Preserve the answer's meaning, remove unsupported claims, "
                "and do not call tools. Add an inline [N] marker after each externally verified claim and finish "
                "with a `**References:**` section. Copy the corresponding allowed reference lines verbatim; never "
                "invent or reconstruct a URL. Return only the repaired report.\n\n"
                f"Allowed reference lines:\n{source_catalog}"
            )
        )
        repair_config: dict[str, Any] = {"tags": [SUPPRESS_OUTPUT_ARTIFACT_TAG]}
        if self.callbacks:
            repair_config["callbacks"] = self.callbacks

        try:
            response = await asyncio.wait_for(
                ainvoke_with_relay(
                    self._get_llm(),
                    [repair_system, *messages, repair_request],
                    callbacks=self.callbacks,
                    config=repair_config,
                ),
                timeout=self.citation_repair_timeout,
            )
        except Exception as ex:
            logger.warning(
                "Shallow citation repair failed (error_type=%s detail_%s)",
                type(ex).__name__,
                log_content_metadata(ex),
            )
            raise CitationIntegrityError() from ex

        repaired_content = getattr(response, "content", None)
        if not isinstance(repaired_content, str) or not repaired_content.strip():
            raise CitationIntegrityError()
        return repaired_content

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph StateGraph."""

        source_tool_names = {tool.name for tool in self.tools}

        async def agent_node(state: ShallowResearchAgentState) -> dict[str, Any]:
            """Execute the agent with parallel call tracking and context anchoring."""
            messages = state.messages
            user_info = state.user_info
            iterations = state.tool_iterations

            tools_info = state.tools_info if state.tools_info else self.tools_info

            # Get available documents (user-uploaded files with summaries)
            available_documents = state.available_documents or []

            if available_documents:
                logger.debug("ShallowResearcher received %d available documents", len(available_documents))
                for doc in available_documents:
                    logger.debug("  - [file]: %s", "summary available" if doc.summary else "no summary")
            else:
                logger.debug("ShallowResearcher received no available documents")

            # Render system prompt with current datetime and available documents
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rendered_system_prompt = render_prompt_template(
                self.system_prompt,
                tools=tools_info,
                user_info=user_info,
                current_datetime=current_datetime,
                available_documents=[doc.model_dump() for doc in available_documents],
            )
            # Preserve prompt-shape diagnostics without writing customer or
            # configuration content to logs.
            if os.environ.get("DEBUG_PROMPTS"):
                logger.debug("Rendered system prompt: %s", log_content_metadata(rendered_system_prompt))

            system_message = SystemMessage(content=rendered_system_prompt)

            processed_history = list(messages)

            try:
                draft_config = {"tags": [SUPPRESS_OUTPUT_ARTIFACT_TAG]}
                if iterations >= self.max_tool_iterations:
                    logger.warning("Max iterations (%d) reached. Forcing synthesis.", iterations)

                    # Anchor instruction at the end to combat "Loss in the Middle"
                    synthesis_anchor = HumanMessage(
                        content=(
                            "You have exhausted your research budget. Synthesize the final answer now "
                            "using the citations [1], [2] and the '## References' format. "
                            "Do not attempt any further tool calls."
                        )
                    )

                    full_messages = [system_message] + processed_history + [synthesis_anchor]
                    response = await ainvoke_with_relay(
                        self._get_llm(),
                        full_messages,
                        callbacks=self.callbacks,
                        config=draft_config,
                    )
                    return {"messages": [response], "tool_iterations": iterations}

                llm = self._get_llm()
                llm_with_tools = llm.bind_tools(self.tools) if self.tools else llm
                full_messages = [system_message] + processed_history
                response = await ainvoke_with_relay(
                    llm_with_tools,
                    full_messages,
                    callbacks=self.callbacks,
                    config=draft_config,
                )

                if self.tools and iterations == 0 and not getattr(response, "tool_calls", None):
                    logger.warning("Shallow researcher returned an answer before collecting evidence; retrying once")
                    tool_required = HumanMessage(
                        content=(
                            "Research is required before answering. Call exactly one available research tool now. "
                            "Do not provide a final answer until the tool result is available."
                        )
                    )
                    retry_llm = llm.bind_tools(self.tools, parallel_tool_calls=False)
                    response = await ainvoke_with_relay(
                        retry_llm,
                        full_messages + [response, tool_required],
                        callbacks=self.callbacks,
                        config=draft_config,
                    )
                    retry_tool_calls = getattr(response, "tool_calls", None) or []
                    if len(retry_tool_calls) != 1 or retry_tool_calls[0].get("name") not in source_tool_names:
                        raise RuntimeError(
                            "shallow_research_tool_required: model did not call exactly one allowed research tool "
                            "after one retry"
                        )

                new_iterations = iterations
                if hasattr(response, "tool_calls") and response.tool_calls:
                    added_calls = len(response.tool_calls)
                    new_iterations += added_calls
                    logger.info("Added %d tool calls to budget. Total: %d", added_calls, new_iterations)

                return {"messages": [response], "tool_iterations": new_iterations}

            except Exception as ex:
                logger.error(
                    "Failed in agent_node (error_type=%s detail_%s)",
                    type(ex).__name__,
                    log_content_metadata(ex),
                )
                raise

        builder = StateGraph(ShallowResearchAgentState)

        builder.set_entry_point("agent")

        tool_node = ToolNode(self.tools, awrap_tool_call=awrap_tool_call_with_relay)

        # Per-agent allowlist mirrors the deep researcher: only tools this
        # agent was loaded with are candidates for source capture. The
        # data_source_registry then decides which of those are configured
        # data sources. Having both gates keeps behavior consistent across
        # agents and safe even if the global registry is ever polluted.
        async def tool_node_with_source_capture(state: ShallowResearchAgentState) -> dict[str, Any]:
            """Execute tools and capture source URLs/citations for verification.

            Source capture is gated by two conditions:

            1. The tool must be in this agent's loaded tool set
               (``source_tool_names``) — mirrors the deep researcher's
               middleware allowlist.
            2. The tool must resolve to a configured data source via
               :func:`get_source_id_for_tool` (i.e. declared under
               ``data_sources`` in the workflow YAML).

            Tools that fail either check (internal scratchpads, ad-hoc
            utilities, unregistered MCP servers) are skipped without
            contributing to the citation registry.
            """
            result = await tool_node.ainvoke(state)
            # Resolve registry at call time (not build time) so each request
            # writes to its own session-scoped registry when available.
            active_registry = get_session_registry() or self.source_registry
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.content:
                    tool_name = getattr(msg, "name", "") or ""
                    if tool_name not in source_tool_names:
                        continue
                    source_id = get_source_id_for_tool(tool_name)
                    if source_id is None:
                        logger.debug(
                            "[CitationRegistry] Skipping non-data-source tool result from %s",
                            tool_name,
                        )
                        continue
                    sources = extract_sources_from_tool_result(
                        tool_name,
                        str(msg.content),
                        source_id=source_id,
                        result_status=getattr(msg, "status", None),
                    )
                    for source in sources:
                        active_registry.add(source)
                    if sources:
                        logger.info(
                            "[CitationRegistry] Captured %d source(s) from %s",
                            len(sources),
                            tool_name,
                        )
            return result

        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node_with_source_capture)

        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", "__end__": "__end__"},
        )
        builder.add_edge("tools", "agent")

        return builder.compile()

    async def run(self, state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        """
        Execute shallow research with tool-calling.

        Args:
            state: ShallowResearchAgentState with conversation messages.

        Returns:
            Updated state with response in messages.
        """
        # Resolve the registry for this request: session-scoped (conversation
        # mode) or instance-scoped with clear (standalone mode).  We use a
        # local variable so we never mutate the shared agent instance.
        session_registry = get_session_registry()
        if session_registry is not None:
            registry = session_registry
        else:
            self.source_registry.clear()
            registry = self.source_registry

        recursion_limit = (self.max_llm_turns * 2) + 10
        config = {"recursion_limit": recursion_limit}

        async def _invoke_graph() -> dict[str, Any]:
            config["callbacks"] = self.callbacks
            return await self._graph.ainvoke(state, config=config)

        result = await run_agent("shallow_research_agent", _invoke_graph, input_value=state)

        # Post-process: verify citations against source registry
        validated_result = dict(result)
        last_msg = validated_result["messages"][-1] if validated_result.get("messages") else None
        content = str(last_msg.content) if last_msg is not None and getattr(last_msg, "content", None) else None

        if not registry.all_sources():
            from deep_researcher_agent.common.citation_verification import classify_empty_source_registry_reason
            from deep_researcher_agent.common.tool_validation import validate_tool_availability

            _, available_count, unavailable = validate_tool_availability(
                self.tools,
                research_type="shallow research",
                enable_logging=False,
            )
            generated_answer = sanitize_report(content).sanitized_report if content is not None else None
            if self.enforce_citations or generated_answer is None:
                raise EmptySourceRegistryError(
                    "shallow research",
                    unavailable_tools=unavailable,
                    available_count=available_count,
                    reason=classify_empty_source_registry_reason(state.data_sources, available_count, unavailable),
                    generated_answer=generated_answer,
                )

            logger.info(
                "Shallow research completed without captured sources; returning generated answer because "
                "enforce_citations is false (available_tools=%d unavailable_tools=%d)",
                available_count,
                len(unavailable),
            )
            content = generated_answer
            if last_msg is not None:
                for cb in self.callbacks:
                    if hasattr(cb, "emit_final_report"):
                        cb.emit_final_report(content, cited_urls=[])
                        break

                if hasattr(last_msg, "model_copy"):
                    validated_result["messages"][-1] = last_msg.model_copy(update={"content": content})
                else:
                    validated_result["messages"][-1] = type(last_msg)(content=content)
            return ShallowResearchAgentState.model_validate(validated_result)

        if validated_result.get("messages"):
            if content is not None:
                # Step 1: verify citations against registry
                if registry.all_sources():
                    verification = verify_citations(content, registry)
                    logger.debug(
                        "Shallow researcher: citation verification complete — "
                        "%d valid, %d removed, %d sources in registry",
                        len(verification.valid_citations),
                        len(verification.removed_citations),
                        len(registry.all_sources()),
                    )
                    content = verification.verified_report
                    sources = registry.all_sources()
                    citation_integrity = _has_citation_integrity(content, verification.valid_citations)
                    if not citation_integrity and len(sources) == 1:
                        content = _append_minimal_citation(content, sources[0])
                    elif not citation_integrity and self.enforce_citations:
                        logger.info(
                            "Shallow report is missing citation integrity; attempting one bounded repair "
                            "(registered_sources=%d)",
                            len(sources),
                        )
                        content = await self._repair_missing_citations(validated_result["messages"], sources)
                        repair_verification = verify_citations(content, registry, reference_sources=sources)
                        content = repair_verification.verified_report
                        if not _has_citation_integrity(content, repair_verification.valid_citations):
                            logger.warning(
                                "Shallow citation repair did not restore integrity "
                                "(registered_sources=%d verified_sources=%d)",
                                len(sources),
                                len(repair_verification.valid_citations),
                            )
                    elif not citation_integrity:
                        logger.info(
                            "Shallow report is missing citation integrity; returning generated answer because "
                            "enforce_citations is false (registered_sources=%d)",
                            len(sources),
                        )
                # Step 2: sanitize report (strip body URLs, shortened URLs, unsafe URLs)
                sanitization = sanitize_report(content)
                content = sanitization.sanitized_report
                final_verification = verify_citations(content, registry)
                if not _has_citation_integrity(
                    final_verification.verified_report,
                    final_verification.valid_citations,
                ):
                    logger.warning(
                        "Shallow report failed final citation integrity check "
                        "(registered_sources=%d verified_sources=%d)",
                        len(registry.all_sources()),
                        len(final_verification.valid_citations),
                    )
                    if self.enforce_citations:
                        raise CitationIntegrityError()
                content = _format_chat_references(final_verification.verified_report)
                final_cited_urls = list(
                    dict.fromkeys(
                        citation["url"] for citation in final_verification.valid_citations if citation.get("url")
                    )
                )

                # Emit verified/sanitized report so the frontend shows the
                # cleaned version (overwrites the raw draft auto-emitted
                # during ainvoke).
                for cb in self.callbacks:
                    if hasattr(cb, "emit_final_report"):
                        cb.emit_final_report(content, cited_urls=final_cited_urls)
                        break

                if hasattr(last_msg, "model_copy"):
                    validated_result["messages"][-1] = last_msg.model_copy(update={"content": content})
                else:
                    validated_result["messages"][-1] = type(last_msg)(content=content)

        return ShallowResearchAgentState.model_validate(validated_result)

    @property
    def graph(self) -> CompiledStateGraph:
        """Get the compiled LangGraph for direct access."""
        return self._graph
