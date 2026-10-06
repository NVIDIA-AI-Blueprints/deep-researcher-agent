# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal report rewriter agent."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole
from deep_researcher_agent.common import load_prompt
from deep_researcher_agent.common import render_prompt_template
from deep_researcher_agent.common.citation_verification import SourceEntry
from deep_researcher_agent.common.citation_verification import SourceRegistry
from deep_researcher_agent.common.citation_verification import extract_source_entries_from_report
from deep_researcher_agent.common.citation_verification import report_has_citations
from deep_researcher_agent.common.citation_verification import sanitize_report
from deep_researcher_agent.common.citation_verification import source_entries_from_parent_context
from deep_researcher_agent.common.citation_verification import verify_citations
from deep_researcher_agent.common.logging_utils import log_identifier_ref
from deep_researcher_agent.relay import ainvoke_with_relay
from deep_researcher_agent.relay import run_agent

from .models import ReportRewriterAgentState

logger = logging.getLogger(__name__)


def _job_ref(job_id: str | None) -> str:
    """Redacted correlation ref for the rewriter job UUID (an opaque bearer capability).

    The raw UUID must never enter logs; ``log_identifier_ref`` yields a stable
    ``sha256:<12hex>`` reference instead. ``job_id`` is optional, so fall back to
    ``"none"`` rather than hashing ``None``.
    """
    return log_identifier_ref(job_id) if job_id else "none"


AGENT_DIR = Path(__file__).parent
ORIGINAL_REPORT_PATH = "/shared/original_report.md"
SOURCE_SUMMARY_PATH = "/shared/source_summary.md"
PARENT_CONTEXT_PATH = "/shared/parent_report_context.json"
EDIT_INSTRUCTION_PATH = "/shared/edit_instruction.txt"
OUTPUT_REPORT_PATH = "/shared/output.md"

_DEFAULT_SOURCE_SUMMARY = "No durable source metadata was found for the parent report."


def _effective_parent_sources(original_report: str, parent_context: str) -> list[SourceEntry]:
    """Build the rewrite allowlist from the canonical report and durable context."""
    report_sources = extract_source_entries_from_report(original_report)
    context_sources = source_entries_from_parent_context(parent_context)
    if report_has_citations(original_report) and not (report_sources or context_sources):
        raise ValueError("Cannot rewrite a cited parent report because it cannot reconstruct its source registry")

    registry = SourceRegistry()
    for source in [*report_sources, *context_sources]:
        registry.add(source)
    return registry.all_sources()


def _post_process_revised_report(revised_report: str, parent_sources: Sequence[SourceEntry]) -> str:
    if parent_sources:
        registry = SourceRegistry()
        for source in parent_sources:
            registry.add(source)
        verification = verify_citations(revised_report, registry, reference_sources=parent_sources)
        if verification.removed_citations:
            logger.info(
                "Report rewrite citation verification removed %d invalid citation(s)",
                len(verification.removed_citations),
            )
        revised_report = verification.verified_report
    elif report_has_citations(revised_report):
        raise ValueError("Cannot publish a rewritten report with citations without a verified parent source registry")

    return sanitize_report(revised_report).sanitized_report


def _verified_cited_urls(report: str, sources: Sequence[SourceEntry]) -> list[str]:
    """Return the verified URL citations still present in a finalized rewrite."""
    if not sources:
        return []
    registry = SourceRegistry()
    for source in sources:
        registry.add(source)
    verification = verify_citations(report, registry)
    return list(dict.fromkeys(citation["url"] for citation in verification.valid_citations if citation.get("url")))


async def rewrite_report(
    *,
    llm: Any,
    original_report: str,
    edit_instruction: str,
    source_summary: str = _DEFAULT_SOURCE_SUMMARY,
    parent_context: str = "{}",
    system_prompt: str | None = None,
    callbacks: list[Any] | None = None,
) -> str:
    """Rewrite a report per an edit instruction with one bounded LLM call.

    Shared by the async ``report_rewriter`` job and the synchronous in-session CLI edit path so
    both produce identical revisions. Needs no filesystem, job store, or scheduler.
    """
    instruction = (edit_instruction or "").strip()
    if not instruction:
        raise ValueError("Report rewrite requires a non-empty edit instruction")
    if system_prompt is None:
        system_prompt = load_prompt(AGENT_DIR / "prompts", "edit")
    rendered_prompt = render_prompt_template(
        system_prompt,
        original_report=original_report,
        source_summary=source_summary,
        parent_context=parent_context,
        edit_instruction=instruction,
    )
    messages = [
        SystemMessage(content=rendered_prompt),
        HumanMessage(content=instruction),
    ]
    response = await ainvoke_with_relay(llm, messages, callbacks=callbacks)
    revised_report = response.content if hasattr(response, "content") else str(response)
    revised_report = revised_report if isinstance(revised_report, str) else str(revised_report)
    revised_report = revised_report.strip()
    if not revised_report:
        raise ValueError("Report writer returned an empty revised report")
    return _post_process_revised_report(
        revised_report,
        parent_sources=_effective_parent_sources(original_report, parent_context),
    )


class ReportRewriterAgent:
    """Rewrite a completed parent report into a full revised child report."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[Any] | None = None,
        *,
        callbacks: list[Any] | None = None,
        job_id: str | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.callbacks = callbacks or []
        self.job_id = job_id
        self.system_prompt = load_prompt(AGENT_DIR / "prompts", "edit")

    @staticmethod
    def _read_text_file(files: dict[str, Any], path: str) -> str | None:
        value = files.get(path)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _latest_user_message(state: ReportRewriterAgentState) -> str:
        for message in reversed(state.messages):
            if isinstance(message, HumanMessage):
                content = message.content
                return content if isinstance(content, str) else str(content)
        return ""

    async def run(self, state: ReportRewriterAgentState) -> ReportRewriterAgentState:
        original_report = self._read_text_file(state.files, ORIGINAL_REPORT_PATH)
        if original_report is None:
            raise ValueError(f"Report rewrite requires {ORIGINAL_REPORT_PATH}")

        instruction = (
            self._read_text_file(state.files, EDIT_INSTRUCTION_PATH) or self._latest_user_message(state)
        ).strip()

        source_summary = self._read_text_file(state.files, SOURCE_SUMMARY_PATH) or _DEFAULT_SOURCE_SUMMARY
        parent_context = self._read_text_file(state.files, PARENT_CONTEXT_PATH) or "{}"

        async def _rewrite() -> str:
            return await rewrite_report(
                llm=self.llm_provider.get(LLMRole.REPORT_WRITER),
                original_report=original_report,
                edit_instruction=instruction,
                source_summary=source_summary,
                parent_context=parent_context,
                system_prompt=self.system_prompt,
                callbacks=self.callbacks,
            )

        revised_report = await run_agent("report_rewriter_agent", _rewrite, input_value=state)
        cited_urls = _verified_cited_urls(
            revised_report,
            _effective_parent_sources(original_report, parent_context),
        )

        for callback in self.callbacks:
            if hasattr(callback, "emit_final_report"):
                callback.emit_final_report(revised_report, cited_urls=cited_urls)
                break

        files = dict(state.files)
        files[OUTPUT_REPORT_PATH] = revised_report
        logger.info("Report rewrite complete (job_ref=%s, chars=%d)", _job_ref(self.job_id), len(revised_report))
        return ReportRewriterAgentState(
            messages=[*state.messages, AIMessage(content=revised_report)],
            files=files,
        )
