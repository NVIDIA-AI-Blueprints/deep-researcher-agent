# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from deep_researcher_agent.common import LLMProvider
from deep_researcher_agent.common import LLMRole
from deep_researcher_agent.common.logging_utils import log_identifier_ref


class FakeWriterLLM:
    def __init__(self, content: str = "# Revised Report\n\nUpdated body.") -> None:
        self.content = content
        self.seen_messages = None

    async def ainvoke(self, messages):
        self.seen_messages = messages
        return AIMessage(content=self.content)


@pytest.mark.asyncio
async def test_report_rewriter_uses_parent_report_context_and_emits_output_file():
    from deep_researcher_agent.agents.report_rewriter.agent import ReportRewriterAgent
    from deep_researcher_agent.agents.report_rewriter.models import ReportRewriterAgentState

    writer_llm = FakeWriterLLM()
    provider = LLMProvider()
    provider.configure(LLMRole.REPORT_WRITER, writer_llm)
    callback = MagicMock()
    duplicate_callback = MagicMock()
    agent = ReportRewriterAgent(llm_provider=provider, tools=[], callbacks=[callback, duplicate_callback])

    state = ReportRewriterAgentState(
        messages=[HumanMessage(content="Remove the appendix.")],
        files={
            "/shared/original_report.md": "# Parent Report\n\nBody.\n\n## Appendix\n\nRemove me.",
            "/shared/source_summary.md": "- [1] Example: https://example.com",
            "/shared/edit_instruction.txt": "Remove the appendix.",
        },
    )

    result = await agent.run(state)

    assert result.files["/shared/output.md"] == "# Revised Report\n\nUpdated body."
    assert result.messages[-1].content == "# Revised Report\n\nUpdated body."
    rendered_prompt = writer_llm.seen_messages[0].content
    assert "# Parent Report" in rendered_prompt
    assert "Remove the appendix." in rendered_prompt
    assert "complete standalone Markdown report" in rendered_prompt
    callback.emit_final_report.assert_called_once_with("# Revised Report\n\nUpdated body.", cited_urls=[])
    duplicate_callback.emit_final_report.assert_not_called()


@pytest.mark.asyncio
async def test_rewrite_report_core_returns_revised_markdown_and_renders_prompt():
    """The extracted rewrite core (shared by the job and the inline CLI path) is a single LLM call."""
    from deep_researcher_agent.agents.report_rewriter.agent import rewrite_report

    writer_llm = FakeWriterLLM(content="# Edited\n\nNew body.")
    revised = await rewrite_report(
        llm=writer_llm,
        original_report="# Parent\n\nBody.",
        edit_instruction="Add a Messi-Ronaldo joke under the title.",
        source_summary="- [1] https://example.com",
    )

    assert revised == "# Edited\n\nNew body."
    rendered_prompt = writer_llm.seen_messages[0].content
    assert "# Parent" in rendered_prompt
    assert "Add a Messi-Ronaldo joke under the title." in rendered_prompt


@pytest.mark.asyncio
async def test_report_rewriter_verifies_and_sanitizes_revised_report_against_parent_sources():
    from deep_researcher_agent.agents.report_rewriter.agent import ReportRewriterAgent
    from deep_researcher_agent.agents.report_rewriter.models import ReportRewriterAgentState

    writer_llm = FakeWriterLLM(
        content=(
            "# Revised Report\n\n"
            "Supported claim [1]. Unsupported claim [2].\n\n"
            "## Sources\n"
            "[1] Valid Source: https://valid.example/source\n"
            "[2] Fabricated Source: https://fabricated.example/source"
        )
    )
    provider = LLMProvider()
    provider.configure(LLMRole.REPORT_WRITER, writer_llm)
    callback = MagicMock()
    agent = ReportRewriterAgent(llm_provider=provider, tools=[], callbacks=[callback])

    parent_context = {
        "parent_job_id": "parent-job",
        "sources": [
            {
                "url": "https://valid.example/source",
                "title": "Valid Source",
                "source_type": "parent_report",
                "tool_name": "parent_report",
            }
        ],
    }
    state = ReportRewriterAgentState(
        messages=[HumanMessage(content="Make it clearer.")],
        files={
            "/shared/original_report.md": "# Parent\n\nSupported claim [1].\n\n## Sources\n[1] Valid Source: https://valid.example/source",
            "/shared/source_summary.md": "- [1] Valid Source: https://valid.example/source",
            "/shared/parent_report_context.json": json.dumps(parent_context),
        },
    )

    result = await agent.run(state)

    revised_report = result.files["/shared/output.md"]
    assert "https://valid.example/source" in revised_report
    assert "https://fabricated.example/source" not in revised_report
    assert "[2]" not in revised_report
    assert "Unsupported claim." in revised_report
    assert result.messages[-1].content == revised_report
    callback.emit_final_report.assert_called_once_with(
        revised_report,
        cited_urls=["https://valid.example/source"],
    )


@pytest.mark.asyncio
async def test_rewrite_report_reconstructs_registry_from_original_report_when_parent_context_is_malformed():
    from deep_researcher_agent.agents.report_rewriter.agent import rewrite_report

    revised = await rewrite_report(
        llm=FakeWriterLLM(
            content=(
                "# Revised Report\n\n"
                "Supported claim [1]. Unsupported claim [2].\n\n"
                "## Sources\n"
                "[1] Valid Source: https://valid.example/source\n"
                "[2] Fabricated Source: https://fabricated.example/source"
            )
        ),
        original_report=(
            "# Parent Report\n\nSupported claim [1].\n\n## Sources\n[1] Valid Source: https://valid.example/source"
        ),
        edit_instruction="Make it clearer.",
        parent_context="{not valid JSON",
    )

    assert "https://valid.example/source" in revised
    assert "https://fabricated.example/source" not in revised
    assert "[2]" not in revised


@pytest.mark.asyncio
async def test_rewrite_report_preserves_sources_missing_from_parent_context():
    from deep_researcher_agent.agents.report_rewriter.agent import rewrite_report

    revised = await rewrite_report(
        llm=FakeWriterLLM(
            content=(
                "# Revised Report\n\n"
                "First claim [1]. Second claim [2]. Unsupported claim [3].\n\n"
                "## Sources\n"
                "[1] First Source: https://first.example/source\n"
                "[2] Second Source: https://second.example/source\n"
                "[3] Fabricated Source: https://fabricated.example/source"
            )
        ),
        original_report=(
            "# Parent Report\n\nFirst claim [1]. Second claim [2].\n\n"
            "## Sources\n"
            "[1] First Source: https://first.example/source\n"
            "[2] Second Source: https://second.example/source"
        ),
        edit_instruction="Make it clearer.",
        parent_context=json.dumps(
            {
                "parent_job_id": "parent-job",
                "sources": [{"url": "https://first.example/source"}],
            }
        ),
    )

    assert "https://first.example/source" in revised
    assert "https://second.example/source" in revised
    assert "https://fabricated.example/source" not in revised
    assert "[3]" not in revised


@pytest.mark.asyncio
async def test_rewrite_report_rejects_cited_parent_without_reconstructable_sources():
    from deep_researcher_agent.agents.report_rewriter.agent import rewrite_report

    with pytest.raises(ValueError, match="cannot reconstruct its source registry"):
        await rewrite_report(
            llm=FakeWriterLLM(content="# Revised Report\n\nUpdated body."),
            original_report="# Parent Report\n\nSupported claim [1].\n\n## Sources\n[1] Missing locator",
            edit_instruction="Make it clearer.",
            parent_context="{}",
        )


@pytest.mark.asyncio
async def test_rewrite_report_core_rejects_empty_instruction():
    from deep_researcher_agent.agents.report_rewriter.agent import rewrite_report

    with pytest.raises(ValueError, match="non-empty edit instruction"):
        await rewrite_report(llm=FakeWriterLLM(), original_report="# R", edit_instruction="   ")


@pytest.mark.asyncio
async def test_report_rewriter_redacts_job_uuid_in_completion_log(caplog):
    """The success-path completion log must emit a redacted ref, never the raw job UUID."""
    from deep_researcher_agent.agents.report_rewriter.agent import ReportRewriterAgent
    from deep_researcher_agent.agents.report_rewriter.models import ReportRewriterAgentState

    provider = LLMProvider()
    provider.configure(LLMRole.REPORT_WRITER, FakeWriterLLM())
    agent = ReportRewriterAgent(llm_provider=provider, tools=[], job_id="job-1")

    state = ReportRewriterAgentState(
        messages=[HumanMessage(content="Remove the appendix.")],
        files={
            "/shared/original_report.md": "# Parent Report\n\nBody.",
            "/shared/edit_instruction.txt": "Remove the appendix.",
        },
    )

    with caplog.at_level(logging.INFO):
        await agent.run(state)

    assert "Report rewrite complete" in caplog.text
    assert "job-1" not in caplog.text
    assert log_identifier_ref("job-1") in caplog.text


@pytest.mark.asyncio
async def test_report_rewriter_completion_log_omits_job_id_when_unset(caplog):
    """A ``None`` job id must not hash to a ref or crash; it logs the ``none`` sentinel."""
    from deep_researcher_agent.agents.report_rewriter.agent import ReportRewriterAgent
    from deep_researcher_agent.agents.report_rewriter.models import ReportRewriterAgentState

    provider = LLMProvider()
    provider.configure(LLMRole.REPORT_WRITER, FakeWriterLLM())
    agent = ReportRewriterAgent(llm_provider=provider, tools=[])

    state = ReportRewriterAgentState(
        messages=[HumanMessage(content="Remove the appendix.")],
        files={
            "/shared/original_report.md": "# Parent Report\n\nBody.",
            "/shared/edit_instruction.txt": "Remove the appendix.",
        },
    )

    with caplog.at_level(logging.INFO):
        await agent.run(state)

    assert "job_ref=none" in caplog.text


@pytest.mark.asyncio
async def test_report_rewriter_requires_original_report_file():
    from deep_researcher_agent.agents.report_rewriter.agent import ReportRewriterAgent
    from deep_researcher_agent.agents.report_rewriter.models import ReportRewriterAgentState

    writer_llm = FakeWriterLLM()
    provider = LLMProvider()
    provider.configure(LLMRole.REPORT_WRITER, writer_llm)
    agent = ReportRewriterAgent(llm_provider=provider, tools=[])

    with pytest.raises(ValueError, match="/shared/original_report.md"):
        await agent.run(ReportRewriterAgentState(messages=[HumanMessage(content="Shorten it.")], files={}))
