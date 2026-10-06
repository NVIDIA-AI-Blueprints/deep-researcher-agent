# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_extract_report_from_output_prefers_job_output():
    from deep_researcher_api.jobs.report_context import _extract_report_from_output

    assert _extract_report_from_output({"report": "# Stored report"}) == "# Stored report"


def test_report_from_events_prefers_final_report():
    from deep_researcher_api.jobs import report_context

    events = [
        {"type": "artifact.update", "data": {"type": "output", "content": "draft"}},
        {
            "type": "artifact.update",
            "data": {"type": "output", "content": "# Final", "output_category": "final_report"},
        },
    ]

    assert report_context._report_from_events(events) == "# Final"


def test_sources_from_events_preserves_duplicates_until_context_boundary():
    from deep_researcher_api.jobs import report_context

    events = [
        {
            "type": "artifact.update",
            "name": "Example",
            "data": {"type": "citation_source", "content": "https://example.com/", "url": "https://example.com/"},
        },
        {
            "type": "artifact.update",
            "name": "Example again",
            "data": {"type": "citation_source", "content": "https://example.com", "url": "https://example.com"},
        },
        {
            "type": "artifact.update",
            "name": "internal.pdf",
            "data": {
                "type": "citation_source",
                "content": "internal.pdf, p.3",
                "citation_key": "internal.pdf, p.3",
            },
        },
        {
            "type": "artifact.update",
            "name": "internal.pdf duplicate",
            "data": {
                "type": "citation_source",
                "content": "internal.pdf, p.3",
                "citation_key": "internal.pdf, p.3",
            },
        },
    ]

    sources = report_context._sources_from_events(events)

    assert [(source.url, source.citation_key) for source in sources] == [
        ("https://example.com/", None),
        ("https://example.com", None),
        (None, "internal.pdf, p.3"),
        (None, "internal.pdf, p.3"),
    ]


def test_extract_sources_from_report_markdown_finds_urls_and_citation_keys():
    from deep_researcher_api.jobs.report_context import _extract_sources_from_report_markdown

    report = """# Report

Body [1].

## Sources

[1] Example: https://example.com/path
[2] internal.pdf, p.3

## Appendix

Not a source: https://ignored.example
"""

    sources = _extract_sources_from_report_markdown(report)

    assert [(source.url, source.citation_key) for source in sources] == [
        ("https://example.com/path", None),
        (None, "internal.pdf, p.3"),
    ]


def test_extract_sources_from_report_markdown_preserves_duplicates_until_context_boundary():
    from deep_researcher_api.jobs.report_context import _extract_sources_from_report_markdown

    report = """# Report

## Sources

[1] https://example.com/path
[2] https://example.com/path/
"""

    sources = _extract_sources_from_report_markdown(report)

    assert [source.url for source in sources] == ["https://example.com/path", "https://example.com/path/"]


@pytest.mark.asyncio
async def test_resolve_report_context_raises_409_without_report(monkeypatch):
    from deep_researcher_api.jobs import report_context

    async def _no_events(_db_url: str, _job_id: str, _after_id: int, _limit: int):
        return []

    monkeypatch.setattr(report_context.EventStore, "get_events_async", _no_events)

    job = type("Job", (), {"output": None})()

    with pytest.raises(HTTPException) as exc:
        await report_context.resolve_report_context(job, "sqlite:///unused.db", "job-1")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_resolve_report_context_fetches_events_once_in_fallback(monkeypatch):
    """When the report is reconstructed from events, the event log is fetched once, not twice."""
    from deep_researcher_api.jobs import report_context

    calls = {"n": 0}

    async def _events(_db_url: str, _job_id: str, _after_id: int, _limit: int):
        calls["n"] += 1
        return [
            {
                "type": "artifact.update",
                "data": {"type": "output", "content": "# Final", "output_category": "final_report"},
            },
            {
                "type": "artifact.update",
                "name": "Ex",
                "data": {"type": "citation_source", "url": "https://example.com"},
            },
        ]

    monkeypatch.setattr(report_context.EventStore, "get_events_async", _events)

    job = type("Job", (), {"output": None})()
    ctx = await report_context.resolve_report_context(job, "sqlite:///unused.db", "job-1")

    assert ctx.report_markdown == "# Final"
    assert [s.url for s in ctx.sources] == ["https://example.com"]
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_resolve_report_context_skips_event_scan_when_output_has_inline_sources(monkeypatch):
    """Report in job.output with its own ## Sources -> skip the (potentially large) event scan."""
    from deep_researcher_api.jobs import report_context

    calls = {"n": 0}

    async def _events(_db_url: str, _job_id: str, _after_id: int, _limit: int):
        calls["n"] += 1
        return []

    monkeypatch.setattr(report_context.EventStore, "get_events_async", _events)

    report = "# Report\n\nBody [1].\n\n## Sources\n\n[1] https://example.com/path\n"
    job = type("Job", (), {"output": {"report": report}})()

    ctx = await report_context.resolve_report_context(job, "sqlite:///unused.db", "job-1")

    assert calls["n"] == 0
    assert [s.url for s in ctx.sources] == ["https://example.com/path"]


@pytest.mark.asyncio
async def test_resolve_report_context_decrypts_encrypted_parent_output(monkeypatch):
    """Report follow-up reads the encrypted parent output instead of relying on event fallback."""
    import base64
    import json
    from unittest.mock import AsyncMock

    from deep_researcher_api.jobs import crypto
    from deep_researcher_api.jobs import report_context

    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv(
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(b"k" * crypto.DEK_BYTES).decode(),
    )
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID", "test-key")
    crypto.reset_content_encryption_manager_for_tests()

    report = "# Encrypted report\n\n## Sources\n\n[1] https://example.com/encrypted\n"
    stored_output = crypto.create_job_content_cipher("job-1").encrypt_output_json(json.dumps({"report": report}))
    job = type("Job", (), {"output": stored_output})()
    get_events = AsyncMock(return_value=[])
    monkeypatch.setattr(report_context.EventStore, "get_events_async", get_events)

    try:
        context = await report_context.resolve_report_context(job, "sqlite:///unused.db", "job-1")
    finally:
        crypto.reset_content_encryption_manager_for_tests()

    assert context.report_markdown == report.strip()
    assert [source.url for source in context.sources] == ["https://example.com/encrypted"]
    get_events.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_report_context_scans_events_when_output_report_lacks_sources(monkeypatch):
    """Report in job.output but no inline ## Sources -> fall back to the event scan for enrichment."""
    from deep_researcher_api.jobs import report_context

    calls = {"n": 0}

    async def _events(_db_url: str, _job_id: str, _after_id: int, _limit: int):
        calls["n"] += 1
        return [
            {
                "type": "artifact.update",
                "name": "Ex",
                "data": {"type": "citation_source", "url": "https://events.example/x"},
            },
        ]

    monkeypatch.setattr(report_context.EventStore, "get_events_async", _events)

    job = type("Job", (), {"output": {"report": "# Report\n\nNo sources section here.\n"}})()

    ctx = await report_context.resolve_report_context(job, "sqlite:///unused.db", "job-1")

    assert calls["n"] == 1
    assert [s.url for s in ctx.sources] == ["https://events.example/x"]


def test_to_initial_files_uses_shared_paths_only():
    from deep_researcher_api.jobs.report_context import ReportContext
    from deep_researcher_api.jobs.report_context import ReportContextSource
    from deep_researcher_api.jobs.report_context import to_initial_files

    context = ReportContext(
        parent_job_id="job-1",
        report_markdown="# Report",
        source_summary_markdown="- https://example.com",
        sources=[ReportContextSource(url="https://example.com")],
    )

    files = to_initial_files(context, instruction="Remove the appendix.")

    assert files["/shared/original_report.md"] == "# Report"
    assert files["/shared/source_summary.md"] == "- https://example.com"
    assert files["/shared/edit_instruction.txt"] == "Remove the appendix."
    assert "/report.md" not in files


def test_report_context_from_markdown_builds_jobless_context():
    """In-session (CLI) report context: built from markdown, no job store / auth / scheduler."""
    from deep_researcher_api.jobs.report_context import report_context_from_markdown

    md = "# Findings\n\nThe key risk is X.\n\n## Sources\n\n[1] https://example.com/a\n[2] https://example.com/b\n"
    ctx = report_context_from_markdown(md)

    assert ctx.report_markdown == md
    assert ctx.parent_job_id == "in-session"
    urls = {s.url for s in ctx.sources}
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls
    assert ctx.source_summary_markdown.strip()


def test_report_context_from_markdown_dedupes_sources_at_context_boundary():
    from deep_researcher_api.jobs.report_context import report_context_from_markdown

    md = "# Findings\n\n## Sources\n\n[1] https://example.com/a\n[2] https://example.com/a/\n"
    ctx = report_context_from_markdown(md)

    assert [source.url for source in ctx.sources] == ["https://example.com/a"]


def test_report_context_from_markdown_no_sources_section():
    from deep_researcher_api.jobs.report_context import report_context_from_markdown

    ctx = report_context_from_markdown("# Findings\n\nNo sources here.")
    assert ctx.sources == []
    assert "No durable source metadata" in ctx.source_summary_markdown


@pytest.mark.asyncio
async def test_resolve_authorized_report_context_caches_job_store(monkeypatch):
    """JobStore is built once per (scheduler_address, db_url), not per request.

    JobStore eagerly builds a SQLAlchemy AsyncEngine (own connection pool), so per-request
    construction wastes work and leaks pooled connections.
    """
    from deep_researcher_api.jobs import report_context

    report_context._JOB_STORE_CACHE.clear()

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setenv("NAT_JOB_STORE_DB_URL", "sqlite:///unused.db")

    constructions = {"n": 0}

    class _FakeJobStore:
        def __init__(self, *, scheduler_address, db_url):
            constructions["n"] += 1

    class _FakeJobStatus:
        SUCCESS = type("E", (), {"value": "success"})

    import nat.front_ends.fastapi.async_jobs.job_store as js

    monkeypatch.setattr(js, "JobStore", _FakeJobStore)
    monkeypatch.setattr(js, "JobStatus", _FakeJobStatus)

    job = type("Job", (), {"status": "success"})()

    async def _authorize(_store, _db_url, _job_id, _principal):
        return job

    async def _resolve(_job, _db_url, _job_id):
        return report_context.ReportContext(parent_job_id=_job_id, report_markdown="# R", source_summary_markdown="")

    monkeypatch.setattr(report_context, "authorize_job_access", _authorize)
    monkeypatch.setattr(report_context, "resolve_report_context", _resolve)

    await report_context.resolve_authorized_report_context("job-1", None)
    await report_context.resolve_authorized_report_context("job-1", None)

    assert constructions["n"] == 1
    report_context._JOB_STORE_CACHE.clear()
