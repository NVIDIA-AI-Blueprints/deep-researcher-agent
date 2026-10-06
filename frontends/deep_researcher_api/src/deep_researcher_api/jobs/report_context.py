# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable report context reconstruction for report-aware follow-up."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from deep_researcher_agent.auth import Principal

from .access import authorize_job_access
from .crypto import read_job_output_async
from .event_store import EventStore

_EVENT_SCAN_LIMIT = 10000
_URL_RE = re.compile(r"https?://[^\s<>)\]]+")
_SOURCES_HEADING_RE = re.compile(r"^##\s+(sources|references)\s*$", re.IGNORECASE | re.MULTILINE)
_NEXT_HEADING_RE = re.compile(r"^##\s+\S", re.MULTILINE)
_CITATION_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?\[\d+\]\s*(?P<text>.+?)\s*$")
_URL_TRIM_CHARS = ".,;:"


class ReportContextSource(BaseModel):
    """One source available to a report-aware child interaction."""

    url: str | None = None
    citation_key: str | None = None
    title: str | None = None
    source_type: str = "parent_report"
    tool_name: str = "parent_report"


class ReportContext(BaseModel):
    """Parent report and compact source context reconstructed from durable state."""

    parent_job_id: str
    report_markdown: str
    source_summary_markdown: str
    sources: list[ReportContextSource] = Field(default_factory=list)


def _decode_job_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if isinstance(output, str) and output.strip():
        try:
            decoded = json.loads(output)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _extract_report_from_output(output: Any) -> str | None:
    output = _decode_job_output(output)
    report = output.get("report")
    return report.strip() if isinstance(report, str) and report.strip() else None


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _report_from_events(events: list[dict[str, Any]]) -> str | None:
    fallback: str | None = None
    final_report: str | None = None
    for event in events:
        if event.get("type") != "artifact.update":
            continue
        data = _event_data(event)
        if data.get("type") != "output":
            continue
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        fallback = content.strip()
        if data.get("output_category") == "final_report":
            final_report = content.strip()
    return final_report or fallback


def _is_url(value: str | None) -> bool:
    return bool(value and value.lower().startswith(("http://", "https://")))


def _normalize_url_key(url: str) -> str:
    return url.rstrip("/").lower()


def _dedupe_sources(sources: list[ReportContextSource]) -> list[ReportContextSource]:
    seen: set[str] = set()
    deduped: list[ReportContextSource] = []
    for source in sources:
        if source.url:
            key = f"url:{_normalize_url_key(source.url)}"
        elif source.citation_key:
            key = f"citation_key:{source.citation_key.strip().lower()}"
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _sources_from_events(events: list[dict[str, Any]]) -> list[ReportContextSource]:
    sources: list[ReportContextSource] = []
    for event in events:
        if event.get("type") != "artifact.update":
            continue
        data = _event_data(event)
        if data.get("type") != "citation_source":
            continue
        content = data.get("content")
        content_text = content.strip() if isinstance(content, str) else None
        url = data.get("url") or content_text
        citation_key = data.get("citation_key")
        if not _is_url(url):
            url = None
            citation_key = citation_key or content_text
        sources.append(
            ReportContextSource(
                url=url,
                citation_key=citation_key.strip() if isinstance(citation_key, str) and citation_key.strip() else None,
                title=data.get("title") or event.get("name"),
                source_type=str(data.get("source_type") or "parent_report"),
                tool_name=str(data.get("tool_name") or "parent_report"),
            )
        )
    return sources


def _sources_section(report_markdown: str) -> str:
    match = _SOURCES_HEADING_RE.search(report_markdown)
    if not match:
        return ""
    rest = report_markdown[match.end() :]
    next_heading = _NEXT_HEADING_RE.search(rest)
    return rest[: next_heading.start()] if next_heading else rest


def _extract_sources_from_report_markdown(report_markdown: str) -> list[ReportContextSource]:
    section = _sources_section(report_markdown)
    if not section:
        return []
    sources: list[ReportContextSource] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        citation_match = _CITATION_LINE_RE.match(stripped)
        if not citation_match:
            continue
        ref_text = citation_match.group("text").strip()
        url_match = _URL_RE.search(ref_text)
        if url_match:
            sources.append(ReportContextSource(url=url_match.group(0).rstrip(_URL_TRIM_CHARS), title=ref_text))
        else:
            sources.append(ReportContextSource(citation_key=ref_text, title=ref_text))
    return sources


def _source_summary_markdown(sources: list[ReportContextSource]) -> str:
    if not sources:
        return "No durable source metadata was found for the parent report."
    lines = []
    for index, source in enumerate(sources, start=1):
        locator = source.url or source.citation_key or "(unknown source)"
        title = f"{source.title}: " if source.title and source.title != locator else ""
        lines.append(f"- [{index}] {title}{locator}")
    return "\n".join(lines)


async def resolve_report_context(job: Any, db_url: str, parent_job_id: str) -> ReportContext:
    """Build report context from a previously authorized parent job."""

    output = await read_job_output_async(parent_job_id, getattr(job, "output", None))
    report = _extract_report_from_output(output)
    # Durable events are only needed when the report isn't already in job output,
    # or to reconstruct sources. Fetch the (potentially large) event log at most once.
    events: list[dict[str, Any]] | None = None
    if not report:
        events = await EventStore.get_events_async(db_url, parent_job_id, 0, _EVENT_SCAN_LIMIT)
        report = _report_from_events(events)
    if not report:
        raise HTTPException(409, f"Parent job has no durable report: {parent_job_id}")

    report_sources = _extract_sources_from_report_markdown(report)

    if events is None:
        # The report came from job output. Its own ## Sources section is authoritative for
        # follow-up context, so only pay the (potentially large) event scan when the report
        # carries no inline sources. A transient fetch failure must not abort follow-up.
        if report_sources:
            sources = _dedupe_sources(report_sources)
        else:
            try:
                events = await EventStore.get_events_async(db_url, parent_job_id, 0, _EVENT_SCAN_LIMIT)
            except Exception:
                events = []
            sources = _dedupe_sources([*_sources_from_events(events), *report_sources])
    else:
        # The event log was already fetched to reconstruct the report; reuse it for sources.
        sources = _dedupe_sources([*_sources_from_events(events), *report_sources])

    return ReportContext(
        parent_job_id=parent_job_id,
        report_markdown=report,
        source_summary_markdown=_source_summary_markdown(sources),
        sources=sources,
    )


def report_context_from_markdown(report_markdown: str, parent_job_id: str = "in-session") -> ReportContext:
    """Build report context directly from report markdown — no job store, auth, or scheduler.

    Used for report follow-up in the synchronous CLI, where the report was produced inline in the
    current session rather than as a durable async job. Sources are reconstructed from the report's
    own ``## Sources`` section via the same parser used for job-backed reports.
    """
    sources = _dedupe_sources(_extract_sources_from_report_markdown(report_markdown))
    return ReportContext(
        parent_job_id=parent_job_id,
        report_markdown=report_markdown,
        source_summary_markdown=_source_summary_markdown(sources),
        sources=sources,
    )


# Cache JobStore instances by (scheduler_address, db_url). JobStore eagerly builds a SQLAlchemy
# AsyncEngine with its own connection pool, so constructing one per request both wastes work and
# leaks pooled connections. Engines are designed to be long-lived and shared, and JobStore's
# sessions are task-scoped (async_scoped_session), so a process-wide cache is safe.
_JOB_STORE_CACHE: dict[tuple[str, str], Any] = {}


def _get_job_store(scheduler_address: str, db_url: str) -> Any:
    from nat.front_ends.fastapi.async_jobs.job_store import JobStore

    key = (scheduler_address, db_url)
    store = _JOB_STORE_CACHE.get(key)
    if store is None:
        store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
        _JOB_STORE_CACHE[key] = store
    return store


async def resolve_authorized_report_context(parent_job_id: str, principal: Principal) -> ReportContext:
    """Authorize and reconstruct a parent report context using configured async job storage."""

    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

    scheduler_address = os.environ.get("NAT_DASK_SCHEDULER_ADDRESS")
    if not scheduler_address:
        raise HTTPException(503, "Async job storage is not configured")

    db_url = os.environ.get("NAT_JOB_STORE_DB_URL", "sqlite:///./data/jobs.db")
    job_store = _get_job_store(scheduler_address, db_url)
    job = await authorize_job_access(job_store, db_url, parent_job_id, principal)
    if getattr(job, "status", None) != JobStatus.SUCCESS.value:
        raise HTTPException(409, f"Parent job is not complete: {parent_job_id}")
    return await resolve_report_context(job, db_url, parent_job_id)


def to_initial_files(context: ReportContext, instruction: str | None = None) -> dict[str, str]:
    """Convert report context into DeepAgents virtual filesystem seed files."""

    # Exclude report_markdown from the JSON context: the full report is already seeded
    # verbatim into /shared/original_report.md, so embedding it again here would roughly
    # double the report tokens carried into the rewrite/research prompt for no benefit.
    files = {
        "/shared/original_report.md": context.report_markdown,
        "/shared/parent_report_context.json": context.model_dump_json(indent=2, exclude={"report_markdown"}),
        "/shared/source_summary.md": context.source_summary_markdown,
    }
    if instruction is not None:
        files["/shared/edit_instruction.txt"] = instruction
    return files


def report_output_metadata(parent_job_id: str, action: str) -> dict[str, str]:
    """Durable output metadata persisted with a report follow-up child job."""
    return {
        "parent_job_id": parent_job_id,
        "interaction_action": action,
        "result_kind": "report",
    }
