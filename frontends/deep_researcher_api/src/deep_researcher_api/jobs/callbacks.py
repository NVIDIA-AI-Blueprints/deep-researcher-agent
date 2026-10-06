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
LangChain callback handlers for SSE event streaming.

Uses NAT's IntermediateStep structure for consistent event types:
- Category: LLM, TOOL, WORKFLOW, ARTIFACT
- State: START, CHUNK, END, UPDATE

All verbs are present tense for consistency.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from deep_researcher_agent.common.callbacks import SUPPRESS_OUTPUT_ARTIFACT_TAG
from deep_researcher_agent.common.citation_verification import extract_http_urls
from deep_researcher_agent.common.citation_verification import get_session_registry

if TYPE_CHECKING:
    from .event_store import EventStore

logger = logging.getLogger(__name__)


class EventCategory(StrEnum):
    """Event categories aligned with NAT's IntermediateStepCategory."""

    LLM = "llm"
    TOOL = "tool"
    WORKFLOW = "workflow"
    ARTIFACT = "artifact"
    JOB = "job"


class EventState(StrEnum):
    """Event states aligned with NAT's IntermediateStepState. All present tense."""

    START = "start"
    CHUNK = "chunk"
    END = "end"
    UPDATE = "update"


class ArtifactType(StrEnum):
    """Types of artifacts that can be tracked via artifact.update events."""

    FILE = "file"
    OUTPUT = "output"
    CITATION_SOURCE = "citation_source"
    CITATION_USE = "citation_use"
    TODO = "todo"


class EventData(BaseModel):
    """Generic event data payload."""

    model_config = ConfigDict(extra="allow")

    input: Any | None = None
    output: Any | None = None
    chunk: Any | None = None


class IntermediateStepEvent(BaseModel):
    """
    SSE event structure aligned with NAT's IntermediateStep.

    This provides a consistent event format that frontends can consume
    without needing deep knowledge of specific tool/agent internals.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: EventCategory
    state: EventState
    name: str | None = None
    timestamp: float = Field(default_factory=time.time)
    data: EventData | None = None
    metadata: dict[str, Any] | None = None

    @property
    def event_type(self) -> str:
        """Returns event type string for SSE: category.state"""
        return f"{self.category.value}.{self.state.value}"

    def to_sse_dict(self) -> dict:
        """Convert to dict for SSE transmission."""
        result = {
            "type": self.event_type,
            "id": self.id,
            "name": self.name,
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=UTC).isoformat(),
        }
        if self.data:
            result["data"] = self.data.model_dump(exclude_none=True)
        if self.metadata:
            result["metadata"] = self.metadata
        return {k: v for k, v in result.items() if v is not None}


def build_final_report_event(content: str, *, cited_urls: Sequence[str] | None = None) -> IntermediateStepEvent:
    """Build the canonical final-report artifact event."""
    data: dict[str, Any] = {
        "type": ArtifactType.OUTPUT.value,
        "content": content,
        "output_category": "final_report",
    }
    if cited_urls is not None:
        data["cited_urls"] = list(cited_urls)
    return IntermediateStepEvent(
        category=EventCategory.ARTIFACT,
        state=EventState.UPDATE,
        data=EventData(**data),
    )


class ToolArtifactMapping:
    """
    Maps tool names to artifact types for automatic artifact emission.

    This provides a generic way to handle middleware-managed tools and
    any other tools that produce trackable artifacts. New mappings can
    be added without modifying the callback handler logic.
    """

    def __init__(self):
        """Initialize an empty tool-to-artifact mapping and register the defaults."""
        self._mappings: dict[str, dict] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default mappings for deepagents middleware tools."""
        self.register(
            "write_todos",
            artifact_type=ArtifactType.TODO,
            content_key="todos",
        )
        self.register(
            "write_file",
            artifact_type=ArtifactType.FILE,
            content_key="content",
            name_key="file_path",
            extra_keys=["file_path", "path", "filename"],
        )

    def register(
        self,
        tool_name: str,
        artifact_type: ArtifactType,
        content_key: str | None = None,
        name_key: str | None = None,
        extra_keys: list[str] | None = None,
    ):
        """
        Register a tool-to-artifact mapping.

        Args:
            tool_name: Name of the tool (case-insensitive matching)
            artifact_type: Type of artifact to emit
            content_key: Key in tool input containing the content
            name_key: Key in tool input containing the artifact name
            extra_keys: Additional keys to include in artifact data
        """
        self._mappings[tool_name.lower()] = {
            "artifact_type": artifact_type,
            "content_key": content_key,
            "name_key": name_key,
            "extra_keys": extra_keys or [],
        }

    def get_mapping(self, tool_name: str) -> dict | None:
        """Get mapping for a tool, or None if not registered."""
        return self._mappings.get(tool_name.lower())

    def is_artifact_tool(self, tool_name: str) -> bool:
        """Check if a tool produces artifacts."""
        return tool_name.lower() in self._mappings


class AgentEventCallback(BaseCallbackHandler):
    """
    Callback handler that emits NAT-aligned IntermediateStep events for SSE streaming.

    Event model:
    - workflow.start/end: Agent execution boundaries (uses LangChain run_id as agent_id)
    - llm.start/chunk/end: LLM inference
    - tool.start/end: Tool invocations
    - artifact.update: Structured artifacts (files, output, citations, todos)

    Agent tracking:
    - Uses LangChain's run_id as the unique agent identifier
    - Tracks parent_run_id hierarchy to handle parallel agent execution
    - Each operation traces back to its owning agent via parent chain
    """

    OUTPUT_MIN_LENGTH = 200
    SEARCH_TOOL_PATTERNS = {"search", "tavily", "web_search", "google", "bing"}
    TOOL_CALL_PATTERN = re.compile(r'\b[a-z][a-z0-9_]*\s*\(\s*(?:["\'{]|[a-z_]+\s*=)', re.IGNORECASE)
    SANDBOX_EXEC_TOOLS = frozenset({"execute"})
    SANDBOX_FILE_TOOLS = frozenset({"write_file", "read_file", "edit_file", "ls"})
    SHARED_FS_PREFIX = "/shared"

    AGENT_PATTERNS = {"agent"}
    AGENT_EXCLUDE_PATTERNS = {"middleware", "handler", "callback"}

    _job_discovered_urls: dict[str, set[str]] = {}
    _job_cited_urls: dict[str, set[str]] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        event_store: EventStore | None = None,
        tool_artifact_mapping: ToolArtifactMapping | None = None,
    ):
        """Wire the event store and tool/artifact mapping and init per-job URL caches.

        Args:
            event_store: Sink for SSE events; None disables emission.
            tool_artifact_mapping: Mapping of tools to artifact types; a default is used when omitted.
        """
        super().__init__()
        self._event_store = event_store
        self._tool_mapping = tool_artifact_mapping or ToolArtifactMapping()

        self._run_id_to_name: dict[str, str] = {}
        self._run_id_to_parent: dict[str, str] = {}
        self._agent_run_ids: dict[str, str] = {}  # {run_id: name}
        self._run_id_to_sandbox_tool: dict[str, bool] = {}

        self._job_id = event_store.job_id if event_store else None
        self._instance_discovered_urls: set[str] = set()
        self._instance_cited_urls: set[str] = set()
        self._init_job_url_sets()

    def _init_job_url_sets(self) -> None:
        """Initialize URL sets for this job."""
        if not self._job_id:
            return
        with AgentEventCallback._cache_lock:
            if self._job_id not in AgentEventCallback._job_discovered_urls:
                AgentEventCallback._job_discovered_urls[self._job_id] = set()
            if self._job_id not in AgentEventCallback._job_cited_urls:
                AgentEventCallback._job_cited_urls[self._job_id] = set()

    @property
    def _discovered_urls(self) -> set[str]:
        """Get discovered URLs set for this job."""
        if self._job_id and self._job_id in AgentEventCallback._job_discovered_urls:
            return AgentEventCallback._job_discovered_urls[self._job_id]
        return self._instance_discovered_urls

    @property
    def _cited_urls(self) -> set[str]:
        """Get cited URLs set for this job."""
        if self._job_id and self._job_id in AgentEventCallback._job_cited_urls:
            return AgentEventCallback._job_cited_urls[self._job_id]
        return self._instance_cited_urls

    @classmethod
    def cleanup_job_urls(cls, job_id: str) -> None:
        """Clean up URL caches for a completed job."""
        with cls._cache_lock:
            cls._job_discovered_urls.pop(job_id, None)
            cls._job_cited_urls.pop(job_id, None)

    def _is_agent_like(self, name: str) -> bool:
        """Check if name represents an agent (contains 'agent' but not middleware/handler)."""
        name_lower = name.lower()
        if any(exclude in name_lower for exclude in self.AGENT_EXCLUDE_PATTERNS):
            return False
        return any(pattern in name_lower for pattern in self.AGENT_PATTERNS)

    def _find_agent_for_run(self, run_id: str) -> tuple[str, str] | None:
        """
        Find the agent that owns a given run_id by traversing the parent chain.
        Returns (agent_name, agent_run_id) or None if no agent found.
        """
        current = run_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            if current in self._agent_run_ids:
                return (self._agent_run_ids[current], current)
            current = self._run_id_to_parent.get(current)
        return None

    def _build_metadata_for_run(self, run_id: str, **extra: Any) -> dict[str, Any] | None:
        """Build metadata with agent context for a specific run_id."""
        metadata: dict[str, Any] = {}

        agent_info = self._find_agent_for_run(run_id)
        if agent_info:
            metadata["workflow"] = agent_info[0]
            metadata["agent_id"] = agent_info[1]

        metadata.update({k: v for k, v in extra.items() if v is not None})
        return metadata if metadata else None

    def _emit(self, event: IntermediateStepEvent):
        """Store an event as an SSE dict when an event store is configured."""
        if self._event_store:
            self._event_store.store(event.to_sse_dict())

    def _emit_artifact(
        self,
        artifact_type: ArtifactType,
        content: Any,
        name: str | None = None,
        **extra_data,
    ):
        """
        Emit an artifact.update event for tracking structured outputs.

        Args:
            artifact_type: Type of artifact (file, output, citation_source, etc.)
            content: The artifact content
            name: Optional artifact name (e.g., filename, URL)
            **extra_data: Additional type-specific fields (workflow, agent_id, etc.)
        """
        workflow = extra_data.pop("workflow_source", None) or extra_data.get("workflow")
        agent_id = extra_data.pop("agent_id", None)
        data = {"type": artifact_type.value, "content": content, **extra_data}

        metadata: dict[str, Any] | None = None
        if workflow or agent_id:
            metadata = {}
            if workflow:
                metadata["workflow"] = workflow
            if agent_id:
                metadata["agent_id"] = agent_id

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.ARTIFACT,
                state=EventState.UPDATE,
                name=name,
                data=EventData(**data),
                metadata=metadata,
            )
        )

    def _get_chain_name(self, serialized: dict | None, **kwargs) -> str:
        """Resolve a human-readable chain name from serialized data or kwargs."""
        if serialized:
            name = serialized.get("name") or serialized.get("id", [""])[-1]
            if name:
                return name
        return kwargs.get("name", "unknown")

    def _is_sandbox_tool(self, tool_name: str, parsed_input: Any) -> bool:
        """Return whether a tool call runs against the provisioned sandbox."""
        if tool_name in self.SANDBOX_EXEC_TOOLS:
            return True
        if tool_name not in self.SANDBOX_FILE_TOOLS or not isinstance(parsed_input, dict):
            return False
        path = str(parsed_input.get("file_path") or parsed_input.get("path") or parsed_input.get("filename") or "")
        in_shared = path == self.SHARED_FS_PREFIX or path.startswith(self.SHARED_FS_PREFIX + "/")
        return bool(path and not in_shared)

    def _get_source_registry(self):
        """Return the session-scoped SourceRegistry if set, otherwise None."""
        return get_session_registry()

    def emit_final_report(self, content: str, *, cited_urls: Sequence[str] | None = None) -> None:
        """Emit the post-processed final report as an OUTPUT artifact.

        Call this after citation verification and sanitisation so the
        frontend receives the verified content (overwrites the earlier
        auto-emitted version). ``cited_urls`` must come from citation
        verification; when supplied, it is the authoritative citation set for
        the published report.
        """
        verified_visible_urls = self._verified_visible_urls(content, cited_urls) if cited_urls is not None else None
        self._emit(build_final_report_event(content, cited_urls=verified_visible_urls))
        if verified_visible_urls is not None:
            for url in verified_visible_urls:
                self._cited_urls.add(self._normalize_url(url))
                self._emit_artifact(
                    ArtifactType.CITATION_USE,
                    url,
                    name=url,
                    url=url,
                    final_report=True,
                )

    def _is_search_tool(self, tool_name: str) -> bool:
        """Check if tool is a search-related tool that returns URLs."""
        tool_lower = tool_name.lower()
        return any(pattern in tool_lower for pattern in self.SEARCH_TOOL_PATTERNS)

    def _contains_tool_call_syntax(self, content: str) -> bool:
        """
        Check if content contains tool call syntax that shouldn't be emitted as output.

        This prevents the LLM's text-based tool call attempts from being captured
        as output artifacts when the model outputs function call syntax as text
        instead of properly invoking the tool.
        """
        if not content:
            return False
        return bool(self.TOOL_CALL_PATTERN.search(content))

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for consistent matching.

        Handles:
        - Trailing slashes
        - Case normalization for domain
        - Query parameter sorting (basic)
        """
        from urllib.parse import urlparse
        from urllib.parse import urlunparse

        try:
            parsed = urlparse(url)
            # Normalize: lowercase scheme and netloc, remove trailing slash from path
            normalized_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
            normalized = urlunparse(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    normalized_path,
                    parsed.params,
                    parsed.query,
                    "",  # Remove fragment
                )
            )
            return normalized
        except Exception:
            return url

    def _extract_urls(self, text: str) -> list[str]:
        """Extract unique URLs from text content."""
        return [url for url in extract_http_urls(text) if len(url) > 10 and "." in url]

    def _verified_visible_urls(self, content: str, cited_urls: Sequence[str]) -> list[str]:
        """Keep only verified citation URLs that remain visible in the final report."""
        visible_urls = {self._normalize_url(url) for url in self._extract_urls(content)}
        result: list[str] = []
        seen: set[str] = set()
        for url in cited_urls:
            normalized = self._normalize_url(url)
            if normalized in visible_urls and normalized not in seen:
                seen.add(normalized)
                result.append(url)
        return result

    def _get_output_category(self, agent_info: tuple[str, str] | None = None) -> str:
        """
        Determine output category based on workflow.

        Args:
            agent_info: Optional (name, run_id) tuple for the owning agent

        Returns:
            - "research_notes" for researcher-agent outputs
            - "draft" for orchestrator or unattributed outputs
        """
        workflow_name = agent_info[0] if agent_info else None
        if not workflow_name:
            return "draft"
        workflow_lower = workflow_name.lower()
        if "researcher" in workflow_lower or "research" in workflow_lower:
            return "research_notes"
        if "orchestrator" in workflow_lower:
            return "draft"
        return "intermediate"

    def _emit_cited_urls(self, content: str) -> None:
        """Extract URLs from output content and emit citation_use events.

        When a SourceRegistry is attached, validates against it (consistent
        with verify_citations). Otherwise falls back to the callback's own
        _discovered_urls set.
        """
        urls = self._extract_urls(content)
        for url in urls:
            normalized = self._normalize_url(url)
            if normalized in self._cited_urls:
                continue

            is_valid = False
            registry = self._get_source_registry()
            if registry is not None:
                is_valid = registry.has_url(url)
            else:
                is_valid = normalized in self._discovered_urls

            if is_valid:
                self._cited_urls.add(normalized)
                self._emit_artifact(
                    ArtifactType.CITATION_USE,
                    url,
                    name=url,
                    url=url,
                )

    def _emit_tool_artifact(self, tool_name: str, tool_input: Any, run_id: str = "") -> None:
        """
        Emit artifacts for tools registered in the mapping.

        This provides generic handling for middleware-managed tools and any
        other tools that produce trackable artifacts. The mapping determines
        how to extract content and metadata from the tool input.
        """
        mapping = self._tool_mapping.get_mapping(tool_name)
        if not mapping:
            return

        artifact_type = mapping["artifact_type"]
        content_key = mapping["content_key"]
        name_key = mapping["name_key"]
        extra_keys = mapping["extra_keys"]

        content = None
        name = None
        extra_data = {}

        if isinstance(tool_input, dict):
            if content_key:
                content = tool_input.get(content_key)
            if name_key:
                name = tool_input.get(name_key)
            for key in extra_keys:
                if key in tool_input:
                    extra_data[key] = tool_input[key]
                    if not name and key in ("file_path", "path", "filename", "name"):
                        name = tool_input[key]
        elif isinstance(tool_input, list) and content_key == "todos":
            content = tool_input

        if content is not None:
            # For todo artifacts, include source context so the UI can
            # distinguish orchestrator-level todos from sub-agent todos.
            if artifact_type == ArtifactType.TODO and run_id:
                agent_info = self._find_agent_for_run(run_id)
                if agent_info:
                    extra_data["workflow"] = agent_info[0]
                    extra_data["agent_id"] = agent_info[1]
                    extra_data["source"] = "agent"
                else:
                    extra_data["source"] = "orchestrator"
            self._emit_artifact(artifact_type, content, name=name, **extra_data)

    def on_chain_start(self, serialized: dict | None, inputs: dict, **kwargs) -> None:
        """Track the run/parent lineage and emit an agent.start event for agent-like chains."""
        name = self._get_chain_name(serialized, **kwargs)
        run_id = str(kwargs.get("run_id", ""))
        parent_run_id = str(kwargs.get("parent_run_id", "")) if kwargs.get("parent_run_id") else ""

        if run_id:
            self._run_id_to_name[run_id] = name
            if parent_run_id:
                self._run_id_to_parent[run_id] = parent_run_id

        if self._is_agent_like(name):
            self._agent_run_ids[run_id] = name

            self._emit(
                IntermediateStepEvent(
                    category=EventCategory.WORKFLOW,
                    state=EventState.START,
                    name=name,
                    data=EventData(input=self._extract_input(inputs)),
                    metadata=self._build_metadata_for_run(run_id),
                )
            )

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        """Emit an agent.end event for agent-like chains and clear run bookkeeping."""
        run_id = str(kwargs.get("run_id", ""))
        name = self._run_id_to_name.pop(run_id, kwargs.get("name", ""))

        if self._is_agent_like(name):
            self._emit(
                IntermediateStepEvent(
                    category=EventCategory.WORKFLOW,
                    state=EventState.END,
                    name=name,
                    data=EventData(output=self._extract_output(outputs)),
                    metadata=self._build_metadata_for_run(run_id),
                )
            )

            if run_id in self._agent_run_ids:
                del self._agent_run_ids[run_id]

        self._run_id_to_parent.pop(run_id, None)

    def on_chain_error(self, error: BaseException, **kwargs) -> None:
        """Emit a retry hint when a chain (typically an LLM call) fails."""
        run_id = str(kwargs.get("run_id", ""))
        name = self._run_id_to_name.get(run_id, "unknown")

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.JOB,
                state=EventState.UPDATE,
                name="retry",
                data=EventData(output=f"Retrying {name}: {type(error).__name__}"),
                metadata=self._build_metadata_for_run(run_id),
            )
        )

    TOOL_INPUT_TRIM_LIMIT = 500

    def _trim_tool_input(self, parsed_input: Any) -> Any:
        """Trim tool input to a reasonable size for SSE streaming."""
        if parsed_input is None:
            return None
        serialized = str(parsed_input)
        if len(serialized) <= self.TOOL_INPUT_TRIM_LIMIT:
            return parsed_input
        if isinstance(parsed_input, str):
            return parsed_input[: self.TOOL_INPUT_TRIM_LIMIT] + "..."
        return serialized[: self.TOOL_INPUT_TRIM_LIMIT] + "..."

    def on_tool_start(self, serialized: dict | None, input_str: str, **kwargs) -> None:
        """Emit a tool.start event and record sandbox/lineage state for the tool run."""
        tool_name = serialized.get("name", "unknown") if serialized else "unknown"
        run_id = str(kwargs.get("run_id", ""))
        parent_run_id = str(kwargs.get("parent_run_id", "")) if kwargs.get("parent_run_id") else ""

        if run_id:
            self._run_id_to_name[run_id] = tool_name
            if parent_run_id:
                self._run_id_to_parent[run_id] = parent_run_id

        parsed_input = self._parse_tool_input(input_str)

        emit_input = self._trim_tool_input(parsed_input)
        is_sandbox_tool = self._is_sandbox_tool(tool_name, parsed_input)
        if run_id:
            self._run_id_to_sandbox_tool[run_id] = is_sandbox_tool

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.TOOL,
                state=EventState.START,
                name=tool_name,
                data=EventData(input=emit_input) if emit_input else None,
                metadata=self._build_metadata_for_run(run_id, sandbox=True if is_sandbox_tool else None),
            )
        )

        self._emit_tool_artifact(tool_name, parsed_input, run_id=run_id)

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Emit a tool.end event and clear the tool run's lineage/sandbox state."""
        run_id = str(kwargs.get("run_id", ""))
        tool_name = self._run_id_to_name.pop(run_id, kwargs.get("name", "unknown"))
        is_sandbox_tool = self._run_id_to_sandbox_tool.pop(run_id, False)

        agent_info = self._find_agent_for_run(run_id)

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.TOOL,
                state=EventState.END,
                name=tool_name,
                data=None,
                metadata=self._build_metadata_for_run(run_id, sandbox=True if is_sandbox_tool else None),
            )
        )

        if self._is_search_tool(tool_name) and output:
            urls = self._extract_urls(str(output))
            for url in urls:
                normalized = self._normalize_url(url)
                if normalized not in self._discovered_urls:
                    self._discovered_urls.add(normalized)
                    self._emit_artifact(
                        ArtifactType.CITATION_SOURCE,
                        url,
                        name=url,
                        url=url,
                        tool=tool_name,
                        agent_id=agent_info[1] if agent_info else None,
                        workflow=agent_info[0] if agent_info else None,
                    )

        self._run_id_to_parent.pop(run_id, None)

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        """Emit an llm.start event for a completion-style model call."""
        model_name = "unknown"
        if serialized:
            model_name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]

        run_id = str(kwargs.get("run_id", ""))
        parent_run_id = str(kwargs.get("parent_run_id", "")) if kwargs.get("parent_run_id") else ""

        if run_id:
            self._run_id_to_name[run_id] = model_name
            if parent_run_id:
                self._run_id_to_parent[run_id] = parent_run_id

        metadata = self._build_metadata_for_run(run_id) or {}
        metadata["prompt_count"] = len(prompts) if prompts else 0

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.LLM,
                state=EventState.START,
                name=model_name,
                metadata=metadata if metadata else None,
            )
        )

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Emit a streaming llm.chunk event for each non-empty token."""
        if token and SUPPRESS_OUTPUT_ARTIFACT_TAG not in (kwargs.get("tags") or []):
            self._emit(
                IntermediateStepEvent(
                    category=EventCategory.LLM,
                    state=EventState.CHUNK,
                    data=EventData(chunk=token),
                )
            )

    THINKING_TRIM_LIMIT = 300
    THINKING_TRIM_SUFFIX = " [Trimmed - check traces for full logs]"

    def on_llm_end(self, response, **kwargs) -> None:
        """Emit an llm.end event with content, thinking, and token usage."""
        run_id = str(kwargs.get("run_id", ""))
        model_name = self._run_id_to_name.pop(run_id, "unknown")

        content, thinking, usage, has_tool_calls = self._extract_llm_response(response)

        agent_info = self._find_agent_for_run(run_id)
        metadata = self._build_metadata_for_run(run_id) or {}
        if thinking:
            if len(thinking) > self.THINKING_TRIM_LIMIT:
                metadata["thinking"] = thinking[: self.THINKING_TRIM_LIMIT] + self.THINKING_TRIM_SUFFIX
            else:
                metadata["thinking"] = thinking
        if usage:
            metadata["usage"] = usage

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.LLM,
                state=EventState.END,
                name=model_name,
                data=None,
                metadata=metadata if any(v for v in metadata.values()) else None,
            )
        )

        if (
            content
            and len(content) >= self.OUTPUT_MIN_LENGTH
            and not has_tool_calls
            and not self._contains_tool_call_syntax(content)
            and SUPPRESS_OUTPUT_ARTIFACT_TAG not in (kwargs.get("tags") or [])
        ):
            output_category = self._get_output_category(agent_info)
            self._emit_artifact(
                ArtifactType.OUTPUT,
                content,
                output_category=output_category,
                workflow_source=agent_info[0] if agent_info else None,
                agent_id=agent_info[1] if agent_info else None,
            )
            self._emit_cited_urls(content)

        self._run_id_to_parent.pop(run_id, None)

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs) -> None:
        """Emit an llm.start event for a chat-model call."""
        model_name = "unknown"
        if serialized:
            model_name = serialized.get("name") or serialized.get("kwargs", {}).get("model", "unknown")

        run_id = str(kwargs.get("run_id", ""))
        parent_run_id = str(kwargs.get("parent_run_id", "")) if kwargs.get("parent_run_id") else ""

        if run_id:
            self._run_id_to_name[run_id] = model_name
            if parent_run_id:
                self._run_id_to_parent[run_id] = parent_run_id

        metadata = self._build_metadata_for_run(run_id) or {}
        metadata["message_count"] = len(messages) if messages else 0

        self._emit(
            IntermediateStepEvent(
                category=EventCategory.LLM,
                state=EventState.START,
                name=model_name,
                metadata=metadata if metadata else None,
            )
        )

    def _extract_input(self, inputs: dict | Any) -> Any:
        """Extract input data without preprocessing. Frontend handles display."""
        if not inputs:
            return None
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, dict):
            if "input" in inputs:
                return inputs["input"]
            if "messages" in inputs and inputs["messages"]:
                msg = inputs["messages"][-1]
                if hasattr(msg, "content"):
                    return msg.content
            return inputs
        return inputs

    def _extract_output(self, outputs: Any) -> Any:
        """Extract output data without preprocessing. Frontend handles display."""
        if outputs is None:
            return None
        if isinstance(outputs, str):
            return outputs
        if isinstance(outputs, dict):
            for key in ("output", "result", "answer", "response"):
                if key in outputs:
                    return outputs[key]
            if "messages" in outputs and outputs["messages"]:
                msg = outputs["messages"][-1]
                if hasattr(msg, "content"):
                    return msg.content
            return outputs
        return outputs

    def _parse_tool_input(self, input_str: str) -> dict | str:
        """Parse tool input string to structured data if possible."""
        import ast

        if not input_str:
            return ""
        try:
            return ast.literal_eval(input_str)
        except Exception:
            return input_str

    def _extract_tool_output(self, output: Any) -> Any:
        """Extract tool output without preprocessing. Frontend handles display."""
        if output is None:
            return None
        if isinstance(output, str):
            return output
        if hasattr(output, "update") and isinstance(output.update, dict):
            messages = output.update.get("messages", [])
            if messages and hasattr(messages[-1], "content"):
                return messages[-1].content
        if hasattr(output, "content"):
            return output.content
        return str(output)

    def _extract_llm_response(self, response) -> tuple[str | None, str | None, dict | None, bool]:
        """
        Extract content, thinking, usage, and tool_call flag from LLM response.

        Returns raw data - let frontend decide how to display it.
        The has_tool_calls flag indicates if the response triggered tool execution.
        """
        content = None
        thinking = None
        usage = None
        has_tool_calls = False

        try:
            if not hasattr(response, "generations") or not response.generations:
                return content, thinking, usage, has_tool_calls

            gen = response.generations[0][0] if response.generations[0] else None
            if not gen:
                return content, thinking, usage, has_tool_calls

            msg = getattr(gen, "message", None)
            if not msg and hasattr(gen, "generation_info"):
                msg = gen.generation_info.get("message") if gen.generation_info else None

            if msg:
                if hasattr(msg, "content"):
                    content = str(msg.content) if msg.content else None

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    has_tool_calls = True
                elif hasattr(msg, "additional_kwargs"):
                    tool_calls = msg.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        has_tool_calls = True

                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                    thinking = (
                        msg.additional_kwargs.get("reasoning_content")
                        or msg.additional_kwargs.get("thinking")
                        or msg.additional_kwargs.get("reasoning")
                    )

                if not thinking and hasattr(msg, "response_metadata") and msg.response_metadata:
                    thinking = msg.response_metadata.get("reasoning_content") or msg.response_metadata.get("thinking")

                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    usage = dict(msg.usage_metadata)
                elif hasattr(msg, "response_metadata") and msg.response_metadata:
                    token_usage = msg.response_metadata.get("token_usage")
                    if token_usage:
                        usage = dict(token_usage)

            elif hasattr(gen, "text"):
                content = gen.text

        except Exception as e:
            logger.debug("Error extracting LLM response: %s", e)

        return content, thinking, usage, has_tool_calls


# Alias for backwards compatibility
DeepResearchEventCallback = AgentEventCallback
