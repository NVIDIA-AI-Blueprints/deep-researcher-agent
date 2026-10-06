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

"""Custom middleware for the deep research agent."""

import asyncio
import hashlib
import json
import logging
import posixpath
import re
import threading
from pathlib import Path
from pathlib import PurePosixPath

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import hook_config
from langchain.agents.middleware.types import ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from pydantic import BaseModel
from pydantic import ValidationError

from deep_researcher_agent.common import get_source_id_for_tool
from deep_researcher_agent.common import load_prompt
from deep_researcher_agent.common import render_prompt_template
from deep_researcher_agent.common.citation_verification import SourceEntry
from deep_researcher_agent.common.citation_verification import SourceRegistry
from deep_researcher_agent.common.citation_verification import extract_sources_from_tool_result
from deep_researcher_agent.common.citation_verification import is_non_citable_status_output
from deep_researcher_agent.common.logging_utils import log_content_metadata

from .models import ResearchNotes
from .resource_limits import DeepResearchResourceLimits
from .resource_limits import StateBudgetLedger

logger = logging.getLogger(__name__)

# Path to this agent's prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SOURCE_ROUTING_PATH = "/shared/source_routing.json"
# When a sandbox provider is configured, CompositeBackend strips the /shared/ route
# before delegating to StateBackend, so the router's file is stored under the
# route-local key. The guard reads raw state, so it must accept both forms or it
# blocks the orchestrator forever on sandboxed runs.
_SOURCE_ROUTING_STATE_KEYS = (_SOURCE_ROUTING_PATH, "/source_routing.json")
FINAL_REPORT_PATH = "/shared/output.md"
FINAL_REPORT_STATE_PATHS = (FINAL_REPORT_PATH, "/output.md")
_GENERATED_RETRY_MARKER = "deep_researcher_generated_retry"
_RESEARCHER_FINALIZATION_MARKER = "deep_researcher_researcher_finalization"
RESEARCHER_FINALIZATION_MODEL_CALLS = 1
_RESEARCHER_FINALIZATION_PROMPT = (
    "Your research model-call budget is exhausted. Do not call tools or continue researching. "
    "Return your ResearchNotes now using only the existing conversation and tool-result history. "
    "Preserve useful findings and sources gathered so far, identify unresolved gaps, and lower the "
    "evidence confidence when support is incomplete."
)
_UNRESOLVED_SANDBOX_PATH_PATTERN = re.compile(
    r"<\s*sandbox_(?:artifact_dir|workdir)\s*>|\{\{\s*sandbox_(?:artifact_dir|workdir)\s*\}\}"
)


def _normalized_virtual_path(path: object) -> str | None:
    """Return a canonical virtual path without weakening backend validation."""
    if not isinstance(path, str) or not path:
        return None
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return normalized if normalized.startswith("/") else f"/{normalized}"


def _tool_file_path(tool_call: object) -> str | None:
    """Read and normalize a filesystem tool's target path."""
    if not isinstance(tool_call, dict):
        return None
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return None
    return _normalized_virtual_path(args.get("file_path", args.get("path")))


def _entry_text(entry: object) -> str | None:
    """Read exact text from a DeepAgents state-file entry."""
    if isinstance(entry, dict):
        entry = entry.get("content")
    if isinstance(entry, bytes):
        try:
            return entry.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(entry, str):
        return entry
    return None


def _tool_result_failed(result: object) -> bool:
    """Return whether a filesystem tool result reports an error."""
    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
    return status == "error"


class StructuredResponseTextFallbackMiddleware(AgentMiddleware):
    """Recover one exact JSON response when a provider skips the output tool."""

    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"), ensure_ascii=False)
        self._correction = (
            "The previous response did not produce the required structured result. Do not call tools. "
            "Return exactly one JSON object matching this JSON Schema, with no Markdown fences or prose:\n"
            f"{schema_json}"
        )

    def _promote(self, response: ModelResponse) -> ModelResponse:
        if response.structured_response is not None or len(response.result) != 1:
            return response
        message = response.result[0]
        if not isinstance(message, AIMessage) or message.tool_calls or not isinstance(message.content, str):
            return response
        try:
            structured = self.schema.model_validate_json(message.content)
        except ValidationError:
            return response
        logger.info("Recovered %s from schema-valid JSON message content", self.schema.__name__)
        return ModelResponse(result=response.result, structured_response=structured)

    @staticmethod
    def _needs_correction(response: ModelResponse) -> bool:
        if response.structured_response is not None or len(response.result) != 1:
            return False
        message = response.result[0]
        return isinstance(message, AIMessage) and not message.tool_calls and isinstance(message.content, str)

    def _correction_request(self, request):
        return request.override(
            messages=[*request.messages, HumanMessage(content=self._correction)],
            tools=[],
            tool_choice=None,
            response_format=None,
        )

    def wrap_model_call(self, request, handler):
        """Promote JSON text, with one tools-disabled corrective call when needed."""
        response = self._promote(handler(request))
        if not self._needs_correction(response) or _is_researcher_finalization_request(request):
            return response
        logger.warning("Retrying %s as a tools-disabled JSON response", self.schema.__name__)
        return self._promote(handler(self._correction_request(request)))

    async def awrap_model_call(self, request, handler):
        """Promote JSON text, with one tools-disabled corrective call when needed."""
        response = self._promote(await handler(request))
        if not self._needs_correction(response) or _is_researcher_finalization_request(request):
            return response
        logger.warning("Retrying %s as a tools-disabled JSON response", self.schema.__name__)
        return self._promote(await handler(self._correction_request(request)))


def _is_researcher_finalization_request(request) -> bool:
    """Return whether this request is the researcher's single reserved finalization turn."""
    return bool(
        request.messages
        and isinstance(request.messages[-1], HumanMessage)
        and request.messages[-1].additional_kwargs.get(_RESEARCHER_FINALIZATION_MARKER)
    )


class ResearcherBudgetExhaustedError(Exception):
    """Raised when the reserved researcher finalization turn does not produce notes."""

    def __init__(self, model_calls: int, max_model_calls: int) -> None:
        self.model_calls = model_calls
        self.max_model_calls = max_model_calls
        super().__init__(f"Researcher exhausted its {max_model_calls}-model-call budget after {model_calls} calls")


class ResearcherFinalizationMiddleware(AgentMiddleware):
    """Reserve one tools-disabled model turn for finalizing partial research.

    ``ModelCallLimitMiddleware`` owns ``run_model_call_count``; the two middleware
    must be installed as a pair.
    """

    def __init__(self, *, max_model_calls: int) -> None:
        self.max_model_calls = max_model_calls

    def _request(self, request):
        """Return the original request until the budget binds, then force finalization."""
        calls_made = request.state.get("run_model_call_count", 0)
        if calls_made < self.max_model_calls:
            return request
        logger.warning(
            "Researcher exhausted its %d-model-call budget after %d calls; entering finalization",
            self.max_model_calls,
            calls_made,
        )
        finalization_message = HumanMessage(
            content=_RESEARCHER_FINALIZATION_PROMPT,
            additional_kwargs={_RESEARCHER_FINALIZATION_MARKER: True},
        )
        return request.override(
            messages=[*request.messages, finalization_message],
            tools=[],
            tool_choice=None,
            response_format=ToolStrategy(ResearchNotes),
        )

    @staticmethod
    def _finalized(response: ModelResponse) -> bool:
        """Return whether the model produced schema-valid research notes."""
        return response.structured_response is not None

    @staticmethod
    def _has_tool_calls(response: ModelResponse) -> bool:
        """Return whether finalization attempted a tool call instead of returning notes."""
        return any(isinstance(message, AIMessage) and message.tool_calls for message in response.result)

    def _result(self, request, response: ModelResponse) -> ModelResponse:
        """Accept notes or tool calls, and type prose-only exhaustion for fallback handling."""
        if not _is_researcher_finalization_request(request) or self._finalized(response):
            return response
        if self._has_tool_calls(response):
            return response
        calls_made = request.state.get("run_model_call_count", 0)
        raise ResearcherBudgetExhaustedError(calls_made + RESEARCHER_FINALIZATION_MODEL_CALLS, self.max_model_calls)

    @staticmethod
    def _refused(request) -> ToolMessage:
        """Build the shared result for a tool call refused after finalization."""
        return ToolMessage(
            content="Researcher model-call budget exhausted; this tool was not executed.",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call.get("name"),
            status="error",
        )

    def wrap_model_call(self, request, handler):
        """Force one synchronous finalization call after the normal-turn budget."""
        finalization_request = self._request(request)
        return self._result(finalization_request, handler(finalization_request))

    async def awrap_model_call(self, request, handler):
        """Force one asynchronous finalization call after the normal-turn budget."""
        finalization_request = self._request(request)
        return self._result(finalization_request, await handler(finalization_request))

    def wrap_tool_call(self, request, handler):
        """Prevent a hallucinated tool call from executing after finalization."""
        if request.state.get("run_model_call_count", 0) <= self.max_model_calls:
            return handler(request)
        return self._refused(request)

    async def awrap_tool_call(self, request, handler):
        """Prevent a hallucinated tool call from executing after async finalization."""
        if request.state.get("run_model_call_count", 0) <= self.max_model_calls:
            return await handler(request)
        return self._refused(request)


class FinalReportCommitTracker:
    """Run-local proof of the writer's most recent successful report mutation."""

    def __init__(self) -> None:
        self._digest: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _digest_text(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def record(self, content: str) -> str:
        """Record the exact UTF-8 digest after a successful writer mutation."""
        digest = self._digest_text(content)
        with self._lock:
            self._digest = digest
        return digest

    @property
    def digest(self) -> str | None:
        """Return the most recently committed digest for this run."""
        with self._lock:
            return self._digest

    def committed_text(
        self,
        files: object,
        *,
        paths: tuple[str, ...] = FINAL_REPORT_STATE_PATHS,
    ) -> str | None:
        """Return the non-empty state file matching the writer's exact digest."""
        digest = self.digest
        if digest is None or not isinstance(files, dict):
            return None
        for path in paths:
            content = _entry_text(files.get(path))
            if content is not None and content.strip() and self._digest_text(content) == digest:
                return content
        return None


class SourceRoutingGuardMiddleware(AgentMiddleware):
    """Require the source-router handoff before other orchestrator tool calls."""

    def __init__(self, *, enabled: bool, required_subagent: str = "source-router-agent") -> None:
        self.enabled = enabled
        self.required_subagent = required_subagent

    @staticmethod
    def _routing_complete(state: object) -> bool:
        files = state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})
        return isinstance(files, dict) and any(key in files for key in _SOURCE_ROUTING_STATE_KEYS)

    async def awrap_tool_call(self, request, handler):
        """Block out-of-order calls until the source router writes its route file."""
        if not self.enabled or self._routing_complete(request.state):
            return await handler(request)

        tool_call = request.tool_call
        args = tool_call.get("args") or {}
        if tool_call.get("name") == "task" and args.get("subagent_type") == self.required_subagent:
            return await handler(request)

        return ToolMessage(
            content=(
                "Source routing is required before any other tool call. "
                f"Call task with subagent_type={self.required_subagent!r}."
            ),
            tool_call_id=tool_call.get("id", "source-routing-guard"),
            name=tool_call.get("name"),
            status="error",
        )


class EmptyContentFixMiddleware(AgentMiddleware):
    """
    Middleware that fixes empty ToolMessage content.

    Some LLM APIs (e.g., NVIDIA, OpenAI) reject messages with empty content.
    This middleware ensures all ToolMessages have non-empty content by
    replacing empty strings with a placeholder.
    """

    def __init__(self, placeholder: str = "empty content received."):
        """
        Initialize the middleware.

        Args:
            placeholder: Text to use when ToolMessage content is empty.
        """
        self.placeholder = placeholder

    async def awrap_model_call(self, request, handler):
        """Fix empty ToolMessage content before sending to the model."""
        fixed_messages = []
        for msg in request.messages:
            if isinstance(msg, ToolMessage) and not msg.content:
                # Create a new ToolMessage with placeholder content
                fixed_messages.append(
                    ToolMessage(
                        content=self.placeholder,
                        tool_call_id=msg.tool_call_id,
                        name=getattr(msg, "name", None),
                        id=msg.id,
                    )
                )
            else:
                fixed_messages.append(msg)

        return await handler(request.override(messages=fixed_messages))


class ExecuteTimeoutClampMiddleware(AgentMiddleware):
    """Clamp the sandbox ``execute`` tool's per-call timeout to a configured ceiling.

    The deepagents ``execute`` tool forwards a model-supplied ``timeout`` straight to the
    sandbox backend, and providers cap it (OpenShell rejects an ``exec`` timeout above the
    gateway maximum). LLMs routinely pass an oversized value -- e.g. milliseconds, where the
    backend expects seconds, or an arbitrarily large round number -- so an unclamped timeout
    makes every ``execute`` fail with a "timeout exceeds maximum" error and no sandbox code
    ever runs. Bound the argument to the configured sandbox lifetime (seconds).

    This guards a different boundary than ``SandboxProvider._clamp_timeout`` in
    ``sandbox/base.py``: that clamp covers Deep Researcher Agent's own provider-mediated calls (e.g. workspace
    prep), whereas the deepagents ``execute`` tool reaches the backend without passing through
    it, so the untrusted agent argument must be sanitized here at the tool-call boundary.
    """

    def __init__(self, *, max_timeout_seconds: int) -> None:
        """Store the ceiling (in seconds) that a single ``execute`` call may request."""
        self.max_timeout_seconds = max(1, int(max_timeout_seconds))

    async def awrap_tool_call(self, request, handler):
        """Clamp an oversized ``timeout`` argument on ``execute`` tool calls."""
        tool_call = request.tool_call
        if tool_call.get("name") != "execute":
            return await handler(request)
        args = tool_call.get("args")
        if not isinstance(args, dict) or not isinstance(args.get("timeout"), (int, float)):
            return await handler(request)
        requested = int(args["timeout"])
        # A non-positive value means "no timeout" to the backend; leave it alone.
        if requested <= 0 or requested <= self.max_timeout_seconds:
            return await handler(request)
        logger.warning(
            "Clamping execute timeout %ss -> %ss (agent-supplied value exceeds the sandbox ceiling)",
            requested,
            self.max_timeout_seconds,
        )
        modified = {**tool_call, "args": {**args, "timeout": self.max_timeout_seconds}}
        return await handler(request.override(tool_call=modified))


class FilesystemToolCallGuardMiddleware(AgentMiddleware):
    """Normalize safe filesystem aliases and reject unresolved sandbox path templates."""

    async def awrap_tool_call(self, request, handler):
        """Repair ``read_file(path=...)`` and fail before executing placeholder paths."""
        tool_call = request.tool_call
        if not isinstance(tool_call, dict):
            return await handler(request)
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return await handler(request)

        if tool_call.get("name") == "read_file" and isinstance(args.get("path"), str):
            normalized_args = {key: value for key, value in args.items() if key != "path"}
            normalized_args.setdefault("file_path", args["path"])
            modified = {**tool_call, "args": normalized_args}
            return await handler(request.override(tool_call=modified))

        if tool_call.get("name") == "execute" and isinstance(args.get("command"), str):
            command = args["command"]
            unresolved = _UNRESOLVED_SANDBOX_PATH_PATTERN.search(command)
            if unresolved is not None:
                return ToolMessage(
                    content=(
                        f"Command not executed: unresolved sandbox path placeholder {unresolved.group(0)}. "
                        "Use the exact sandbox_workdir or sandbox_artifact_dir path from your instructions."
                    ),
                    tool_call_id=tool_call.get("id", "filesystem-tool-call-guard"),
                    name="execute",
                    status="error",
                )

        return await handler(request)


class FinalReportOwnershipGuardMiddleware(AgentMiddleware):
    """Reserve final-report mutation for the writer role."""

    async def awrap_tool_call(self, request, handler):
        """Reject non-writer mutations of either final-report state path."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        if tool_call.get("name") not in {"write_file", "edit_file"}:
            return await handler(request)
        if _tool_file_path(tool_call) not in FINAL_REPORT_STATE_PATHS:
            return await handler(request)
        return ToolMessage(
            content=(
                "final_report_writer_only: only writer-agent may write or edit "
                f"{FINAL_REPORT_PATH}; hand off evidence through the normal research workflow."
            ),
            tool_call_id=tool_call.get("id", "final-report-ownership"),
            name=tool_call.get("name"),
            status="error",
        )


class StateMutationGuardMiddleware(AgentMiddleware):
    """Restrict model-issued mutations of the StateBackend filesystem by role."""

    def __init__(self, *, writer: bool, sandbox_enabled: bool) -> None:
        self.writer = writer
        self.sandbox_enabled = sandbox_enabled

    def _is_state_backed(self, path: str | None) -> bool:
        if path is None:
            return False
        return not self.sandbox_enabled or path == "/shared" or path.startswith("/shared/")

    @staticmethod
    def _tool_error(tool_call: dict[str, object], reason: str, guidance: str) -> ToolMessage:
        return ToolMessage(
            content=f"{reason}: {guidance}",
            tool_call_id=tool_call.get("id", "state-mutation-guard"),
            name=tool_call.get("name"),
            status="error",
        )

    def _rejection(self, request: object) -> ToolMessage | None:
        """Return a denial for a guarded mutation, otherwise allow delegation."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        tool_name = tool_call.get("name")
        if tool_name not in {"write_file", "edit_file"}:
            return None

        target = _tool_file_path(tool_call)
        if not self._is_state_backed(target):
            return None
        if not self.writer:
            return self._tool_error(
                tool_call,
                "state_mutation_role_denied",
                "this agent has read-only access to shared research state; return structured output instead",
            )
        if target != FINAL_REPORT_PATH:
            return self._tool_error(
                tool_call,
                "writer_state_path_denied",
                f"writer-agent may mutate only {FINAL_REPORT_PATH}",
            )
        if tool_name == "edit_file":
            return self._tool_error(
                tool_call,
                "writer_output_edit_not_supported",
                f"use write_file with file_path={FINAL_REPORT_PATH} and the complete bounded report",
            )
        return None

    def wrap_tool_call(self, request, handler):
        """Reject unauthorized state writes before a synchronous backend call."""
        rejection = self._rejection(request)
        if rejection is not None:
            return rejection
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """Reject unauthorized state writes before an asynchronous backend call."""
        rejection = self._rejection(request)
        if rejection is not None:
            return rejection
        return await handler(request)


class FinalReportCommitMiddleware(AgentMiddleware):
    """Commit writer-owned output with overwrite and exact-digest verification."""

    def __init__(
        self,
        *,
        backend: object,
        tracker: FinalReportCommitTracker,
        state_budget: StateBudgetLedger | None = None,
        resource_limits: DeepResearchResourceLimits | None = None,
    ) -> None:
        self.backend = backend
        self.tracker = tracker
        self.resource_limits = resource_limits or DeepResearchResourceLimits()
        self.state_budget = state_budget or StateBudgetLedger(
            limits=self.resource_limits,
            files={},
            sandbox_enabled=True,
        )
        self._mutation_lock = asyncio.Lock()

    @staticmethod
    def _tool_error(tool_call: dict[str, object], reason: str, guidance: str) -> ToolMessage:
        return ToolMessage(
            content=f"{reason}: {guidance}",
            tool_call_id=tool_call.get("id", "final-report-commit"),
            name=tool_call.get("name"),
            status="error",
        )

    @staticmethod
    def _response_error(response: object) -> object:
        return response.get("error") if isinstance(response, dict) else getattr(response, "error", None)

    async def _commit_write(self, tool_call: dict[str, object]) -> ToolMessage:
        args = tool_call.get("args")
        content = args.get("content") if isinstance(args, dict) else None
        if not isinstance(content, str):
            return self._tool_error(tool_call, "writer_output_commit_failed", "report content must be text")
        encoded = content.encode("utf-8")
        if len(encoded) > self.resource_limits.max_final_report_bytes:
            return self._tool_error(
                tool_call,
                "writer_output_limit_exceeded",
                f"report exceeds the {self.resource_limits.max_final_report_bytes}-byte UTF-8 limit",
            )
        try:
            reservation = self.state_budget.reserve([(FINAL_REPORT_PATH, encoded)])
        except ValueError:
            return self._tool_error(
                tool_call,
                "writer_output_limit_exceeded",
                "report would exceed the shared-state resource limit",
            )
        try:
            responses = await self.backend.aupload_files([(FINAL_REPORT_PATH, encoded)])
        except Exception as exc:  # noqa: BLE001 - return a stable, sanitized tool error
            self.state_budget.rollback(reservation)
            logger.warning("Writer final-report commit failed (%s)", type(exc).__name__)
            return self._tool_error(tool_call, "writer_output_commit_failed", "the backend rejected the write")
        if not isinstance(responses, list) or len(responses) != 1 or self._response_error(responses[0]):
            self.state_budget.rollback(reservation)
            logger.warning("Writer final-report commit returned an unsuccessful upload response")
            return self._tool_error(tool_call, "writer_output_commit_failed", "the backend rejected the write")
        self.tracker.record(content)
        return ToolMessage(
            content=f"Updated file {FINAL_REPORT_PATH}",
            tool_call_id=tool_call.get("id", "final-report-commit"),
            name="write_file",
            status="success",
        )

    async def awrap_tool_call(self, request, handler):
        """Upsert bounded writer output; edits are rejected before mutation."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        tool_name = tool_call.get("name")
        if tool_name not in {"write_file", "edit_file"}:
            return await handler(request)
        target = _tool_file_path(tool_call)
        if target not in FINAL_REPORT_STATE_PATHS:
            return await handler(request)
        if target != FINAL_REPORT_PATH:
            return self._tool_error(
                tool_call,
                "writer_output_path_invalid",
                f"write the final report to {FINAL_REPORT_PATH}",
            )

        async with self._mutation_lock:
            if tool_name == "write_file":
                return await self._commit_write(tool_call)
            return self._tool_error(
                tool_call,
                "writer_output_edit_not_supported",
                f"use write_file with file_path={FINAL_REPORT_PATH} and the complete bounded report",
            )


class RequiredOutputFileMiddleware(AgentMiddleware):
    """Verify a model's file-backed completion marker before ending its run.

    A model can claim that it wrote a file without making the filesystem tool call.
    Keep recovery local to that agent: request one corrective model turn, then fail
    with a stable reason code instead of restarting the surrounding workflow.
    """

    def __init__(
        self,
        *,
        tracker: FinalReportCommitTracker,
        paths: tuple[str, ...] = FINAL_REPORT_STATE_PATHS,
        completion_marker: str = "Wrote /shared/output.md",
        max_retries: int = 1,
        reason_code: str = "writer_output_not_committed",
    ) -> None:
        """Configure the accepted state paths and bounded corrective turns."""
        if not paths:
            raise ValueError("paths must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.tracker = tracker
        self.paths = paths
        self.completion_marker = completion_marker
        self.max_retries = max_retries
        self.reason_code = reason_code
        self._retry_message = (
            "The final report is missing, empty, or was not committed by this writer run. "
            "Do not repeat research or regenerate artifacts. "
            f"Call write_file with file_path={paths[0]} and the complete final Markdown, confirm the tool "
            f"succeeds, and only then return `{completion_marker}`."
        )

    @staticmethod
    def _files_from_state(state: object) -> object:
        return state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})

    def _required_output_is_committed(self, state: object) -> bool:
        files = self._files_from_state(state)
        return self.tracker.committed_text(files, paths=self.paths) is not None

    def _retry_count(self, messages: list[object]) -> int:
        return sum(
            isinstance(message, HumanMessage)
            and message.additional_kwargs.get(_GENERATED_RETRY_MARKER) == "required_output_file"
            for message in messages
        )

    def _check_after_model(self, state: object) -> dict[str, object] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not isinstance(messages, list) or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or last_message.tool_calls:
            return None
        if last_message.text.strip() != self.completion_marker:
            return None
        if self._required_output_is_committed(state):
            return None

        retry_count = self._retry_count(messages)
        if retry_count >= self.max_retries:
            raise RuntimeError(self.reason_code)

        logger.warning("Agent reported completion before committing the required output; requesting corrective turn")
        return {
            "messages": [
                HumanMessage(
                    content=self._retry_message,
                    additional_kwargs={_GENERATED_RETRY_MARKER: "required_output_file"},
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        """Verify synchronous writer completion and request one local repair when needed."""
        return self._check_after_model(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        """Verify asynchronous writer completion and request one local repair when needed."""
        return self._check_after_model(state)


class RequiredWriterDelegationMiddleware(AgentMiddleware):
    """Prevent the orchestrator from terminating before writer-owned publication.

    Source failures can make an orchestrator conclude that no further research
    is useful and return ordinary assistant text without ever delegating to the
    writer. Give it one bounded corrective turn that forbids more research and
    requires writer delegation. The writer's existing commit middleware remains
    the only component allowed to publish the final report.
    """

    def __init__(
        self,
        *,
        tracker: FinalReportCommitTracker,
        max_retries: int = 1,
        reason_code: str = "writer_output_not_committed",
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.tracker = tracker
        self.max_retries = max_retries
        self.reason_code = reason_code
        self._retry_message = (
            "The run cannot finish because writer-agent has not committed /shared/output.md. "
            "Do not perform or retry source research. Delegate to writer-agent now using the Writer Delegation "
            "Template and the plan, research notes, verified sources, and explicit evidence gaps already available. "
            "After writer-agent returns, return only its completion marker."
        )

    def _retry_count(self, messages: list[object]) -> int:
        return sum(
            isinstance(message, HumanMessage)
            and message.additional_kwargs.get(_GENERATED_RETRY_MARKER) == "required_writer_delegation"
            for message in messages
        )

    def _check_after_model(self, state: object) -> dict[str, object] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        files = state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})
        if not isinstance(messages, list) or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or last_message.tool_calls:
            return None
        if self.tracker.committed_text(files, paths=FINAL_REPORT_STATE_PATHS) is not None:
            return None
        if self._retry_count(messages) >= self.max_retries:
            raise RuntimeError(self.reason_code)

        logger.warning("Orchestrator ended before writer delegation; requesting one corrective turn")
        return {
            "messages": [
                HumanMessage(
                    content=self._retry_message,
                    additional_kwargs={_GENERATED_RETRY_MARKER: "required_writer_delegation"},
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        """Require a synchronous orchestrator to delegate writer publication."""
        return self._check_after_model(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        """Require an asynchronous orchestrator to delegate writer publication."""
        return self._check_after_model(state)


# Common hallucinated tool name mappings
_TOOL_NAME_ALIASES: dict[str, str] = {
    "open_file": "read_file",
    "find": "grep",
    "find_file": "glob",
}


class ToolNameSanitizationMiddleware(AgentMiddleware):
    """
    Middleware that sanitizes corrupted tool names in LLM responses.

    LLMs sometimes generate malformed tool calls with suffixes like
    <|channel|>commentary or .exec, or hallucinate tool names like
    open_file or find. This middleware intercepts the model response
    and fixes tool names before the framework dispatches them.
    """

    def __init__(self, valid_tool_names: list[str]):
        """Store the set of valid tool names used to correct malformed tool calls."""
        self.valid_tool_names = set(valid_tool_names)

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize a potentially corrupted tool name.

        Returns the cleaned name if it maps to a valid tool,
        otherwise returns the original name unchanged.
        """
        # 1. Strip <|channel|> and everything after
        if "<|channel|>" in name:
            candidate = name.split("<|channel|>", maxsplit=1)[0]
            if candidate in self.valid_tool_names:
                logger.info("Sanitized tool name (original_%s) -> '%s'", log_content_metadata(name), candidate)
                return candidate

        # 2. Strip dot suffix if base name is valid
        if "." in name:
            candidate = name.split(".", maxsplit=1)[0]
            if candidate in self.valid_tool_names:
                logger.info("Sanitized tool name (original_%s) -> '%s'", log_content_metadata(name), candidate)
                return candidate

        # 3. Map common hallucinated names
        if name in _TOOL_NAME_ALIASES:
            mapped = _TOOL_NAME_ALIASES[name]
            if mapped in self.valid_tool_names:
                logger.info("Mapped tool name (original_%s) -> '%s'", log_content_metadata(name), mapped)
                return mapped

        return name

    async def awrap_model_call(self, request, handler):
        """Intercept model response and sanitize tool names."""
        response = await handler(request)

        needs_fix = False
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    sanitized = self._sanitize_tool_name(tc["name"])
                    if sanitized != tc["name"]:
                        needs_fix = True
                        break
                if needs_fix:
                    break

        if not needs_fix:
            return response

        new_result = []
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                new_tool_calls = []
                for tc in msg.tool_calls:
                    new_tool_calls.append({**tc, "name": self._sanitize_tool_name(tc["name"])})
                additional_kwargs = dict(msg.additional_kwargs)
                raw_tool_calls = additional_kwargs.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    sanitized_raw_tool_calls = []
                    for raw_tool_call in raw_tool_calls:
                        if not isinstance(raw_tool_call, dict) or not isinstance(raw_tool_call.get("function"), dict):
                            sanitized_raw_tool_calls.append(raw_tool_call)
                            continue
                        function = dict(raw_tool_call["function"])
                        function["name"] = self._sanitize_tool_name(str(function.get("name") or ""))
                        sanitized_raw_tool_calls.append({**raw_tool_call, "function": function})
                    additional_kwargs["tool_calls"] = sanitized_raw_tool_calls
                new_msg = msg.model_copy(
                    update={
                        "additional_kwargs": additional_kwargs,
                        "tool_calls": new_tool_calls,
                    }
                )
                new_result.append(new_msg)
            else:
                new_result.append(msg)

        return ModelResponse(result=new_result, structured_response=response.structured_response)


def _request_tool_name(tool: object) -> str | None:
    """Return a LangChain model-request tool name across common tool shapes."""
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(tool, dict):
        dict_name = tool.get("name")
        if isinstance(dict_name, str):
            return dict_name
        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str):
                return function_name
    return None


class ToolVisibilityMiddleware(AgentMiddleware):
    """Hide selected tools from model requests without removing scaffolding middleware."""

    def __init__(self, hidden_tool_names: set[str]) -> None:
        """Store the tool names to hide from model requests."""
        self.hidden_tool_names = hidden_tool_names

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Return the tool list with hidden tools removed."""
        if not self.hidden_tool_names:
            return tools
        return [tool for tool in tools if _request_tool_name(tool) not in self.hidden_tool_names]

    def wrap_model_call(self, request, handler):
        """Filter hidden tools before a synchronous model call."""
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(self, request, handler):
        """Filter hidden tools before an asynchronous model call."""
        return await handler(request.override(tools=self._filter_tools(request.tools)))


class TodoSuppressionMiddleware(AgentMiddleware):
    """Strip the framework's ``write_todos`` tool and its injected prompt for a subagent.

    deepagents attaches ``TodoListMiddleware`` to every subagent, which adds the
    ``write_todos`` tool plus a system-prompt block telling the agent to use it.
    Agents that own no progress list - e.g. the planner, which returns a single
    structured ``ResearchPlan`` - should not have it. Placed after the framework's
    ``TodoListMiddleware`` in the stack, this removes both the tool and the injected
    prompt block from the model request, keeping todo tracking solely with the
    orchestrator. It is a no-op when neither is present.
    """

    _TODO_TOOL = "write_todos"
    _TODO_PROMPT_MARKER = "## `write_todos`"

    def _clean_request(self, request: object) -> object:
        """Return the request with the write_todos tool and its prompt block removed."""
        overrides: dict[str, object] = {
            "tools": [tool for tool in request.tools if _request_tool_name(tool) != self._TODO_TOOL]
        }
        system_message = getattr(request, "system_message", None)
        if system_message is not None:
            blocks = system_message.content_blocks
            kept = [
                block
                for block in blocks
                if not (isinstance(block, dict) and self._TODO_PROMPT_MARKER in str(block.get("text", "")))
            ]
            if len(kept) != len(blocks):
                overrides["system_message"] = SystemMessage(content=kept)
        return request.override(**overrides)

    def wrap_model_call(self, request, handler):
        """Strip write_todos and its prompt before a synchronous model call."""
        return handler(self._clean_request(request))

    async def awrap_model_call(self, request, handler):
        """Strip write_todos and its prompt before an asynchronous model call."""
        return await handler(self._clean_request(request))


class TodoQuotaMiddleware(AgentMiddleware):
    """Reject oversized top-level todo replacements before they mutate graph state."""

    _TODO_TOOL = "write_todos"

    def __init__(self, *, resource_limits: DeepResearchResourceLimits | None = None) -> None:
        """Configure the hard job-local todo count and content ceilings."""
        self.resource_limits = resource_limits or DeepResearchResourceLimits()

    def _validate(self, request: object) -> None:
        """Validate raw write_todos arguments before delegating to the framework tool."""
        tool_call = getattr(request, "tool_call", {})
        if not isinstance(tool_call, dict) or tool_call.get("name") != self._TODO_TOOL:
            return
        args = tool_call.get("args", {})
        todos = args.get("todos") if isinstance(args, dict) else None
        if not isinstance(todos, list):
            raise ValueError("write_todos requires a todos list")
        if len(todos) > self.resource_limits.max_todo_items:
            raise ValueError(f"write_todos exceeds the {self.resource_limits.max_todo_items}-item limit")

        total_chars = 0
        for todo in todos:
            content = todo.get("content") if isinstance(todo, dict) else None
            if not isinstance(content, str):
                raise ValueError("write_todos item content must be a string")
            if len(content) > self.resource_limits.max_todo_item_chars:
                raise ValueError(
                    f"write_todos item exceeds the {self.resource_limits.max_todo_item_chars}-character limit"
                )
            total_chars += len(content)
        if total_chars > self.resource_limits.max_total_todo_chars:
            raise ValueError(
                f"write_todos exceeds the {self.resource_limits.max_total_todo_chars}-character aggregate content limit"
            )

    @staticmethod
    def _tool_error(request: object, error: ValueError) -> ToolMessage:
        """Return a recoverable tool error for a rejected todo replacement."""
        tool_call = getattr(request, "tool_call", {})
        if not isinstance(tool_call, dict):
            tool_call = {}
        return ToolMessage(
            content=f"write_todos_quota_rejected: {error}",
            tool_call_id=tool_call.get("id", "todo-quota"),
            name=tool_call.get("name", TodoQuotaMiddleware._TODO_TOOL),
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        """Validate a synchronous todo update before graph-state mutation."""
        try:
            self._validate(request)
        except ValueError as exc:
            return self._tool_error(request, exc)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """Validate an asynchronous todo update before graph-state mutation."""
        try:
            self._validate(request)
        except ValueError as exc:
            return self._tool_error(request, exc)
        return await handler(request)


class ToolRetryMiddleware(AgentMiddleware):
    """Retries failed tool calls with exponential backoff.

    Provides uniform retry coverage for all tools. Some tools (e.g., Tavily)
    have their own internal retry; this middleware wraps the outer call so
    tools without retry (knowledge layer, paper search) are also covered.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
    ):
        """Configure retry count and exponential backoff for failed tool calls."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay

    async def awrap_tool_call(self, request, handler):
        """Retry tool calls on failure with exponential backoff."""
        delay = self.initial_delay
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    tool_name = request.tool_call.get("name", "?") if hasattr(request, "tool_call") else "?"
                    logger.warning(
                        "Tool call failed (tool_%s, attempt %d/%d, error_type=%s, detail_%s)",
                        log_content_metadata(tool_name),
                        attempt + 1,
                        self.max_retries + 1,
                        type(e).__name__,
                        log_content_metadata(e),
                    )
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor
        raise last_exception


class SourceRegistryMiddleware(AgentMiddleware):
    """Intercepts tool call results to build a registry of actual sources.

    Two responsibilities:
    1. awrap_tool_call: Capture URLs/citation keys from tool results
    2. awrap_model_call: Inject a consolidated source list into the LLM context
       so the orchestrator has a single, authoritative reference list when
       writing the final report (no manual reconciliation across research-note files)

    Source capture is gated only by the agent's loaded tool set
    (``source_tool_names``). Internal scratchpad/runtime tools (think,
    write_file, read_file, etc.) are added by deepagents itself and never
    appear in that set, so they are implicitly excluded. Tools registered as
    configured data sources additionally carry a ``source_id`` label, but a
    tool does *not* have to be declared under ``data_sources`` to contribute
    sources — agents can be passed citable tools directly.

    The registry is also used by verify_citations() to strip fabricated,
    stale, or intermediate-artifact citations from the final report.
    """

    def __init__(self, source_tool_names: set[str] | None = None) -> None:
        """Create a source registry scoped to the given source-producing tool names."""
        self.registry = SourceRegistry()
        self._source_tool_names = source_tool_names or set()
        self._compact_source_keys: set[str] = set()
        self._lock = asyncio.Lock()

    def active_registry(self) -> SourceRegistry:
        """Return the session-scoped registry if set, otherwise the instance registry."""
        from deep_researcher_agent.common.citation_verification import get_session_registry

        return get_session_registry() or self.registry

    def has_sources(self) -> bool:
        """Return True when the active source registry contains captured sources."""
        return bool(self.active_registry().all_sources())

    @staticmethod
    def _locator_key(locator: str) -> str:
        """Return the comparable key used for source locators and registry entries."""
        locator = locator.strip()
        if locator.startswith(("http://", "https://")):
            from deep_researcher_agent.common.citation_verification import _normalize_url

            return _normalize_url(locator)
        return locator

    @classmethod
    def _entry_key(cls, entry: SourceEntry) -> str | None:
        """Return the comparable key for a registered source entry."""
        if entry.url:
            return cls._locator_key(entry.url)
        if entry.citation_key:
            return entry.citation_key.strip()
        return None

    def register_research_note_sources(self, notes: list[object]) -> None:
        """Mark ResearchNotes source locators as the compact writer-facing citation set."""
        for note in notes:
            sources = getattr(note, "sources", None) or []
            for source in sources:
                locator = getattr(source, "locator", "")
                if isinstance(locator, str) and locator.strip():
                    self._compact_source_keys.add(self._locator_key(locator))

    def register_compact_sources(self, sources: list[SourceEntry]) -> int:
        """Register seeded sources and expose them in the compact citation source list."""
        registry = self.active_registry()
        registered = 0
        for source in sources:
            key = self._entry_key(source)
            if not key:
                continue
            registry.add(source)
            self._compact_source_keys.add(key)
            registered += 1
        return registered

    async def awrap_tool_call(self, request, handler):
        """Capture sources from tool results after execution.

        Capture is gated only by the agent's loaded tool set
        (``source_tool_names``). Internal scratchpad/runtime tools (think,
        write_file, read_file, etc.) are added by deepagents itself and never
        appear in that set, so they are implicitly excluded.

        Tools that resolve to a configured data source via
        :func:`get_source_id_for_tool` get a ``source_id`` label. Tools passed
        directly to the agent without a data-source declaration are still
        captured — their results are real, citable evidence even when
        ``data_source_registry`` does not know about them — but their entries
        carry no ``source_id``.
        """
        result = await handler(request)
        if isinstance(result, ToolMessage) and result.content:
            tool_name = ""
            if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", "")
            if tool_name not in self._source_tool_names:
                return result
            content = str(result.content)
            if is_non_citable_status_output(content):
                return result
            source_id = get_source_id_for_tool(tool_name)
            sources = extract_sources_from_tool_result(
                tool_name,
                content,
                source_id=source_id,
                result_status=getattr(result, "status", None),
            )
            async with self._lock:
                active_registry = self.active_registry()
                for source in sources:
                    active_registry.add(source)
            if sources:
                logger.info(
                    "[CitationRegistry] Captured %d source(s) from %s",
                    len(sources),
                    tool_name,
                )
        return result

    def _render_source_list_text(self, sources: list[SourceEntry]) -> str | None:
        """Render a consolidated source list from registry entries.

        Returns rendered template text, or None if no sources captured.
        Used by agent.run() to include the source list in retry messages
        when citation quality is poor.
        """
        from urllib.parse import urlparse

        from deep_researcher_agent.common.citation_verification import _normalize_url

        if not sources:
            return None

        seen: set[str] = set()
        template_sources = []
        for entry in sources:
            if entry.url:
                normalized = _normalize_url(entry.url)
                if normalized in seen:
                    continue
                seen.add(normalized)
                if entry.title:
                    title = entry.title
                else:
                    try:
                        title = urlparse(entry.url).netloc.replace("www.", "")
                    except Exception:
                        title = entry.url
                template_sources.append({"title": title, "url": entry.url})
            elif entry.citation_key:
                key = entry.citation_key
                if key in seen:
                    continue
                seen.add(key)
                template_sources.append({"title": key, "url": key})

        if not template_sources:
            return None

        try:
            template = load_prompt(_PROMPTS_DIR, "source_registry")
            return render_prompt_template(template, sources=template_sources)
        except Exception:
            logger.warning("Failed to load source_registry prompt template", exc_info=True)
            return None

    def get_source_entries(self, mode: str = "compact") -> list[SourceEntry]:
        """Return the source entries represented by the writer-facing source list."""
        sources = self.active_registry().all_sources()
        if mode == "full" or not self._compact_source_keys:
            return sources
        compact_sources = [source for source in sources if self._entry_key(source) in self._compact_source_keys]
        return compact_sources or sources

    def get_source_list_text(self, mode: str = "compact") -> str | None:
        """Build a writer-facing verified source list.

        Compact mode returns the subset of registered sources that researcher
        workers actually carried forward in structured ResearchNotes. Full mode
        returns the complete registry.
        """
        return self._render_source_list_text(self.get_source_entries(mode=mode))


class ArtifactHarvestMiddleware(AgentMiddleware):
    """Checkpoint durable artifacts after successful sandbox execute calls."""

    def __init__(self, artifact_manager: object) -> None:
        """Store the artifact manager used for best-effort checkpoints."""
        self.artifact_manager = artifact_manager

    async def awrap_tool_call(self, request, handler):
        """Run the tool, then checkpoint manifest-declared artifacts after execute."""
        result = await handler(request)
        tool_name = ""
        if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", "")
        result_status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
        if tool_name == "execute" and result_status != "error":
            try:
                captured = await asyncio.to_thread(self.artifact_manager.harvest_after_execute)
            except Exception as exc:  # noqa: BLE001 - artifact capture must not fail the agent
                logger.warning("Artifact checkpoint harvest failed (%s)", type(exc).__name__)
            else:
                result = self._append_checkpoint_result(result, captured)
        return result

    @staticmethod
    def _append_checkpoint_result(result: object, captured: object) -> object:
        """Tell the model the exact safe filenames captured from a valid manifest."""
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        if not isinstance(captured, (list, tuple)) or not captured:
            return result

        lines = ["Artifact checkpoint captured these exact filenames:"]
        for artifact in captured[:10]:
            filename = PurePosixPath(str(getattr(artifact, "filename", ""))).name
            if not filename:
                continue
            if bool(getattr(artifact, "inline", False)):
                lines.append(f"- {filename} (inline): embed as ![caption](artifact://{filename})")
            else:
                lines.append(f"- {filename} (downloadable; not marked inline)")
        if len(lines) == 1:
            return result
        content = result.content.rstrip() + "\n\n" + "\n".join(lines)
        return result.model_copy(update={"content": content})


class SourceRoutingPersistenceMiddleware(AgentMiddleware):
    """Persist the source router's schema-validated response to shared state."""

    def __init__(
        self,
        backend: object,
        *,
        state_budget: StateBudgetLedger | None = None,
        resource_limits: DeepResearchResourceLimits | None = None,
        path: str = _SOURCE_ROUTING_PATH,
    ) -> None:
        self.backend = backend
        self.resource_limits = resource_limits or DeepResearchResourceLimits()
        self.state_budget = state_budget or StateBudgetLedger(
            limits=self.resource_limits,
            files={},
            sandbox_enabled=True,
        )
        self.path = path

    @staticmethod
    def _routing_from_state(state: object) -> object:
        if isinstance(state, dict):
            return state.get("structured_response")
        return getattr(state, "structured_response", None)

    def _persist_routing(self, routing: object) -> None:
        if routing is None:
            return
        if hasattr(routing, "model_dump"):
            payload = routing.model_dump(mode="json", exclude_none=True)
        elif isinstance(routing, dict):
            payload = routing
        else:
            return

        content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        if len(content) > self.resource_limits.max_source_routing_bytes:
            raise ValueError(
                f"Source routing exceeds the {self.resource_limits.max_source_routing_bytes}-byte serialized size limit"
            )
        reservation = self.state_budget.reserve([(self.path, content)])
        try:
            responses = self.backend.upload_files([(self.path, content)])
        except Exception:
            self.state_budget.rollback(reservation)
            raise
        errors = [f"{response.path}: {response.error}" for response in responses if getattr(response, "error", None)]
        if errors:
            self.state_budget.rollback(reservation)
            logger.error(
                "Failed to persist source routing to %s (error_count=%d detail_%s)",
                self.path,
                len(errors),
                log_content_metadata("; ".join(errors)),
            )
            raise RuntimeError(f"Failed to persist source routing to {self.path}")

    def after_agent(self, state, runtime):
        """Persist source routing after a synchronous source-router run."""
        self._persist_routing(self._routing_from_state(state))

    async def aafter_agent(self, state, runtime):
        """Persist source routing after an asynchronous source-router run."""
        await asyncio.to_thread(self._persist_routing, self._routing_from_state(state))


class PlanPersistenceMiddleware(AgentMiddleware):
    """Persists the planner's structured ResearchPlan to the shared filesystem.

    The planner returns a schema-validated ``ResearchPlan`` (``response_format``).
    This middleware writes that plan to ``/shared/plan.json`` deterministically via
    the overwrite-safe ``backend.upload_files`` (the same state-channel write
    ``run_research_batch`` uses for ResearchNotes), so the planner never performs
    file I/O itself. Keeping the write off the LLM removes the ``write_file`` /
    ``edit_file`` loop the planner otherwise hits when ``/shared/plan.json`` already
    exists, since the LLM ``write_file`` tool refuses to overwrite while
    ``upload_files`` overwrites in place.

    Persistence failures propagate so the planner task fails before the
    orchestrator reads a missing or stale ``/shared/plan.json``.
    """

    def __init__(
        self,
        backend: object,
        *,
        state_budget: StateBudgetLedger | None = None,
        resource_limits: DeepResearchResourceLimits | None = None,
        path: str = "/shared/plan.json",
    ) -> None:
        """Initialize the middleware.

        Args:
            backend: Shared filesystem backend exposing ``upload_files``.
            resource_limits: Hard plan/query limits enforced before state mutation.
            path: Shared path the serialized plan is written to.
        """
        self.backend = backend
        self.resource_limits = resource_limits or DeepResearchResourceLimits()
        self.state_budget = state_budget or StateBudgetLedger(
            limits=self.resource_limits,
            files={},
            sandbox_enabled=True,
        )
        self.path = path

    @staticmethod
    def _plan_from_state(state: object) -> object:
        """Extract the planner's ``structured_response`` from dict or attribute state."""
        if isinstance(state, dict):
            return state.get("structured_response")
        return getattr(state, "structured_response", None)

    def _persist_plan(self, plan: object) -> None:
        """Serialize a structured ResearchPlan and upload it to shared state."""
        if plan is None:
            return
        if hasattr(plan, "model_dump"):
            payload = plan.model_dump(mode="json", exclude_none=True)
        elif isinstance(plan, dict):
            payload = plan
        else:
            return

        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("Research plan queries must be a list")
        if len(queries) > self.resource_limits.max_research_queries:
            raise ValueError(f"Research plan exceeds the {self.resource_limits.max_research_queries}-query job limit")
        query_chars = 0
        for query in queries:
            if not isinstance(query, dict):
                raise ValueError("Research plan contains an invalid query")
            query_text = query.get("query")
            if not isinstance(query_text, str) or not query_text:
                raise ValueError("Research plan contains an invalid query")
            query_chars += len(query_text)
            subqueries = query.get("subqueries", [])
            if not isinstance(subqueries, list) or any(
                not isinstance(subquery, str) or not subquery for subquery in subqueries
            ):
                raise ValueError("Research plan contains invalid subqueries")
            query_chars += sum(len(subquery) for subquery in subqueries)
        if query_chars > self.resource_limits.max_total_query_chars:
            raise ValueError(
                "Research plan exceeds the "
                f"{self.resource_limits.max_total_query_chars}-character aggregate query limit"
            )

        content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        if len(content) > self.resource_limits.max_plan_bytes:
            raise ValueError(
                f"Research plan exceeds the {self.resource_limits.max_plan_bytes}-byte serialized size limit"
            )
        reservation = self.state_budget.reserve([(self.path, content)])
        try:
            responses = self.backend.upload_files([(self.path, content)])
        except Exception:
            self.state_budget.rollback(reservation)
            raise
        errors = [f"{response.path}: {response.error}" for response in responses if getattr(response, "error", None)]
        if errors:
            self.state_budget.rollback(reservation)
            logger.error(
                "Failed to persist plan to %s (error_count=%d detail_%s)",
                self.path,
                len(errors),
                log_content_metadata("; ".join(errors)),
            )
            raise RuntimeError(f"Failed to persist the research plan to {self.path}")

    def after_agent(self, state, runtime):
        """Persist the plan once the synchronous planner run completes."""
        self._persist_plan(self._plan_from_state(state))

    async def aafter_agent(self, state, runtime):
        """Persist the plan once the asynchronous planner run completes."""
        await asyncio.to_thread(self._persist_plan, self._plan_from_state(state))


class ToolResultPruningMiddleware(AgentMiddleware):
    """Truncates older tool results to keep context manageable.

    Keeps the last N tool results intact and truncates older ones to
    reduce "lost in the middle" degradation. Operates on awrap_model_call
    so the full results are still available for SourceRegistryMiddleware.
    """

    def __init__(self, keep_last_n: int = 3, max_chars: int = 500):
        """Configure how many recent tool results to keep intact and the truncation cap."""
        self.keep_last_n = keep_last_n
        self.max_chars = max_chars

    async def awrap_model_call(self, request, handler):
        """Truncate older ToolMessage content before sending to the model."""
        # Find all ToolMessage indices
        tool_indices = [i for i, msg in enumerate(request.messages) if isinstance(msg, ToolMessage)]

        if len(tool_indices) <= self.keep_last_n:
            return await handler(request)

        # Indices to truncate: all but the last keep_last_n
        truncate_indices = set(tool_indices[: -self.keep_last_n])

        pruned_messages = []
        for i, msg in enumerate(request.messages):
            if i in truncate_indices and isinstance(msg, ToolMessage) and msg.content:
                content = str(msg.content)
                if len(content) > self.max_chars:
                    truncated_content = content[: self.max_chars] + "\n\n[... truncated ...]"
                    pruned_messages.append(
                        ToolMessage(
                            content=truncated_content,
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", None),
                            id=msg.id,
                        )
                    )
                else:
                    pruned_messages.append(msg)
            else:
                pruned_messages.append(msg)

        return await handler(request.override(messages=pruned_messages))
