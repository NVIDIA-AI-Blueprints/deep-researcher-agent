#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a preinstalled AI-Q workflow inside a Harbor task environment.

The runner intentionally imports NAT and AI-Q lazily. The Harbor host package
does not need those dependencies; they are supplied by the pinned AI-Q image.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import UTC
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("aiq-harbor-runner")
SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
PROGRESS_HEARTBEAT_SEC = 10.0

_CALL_EVENTS = {
    "LLM_START": ("llm", "start"),
    "LLM_END": ("llm", "end"),
    "TOOL_START": ("tool", "start"),
    "TOOL_END": ("tool", "end"),
    "FUNCTION_START": ("function", "start"),
    "FUNCTION_END": ("function", "end"),
}
_DEEP_RESEARCH_SUBAGENTS = frozenset(
    {
        "planner-agent",
        "researcher-agent",
        "source-router-agent",
        "writer-agent",
    }
)
_MISSING = object()
_LLM_RUNTIME_FIELDS = (
    "model_name",
    "base_url",
    "temperature",
    "top_p",
    "max_tokens",
    "num_retries",
    "timeout",
    "parallel_tool_calls",
    "chat_template_kwargs",
)
_FUNCTION_RUNTIME_FIELDS = {
    "tavily_web_search": (
        "api_base_url",
        "include_answer",
        "advanced_search",
        "max_results",
        "max_content_length",
        "max_retries",
    ),
    "knowledge_retrieval": (
        "backend",
        "collection_name",
        "top_k",
        "rag_url",
        "ingest_url",
        "timeout",
    ),
    "shallow_research_agent": (
        "llm",
        "tools",
        "max_llm_turns",
        "max_tool_iterations",
    ),
    "paper_search": (
        "provider",
        "timeout",
        "max_results",
    ),
    "deep_research_agent": (
        "orchestrator_llm",
        "source_router_llm",
        "researcher_llm",
        "planner_llm",
        "writer_llm",
        "tools",
        "enable_citation_verification",
    ),
}


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write a JSON-serializable value."""
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _format_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _format_duration(seconds: float) -> str:
    total_milliseconds = round(max(seconds, 0.0) * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _plugin_inventory() -> list[dict[str, str | None]]:
    plugins: list[dict[str, str | None]] = []
    for entry_point in importlib_metadata.entry_points(group="nat.plugins"):
        distribution = getattr(entry_point, "dist", None)
        distribution_name = None
        distribution_version = None
        if distribution is not None:
            distribution_name = distribution.metadata.get("Name")
            distribution_version = distribution.version
        plugins.append(
            {
                "name": entry_point.name,
                "value": entry_point.value,
                "distribution": distribution_name,
                "version": distribution_version,
            }
        )
    return sorted(plugins, key=lambda item: (item["name"] or "", item["value"] or ""))


def _redact_sensitive_text(message: str) -> str:
    for name, value in os.environ.items():
        if not value or not any(marker in name.upper() for marker in SENSITIVE_ENV_MARKERS):
            continue
        message = message.replace(value, f"${{{name}}}")
    return message


def _safe_error(exc: BaseException) -> str:
    return _redact_sensitive_text(str(exc))


def _step_value(step: Any, name: str, default: Any = None) -> Any:
    value = getattr(step, name, _MISSING)
    if value is _MISSING:
        payload = getattr(step, "payload", None)
        value = getattr(payload, name, _MISSING)
    return default if value is _MISSING else value


def _event_type_name(step: Any) -> str:
    value = _step_value(step, "event_type", "")
    return str(getattr(value, "value", value) or "")


def _trajectory_scope(step: Any) -> tuple[str, str] | None:
    if isinstance(step, dict):
        parent_id = step.get("parent_id")
        ancestry = step.get("function_ancestry")
    else:
        parent_id = getattr(step, "parent_id", None)
        ancestry = getattr(step, "function_ancestry", None)
    if isinstance(ancestry, dict):
        function_id = ancestry.get("function_id")
    else:
        function_id = getattr(ancestry, "function_id", None)
    if not parent_id or not function_id:
        return None
    return str(parent_id), str(function_id)


def _trajectory_steps_for_atif(steps: list[Any]) -> list[Any]:
    """Keep only model-visible tool spans for ATIF trajectory conversion.

    NAT emits both TOOL and FUNCTION spans for a LangChain tool invocation, and
    the tool implementation can emit additional nested TOOL spans. A tool span
    is model-visible when it shares both the parent span and function invocation
    of an LLM turn. Function spans and tool spans from implementation-only
    descendants would otherwise be flattened into duplicate top-level ATIF tool
    calls.
    """
    llm_scopes = {
        scope for step in steps if _event_type_name(step) == "LLM_END" if (scope := _trajectory_scope(step)) is not None
    }

    filtered_steps: list[Any] = []
    for step in steps:
        event_type = _event_type_name(step)
        if event_type in {"FUNCTION_START", "FUNCTION_END"}:
            continue
        if event_type in {"TOOL_START", "TOOL_END"}:
            if _trajectory_scope(step) not in llm_scopes:
                continue
        filtered_steps.append(step)
    return filtered_steps


def _safe_event_name(value: Any, limit: int = 160) -> str | None:
    if value is None:
        return None
    name = _redact_sensitive_text(" ".join(str(value).split()).strip())
    return name[:limit] or None


def _object_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _allowlisted_subagent(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in _DEEP_RESEARCH_SUBAGENTS else None


def _task_subagent(step: Any) -> str | None:
    """Extract only the canonical role from a DeepAgents task call."""
    metadata = _step_value(step, "metadata")
    data = _step_value(step, "data")
    candidates = (
        _object_value(metadata, "tool_inputs"),
        _object_value(data, "input"),
    )
    for candidate in candidates:
        direct = _allowlisted_subagent(_object_value(candidate, "subagent_type"))
        if direct is not None:
            return direct
        if isinstance(candidate, str):
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
            direct = _allowlisted_subagent(_object_value(parsed, "subagent_type"))
            if direct is not None:
                return direct
    return None


class _LiveProgress:
    """Persist compact, content-free runtime progress for a Harbor trial."""

    def __init__(
        self,
        *,
        state_path: Path,
        events_path: Path,
        session_id: str,
        started_at_epoch: float,
    ) -> None:
        self._state_path = state_path
        self._events_path = events_path
        self._session_id = session_id
        self._started_at_epoch = started_at_epoch
        self._lock = threading.Lock()
        self._status = "running"
        self._phase = "preflight"
        self._events_observed = 0
        self._last_event: dict[str, Any] | None = None
        self._details: dict[str, Any] = {}
        self._calls = {kind: {"started": 0, "completed": 0} for kind in ("llm", "tool", "function")}
        self._active_names: dict[str, dict[str, str | None]] = {kind: {} for kind in self._calls}
        self._llm_scopes: set[tuple[str, str]] = set()
        self._tool_span_ids: set[str] = set()
        self._visible_tool_ids: set[str] = set()
        self._hidden_function_ids: set[str] = set()
        self._default_agent: str | None = None
        self._call_agents: dict[str, str] = {}
        self._agent_scopes: dict[str, str] = {}
        self._warned = False

        self._best_effort_write(lambda: self._events_path.parent.mkdir(parents=True, exist_ok=True))
        self._best_effort_write(lambda: _atomic_write_text(self._events_path, ""))
        self.heartbeat()

    def _best_effort_write(self, operation: Any) -> None:
        try:
            operation()
        except OSError as exc:
            if not self._warned:
                LOGGER.warning("Unable to persist live AI-Q progress: %s", _safe_error(exc))
                self._warned = True

    def _snapshot_locked(self, now: float) -> dict[str, Any]:
        calls = {
            kind: {
                **counts,
                "in_flight": max(counts["started"] - counts["completed"], 0),
            }
            for kind, counts in self._calls.items()
        }
        return {
            "schema_version": 2,
            "status": self._status,
            "phase": self._phase,
            "pid": os.getpid(),
            "session_id": self._session_id,
            "started_at": _format_timestamp(self._started_at_epoch),
            "updated_at": _format_timestamp(now),
            "elapsed": _format_duration(now - self._started_at_epoch),
            "events_observed": self._events_observed,
            "calls": calls,
            "last_event": self._last_event,
            **self._details,
        }

    def _write_state_locked(self, now: float | None = None) -> None:
        snapshot = self._snapshot_locked(time.time() if now is None else now)
        self._best_effort_write(lambda: _atomic_write_json(self._state_path, snapshot))

    def _append_event_locked(self, line: str) -> None:
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def heartbeat(self) -> None:
        with self._lock:
            self._write_state_locked()

    def set_phase(self, phase: str, **details: Any) -> None:
        with self._lock:
            self._phase = phase
            self._details.update(details)
            if details.get("workflow_type") == "deep_research_workflow":
                self._default_agent = "orchestrator"
            self._write_state_locked()

    def _current_agent_locked(self) -> str | None:
        if self._agent_scopes:
            active_agents = set(self._agent_scopes.values())
            return active_agents.pop() if len(active_agents) == 1 else "unknown"
        return self._default_agent

    def _clear_abandoned_agent_scopes_locked(self) -> None:
        if self._default_agent is None:
            return
        stale_ids = {call_id for call_id, agent in self._call_agents.items() if agent != self._default_agent}
        for call_id in stale_ids:
            self._call_agents.pop(call_id, None)
        self._agent_scopes.clear()
        self._tool_span_ids.difference_update(stale_ids)
        self._visible_tool_ids.difference_update(stale_ids)
        self._hidden_function_ids.difference_update(stale_ids)
        for active_names in self._active_names.values():
            for call_id in stale_ids:
                active_names.pop(call_id, None)

    def _agent_for_event_locked(
        self,
        step: Any,
        event_type: str,
        name: str | None,
        call_id: str,
    ) -> str | None:
        if self._default_agent is None:
            return None

        if event_type.endswith("_END") and call_id in self._call_agents:
            return self._call_agents[call_id]

        scoped_agent = None
        if event_type == "TOOL_START":
            if name == "task":
                scoped_agent = _task_subagent(step) or "unknown"
            elif name == "run_research_batch":
                scoped_agent = "researcher-agent"

        parent_id = str(_step_value(step, "parent_id", "") or "")
        parent_agent = self._call_agents.get(parent_id)
        if (
            event_type == "LLM_START"
            and scoped_agent is None
            and parent_agent == self._default_agent
            and self._agent_scopes
        ):
            self._clear_abandoned_agent_scopes_locked()
        agent = scoped_agent or parent_agent or self._current_agent_locked()
        if event_type.endswith("_START") and call_id and agent is not None:
            self._call_agents[call_id] = agent
            if scoped_agent is not None:
                self._agent_scopes[call_id] = scoped_agent
        return agent

    def _finish_agent_event_locked(self, event_type: str, call_id: str) -> None:
        if not event_type.endswith("_END") or not call_id:
            return
        self._call_agents.pop(call_id, None)
        self._agent_scopes.pop(call_id, None)

    def observe(self, step: Any) -> None:
        """Capture event metadata only; never persist inputs, outputs, or arguments."""
        try:
            event_type = _event_type_name(step)
            event = _CALL_EVENTS.get(event_type)
            timestamp = float(_step_value(step, "event_timestamp", time.time()))
            name = _safe_event_name(_step_value(step, "name"))
            call_id = str(_step_value(step, "UUID", "") or "")

            with self._lock:
                self._events_observed += 1
                agent = self._agent_for_event_locked(step, event_type, name, call_id)
                try:
                    if event is None:
                        return
                    if event_type in {"LLM_START", "LLM_END"}:
                        if (scope := _trajectory_scope(step)) is not None:
                            self._llm_scopes.add(scope)
                    elif event_type in {"TOOL_START", "TOOL_END"}:
                        if event_type == "TOOL_START" and call_id:
                            self._tool_span_ids.add(call_id)
                            if _trajectory_scope(step) in self._llm_scopes:
                                self._visible_tool_ids.add(call_id)
                        is_model_visible = call_id in self._visible_tool_ids
                        if event_type == "TOOL_END":
                            self._tool_span_ids.discard(call_id)
                            self._visible_tool_ids.discard(call_id)
                        if not is_model_visible:
                            return
                    elif event_type in {"FUNCTION_START", "FUNCTION_END"}:
                        parent_id = str(_step_value(step, "parent_id", "") or "")
                        is_tool_wrapper = parent_id in self._tool_span_ids
                        if event_type == "FUNCTION_START" and is_tool_wrapper:
                            if call_id:
                                self._hidden_function_ids.add(call_id)
                            return
                        if event_type == "FUNCTION_END":
                            is_hidden_function = call_id in self._hidden_function_ids
                            self._hidden_function_ids.discard(call_id)
                            if is_hidden_function or is_tool_wrapper:
                                return

                    kind, state = event
                    if state == "start":
                        self._calls[kind]["started"] += 1
                        if call_id:
                            self._active_names[kind][call_id] = name
                    else:
                        self._calls[kind]["completed"] += 1
                        if call_id:
                            name = name or self._active_names[kind].get(call_id)
                            self._active_names[kind].pop(call_id, None)

                    compact_event = {
                        "timestamp": _format_timestamp(timestamp),
                        "elapsed": _format_duration(timestamp - self._started_at_epoch),
                        "event_type": event_type,
                        "name": name,
                    }
                    if agent is not None:
                        compact_event["agent"] = agent
                    self._last_event = compact_event
                    line = json.dumps(compact_event, ensure_ascii=False, separators=(",", ":")) + "\n"
                    self._best_effort_write(lambda: self._append_event_locked(line))
                    self._write_state_locked()
                finally:
                    self._finish_agent_event_locked(event_type, call_id)
        except Exception as exc:  # Progress reporting must never fail the workflow.
            if not self._warned:
                LOGGER.warning("Unable to summarize an AI-Q progress event: %s", _safe_error(exc))
                self._warned = True

    def finish(self, status: str, **details: Any) -> None:
        with self._lock:
            self._status = status
            self._phase = status
            self._details.update(details)
            self._write_state_locked()


async def _heartbeat_progress(progress: _LiveProgress) -> None:
    while True:
        await asyncio.sleep(PROGRESS_HEARTBEAT_SEC)
        progress.heartbeat()


def _extract_text(result: Any) -> str:
    """Extract the final text without stringifying an unknown response object."""
    if isinstance(result, str):
        text = result
    else:
        try:
            from nat.data_models.api_server import ChatResponse
        except ImportError:  # pragma: no cover - NAT always exists in the runtime image.
            ChatResponse = ()  # type: ignore[assignment,misc]

        if isinstance(result, ChatResponse):
            if not result.choices:
                raise ValueError("AI-Q returned a ChatResponse with no choices")
            text = result.choices[0].message.content or ""
        else:
            raise TypeError(f"Unsupported AI-Q workflow result type: {type(result).__name__}")

    text = text.strip()
    if not text:
        raise ValueError("AI-Q workflow returned an empty response")
    return text


def _data_source_map(config: Any) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for function_config in config.functions.values():
        if getattr(function_config, "type", None) != "data_source_registry":
            continue
        for source in getattr(function_config, "sources", []):
            source_id = str(getattr(source, "id", "")).strip()
            if source_id:
                sources[source_id] = [str(tool) for tool in getattr(source, "tools", [])]
    return sources


def _runtime_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _runtime_value(item) for key, item in value.items()}
    enum_value = getattr(value, "value", _MISSING)
    if enum_value is not _MISSING:
        return _runtime_value(enum_value)
    return str(value)


def _component_runtime_settings(
    components: Any,
    fields_by_type: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    for name, component in components.items():
        component_type = str(getattr(component, "type", type(component).__name__))
        fields = fields_by_type.get(component_type, ()) if fields_by_type is not None else _LLM_RUNTIME_FIELDS
        values = {
            field: _runtime_value(value) for field in fields if (value := getattr(component, field, None)) is not None
        }
        if values:
            settings[str(name)] = {
                "type": component_type,
                **values,
            }
    return settings


def _runtime_summary(config: Any) -> dict[str, Any]:
    return {
        "aiq_version": _package_version("aiq-agent"),
        "nat_version": _package_version("nvidia-nat"),
        "workflow_type": getattr(config.workflow, "type", type(config.workflow).__name__),
        "functions": sorted(config.functions),
        "function_groups": sorted(config.function_groups),
        "data_sources": _data_source_map(config),
        "llm_settings": _component_runtime_settings(config.llms),
        "function_settings": _component_runtime_settings(
            config.functions,
            _FUNCTION_RUNTIME_FIELDS,
        ),
        "plugins": _plugin_inventory(),
    }


def _write_trajectory(steps: list[Any], output_path: Path, session_id: str) -> None:
    from nat.utils.atif_converter import IntermediateStepToATIFConverter

    trajectory = IntermediateStepToATIFConverter().convert(
        _trajectory_steps_for_atif(steps),
        session_id=session_id,
        agent_name="aiq-harbor",
    )
    aiq_version = _package_version("aiq-agent")
    if aiq_version:
        trajectory.agent.version = aiq_version
    _atomic_write_json(output_path, trajectory.to_json_dict(exclude_none=True))


async def _load_workflow(config_file: Path):
    from nat.runtime.loader import load_workflow

    workflow_context = load_workflow(config_file, max_concurrency=1)
    session_manager = await workflow_context.__aenter__()
    return workflow_context, session_manager, session_manager.config


async def validate_configuration(args: argparse.Namespace) -> None:
    config_file = Path(args.config_file)
    metadata_path = Path(args.metadata_output)
    started = time.time()
    metadata: dict[str, Any] = {
        "status": "validating",
        "config_file": str(config_file),
        "started_at_epoch": started,
    }
    _atomic_write_json(metadata_path, metadata)

    workflow_context = None
    try:
        workflow_context, _, config = await _load_workflow(config_file)
        metadata.update(_runtime_summary(config))
        metadata.update(
            {
                "status": "valid",
                "duration_sec": time.time() - started,
            }
        )
        _atomic_write_json(metadata_path, metadata)
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": _safe_error(exc),
                "duration_sec": time.time() - started,
            }
        )
        _atomic_write_json(metadata_path, metadata)
        raise
    finally:
        if workflow_context is not None:
            await workflow_context.__aexit__(None, None, None)


async def run_workflow(args: argparse.Namespace) -> None:
    config_file = Path(args.config_file)
    instruction_file = Path(args.instruction_file)
    output_file = Path(args.output_file)
    trajectory_file = Path(args.trajectory_output)
    metadata_file = Path(args.metadata_output)
    state_file = Path(args.state_output)
    events_output = getattr(args, "events_output", None)
    events_file = Path(events_output) if events_output else state_file.with_name("aiq_events.jsonl")
    session_id = args.session_id or str(uuid.uuid4())
    started = time.time()
    metadata: dict[str, Any] = {
        "status": "running",
        "session_id": session_id,
        "config_file": str(config_file),
        "output_file": str(output_file),
        "started_at_epoch": started,
    }
    _atomic_write_json(metadata_file, metadata)
    progress = _LiveProgress(
        state_path=state_file,
        events_path=events_file,
        session_id=session_id,
        started_at_epoch=started,
    )
    heartbeat_task = asyncio.create_task(_heartbeat_progress(progress))

    workflow_context = None
    captured_steps: list[Any] = []
    try:
        instruction = instruction_file.read_text(encoding="utf-8")
        if not instruction.strip():
            raise ValueError("Harbor instruction file is empty")

        workflow_context, session_manager, config = await _load_workflow(config_file)
        metadata.update(_runtime_summary(config))
        progress.set_phase("workflow", workflow_type=metadata.get("workflow_type"))

        # Keep Harbor session IDs in job metadata without setting NAT conversation context.
        # AI-Q knowledge retrieval treats conversation_id as a session collection name.
        async with session_manager.session(user_id="harbor") as session:
            async with session.run(instruction) as runner:

                def capture_step(step: Any) -> None:
                    captured_steps.append(step)
                    progress.observe(step)

                subscription = runner.context.intermediate_step_manager.subscribe(on_next=capture_step)
                try:
                    result = await runner.result(to_type=str)
                finally:
                    subscription.unsubscribe()

        progress.set_phase("finalizing")
        output = _extract_text(result)
        _write_trajectory(captured_steps, trajectory_file, session_id)
        _atomic_write_text(output_file, output + "\n")
        progress.finish(
            "completed",
            workflow_type=metadata.get("workflow_type"),
            result_type=type(result).__name__,
            output_chars=len(output),
            intermediate_steps=len(captured_steps),
        )
        metadata.update(
            {
                "status": "completed",
                "output_chars": len(output),
                "intermediate_steps": len(captured_steps),
                "duration_sec": time.time() - started,
            }
        )
        _atomic_write_json(metadata_file, metadata)
    except Exception as exc:
        trajectory_error = None
        if captured_steps:
            try:
                _write_trajectory(captured_steps, trajectory_file, session_id)
            except Exception as trajectory_exc:  # Preserve the original workflow error.
                trajectory_error = _safe_error(trajectory_exc)
        metadata.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": _safe_error(exc),
                "intermediate_steps": len(captured_steps),
                "duration_sec": time.time() - started,
            }
        )
        if trajectory_error:
            metadata["trajectory_error"] = trajectory_error
        progress.finish(
            "failed",
            error_type=type(exc).__name__,
            intermediate_steps=len(captured_steps),
        )
        _atomic_write_json(metadata_file, metadata)
        raise
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        if workflow_context is not None:
            await workflow_context.__aexit__(None, None, None)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--metadata-output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Load and validate an AI-Q profile")
    _add_common_arguments(validate_parser)

    run_parser = subparsers.add_parser("run", help="Execute one AI-Q workflow invocation")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--instruction-file", required=True)
    run_parser.add_argument("--output-file", required=True)
    run_parser.add_argument("--trajectory-output", required=True)
    run_parser.add_argument("--state-output", required=True)
    run_parser.add_argument("--events-output")
    run_parser.add_argument("--session-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("NVIDIA_NAT_LOG_LEVEL", "WARNING").upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            asyncio.run(validate_configuration(args))
        else:
            asyncio.run(run_workflow(args))
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, _safe_error(exc))
        return 1
    return 0


def _exit_one_shot_process(status: int) -> None:
    """Exit without waiting on service-lifetime dependency threads.

    AI-Q components such as the cached SQLite checkpointer intentionally keep
    non-daemon worker threads alive for a long-running server. This runner is a
    one-shot Harbor subprocess: all workflow cleanup and artifact writes have
    completed when ``main`` returns, so interpreter shutdown must not wait for
    those service-lifetime threads.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    _exit_one_shot_process(main())
