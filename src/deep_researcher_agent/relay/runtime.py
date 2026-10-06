# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NeMo Relay framework integration helpers."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import islice
from typing import Any
from typing import TypeVar
from uuid import uuid4

import nemo_relay
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import messages_to_dict
from langchain_core.runnables import RunnableBinding
from langchain_core.runnables.config import merge_configs
from nemo_relay.integrations.langchain import NemoRelayMiddleware
from pydantic import BaseModel

_T = TypeVar("_T")
_deep_researcher_scope_active: ContextVar[bool] = ContextVar("deep_researcher_relay_scope_active", default=False)
logger = logging.getLogger(__name__)
_SAFE_VALUE_MAX_DEPTH = 12
_SAFE_VALUE_MAX_ITEMS = 100
_SAFE_VALUE_MAX_STRING_LENGTH = 16_384


@dataclass
class _AgentScopeLifecycle:
    handle: Any
    output: Any = None


def _log_capture_failure(operation: str, error: Exception) -> None:
    """Report Relay capture failures without exposing payloads or changing execution."""
    logger.warning("NeMo Relay: %s failed (error_type=%s)", operation, type(error).__name__)


@dataclass
class _NamedModelAdapter:
    """Give otherwise valid chat-model implementations a Relay call name."""

    wrapped: Any
    model_name: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


def _normalize_chat_nvidia_binding(
    runnable: Any,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Expose a bound ChatNVIDIA model to Relay without losing bound tools.

    Relay 0.7.3 handles propagation headers through ``ChatNVIDIA.default_headers``
    only when the model is a direct ChatNVIDIA instance. LangChain's
    ``bind_tools()`` returns a RunnableBinding, which otherwise makes Relay fall
    back to the unsupported ``extra_headers`` model parameter.
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError:
        return runnable, {}, config

    if (
        not isinstance(runnable, RunnableBinding)
        or not isinstance(runnable.bound, ChatNVIDIA)
        or runnable.config_factories
    ):
        return runnable, {}, config

    return runnable.bound, dict(runnable.kwargs), merge_configs(runnable.config, config)


def deepagents_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Attach Relay's supported DeepAgents middleware."""

    from nemo_relay.integrations.deepagents import add_nemo_relay_integration

    return add_nemo_relay_integration(kwargs)


def merge_langchain_middleware(middleware: Sequence[Any] | None) -> list[Any]:
    """Attach Relay managed execution to an application-owned LangChain agent."""
    merged = list(middleware or ())
    if not any(isinstance(item, NemoRelayMiddleware) for item in merged):
        merged.insert(0, NemoRelayMiddleware())
    return merged


async def ainvoke_with_relay(
    runnable: Any,
    input_value: Any,
    *,
    callbacks: Sequence[Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Run a direct LangChain model call through Relay's maintained middleware."""
    effective_config = dict(config or {})
    configured_callbacks = callbacks if callbacks is not None else effective_config.get("callbacks")
    if configured_callbacks:
        effective_config["callbacks"] = list(configured_callbacks)
    else:
        effective_config.pop("callbacks", None)
    if isinstance(input_value, str | BaseMessage):
        messages = [input_value]
    else:
        messages = list(input_value)
    system_message = (
        messages.pop(0) if messages and isinstance(messages[0], BaseMessage) and messages[0].type == "system" else None
    )
    model, model_settings, effective_config = _normalize_chat_nvidia_binding(runnable, effective_config)
    named_source = getattr(model, "bound", model)
    resolved_name = next(
        (
            value
            for attribute in ("model", "model_name", "model_id", "deployment_name")
            if isinstance(value := getattr(named_source, attribute, None), str) and value
        ),
        None,
    )
    if not any(
        isinstance(getattr(model, attribute, None), str) and getattr(model, attribute)
        for attribute in ("model", "model_name", "model_id", "deployment_name")
    ):
        model = _NamedModelAdapter(model, resolved_name or type(model).__name__)
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=system_message,
        model_settings=model_settings,
    )

    async def invoke_call(next_request: ModelRequest[Any]) -> ModelResponse[Any]:
        next_messages = list(next_request.messages)
        if next_request.system_message is not None:
            next_messages.insert(0, next_request.system_message)
        parameters = inspect.signature(next_request.model.ainvoke).parameters
        accepts_config = "config" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs = dict(next_request.model_settings)
        if isinstance(next_request.model, _NamedModelAdapter) and not isinstance(
            next_request.model.wrapped,
            BaseChatModel,
        ):
            kwargs = {}
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        if accepts_config:
            response = await next_request.model.ainvoke(next_messages, config=effective_config, **kwargs)
        else:
            response = await next_request.model.ainvoke(next_messages, **kwargs)
        if not isinstance(response, BaseMessage):
            content = getattr(response, "content", None)
            if not isinstance(content, str | list):
                message = f"Relay-managed LangChain model returned {type(response).__name__}, expected BaseMessage"
                raise TypeError(message)
            response = AIMessage(content=content)
        return ModelResponse(result=[response])

    invocation_started = False
    invocation_error: BaseException | None = None

    async def invoke(next_request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal invocation_error
        nonlocal invocation_started
        invocation_started = True
        try:
            return await invoke_call(next_request)
        except BaseException as error:
            invocation_error = error
            raise

    try:
        response = await NemoRelayMiddleware().awrap_model_call(request, invoke)
    except Exception as error:
        if invocation_started:
            if invocation_error is not None:
                raise invocation_error
            raise
        _log_capture_failure("model middleware", error)
        response = await invoke(request)
    if not response.result:
        raise RuntimeError("Relay-managed LangChain model returned no messages")
    return response.result[-1]


async def awrap_tool_call_with_relay(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[_T]]
) -> _T:
    """Capture a LangChain tool call while preserving execution if Relay setup fails."""
    invocation_started = False
    invocation_error: BaseException | None = None

    async def invoke(next_request: ToolCallRequest) -> _T:
        nonlocal invocation_error
        nonlocal invocation_started
        invocation_started = True
        try:
            return await handler(next_request)
        except BaseException as error:
            invocation_error = error
            raise

    try:
        return await NemoRelayMiddleware().awrap_tool_call(request, invoke)
    except Exception as error:
        if invocation_started:
            if invocation_error is not None:
                raise invocation_error
            raise
        _log_capture_failure("tool middleware", error)
        return await invoke(request)


async def ainvoke_tool_with_relay(tool: Any, args: dict[str, Any]) -> Any:
    """Run a direct LangChain tool call through Relay's maintained middleware."""
    request = ToolCallRequest(
        tool_call={"name": tool.name, "args": args, "id": f"deep-researcher-{uuid4()}"},
        tool=tool,
        state={},
        runtime=None,
    )

    async def invoke_call(next_request: ToolCallRequest) -> Any:
        if next_request.tool is None:
            raise RuntimeError(f"Relay-managed tool {next_request.tool_call['name']!r} is unavailable")
        return await next_request.tool.ainvoke(next_request.tool_call.get("args") or {})

    return await awrap_tool_call_with_relay(request, invoke_call)


@contextmanager
def _semantic_scope(
    name: str,
    scope_type: Any,
    component_type: str,
    *,
    session_id: str | None = None,
    input_value: Any = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a semantic scope and mark nested Deep Researcher Agent scope execution."""
    scope_token = None
    if not _deep_researcher_scope_active.get():
        scope_token = _deep_researcher_scope_active.set(True)
    scope_metadata = {
        "deep-researcher.component.name": name,
        "deep-researcher.component.type": component_type,
        "deep-researcher.framework": "nemo-agent-toolkit",
    }
    scope_metadata.update(metadata or {})
    if session_id:
        scope_metadata["session_id"] = session_id
    lifecycle = _AgentScopeLifecycle(None)
    status_metadata: dict[str, Any] = {"otel.status_code": "UNSET"}
    try:
        try:
            lifecycle.handle = nemo_relay.scope.push(
                name,
                scope_type,
                metadata=_safe_value(scope_metadata),
                input=_safe_value(input_value) if input_value is not None else None,
            )
        except Exception as capture_error:
            _log_capture_failure("semantic scope start", capture_error)
        try:
            yield lifecycle
        except BaseException as error:
            status_metadata = {
                "error_type": type(error).__name__,
                "otel.status_code": "ERROR",
                "otel.status_description": type(error).__name__,
            }
            raise
        else:
            status_metadata = {"otel.status_code": "OK"}
    finally:
        try:
            if lifecycle.handle is not None:
                try:
                    output = _safe_value(lifecycle.output) if lifecycle.output is not None else None
                    nemo_relay.scope.pop(lifecycle.handle, output=output, metadata=status_metadata)
                except Exception as capture_error:
                    _log_capture_failure("semantic scope end", capture_error)
        finally:
            if scope_token is not None:
                _deep_researcher_scope_active.reset(scope_token)


@contextmanager
def agent_scope(name: str, *, session_id: str | None = None, input_value: Any = None):
    """Create an Agent scope around an application-owned agent boundary."""
    with _semantic_scope(
        name,
        nemo_relay.ScopeType.Agent,
        "agent",
        session_id=session_id,
        input_value=input_value,
    ) as lifecycle:
        yield lifecycle


@contextmanager
def workflow_scope(
    name: str,
    *,
    session_id: str | None = None,
    input_value: Any = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a NAT workflow scope above application-owned agent scopes."""
    with _semantic_scope(
        name,
        nemo_relay.ScopeType.Function,
        "workflow",
        session_id=session_id,
        input_value=input_value,
        metadata=metadata,
    ) as lifecycle:
        yield lifecycle


async def run_agent(
    name: str,
    operation: Callable[[], Awaitable[_T]],
    *,
    session_id: str | None = None,
    input_value: Any = None,
) -> _T:
    """Run an agent with a fresh stack at a request boundary and shared stack when nested."""

    async def _run() -> _T:
        with agent_scope(name, session_id=session_id, input_value=input_value) as lifecycle:
            result = await operation()
            lifecycle.output = result
            return result

    return await _run_at_request_boundary(_run)


async def run_workflow(
    name: str,
    operation: Callable[[], Awaitable[_T]],
    *,
    session_id: str | None = None,
    input_value: Any = None,
    metadata: dict[str, Any] | None = None,
) -> _T:
    """Run one NAT request as a Relay workflow root with semantic input and output."""

    async def _run() -> _T:
        with workflow_scope(
            name,
            session_id=session_id,
            input_value=input_value,
            metadata=metadata,
        ) as lifecycle:
            result = await operation()
            lifecycle.output = result
            return result

    return await _run_at_request_boundary(_run)


async def _run_at_request_boundary(operation: Callable[[], Awaitable[_T]]) -> _T:
    """Reuse a nested scope or isolate a new request on its own task and stack."""
    if _deep_researcher_scope_active.get():
        return await operation()

    async def _run_isolated() -> _T:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            return await operation()

    return await asyncio.create_task(_run_isolated())


def _safe_value(value: Any, *, _depth: int = 0) -> Any:
    """Project framework state to JSON-compatible Relay event values."""
    if _depth >= _SAFE_VALUE_MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, str):
        return value[:_SAFE_VALUE_MAX_STRING_LENGTH]
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, BaseMessage):
        return _safe_value(messages_to_dict([value])[0], _depth=_depth + 1)
    if isinstance(value, BaseModel):
        return _safe_value(value.model_dump(mode="json"), _depth=_depth + 1)
    if isinstance(value, dict):
        return {
            str(key)[:_SAFE_VALUE_MAX_STRING_LENGTH]: _safe_value(item, _depth=_depth + 1)
            for key, item in islice(value.items(), _SAFE_VALUE_MAX_ITEMS)
        }
    if isinstance(value, list | tuple | set):
        return [_safe_value(item, _depth=_depth + 1) for item in islice(value, _SAFE_VALUE_MAX_ITEMS)]
    return {"type": type(value).__name__}
