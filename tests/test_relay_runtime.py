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

import asyncio
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from uuid import uuid4

import nemo_relay
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from nemo_relay import plugin
from nemo_relay.integrations.deepagents import NemoRelayDeepAgentsCallbackHandler
from nemo_relay.integrations.langchain._serialization import payload_to_model_request
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pydantic import ValidationError

from deep_researcher_agent.agents.deep_researcher.models import ResearchQuery
from deep_researcher_agent.agents.deep_researcher.tools.research import _run_research_queries
from deep_researcher_agent.relay.bootstrap import ensure_started
from deep_researcher_agent.relay.bootstrap import shutdown_async
from deep_researcher_agent.relay.config import RelayConfig
from deep_researcher_agent.relay.config import RelayOpenTelemetryEndpointConfig
from deep_researcher_agent.relay.logging import log_event
from deep_researcher_agent.relay.privacy import request_privacy_context
from deep_researcher_agent.relay.runtime import _normalize_chat_nvidia_binding
from deep_researcher_agent.relay.runtime import _safe_value
from deep_researcher_agent.relay.runtime import ainvoke_tool_with_relay
from deep_researcher_agent.relay.runtime import ainvoke_with_relay
from deep_researcher_agent.relay.runtime import deepagents_kwargs
from deep_researcher_agent.relay.runtime import merge_langchain_middleware
from deep_researcher_agent.relay.runtime import run_agent
from deep_researcher_agent.relay.runtime import run_workflow


def test_deepagents_integration_is_enabled() -> None:
    kwargs = deepagents_kwargs(
        {
            "model": "test",
            "tools": [],
            "name": "test-agent",
            "subagents": [{"name": "runtime-agent", "description": "test", "model": "test", "tools": []}],
        }
    )
    assert [type(middleware).__name__ for middleware in kwargs["middleware"][-1:]] == ["NemoRelayDeepAgentsMiddleware"]
    assert [type(middleware).__name__ for middleware in kwargs["subagents"][0]["middleware"][-1:]] == [
        "NemoRelayDeepAgentsMiddleware"
    ]


def test_langchain_managed_execution_middleware_is_enabled() -> None:
    middleware = merge_langchain_middleware([])

    assert [type(item).__name__ for item in middleware] == ["NemoRelayMiddleware"]


def test_bound_chat_nvidia_uses_supported_relay_header_path() -> None:
    from langchain.agents.middleware import ModelRequest
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    @tool
    def search(query: str) -> str:
        """Search for a query."""
        return query

    model = ChatNVIDIA(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://example.invalid/v1",
    )
    bound_model = model.bind_tools([search], parallel_tool_calls=False)
    normalized, model_settings, config = _normalize_chat_nvidia_binding(
        bound_model,
        {"tags": ["request"]},
    )

    assert normalized is model
    assert model_settings["tools"][0]["function"]["name"] == "search"
    assert model_settings["parallel_tool_calls"] is False
    assert config["tags"] == ["request"]

    request = ModelRequest(
        model=normalized,
        messages=[HumanMessage(content="test")],
        model_settings=model_settings,
    )
    relay_request = nemo_relay.LLMRequest(
        {"traceparent": "00-test-trace-test-span-01"},
        {"model_settings": model_settings},
    )
    converted = payload_to_model_request(request, relay_request)

    assert isinstance(converted.model, ChatNVIDIA)
    assert converted.model.default_headers["traceparent"] == "00-test-trace-test-span-01"
    assert "extra_headers" not in converted.model_settings
    assert converted.model_settings["tools"][0]["function"]["name"] == "search"


def test_relay_logging_subscriber_matches_verbose_trace_labels(caplog) -> None:
    caplog.set_level("INFO")
    root_uuid = str(uuid4())
    llm_uuid = str(uuid4())
    tool_uuid = str(uuid4())
    researcher_uuid = str(uuid4())
    events = [
        SimpleNamespace(
            kind="scope",
            category="agent",
            name="test-agent",
            scope_category="start",
            metadata={},
            uuid=root_uuid,
            parent_uuid=None,
        ),
        SimpleNamespace(
            kind="scope",
            category="llm",
            name="ChatNVIDIA",
            scope_category="start",
            metadata={},
            uuid=llm_uuid,
            parent_uuid=root_uuid,
            data={"messages": "redacted Relay input"},
            category_profile={},
        ),
        SimpleNamespace(
            kind="scope",
            category="llm",
            name="ChatNVIDIA",
            scope_category="end",
            metadata={"otel.status_code": "OK"},
            uuid=llm_uuid,
            parent_uuid=root_uuid,
            data={
                "generations": [
                    [
                        {
                            "message": {
                                "content": "redacted Relay response",
                                "additional_kwargs": {"reasoning_content": "redacted Relay reasoning"},
                                "response_metadata": {
                                    "model_name": "test-model",
                                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 4},
                                },
                            }
                        }
                    ]
                ]
            },
            category_profile={
                "model_name": "test-model",
                "annotated_response": {
                    "tool_calls": [
                        {
                            "name": "web_search",
                            "arguments": {"query": "redacted Relay query"},
                        }
                    ]
                },
            },
        ),
        SimpleNamespace(
            kind="scope",
            category="tool",
            name="web_search",
            scope_category="start",
            metadata={},
            uuid=tool_uuid,
            parent_uuid=root_uuid,
            data={"query": "redacted Relay query"},
        ),
        SimpleNamespace(
            kind="scope",
            category="tool",
            name="web_search",
            scope_category="end",
            metadata={"otel.status_code": "OK"},
            uuid=tool_uuid,
            parent_uuid=root_uuid,
            data={"result": "redacted Relay result"},
        ),
        SimpleNamespace(
            kind="scope",
            category="agent",
            name="researcher-agent",
            scope_category="start",
            uuid=researcher_uuid,
            parent_uuid=root_uuid,
            metadata={},
        ),
        SimpleNamespace(
            kind="scope",
            category="agent",
            name="researcher-agent",
            scope_category="end",
            uuid=researcher_uuid,
            parent_uuid=root_uuid,
            metadata={"otel.status_code": "OK"},
        ),
        SimpleNamespace(
            kind="scope",
            category="agent",
            name="test-agent",
            scope_category="end",
            metadata={"otel.status_code": "OK"},
            uuid=root_uuid,
            parent_uuid=None,
        ),
    ]
    for event in events:
        log_event(event)

    for label in (
        "[Chain Start] test-agent",
        "[AGENT]",
        "[Reasoning]",
        "[Agent Response]",
        "[Tool Calls] 1 tool(s) requested",
        "→ web_search",
        "Args: chars=33",
        "[Tokens] prompt=10, completion=4, model=test-model",
        "[Tool Start] web_search",
        "[Tool Result]",
        "[Chain Start] researcher-agent",
        "[Chain End] researcher-agent",
        "[Chain End] test-agent",
    ):
        assert label in caplog.text
    assert "[Relay Scope]" not in caplog.text
    assert "[Relay LLM]" not in caplog.text
    assert "[Relay Tool]" not in caplog.text


def test_safe_value_projects_pydantic_like_state() -> None:
    class State:
        pass

    assert _safe_value({"state": State()}) == {"state": {"type": "State"}}


def test_safe_value_bounds_nested_and_large_values() -> None:
    nested: object = "leaf"
    for _ in range(20):
        nested = {"nested": nested}

    projected = _safe_value({"nested": nested, "items": list(range(101)), "text": "x" * 20_000})

    assert len(projected["items"]) < 101
    assert len(projected["text"]) < 20_000
    current = projected["nested"]
    while "nested" in current:
        current = current["nested"]
    assert current["truncated"] is True


@pytest.mark.asyncio
async def test_relay_model_call_accepts_scalar_inputs(monkeypatch) -> None:
    calls = []

    class Model:
        model_name = "test-model"

        async def ainvoke(self, messages, config=None):  # noqa: ARG002
            calls.append(messages)
            return AIMessage(content="done")

    async def passthrough(_self, request, handler):
        return await handler(request)

    monkeypatch.setattr("deep_researcher_agent.relay.runtime.NemoRelayMiddleware.awrap_model_call", passthrough)
    message = HumanMessage(content="question")

    await ainvoke_with_relay(Model(), "question")
    await ainvoke_with_relay(Model(), message)

    assert calls == [["question"], [message]]


@pytest.mark.asyncio
async def test_relay_model_name_comes_from_bound_model(monkeypatch) -> None:
    observed_names: list[str] = []

    class BoundModel:
        model_name = "provider-model"

    class Binding:
        bound = BoundModel()

        async def ainvoke(self, messages, config=None):  # noqa: ARG002
            return AIMessage(content="done")

    async def capture_name(_self, request, handler):
        observed_names.append(request.model.model_name)
        return await handler(request)

    monkeypatch.setattr("deep_researcher_agent.relay.runtime.NemoRelayMiddleware.awrap_model_call", capture_name)

    await ainvoke_with_relay(Binding(), [])

    assert observed_names == ["provider-model"]


@pytest.mark.asyncio
async def test_relay_model_middleware_fallback_does_not_retry_started_calls(monkeypatch, caplog) -> None:
    calls = 0

    class Model:
        model_name = "test-model"

        async def ainvoke(self, messages, config=None):  # noqa: ARG002
            nonlocal calls
            calls += 1
            return AIMessage(content="done")

    async def fail_before(_self, request, handler):  # noqa: ARG001
        raise RuntimeError("private middleware detail")

    monkeypatch.setattr("deep_researcher_agent.relay.runtime.NemoRelayMiddleware.awrap_model_call", fail_before)
    assert (await ainvoke_with_relay(Model(), [])).content == "done"
    assert calls == 1
    assert "private middleware detail" not in caplog.text

    async def fail_after(_self, request, handler):
        await handler(request)
        raise RuntimeError("post-invocation failure")

    monkeypatch.setattr("deep_researcher_agent.relay.runtime.NemoRelayMiddleware.awrap_model_call", fail_after)
    with pytest.raises(RuntimeError, match="post-invocation failure"):
        await ainvoke_with_relay(Model(), [])
    assert calls == 2


@pytest.mark.asyncio
async def test_relay_tool_middleware_fallback_does_not_retry_started_calls(monkeypatch, caplog) -> None:
    calls = 0

    class Tool:
        name = "test_tool"

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            return args["value"]

    async def fail_before(_self, request, handler):  # noqa: ARG001
        raise RuntimeError("private middleware detail")

    monkeypatch.setattr("deep_researcher_agent.relay.runtime.NemoRelayMiddleware.awrap_tool_call", fail_before)
    assert await ainvoke_tool_with_relay(Tool(), {"value": "done"}) == "done"
    assert calls == 1
    assert "private middleware detail" not in caplog.text

    async def fail_after(_self, request, handler):
        await handler(request)
        raise RuntimeError("post-invocation failure")

    monkeypatch.setattr("deep_researcher_agent.relay.runtime.NemoRelayMiddleware.awrap_tool_call", fail_after)
    with pytest.raises(RuntimeError, match="post-invocation failure"):
        await ainvoke_tool_with_relay(Tool(), {"value": "done"})
    assert calls == 2


@pytest.mark.asyncio
async def test_callback_does_not_duplicate_middleware_managed_llm_and_tool_scopes(tmp_path: Path) -> None:
    class TestChatModel(FakeMessagesListChatModel):
        model_name: str = "managed-model"

    @tool
    def managed_tool(value: str) -> str:
        """Return the supplied value."""
        return value

    config = RelayConfig()
    config.logging = False
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "managed.jsonl"
    config.observability.opentelemetry.enabled = False

    async def operation() -> None:
        model = TestChatModel(
            responses=[
                AIMessage(
                    content="response",
                    response_metadata={"model_name": "managed-model"},
                    usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                )
            ]
        )
        response = await ainvoke_with_relay(
            model,
            [HumanMessage(content="request")],
        )
        assert response.content == "response"
        assert await ainvoke_tool_with_relay(managed_tool, {"value": "result"}) == "result"

    await ensure_started(config)
    try:
        await run_agent("managed-agent", operation)
    finally:
        await shutdown_async()

    events = [json.loads(line) for line in (tmp_path / "managed.jsonl").read_text().splitlines()]
    starts = [event for event in events if event["kind"] == "scope" and event["scope_category"] == "start"]
    assert [(event["category"], event["name"]) for event in starts] == [
        ("agent", "managed-agent"),
        ("llm", "managed-model"),
        ("tool", "managed_tool"),
    ]
    llm_end = next(event for event in events if event["category"] == "llm" and event["scope_category"] == "end")
    assert llm_end["category_profile"]["model_name"] == "managed-model"
    assert llm_end["category_profile"]["annotated_response"]["usage"] == {
        "completion_tokens": 4,
        "prompt_tokens": 10,
        "total_tokens": 14,
    }


@pytest.mark.asyncio
async def test_concurrent_researchers_do_not_share_mutable_relay_agent_scopes(tmp_path: Path) -> None:
    config = RelayConfig()
    config.logging = False
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "concurrent-researchers.jsonl"
    config.observability.opentelemetry.enabled = False

    class Researcher:
        async def ainvoke(self, state, config=None):  # noqa: ARG002
            await asyncio.sleep(0.01 if "slow" in state["messages"][0].content else 0)
            return {
                "structured_response": {
                    "query_topic": "test",
                    "target_components": ["test"],
                    "summary": "test",
                    "findings": [],
                    "gaps": [],
                    "sources": [],
                    "narrative_notes": "test",
                    "language": "English",
                }
            }

    queries = [
        ResearchQuery(
            query=query,
            preferred_tools=["test_tool"],
            target_components=["test"],
            rationale="test",
        )
        for query in ("slow query", "fast query")
    ]

    async def operation() -> None:
        successful, notes, errors = await _run_research_queries(
            queries=queries,
            researcher_runnable=Researcher(),
            runtime=None,
            callbacks=[],
            max_concurrency=2,
        )
        assert successful == queries
        assert len(notes) == 2
        assert errors == []

    await ensure_started(config)
    try:
        await run_agent("deep_research_agent", operation)
    finally:
        await shutdown_async()

    events = [json.loads(line) for line in (tmp_path / "concurrent-researchers.jsonl").read_text().splitlines()]
    scope_events = [event for event in events if event["kind"] == "scope"]
    starts = [event for event in scope_events if event["scope_category"] == "start"]
    assert [event["name"] for event in starts] == [
        "deep_research_agent",
        "researcher-agent",
        "researcher-agent",
    ]
    researcher_scopes = [event for event in scope_events if event["name"] == "researcher-agent"]
    assert Counter(event["scope_category"] for event in researcher_scopes) == {"start": 2, "end": 2}
    assert len({event["uuid"] for event in researcher_scopes}) == 2
    assert {event["parent_uuid"] for event in researcher_scopes} == {starts[0]["uuid"]}


@pytest.mark.asyncio
async def test_semantic_scope_capture_failures_do_not_change_agent_execution(monkeypatch) -> None:
    operation_calls = 0

    async def operation() -> str:
        nonlocal operation_calls
        operation_calls += 1
        return "result"

    def fail_start(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("synthetic Relay start failure")

    monkeypatch.setattr(nemo_relay.scope, "push", fail_start)
    assert await run_agent("test-agent", operation) == "result"

    monkeypatch.setattr(nemo_relay.scope, "push", lambda *args, **kwargs: object())

    def fail_end(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("synthetic Relay end failure")

    monkeypatch.setattr(nemo_relay.scope, "pop", fail_end)
    assert await run_agent("test-agent", operation) == "result"
    assert operation_calls == 2


def test_relay_config_is_accepted_by_plugin_validator() -> None:
    report = plugin.validate(RelayConfig().to_plugin_config())

    assert not [diagnostic for diagnostic in report["diagnostics"] if diagnostic["level"] == "error"]


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": True},
        {"observability": {"unknown": True}},
        {"observability": {"opentelemetry": {"endpoints": [{"endpoint": "not-a-url"}]}}},
        {"redaction": {"request_privacy_attributes": ["input.value"]}},
    ],
)
def test_relay_config_rejects_unknown_or_invalid_values(value: dict) -> None:
    with pytest.raises(ValidationError):
        RelayConfig.model_validate(value)


@pytest.mark.asyncio
async def test_plugin_managed_atof_redacts_before_export(tmp_path: Path) -> None:
    config = RelayConfig()
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "events.jsonl"
    config.observability.opentelemetry.enabled = False

    await ensure_started(config)
    try:
        with nemo_relay.scope.scope(
            "redaction-test",
            nemo_relay.ScopeType.Agent,
            input={"email": "person@example.com"},
        ):
            nemo_relay.scope.event("secret", data={"api_key": "sk-1234567890abcdef"})  # pragma: allowlist secret
    finally:
        await shutdown_async()

    exported = (tmp_path / "events.jsonl").read_text()
    assert "redaction-test" in exported
    assert "person@example.com" not in exported
    assert "sk-1234567890abcdef" not in exported


@pytest.mark.asyncio
async def test_request_privacy_sanitizes_relay_without_changing_execution(tmp_path: Path) -> None:
    config = RelayConfig()
    config.logging = False
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "private-events.jsonl"
    config.observability.opentelemetry.enabled = False

    @tool
    def echo_private(value: str) -> str:
        """Echo a value for privacy testing."""
        return f"tool-result:{value}"

    class TestChatModel(FakeMessagesListChatModel):
        model_name: str = "test-model"

    model = TestChatModel(responses=[AIMessage(content="private-model-output")])

    async def operation() -> dict[str, str]:
        message = await ainvoke_with_relay(model, [HumanMessage(content="private-model-input")])
        tool_result = await ainvoke_tool_with_relay(echo_private, {"value": "private-tool-input"})
        return {"model": str(message.content), "tool": tool_result}

    await ensure_started(config)
    try:
        with request_privacy_context(True):
            result = await run_workflow("private-workflow", operation, input_value="private-workflow-input")
    finally:
        await shutdown_async()

    assert result == {"model": "private-model-output", "tool": "tool-result:private-tool-input"}
    exported = (tmp_path / "private-events.jsonl").read_text()
    for private_value in (
        "private-workflow-input",
        "private-model-input",
        "private-model-output",
        "private-tool-input",
        "tool-result:private-tool-input",
    ):
        assert private_value not in exported


@pytest.mark.asyncio
async def test_request_privacy_redacts_deepagents_error_description(tmp_path: Path) -> None:
    config = RelayConfig()
    config.logging = False
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "private-error.jsonl"
    config.observability.opentelemetry.enabled = False
    private_error = "proprietary-error-canary"

    await ensure_started(config)
    try:
        run_id = uuid4()
        with request_privacy_context(True), nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            callback = NemoRelayDeepAgentsCallbackHandler()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                name="DeepAgent",
                metadata={"lc_versions": {"deepagents": "test"}, "ls_integration": "deepagents"},
            )
            callback.on_chain_error(RuntimeError(private_error), run_id=run_id)
    finally:
        await shutdown_async()

    exported = (tmp_path / "private-error.jsonl").read_text()
    assert private_error not in exported
    events = [json.loads(line) for line in exported.splitlines()]
    assert [(event["name"], event["scope_category"]) for event in events] == [
        ("DeepAgent", "start"),
        ("DeepAgent", "end"),
    ]
    assert len({event["uuid"] for event in events}) == 1
    assert events[-1]["metadata"]["otel.status_code"] == "ERROR"
    assert events[-1]["metadata"]["otel.status_description"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_two_turn_parity_has_two_traces_one_session_no_duplicates_and_balanced_scopes(tmp_path: Path) -> None:
    @tool
    def echo(text: str) -> str:
        """Return the supplied text."""
        return text

    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            received.append(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    config = RelayConfig()
    config.logging = False
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "two-turns.jsonl"
    config.observability.opentelemetry.enabled = True
    config.observability.opentelemetry.endpoints = [
        RelayOpenTelemetryEndpointConfig(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
            timeout_millis=1000,
        )
    ]

    async def turn(turn_number: int) -> dict[str, int]:
        class TestChatModel(FakeMessagesListChatModel):
            model_name: str = "ChatNVIDIA"

        async def classify_intent() -> dict[str, str]:
            model = TestChatModel(
                responses=[
                    AIMessage(
                        content=f"answer-{turn_number}",
                        response_metadata={
                            "model_name": "test-model",
                            "token_usage": {"prompt_tokens": 10, "completion_tokens": 4},
                        },
                    )
                ]
            )
            await ainvoke_with_relay(model, [HumanMessage(content=f"question-{turn_number}")])
            return {"intent": "research"}

        await run_agent("intent_classifier", classify_intent, input_value={"turn": turn_number})

        await ainvoke_tool_with_relay(echo, {"text": f"turn-{turn_number}"})

        async def nested_agent() -> None:
            return None

        await run_agent("shallow_research_agent", nested_agent)
        return {"turn": turn_number}

    try:
        await ensure_started(config)
        await run_workflow(
            "<workflow>",
            lambda: run_agent(
                "chat_deepresearcher_agent",
                lambda: turn(1),
                input_value={"question": "question-1"},
            ),
            session_id="same-session",
            input_value={"question": "question-1"},
        )
        await run_workflow(
            "<workflow>",
            lambda: run_agent(
                "chat_deepresearcher_agent",
                lambda: turn(2),
                input_value={"question": "question-2"},
            ),
            session_id="same-session",
            input_value={"question": "question-2"},
        )
    finally:
        try:
            await shutdown_async()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            assert not server_thread.is_alive()

    events = [json.loads(line) for line in (tmp_path / "two-turns.jsonl").read_text().splitlines()]
    scope_events = [event for event in events if event["kind"] == "scope"]
    lifecycle_counts = Counter((event["uuid"], event["scope_category"]) for event in scope_events)
    scope_uuids = {event["uuid"] for event in scope_events}
    assert all(lifecycle_counts[(scope_uuid, phase)] == 1 for scope_uuid in scope_uuids for phase in ("start", "end"))

    root_starts = [
        event for event in scope_events if event["name"] == "<workflow>" and event["scope_category"] == "start"
    ]
    assert len(root_starts) == 2
    assert len({event["uuid"] for event in root_starts}) == 2
    assert {event["metadata"]["session_id"] for event in root_starts} == {"same-session"}
    assert {event["metadata"]["deep-researcher.framework"] for event in root_starts} == {"nemo-agent-toolkit"}
    assert all(event["data"] is not None for event in root_starts)
    root_ends = [event for event in scope_events if event["name"] == "<workflow>" and event["scope_category"] == "end"]
    assert all(event["data"] is not None for event in root_ends)
    assert sum(event["category"] == "llm" and event["scope_category"] == "start" for event in scope_events) == 2
    assert sum(event["category"] == "tool" and event["scope_category"] == "start" for event in scope_events) == 2
    assert not {"LangGraph", "tools_condition", "should_escalate", "agent", "tools"}.intersection(
        event["name"] for event in scope_events
    )
    classifier_starts = [
        event
        for event in scope_events
        if event["category"] == "agent" and event["name"] == "intent_classifier" and event["scope_category"] == "start"
    ]
    assert len(classifier_starts) == 2
    assert all(event["data"] is not None for event in classifier_starts)
    classifier_ends = [
        event
        for event in scope_events
        if event["category"] == "agent" and event["name"] == "intent_classifier" and event["scope_category"] == "end"
    ]
    assert all(event["data"] is not None for event in classifier_ends)
    root_uuids = {event["uuid"] for event in root_starts}
    agent_starts = [
        event
        for event in scope_events
        if event["category"] == "agent"
        and event["name"] == "chat_deepresearcher_agent"
        and event["scope_category"] == "start"
    ]
    assert len(agent_starts) == 2
    assert {event["parent_uuid"] for event in agent_starts} == root_uuids
    agent_uuids = {event["uuid"] for event in agent_starts}
    classifier_uuids = {event["uuid"] for event in classifier_starts}
    assert {event["parent_uuid"] for event in classifier_starts} == agent_uuids
    llm_starts = [event for event in scope_events if event["category"] == "llm" and event["scope_category"] == "start"]
    assert {event["parent_uuid"] for event in llm_starts} == classifier_uuids

    spans = []
    for body in received:
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        spans.extend(
            span
            for resource_spans in request.resource_spans
            for scope_spans in resource_spans.scope_spans
            for span in scope_spans.spans
        )
    assert len({span.trace_id for span in spans}) == 2
    root_spans = [span for span in spans if span.name == "<workflow>"]
    assert len(root_spans) == 2
    assert all(
        {"input.value", "output.value"}.issubset({attribute.key for attribute in span.attributes})
        for span in root_spans
    )
    assert all(
        any(
            attribute.key == "openinference.span.kind" and attribute.value.string_value == "CHAIN"
            for attribute in span.attributes
        )
        for span in root_spans
    )
    llm_spans = [span for span in spans if span.name == "ChatNVIDIA"]
    tool_spans = [span for span in spans if span.name == "echo"]
    assert len(llm_spans) == 2
    assert len(tool_spans) == 2
    assert all(
        {"input.value", "output.value"}.issubset({attribute.key for attribute in span.attributes})
        for span in [*llm_spans, *tool_spans]
    )
    session_ids = {
        attribute.value.string_value
        for span in root_spans
        for attribute in span.attributes
        if attribute.key == "session.id"
    }
    assert session_ids == {"same-session"}


@pytest.mark.asyncio
async def test_request_scope_closes_on_error_timeout_and_cancellation(tmp_path: Path) -> None:
    config = RelayConfig()
    config.logging = False
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.atof.filename = "interruptions.jsonl"
    config.observability.opentelemetry.enabled = False

    async def fail() -> None:
        raise RuntimeError("synthetic failure")

    async def wait_forever(started: asyncio.Event | None = None) -> None:
        if started is not None:
            started.set()
        await asyncio.Event().wait()

    await ensure_started(config)
    try:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            await run_agent("error-agent", fail, session_id="same-session")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                run_agent("timeout-agent", wait_forever, session_id="same-session"),
                timeout=0.01,
            )

        started = asyncio.Event()
        cancelled = asyncio.create_task(
            run_agent("cancelled-agent", lambda: wait_forever(started), session_id="same-session")
        )
        await started.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        await run_agent("recovery-agent", _completed_operation, session_id="same-session")
    finally:
        await shutdown_async()

    events = [json.loads(line) for line in (tmp_path / "interruptions.jsonl").read_text().splitlines()]
    roots = [event for event in events if event["kind"] == "scope" and event["category"] == "agent"]
    counts = Counter((event["uuid"], event["scope_category"]) for event in roots)
    assert {event["name"] for event in roots} == {
        "error-agent",
        "timeout-agent",
        "cancelled-agent",
        "recovery-agent",
    }
    assert all(counts[(scope_uuid, phase)] == 1 for scope_uuid, _ in counts for phase in ("start", "end"))
    end_status = {
        event["name"]: event["metadata"]["otel.status_code"] for event in roots if event["scope_category"] == "end"
    }
    assert end_status == {
        "error-agent": "ERROR",
        "timeout-agent": "ERROR",
        "cancelled-agent": "ERROR",
        "recovery-agent": "OK",
    }


async def _completed_operation() -> None:
    return None


@pytest.mark.asyncio
async def test_plugin_managed_otel_exports_protobuf_trace(tmp_path: Path) -> None:
    received: list[tuple[str, str | None, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, self.headers.get("content-type"), body))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    config = RelayConfig()
    config.observability.atof.output_directory = str(tmp_path)
    config.observability.opentelemetry.enabled = True
    config.observability.opentelemetry.endpoints = [
        RelayOpenTelemetryEndpointConfig(
            type=projection,
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces?projection={projection}",
            timeout_millis=1000,
        )
        for projection in ("openinference", "full", "gen_ai")
    ]
    try:
        await ensure_started(config)
        with nemo_relay.scope.scope("otel-test", nemo_relay.ScopeType.Agent):
            pass
    finally:
        try:
            await shutdown_async()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            assert not server_thread.is_alive()

    assert len(received) >= 3
    assert {path for path, _, _ in received} == {
        "/v1/traces?projection=openinference",
        "/v1/traces?projection=full",
        "/v1/traces?projection=gen_ai",
    }
    assert all(content_type == "application/x-protobuf" for _, content_type, _ in received)
    assert all(body for _, _, body in received)
