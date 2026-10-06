# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa: E501
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

"""Unit tests for reconnectable websocket handling."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from deep_researcher_api import websocket_reconnect
from deep_researcher_api.auth.middleware import get_current_user
from deep_researcher_api.auth.request_trace import get_request_trace_tags
from deep_researcher_api.websocket_reconnect import ReconnectableWebSocketMessageHandler
from deep_researcher_api.websocket_reconnect import WebSocketSessionRegistry
from deep_researcher_api.websocket_reconnect import authenticate_websocket_connection
from deep_researcher_api.websocket_reconnect import configure_websocket_auth
from nat.data_models.api_server import TextContent
from nat.data_models.api_server import UserMessageContent
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.api_server import UserMessages
from nat.data_models.api_server import WebSocketMessageType
from nat.data_models.api_server import WebSocketUserInteractionResponseMessage
from nat.data_models.api_server import WebSocketUserMessage
from nat.data_models.api_server import WorkflowSchemaType


class DummySocket:
    """Minimal websocket stand-in for handler testing."""

    def __init__(
        self,
        messages: list[dict] | None = None,
        raise_on_send: bool = False,
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        self._messages = messages or []
        self._index = 0
        self.raise_on_send = raise_on_send
        self.sent: list[dict] = []
        self.closed_with: int | None = None
        self.scope = {"headers": headers or [], "type": "websocket"}

    async def receive_json(self) -> dict:
        if self._index >= len(self._messages):
            raise WebSocketDisconnect(1000)
        value = self._messages[self._index]
        self._index += 1
        return value

    async def send_json(self, payload: dict) -> None:
        if self.raise_on_send:
            raise RuntimeError("send failed")
        self.sent.append(payload)

    async def close(self, code: int) -> None:
        self.closed_with = code


class DummySessionManager:
    """Minimal session manager stub for handler initialization."""

    def get_workflow_single_output_schema(self):
        return None

    def get_workflow_streaming_output_schema(self):
        return None

    @asynccontextmanager
    async def session(self, **_kwargs):
        yield object()


class DummyStepAdaptor:
    """Minimal step adaptor stub."""


class DummyWorker:
    """Minimal FastApiFrontEndPluginWorker stand-in (NAT handler requires worker)."""

    def set_conversation_handler(self, _conversation_id: str, _handler: object) -> None:
        return None

    def get_conversation_handler(self, _conversation_id: str) -> object | None:
        return None

    def remove_conversation_handler(self, _conversation_id: str) -> None:
        return None


class DummyMessage(BaseModel):
    """Simple pydantic message for tests."""

    content: str = "ok"


class DummyValidator:
    def __init__(self, user: dict | None = None) -> None:
        self.user = user
        self.tokens: list[str] = []

    def can_handle(self, token: str) -> bool:
        self.tokens.append(token)
        return True

    async def validate(self, token: str) -> dict | None:
        self.tokens.append(token)
        return self.user


@pytest.fixture(name="event_loop")
def fixture_event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(name="dummy_socket")
def fixture_dummy_socket() -> DummySocket:
    return DummySocket()


@pytest.mark.asyncio
async def test_registry_set_send_clear_socket(
    dummy_socket: DummySocket,
) -> None:
    registry = WebSocketSessionRegistry()
    await registry.set_socket("conv-1", dummy_socket)

    message = DummyMessage()
    assert await registry.send("conv-1", message) is True
    assert dummy_socket.sent == [message.model_dump()]

    await registry.clear_socket("conv-1", dummy_socket)
    assert await registry.send("conv-1", message) is False


@pytest.mark.asyncio
async def test_registry_clear_socket_mismatch(
    dummy_socket: DummySocket,
) -> None:
    registry = WebSocketSessionRegistry()
    other_socket = DummySocket()
    await registry.set_socket("conv-1", dummy_socket)
    await registry.clear_socket("conv-1", other_socket)

    assert await registry.send("conv-1", DummyMessage()) is True


@pytest.mark.asyncio
async def test_registry_send_handles_missing_and_error() -> None:
    registry = WebSocketSessionRegistry()
    assert await registry.send(None, DummyMessage()) is False
    assert await registry.send("missing", DummyMessage()) is False

    failing_socket = DummySocket(raise_on_send=True)
    await registry.set_socket("conv-1", failing_socket)
    assert await registry.send("conv-1", DummyMessage()) is False


@pytest.mark.asyncio
async def test_registry_pending_interaction_resolve() -> None:
    registry = WebSocketSessionRegistry()
    future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()
    await registry.register_pending_interaction("conv-1", future)

    response = TextContent(text="hello")
    assert await registry.resolve_pending_interaction("conv-1", response) is True
    assert future.result() == response
    assert await registry.resolve_pending_interaction("conv-1", response) is False

    await registry.clear_pending_interaction("conv-1")
    assert await registry.resolve_pending_interaction("conv-1", response) is False


@pytest.mark.asyncio
async def test_handler_create_websocket_message_uses_registry_send(
    monkeypatch,
    dummy_socket: DummySocket,
) -> None:
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = "conv-1"

    async def fake_send(_conversation_id, _message):
        return True

    async def fake_resolve_message_type(_data_model):
        return "response"

    async def fake_get_schema(_message_type):
        from nat.data_models.api_server import WebSocketSystemResponseTokenMessage

        return WebSocketSystemResponseTokenMessage

    async def fake_convert_data(_data_model):
        return DummyMessage()

    async def fake_create_response_message(**_kwargs):
        return DummyMessage(content="sent")

    monkeypatch.setattr(
        websocket_reconnect,
        "_registry",
        WebSocketSessionRegistry(),
    )
    monkeypatch.setattr(websocket_reconnect._registry, "send", fake_send)
    monkeypatch.setattr(
        handler._message_validator,
        "resolve_message_type_by_data",
        fake_resolve_message_type,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "get_message_schema_by_type",
        fake_get_schema,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "convert_data_to_message_content",
        fake_convert_data,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "create_system_response_token_message",
        fake_create_response_message,
    )

    await handler.create_websocket_message(data_model=DummyMessage())
    assert dummy_socket.sent == []


@pytest.mark.asyncio
async def test_handler_create_websocket_message_drops_for_disconnected_conversation(
    monkeypatch,
) -> None:
    """When registry.send fails for a valid conversation_id the message is dropped
    instead of falling back to the (likely dead) direct socket."""
    failing_registry = WebSocketSessionRegistry()
    dummy_socket = DummySocket()
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = "conv-1"

    async def fake_send(_conversation_id, _message):
        return False

    async def fake_resolve_message_type(_data_model):
        return "response"

    async def fake_get_schema(_message_type):
        from nat.data_models.api_server import WebSocketSystemResponseTokenMessage

        return WebSocketSystemResponseTokenMessage

    async def fake_convert_data(_data_model):
        return DummyMessage()

    async def fake_create_response_message(**_kwargs):
        return DummyMessage(content="sent")

    monkeypatch.setattr(websocket_reconnect, "_registry", failing_registry)
    monkeypatch.setattr(websocket_reconnect._registry, "send", fake_send)
    monkeypatch.setattr(
        handler._message_validator,
        "resolve_message_type_by_data",
        fake_resolve_message_type,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "get_message_schema_by_type",
        fake_get_schema,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "convert_data_to_message_content",
        fake_convert_data,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "create_system_response_token_message",
        fake_create_response_message,
    )

    await handler.create_websocket_message(data_model=DummyMessage())
    assert dummy_socket.sent == []


@pytest.mark.asyncio
async def test_handler_create_websocket_message_falls_back_to_socket_without_conversation(
    monkeypatch,
) -> None:
    """When conversation_id is None, fall back to the direct socket."""
    failing_registry = WebSocketSessionRegistry()
    dummy_socket = DummySocket()
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = None

    async def fake_send(_conversation_id, _message):
        return False

    async def fake_resolve_message_type(_data_model):
        return "response"

    async def fake_get_schema(_message_type):
        from nat.data_models.api_server import WebSocketSystemResponseTokenMessage

        return WebSocketSystemResponseTokenMessage

    async def fake_convert_data(_data_model):
        return DummyMessage()

    async def fake_create_response_message(**_kwargs):
        return DummyMessage(content="sent")

    monkeypatch.setattr(websocket_reconnect, "_registry", failing_registry)
    monkeypatch.setattr(websocket_reconnect._registry, "send", fake_send)
    monkeypatch.setattr(
        handler._message_validator,
        "resolve_message_type_by_data",
        fake_resolve_message_type,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "get_message_schema_by_type",
        fake_get_schema,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "convert_data_to_message_content",
        fake_convert_data,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "create_system_response_token_message",
        fake_create_response_message,
    )

    await handler.create_websocket_message(data_model=DummyMessage())
    assert dummy_socket.sent == [{"content": "sent"}]


@pytest.mark.asyncio
async def test_handler_create_websocket_message_handles_socket_failure(
    monkeypatch,
) -> None:
    failing_registry = WebSocketSessionRegistry()
    dummy_socket = DummySocket(raise_on_send=True)
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = "conv-1"

    async def fake_send(_conversation_id, _message):
        return False

    async def fake_resolve_message_type(_data_model):
        return "response"

    async def fake_get_schema(_message_type):
        from nat.data_models.api_server import WebSocketSystemResponseTokenMessage

        return WebSocketSystemResponseTokenMessage

    async def fake_convert_data(_data_model):
        return DummyMessage()

    async def fake_create_response_message(**_kwargs):
        return DummyMessage(content="sent")

    monkeypatch.setattr(websocket_reconnect, "_registry", failing_registry)
    monkeypatch.setattr(websocket_reconnect._registry, "send", fake_send)
    monkeypatch.setattr(
        handler._message_validator,
        "resolve_message_type_by_data",
        fake_resolve_message_type,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "get_message_schema_by_type",
        fake_get_schema,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "convert_data_to_message_content",
        fake_convert_data,
    )
    monkeypatch.setattr(
        handler._message_validator,
        "create_system_response_token_message",
        fake_create_response_message,
    )

    await handler.create_websocket_message(data_model=DummyMessage())
    assert dummy_socket.sent == []


@pytest.mark.asyncio
async def test_handler_run_resolves_pending_future(monkeypatch) -> None:
    dummy_socket = DummySocket(messages=[{"ok": True}])
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = "conv-1"

    content = UserMessageContent(
        messages=[
            UserMessages(
                role=UserMessageContentRoleType.USER,
                content=[TextContent(text="response")],
            )
        ]
    )
    response_message = WebSocketUserInteractionResponseMessage(
        type=WebSocketMessageType.USER_INTERACTION_MESSAGE,
        id="msg-1",
        thread_id="thread-1",
        parent_id="parent-1",
        conversation_id="conv-1",
        content=content,
    )

    future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()
    handler._user_interaction_response = future

    async def fake_validate_message(_message):
        return response_message

    async def fake_set_socket(_conversation_id, _socket):
        return None

    monkeypatch.setattr(
        handler._message_validator,
        "validate_message",
        fake_validate_message,
    )
    monkeypatch.setattr(
        websocket_reconnect._registry,
        "set_socket",
        fake_set_socket,
    )

    await handler.run()
    assert future.done()
    assert future.result().text == "response"


@pytest.mark.asyncio
async def test_handler_run_uses_registry_when_no_future(
    monkeypatch,
) -> None:
    dummy_socket = DummySocket(messages=[{"ok": True}])
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = "conv-1"

    content = UserMessageContent(
        messages=[
            UserMessages(
                role=UserMessageContentRoleType.USER,
                content=[TextContent(text="response")],
            )
        ]
    )
    response_message = WebSocketUserInteractionResponseMessage(
        type=WebSocketMessageType.USER_INTERACTION_MESSAGE,
        id="msg-1",
        thread_id="thread-1",
        parent_id="parent-1",
        conversation_id="conv-1",
        content=content,
    )

    async def fake_validate_message(_message):
        return response_message

    async def fake_set_socket(_conversation_id, _socket):
        return None

    async def fake_resolve_pending(_conversation_id, _user_content):
        return True

    monkeypatch.setattr(
        handler._message_validator,
        "validate_message",
        fake_validate_message,
    )
    monkeypatch.setattr(
        websocket_reconnect._registry,
        "set_socket",
        fake_set_socket,
    )
    monkeypatch.setattr(
        websocket_reconnect._registry,
        "resolve_pending_interaction",
        fake_resolve_pending,
    )

    await handler.run()


@pytest.mark.asyncio
async def test_handler_run_processes_user_message(monkeypatch) -> None:
    dummy_socket = DummySocket(messages=[{"ok": True}])
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )

    content = UserMessageContent(
        messages=[
            UserMessages(
                role=UserMessageContentRoleType.USER,
                content=[TextContent(text="start")],
            )
        ]
    )
    user_message = WebSocketUserMessage(
        type=WebSocketMessageType.USER_MESSAGE,
        schema_type=WorkflowSchemaType.CHAT_STREAM,
        id="msg-1",
        conversation_id="conv-1",
        content=content,
    )
    processed: list[WebSocketUserMessage] = []
    sockets: list[str] = []

    async def fake_validate_message(_message):
        return user_message

    async def fake_process_workflow(_message):
        processed.append(_message)

    async def fake_set_socket(conversation_id, _socket):
        sockets.append(conversation_id)

    monkeypatch.setattr(handler._message_validator, "validate_message", fake_validate_message)
    monkeypatch.setattr(handler, "process_workflow_request", fake_process_workflow)
    monkeypatch.setattr(websocket_reconnect._registry, "set_socket", fake_set_socket)

    await handler.run()

    assert processed == [user_message]
    assert sockets == ["conv-1"]


@pytest.mark.asyncio
async def test_handler_run_cancels_workflow_on_disconnect(monkeypatch) -> None:
    """When the socket disconnects, in-flight workflow tasks are cancelled."""
    dummy_socket = DummySocket()  # no messages → immediate WebSocketDisconnect
    handler = ReconnectableWebSocketMessageHandler(
        socket=dummy_socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._conversation_id = "conv-1"

    # Simulate a long-running workflow task (sleeps forever)
    workflow_task = asyncio.create_task(asyncio.sleep(999))
    handler._running_workflow_task = workflow_task

    # Also register the task in the global registry
    await websocket_reconnect._registry.set_workflow_task("conv-1", workflow_task)

    await handler.run()

    # Let the event loop process the cancellation
    await asyncio.sleep(0)
    assert workflow_task.cancelled()


@pytest.mark.asyncio
async def test_registry_set_workflow_task_cancels_stale() -> None:
    """Registering a new workflow task cancels any previous stale one."""
    registry = WebSocketSessionRegistry()

    stale_task = asyncio.create_task(asyncio.sleep(999))
    await registry.set_workflow_task("conv-1", stale_task)
    await asyncio.sleep(0)
    assert not stale_task.cancelled()

    new_task = asyncio.create_task(asyncio.sleep(999))
    await registry.set_workflow_task("conv-1", new_task)
    await asyncio.sleep(0)  # let cancellation propagate to stale_task

    assert stale_task.cancelled()
    assert not new_task.cancelled()

    new_task.cancel()
    await asyncio.sleep(0)  # let cancellation propagate


@pytest.mark.asyncio
async def test_registry_cancel_workflow_task() -> None:
    """cancel_workflow_task removes and cancels the tracked task."""
    registry = WebSocketSessionRegistry()

    task = asyncio.create_task(asyncio.sleep(999))
    await registry.set_workflow_task("conv-1", task)
    await asyncio.sleep(0)
    assert not task.cancelled()

    await registry.cancel_workflow_task("conv-1")
    await asyncio.sleep(0)  # let cancellation propagate
    assert task.cancelled()

    # Idempotent: calling again is a no-op
    await registry.cancel_workflow_task("conv-1")


@pytest.mark.asyncio
async def test_registry_cancel_workflow_task_noop_for_missing() -> None:
    """cancel_workflow_task is safe for missing conversation IDs."""
    registry = WebSocketSessionRegistry()
    await registry.cancel_workflow_task(None)
    await registry.cancel_workflow_task("nonexistent")


def test_install_reconnectable_handler(monkeypatch) -> None:
    from nat.front_ends.fastapi import fastapi_front_end_plugin_worker as worker_module

    original_handler = worker_module.WebSocketMessageHandler
    monkeypatch.setattr(websocket_reconnect, "_installed", False)
    websocket_reconnect.install_reconnectable_handler()

    assert worker_module.WebSocketMessageHandler is ReconnectableWebSocketMessageHandler
    worker_module.WebSocketMessageHandler = original_handler
    monkeypatch.setattr(websocket_reconnect, "_installed", False)


@pytest.mark.asyncio
async def test_authenticate_websocket_connection_validates_external_token() -> None:
    validator = DummyValidator(user={"type": "oidc", "sub": "user-1", "skip_clarifier": False})
    configure_websocket_auth(validators=[validator], require_auth=True, external_hostnames={"localhost"})
    socket = DummySocket(headers=[(b"host", b"localhost"), (b"cookie", b"idToken=test-token")])

    user, close_code = await authenticate_websocket_connection(socket)

    assert close_code is None
    assert user == {"type": "oidc", "sub": "user-1", "skip_clarifier": False}
    assert validator.tokens == ["test-token", "test-token"]


@pytest.mark.asyncio
async def test_authenticate_websocket_connection_rejects_missing_external_token() -> None:
    configure_websocket_auth(validators=[], require_auth=True, external_hostnames={"localhost"})
    socket = DummySocket(headers=[(b"host", b"localhost")])

    user, close_code = await authenticate_websocket_connection(socket)

    assert user is None
    assert close_code == 1008


@pytest.mark.asyncio
async def test_authenticate_websocket_connection_validates_internal_token_when_present() -> None:
    validator = DummyValidator(user={"type": "oidc", "sub": "user-1", "skip_clarifier": False})
    configure_websocket_auth(validators=[validator], require_auth=True, external_hostnames={"localhost"})
    socket = DummySocket(headers=[(b"host", b"deep-researcher-agent"), (b"cookie", b"idToken=test-token")])

    user, close_code = await authenticate_websocket_connection(socket)

    assert close_code is None
    assert user == {"type": "oidc", "sub": "user-1", "skip_clarifier": False}


@pytest.mark.asyncio
async def test_run_closes_socket_when_websocket_auth_fails() -> None:
    configure_websocket_auth(validators=[], require_auth=True, external_hostnames={"localhost"})
    socket = DummySocket(headers=[(b"host", b"localhost")])
    handler = ReconnectableWebSocketMessageHandler(
        socket=socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )

    await handler.run()

    assert socket.closed_with == 1008


@pytest.mark.asyncio
async def test_run_workflow_binds_authenticated_user_context(monkeypatch) -> None:
    configure_websocket_auth(validators=[], require_auth=False, external_hostnames={"localhost"})
    socket = DummySocket(headers=[(b"host", b"localhost")])
    handler = ReconnectableWebSocketMessageHandler(
        socket=socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._authenticated_user = {"type": "oidc", "sub": "user-1", "skip_clarifier": False}

    seen_users: list[dict] = []

    async def fake_stream(*_args, **_kwargs):
        seen_users.append(get_current_user())
        if False:
            yield None

    monkeypatch.setattr(websocket_reconnect, "generate_streaming_response", fake_stream)

    await handler._run_workflow(payload="hello", conversation_id="conv-1")

    assert seen_users == [{"type": "oidc", "sub": "user-1", "skip_clarifier": False}]


@pytest.mark.asyncio
async def test_process_workflow_request_binds_request_trace_tags(monkeypatch) -> None:
    configure_websocket_auth(validators=[], require_auth=False, external_hostnames={"localhost"})
    socket = DummySocket(headers=[(b"host", b"localhost"), (b"cookie", b"idToken=test-token")])
    handler = ReconnectableWebSocketMessageHandler(
        socket=socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )
    handler._authenticated_user = {"type": "oidc", "sub": "user-1", "skip_clarifier": False}
    user_message = WebSocketUserMessage(
        type=WebSocketMessageType.USER_MESSAGE,
        schema_type=WorkflowSchemaType.CHAT_STREAM,
        id="msg-1",
        conversation_id="conv-1",
        content=UserMessageContent(
            messages=[
                UserMessages(
                    role=UserMessageContentRoleType.USER,
                    content=[TextContent(text="hello")],
                )
            ]
        ),
    )
    seen_tags: list[dict[str, str]] = []

    async def fake_process_workflow_request(_self, _message):
        seen_tags.append(get_request_trace_tags())

    monkeypatch.setattr(
        websocket_reconnect.WebSocketMessageHandler,
        "process_workflow_request",
        fake_process_workflow_request,
    )

    await handler.process_workflow_request(user_message)

    assert seen_tags == [
        {
            "deep-researcher.caller.type": "oidc",
            "deep-researcher.auth.transport": "cookie",
            "deep-researcher.auth.verified": "true",
            "deep-researcher.access.channel": "ui",
        }
    ]
