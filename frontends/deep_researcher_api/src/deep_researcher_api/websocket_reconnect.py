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

"""Reconnectable WebSocket handler for HITL interactions."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from deep_researcher_api.auth.errors import AuthError
from deep_researcher_api.auth.middleware import build_request_trace_tags
from deep_researcher_api.auth.middleware import detect_internal_caller
from deep_researcher_api.auth.middleware import resolve_request_user
from deep_researcher_api.auth.middleware import user_context
from deep_researcher_api.auth.request_trace import request_trace_tag_context
from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.api_server import ResponseObservabilityTrace
from nat.data_models.api_server import SystemResponseContent
from nat.data_models.api_server import TextContent
from nat.data_models.api_server import WebSocketMessageStatus
from nat.data_models.api_server import WebSocketMessageType
from nat.data_models.api_server import WebSocketObservabilityTraceMessage
from nat.data_models.api_server import WebSocketSystemInteractionMessage
from nat.data_models.api_server import WebSocketSystemIntermediateStepMessage
from nat.data_models.api_server import WebSocketSystemResponseTokenMessage
from nat.data_models.api_server import WebSocketUserInteractionResponseMessage
from nat.data_models.api_server import WebSocketUserMessage
from nat.data_models.interactive import HumanPromptNotification
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import HumanResponseNotification
from nat.data_models.interactive import InteractionPrompt
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import WebSocketAuthenticationFlowHandler
from nat.front_ends.fastapi.message_handler import WebSocketMessageHandler
from nat.front_ends.fastapi.response_helpers import generate_streaming_response

logger = logging.getLogger(__name__)

_auth_validators: list = []
_require_auth = False
_external_hostnames: set[str] | None = None
WS_POLICY_VIOLATION = 1008
SESSION_COOKIE_NAME = "nat-session"


def configure_websocket_auth(
    *,
    validators: list | None = None,
    require_auth: bool = False,
    external_hostnames: set[str] | None = None,
) -> None:
    """Configure WebSocket auth to mirror the HTTP middleware validator chain."""
    global _auth_validators, _require_auth, _external_hostnames
    _auth_validators = list(validators or [])
    _require_auth = require_auth
    _external_hostnames = external_hostnames


async def authenticate_websocket_connection(socket: WebSocket) -> tuple[dict[str, Any] | None, int | None]:
    """Resolve the caller identity for a WebSocket handshake."""
    headers = dict(socket.scope.get("headers", []))
    user, error_status, is_external, _ = await resolve_request_user(
        headers,
        validators=_auth_validators,
        require_auth=_require_auth,
        external_hostnames=_external_hostnames,
    )
    if user is not None:
        return user, None

    if not is_external:
        return detect_internal_caller(headers), None

    if error_status == 401:
        return None, WS_POLICY_VIOLATION

    return None, WS_POLICY_VIOLATION


class WebSocketSessionRegistry:
    """Keep track of active sockets, pending HITL responses, and running workflow tasks."""

    def __init__(self) -> None:
        self._sockets: dict[str, WebSocket] = {}
        self._pending_interactions: dict[str, asyncio.Future[TextContent]] = {}
        self._workflow_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def set_socket(self, conversation_id: str | None, socket: WebSocket) -> None:
        """Register the latest socket for a conversation."""
        if not conversation_id:
            return
        async with self._lock:
            self._sockets[conversation_id] = socket

    async def clear_socket(self, conversation_id: str | None, socket: WebSocket) -> None:
        """Clear the socket only if it matches the current one."""
        if not conversation_id:
            return
        async with self._lock:
            current = self._sockets.get(conversation_id)
            if current is socket:
                self._sockets.pop(conversation_id, None)

    async def send(self, conversation_id: str | None, message: BaseModel) -> bool:
        """Send a message to the current socket for a conversation."""
        if not conversation_id:
            return False
        async with self._lock:
            socket = self._sockets.get(conversation_id)
        if not socket:
            return False
        try:
            await socket.send_json(message.model_dump())
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to send websocket message after reconnect: %s", exc)
            return False

    async def register_pending_interaction(
        self,
        conversation_id: str | None,
        future: asyncio.Future[TextContent],
    ) -> None:
        """Store the pending HITL future for a conversation."""
        if not conversation_id:
            return
        async with self._lock:
            self._pending_interactions[conversation_id] = future

    async def resolve_pending_interaction(
        self,
        conversation_id: str | None,
        user_content: TextContent,
    ) -> bool:
        """Resolve a pending HITL future if it exists."""
        if not conversation_id:
            return False
        async with self._lock:
            future = self._pending_interactions.get(conversation_id)
            if future is None or future.done():
                return False
            future.set_result(user_content)
            self._pending_interactions.pop(conversation_id, None)
            return True

    async def clear_pending_interaction(self, conversation_id: str | None) -> None:
        """Clear pending interaction state once resolved."""
        if not conversation_id:
            return
        async with self._lock:
            self._pending_interactions.pop(conversation_id, None)

    async def set_workflow_task(self, conversation_id: str | None, task: asyncio.Task) -> None:
        """Register the running workflow task, cancelling any stale one first."""
        if not conversation_id:
            return
        async with self._lock:
            old_task = self._workflow_tasks.get(conversation_id)
            if old_task is not None and not old_task.done():
                old_task.cancel()
                logger.info("Cancelled stale workflow task for conversation %s", conversation_id)
            self._workflow_tasks[conversation_id] = task

    async def cancel_workflow_task(self, conversation_id: str | None) -> None:
        """Cancel and remove the workflow task for a conversation."""
        if not conversation_id:
            return
        async with self._lock:
            task = self._workflow_tasks.pop(conversation_id, None)
            if task is not None and not task.done():
                task.cancel()
                logger.info("Cancelled workflow task for conversation %s", conversation_id)


_registry = WebSocketSessionRegistry()
_installed = False


class ReconnectableWebSocketMessageHandler(WebSocketMessageHandler):
    """WebSocket handler that supports HITL reconnects per conversation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._user_interaction_response: asyncio.Future[TextContent] | None = None
        self._authenticated_user: dict[str, Any] | None = None

    def _is_handshake_token_expired(self) -> bool:
        """Return True if the JWT used at handshake has since passed its ``exp``.

        Backend stores verified JWT claims in ``_authenticated_user`` after the
        handshake (see ``JWTValidator.validate``). The ``exp`` claim is seconds
        since epoch. We re-check it on every inbound message so a long-lived
        socket cannot keep accepting work indefinitely under a dead token --
        WebSocket browsers do NOT replay HTTP cookies on subsequent frames, so
        the only way to refresh credentials is to close + reopen the socket.

        Internal/anonymous callers do not carry an ``exp`` claim; for them
        this returns ``False`` (no re-auth required).
        """
        user = self._authenticated_user
        if not user:
            return False
        exp = user.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return time.time() >= exp

    async def _send_auth_expired_error(self, conversation_id: str | None) -> None:
        """Notify the client that its handshake token has expired.

        The client (frontend) listens for ``message == "auth_expired"`` on the
        ERROR channel and reacts by refreshing the NextAuth session, closing
        the socket, and reconnecting. The fresh handshake carries an updated
        idToken cookie. After the new socket is open, the client drains any
        buffered outgoing message it had queued during the rotation.
        """
        error = Error(
            code=ErrorTypes.USER_AUTH_ERROR,
            message="auth_expired",
            details="Handshake token has expired; reconnect to refresh credentials.",
        )
        try:
            error_message = await self._message_validator.create_system_response_token_message(
                message_type=WebSocketMessageType.ERROR_MESSAGE,
                conversation_id=conversation_id,
                content=error,
            )
        except Exception:  # pragma: no cover - validator never fails on this contract
            logger.exception("Failed to build auth_expired error message")
            return

        try:
            await self._socket.send_json(error_message.model_dump())
        except Exception as exc:  # pragma: no cover - socket may already be closed
            logger.warning("Failed to send auth_expired: %s", exc)

    async def run(self) -> None:
        """Process websocket messages and allow reconnect HITL responses."""
        if self._authenticated_user is None:
            self._authenticated_user, close_code = await authenticate_websocket_connection(self._socket)
            if close_code is not None:
                await self._socket.close(code=close_code)
                return

        while True:
            try:
                message: dict[str, Any] = await self._socket.receive_json()
                validated_message: BaseModel = await self._message_validator.validate_message(message)

                if isinstance(validated_message, WebSocketUserMessage):
                    # Per-message re-auth: the handshake-time JWT may have
                    # expired since the socket was opened. Reject the work
                    # and ask the client to reconnect with a fresh token.
                    if self._is_handshake_token_expired():
                        logger.info(
                            "Rejecting user_message: handshake token expired (conversation %s)",
                            validated_message.conversation_id,
                        )
                        await self._send_auth_expired_error(validated_message.conversation_id)
                        continue

                    await self.process_workflow_request(validated_message)
                    await _registry.set_socket(validated_message.conversation_id, self._socket)

                elif isinstance(
                    validated_message,
                    WebSocketSystemResponseTokenMessage
                    | WebSocketSystemIntermediateStepMessage
                    | WebSocketSystemInteractionMessage,
                ):
                    pass

                elif isinstance(validated_message, WebSocketUserInteractionResponseMessage):
                    # Same re-auth gate for HITL interaction responses --
                    # otherwise an idle clarification prompt could be
                    # answered hours later under an expired token.
                    if self._is_handshake_token_expired():
                        logger.info(
                            "Rejecting user_interaction: handshake token expired (conversation %s)",
                            validated_message.conversation_id,
                        )
                        await self._send_auth_expired_error(validated_message.conversation_id)
                        continue

                    user_content = await self._process_websocket_user_interaction_response_message(validated_message)
                    await _registry.set_socket(validated_message.conversation_id, self._socket)
                    if self._user_interaction_response is not None:
                        self._user_interaction_response.set_result(user_content)
                    else:
                        resolved = await _registry.resolve_pending_interaction(
                            validated_message.conversation_id, user_content
                        )
                        if not resolved:
                            logger.warning(
                                "No pending HITL interaction to resume for conversation %s",
                                validated_message.conversation_id,
                            )
            except (asyncio.CancelledError, WebSocketDisconnect):
                await _registry.clear_socket(self._conversation_id, self._socket)
                await _registry.cancel_workflow_task(self._conversation_id)
                self._cancel_running_workflow()
                break
            except ValidationError as exc:
                logger.warning("Invalid websocket message payload: %s", str(exc))

    def _cancel_running_workflow(self) -> None:
        """Cancel the background workflow task spawned by NAT's create_task."""
        task = self._running_workflow_task
        if task is not None and not task.done():
            task.cancel()
            logger.info(
                "Cancelled in-flight workflow task for conversation %s",
                self._conversation_id,
            )

    async def process_workflow_request(self, user_message_as_validated_type: WebSocketUserMessage) -> None:
        """Process user messages and register sockets for reconnect."""
        await _registry.set_socket(user_message_as_validated_type.conversation_id, self._socket)
        headers = dict(self._socket.scope.get("headers", []))
        current_user = self._authenticated_user or detect_internal_caller(headers)
        request_trace_tags = build_request_trace_tags(
            headers,
            self._socket.scope,
            current_user,
            external_hostnames=_external_hostnames,
        )
        with user_context(current_user), request_trace_tag_context(request_trace_tags):
            await super().process_workflow_request(user_message_as_validated_type)
        # TODO(NAT-upstream): _running_workflow_task is currently always None
        # because NAT's message_handler.py assigns via method chaining:
        #   self._running_workflow_task = asyncio.create_task(...).add_done_callback(cb)
        # add_done_callback() returns None. Blocked on NeMo-Agent-Toolkit#1744.
        task = self._running_workflow_task
        if task is not None and not task.done():
            await _registry.set_workflow_task(user_message_as_validated_type.conversation_id, task)

    async def create_websocket_message(
        self,
        data_model: BaseModel,
        message_type: str | None = None,
        status: WebSocketMessageStatus = WebSocketMessageStatus.IN_PROGRESS,
    ) -> None:
        """Create a websocket message and send via the registry."""
        message: BaseModel | None = None
        try:
            if message_type is None:
                message_type = await self._message_validator.resolve_message_type_by_data(data_model)

            message_schema: type[BaseModel] = await self._message_validator.get_message_schema_by_type(message_type)

            if hasattr(data_model, "id"):
                message_id: str = str(getattr(data_model, "id"))
            else:
                message_id = str(uuid.uuid4())

            content: BaseModel = await self._message_validator.convert_data_to_message_content(data_model)

            if issubclass(message_schema, WebSocketSystemResponseTokenMessage):
                message = await self._message_validator.create_system_response_token_message(
                    message_id=message_id,
                    parent_id=self._message_parent_id,
                    conversation_id=self._conversation_id,
                    content=content,
                    status=status,
                )

            elif issubclass(message_schema, WebSocketSystemIntermediateStepMessage):
                message = await self._message_validator.create_system_intermediate_step_message(
                    message_id=message_id,
                    parent_id=await self._message_validator.get_intermediate_step_parent_id(data_model),
                    conversation_id=self._conversation_id,
                    content=content,
                    status=status,
                )

            elif issubclass(message_schema, WebSocketSystemInteractionMessage):
                message = await self._message_validator.create_system_interaction_message(
                    message_id=message_id,
                    parent_id=self._message_parent_id,
                    conversation_id=self._conversation_id,
                    content=content,
                    status=status,
                )

            elif issubclass(message_schema, WebSocketObservabilityTraceMessage):
                message = await self._message_validator.create_observability_trace_message(
                    message_id=message_id,
                    parent_id=self._message_parent_id,
                    conversation_id=self._conversation_id,
                    content=content,
                )

            elif isinstance(content, Error):
                raise ValueError(f"Invalid input data creating websocket message. {data_model.model_dump_json()}")

            elif issubclass(message_schema, Error):
                raise TypeError(f"Invalid message type: {message_type}")

            elif message is None:
                raise ValueError(
                    f"Message type could not be resolved by input data model: {data_model.model_dump_json()}"
                )

        except (ValidationError, ValueError, TypeError) as exc:
            logger.exception("A data validation error occurred creating websocket message: %s", str(exc))
            message = await self._message_validator.create_system_response_token_message(
                message_type=WebSocketMessageType.ERROR_MESSAGE,
                conversation_id=self._conversation_id,
                content=Error(code=ErrorTypes.UNKNOWN_ERROR, message="default", details=str(exc)),
            )

        finally:
            if message is not None:
                sent = await _registry.send(self._conversation_id, message)
                if not sent:
                    if not self._conversation_id:
                        try:
                            await self._socket.send_json(message.model_dump())
                        except Exception as exc:  # pragma: no cover - socket may be closed
                            logger.warning("Failed to send websocket message: %s", exc)
                    else:
                        logger.debug(
                            "Dropping message for disconnected conversation %s",
                            self._conversation_id,
                        )

    async def human_interaction_callback(self, prompt: InteractionPrompt) -> HumanResponse:
        """
        Handle HITL prompts and register response futures for reconnect.
        """
        human_response_future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()
        self._user_interaction_response = human_response_future
        await _registry.register_pending_interaction(self._conversation_id, human_response_future)

        try:
            await self.create_websocket_message(
                data_model=prompt.content,
                message_type=WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE,
                status=WebSocketMessageStatus.IN_PROGRESS,
            )

            if isinstance(prompt.content, HumanPromptNotification):
                return HumanResponseNotification()

            text_content: TextContent = await human_response_future
            interaction_response: HumanResponse = await self._message_validator.convert_text_content_to_human_response(
                text_content, prompt.content
            )
            return interaction_response
        finally:
            await _registry.clear_pending_interaction(self._conversation_id)
            self._user_interaction_response = None

    async def _run_workflow(
        self,
        payload: Any,
        user_message_id: str | None = None,
        conversation_id: str | None = None,
        streaming: bool = True,
        result_type: type | None = None,
        output_type: type | None = None,
    ) -> None:
        """Run the workflow without breaking reconnect message delivery."""
        socket_scope = getattr(getattr(self, "_socket", None), "scope", {})
        current_user = self._authenticated_user or detect_internal_caller(dict(socket_scope.get("headers", [])))
        with user_context(current_user):
            try:
                auth_callback = self._flow_handler.authenticate if self._flow_handler else None
                async with self._session_manager.session(
                    user_message_id=user_message_id,
                    conversation_id=conversation_id,
                    http_connection=self._socket,
                    user_input_callback=self.human_interaction_callback,
                    user_authentication_callback=auth_callback,
                ) as session:
                    async for value in generate_streaming_response(
                        payload,
                        session=session,
                        streaming=streaming,
                        step_adaptor=self._step_adaptor,
                        result_type=result_type,
                        output_type=output_type,
                    ):
                        if isinstance(value, ResponseObservabilityTrace):
                            if self._pending_observability_trace is None:
                                self._pending_observability_trace = value
                        else:
                            await self.create_websocket_message(
                                data_model=value,
                                status=WebSocketMessageStatus.IN_PROGRESS,
                            )

                await self.create_websocket_message(
                    data_model=SystemResponseContent(),
                    message_type=WebSocketMessageType.RESPONSE_MESSAGE,
                    status=WebSocketMessageStatus.COMPLETE,
                )

                if self._pending_observability_trace:
                    await self.create_websocket_message(
                        data_model=self._pending_observability_trace,
                        message_type=WebSocketMessageType.OBSERVABILITY_TRACE_MESSAGE,
                    )
                    self._pending_observability_trace = None
            except Exception as exc:
                if not isinstance(exc, AuthError):
                    logger.exception("Error running workflow")
                    return

                logger.warning("Auth error during workflow: %s", exc)
                try:
                    await self.create_websocket_message(
                        data_model=Error(
                            code=ErrorTypes.UNKNOWN_ERROR,
                            message=exc.error_code,
                            details=str(exc),
                        ),
                        message_type=WebSocketMessageType.ERROR_MESSAGE,
                        status=WebSocketMessageStatus.COMPLETE,
                    )
                except Exception:  # pragma: no cover - socket may already be closed
                    pass


def install_reconnectable_handler() -> None:  # TODO: upstream to NAT
    """Monkeypatch NAT to use reconnectable websocket handler."""
    global _installed
    if _installed:
        return
    from nat.front_ends.fastapi import fastapi_front_end_plugin_worker as worker_module
    from nat.front_ends.fastapi.routes import websocket as websocket_routes

    worker_module.WebSocketMessageHandler = ReconnectableWebSocketMessageHandler
    websocket_routes.WebSocketMessageHandler = ReconnectableWebSocketMessageHandler

    def patched_websocket_endpoint(*, worker: Any, session_manager: Any):
        """Build websocket endpoint handler with reconnect support and verified auth."""

        async def _websocket_endpoint(websocket: WebSocket):
            session_id = websocket.query_params.get("session")
            if session_id and not websocket_routes._SAFE_SESSION_ID_RE.match(session_id):
                logger.warning("WebSocket: Rejected session ID with unsafe characters")
                await websocket.close(code=WS_POLICY_VIOLATION, reason="Invalid session ID")
                return

            if session_id:
                headers = list(websocket.scope.get("headers", []))
                cookie_header = f"{SESSION_COOKIE_NAME}={session_id}"

                cookie_exists = False
                existing_session_cookie = False

                for i, (name, value) in enumerate(headers):
                    if name != b"cookie":
                        continue

                    cookie_exists = True
                    cookie_str = value.decode()

                    if f"{SESSION_COOKIE_NAME}=" in cookie_str:
                        existing_session_cookie = True
                        logger.info("WebSocket: Session cookie already present in headers (same-origin)")
                    else:
                        headers[i] = (name, f"{cookie_str}; {cookie_header}".encode())
                        logger.info(
                            "WebSocket: Added session cookie to existing cookie header: %s",
                            session_id[:10] + "...",
                        )
                    break

                if not cookie_exists and not existing_session_cookie:
                    headers.append((b"cookie", cookie_header.encode()))
                    logger.info("WebSocket: Added new session cookie header: %s", session_id[:10] + "...")

                websocket.scope["headers"] = headers

            user, close_code = await authenticate_websocket_connection(websocket)
            if close_code is not None:
                await websocket.close(code=close_code)
                return

            async with ReconnectableWebSocketMessageHandler(
                websocket,
                session_manager,
                worker.get_step_adaptor(),
                worker,
            ) as handler:
                handler._authenticated_user = user
                flow_handler = WebSocketAuthenticationFlowHandler(worker._add_flow, worker._remove_flow, handler)
                handler.set_flow_handler(flow_handler)
                with user_context(user or detect_internal_caller(dict(websocket.scope.get("headers", [])))):
                    await handler.run()

        return _websocket_endpoint

    websocket_routes.websocket_endpoint = patched_websocket_endpoint
    _installed = True
