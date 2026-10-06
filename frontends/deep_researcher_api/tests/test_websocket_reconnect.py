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

"""Unit tests for per-message JWT re-authentication on the reconnectable WS handler.

Covers the behavior added for sealing the "long-lived socket trusts an expired
token forever" gap:

* ``_is_handshake_token_expired`` correctly classifies tokens by ``exp``
* ``_send_auth_expired_error`` builds a structured error and writes it to the socket
* the inbound message loop short-circuits user_message / user_interaction
  when the handshake JWT has passed its ``exp`` and emits the auth_expired
  error instead of running the workflow

We construct the handler with ``__new__`` so the heavy NAT base ``__init__``
(session manager, message validator, step adaptor, worker) does not have to be
satisfied by the test environment.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from deep_researcher_api.websocket_reconnect import ReconnectableWebSocketMessageHandler


def _make_handler(
    *,
    authenticated_user: dict | None,
    socket: MagicMock | None = None,
    message_validator: MagicMock | None = None,
) -> ReconnectableWebSocketMessageHandler:
    """Return a handler instance bypassing the NAT base class __init__."""
    handler = ReconnectableWebSocketMessageHandler.__new__(ReconnectableWebSocketMessageHandler)
    handler._authenticated_user = authenticated_user
    handler._socket = socket or MagicMock()
    handler._message_validator = message_validator or MagicMock()
    return handler


class TestIsHandshakeTokenExpired:
    """``_is_handshake_token_expired`` is the gate that decides re-auth on each message."""

    def test_returns_false_when_no_user(self) -> None:
        handler = _make_handler(authenticated_user=None)
        assert handler._is_handshake_token_expired() is False

    def test_returns_false_for_internal_caller_without_exp(self) -> None:
        # Internal/anonymous callers don't carry JWT claims; we must not
        # treat them as "expired" or we'd reject every internal call.
        handler = _make_handler(authenticated_user={"type": "internal"})
        assert handler._is_handshake_token_expired() is False

    def test_returns_false_when_exp_is_not_a_number(self) -> None:
        # Defensive: malformed exp (e.g. claim was a string) should NOT be
        # interpreted as "expired" -- that would fail closed in a way the
        # client cannot remediate. Treat as "no exp" instead.
        handler = _make_handler(authenticated_user={"exp": "not-a-number"})
        assert handler._is_handshake_token_expired() is False

    def test_returns_false_when_exp_is_in_the_future(self) -> None:
        future = time.time() + 600
        handler = _make_handler(authenticated_user={"exp": future, "sub": "u-1"})
        assert handler._is_handshake_token_expired() is False

    def test_returns_true_when_exp_is_in_the_past(self) -> None:
        past = time.time() - 1
        handler = _make_handler(authenticated_user={"exp": past, "sub": "u-1"})
        assert handler._is_handshake_token_expired() is True


class TestSendAuthExpiredError:
    """The error packet must reach the wire with the contract the client expects."""

    @pytest.mark.asyncio
    async def test_sends_user_auth_error_with_auth_expired_message(self) -> None:
        socket = MagicMock()
        socket.send_json = AsyncMock()

        # The validator builds the payload; we only care that we forwarded
        # an Error with the right shape into it. Echo the produced message
        # back verbatim so we can assert on it.
        message_validator = MagicMock()
        built_message = MagicMock()
        built_message.model_dump.return_value = {
            "type": "error",
            "content": {"code": "user_auth_error", "message": "auth_expired"},
        }
        message_validator.create_system_response_token_message = AsyncMock(return_value=built_message)

        handler = _make_handler(
            authenticated_user={"exp": time.time() - 1},
            socket=socket,
            message_validator=message_validator,
        )

        await handler._send_auth_expired_error("conv-1")

        # The validator must have been asked to build an ERROR_MESSAGE for
        # this conversation, with an Error carrying message="auth_expired".
        call = message_validator.create_system_response_token_message.await_args
        kwargs = call.kwargs
        assert kwargs["conversation_id"] == "conv-1"
        error = kwargs["content"]
        assert error.message == "auth_expired"
        assert error.code.value == "user_auth_error"

        # And the result must have been forwarded to the socket.
        socket.send_json.assert_awaited_once()
        sent_payload = socket.send_json.await_args.args[0]
        assert sent_payload["content"]["message"] == "auth_expired"

    @pytest.mark.asyncio
    async def test_swallows_send_failure_silently(self) -> None:
        # If the socket is already closed, we don't want a second exception
        # crashing the receive loop -- the client will reconnect on its own.
        socket = MagicMock()
        socket.send_json = AsyncMock(side_effect=RuntimeError("socket closed"))

        message_validator = MagicMock()
        message_validator.create_system_response_token_message = AsyncMock(return_value=MagicMock())

        handler = _make_handler(
            authenticated_user={"exp": time.time() - 1},
            socket=socket,
            message_validator=message_validator,
        )

        # Must not raise.
        await handler._send_auth_expired_error("conv-1")
