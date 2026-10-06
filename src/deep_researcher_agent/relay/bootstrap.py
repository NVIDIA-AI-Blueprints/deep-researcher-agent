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

"""Deep Researcher Agent-owned lifecycle for NeMo Relay's plugin host."""

from __future__ import annotations

import asyncio
import atexit
import json

import nemo_relay
from nemo_relay import plugin

from .config import RelayConfig
from .logging import register_logging_subscriber
from .privacy import deregister_privacy_sanitizers
from .privacy import register_privacy_sanitizers

_lock = asyncio.Lock()
_active_config: str | None = None


async def ensure_started(config: RelayConfig | None = None) -> None:
    """Initialize Relay's supported plugin host once for the effective Deep Researcher Agent config."""
    global _active_config
    relay_config = config or RelayConfig()
    plugin_config = relay_config.to_plugin_config()
    serialized = json.dumps(plugin_config, sort_keys=True)
    async with _lock:
        if _active_config == serialized:
            return
        plugin.validate(plugin_config)
        await plugin.initialize(plugin_config)
        register_privacy_sanitizers(relay_config.redaction)
        if relay_config.logging:
            register_logging_subscriber()
        else:
            nemo_relay.subscribers.deregister("deep-researcher-relay-logging")
        _active_config = serialized


def _reset_state() -> None:
    global _active_config
    nemo_relay.subscribers.deregister("deep-researcher-relay-logging")
    deregister_privacy_sanitizers()
    _active_config = None


async def shutdown_async() -> None:
    """Flush Relay exporters from an asynchronous application lifecycle."""
    await plugin.clear_async()
    _reset_state()


def shutdown() -> None:
    """Flush Relay exporters after the application event loop has stopped."""
    plugin.clear()
    _reset_state()


atexit.register(shutdown)
