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

"""SSE connection manager for graceful shutdown support."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class SSEConnectionManager:
    """
    Manages active SSE connections for graceful shutdown.

    Tracks active SSE streams and provides a mechanism to signal
    shutdown to all active connections, allowing them to terminate
    gracefully instead of hanging during server shutdown.
    """

    def __init__(self):
        self._active_connections: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown_event.is_set()

    @property
    def active_count(self) -> int:
        """Return the number of active SSE connections."""
        return len(self._active_connections)

    def signal_shutdown(self) -> None:
        """
        Signal shutdown to all SSE connections (sync, for use from signal handlers).

        This is a synchronous method that just sets the shutdown event flag.
        SSE generators will notice this flag and exit gracefully.
        """
        self._shutdown_event.set()

    async def wait_or_shutdown(self, timeout: float) -> bool:
        """
        Wait for the specified timeout or until shutdown is signaled.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if shutdown was signaled, False if timeout elapsed normally.
        """
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            return True  # Shutdown was signaled
        except TimeoutError:
            return False  # Normal timeout, no shutdown

    async def register(self, task: asyncio.Task) -> None:
        """Register an SSE connection task."""
        async with self._lock:
            self._active_connections.add(task)
            logger.debug("SSE connection registered, active count: %d", len(self._active_connections))

    async def unregister(self, task: asyncio.Task) -> None:
        """Unregister an SSE connection task."""
        async with self._lock:
            self._active_connections.discard(task)
            logger.debug("SSE connection unregistered, active count: %d", len(self._active_connections))

    @asynccontextmanager
    async def track_connection(self) -> AsyncGenerator[None, None]:
        """Context manager to track an SSE connection's lifecycle."""
        task = asyncio.current_task()
        if task:
            await self.register(task)
        try:
            yield
        finally:
            if task:
                await self.unregister(task)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """
        Signal shutdown to all active SSE connections.

        Args:
            timeout: Maximum seconds to wait for connections to close gracefully.
        """
        logger.info("SSE connection manager shutting down, %d active connections", len(self._active_connections))

        # Signal shutdown to all generators
        self._shutdown_event.set()

        if not self._active_connections:
            return

        # Give connections a moment to notice the shutdown signal
        await asyncio.sleep(0.1)

        # Cancel any remaining connections
        async with self._lock:
            tasks_to_cancel = list(self._active_connections)

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                logger.debug("Cancelled SSE connection task")

        # Wait for all tasks to complete
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=timeout,
                )
                logger.info("All SSE connections closed gracefully")
            except TimeoutError:
                logger.warning(
                    "Timeout waiting for SSE connections to close, %d remaining", len(self._active_connections)
                )

    def reset(self) -> None:
        """Reset the manager state (for testing)."""
        self._active_connections.clear()
        self._shutdown_event.clear()


# Global instance for the application
_connection_manager: SSEConnectionManager | None = None


def get_connection_manager() -> SSEConnectionManager:
    """Get the global SSE connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = SSEConnectionManager()
    return _connection_manager


def reset_connection_manager() -> None:
    """Reset the global connection manager (for testing)."""
    global _connection_manager
    if _connection_manager is not None:
        _connection_manager.reset()
    _connection_manager = None
