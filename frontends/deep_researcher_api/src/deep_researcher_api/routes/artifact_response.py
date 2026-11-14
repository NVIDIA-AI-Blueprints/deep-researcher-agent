# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact streaming with errors resolved before HTTP headers are sent."""

from collections.abc import Iterator
from itertools import chain
from typing import Any

import anyio
from starlette.concurrency import iterate_in_threadpool
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send


class ArtifactStreamingResponse(StreamingResponse):
    """Own the artifact iterator across prefetch, streaming, and cancellation."""

    def __init__(self, content: Iterator[bytes], **kwargs: Any) -> None:
        """Keep the provider iterator available for deterministic cleanup."""
        self._content = content
        super().__init__(content, **kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Prefetch before headers and close the iterator on every exit path."""
        try:
            try:
                # AnyIO waits for the worker before honoring cancellation, so an
                # opened provider body cannot be abandoned in a detached thread.
                first = await run_in_threadpool(next, self._content, b"")
            except FileNotFoundError:
                await JSONResponse({"detail": "Artifact content not found"}, status_code=404)(scope, receive, send)
                return
            self.body_iterator = iterate_in_threadpool(chain((first,), self._content))
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self._content, "close", None)
            if close is not None:
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(close)
