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

"""Request-scoped hook for reusing a caller's research depth decision.

A caller that has already classified a query (for example the MCP ``JobManager``,
which classifies once in ``submit()`` to persist ``depth`` and choose a polling
cadence) can set the resulting depth here for the duration of a single workflow
run. The ``intent_classifier`` node reuses it instead of re-invoking the intent
LLM, so the persisted depth and the executed route are the same decision.

The value rides a plain :class:`contextvars.ContextVar`, which propagates
uninterrupted through NAT's ``SessionManager.session()``/``.run()`` and the
compiled LangGraph (the same mechanism NAT itself relies on for
``conversation_id`` and its active-function stack). It defaults to ``None`` and
is never set by API/CLI callers, so the short-circuit is strictly opt-in.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

Depth = Literal["shallow", "deep"]

_preclassified_depth: ContextVar[Depth | None] = ContextVar("deep_researcher_preclassified_depth", default=None)


@contextmanager
def preclassified_depth(depth: str | None):
    """Bind a caller-supplied depth for the enclosed workflow run.

    Values other than ``"shallow"`` or ``"deep"`` (including ``None``) bind to
    ``None``, which makes the block a no-op and leaves the classifier to decide.
    """
    value: Depth | None = depth if depth in ("shallow", "deep") else None
    token = _preclassified_depth.set(value)
    try:
        yield
    finally:
        _preclassified_depth.reset(token)


def get_preclassified_depth() -> Depth | None:
    """Return the caller-supplied depth for the current run, or ``None``."""
    return _preclassified_depth.get()
