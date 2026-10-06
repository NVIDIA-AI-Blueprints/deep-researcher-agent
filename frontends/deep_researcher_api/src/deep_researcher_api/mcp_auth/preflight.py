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

"""Transport-agnostic preflight for per-user MCP auth.

This is the single source of truth for "may this job be enqueued given which
protected sources the caller selected?". It is reused by two enforcement points
so they cannot drift:

  * the REST ``/v1/jobs/async/submit`` route, which wraps a block in a 409
    JSON response (``_preflight_mcp_auth``); and
  * ``submit_agent_job`` itself, which raises :class:`McpAuthRequiredError` so
    that programmatic callers (e.g. the chat researcher's async deep-research
    submit) cannot bypass the route-level check.
"""

from __future__ import annotations

import logging

from deep_researcher_agent.auth import Principal
from deep_researcher_api.auth.errors import AuthError

from .models import McpAuthRequiredResponse
from .models import McpAuthRequiredSource
from .provider import ProtectedSourceAuthProvider
from .serialize import connect_url_for

logger = logging.getLogger(__name__)


async def evaluate_mcp_auth(
    provider: ProtectedSourceAuthProvider,
    principal: Principal,
    data_sources: list[str] | None,
) -> McpAuthRequiredResponse | None:
    """Return a block descriptor if a selected protected source needs connecting.

    ``data_sources is None`` means the job may use any tool, so every protected
    source must be connected; an explicit list restricts the check to those ids.
    Returns ``None`` when nothing is blocked (no protected sources selected, or
    all connected).
    """
    from deep_researcher_agent.common.data_source_registry import get_all_sources
    from deep_researcher_agent.common.data_source_registry import get_source

    if data_sources is None:
        protected_ids = [s.id for s in get_all_sources() if s.per_user_auth and s.per_user_auth.required]
    else:
        protected_ids = [
            sid for sid in data_sources if (src := get_source(sid)) and src.per_user_auth and src.per_user_auth.required
        ]
    if not protected_ids:
        return None

    challenges = await provider.require_connected(principal, protected_ids)
    if not challenges:
        return None

    blocked: list[McpAuthRequiredSource] = []
    for challenge in challenges:
        state = await provider.get_status(principal, challenge.source_id)
        blocked.append(
            McpAuthRequiredSource(
                source_id=challenge.source_id,
                status=state.status if state.status != "connected" else "not_connected",
                connect_url=connect_url_for(challenge.source_id),
                auth_url=challenge.auth_url or None,
            )
        )
    return McpAuthRequiredResponse(sources=blocked)


class McpAuthRequiredError(AuthError):
    """Raised by ``submit_agent_job`` when a selected protected source is not connected.

    Subclasses :class:`AuthError` so the chat researcher's deep-research node
    surfaces ``str(self)`` to the user (instead of a generic failure) and the
    structured ``response`` is available for clients that want connect URLs.
    """

    error_code = "mcp_auth_required"

    def __init__(self, response: McpAuthRequiredResponse) -> None:
        self.response = response
        names = ", ".join(s.source_id for s in response.sources) or "the selected data source"
        super().__init__(
            f"Connect the following data source(s) before starting deep research: {names}. "
            "Open the data sources panel, click Connect to sign in, then try again."
        )
