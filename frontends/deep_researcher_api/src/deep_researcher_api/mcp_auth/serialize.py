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

"""Helpers to project registry + auth state into API models."""

from __future__ import annotations

from deep_researcher_agent.auth import Principal
from deep_researcher_agent.common.data_source_registry import DataSourceMeta

from .models import PerUserAuthInfo
from .provider import ProtectedSourceAuthProvider

# Statuses for which the client should be offered a way to (re)connect.
_ACTIONABLE = {"not_connected", "expired", "error"}


def connect_url_for(source_id: str) -> str:
    return f"/v1/auth/mcp/{source_id}/connect"


def status_url_for(source_id: str) -> str:
    return f"/v1/auth/mcp/{source_id}/status"


async def build_listing_auth_info(
    provider: ProtectedSourceAuthProvider,
    principal: Principal,
    source: DataSourceMeta,
) -> PerUserAuthInfo | None:
    """Build the ``per_user_auth`` block for ``GET /v1/data_sources``.

    Read-only: never mints ``auth_url`` (no OAuth state is created here). Returns
    ``None`` for sources with no per-user auth declaration so the field is
    omitted entirely.
    """
    pua = source.per_user_auth
    if pua is None:
        return None

    info = PerUserAuthInfo(
        required=pua.required,
        type=pua.type,
        provider=pua.provider,
        mcp_server_id=pua.mcp_server_id,
    )
    if not pua.required:
        return info

    state = await provider.get_status(principal, source.id)
    info.status = state.status
    info.expires_at = state.expires_at
    info.last_error = state.last_error
    if state.status in _ACTIONABLE:
        info.connect_url = connect_url_for(source.id)
    return info
