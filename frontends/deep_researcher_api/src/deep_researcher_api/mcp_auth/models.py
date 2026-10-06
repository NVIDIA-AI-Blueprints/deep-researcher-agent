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

"""API response models for per-user MCP auth."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

AuthStatusLiteral = Literal["connected", "not_connected", "expired", "error"]


class PerUserAuthInfo(BaseModel):
    """Per-user MCP auth block attached to a data source in API responses.

    Combines the static declaration (``required``/``type``/``provider``/
    ``mcp_server_id`` from the registry) with the current user's dynamic state
    (``status``/``expires_at``/``last_error``) and the action URLs.

    ``connect_url`` is a stable Deep Researcher Agent API surface. ``auth_url`` is a short-lived
    provider/NAT login URL and is only populated when Deep Researcher Agent has intentionally
    started or reused an auth challenge — never from a read-only listing.
    """

    required: bool = False
    type: Literal["mcp_oauth2"] = "mcp_oauth2"
    provider: str | None = None
    mcp_server_id: str | None = None
    status: AuthStatusLiteral | None = None
    connect_url: str | None = None
    auth_url: str | None = None
    expires_at: datetime | None = None
    last_error: str | None = None


class SourceAuthStatusResponse(BaseModel):
    """Response for ``GET /v1/auth/mcp/{source_id}/status``."""

    source_id: str
    status: AuthStatusLiteral
    expires_at: datetime | None = None
    connect_url: str | None = None
    last_error: str | None = None


class SourceConnectResponse(BaseModel):
    """Response for ``POST /v1/auth/mcp/{source_id}/connect``."""

    source_id: str
    status: Literal["auth_required", "connected"] = "auth_required"
    auth_url: str | None = None
    expires_at: datetime | None = Field(
        default=None,
        description="Expiry of the auth challenge (auth_url), not of the eventual token",
    )


class McpAuthRequiredSource(BaseModel):
    """A single blocked source in a 409 mcp_auth_required response."""

    source_id: str
    status: AuthStatusLiteral
    connect_url: str
    auth_url: str | None = None


class McpAuthRequiredResponse(BaseModel):
    """Body of the 409 returned by submit preflight when sources need auth."""

    error: Literal["mcp_auth_required"] = "mcp_auth_required"
    message: str = "One or more selected data sources require connection before this job can start."
    sources: list[McpAuthRequiredSource] = Field(default_factory=list)
