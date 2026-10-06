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

"""Per-user MCP auth control plane for DeepResearcher.

Deep Researcher Agent owns the product control plane (status surfacing, connect actions,
structured auth-required responses, submit-time preflight). The actual OAuth
mechanics and per-user token storage are NAT's. Deep Researcher Agent mints the provider
authorization URL and completes the callback using NAT's public OAuth/token
primitives, then writes the resulting token into the *same* NAT token storage
that the headless job-time ``per_user_mcp_client`` reads from. See
``deep-researcher-nat-oauth-execution-gap`` design notes.
"""

from .models import McpAuthRequiredResponse
from .models import McpAuthRequiredSource
from .models import PerUserAuthInfo
from .models import SourceAuthStatusResponse
from .models import SourceConnectResponse
from .provider import AuthStatus
from .provider import ProtectedSourceAuthProvider
from .provider import SourceAuthChallenge
from .provider import SourceAuthState
from .provider import principal_user_id

__all__ = [
    "AuthStatus",
    "McpAuthRequiredResponse",
    "McpAuthRequiredSource",
    "PerUserAuthInfo",
    "ProtectedSourceAuthProvider",
    "SourceAuthChallenge",
    "SourceAuthState",
    "SourceAuthStatusResponse",
    "SourceConnectResponse",
    "principal_user_id",
]
