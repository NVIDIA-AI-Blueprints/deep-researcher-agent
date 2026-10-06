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

"""Process-global handle to the per-user MCP auth provider.

The provider is built once at route registration in the API process. Both the
REST submit route and programmatic submitters (e.g. the chat researcher's async
deep-research submit) run in that same process, so they can reach the live
provider through this module to run the connect-state preflight before enqueue.

It is intentionally a simple module-global: there is one provider per API
process and no per-request state. Returns ``None`` when MCP auth was never
registered (e.g. unit tests that call ``submit_agent_job`` directly), in which
case callers skip the preflight — there is nothing to enforce.
"""

from __future__ import annotations

from .provider import ProtectedSourceAuthProvider

_active_provider: ProtectedSourceAuthProvider | None = None


def set_active_mcp_auth_provider(provider: ProtectedSourceAuthProvider | None) -> None:
    """Register the provider built at route setup as the process-wide instance."""
    global _active_provider
    _active_provider = provider


def get_active_mcp_auth_provider() -> ProtectedSourceAuthProvider | None:
    """Return the registered provider, or ``None`` if MCP auth is not configured."""
    return _active_provider
