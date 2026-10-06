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

"""SandboxProvider compliance suite.

Mirrors the knowledge-layer adapter compliance harness: every registered provider
must satisfy the same contract. Providers whose optional SDK is not installed
(e.g. OpenShell) are skipped rather than failed, so this runs without a live gateway.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxCapabilities
from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxConfig
from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxProvider
from deep_researcher_agent.agents.deep_researcher.sandbox import create_sandbox_backend

_BUILTIN_PROVIDERS = ("modal", "openshell")


def assert_provider_contract(provider: SandboxProvider) -> None:
    """Assert a provider honors the SandboxProvider contract (no live session needed)."""
    # Declared capabilities are a real SandboxCapabilities model.
    assert isinstance(provider.capabilities, SandboxCapabilities)

    # Identity is a non-empty job-scoped name before any session exists.
    assert isinstance(provider.sandbox_name, str) and provider.sandbox_name
    assert provider.id == provider.sandbox_name

    # Error classification is conservative for unrelated errors.
    assert provider.is_recoverable_error(ValueError("unrelated")) is False

    # close() is idempotent and safe with no live session.
    provider.close()
    provider.close()

    # The shared resilience path delegates to the session created by _create_session, and
    # prepares the per-job workspace (idempotent mkdir -p) before the first real call.
    session = MagicMock()
    session.execute.return_value = "ok"
    provider._create_session = lambda: session  # type: ignore[method-assign]
    assert provider.execute("echo ok", timeout=5) == "ok"
    first_cmd = session.execute.call_args_list[0].args[0]
    assert first_cmd.startswith("mkdir -p") and provider.workdir in first_cmd
    session.execute.assert_called_with("echo ok", timeout=5)  # most recent call is the real command


@pytest.mark.parametrize("provider_name", _BUILTIN_PROVIDERS)
def test_builtin_provider_compliance(provider_name: str) -> None:
    config = SandboxConfig(provider=provider_name, block_network=False)
    try:
        provider = create_sandbox_backend(config, "compliance-job-123")
    except ImportError:
        pytest.skip(f"{provider_name} SDK/adapter not installed")
    assert_provider_contract(provider)
