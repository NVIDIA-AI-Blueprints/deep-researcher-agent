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

"""Sandbox capability declarations and the fail-closed verification gate.

A provider declares what security/lifecycle guarantees it can enforce. The runtime
verifies the active :class:`SandboxConfig` against those declared capabilities
*before* creating a backend and refuses to run when a required guarantee is not
available (fail-closed). This keeps the provider interface thin while keeping the
security floor enforceable across every provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from .config import SandboxConfig


class SandboxCapabilities(BaseModel):
    """Security and lifecycle guarantees a provider declares it can enforce.

    Defaults are conservative: an unknown provider is assumed to support nothing,
    so the fail-closed gate refuses workloads that require guarantees the provider
    has not explicitly claimed.

    Attributes:
        supports_network_policy: Provider can enforce outbound network blocking.
        supports_network_allowlist: Provider can enforce a per-host egress allowlist.
        supports_filesystem_policy: Provider can restrict filesystem access at creation.
        supports_process_policy: Provider can restrict process/exec behavior.
        supports_resource_limits: Provider can cap CPU/memory/disk.
        supports_artifact_download: Provider implements byte-accurate ``download_files``.
        supports_cleanup: Provider implements ``close`` for deterministic teardown.
        supports_terminate: Provider can forcibly terminate a running execution.
    """

    supports_network_policy: bool = Field(default=False)
    supports_network_allowlist: bool = Field(default=False)
    supports_filesystem_policy: bool = Field(default=False)
    supports_process_policy: bool = Field(default=False)
    supports_resource_limits: bool = Field(default=False)
    supports_artifact_download: bool = Field(default=False)
    supports_cleanup: bool = Field(default=False)
    supports_terminate: bool = Field(default=False)


class CapabilityError(ValueError):
    """Raised when a sandbox config requires a guarantee the provider cannot enforce."""


def verify_capabilities(config: SandboxConfig, capabilities: SandboxCapabilities) -> None:
    """Fail closed if the config demands a guarantee the provider does not declare.

    Args:
        config: The resolved sandbox configuration for the job.
        capabilities: The selected provider's declared capabilities.

    Raises:
        CapabilityError: If a required guarantee (e.g. network policy, allowlist, or
            artifact capture) is requested but unsupported by the provider.
    """
    mode = config.network.mode
    if mode == "blocked" and not capabilities.supports_network_policy:
        raise CapabilityError(
            f"Provider '{config.provider}' cannot enforce network.mode='blocked' (block_network). "
            "Refusing to run code with un-enforceable network policy. "
            "Choose a provider that declares supports_network_policy, or set network.mode='open' explicitly."
        )

    if mode == "allowlist" and not capabilities.supports_network_allowlist:
        raise CapabilityError(
            f"Provider '{config.provider}' cannot enforce network.mode='allowlist'. "
            "This provider's network policy is all-or-nothing. "
            "Choose a provider that declares supports_network_allowlist, or use network.mode='blocked'/'open'."
        )

    if config.resources.any_set() and not capabilities.supports_resource_limits:
        raise CapabilityError(
            f"Provider '{config.provider}' cannot enforce resource limits (CPU/memory), but "
            "sandbox.resources requests one. Refusing to run with un-enforceable limits. "
            "Choose a provider that declares supports_resource_limits, or remove sandbox.resources."
        )

    if config.artifact_capture.enabled and not capabilities.supports_artifact_download:
        raise CapabilityError(
            f"Provider '{config.provider}' cannot download artifacts (no download_files support), "
            "but artifact_capture.enabled=true. Disable artifact capture or choose a provider that supports it."
        )
