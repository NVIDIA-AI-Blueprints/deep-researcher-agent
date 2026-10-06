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

"""Provider-neutral sandbox + artifact runtime for deep research.

Layout:
    base.py          SandboxProvider contract (execute required; optional hooks/capabilities)
    registry.py      register_sandbox_provider / create_sandbox_backend (the config-driven seam)
    config.py        SandboxConfig (common + providers.<name> + artifact_capture + lifecycle_scope)
    capabilities.py  SandboxCapabilities + fail-closed verification gate
    providers/       one module per provider (modal, openshell, ...) — each self-registers
    artifacts/       durable artifact records, manifest parsing, store, and harvester

Importing this package registers the built-in providers, so the registry is
populated before any config is validated against it.
"""

from __future__ import annotations

# Import providers for their registration side effects (built-ins self-register).
from . import providers as _providers  # noqa: E402,F401
from .artifacts import Artifact
from .artifacts import ArtifactManager
from .artifacts import ArtifactStore
from .artifacts import LocalArtifactStore
from .base import SandboxProvider
from .base import SandboxTerminatedError
from .capabilities import CapabilityError
from .capabilities import SandboxCapabilities
from .capabilities import verify_capabilities
from .config import NetworkPolicy
from .config import SandboxConfig
from .registry import SANDBOX_PROVIDER_ENTRY_POINT_GROUP
from .registry import create_sandbox_backend
from .registry import register_sandbox_provider
from .registry import registered_providers

__all__ = [
    "SandboxProvider",
    "SandboxTerminatedError",
    "SandboxConfig",
    "NetworkPolicy",
    "SandboxCapabilities",
    "CapabilityError",
    "verify_capabilities",
    "register_sandbox_provider",
    "create_sandbox_backend",
    "registered_providers",
    "SANDBOX_PROVIDER_ENTRY_POINT_GROUP",
    "Artifact",
    "ArtifactStore",
    "LocalArtifactStore",
    "ArtifactManager",
]
