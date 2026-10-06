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

"""Provider registry — the single, config-driven swap point.

A provider registers its class under a name; the config's ``provider`` field is
validated against this registry; and :func:`create_sandbox_backend` is the one
place the runtime resolves a name to a backend. Adding a provider never edits this
module — the provider self-registers at import.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .capabilities import verify_capabilities
from .logging_utils import log_sandbox_failure

if TYPE_CHECKING:
    from .base import SandboxProvider
    from .config import SandboxConfig

logger = logging.getLogger(__name__)

#: Python entry-point group third-party packages use to contribute providers.
#: Declare in a distribution's pyproject as, e.g.::
#:
#:     [project.entry-points."deep-researcher.sandbox_providers"]
#:     mybox = "my_pkg.provider:MySandboxProvider"
SANDBOX_PROVIDER_ENTRY_POINT_GROUP = "deep-researcher.sandbox_providers"

_SANDBOX_PROVIDERS: dict[str, type[SandboxProvider]] = {}
_entry_points_loaded = False


def register_sandbox_provider(name: str, provider_cls: type[SandboxProvider]) -> None:
    """Register a provider class under a config-facing name.

    Args:
        name: Provider key used in ``SandboxConfig.provider`` and YAML config.
        provider_cls: A ``SandboxProvider`` subclass constructed as ``(config, job_id)``.
    """
    key = name.lower()
    if key in _SANDBOX_PROVIDERS and _SANDBOX_PROVIDERS[key] is not provider_cls:
        logger.warning("Overriding already-registered sandbox provider '%s'", key)
    _SANDBOX_PROVIDERS[key] = provider_cls


def _load_entry_point_providers() -> None:
    """Discover and register third-party providers via the entry-point group.

    Idempotent and best-effort: a broken or missing plugin is logged and skipped so
    it can never take down provider resolution for the built-ins. Built-in providers
    register eagerly at package import; this only adds external contributions.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    from importlib.metadata import entry_points

    try:
        discovered = entry_points(group=SANDBOX_PROVIDER_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 select-by-group fallback
        discovered = entry_points().get(SANDBOX_PROVIDER_ENTRY_POINT_GROUP, [])

    for entry_point in discovered:
        try:
            provider_cls = entry_point.load()
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break built-in resolution
            log_sandbox_failure(
                logger,
                operation="provider_entry_point_load",
                reason_code="provider_load_failed",
                exc=exc,
                provider=entry_point.name,
            )
            continue
        register_sandbox_provider(entry_point.name, provider_cls)
        logger.info("Registered sandbox provider '%s' from entry point", entry_point.name)


def is_registered(name: str) -> bool:
    """Return whether a provider name is registered (loading entry points on first use)."""
    _load_entry_point_providers()
    return name.lower() in _SANDBOX_PROVIDERS


def registered_providers() -> list[str]:
    """Return the sorted list of registered provider names (built-in + entry-point)."""
    _load_entry_point_providers()
    return sorted(_SANDBOX_PROVIDERS)


def create_sandbox_backend(config: SandboxConfig, job_id: str) -> SandboxProvider:
    """Resolve the configured provider to a backend and verify its capabilities.

    Args:
        config: Resolved sandbox configuration.
        job_id: Async job identifier used to scope the sandbox.

    Returns:
        A constructed, capability-verified ``SandboxProvider``.

    Raises:
        ValueError: If the provider name is not registered.
        CapabilityError: If the provider cannot enforce a required guarantee (fail-closed).
    """
    _load_entry_point_providers()
    provider_cls = _SANDBOX_PROVIDERS.get(config.provider)
    if provider_cls is None:
        registered = ", ".join(registered_providers()) or "(none registered)"
        raise ValueError(f"Unsupported sandbox provider: {config.provider}. Registered providers: {registered}")
    backend = provider_cls(config, job_id)
    verify_capabilities(config, backend.capabilities)
    return backend
