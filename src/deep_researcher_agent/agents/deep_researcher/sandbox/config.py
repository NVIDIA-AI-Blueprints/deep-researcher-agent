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

"""Provider-neutral sandbox configuration models.

Common fields apply to every provider; provider-specific settings live under
``providers.<name>``. A backward-compatible validator lifts legacy flat Modal
fields into ``providers.modal`` so pre-existing configs keep loading. The
``provider`` field is validated against the registry, so any registered provider
is automatically accepted with no edits here.
"""

from __future__ import annotations

import math
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

DEFAULT_WORKDIR = "/workspace"


def _safe_job_segment(job_id: str) -> str:
    """Return a filename-safe single path segment for ``job_id``.

    The job id becomes a directory name, so keep only filename-safe characters: a
    crafted id containing ``/`` or ``..`` must not be able to escape the base workdir.
    """
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in job_id) or "job"


def job_scoped_workdir(base_workdir: str, job_id: str) -> str:
    """Return the per-job working directory ``<base_workdir>/<safe job id>``.

    Per-job directories prevent accidental filename collisions in a shared sandbox.
    They are organization, not an access-control boundary: code running in one job can
    still access sibling directories unless the provider enforces stronger isolation.
    """
    return f"{base_workdir.rstrip('/')}/{_safe_job_segment(job_id)}"


def job_scoped_artifact_dir(base_workdir: str, job_id: str) -> str:
    """Return the per-job artifact directory nested under the job working directory."""
    return f"{job_scoped_workdir(base_workdir, job_id)}/deep-researcher-artifacts"


# Allowed artifact extensions for the first capture milestone (validated MIME-from-bytes
# happens in the ArtifactManager; this is the coarse filename allowlist).
_DEFAULT_ALLOW_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".svg",
    ".csv",
    ".json",
    ".md",
    ".ipynb",
    ".pdf",
)


class ModalProviderConfig(BaseModel):
    """Modal-specific sandbox settings."""

    app_name: str = Field(
        default="deep-researcher-deep-research", description="Modal app name for deep research sandboxes"
    )
    image: str = Field(default="python:3.12-slim", description="Container image for Modal sandboxes")
    python_packages: tuple[str, ...] = Field(
        default=(),
        description="Python packages to install into the Modal sandbox image (e.g. matplotlib, pandas).",
    )


class OpenShellProviderConfig(BaseModel):
    """OpenShell-specific sandbox settings (enterprise/on-prem example provider)."""

    gateway: str | None = Field(default=None, description="OpenShell gateway/cluster endpoint or name")
    workspace: str = Field(
        default="default",
        min_length=1,
        description="OpenShell workspace that scopes sandbox lifecycle operations.",
    )
    existing_sandbox_name: str | None = Field(
        default=None,
        description="Debug-only existing OpenShell sandbox to attach to; requires allow_shared_sandbox=true.",
    )
    sandbox_name: str | None = Field(
        default=None,
        description="Deprecated alias for existing_sandbox_name; requires allow_shared_sandbox=true.",
    )
    allow_shared_sandbox: bool = Field(
        default=False,
        description="Allow debug attachment to a shared sandbox; this does not provide per-job isolation.",
    )
    policy: str | None = Field(
        default=None,
        description="OpenShell policy YAML applied when the per-job sandbox is created.",
    )
    image: str = Field(default="deep-researcher-openshell-demo:latest", description="OpenShell image identifier")
    ready_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        description="Seconds to wait for the sandbox to become ready",
    )
    policy_load_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Seconds to wait for the authoritative policy revision to become loaded.",
    )
    cleanup_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Seconds to wait for an in-flight OpenShell context teardown to finish.",
    )
    delete_on_exit: bool = Field(default=True, description="Delete the sandbox when its session context closes")
    attest: bool = Field(default=True, description="Require READY state and a loaded policy revision before execution")
    expected_policy_version: int | None = Field(
        default=None,
        ge=1,
        description="Optional exact OpenShell policy revision pin.",
    )
    require_hard_landlock: bool = Field(
        default=True,
        description="Reject policy files that do not require Landlock filesystem enforcement.",
    )
    shell: tuple[str, ...] = Field(
        default=("bash", "-c"),
        description="Shell argv prefix passed to the langchain-nvidia-openshell adapter.",
    )

    @field_validator("cleanup_timeout_seconds")
    @classmethod
    def _cleanup_timeout_must_be_finite(cls, value: float) -> float:
        """Reject non-finite teardown deadlines that would defeat bounded finalization."""
        if not math.isfinite(value):
            raise ValueError("cleanup_timeout_seconds must be finite")
        return value

    @field_validator("policy_load_timeout_seconds")
    @classmethod
    def _policy_load_timeout_must_be_finite(cls, value: float) -> float:
        """Reject non-finite attestation deadlines that would defeat fail-closed startup."""
        if not math.isfinite(value):
            raise ValueError("policy_load_timeout_seconds must be finite")
        return value

    @model_validator(mode="after")
    def _validate_lifecycle_mode(self) -> OpenShellProviderConfig:
        """Keep shared attachment explicit and per-job creation fail-closed."""
        if self.sandbox_name and self.existing_sandbox_name and self.sandbox_name != self.existing_sandbox_name:
            raise ValueError("Set only one of sandbox_name or existing_sandbox_name.")
        shared_name = self.existing_sandbox_name or self.sandbox_name
        if shared_name and not self.allow_shared_sandbox:
            raise ValueError(
                "Attaching to an existing OpenShell sandbox is debug-only and requires allow_shared_sandbox=true."
            )
        if shared_name and self.policy and not self.attest:
            raise ValueError("A policy-configured shared OpenShell sandbox requires attest=true.")
        if not shared_name:
            if not self.delete_on_exit:
                raise ValueError("Per-job OpenShell sandboxes require delete_on_exit=true.")
            if not self.attest:
                raise ValueError("Per-job OpenShell sandboxes require attest=true.")
        if self.expected_policy_version is not None and not self.attest:
            raise ValueError("expected_policy_version requires attest=true.")
        if not self.shell or any(not part.strip() for part in self.shell):
            raise ValueError("OpenShell shell must contain at least one non-empty argv element.")
        return self

    @property
    def shared_sandbox_name(self) -> str | None:
        """Return the explicitly configured debug attachment target, if any."""
        return self.existing_sandbox_name or self.sandbox_name


class SandboxProvidersConfig(BaseModel):
    """Per-provider configuration blocks. Add a provider by adding an optional field here."""

    modal: ModalProviderConfig = Field(default_factory=ModalProviderConfig)
    openshell: OpenShellProviderConfig = Field(default_factory=OpenShellProviderConfig)


class ArtifactCaptureConfig(BaseModel):
    """Controls durable harvesting of generated binary/rich artifacts."""

    enabled: bool = Field(default=False, description="Enable artifact harvesting from the sandbox")
    max_file_bytes: int = Field(default=50_000_000, description="Maximum size of a single harvested artifact")
    max_total_bytes: int = Field(default=500_000_000, description="Maximum total artifact bytes per job (quota)")
    max_file_count: int = Field(default=200, description="Maximum number of artifacts per job")
    allow_extensions: tuple[str, ...] = Field(
        default=_DEFAULT_ALLOW_EXTENSIONS,
        description="Filename extension allowlist for captured artifacts.",
    )


class NetworkPolicy(BaseModel):
    """Provider-neutral outbound network policy.

    One normalized shape every provider maps to its native mechanism (Modal's
    ``block_network`` flag, OpenShell's gateway policy file, etc.):

    * ``blocked`` - no outbound network (the safe default).
    * ``allowlist`` - only the hosts in ``allow`` are reachable.
    * ``open`` - unrestricted (use only for trusted workloads).
    """

    mode: Literal["blocked", "allowlist", "open"] = Field(
        default="blocked",
        description="Outbound network policy mode enforced inside the sandbox.",
    )
    allow: tuple[str, ...] = Field(
        default=(),
        description="Allowed hostnames/domains; only used (and required) when mode='allowlist'.",
    )

    @model_validator(mode="after")
    def _validate_allowlist(self) -> NetworkPolicy:
        """An allowlist policy is meaningless without hosts; fail loudly at config time."""
        if self.mode == "allowlist" and not self.allow:
            raise ValueError("network.mode='allowlist' requires a non-empty network.allow list of hosts.")
        return self


class ResourceLimits(BaseModel):
    """Provider-neutral CPU/memory caps for the sandbox (opt-in).

    Both default to ``None`` (no limit), so an unset ``resources`` block changes nothing.
    When a limit is set, the fail-closed capability gate refuses to run on a provider that
    cannot enforce it (``supports_resource_limits``), rather than silently ignoring it.
    Disk quotas are intentionally omitted: no current provider can enforce them, so a disk
    field would be unenforceable.
    """

    cpu: float | None = Field(default=None, gt=0, description="Max CPU cores (provider-enforced).")
    memory_mb: int | None = Field(default=None, gt=0, description="Max memory in MB (provider-enforced).")

    def any_set(self) -> bool:
        """Whether any limit is requested (so the capability gate applies)."""
        return self.cpu is not None or self.memory_mb is not None


class SandboxConfig(BaseModel):
    """Provider-neutral configuration for a DeepAgents sandbox backend.

    Swapping providers is a config-only change: set ``provider`` and the matching
    ``providers.<name>`` block. Common fields (workdir, network, timeouts, artifact
    capture, lifecycle scope) apply to every provider.
    """

    enabled: bool = Field(default=True, description="Whether the sandbox is active for this agent")
    provider: str = Field(default="modal", description="Sandbox backend provider (must be registered).")
    lifecycle_scope: Literal["job", "skill", "subagent"] = Field(
        default="job",
        description="Isolation scope for the sandbox. 'job' shares one sandbox across subagents.",
    )
    workdir: str = Field(default=DEFAULT_WORKDIR, description="Writable working directory inside the sandbox")
    network: NetworkPolicy = Field(
        default_factory=NetworkPolicy,
        description="Normalized outbound network policy. Legacy `block_network: bool` is lifted into this.",
    )
    timeout: int = Field(default=1200, description="Maximum sandbox lifetime in seconds")
    idle_timeout: int = Field(default=1800, description="Sandbox idle timeout in seconds")
    resources: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Optional CPU/memory caps; enforced only by providers declaring supports_resource_limits.",
    )
    artifact_capture: ArtifactCaptureConfig = Field(default_factory=ArtifactCaptureConfig)
    providers: SandboxProvidersConfig = Field(default_factory=SandboxProvidersConfig)

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_block_network(cls, data: Any) -> Any:
        """Lift legacy ``block_network: bool`` into the normalized ``network`` policy.

        Explicit ``network`` always wins; ``block_network`` is only consulted when
        ``network`` is not provided, so old configs keep working unchanged.
        """
        if not isinstance(data, dict) or "block_network" not in data:
            return data
        data = dict(data)
        legacy_block = data.pop("block_network")
        if isinstance(legacy_block, str):
            # Env-interpolated values arrive as strings. Reject anything unrecognized rather
            # than mapping it to falsy, so a typo (e.g. "flase") cannot silently open egress
            # on what the operator intended to be a network-blocked sandbox.
            raw = legacy_block.strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                legacy_block = True
            elif raw in {"0", "false", "no", "off", ""}:
                legacy_block = False
            else:
                raise ValueError(
                    f"Invalid block_network value {legacy_block!r}; expected a boolean "
                    "(true/false). Use the 'network' policy for finer control."
                )
        if "network" not in data or data.get("network") is None:
            data["network"] = {"mode": "blocked" if legacy_block else "open"}
        return data

    @field_validator("provider")
    @classmethod
    def _provider_must_be_registered(cls, value: str) -> str:
        """Validate the provider name against the registry (the single source of truth)."""
        from .registry import is_registered
        from .registry import registered_providers

        provider = value.lower()
        if not is_registered(provider):
            registered = ", ".join(registered_providers()) or "(none registered)"
            raise ValueError(f"Unsupported sandbox provider: {value}. Registered providers: {registered}")
        return provider

    @property
    def block_network(self) -> bool:
        """Back-compat accessor: any non-``open`` network policy blocks unrestricted egress."""
        return self.network.mode != "open"

    @property
    def python_packages(self) -> tuple[str, ...]:
        """Active provider's package list (Modal). Empty for providers without one."""
        return self.providers.modal.python_packages
