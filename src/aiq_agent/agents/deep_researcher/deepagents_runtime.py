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

"""DeepAgents skills and sandbox runtime support for deep research."""

from __future__ import annotations

import importlib.util
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Literal
from uuid import uuid4

from deepagents.backends import CompositeBackend
from deepagents.backends import FilesystemBackend
from deepagents.backends import StateBackend
from deepagents.backends.state import create_file_data
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nat.data_models.function import FunctionBaseConfig

from .sandbox.config import ArtifactCaptureConfig
from .sandbox.logging_utils import log_sandbox_failure

logger = logging.getLogger(__name__)

BUILTIN_SKILLS_DIR = Path(__file__).with_name("skills")
BUILTIN_SKILL_SOURCE = "/skills/"
SHARED_ROUTE = "/shared/"
SKILL_AGENT_NAMES = frozenset({"researcher-agent", "writer-agent"})
CHART_SKILL_NAME = "chart-generation"
DEFAULT_WORKDIR = "/workspace"


class DeepResearchSkillsConfig(FunctionBaseConfig, name="deep_research_skills"):
    """AI-Q config surface for assigning built-in DeepAgents skill collections."""

    model_config = ConfigDict(extra="forbid")

    agents: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Per-agent built-in skill collection names keyed by DeepAgents agent name.",
    )
    require_sandbox: tuple[str, ...] = Field(
        default=(),
        description="Skill collection names that require a sandbox when assigned to any agent.",
    )

    @field_validator("agents")
    @classmethod
    def _validate_agent_names(cls, value: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        """Reject skill assignments to agent names that are not skill-bearing agents."""
        unknown = sorted(set(value) - SKILL_AGENT_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown deep research skill agent(s): {unknown}. Known agents: {sorted(SKILL_AGENT_NAMES)}"
            )
        return value


class DeepResearchSandboxConfig(FunctionBaseConfig, name="deep_research_sandbox"):
    """AI-Q config surface for the optional DeepAgents sandbox backend."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="openshell", description="Sandbox backend provider (resolved by registry).")
    # Modal-specific (used when provider == "modal").
    app_name: str = Field(default="aiq-deep-research", description="Modal app name for deep research sandboxes")
    image: str = Field(default="python:3.13-slim", description="Container image for Modal sandboxes")
    packages: tuple[str, ...] = Field(
        default=(),
        description="Python packages to install into the Modal sandbox image.",
    )
    # OpenShell-specific (used when provider == "openshell"). The secure default creates
    # a new policy-bound physical sandbox for each AI-Q job.
    openshell_image: str = Field(
        default="aiq-openshell-demo:latest",
        description="Container image used for per-job OpenShell sandboxes.",
    )
    existing_sandbox_name: str | None = Field(
        default=None,
        description="Debug-only existing OpenShell sandbox; requires allow_shared_sandbox=true.",
    )
    sandbox_name: str | None = Field(
        default=None,
        description="Deprecated alias for existing_sandbox_name; requires allow_shared_sandbox=true.",
    )
    allow_shared_sandbox: bool = Field(
        default=False,
        description="Allow debug attachment to a shared OpenShell sandbox; not job-isolated.",
    )
    gateway: str | None = Field(
        default=None,
        description="OpenShell gateway endpoint/name (null uses the locally selected gateway).",
    )
    workspace: str = Field(
        default="default",
        min_length=1,
        description="OpenShell workspace that scopes sandbox lifecycle operations.",
    )
    policy: str | None = Field(default=None, description="OpenShell policy YAML applied at per-job creation.")
    ready_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        description="Seconds to wait for the OpenShell sandbox to become ready.",
    )
    policy_load_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
        description="Seconds to wait for the authoritative OpenShell policy revision to become loaded.",
    )
    delete_on_exit: bool = Field(default=True, description="Delete the OpenShell sandbox when the session closes.")
    attest: bool = Field(default=True, description="Fail closed unless the OpenShell policy revision is loaded.")
    expected_policy_version: int | None = Field(
        default=None,
        ge=1,
        description="Optional exact OpenShell policy revision pin.",
    )
    require_hard_landlock: bool = Field(
        default=True,
        description="Require landlock.compatibility=hard_requirement in the configured policy.",
    )
    shell: tuple[str, ...] = Field(
        default=("bash", "-c"),
        description="Shell argv prefix passed to the langchain-nvidia-openshell adapter.",
    )
    # Shared across providers.
    workdir: str | None = Field(default=None, description="Working directory inside the sandbox")
    timeout: int = Field(default=1200, description="Maximum sandbox lifetime in seconds")
    idle_timeout: int = Field(default=1800, description="Sandbox idle timeout in seconds")
    network: Literal["blocked", "allowlist", "open", "enabled"] = Field(
        default="blocked",
        description="Outbound network policy; 'enabled' is a deprecated alias for 'open'.",
    )
    network_allow: tuple[str, ...] = Field(
        default=(),
        description="Hostnames allowed when network='allowlist'.",
    )
    artifact_capture: ArtifactCaptureConfig = Field(
        default_factory=ArtifactCaptureConfig,
        description="Durable harvesting of generated artifacts (charts/CSVs). Disabled by default.",
    )

    @model_validator(mode="after")
    def _validate_openshell_and_network(self) -> DeepResearchSandboxConfig:
        if self.network == "allowlist" and not self.network_allow:
            raise ValueError("network='allowlist' requires a non-empty network_allow list")
        if self.provider == "openshell":
            if (
                self.existing_sandbox_name is not None
                and self.sandbox_name is not None
                and self.existing_sandbox_name != self.sandbox_name
            ):
                raise ValueError("existing_sandbox_name and sandbox_name must match when both are set")
            shared_name = self.existing_sandbox_name or self.sandbox_name
            if shared_name and not self.allow_shared_sandbox:
                raise ValueError("existing_sandbox_name/sandbox_name requires allow_shared_sandbox=true")
            if not shared_name and not self.attest:
                raise ValueError("Per-job OpenShell creation requires attest=true")
        return self

    @property
    def network_mode(self) -> Literal["blocked", "allowlist", "open"]:
        """Return the normalized provider-neutral network mode."""
        return "open" if self.network == "enabled" else self.network


class DeepAgentsRuntime:
    """Build DeepAgents backend wiring and resolve AI-Q skill collections."""

    def __init__(
        self,
        *,
        skills: DeepResearchSkillsConfig | None = None,
        sandbox: DeepResearchSandboxConfig | None = None,
        job_id: str | None = None,
        artifact_db_url: str | None = None,
        artifact_emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Resolve skill sources and eagerly build the sandbox provider/artifact manager.

        Args:
            skills: Skill collections assigned per agent, or None for no skills.
            sandbox: Sandbox configuration, or None to run without a sandbox.
            job_id: Owning job id; a random id is generated when omitted.
            artifact_db_url: Database URL for durable artifact storage.
            artifact_emit: Optional SSE emitter for artifact events.
        """
        self._skills = skills
        self._sandbox = sandbox
        self._job_id = str(job_id) if job_id is not None else str(uuid4())
        self._backend: Any | None = None
        self._sandbox_provider: Any | None = None
        self._event_emit = artifact_emit
        self._finalize_lock = threading.Lock()
        self._finalized = False
        self._finalize_result: bool | None = None
        self.artifact_manager: Any | None = None
        self._artifact_finalize_lock = threading.Lock()
        self._artifact_finalize_attempted = False
        self._skill_sources_by_agent = _resolve_agent_skill_sources(skills)
        self._skill_sources = tuple(
            dict.fromkeys(source for sources in self._skill_sources_by_agent.values() for source in sources)
        )
        _validate_sandbox_requirements(
            skills=skills,
            sandbox=sandbox,
        )
        # Build the provider-neutral sandbox provider eagerly (lazy SDK session) so the
        # runtime can expose its job-scoped workdir/artifact_dir and own its lifecycle.
        if sandbox is not None:
            self._sandbox_provider = _create_sandbox_backend(sandbox, self._job_id)
            if hasattr(self._sandbox_provider, "set_event_emitter"):
                self._sandbox_provider.set_event_emitter(artifact_emit)
            self.artifact_manager = _maybe_build_artifact_manager(
                provider=self._sandbox_provider,
                job_id=self._job_id,
                artifact_dir=self.artifact_dir,
                artifact_db_url=artifact_db_url,
                artifact_emit=artifact_emit,
            )

    @property
    def execution_enabled(self) -> bool:
        """Return true when the runtime has a sandbox backend for execute."""
        return self._sandbox is not None

    @property
    def skills_enabled(self) -> bool:
        """Return true when any agent has configured skill sources."""
        return bool(self._skill_sources)

    @property
    def execute_timeout_seconds(self) -> int | None:
        """Per-call ``execute`` timeout ceiling (seconds), sourced from the sandbox config.

        Agent-supplied execute timeouts are unreliable (LLMs pass milliseconds where the
        backend expects seconds, or an arbitrary large value), so callers clamp to this
        configured sandbox lifetime to keep a single execute under the provider's hard cap.
        Returns None when no sandbox is configured (execute is unavailable anyway).
        """
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "timeout", None)

    def skill_sources_for(self, agent_name: str) -> list[str] | None:
        """Return DeepAgents source paths for an agent/subagent name."""
        sources = self._skill_sources_by_agent.get(agent_name)
        return list(sources) if sources else None

    def agent_has_chart_skill(self, agent_name: str) -> bool:
        """Return true when the agent's assigned collections provide the chart skill on disk."""
        sources = self._skill_sources_by_agent.get(agent_name)
        if not sources:
            return False
        collection_for_source = {source: name for name, source in discover_skill_collections().items()}
        return any(
            (BUILTIN_SKILLS_DIR / collection_for_source[source] / CHART_SKILL_NAME / "SKILL.md").exists()
            for source in sources
            if source in collection_for_source
        )

    @property
    def workdir(self) -> str:
        """Job-scoped sandbox working directory (or the default when no sandbox)."""
        if self._sandbox_provider is None:
            return DEFAULT_WORKDIR
        return self._sandbox_provider.workdir

    @property
    def artifact_dir(self) -> str:
        """Job-scoped sandbox artifact directory (harvest root) or the default."""
        if self._sandbox_provider is None:
            return f"{DEFAULT_WORKDIR}/aiq-artifacts"
        return self._sandbox_provider.artifact_dir

    @property
    def backend(self) -> Any:
        """Return the concrete backend instance passed to DeepAgents."""
        if self._backend is None:
            self._backend = _build_backend(
                provider=self._sandbox_provider,
                skills_enabled=self.skills_enabled,
            )
        return self._backend

    @property
    def sandbox_backend(self) -> Any | None:
        """Return the provider sandbox without DeepAgents virtual-route wrapping."""

        return self._sandbox_provider

    def final_harvest(self) -> None:
        """Best-effort final artifact harvest before cleanup (terminal job path)."""
        manager = self.artifact_manager
        if manager is None:
            return
        try:
            manager.final_harvest()
        except Exception as exc:  # noqa: BLE001 - harvest is best-effort on the terminal path
            log_sandbox_failure(
                logger,
                operation="final_artifact_harvest",
                reason_code="artifact_harvest_failed",
                exc=exc,
                provider=getattr(self._sandbox_provider, "provider_name", None),
                sandbox=self._job_id,
            )

    def close(self) -> None:
        """Release the sandbox provider on a normal terminal job path (idempotent)."""
        provider = self._sandbox_provider
        if provider is not None and hasattr(provider, "close"):
            provider.close()

    def terminate(self) -> None:
        """Forcibly stop the sandbox on an interrupted job (cancel/timeout), idempotent."""
        provider = self._sandbox_provider
        if provider is not None and hasattr(provider, "terminate"):
            provider.terminate()

    def finalize_artifacts(self, *, interrupted: bool) -> bool:
        """Run the terminal artifact scan without delaying cancellation.

        Normal success/failure paths perform the idempotent final scan. On cancellation,
        only scan when the provider operation lock is immediately available; otherwise
        execution teardown takes priority. Successful execute calls are checkpointed by
        middleware, so already completed artifacts remain durable in either case.
        """
        with self._artifact_finalize_lock:
            if self._artifact_finalize_attempted:
                return False
            self._artifact_finalize_attempted = True
        if self.artifact_manager is None or self._sandbox_provider is None:
            return False
        if not interrupted:
            self.final_harvest()
            return True
        lease = getattr(self._sandbox_provider, "try_operation_lease", None)
        if not callable(lease):
            return False
        with lease() as acquired:
            if not acquired:
                logger.info("Skipping terminal harvest for busy cancelled sandbox job %s", self._job_id)
                return False
            self.final_harvest()
            return True

    def finalize(self, *, interrupted: bool) -> bool:
        """Release the sandbox once and emit a truthful, sanitized cleanup outcome."""
        with self._finalize_lock:
            if self._finalized:
                assert self._finalize_result is not None
                return self._finalize_result

            # No provider means there is no sandbox lifecycle to report; skip the
            # cleanup events so non-sandbox jobs never emit an empty sandbox.cleanup.
            if self._sandbox_provider is None:
                self._finalize_result = True
                self._finalized = True
                return True

            self._emit_cleanup("started", interrupted=interrupted)
            try:
                if interrupted:
                    self.terminate()
                else:
                    self.close()
            except Exception as exc:  # noqa: BLE001 - terminal cleanup cannot replace the job result
                logger.warning("Sandbox cleanup failed for job %s (%s)", self._job_id, type(exc).__name__)
                succeeded = False
            else:
                succeeded = bool(getattr(self._sandbox_provider, "cleanup_succeeded", True))

            self._emit_cleanup("succeeded" if succeeded else "failed", interrupted=interrupted)
            self._finalize_result = succeeded
            self._finalized = True
            return succeeded

    def _emit_cleanup(self, status: str, *, interrupted: bool) -> None:
        """Emit cleanup metadata without policy contents, environment, or error text."""
        if self._event_emit is None:
            return
        try:
            self._event_emit(
                {
                    "type": "sandbox.cleanup",
                    "data": {
                        "status": status,
                        "interrupted": interrupted,
                        "provider": getattr(self._sandbox_provider, "provider_name", None),
                        "sandbox": getattr(self._sandbox_provider, "physical_sandbox_name", None)
                        or getattr(self._sandbox_provider, "sandbox_name", None),
                        "reason_codes": (
                            list(getattr(self._sandbox_provider, "cleanup_failure_reason_codes", ()))
                            if status == "failed"
                            else []
                        ),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - observability must not break terminal cleanup
            logger.warning("Sandbox cleanup event emission failed for job %s (%s)", self._job_id, type(exc).__name__)

    def prepare_state_files(self, files: dict[str, Any]) -> dict[str, Any]:
        """Normalize seeded virtual filesystem files for the configured backend."""
        return _normalize_state_files(files, strip_shared_route=self._sandbox is not None)


def _build_backend(
    *,
    provider: Any | None,
    skills_enabled: bool,
) -> Any:
    """Build the smallest stock DeepAgents backend needed for this run.

    The sandbox provider (a ``BaseSandbox``) is created once by the runtime so it can
    own the artifact manager and lifecycle; here it is simply used as the default backend.
    """
    default = provider if provider is not None else StateBackend()
    routes: dict[str, Any] = {}
    if skills_enabled:
        routes[BUILTIN_SKILL_SOURCE] = _skills_backend()
    if provider is not None:
        routes[SHARED_ROUTE] = StateBackend()
    if not routes:
        return default
    return CompositeBackend(default=default, routes=routes)


def _maybe_build_artifact_manager(
    *,
    provider: Any | None,
    job_id: str,
    artifact_dir: str,
    artifact_db_url: str | None,
    artifact_emit: Callable[[dict[str, Any]], None] | None,
) -> Any | None:
    """Build an ArtifactManager only when capture is enabled and a store URL is provided.

    Defaults to ``None`` (no harvesting) so adding the sandbox alone never requires a DB.
    """
    if provider is None or artifact_db_url is None:
        return None
    capture = getattr(getattr(provider, "config", None), "artifact_capture", None)
    if capture is None or not getattr(capture, "enabled", False):
        return None
    from .sandbox.artifacts import ArtifactManager
    from .sandbox.artifacts import build_artifact_store

    return ArtifactManager(
        job_id=job_id,
        backend=provider,
        store=build_artifact_store(artifact_db_url),
        config=capture,
        artifact_dir=artifact_dir,
        emit=artifact_emit,
    )


def _skills_backend() -> FilesystemBackend:
    """Return the filesystem-backed built-in skills route."""
    return FilesystemBackend(root_dir=BUILTIN_SKILLS_DIR.resolve(), virtual_mode=True)


def _normalize_state_files(files: dict[str, Any], *, strip_shared_route: bool) -> dict[str, Any]:
    """Return files in the shape expected by DeepAgents StateBackend.

    CompositeBackend strips a matched route before delegating to the route backend.
    When /shared/ is backed by StateBackend, seeded files must therefore be stored
    at the route-local path so reads for /shared/foo.md find /foo.md internally.
    """
    normalized: dict[str, Any] = {}
    for file_path, file_data in files.items():
        normalized_path = file_path
        if strip_shared_route and file_path == SHARED_ROUTE.rstrip("/"):
            normalized_path = "/"
        elif strip_shared_route and file_path.startswith(SHARED_ROUTE):
            normalized_path = f"/{file_path.removeprefix(SHARED_ROUTE)}"
        normalized[normalized_path] = _normalize_file_data(file_data)
    return normalized


def _normalize_file_data(file_data: Any) -> Any:
    """Return a DeepAgents file-data dict for raw seeded content."""
    if isinstance(file_data, dict):
        return file_data
    if isinstance(file_data, bytes):
        file_data = file_data.decode("utf-8")
    return create_file_data(str(file_data))


def discover_skill_collections(root: Path = BUILTIN_SKILLS_DIR) -> dict[str, str]:
    """Return built-in skill collection names mapped to DeepAgents source paths."""
    if not root.exists():
        return {}

    collections: dict[str, str] = {}
    for skill_file in sorted(root.glob("**/SKILL.md")):
        collection_dir = skill_file.parent.parent
        if collection_dir == root:
            continue
        name = collection_dir.relative_to(root).as_posix()
        collections[name] = f"{BUILTIN_SKILL_SOURCE}{name}/"
    return collections


def resolve_skill_collections(collection_names: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve public skill collection names to DeepAgents source paths."""
    known = discover_skill_collections()
    unknown = sorted(set(collection_names) - set(known))
    if unknown:
        raise ValueError(f"Unknown deep research skill collection(s): {unknown}. Known collections: {sorted(known)}")
    return tuple(known[name] for name in collection_names)


def _resolve_agent_skill_sources(skills: DeepResearchSkillsConfig | None) -> dict[str, tuple[str, ...]]:
    """Map each agent to its resolved skill source paths, skipping empty assignments."""
    if skills is None:
        return {}
    return {
        agent_name: resolve_skill_collections(collection_names)
        for agent_name, collection_names in skills.agents.items()
        if collection_names
    }


def _validate_sandbox_requirements(
    *,
    skills: DeepResearchSkillsConfig | None,
    sandbox: DeepResearchSandboxConfig | None,
) -> None:
    """Fail fast when a skill collection that requires a sandbox is assigned without one."""
    if skills is None or not skills.require_sandbox:
        return

    resolve_skill_collections(skills.require_sandbox)
    if sandbox is not None:
        return

    required_collections = set(skills.require_sandbox)
    violating_agents = sorted(
        agent_name
        for agent_name, collections in skills.agents.items()
        if required_collections.intersection(collections)
    )
    if violating_agents:
        raise ValueError(
            "Deep research skill collection(s) require a sandbox for agent(s) "
            f"{violating_agents}: {sorted(skills.require_sandbox)}. Configure sandbox or remove the collection(s)."
        )


def _create_sandbox_backend(config: DeepResearchSandboxConfig, job_id: str) -> Any:
    """Resolve the AI-Q sandbox config to a provider-neutral sandbox backend.

    Keeps the Modal dependency pre-check (clear early error when Modal is configured but
    not installed), then maps the config to the provider-neutral ``SandboxConfig`` and
    dispatches through the sandbox provider registry.
    """
    from .sandbox import create_sandbox_backend as registry_create
    from .sandbox.config import SandboxConfig as ProviderSandboxConfig

    provider = config.provider.lower()
    workdir = config.workdir or ("/workspace" if provider == "modal" else "/sandbox")

    if provider == "modal":
        _ensure_modal_dependencies()
        providers = {
            "modal": {
                "app_name": config.app_name,
                "image": config.image,
                "python_packages": config.packages,
            }
        }
    elif provider == "openshell":
        providers = {
            "openshell": {
                "gateway": config.gateway,
                "workspace": config.workspace,
                "existing_sandbox_name": config.existing_sandbox_name,
                "sandbox_name": config.sandbox_name,
                "allow_shared_sandbox": config.allow_shared_sandbox,
                "policy": config.policy,
                "image": config.openshell_image,
                "ready_timeout_seconds": config.ready_timeout_seconds,
                "policy_load_timeout_seconds": config.policy_load_timeout_seconds,
                "delete_on_exit": config.delete_on_exit,
                "attest": config.attest,
                "expected_policy_version": config.expected_policy_version,
                "require_hard_landlock": config.require_hard_landlock,
                "shell": list(config.shell),
            }
        }
    else:
        providers = {}

    provider_config = ProviderSandboxConfig.model_validate(
        {
            "provider": provider,
            "workdir": workdir,
            "timeout": config.timeout,
            "idle_timeout": config.idle_timeout,
            "network": {"mode": config.network_mode, "allow": config.network_allow},
            "artifact_capture": config.artifact_capture.model_dump(),
            "providers": providers,
        }
    )
    return registry_create(provider_config, job_id)


def _ensure_modal_dependencies() -> None:
    """Raise ImportError listing any missing Modal packages when Modal is configured."""
    missing = [
        package
        for module_name, package in (("modal", "modal"), ("langchain_modal", "langchain-modal"))
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        packages = ", ".join(missing)
        raise ImportError(
            "Modal sandbox is configured, but required package(s) are missing: "
            f"{packages}. Install the Modal sandbox dependencies or remove the sandbox config."
        )
