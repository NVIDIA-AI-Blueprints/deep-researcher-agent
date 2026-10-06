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

"""OpenShell sandbox provider (enterprise/on-prem).

Governed, policy-enforced execution on local Docker/Podman/Kubernetes/microVM via
the OpenShell gateway. The deepagents ``BaseSandbox`` adapter is the official
``langchain-nvidia-openshell`` partner package (``OpenShellSandbox``), the same
adapter Deep Researcher Agent PR #274 integrates. Both the ``openshell`` SDK and the adapter are
intentionally NOT declared in ``pyproject``; they are optional dependencies
imported lazily, so this provider is never force-installed. The canonical
operator workflow lives in ``docs/source/deployment/openshell.md``.
"""

from __future__ import annotations

import base64
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from typing import Any

from deepagents.backends.protocol import FileDownloadResponse
from deepagents.backends.protocol import FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from ..base import SandboxProvider
from ..base import SandboxTerminatedError
from ..capabilities import SandboxCapabilities
from ..logging_utils import log_sandbox_failure
from ..registry import register_sandbox_provider

if TYPE_CHECKING:
    from ..config import SandboxConfig

logger = logging.getLogger(__name__)

# Migration switch: when set truthy, delegate file transfer to the official adapter's
# upload_files/download_files instead of the local env-free shim. Use this to validate the
# upstream argv fix (langchain-ai/langchain-nvidia PR #303); once that ships, the shim and
# this switch can be removed and the adapter used unconditionally.
_ADAPTER_FILE_TRANSFER_ENV = "DEEP_RESEARCHER_OPENSHELL_ADAPTER_FILE_TRANSFER"


def _adapter_file_transfer_enabled() -> bool:
    """True only when the toggle env var is an explicit truthy value (not just any string)."""
    return os.getenv(_ADAPTER_FILE_TRANSFER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# File-transfer bootstraps pass the path via argv (not env): OpenShell <=0.0.67 strips
# OPENSHELL_* env before exec, breaking the adapter's env-based transfer. We keep the
# adapter for execute and override only these two methods until the SDK propagates env.
# The download bootstrap fails closed before reading untrusted bytes: reject a symlink
# leaf (exit 5) or directory (exit 3), and read at most cap+1 bytes (exit 4 if over) so
# an oversized/out-of-tree file is never pulled into host memory.
_UPLOAD_CODE = (
    "import base64,os,sys;"
    "p=sys.argv[1];"
    "d=os.path.dirname(p);"
    "(os.makedirs(d,exist_ok=True) if d else None);"
    "open(p,'wb').write(base64.b64decode(sys.stdin.buffer.read()))"
)
_DOWNLOAD_CODE = (
    "import base64,os,sys;"
    "p=sys.argv[1];"
    "limit=int(sys.argv[2]);"
    "root=os.path.realpath(sys.argv[3]);"
    "rp=os.path.realpath(p);"
    "(sys.exit(5) if not (rp==root or rp.startswith(root+os.sep)) else None);"
    "(sys.exit(3) if os.path.isdir(rp) else None);"
    "b=open(rp,'rb').read(limit+1);"
    "(sys.exit(4) if len(b)>limit else None);"
    "sys.stdout.write(base64.b64encode(b).decode())"
)

# Bootstrap exit codes mapped to a download error reason (see _DOWNLOAD_CODE).
_DOWNLOAD_EXIT_ERRORS = {3: "is_directory", 4: "too_large", 5: "symlink_rejected"}


def _classify_fs_error(text: str) -> str:
    """Map sandbox-side stderr to a deepagents FileOperationError literal."""
    lowered = text.lower()
    if "no such file" in lowered or "file not found" in lowered or "filenotfounderror" in lowered:
        return "file_not_found"
    if "is a directory" in lowered or "isadirectoryerror" in lowered:
        return "is_directory"
    if "invalid" in lowered and "path" in lowered:
        return "invalid_path"
    return "permission_denied"


_OPENSHELL_IMPORT_HINT = (
    "The OpenShell sandbox provider requires the `openshell>=0.0.88,<0.1` SDK and the "
    "`langchain-nvidia-openshell` adapter (published on PyPI). They are optional, separately installed "
    "dependencies. Install them with `./scripts/openshell/setup_openshell.sh` (which installs "
    "`langchain-nvidia-openshell` from PyPI; override the source via `LANGCHAIN_NVIDIA_REPO`), "
    "and configure an OpenShell gateway before enabling this provider."
)


def _normalize_openshell_name(job_id: str, prefix: str = "deep-researcher-deep-research") -> str:
    """Normalize a job id into a DNS-style, length-bounded OpenShell sandbox name."""
    raw = f"{prefix}-{job_id}" if prefix else job_id
    normalized = re.sub(r"[^a-z0-9-]+", "-", raw.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return (normalized[:63].rstrip("-")) or prefix


def _sandbox_labels(job_id: str) -> dict[str, str]:
    """Return stable gateway and runtime ownership labels for one Deep Researcher Agent job."""
    return {
        "deep-researcher": "deep-research",
        "deep-researcher-job-id": _normalize_openshell_name(job_id, prefix=""),
    }


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    """Return whether a public SDK callable accepts one named keyword."""
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _is_openshell_not_found_error(exc: Exception) -> bool:
    """Best-effort classification of OpenShell stale-sandbox errors."""
    text = str(exc).lower()
    return "not found" in text and ("sandbox" in text or exc.__class__.__module__.startswith("openshell"))


def _read_policy_data(policy_path: str, *, require_hard_landlock: bool) -> dict[str, Any]:
    """Read and normalize OpenShell policy YAML without importing the optional SDK."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    path = Path(policy_path).expanduser()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read OpenShell policy file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid OpenShell policy YAML: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"OpenShell policy must be a YAML mapping: {path}")
    if raw.get("version") != 1:
        raise ValueError("OpenShell policy version must be exactly 1")

    landlock = raw.get("landlock")
    compatibility = landlock.get("compatibility") if isinstance(landlock, dict) else None
    if require_hard_landlock and compatibility != "hard_requirement":
        raise ValueError(
            "OpenShell production policy requires landlock.compatibility=hard_requirement; "
            "set require_hard_landlock=false only for an explicit local demo."
        )

    filesystem = raw.get("filesystem_policy", raw.get("filesystem"))
    if not isinstance(filesystem, dict) or not any(filesystem.get(key) for key in ("read_only", "read_write")):
        raise ValueError("OpenShell production policy requires non-empty filesystem read_only or read_write rules")

    process = raw.get("process")
    if not isinstance(process, dict):
        raise ValueError("OpenShell production policy requires a process policy")
    for field in ("run_as_user", "run_as_group"):
        identity = process.get(field)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"OpenShell production policy requires process.{field} to be a non-empty string")
        if identity.strip().lower() in {"0", "root"}:
            raise ValueError(f"OpenShell production policy requires a non-root process.{field}")

    network_policies = raw.get("network_policies") or {}
    if not isinstance(network_policies, dict):
        raise ValueError("OpenShell network_policies must be a mapping")
    for policy_name, network_policy in network_policies.items():
        if not isinstance(network_policy, dict):
            raise ValueError(f"OpenShell network policy {policy_name!r} must be a mapping")
        endpoints = network_policy.get("endpoints") or []
        if not isinstance(endpoints, list):
            raise ValueError(f"OpenShell network policy {policy_name!r} endpoints must be a list")
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                raise ValueError(f"OpenShell network policy {policy_name!r} contains an invalid endpoint")
            if endpoint.get("enforcement") != "enforce" or endpoint.get("access") != "read-only":
                raise ValueError(
                    f"OpenShell network policy {policy_name!r} endpoints must use "
                    "enforcement=enforce and access=read-only"
                )

    # OpenShell's YAML schema calls this field filesystem_policy while the
    # Python proto calls it filesystem. Keep this compatibility translation in one
    # place and reject every other unknown field through ParseDict below.
    policy_data = dict(raw)
    if "filesystem_policy" in policy_data:
        if "filesystem" in policy_data:
            raise ValueError("OpenShell policy cannot contain both filesystem_policy and filesystem")
        policy_data["filesystem"] = policy_data.pop("filesystem_policy")
    return policy_data


def _parse_policy_proto(policy_data: dict[str, Any], *, policy_path: str) -> Any:
    """Parse one validated policy snapshot into the SDK proto with strict field validation."""
    try:
        from google.protobuf.json_format import ParseDict
        from openshell._proto import sandbox_pb2
    except ImportError as exc:
        raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    try:
        return ParseDict(policy_data, sandbox_pb2.SandboxPolicy(), ignore_unknown_fields=False)
    except Exception as exc:  # noqa: BLE001 - protobuf raises several parse exception types
        raise ValueError(f"OpenShell policy does not match the installed SDK schema: {policy_path}") from exc


def _policy_network_hosts(policy_data: dict[str, Any]) -> set[str]:
    """Return every hostname authorized by an OpenShell network policy."""
    policies = policy_data.get("network_policies") or {}
    if not isinstance(policies, dict):
        return set()
    hosts: set[str] = set()
    for policy in policies.values():
        if not isinstance(policy, dict):
            continue
        endpoints = policy.get("endpoints") or []
        if not isinstance(endpoints, list):
            continue
        for endpoint in endpoints:
            if isinstance(endpoint, dict) and isinstance(endpoint.get("host"), str):
                host = endpoint["host"].strip().lower().rstrip(".")
                if host:
                    hosts.add(host)
    return hosts


def _policy_network_endpoints(policy_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated endpoint mappings from every OpenShell network policy."""
    policies = policy_data.get("network_policies") or {}
    if not isinstance(policies, dict):
        return []
    endpoints: list[dict[str, Any]] = []
    for policy in policies.values():
        if not isinstance(policy, dict):
            continue
        entries = policy.get("endpoints") or []
        if isinstance(entries, list):
            endpoints.extend(endpoint for endpoint in entries if isinstance(endpoint, dict))
    return endpoints


def _validate_policy_network(policy_data: dict[str, Any], *, mode: str, allow: tuple[str, ...]) -> None:
    """Fail closed when the policy grants more egress than the public config declares."""
    endpoints = _policy_network_endpoints(policy_data)
    policy_hosts = _policy_network_hosts(policy_data)
    if mode == "blocked" and endpoints:
        raise ValueError("OpenShell policy grants network endpoints while sandbox.network is 'blocked'")
    if mode == "allowlist":
        for endpoint in endpoints:
            host = endpoint.get("host")
            if not isinstance(host, str) or not host.strip().rstrip("."):
                raise ValueError("OpenShell allowlist endpoints require a non-empty host")
            allowed_ips = endpoint.get("allowed_ips") or []
            if allowed_ips:
                raise ValueError("OpenShell allowed_ips/CIDR endpoints require an explicit public CIDR contract")
        configured_hosts = {host.strip().lower().rstrip(".") for host in allow}
        unexpected = policy_hosts - configured_hosts
        if unexpected:
            raise ValueError(f"OpenShell policy grants hosts outside sandbox.network.allow: {sorted(unexpected)}")


@dataclass(frozen=True)
class _AttestationResult:
    """Secret-free proof returned only after authoritative policy checks pass."""

    phase: object
    policy_version: int
    policy_hash: str | None
    policy_source: object | None
    assurance: str


def _build_sandbox_spec(
    *,
    policy: Any,
    image: str,
    job_id: str,
    labels: dict[str, str] | None = None,
) -> Any:
    """Build a secret-free per-job OpenShell spec using the installed SDK schema."""
    try:
        from openshell._proto import openshell_pb2
    except ImportError as exc:
        raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    template = openshell_pb2.SandboxTemplate(image=image, labels=labels or _sandbox_labels(job_id))
    # Deliberately omit environment and providers. Research/model credentials stay
    # on the host unless a future explicit credential-provider feature is configured.
    return openshell_pb2.SandboxSpec(template=template, policy=policy)


class OpenShellSandboxProvider(SandboxProvider):
    """OpenShell backend that creates and attests a policy-bound sandbox per job."""

    provider_name = "openshell"

    def __init__(self, config: SandboxConfig, job_id: str) -> None:
        """Initialize the provider, requiring the OpenShell SDK and adapter to import."""
        super().__init__(config, job_id)
        self._os_context: object | None = None
        self._os_context_entering = False
        self._os_context_exit_requested = False
        self._os_context_cleanup_complete: Event | None = None
        try:
            import langchain_nvidia_openshell  # noqa: F401
            import openshell  # noqa: F401
        except ImportError as exc:
            raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    @classmethod
    def _scoped_name(cls, job_id: str) -> str:
        """Return the OpenShell-safe sandbox name derived from the job id."""
        return _normalize_openshell_name(job_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the gateway-enforced guarantees this provider supports."""
        return SandboxCapabilities(
            supports_network_policy=True,
            supports_network_allowlist=True,
            supports_filesystem_policy=True,
            supports_process_policy=True,
            supports_artifact_download=True,
            supports_cleanup=True,
            supports_terminate=True,
        )

    def is_recoverable_error(self, exc: Exception) -> bool:
        """Return whether the error is a missing-sandbox condition worth one retry."""
        return _is_openshell_not_found_error(exc)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files. Uses the local env-free shim by default (OpenShell <=0.0.67 strips
        ``OPENSHELL_*`` env); set ``DEEP_RESEARCHER_OPENSHELL_ADAPTER_FILE_TRANSFER`` to delegate to the
        official adapter (validates the upstream argv fix)."""
        if _adapter_file_transfer_enabled():
            return self._call("upload_files", lambda session: session.upload_files(files), idempotent=True)
        return self._call("upload_files", lambda _s: self._upload_files_envfree(files), idempotent=True)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download artifacts through the bounded, job-confined local shim."""
        return self._call("download_files", lambda _s: self._download_files_envfree(paths), idempotent=True)

    def _upload_files_envfree(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files via argv + stdin so no ``OPENSHELL_*`` env is required."""
        sandbox = self._active_os_context()
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            result = sandbox.exec(  # type: ignore[union-attr]
                ["python3", "-c", _UPLOAD_CODE, path],
                stdin=base64.b64encode(content),
                timeout_seconds=self.config.timeout,
            )
            exit_code = getattr(result, "exit_code", 1)
            error = None if exit_code == 0 else _classify_fs_error(getattr(result, "stderr", "") or "")
            responses.append(FileUploadResponse(path=path, error=error))
        return responses

    def _download_files_envfree(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files via an argv bootstrap that enforces size/symlink limits in-sandbox."""
        sandbox = self._active_os_context()
        # Cap passed to the bootstrap so oversized files are refused before transfer.
        max_bytes = self.config.artifact_capture.max_file_bytes
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            result = sandbox.exec(  # type: ignore[union-attr]
                # Confine resolved paths to this job's artifact directory. The configured
                # workdir may be shared by several jobs in an attached named sandbox.
                ["python3", "-c", _DOWNLOAD_CODE, path, str(max_bytes), self.artifact_dir],
                timeout_seconds=self.config.timeout,
            )
            exit_code = getattr(result, "exit_code", 1)
            if exit_code != 0:
                error = _DOWNLOAD_EXIT_ERRORS.get(exit_code) or _classify_fs_error(getattr(result, "stderr", "") or "")
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
                continue
            # Validate base64 so stray stdout fails closed rather than storing corrupt bytes.
            try:
                content = base64.b64decode((getattr(result, "stdout", "") or "").strip().encode("ascii"), validate=True)
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_content"))
                continue
            responses.append(FileDownloadResponse(path=path, content=content, error=None))
        return responses

    def _create_session(self) -> BaseSandbox:
        """Create and attest a per-job OpenShell sandbox, or explicitly attach for debug."""
        try:
            import openshell
            from langchain_nvidia_openshell import OpenShellSandbox
        except ImportError as exc:
            raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

        cfg = self.config
        oscfg = cfg.providers.openshell

        if not oscfg.shell or any(not part.strip() for part in oscfg.shell):
            raise ValueError("OpenShell shell must contain at least one non-empty argv element")

        # Release any prior context (covers the recoverable-error reset path).
        self._exit_context()

        sandbox_kwargs: dict[str, object] = {
            "cluster": oscfg.gateway,
            "workspace": oscfg.workspace,
            "ready_timeout_seconds": oscfg.ready_timeout_seconds,
        }
        shared_name = oscfg.shared_sandbox_name
        expected_policy: Any | None = None
        if oscfg.policy:
            policy_data = _read_policy_data(oscfg.policy, require_hard_landlock=oscfg.require_hard_landlock)
            _validate_policy_network(
                policy_data,
                mode=cfg.network.mode,
                allow=cfg.network.allow,
            )
            expected_policy = _parse_policy_proto(policy_data, policy_path=oscfg.policy)

        if shared_name is not None:
            logger.warning(
                "OpenShell shared-sandbox debug attachment enabled: sandbox=%s job=%s; physical job isolation is off",
                shared_name,
                self.job_id,
            )
            # Attachment does not transfer ownership: never delete a shared sandbox
            # that this job did not create.
            sandbox_kwargs.update(sandbox=shared_name, delete_on_exit=False)
        else:
            if expected_policy is None:
                raise ValueError("Per-job OpenShell creation requires a policy file")
            labels = _sandbox_labels(self.job_id)
            if not _accepts_keyword(openshell.Sandbox, "labels"):
                self._fail_attestation(
                    phase=None,
                    policy_version=0,
                    assurance="strict",
                    reason_code="request_labels_unsupported",
                )
            sandbox_kwargs.update(
                spec=_build_sandbox_spec(
                    policy=expected_policy,
                    image=oscfg.image,
                    job_id=self.job_id,
                    labels=labels,
                ),
                labels=labels,
                delete_on_exit=oscfg.delete_on_exit,
            )

        os_sandbox = openshell.Sandbox(**sandbox_kwargs)
        self._enter_context(os_sandbox)
        backend: BaseSandbox | None = None
        try:
            self._ensure_context_active(os_sandbox)
            self.physical_sandbox_name = getattr(os_sandbox.sandbox, "name", None)
            attestation = self._attest(
                os_sandbox,
                expected_policy=expected_policy,
                require_sandbox_source=shared_name is None,
            )
            self._ensure_context_active(os_sandbox)
            backend = OpenShellSandbox(sandbox=os_sandbox, timeout=cfg.timeout, shell=oscfg.shell)
            self._ensure_context_active(os_sandbox)
            sandbox_ref = os_sandbox.sandbox
            logger.info(
                "OpenShell sandbox attested: id=%s name=%s policy_version=%s shared=%s",
                backend.id,
                getattr(sandbox_ref, "name", None),
                attestation.policy_version,
                shared_name is not None,
            )
            self._emit_attestation(attestation=attestation, status="succeeded")
            return backend
        except BaseException:
            self._safe_close(backend)
            self._exit_context()
            raise

    def _attest(
        self,
        os_sandbox: Any,
        *,
        expected_policy: Any | None,
        require_sandbox_source: bool,
    ) -> _AttestationResult:
        """Fail closed unless authoritative RPCs prove the effective policy."""
        try:
            from openshell._proto import openshell_pb2
        except ImportError as exc:
            raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

        sandbox_ref = os_sandbox.sandbox
        phase = getattr(sandbox_ref, "phase", None)
        policy_version = getattr(sandbox_ref, "current_policy_version", 0)
        if phase != openshell_pb2.SANDBOX_PHASE_READY:
            self._fail_attestation(
                phase=phase,
                policy_version=policy_version,
                assurance="strict" if expected_policy is not None else "reduced",
                reason_code="not_ready",
            )

        oscfg = self.config.providers.openshell
        assurance = "strict" if expected_policy is not None else "reduced"
        if not oscfg.attest:
            return _AttestationResult(
                phase=phase,
                policy_version=policy_version if isinstance(policy_version, int) else 0,
                policy_hash=None,
                policy_source=None,
                assurance=assurance,
            )
        return self._wait_for_authoritative_policy(
            os_sandbox,
            sandbox_ref,
            expected_policy=expected_policy,
            require_sandbox_source=require_sandbox_source,
            assurance=assurance,
        )

    def _wait_for_authoritative_policy(
        self,
        os_sandbox: Any,
        sandbox_ref: Any,
        *,
        expected_policy: Any | None,
        require_sandbox_source: bool,
        assurance: str,
    ) -> _AttestationResult:
        """Verify status, effective policy, provenance, version, and hash via authoritative RPCs."""
        from openshell._proto import openshell_pb2
        from openshell._proto import sandbox_pb2

        oscfg = self.config.providers.openshell
        phase = getattr(sandbox_ref, "phase", None)
        initial_policy_version = getattr(sandbox_ref, "current_policy_version", 0)
        client = getattr(os_sandbox, "_client", None)
        stub = getattr(client, "_stub", None)
        if (
            client is None
            or stub is None
            or not hasattr(stub, "GetSandboxPolicyStatus")
            or not hasattr(stub, "GetSandboxConfig")
        ):
            self._fail_attestation(
                phase=phase,
                policy_version=initial_policy_version,
                assurance=assurance,
                reason_code="rpc_unavailable",
            )

        status_request = openshell_pb2.GetSandboxPolicyStatusRequest(
            name=sandbox_ref.name,
            version=0,
            workspace=oscfg.workspace,
        )
        sandbox_id = getattr(sandbox_ref, "id", None) or sandbox_ref.name
        config_request = sandbox_pb2.GetSandboxConfigRequest(sandbox_id=sandbox_id)
        deadline = time.monotonic() + oscfg.policy_load_timeout_seconds
        pending_effective_policy = False
        last_revision_version = initial_policy_version
        while True:
            self._ensure_context_active(os_sandbox)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail_attestation(
                    phase=phase,
                    policy_version=last_revision_version,
                    assurance=assurance,
                    reason_code="policy_status_inconsistent" if pending_effective_policy else "attestation_timeout",
                )

            try:
                refreshed = client.get(sandbox_ref.name, workspace=oscfg.workspace)
                phase = getattr(refreshed, "phase", None)
                current_policy_version = getattr(refreshed, "current_policy_version", 0)
                status_response = stub.GetSandboxPolicyStatus(status_request, timeout=min(5.0, remaining))
                config_response = stub.GetSandboxConfig(config_request, timeout=min(5.0, remaining))
            except Exception:  # noqa: BLE001 - SDK errors must not escape with credential-bearing details
                self._fail_attestation(
                    phase=phase,
                    policy_version=initial_policy_version,
                    assurance=assurance,
                    reason_code="rpc_failed",
                )

            if phase != openshell_pb2.SANDBOX_PHASE_READY:
                self._fail_attestation(
                    phase=phase,
                    policy_version=current_policy_version,
                    assurance=assurance,
                    reason_code="not_ready",
                )

            revision = getattr(status_response, "revision", None)
            revision_status = getattr(revision, "status", openshell_pb2.POLICY_STATUS_UNSPECIFIED)
            revision_version = getattr(revision, "version", 0)
            last_revision_version = revision_version
            config_version = getattr(config_response, "version", 0)
            if getattr(revision, "load_error", ""):
                self._fail_attestation(
                    phase=phase,
                    policy_version=revision_version,
                    assurance=assurance,
                    reason_code="policy_load_error",
                )
            if revision_status == openshell_pb2.POLICY_STATUS_FAILED:
                self._fail_attestation(
                    phase=phase,
                    policy_version=revision_version,
                    assurance=assurance,
                    reason_code="policy_status_failed",
                )
            if revision_status != openshell_pb2.POLICY_STATUS_LOADED:
                config_policy = getattr(config_response, "policy", None)
                revision_policy = getattr(revision, "policy", None)
                policy_source = getattr(config_response, "policy_source", sandbox_pb2.POLICY_SOURCE_UNSPECIFIED)
                policy_hash = getattr(config_response, "policy_hash", "")
                revision_hash = getattr(revision, "policy_hash", "")
                pending_effective_policy = (
                    revision_status == openshell_pb2.POLICY_STATUS_PENDING
                    and revision_version > 0
                    and revision_version == config_version
                    and (
                        expected_policy is None
                        or (
                            config_policy == expected_policy
                            and revision_policy == expected_policy
                            and bool(policy_hash)
                            and policy_hash == revision_hash
                            and policy_source != sandbox_pb2.POLICY_SOURCE_UNSPECIFIED
                            and (not require_sandbox_source or policy_source == sandbox_pb2.POLICY_SOURCE_SANDBOX)
                        )
                    )
                )
                time.sleep(min(0.5, remaining))
                continue

            active_version = getattr(status_response, "active_version", 0)
            if not revision_version or revision_version != config_version:
                self._fail_attestation(
                    phase=phase,
                    policy_version=revision_version,
                    assurance=assurance,
                    reason_code="version_mismatch",
                )
            effective_version = revision_version
            for reported_version in (active_version, current_policy_version):
                if (
                    not isinstance(reported_version, int)
                    or reported_version <= 0
                    or reported_version != effective_version
                ):
                    self._fail_attestation(
                        phase=phase,
                        policy_version=revision_version,
                        assurance=assurance,
                        reason_code="version_mismatch",
                    )
            if oscfg.expected_policy_version is not None and effective_version != oscfg.expected_policy_version:
                self._fail_attestation(
                    phase=phase,
                    policy_version=effective_version,
                    assurance=assurance,
                    reason_code="expected_version_mismatch",
                )

            policy_source = getattr(config_response, "policy_source", sandbox_pb2.POLICY_SOURCE_UNSPECIFIED)
            policy_hash = getattr(config_response, "policy_hash", "")
            revision_hash = getattr(revision, "policy_hash", "")
            if expected_policy is not None:
                if require_sandbox_source and policy_source != sandbox_pb2.POLICY_SOURCE_SANDBOX:
                    self._fail_attestation(
                        phase=phase,
                        policy_version=effective_version,
                        assurance=assurance,
                        reason_code="policy_source_mismatch",
                    )
                if policy_source == sandbox_pb2.POLICY_SOURCE_UNSPECIFIED:
                    self._fail_attestation(
                        phase=phase,
                        policy_version=effective_version,
                        assurance=assurance,
                        reason_code="policy_source_mismatch",
                    )
                config_policy = getattr(config_response, "policy", None)
                revision_policy = getattr(revision, "policy", None)
                if config_policy != expected_policy or revision_policy != expected_policy:
                    self._fail_attestation(
                        phase=phase,
                        policy_version=effective_version,
                        assurance=assurance,
                        reason_code="policy_content_mismatch",
                    )
                if not policy_hash or not revision_hash:
                    self._fail_attestation(
                        phase=phase,
                        policy_version=effective_version,
                        assurance=assurance,
                        reason_code="policy_hash_missing",
                    )
                if policy_hash != revision_hash:
                    self._fail_attestation(
                        phase=phase,
                        policy_version=effective_version,
                        assurance=assurance,
                        reason_code="policy_hash_mismatch",
                    )

            return _AttestationResult(
                phase=phase,
                policy_version=effective_version,
                policy_hash=policy_hash or revision_hash or None,
                policy_source=policy_source,
                assurance=assurance,
            )

    def _fail_attestation(
        self,
        *,
        phase: object,
        policy_version: object,
        assurance: str,
        reason_code: str,
    ) -> None:
        """Emit one classified failure and raise without SDK or policy details."""
        self._emit_attestation(
            attestation=_AttestationResult(
                phase=phase,
                policy_version=policy_version if isinstance(policy_version, int) else 0,
                policy_hash=None,
                policy_source=None,
                assurance=assurance,
            ),
            status="failed",
            reason_code=reason_code,
        )
        raise RuntimeError(f"OpenShell sandbox attestation failed: {reason_code}")

    def _emit_attestation(
        self,
        *,
        attestation: _AttestationResult,
        status: str,
        reason_code: str | None = None,
    ) -> None:
        """Emit a secret-free OpenShell attestation outcome."""
        self._emit_event(
            {
                "type": "sandbox.attestation",
                "data": {
                    "provider": self.provider_name,
                    "sandbox": getattr(self, "physical_sandbox_name", None) or self.sandbox_name,
                    "phase": attestation.phase,
                    "policy_version": attestation.policy_version,
                    "policy_hash": attestation.policy_hash,
                    "policy_source": attestation.policy_source,
                    "assurance": attestation.assurance,
                    "reason_code": reason_code,
                    "status": status,
                },
            }
        )

    def close(self) -> None:
        """Terminate the session and exit the OpenShell context manager."""
        super().close()
        self._exit_context()

    def _terminate_session(self, session: BaseSandbox | None) -> None:
        """Close the adapter session and the owning OpenShell context on cancellation."""
        super()._terminate_session(session)
        self._exit_context()

    def _active_os_context(self) -> Any:
        """Return the current SDK context without racing an out-of-band teardown."""
        with self._state_lock:
            ctx = self._os_context
        if ctx is None:
            raise SandboxTerminatedError(f"OpenShell sandbox {self.sandbox_name} has no active context")
        return ctx

    def _enter_context(self, ctx: object) -> None:
        """Enter an SDK context while allowing terminate() to request deferred cleanup."""
        cleanup_complete = Event()
        with self._state_lock:
            if self._terminated:
                raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} has been terminated")
            if self._os_context is not None:
                raise RuntimeError(f"OpenShell sandbox {self.sandbox_name} context creation is already in progress")
            if self._os_context_cleanup_complete is not None and not self._os_context_cleanup_complete.is_set():
                raise RuntimeError(f"OpenShell sandbox {self.sandbox_name} context cleanup is already in progress")
            self._os_context = ctx
            self._os_context_entering = True
            self._os_context_exit_requested = False
            self._os_context_cleanup_complete = cleanup_complete

        try:
            ctx.__enter__()  # type: ignore[attr-defined]
        except BaseException:
            self._finish_context_entry(ctx, cleanup_complete, entered=False)
            raise

        if not self._finish_context_entry(ctx, cleanup_complete, entered=True):
            raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} was closed during creation")

    def _finish_context_entry(self, ctx: object, cleanup_complete: Event, *, entered: bool) -> bool:
        """Publish an entered context, or honor a pending teardown exactly once."""
        with self._state_lock:
            owns_context = self._os_context is ctx
            if owns_context:
                self._os_context_entering = False
                release_context = not entered or self._os_context_exit_requested or self._terminated
                if release_context:
                    self._os_context = None
                    self._os_context_exit_requested = False
            else:
                release_context = entered

        if release_context:
            self._close_os_context(ctx, cleanup_complete)
        return entered and owns_context and not release_context

    def _ensure_context_active(self, ctx: object) -> None:
        """Abort session publication when teardown won a creation race."""
        with self._state_lock:
            active = self._os_context is ctx and not self._os_context_exit_requested and not self._terminated
        if not active:
            raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} was closed during creation")

    def _exit_context(self) -> None:
        """Exit once and boundedly await creator-owned deletion during ``__enter__``."""
        close_context: object | None = None
        with self._state_lock:
            ctx = self._os_context
            cleanup_complete = self._os_context_cleanup_complete
            if ctx is None:
                wait_for_cleanup = cleanup_complete is not None and not cleanup_complete.is_set()
            elif self._os_context_entering:
                self._os_context_exit_requested = True
                if cleanup_complete is None:
                    cleanup_complete = Event()
                    self._os_context_cleanup_complete = cleanup_complete
                wait_for_cleanup = True
            else:
                if cleanup_complete is None:
                    cleanup_complete = Event()
                    self._os_context_cleanup_complete = cleanup_complete
                self._os_context = None
                self._os_context_exit_requested = False
                close_context = ctx
                wait_for_cleanup = False

        if close_context is not None:
            self._close_os_context(close_context, cleanup_complete)
            return
        if wait_for_cleanup and cleanup_complete is not None:
            if not cleanup_complete.wait(timeout=self.config.providers.openshell.cleanup_timeout_seconds):
                self._record_cleanup_failure("cleanup_timeout")
                logger.warning(
                    "OpenShell cleanup timed out: provider=%s sandbox=%s reason=cleanup_timeout",
                    self.provider_name,
                    self.sandbox_name,
                )

    def _close_os_context(self, ctx: object, cleanup_complete: Event) -> None:
        """Drive one detached SDK context exit without replacing the job result."""
        try:
            if hasattr(ctx, "__exit__"):
                ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise on the terminal path
            self._record_cleanup_failure("context_exit_failed")
            log_sandbox_failure(
                logger,
                operation="context_exit",
                reason_code="context_exit_failed",
                exc=exc,
                provider=self.provider_name,
                sandbox=self.sandbox_name,
            )
        finally:
            cleanup_complete.set()
            with self._state_lock:
                if self._os_context_cleanup_complete is cleanup_complete:
                    self._os_context_cleanup_complete = None


register_sandbox_provider("openshell", OpenShellSandboxProvider)
