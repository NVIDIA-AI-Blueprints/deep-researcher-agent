# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove OpenShell version, policy, selector, execution, and cleanup capabilities."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_VERSION_PATTERN = re.compile(
    r"(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:(?:-dev\.|\.dev)(?P<dev>\d+)(?:\+g(?P<sha>[0-9a-fA-F]+))?)?"
)


class ReadinessError(RuntimeError):
    """A sanitized readiness failure safe to print for operators."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ReadinessConfig:
    """Non-secret inputs for one disposable readiness probe."""

    gateway: str | None
    workspace: str
    image: str
    policy: Path
    openshell_bin: Path
    ready_timeout_seconds: float
    policy_load_timeout_seconds: float


def _version_from_cli(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReadinessError("version_check_failed") from exc
    match = _VERSION_PATTERN.search(result.stdout)
    if result.returncode != 0 or match is None:
        raise ReadinessError("version_check_failed")
    return match.group(0)


def _version_identity(value: str) -> tuple[int, int, int, int | None, str | None] | None:
    """Normalize equivalent Python and Cargo development-version spellings."""
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        return None
    dev = match.group("dev")
    sha = match.group("sha")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        int(dev) if dev is not None else None,
        sha.lower() if sha is not None else None,
    )


def _is_not_found(exc: BaseException, grpc: Any) -> bool:
    return isinstance(exc, grpc.Call) and exc.code() == grpc.StatusCode.NOT_FOUND


def _verify_absent(client: Any, name: str, selector: str, grpc: Any, *, workspace: str) -> None:
    try:
        client.get(name, workspace=workspace)
    except grpc.RpcError as exc:
        if not _is_not_found(exc, grpc):
            raise ReadinessError("cleanup_failed") from exc
    else:
        raise ReadinessError("cleanup_failed")
    try:
        selected = client.list(workspace=workspace, label_selector=selector)
    except Exception as exc:  # noqa: BLE001 - never expose SDK response details
        raise ReadinessError("cleanup_failed") from exc
    if any(getattr(item, "name", None) == name for item in selected):
        raise ReadinessError("cleanup_failed")


def _verify_policy(
    *,
    client: Any,
    sandbox: Any,
    workspace: str,
    expected_policy: Any,
    timeout_seconds: float,
    openshell_pb2: Any,
    sandbox_pb2: Any,
) -> None:
    stub = getattr(client, "_stub", None)
    if stub is None or not hasattr(stub, "GetSandboxPolicyStatus") or not hasattr(stub, "GetSandboxConfig"):
        raise ReadinessError("sdk_capability_missing")
    status_request = openshell_pb2.GetSandboxPolicyStatusRequest(
        name=sandbox.name,
        version=0,
        workspace=workspace,
    )
    config_request = sandbox_pb2.GetSandboxConfigRequest(sandbox_id=sandbox.id or sandbox.name)
    deadline = time.monotonic() + timeout_seconds
    pending_effective = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReadinessError("policy_status_inconsistent" if pending_effective else "attestation_timeout")
        try:
            refreshed = client.get(sandbox.name, workspace=workspace)
            status = stub.GetSandboxPolicyStatus(status_request, timeout=min(5.0, remaining))
            config = stub.GetSandboxConfig(config_request, timeout=min(5.0, remaining))
        except Exception as exc:  # noqa: BLE001 - never expose SDK response details
            raise ReadinessError("rpc_failed") from exc
        if getattr(refreshed, "phase", None) != openshell_pb2.SANDBOX_PHASE_READY:
            raise ReadinessError("not_ready")
        revision = getattr(status, "revision", None)
        revision_status = getattr(revision, "status", openshell_pb2.POLICY_STATUS_UNSPECIFIED)
        revision_version = getattr(revision, "version", 0)
        config_version = getattr(config, "version", 0)
        if getattr(revision, "load_error", "") or revision_status == openshell_pb2.POLICY_STATUS_FAILED:
            raise ReadinessError("policy_status_failed")
        if revision_status != openshell_pb2.POLICY_STATUS_LOADED:
            pending_effective = (
                revision_status == openshell_pb2.POLICY_STATUS_PENDING
                and revision_version > 0
                and revision_version == config_version
                and getattr(config, "policy_source", sandbox_pb2.POLICY_SOURCE_UNSPECIFIED)
                == sandbox_pb2.POLICY_SOURCE_SANDBOX
                and getattr(config, "policy", None) == expected_policy
                and getattr(revision, "policy", None) == expected_policy
                and bool(getattr(config, "policy_hash", ""))
                and getattr(config, "policy_hash", "") == getattr(revision, "policy_hash", "")
            )
            time.sleep(min(0.5, remaining))
            continue
        if not revision_version or revision_version != config_version:
            raise ReadinessError("version_mismatch")
        for reported in (
            getattr(status, "active_version", 0),
            getattr(refreshed, "current_policy_version", 0),
        ):
            if not isinstance(reported, int) or reported <= 0 or reported != revision_version:
                raise ReadinessError("version_mismatch")
        if getattr(config, "policy_source", sandbox_pb2.POLICY_SOURCE_UNSPECIFIED) != sandbox_pb2.POLICY_SOURCE_SANDBOX:
            raise ReadinessError("policy_source_mismatch")
        if getattr(config, "policy", None) != expected_policy or getattr(revision, "policy", None) != expected_policy:
            raise ReadinessError("policy_content_mismatch")
        config_hash = getattr(config, "policy_hash", "")
        revision_hash = getattr(revision, "policy_hash", "")
        if not config_hash or not revision_hash:
            raise ReadinessError("policy_hash_missing")
        if config_hash != revision_hash:
            raise ReadinessError("policy_hash_mismatch")
        return


def run_check(config: ReadinessConfig) -> tuple[str, str]:
    """Run one strict probe and return the validated SDK and gateway version."""
    try:
        import grpc
        from openshell._proto import openshell_pb2
        from openshell._proto import sandbox_pb2
        from openshell.sandbox import SandboxClient
    except ImportError as exc:
        raise ReadinessError("sdk_unavailable") from exc

    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _accepts_keyword
    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _build_sandbox_spec
    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _parse_policy_proto
    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _read_policy_data

    try:
        sdk_version = importlib.metadata.version("openshell")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ReadinessError("sdk_unavailable") from exc
    cli_version = _version_from_cli(config.openshell_bin)
    labels = {"deep-researcher": "readiness-probe"}
    selector = "deep-researcher=readiness-probe"
    policy_data = _read_policy_data(str(config.policy), require_hard_landlock=False)
    expected_policy = _parse_policy_proto(policy_data, policy_path=str(config.policy))
    spec = _build_sandbox_spec(
        policy=expected_policy,
        image=config.image,
        job_id=f"readiness-{uuid4().hex[:10]}",
        labels=labels,
    )

    with SandboxClient.from_active_cluster(cluster=config.gateway) as client:
        try:
            gateway_version = client.health().version
        except Exception as exc:  # noqa: BLE001 - never expose SDK response details
            raise ReadinessError("gateway_unavailable") from exc
        version_identities = {
            _version_identity(cli_version),
            _version_identity(sdk_version),
            _version_identity(gateway_version),
        }
        if None in version_identities or len(version_identities) != 1:
            raise ReadinessError("version_mismatch")
        workspace_methods = ("create", "list", "wait_ready", "get", "delete", "wait_deleted")
        if (
            any(
                not _accepts_keyword(getattr(client, method_name, None), "workspace")
                for method_name in workspace_methods
            )
            or not _accepts_keyword(client.create, "name")
            or not _accepts_keyword(client.create, "labels")
            or not _accepts_keyword(client.list, "label_selector")
        ):
            raise ReadinessError("request_labels_unsupported")

        sandbox_name = f"drr-{uuid4().hex[:12]}"
        cleanup_name = sandbox_name
        primary_error: ReadinessError | None = None
        try:
            sandbox = client.create(workspace=config.workspace, spec=spec, name=sandbox_name, labels=labels)
            cleanup_name = getattr(sandbox, "name", None) or sandbox_name
            if sandbox.name != sandbox_name:
                raise ReadinessError("probe_failed")
            sandbox = client.wait_ready(
                sandbox_name, workspace=config.workspace, timeout_seconds=config.ready_timeout_seconds
            )
            selected = client.list(workspace=config.workspace, label_selector=selector)
            matches = [item for item in selected if getattr(item, "name", None) == sandbox_name]
            if len(selected) != 1 or len(matches) != 1 or dict(getattr(matches[0], "labels", {})) != labels:
                raise ReadinessError("selector_mismatch")
            _verify_policy(
                client=client,
                sandbox=sandbox,
                workspace=config.workspace,
                expected_policy=expected_policy,
                timeout_seconds=config.policy_load_timeout_seconds,
                openshell_pb2=openshell_pb2,
                sandbox_pb2=sandbox_pb2,
            )
            result = client.exec(sandbox.id or sandbox.name, ["sh", "-c", "printf %s deep-researcher-openshell-ready"])
            if (
                getattr(result, "exit_code", 1) != 0
                or getattr(result, "stdout", "") != "deep-researcher-openshell-ready"
            ):
                raise ReadinessError("execution_failed")
        except ReadinessError as exc:
            primary_error = exc
        except Exception as exc:  # noqa: BLE001 - never expose SDK response details
            primary_error = ReadinessError("probe_failed")
            primary_error.__cause__ = exc
        finally:
            try:
                try:
                    client.get(cleanup_name, workspace=config.workspace)
                except grpc.RpcError as exc:
                    if not _is_not_found(exc, grpc):
                        raise
                else:
                    if not client.delete(cleanup_name, workspace=config.workspace):
                        raise ReadinessError("cleanup_failed")
                    client.wait_deleted(cleanup_name, workspace=config.workspace)
                _verify_absent(client, cleanup_name, selector, grpc, workspace=config.workspace)
            except Exception as exc:  # noqa: BLE001 - cleanup failure overrides probe failure
                raise ReadinessError("cleanup_failed") from exc
        if primary_error is not None:
            raise primary_error
    return sdk_version, gateway_version


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-name", default=None)
    parser.add_argument("--workspace", default=os.getenv("DEEP_RESEARCHER_OPENSHELL_WORKSPACE") or "default")
    parser.add_argument("--image-name", default="deep-researcher-openshell-demo:latest")
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--openshell-bin", type=Path, required=True)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--policy-load-timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ReadinessConfig(
        gateway=args.gateway_name,
        workspace=args.workspace,
        image=args.image_name,
        policy=args.policy_file,
        openshell_bin=args.openshell_bin,
        ready_timeout_seconds=args.ready_timeout_seconds,
        policy_load_timeout_seconds=args.policy_load_timeout_seconds,
    )
    try:
        sdk_version, gateway_version = run_check(config)
    except ReadinessError as exc:
        print(f"OpenShell readiness check failed: {exc.reason_code}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - unexpected details may contain gateway or SDK data
        print("OpenShell readiness check failed: unexpected_error", file=sys.stderr)
        return 1
    print(f"OpenShell strict readiness succeeded: sdk={sdk_version} gateway={gateway_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
