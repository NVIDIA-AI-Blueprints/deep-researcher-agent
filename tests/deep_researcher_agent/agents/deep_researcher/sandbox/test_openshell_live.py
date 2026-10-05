# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live acceptance tests for OpenShell isolation and lifecycle behavior."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import logging
import os
import platform
import shlex
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
_LIVE_ENABLED = os.getenv("DEEP_RESEARCHER_OPENSHELL_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _LIVE_ENABLED,
        reason="Set DEEP_RESEARCHER_OPENSHELL_LIVE_TESTS=1 to run live OpenShell integration tests.",
    ),
]


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LiveConfig:
    """Non-secret live-test settings."""

    gateway: str | None
    workspace: str
    policy_path: Path
    image: str
    expected_gateway_version: str | None
    allow_best_effort: bool


@dataclass(frozen=True)
class LiveRuntime:
    """Lazy imports and validated policy state used by opted-in tests."""

    config: LiveConfig
    sandbox_client: Any
    grpc: Any
    openshell_pb2: Any
    sandbox_pb2: Any
    expected_policy: Any
    policy_data: dict[str, Any]
    network: dict[str, object]
    build_sandbox_spec: Callable[..., Any]

    def client(self) -> Any:
        return self.sandbox_client.from_active_cluster(cluster=self.config.gateway)


@pytest.fixture(scope="session")
def live_config() -> LiveConfig:
    policy_value = os.getenv(
        "DEEP_RESEARCHER_OPENSHELL_POLICY_FILE",
        "configs/openshell/generated/deep-researcher-openshell-policy.yaml",
    )
    policy_path = Path(policy_value).expanduser()
    if not policy_path.is_absolute():
        policy_path = _REPO_ROOT / policy_path
    return LiveConfig(
        gateway=os.getenv("DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME") or None,
        workspace=os.getenv("DEEP_RESEARCHER_OPENSHELL_WORKSPACE") or "default",
        policy_path=policy_path.resolve(),
        image=os.getenv("DEEP_RESEARCHER_OPENSHELL_IMAGE", "deep-researcher-openshell-demo:latest"),
        expected_gateway_version=os.getenv("DEEP_RESEARCHER_OPENSHELL_EXPECTED_GATEWAY_VERSION") or None,
        allow_best_effort=_env_enabled("DEEP_RESEARCHER_OPENSHELL_LIVE_ALLOW_BEST_EFFORT"),
    )


@pytest.fixture(scope="session")
def live_runtime(live_config: LiveConfig) -> LiveRuntime:
    """Import optional SDK modules only after pytest has evaluated the live opt-in."""
    if not live_config.allow_best_effort and platform.system() != "Linux":
        pytest.fail(
            "Production OpenShell acceptance requires Linux; use the explicit best-effort demo opt-in elsewhere."
        )

    try:
        import grpc
        import langchain_nvidia_openshell  # noqa: F401
        import openshell  # noqa: F401
        from openshell._proto import openshell_pb2
        from openshell._proto import sandbox_pb2
        from openshell.sandbox import SandboxClient
    except ImportError:
        pytest.fail("Opted-in OpenShell live tests require the SDK and DeepAgents adapter.")

    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _build_sandbox_spec
    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _parse_policy_proto
    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _policy_network_hosts
    from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _read_policy_data

    policy_data = _read_policy_data(
        str(live_config.policy_path),
        require_hard_landlock=not live_config.allow_best_effort,
    )
    expected_policy = _parse_policy_proto(policy_data, policy_path=str(live_config.policy_path))
    hosts = tuple(sorted(_policy_network_hosts(policy_data)))
    network: dict[str, object] = {"mode": "allowlist", "allow": hosts} if hosts else {"mode": "blocked"}

    return LiveRuntime(
        config=live_config,
        sandbox_client=SandboxClient,
        grpc=grpc,
        openshell_pb2=openshell_pb2,
        sandbox_pb2=sandbox_pb2,
        expected_policy=expected_policy,
        policy_data=policy_data,
        network=network,
        build_sandbox_spec=_build_sandbox_spec,
    )


def _is_not_found(runtime: LiveRuntime, exc: BaseException) -> bool:
    return isinstance(exc, runtime.grpc.Call) and exc.code() == runtime.grpc.StatusCode.NOT_FOUND


def _assert_deleted(runtime: LiveRuntime, client: Any, name: str) -> None:
    try:
        client.get(name, workspace=runtime.config.workspace)
    except runtime.grpc.RpcError as exc:
        if _is_not_found(runtime, exc):
            return
        raise
    raise AssertionError(f"OpenShell cleanup was not verified for sandbox {name}")


@dataclass
class _TrackedResource:
    kind: str
    value: Any
    owns_remote: bool = True


class ResourceTracker:
    """Own every test-created resource and verify teardown through the gateway."""

    def __init__(self, runtime: LiveRuntime) -> None:
        self.runtime = runtime
        self.resources: list[_TrackedResource] = []

    def provider(self, provider: Any, *, owns_remote: bool) -> Any:
        self.resources.append(_TrackedResource("provider", provider, owns_remote))
        return provider

    def direct_sandbox(self, name: str) -> str:
        self.resources.append(_TrackedResource("direct", name))
        return name

    def cleanup(self) -> None:
        failures: list[str] = []
        for resource in reversed(self.resources):
            if resource.kind == "provider":
                provider = resource.value
                try:
                    provider.terminate()
                except Exception as exc:  # noqa: BLE001 - teardown must continue across every owned resource
                    failures.append(f"provider_terminate:{type(exc).__name__}")
                name = getattr(provider, "physical_sandbox_name", None)
                if resource.owns_remote and name:
                    try:
                        with self.runtime.client() as client:
                            _assert_deleted(self.runtime, client, name)
                    except Exception as exc:  # noqa: BLE001 - report sanitized teardown metadata after all attempts
                        failures.append(f"provider_delete_verify:{type(exc).__name__}")
                continue

            name = resource.value
            try:
                with self.runtime.client() as client:
                    try:
                        client.get(name, workspace=self.runtime.config.workspace)
                    except self.runtime.grpc.RpcError as exc:
                        if not _is_not_found(self.runtime, exc):
                            raise
                    else:
                        if not client.delete(name, workspace=self.runtime.config.workspace):
                            raise RuntimeError("delete_not_acknowledged")
                        client.wait_deleted(name, workspace=self.runtime.config.workspace)
                    _assert_deleted(self.runtime, client, name)
            except Exception as exc:  # noqa: BLE001 - report sanitized teardown metadata after all attempts
                failures.append(f"direct_delete_verify:{type(exc).__name__}")

        if failures:
            pytest.fail("OpenShell fixture teardown failed: " + ", ".join(failures))


@pytest.fixture
def resources(live_runtime: LiveRuntime) -> ResourceTracker:
    tracker = ResourceTracker(live_runtime)
    yield tracker
    tracker.cleanup()


@pytest.fixture(scope="session", autouse=True)
def require_gateway_version(live_runtime: LiveRuntime) -> str:
    with live_runtime.client() as client:
        version = client.health().version
    expected = live_runtime.config.expected_gateway_version or importlib.metadata.version("openshell")
    assert version == expected, "The OpenShell gateway and Python SDK versions must match exactly."
    return version


@pytest.fixture
def provider_factory(
    live_runtime: LiveRuntime,
    resources: ResourceTracker,
) -> Callable[..., Any]:
    from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxConfig
    from deep_researcher_agent.agents.deep_researcher.sandbox import create_sandbox_backend

    def create(
        job_id: str,
        emitter: Callable[[dict[str, object]], None],
        *,
        policy_path: Path | None = None,
        shared_name: str | None = None,
    ) -> Any:
        openshell_config: dict[str, object] = {
            "gateway": live_runtime.config.gateway,
            "workspace": live_runtime.config.workspace,
            "policy": str(policy_path or live_runtime.config.policy_path),
            "image": live_runtime.config.image,
            "require_hard_landlock": not live_runtime.config.allow_best_effort,
        }
        if shared_name is not None:
            openshell_config.update(existing_sandbox_name=shared_name, allow_shared_sandbox=True)
        config = SandboxConfig(
            provider="openshell",
            workdir="/sandbox",
            network=live_runtime.network,
            providers={"openshell": openshell_config},
        )
        provider = create_sandbox_backend(config, job_id)
        resources.provider(provider, owns_remote=shared_name is None)
        provider.set_event_emitter(emitter)
        return provider

    return create


@pytest.fixture
def direct_sandbox_factory(
    live_runtime: LiveRuntime,
    resources: ResourceTracker,
) -> Callable[[str], str]:
    """Create a directly owned sandbox and register it before readiness polling."""

    def create(job_id: str) -> str:
        with live_runtime.client() as client:
            labels = {"deep-researcher": "shared-debug-test"}
            sandbox_ref = client.create(
                workspace=live_runtime.config.workspace,
                spec=live_runtime.build_sandbox_spec(
                    policy=live_runtime.expected_policy,
                    image=live_runtime.config.image,
                    job_id=job_id,
                    labels=labels,
                ),
                labels=labels,
            )
            name = resources.direct_sandbox(sandbox_ref.name)
            client.wait_ready(name, workspace=live_runtime.config.workspace)
        return name

    return create


def _attestation_success(events: list[dict[str, object]], runtime: LiveRuntime, authoritative_hash: str) -> bool:
    for event in events:
        if event.get("type") != "sandbox.attestation":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("status") != "succeeded":
            continue
        return (
            isinstance(data.get("policy_version"), int)
            and data["policy_version"] > 0
            and data.get("policy_hash") == authoritative_hash
            and data.get("policy_source") == runtime.sandbox_pb2.POLICY_SOURCE_SANDBOX
            and data.get("assurance") == "strict"
            and data.get("reason_code") is None
        )
    return False


def _assert_authoritative_policy(runtime: LiveRuntime, client: Any, name: str) -> tuple[Any, int, str]:
    sandbox = client.get(name, workspace=runtime.config.workspace)
    stub = client._stub
    status = stub.GetSandboxPolicyStatus(
        runtime.openshell_pb2.GetSandboxPolicyStatusRequest(
            name=name,
            version=0,
            workspace=runtime.config.workspace,
        ),
        timeout=30,
    )
    config = stub.GetSandboxConfig(
        runtime.sandbox_pb2.GetSandboxConfigRequest(sandbox_id=sandbox.id or name),
        timeout=30,
    )
    revision = status.revision
    assert revision.status == runtime.openshell_pb2.POLICY_STATUS_LOADED
    assert not revision.load_error
    assert revision.version > 0
    assert config.version == revision.version
    assert status.active_version == revision.version
    assert sandbox.current_policy_version == revision.version
    assert config.policy_source == runtime.sandbox_pb2.POLICY_SOURCE_SANDBOX
    assert config.policy == runtime.expected_policy
    assert revision.policy == runtime.expected_policy
    assert config.policy_hash
    assert revision.policy_hash == config.policy_hash
    return sandbox, revision.version, config.policy_hash


def test_live_per_job_isolation_attestation_and_cancellation(
    live_runtime: LiveRuntime,
    provider_factory: Callable[..., Any],
) -> None:
    suffix = uuid4().hex[:10]
    events: tuple[list[dict[str, object]], list[dict[str, object]]] = ([], [])
    providers = (
        provider_factory(f"deep-researcher-isolation-a-{suffix}", events[0].append),
        provider_factory(f"deep-researcher-isolation-b-{suffix}", events[1].append),
    )
    markers = (f"owner-a-{suffix}", f"owner-b-{suffix}")

    def initialize(index: int) -> str:
        provider = providers[index]
        marker_file = f"{provider.workdir}/owner.txt"
        command = (
            f"printf %s {shlex.quote(markers[index])} > {shlex.quote(marker_file)}; cat {shlex.quote(marker_file)}"
        )
        result = provider.execute(command, timeout=30)
        assert result.exit_code == 0
        assert result.output.strip() == markers[index]
        return provider.physical_sandbox_name

    with ThreadPoolExecutor(max_workers=2) as pool:
        names = tuple(pool.map(initialize, range(2)))
    assert names[0] != names[1]

    with live_runtime.client() as client:
        selected = client.list(workspace=live_runtime.config.workspace, label_selector="deep-researcher=deep-research")
        selected_by_name = {item.name: item for item in selected}
        assert set(names) <= selected_by_name.keys()
        for index, name in enumerate(names):
            assert dict(selected_by_name[name].labels) == {
                "deep-researcher": "deep-research",
                "deep-researcher-job-id": f"deep-researcher-isolation-{'a' if index == 0 else 'b'}-{suffix}",
            }
        first, first_revision, first_hash = _assert_authoritative_policy(live_runtime, client, names[0])
        second, second_revision, second_hash = _assert_authoritative_policy(live_runtime, client, names[1])
        assert first.id != second.id
        assert first_revision > 0 and second_revision > 0
        assert _attestation_success(events[0], live_runtime, first_hash)
        assert _attestation_success(events[1], live_runtime, second_hash)

        providers[0].terminate()
        _assert_deleted(live_runtime, client, names[0])
        selected_names = {
            item.name
            for item in client.list(
                workspace=live_runtime.config.workspace, label_selector="deep-researcher=deep-research"
            )
        }
        assert names[0] not in selected_names
        assert names[1] in selected_names
        assert client.get(names[1], workspace=live_runtime.config.workspace).id == second.id
        result = providers[1].execute("printf %s still-alive", timeout=30)
        assert result.exit_code == 0
        assert result.output.strip() == "still-alive"

        providers[1].close()
        _assert_deleted(live_runtime, client, names[1])
        assert names[1] not in {
            item.name
            for item in client.list(
                workspace=live_runtime.config.workspace, label_selector="deep-researcher=deep-research"
            )
        }


def test_live_failure_cleanup_and_log_redaction(
    live_runtime: LiveRuntime,
    provider_factory: Callable[..., Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    suffix = uuid4().hex[:10]
    secret_canary = f"credential=deep-researcher-live-{suffix}"
    events: list[dict[str, object]] = []

    def fail_event_delivery(event: dict[str, object]) -> None:
        events.append(event)
        raise RuntimeError(secret_canary)

    caplog.set_level(logging.DEBUG)
    provider = provider_factory(f"deep-researcher-isolation-failure-{suffix}", fail_event_delivery)
    result = provider.execute("exit 23", timeout=30)
    name = provider.physical_sandbox_name
    assert result.exit_code == 23
    provider.close()

    with live_runtime.client() as client:
        _assert_deleted(live_runtime, client, name)
    assert secret_canary not in caplog.text
    assert secret_canary not in json.dumps(events, default=str)


def test_live_shared_policy_mismatch_is_rejected(
    live_runtime: LiveRuntime,
    provider_factory: Callable[..., Any],
    direct_sandbox_factory: Callable[[str], str],
    tmp_path: Path,
) -> None:
    import yaml

    suffix = uuid4().hex[:10]
    mismatch_data = copy.deepcopy(live_runtime.policy_data)
    filesystem = mismatch_data.get("filesystem")
    assert isinstance(filesystem, dict)
    read_only = list(filesystem.get("read_only") or [])
    read_only.append("/__deep_researcher_attestation_mismatch__")
    filesystem["read_only"] = read_only
    mismatch_path = tmp_path / "mismatched-policy.yaml"
    mismatch_path.write_text(yaml.safe_dump(mismatch_data, sort_keys=False), encoding="utf-8")

    shared_name = direct_sandbox_factory(f"deep-researcher-shared-mismatch-{suffix}")

    mismatch_events: list[dict[str, object]] = []
    provider = provider_factory(
        f"deep-researcher-mismatched-attach-{suffix}",
        mismatch_events.append,
        policy_path=mismatch_path,
        shared_name=shared_name,
    )
    with pytest.raises(RuntimeError, match="policy_content_mismatch"):
        provider.execute("true", timeout=30)

    assert not any(
        event.get("type") == "sandbox.attestation"
        and isinstance(event.get("data"), dict)
        and event["data"].get("status") == "succeeded"
        for event in mismatch_events
    )
    import openshell

    with live_runtime.client() as client:
        shared = client.get(shared_name, workspace=live_runtime.config.workspace)
        assert shared.name == shared_name
    with openshell.Sandbox(
        workspace=live_runtime.config.workspace,
        cluster=live_runtime.config.gateway,
        sandbox=shared_name,
        delete_on_exit=False,
    ) as shared_sandbox:
        result = shared_sandbox.exec(["bash", "-c", "printf %s still-usable"], timeout_seconds=30)
        assert result.exit_code == 0
        assert result.stdout.strip() == "still-usable"
