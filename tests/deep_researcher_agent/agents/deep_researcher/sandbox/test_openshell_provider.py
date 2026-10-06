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

"""OpenShell provider tests: creation/attestation plus confined file transfer."""

from __future__ import annotations

import base64
import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from threading import Thread
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from deep_researcher_agent.agents.deep_researcher.sandbox.base import SandboxProvider
from deep_researcher_agent.agents.deep_researcher.sandbox.base import SandboxTerminatedError
from deep_researcher_agent.agents.deep_researcher.sandbox.config import SandboxConfig
from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import OpenShellSandboxProvider
from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _build_sandbox_spec
from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _parse_policy_proto
from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _read_policy_data
from deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell import _validate_policy_network


@dataclass
class _ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class _FakeProtoPart:
    """Small deterministic protobuf stand-in used without the optional SDK."""

    payload: bytes

    def SerializeToString(self, *, deterministic: bool = False) -> bytes:  # noqa: N802 - protobuf API
        del deterministic
        return self.payload


class _FakePolicy:
    """Structural stand-in for the OpenShell SandboxPolicy protobuf."""

    __hash__ = None

    def __init__(self, *, version: int = 1, network_policies: dict[str, _FakeProtoPart] | None = None) -> None:
        self.version = version
        self.filesystem = _FakeProtoPart(b"filesystem")
        self.landlock = _FakeProtoPart(b"landlock")
        self.process = _FakeProtoPart(b"process")
        self.network_policies = network_policies or {}

    def HasField(self, field: str) -> bool:  # noqa: N802 - protobuf API
        return getattr(self, field, None) is not None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakePolicy) and vars(self) == vars(other)


_FAKE_POLICY = _FakePolicy()
_FAKE_POLICY_HASH = "authoritative-openshell-hash"


class _FakeOpenShellSandbox:
    """Records exec calls and returns scripted results."""

    id = "fake-os-id"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = _ExecResult(exit_code=0)
        self.exit_calls = 0

    def exec(self, command, **kwargs):  # noqa: ANN001 - mirrors openshell.Sandbox.exec
        self.calls.append({"command": list(command), **kwargs})
        return self.result

    def __exit__(self, *_args: object) -> None:
        self.exit_calls += 1


def _provider(**openshell_config: Any) -> OpenShellSandboxProvider:
    cfg = SandboxConfig(
        provider="openshell",
        network={"mode": "blocked"},
        providers={"openshell": openshell_config},
    )
    # Construct through the provider-neutral base so these unit tests run even when
    # the optional OpenShell SDK/adapter are absent from normal CI.
    provider = object.__new__(OpenShellSandboxProvider)
    SandboxProvider.__init__(provider, cfg, "job-1")
    provider._os_context = None
    provider._os_context_entering = False
    provider._os_context_exit_requested = False
    provider._os_context_cleanup_complete = None
    # Avoid real session creation: a non-None _session short-circuits _session_or_create.
    provider._session = MagicMock()  # type: ignore[assignment]
    return provider


class _FakeCreatedContext(_FakeOpenShellSandbox):
    """Entered SDK context with a public sandbox reference for attestation."""

    def __init__(
        self,
        *,
        phase: int = 2,
        policy_version: int = 1,
        name: str = "generated",
        policy: _FakePolicy = _FAKE_POLICY,
        policy_source: int = 1,
    ) -> None:
        super().__init__()
        self.enter_calls = 0
        self.sandbox = SimpleNamespace(
            id=f"{name}-id",
            phase=phase,
            current_policy_version=policy_version,
            name=name,
            workspace="default",
        )
        revision_version = policy_version if policy_version > 0 else 1
        revision = SimpleNamespace(
            version=revision_version,
            status=2,
            load_error="",
            policy_hash=_FAKE_POLICY_HASH,
            policy=policy,
        )
        status = SimpleNamespace(revision=revision, active_version=revision_version)
        config = SimpleNamespace(
            version=revision_version,
            policy_hash=_FAKE_POLICY_HASH,
            policy=policy,
            policy_source=policy_source,
        )
        self._client = SimpleNamespace(
            get=MagicMock(return_value=self.sandbox),
            health=MagicMock(return_value=SimpleNamespace(version="0.0.88")),
            _stub=SimpleNamespace(
                GetSandboxPolicyStatus=MagicMock(return_value=status),
                GetSandboxConfig=MagicMock(return_value=config),
            ),
        )

    def __enter__(self):
        self.enter_calls += 1
        return self


class _FakeAdapter:
    """Minimal langchain-nvidia-openshell adapter used by provider lifecycle tests."""

    def __init__(self, *, sandbox: _FakeCreatedContext, timeout: int, shell: tuple[str, ...]) -> None:
        self.id = sandbox.id
        self.sandbox = sandbox
        self.timeout = timeout
        self.shell = shell


@contextmanager
def _fake_optional_modules(context_factory, adapter_cls: type = _FakeAdapter):
    openshell_module = ModuleType("openshell")
    openshell_module.Sandbox = context_factory  # type: ignore[attr-defined]
    adapter_module = ModuleType("langchain_nvidia_openshell")
    adapter_module.OpenShellSandbox = adapter_cls  # type: ignore[attr-defined]
    proto_module = ModuleType("openshell._proto")
    proto_module.openshell_pb2 = SimpleNamespace(  # type: ignore[attr-defined]
        SANDBOX_PHASE_READY=2,
        SANDBOX_PHASE_ERROR=3,
        POLICY_STATUS_UNSPECIFIED=0,
        POLICY_STATUS_PENDING=1,
        POLICY_STATUS_LOADED=2,
        POLICY_STATUS_FAILED=3,
        GetSandboxPolicyStatusRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    proto_module.sandbox_pb2 = SimpleNamespace(  # type: ignore[attr-defined]
        POLICY_SOURCE_UNSPECIFIED=0,
        POLICY_SOURCE_SANDBOX=1,
        POLICY_SOURCE_GLOBAL=2,
        GetSandboxConfigRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    with patch.dict(
        sys.modules,
        {
            "openshell": openshell_module,
            "openshell._proto": proto_module,
            "langchain_nvidia_openshell": adapter_module,
        },
    ):
        yield


def _write_policy(path: Path, *, compatibility: str = "hard_requirement", extra: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "filesystem_policy:",
                "  include_workdir: true",
                "  read_write: [/sandbox]",
                "landlock:",
                f"  compatibility: {compatibility}",
                "process:",
                "  run_as_user: sandbox",
                "  run_as_group: sandbox",
                "network_policies: {}",
                extra,
            ]
        ),
        encoding="utf-8",
    )


def test_policy_reader_normalizes_filesystem_alias_and_requires_hard_landlock(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)

    policy = _read_policy_data(str(policy_path), require_hard_landlock=True)

    assert "filesystem" in policy
    assert "filesystem_policy" not in policy
    assert policy["landlock"]["compatibility"] == "hard_requirement"


def test_policy_reader_rejects_best_effort_in_production(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, compatibility="best_effort")

    with pytest.raises(ValueError, match="hard_requirement"):
        _read_policy_data(str(policy_path), require_hard_landlock=True)


def test_policy_reader_rejects_missing_file_and_wrong_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read"):
        _read_policy_data(str(tmp_path / "missing.yaml"), require_hard_landlock=True)

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 1"):
        _read_policy_data(str(policy_path), require_hard_landlock=True)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            "version: 1\nlandlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: sandbox, run_as_group: sandbox}\n",
            "filesystem",
        ),
        (
            "version: 1\nfilesystem_policy: {read_write: [/sandbox]}\n"
            "landlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: root, run_as_group: sandbox}\n",
            "non-root",
        ),
        (
            "version: 1\nfilesystem_policy: {read_write: [/sandbox]}\n"
            "landlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: 1000, run_as_group: sandbox}\n",
            "non-empty string",
        ),
        (
            "version: 1\nfilesystem_policy: {read_write: [/sandbox]}\n"
            "landlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: sandbox, run_as_group: sandbox}\n"
            "network_policies: {bad: {endpoints: [{host: example.com, enforcement: audit, access: read-write}]}}\n",
            "enforcement=enforce",
        ),
    ],
)
def test_policy_reader_enforces_production_floor(tmp_path: Path, policy: str, message: str) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(policy, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _read_policy_data(str(policy_path), require_hard_landlock=True)


def test_checked_policy_matches_installed_sdk_schema() -> None:
    if importlib.util.find_spec("openshell") is None:
        pytest.skip("optional OpenShell SDK is not installed")

    policy_path = "configs/openshell/deep-researcher-research-policy.yaml"
    policy_data = _read_policy_data(policy_path, require_hard_landlock=True)
    policy = _parse_policy_proto(policy_data, policy_path=policy_path)

    assert policy.version == 1

    spec = _build_sandbox_spec(policy=policy, image="deep-researcher:test", job_id="job-1")
    assert spec.template.image == "deep-researcher:test"
    assert dict(spec.template.labels) == {"deep-researcher": "deep-research", "deep-researcher-job-id": "job-1"}
    assert not spec.template.environment
    assert not spec.environment
    assert not spec.providers


def test_sandbox_spec_normalizes_long_job_id_label() -> None:
    if importlib.util.find_spec("openshell") is None:
        pytest.skip("optional OpenShell SDK is not installed")
    from openshell._proto import sandbox_pb2

    spec = _build_sandbox_spec(
        policy=sandbox_pb2.SandboxPolicy(version=1),
        image="deep-researcher:test",
        job_id="JOB_WITH_UNSAFE_CHARS_" * 8,
    )
    job_label = spec.template.labels["deep-researcher-job-id"]
    assert len(job_label) <= 63
    assert job_label == job_label.lower()
    assert "_" not in job_label


def test_policy_network_must_not_exceed_declared_allowlist() -> None:
    policy = {"network_policies": {"github": {"endpoints": [{"host": "api.github.com"}, {"host": "github.com"}]}}}

    _validate_policy_network(
        policy,
        mode="allowlist",
        allow=("api.github.com", "github.com"),
    )
    with pytest.raises(ValueError, match="github.com"):
        _validate_policy_network(policy, mode="allowlist", allow=("api.github.com",))
    with pytest.raises(ValueError, match="network endpoints"):
        _validate_policy_network(policy, mode="blocked", allow=())


@pytest.mark.parametrize("host", [None, "", "   ", "."])
def test_allowlist_rejects_hostless_endpoints(host: str | None) -> None:
    endpoint: dict[str, object] = {"allowed_ips": ["0.0.0.0/0"]}
    if host is not None:
        endpoint["host"] = host
    policy = {"network_policies": {"cidr": {"endpoints": [endpoint]}}}

    with pytest.raises(ValueError, match="non-empty host"):
        _validate_policy_network(policy, mode="allowlist", allow=("api.github.com",))
    with pytest.raises(ValueError, match="network endpoints"):
        _validate_policy_network(policy, mode="blocked", allow=())


@pytest.mark.parametrize("cidr", ["0.0.0.0/0", "10.0.0.0/8", "::/0", "2001:db8::/32"])
def test_allowlist_rejects_allowed_ip_overrides(cidr: str) -> None:
    policy = {
        "network_policies": {
            "mixed": {"endpoints": [{"host": "api.github.com", "allowed_ips": [cidr]}]},
        }
    }

    with pytest.raises(ValueError, match="CIDR"):
        _validate_policy_network(policy, mode="allowlist", allow=("api.github.com",))
    _validate_policy_network(policy, mode="open", allow=())


def test_per_job_session_uses_policy_spec_and_attests_before_return(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    created: list[tuple[_FakeCreatedContext, dict[str, Any]]] = []

    def context_factory(**kwargs: Any) -> _FakeCreatedContext:
        context = _FakeCreatedContext()
        created.append((context, kwargs))
        return context

    with (
        _fake_optional_modules(context_factory),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(
                provider="openshell",
                providers={
                    "openshell": {
                        "policy": str(policy_path),
                        "image": "deep-researcher:test",
                        "gateway": "sensitive-gateway-name",
                        "workspace": "research",
                    }
                },
            ),
            "job-123",
        )
        with caplog.at_level("INFO"):
            backend = provider._create_session()

    context, kwargs = created[0]
    assert backend.id == context.id
    assert context.enter_calls == 1
    assert kwargs["spec"] == "job-spec"
    assert kwargs["labels"] == {"deep-researcher": "deep-research", "deep-researcher-job-id": "job-123"}
    assert kwargs["delete_on_exit"] is True
    assert kwargs["workspace"] == "research"
    assert "sandbox" not in kwargs
    assert provider.physical_sandbox_name == "generated"
    assert "sensitive-gateway-name" not in caplog.text
    context._client._stub.GetSandboxPolicyStatus.assert_called_once()  # type: ignore[attr-defined]
    context._client._stub.GetSandboxConfig.assert_called_once()  # type: ignore[attr-defined]


def test_per_job_session_fails_before_creation_when_request_labels_are_unsupported(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)

    def context_factory(
        *,
        cluster: str | None,
        workspace: str,
        ready_timeout_seconds: float,
        spec: object,
        delete_on_exit: bool,
    ) -> _FakeCreatedContext:
        del cluster, workspace, ready_timeout_seconds, spec, delete_on_exit
        raise AssertionError("unsupported SDK must fail before creating a sandbox")

    with (
        _fake_optional_modules(context_factory),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        with pytest.raises(RuntimeError, match="request_labels_unsupported"):
            provider._create_session()

    assert provider._os_context is None


def test_two_jobs_create_distinct_specs_without_named_attachment(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    created: list[dict[str, Any]] = []
    spec_jobs: list[str] = []

    def context_factory(**kwargs: Any) -> _FakeCreatedContext:
        created.append(kwargs)
        return _FakeCreatedContext(name=f"generated-{len(created)}")

    def build_spec(*, policy: Any, image: str, job_id: str, labels: dict[str, str]) -> str:
        del policy, image
        assert labels == {"deep-researcher": "deep-research", "deep-researcher-job-id": job_id}
        spec_jobs.append(job_id)
        return f"spec-{job_id}"

    with (
        _fake_optional_modules(context_factory),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            side_effect=build_spec,
        ),
    ):
        for job_id in ("job-a", "job-b"):
            provider = OpenShellSandboxProvider(
                SandboxConfig(
                    provider="openshell",
                    providers={"openshell": {"policy": str(policy_path), "image": "deep-researcher:test"}},
                ),
                job_id,
            )
            provider._create_session()

    assert spec_jobs == ["job-a", "job-b"]
    assert [kwargs["spec"] for kwargs in created] == ["spec-job-a", "spec-job-b"]
    assert all("sandbox" not in kwargs for kwargs in created)


@pytest.mark.parametrize(
    ("phase", "policy_version", "expected", "message"),
    [
        (3, 1, None, "not_ready"),
        (2, 2, 1, "expected_version_mismatch"),
    ],
)
def test_attestation_failure_deletes_before_raising(
    tmp_path: Path,
    phase: int,
    policy_version: int,
    expected: int | None,
    message: str,
) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext(phase=phase, policy_version=policy_version)

    with (
        _fake_optional_modules(lambda **_kwargs: context),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(
                provider="openshell",
                providers={
                    "openshell": {
                        "policy": str(policy_path),
                        "expected_policy_version": expected,
                    }
                },
            ),
            "job-123",
        )
        with pytest.raises(RuntimeError, match=message):
            provider._create_session()

    assert context.exit_calls == 1
    assert provider._os_context is None


def test_attestation_reads_authoritative_policy_before_success(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext(policy_version=1)
    policy_status = context._client._stub.GetSandboxPolicyStatus  # type: ignore[attr-defined]
    sandbox_config = context._client._stub.GetSandboxConfig  # type: ignore[attr-defined]
    events: list[dict[str, object]] = []

    with (
        _fake_optional_modules(lambda **_kwargs: context),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        provider.set_event_emitter(events.append)

        provider._create_session()

    request = policy_status.call_args.args[0]
    assert request.name == context.sandbox.name
    assert request.version == 0
    assert request.workspace == "default"
    config_request = sandbox_config.call_args.args[0]
    assert config_request.sandbox_id == context.sandbox.id
    succeeded = [event for event in events if event["data"]["status"] == "succeeded"]  # type: ignore[index]
    assert len(succeeded) == 1
    assert succeeded[0]["data"]["policy_version"] == 1  # type: ignore[index]
    assert succeeded[0]["data"]["policy_hash"] == _FAKE_POLICY_HASH  # type: ignore[index]
    assert succeeded[0]["data"]["policy_source"] == 1  # type: ignore[index]
    assert succeeded[0]["data"]["assurance"] == "strict"  # type: ignore[index]


def test_attestation_polls_pending_revision_until_loaded() -> None:
    context = _FakeCreatedContext(policy_version=1)
    status_rpc = context._client._stub.GetSandboxPolicyStatus  # type: ignore[attr-defined]
    loaded = status_rpc.return_value
    pending = SimpleNamespace(
        revision=SimpleNamespace(**{**vars(loaded.revision), "status": 1}),
        active_version=0,
    )
    status_rpc.side_effect = [pending, loaded]

    result = _run_attestation(context, policy_load_timeout_seconds=1.0)

    assert result.policy_version == 1
    assert status_rpc.call_count == 2
    assert context._client._stub.GetSandboxConfig.call_count == 2  # type: ignore[attr-defined]


def test_attestation_accepts_zero_initial_version_after_authoritative_transition() -> None:
    context = _FakeCreatedContext(policy_version=0)
    refreshed = SimpleNamespace(**vars(context.sandbox))
    refreshed.current_policy_version = 1
    context._client.get.return_value = refreshed  # type: ignore[attr-defined]
    status_rpc = context._client._stub.GetSandboxPolicyStatus  # type: ignore[attr-defined]
    loaded = status_rpc.return_value
    pending = SimpleNamespace(
        revision=SimpleNamespace(**{**vars(loaded.revision), "status": 1}),
        active_version=0,
    )
    status_rpc.side_effect = [pending, loaded]

    result = _run_attestation(context, policy_load_timeout_seconds=1.0)

    assert result.policy_version == 1
    assert status_rpc.call_count == 2
    assert context._client.get.call_count == 2  # type: ignore[attr-defined]


def test_attestation_classifies_effective_policy_that_remains_pending() -> None:
    context = _FakeCreatedContext(policy_version=0)
    status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
    status.revision.status = 1
    status.active_version = 0

    with pytest.raises(RuntimeError, match="policy_status_inconsistent"):
        _run_attestation(context, policy_load_timeout_seconds=0.001)


def _run_attestation(
    context: _FakeCreatedContext,
    *,
    expected_policy: _FakePolicy | None = _FAKE_POLICY,
    require_sandbox_source: bool = True,
    **openshell_config: Any,
):
    with _fake_optional_modules(lambda **_kwargs: context):
        provider = _provider(**openshell_config)
        provider._os_context = context
        return provider._attest(
            context,
            expected_policy=expected_policy,
            require_sandbox_source=require_sandbox_source,
        )


def test_attestation_fails_closed_when_authoritative_rpc_is_unavailable() -> None:
    context = _FakeCreatedContext()
    del context._client._stub.GetSandboxConfig  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="rpc_unavailable"):
        _run_attestation(context)


@pytest.mark.parametrize("field", ["active_version", "revision_version", "config_version", "current_version"])
def test_attestation_rejects_version_disagreement(field: str) -> None:
    context = _FakeCreatedContext()
    status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
    config = context._client._stub.GetSandboxConfig.return_value  # type: ignore[attr-defined]
    if field == "active_version":
        status.active_version = 2
    elif field == "revision_version":
        status.revision.version = 2
    elif field == "config_version":
        config.version = 2
    else:
        context.sandbox.current_policy_version = 2

    with pytest.raises(RuntimeError, match="version_mismatch"):
        _run_attestation(context)


def test_attestation_rejects_unreported_current_and_active_versions() -> None:
    context = _FakeCreatedContext(policy_version=0)
    status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
    status.active_version = 0

    with pytest.raises(RuntimeError, match="version_mismatch"):
        _run_attestation(context)


def test_attestation_rejects_unreported_active_version_with_matching_current_version() -> None:
    context = _FakeCreatedContext(policy_version=1)
    status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
    status.active_version = 0

    with pytest.raises(RuntimeError, match="version_mismatch"):
        _run_attestation(context)


@pytest.mark.parametrize(
    ("status", "load_error", "reason"),
    [(3, "", "policy_status_failed"), (2, "credential=do-not-log", "policy_load_error")],
)
def test_attestation_rejects_failed_policy_revision(status: int, load_error: str, reason: str) -> None:
    context = _FakeCreatedContext()
    revision = context._client._stub.GetSandboxPolicyStatus.return_value.revision  # type: ignore[attr-defined]
    revision.status = status
    revision.load_error = load_error

    with pytest.raises(RuntimeError, match=reason) as exc_info:
        _run_attestation(context)
    assert "credential=do-not-log" not in str(exc_info.value)


@pytest.mark.parametrize("surface", ["config", "revision"])
def test_attestation_rejects_effective_policy_content_mismatch(surface: str) -> None:
    context = _FakeCreatedContext()
    if surface == "config":
        context._client._stub.GetSandboxConfig.return_value.policy = _FakePolicy(version=2)  # type: ignore[attr-defined]
    else:
        status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
        status.revision.policy = _FakePolicy(version=2)

    with pytest.raises(RuntimeError, match="policy_content_mismatch"):
        _run_attestation(context)


@pytest.mark.parametrize("surface", ["config", "revision"])
def test_attestation_rejects_policy_hash_mismatch(surface: str) -> None:
    context = _FakeCreatedContext()
    if surface == "config":
        context._client._stub.GetSandboxConfig.return_value.policy_hash = "bad"  # type: ignore[attr-defined]
    else:
        status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
        status.revision.policy_hash = "bad"

    with pytest.raises(RuntimeError, match="policy_hash_mismatch"):
        _run_attestation(context)


@pytest.mark.parametrize("surface", ["config", "revision"])
def test_attestation_rejects_missing_authoritative_policy_hash(surface: str) -> None:
    context = _FakeCreatedContext()
    if surface == "config":
        context._client._stub.GetSandboxConfig.return_value.policy_hash = ""  # type: ignore[attr-defined]
    else:
        status = context._client._stub.GetSandboxPolicyStatus.return_value  # type: ignore[attr-defined]
        status.revision.policy_hash = ""

    with pytest.raises(RuntimeError, match="policy_hash_missing"):
        _run_attestation(context)


def test_attestation_rejects_gateway_global_policy_for_owned_sandbox() -> None:
    context = _FakeCreatedContext(policy_source=2)

    with pytest.raises(RuntimeError, match="policy_source_mismatch"):
        _run_attestation(context)


def test_policy_configured_shared_debug_accepts_matching_global_effective_policy() -> None:
    context = _FakeCreatedContext(policy_source=2)

    result = _run_attestation(context, require_sandbox_source=False)

    assert result.assurance == "strict"
    assert result.policy_source == 2


def test_shared_debug_without_policy_reports_reduced_assurance() -> None:
    context = _FakeCreatedContext(policy_source=2)

    result = _run_attestation(context, expected_policy=None, require_sandbox_source=False)

    assert result.assurance == "reduced"
    assert result.policy_version == 1


def test_shared_attachment_requires_explicit_debug_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_shared_sandbox"):
        SandboxConfig(provider="openshell", providers={"openshell": {"existing_sandbox_name": "shared"}})

    config = SandboxConfig(
        provider="openshell",
        providers={
            "openshell": {
                "existing_sandbox_name": "shared",
                "allow_shared_sandbox": True,
                "delete_on_exit": False,
            }
        },
    )
    assert config.providers.openshell.shared_sandbox_name == "shared"

    with pytest.raises(ValueError, match="requires attest=true"):
        SandboxConfig(
            provider="openshell",
            providers={
                "openshell": {
                    "existing_sandbox_name": "shared",
                    "allow_shared_sandbox": True,
                    "policy": "policy.yaml",
                    "attest": False,
                }
            },
        )


def test_shared_attachment_never_deletes_unowned_sandbox() -> None:
    created: list[dict[str, Any]] = []

    def context_factory(**kwargs: Any) -> _FakeCreatedContext:
        created.append(kwargs)
        return _FakeCreatedContext(name="shared")

    with _fake_optional_modules(context_factory):
        provider = OpenShellSandboxProvider(
            SandboxConfig(
                provider="openshell",
                providers={
                    "openshell": {
                        "existing_sandbox_name": "shared",
                        "allow_shared_sandbox": True,
                        "delete_on_exit": True,
                    }
                },
            ),
            "job-123",
        )
        provider._create_session()
        provider.close()

    assert created[0]["sandbox"] == "shared"
    assert created[0]["delete_on_exit"] is False
    assert "spec" not in created[0]


def test_per_job_mode_rejects_disabled_attestation() -> None:
    with pytest.raises(ValueError, match="attest=true"):
        SandboxConfig(
            provider="openshell",
            providers={"openshell": {"policy": "policy.yaml", "attest": False}},
        )

    with pytest.raises(ValueError, match="delete_on_exit=true"):
        SandboxConfig(
            provider="openshell",
            providers={"openshell": {"policy": "policy.yaml", "delete_on_exit": False}},
        )


def test_adapter_construction_failure_deletes_sandbox(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext()

    class FailingAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("bad adapter configuration")

    with (
        _fake_optional_modules(lambda **_kwargs: context, FailingAdapter),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        events: list[dict[str, object]] = []
        provider.set_event_emitter(events.append)
        with pytest.raises(ValueError, match="bad adapter"):
            provider._create_session()

    assert context.exit_calls == 1
    assert provider._os_context is None
    assert not any(event["data"]["status"] == "succeeded" for event in events)  # type: ignore[index]


def test_attestation_success_is_emitted_after_adapter_construction(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext()
    order: list[str] = []

    class RecordingAdapter(_FakeAdapter):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            order.append("adapter")

    with (
        _fake_optional_modules(lambda **_kwargs: context, RecordingAdapter),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        provider.set_event_emitter(lambda event: order.append(event["data"]["status"]))  # type: ignore[index]
        provider._create_session()

    assert order == ["adapter", "succeeded"]


def test_terminate_before_context_entry_prevents_creation(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext()

    class UnexpectedAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("adapter must not be constructed after termination")

    with (
        _fake_optional_modules(lambda **_kwargs: context, UnexpectedAdapter),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        provider.terminate()

        with pytest.raises(SandboxTerminatedError):
            provider._create_session()

    assert context.enter_calls == 0
    assert context.exit_calls == 0


def test_terminate_during_context_entry_defers_one_exit(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    entry_started = Event()
    allow_entry = Event()

    class BlockingContext(_FakeCreatedContext):
        def __enter__(self):
            self.enter_calls += 1
            entry_started.set()
            if not allow_entry.wait(timeout=2):
                raise AssertionError("context entry was not released")
            return self

    class UnexpectedAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("adapter must not be constructed after termination")

    context = BlockingContext()
    errors: list[BaseException] = []

    with (
        _fake_optional_modules(lambda **_kwargs: context, UnexpectedAdapter),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )

        def create() -> None:
            try:
                provider._session_or_create()
            except BaseException as exc:  # noqa: BLE001 - capture the worker outcome for assertion
                errors.append(exc)

        worker = Thread(target=create)
        worker.start()
        assert entry_started.wait(timeout=2)

        terminate_done = Event()
        terminator = Thread(target=lambda: (provider.terminate(), terminate_done.set()))
        terminator.start()
        assert not terminate_done.wait(timeout=0.05)
        assert context.exit_calls == 0
        allow_entry.set()
        worker.join(timeout=2)
        terminator.join(timeout=2)

    assert not worker.is_alive()
    assert not terminator.is_alive()
    assert terminate_done.is_set()
    assert len(errors) == 1 and isinstance(errors[0], SandboxTerminatedError)
    assert context.exit_calls == 1
    assert provider._session is None


def test_terminate_during_adapter_construction_cannot_publish_session(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    adapter_started = Event()
    allow_adapter = Event()
    context = _FakeCreatedContext()
    errors: list[BaseException] = []
    events: list[dict[str, object]] = []

    class BlockingAdapter(_FakeAdapter):
        def __init__(self, **kwargs: Any) -> None:
            adapter_started.set()
            if not allow_adapter.wait(timeout=2):
                raise AssertionError("adapter construction was not released")
            super().__init__(**kwargs)

    with (
        _fake_optional_modules(lambda **_kwargs: context, BlockingAdapter),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        provider.set_event_emitter(events.append)

        def create() -> None:
            try:
                provider._session_or_create()
            except BaseException as exc:  # noqa: BLE001 - capture the worker outcome for assertion
                errors.append(exc)

        worker = Thread(target=create)
        worker.start()
        assert adapter_started.wait(timeout=2)

        provider.terminate()
        allow_adapter.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SandboxTerminatedError)
    assert context.exit_calls == 1
    assert provider._session is None
    assert not any(event["data"]["status"] == "succeeded" for event in events)  # type: ignore[index]


def test_upload_passes_path_via_argv_and_data_via_stdin_no_env() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    provider._os_context = fake

    result = provider.upload_files([("/sandbox/x.py", b"print('hi')")])

    assert result[0].error is None
    call = fake.calls[0]
    assert call["command"][0] == "python3" and call["command"][-1] == "/sandbox/x.py"
    assert call["stdin"] == base64.b64encode(b"print('hi')")
    assert "env" not in call  # the whole point: never rely on exec env propagation


def test_upload_classifies_failure() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=1, stderr="No such file or directory")
    provider._os_context = fake

    result = provider.upload_files([("/sandbox/x.py", b"data")])
    assert result[0].error == "file_not_found"


def test_upload_rejects_relative_path() -> None:
    provider = _provider()
    provider._os_context = _FakeOpenShellSandbox()
    result = provider.upload_files([("relative.py", b"data")])
    assert result[0].error == "invalid_path"


def test_download_passes_path_via_argv_and_decodes_base64() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout=base64.b64encode(b"chart-bytes").decode())
    provider._os_context = fake

    artifact_path = f"{provider.artifact_dir}/chart.png"
    result = provider.download_files([artifact_path])

    assert result[0].error is None
    assert result[0].content == b"chart-bytes"
    call = fake.calls[0]
    # Path is passed positionally via argv (with the size cap appended); never via env.
    assert artifact_path in call["command"]
    assert call["command"][-1] == provider.artifact_dir
    assert "env" not in call


def test_download_uses_confined_shim_when_adapter_transfer_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout=base64.b64encode(b"chart-bytes").decode())
    provider._os_context = fake
    provider._session.download_files.side_effect = AssertionError("adapter download bypassed confinement")  # type: ignore[union-attr]
    monkeypatch.setenv("DEEP_RESEARCHER_OPENSHELL_ADAPTER_FILE_TRANSFER", "1")

    result = provider.download_files([f"{provider.artifact_dir}/chart.png"])

    assert result[0].content == b"chart-bytes"
    assert fake.calls[0]["command"][-1] == provider.artifact_dir


def test_download_is_directory_exit_code() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=3, stderr="")
    provider._os_context = fake

    result = provider.download_files(["/sandbox"])
    assert result[0].content is None
    assert result[0].error == "is_directory"  # _DOWNLOAD_CODE exits 3 specifically for a directory


def test_download_passes_size_cap_via_argv() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout=base64.b64encode(b"x").decode())
    provider._os_context = fake

    provider.download_files(["/sandbox/deep-researcher-artifacts/chart.png"])

    # The artifact size cap is passed to the bootstrap so it can refuse oversized files
    # before reading them into host memory.
    assert str(provider.config.artifact_capture.max_file_bytes) in fake.calls[0]["command"]


def test_download_rejects_oversized_and_symlink() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    provider._os_context = fake

    fake.result = _ExecResult(exit_code=4, stderr="")
    assert provider.download_files(["/sandbox/deep-researcher-artifacts/huge.bin"])[0].error == "too_large"

    fake.result = _ExecResult(exit_code=5, stderr="")
    assert provider.download_files(["/sandbox/deep-researcher-artifacts/evil.png"])[0].error == "symlink_rejected"


def test_download_rejects_non_base64_stdout() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout="not valid base64 !!!")
    provider._os_context = fake

    result = provider.download_files(["/sandbox/deep-researcher-artifacts/chart.png"])
    assert result[0].content is None
    assert result[0].error == "invalid_content"


def test_terminate_exits_openshell_context_once() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    provider._os_context = fake

    provider.terminate()
    provider.terminate()

    assert fake.exit_calls == 1
    assert provider._os_context is None


def test_context_exit_failure_marks_cleanup_failed() -> None:
    provider = _provider()

    class FailingContext:
        def __exit__(self, *_args: object) -> None:
            raise RuntimeError("gateway delete failed")

    provider._os_context = FailingContext()

    provider.close()

    assert provider.cleanup_succeeded is False
    assert provider.cleanup_failure_reason_codes == ("context_exit_failed",)
    assert provider._os_context is None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_cleanup_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="cleanup_timeout_seconds"):
        SandboxConfig(provider="openshell", providers={"openshell": {"cleanup_timeout_seconds": timeout}})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_policy_load_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="policy_load_timeout_seconds"):
        SandboxConfig(provider="openshell", providers={"openshell": {"policy_load_timeout_seconds": timeout}})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_ready_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="ready_timeout_seconds"):
        SandboxConfig(provider="openshell", providers={"openshell": {"ready_timeout_seconds": timeout}})


def test_deferred_context_cleanup_timeout_is_terminal() -> None:
    provider = _provider(cleanup_timeout_seconds=0.01)
    provider._os_context = _FakeOpenShellSandbox()
    provider._os_context_entering = True
    provider._os_context_cleanup_complete = Event()

    provider._exit_context()

    assert provider.cleanup_succeeded is False
    assert provider.cleanup_failure_reason_codes == ("cleanup_timeout",)
    assert provider._os_context_exit_requested is True


def test_finalize_waits_for_deferred_exit_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from deep_researcher_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
    from deep_researcher_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig

    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    entry_started = Event()
    allow_entry = Event()

    class FailingExitContext(_FakeCreatedContext):
        def __enter__(self):
            self.enter_calls += 1
            entry_started.set()
            if not allow_entry.wait(timeout=2):
                raise AssertionError("context entry was not released")
            return self

        def __exit__(self, *_args: object) -> None:
            self.exit_calls += 1
            raise RuntimeError("credential=do-not-log")

    context = FailingExitContext()
    errors: list[BaseException] = []
    finalize_results: list[bool] = []
    events: list[dict[str, object]] = []

    with (
        _fake_optional_modules(lambda **_kwargs: context),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value=_FAKE_POLICY,
        ),
        patch(
            "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(
                provider="openshell",
                providers={"openshell": {"policy": str(policy_path), "cleanup_timeout_seconds": 1}},
            ),
            "job-123",
        )
        with patch(
            "deep_researcher_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig(), artifact_emit=events.append)

        def create() -> None:
            try:
                provider._session_or_create()
            except BaseException as exc:  # noqa: BLE001 - capture worker outcome
                errors.append(exc)

        creator = Thread(target=create)
        creator.start()
        assert entry_started.wait(timeout=2)

        finalizer = Thread(target=lambda: finalize_results.append(runtime.finalize(interrupted=True)))
        with caplog.at_level("WARNING"):
            finalizer.start()
            assert finalizer.is_alive()
            allow_entry.set()
            creator.join(timeout=2)
            finalizer.join(timeout=2)

    assert not creator.is_alive()
    assert not finalizer.is_alive()
    assert finalize_results == [False]
    assert len(errors) == 1 and isinstance(errors[0], SandboxTerminatedError)
    assert context.exit_calls == 1
    assert provider.cleanup_failure_reason_codes == ("context_exit_failed",)
    assert [event["data"]["status"] for event in events] == ["started", "failed"]  # type: ignore[index]
    assert events[-1]["data"]["reason_codes"] == ["context_exit_failed"]  # type: ignore[index]
    assert "RuntimeError" in caplog.text
    assert "credential=do-not-log" not in caplog.text


def test_retry_cleanup_failure_remains_terminal_failure() -> None:
    from deep_researcher_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
    from deep_researcher_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig

    provider = _provider()
    stale = MagicMock()
    stale.close.side_effect = RuntimeError("stale sandbox close failed")
    replacement = MagicMock()
    provider._session = stale
    provider._create_session = MagicMock(return_value=replacement)  # type: ignore[method-assign]
    provider._prepare_workspace = MagicMock()  # type: ignore[method-assign]
    events: list[dict[str, object]] = []

    with patch(
        "deep_researcher_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
        return_value=provider,
    ):
        runtime = DeepAgentsRuntime(
            sandbox=DeepResearchSandboxConfig(),
            artifact_emit=events.append,
        )

    provider._reset_session()

    assert runtime.finalize(interrupted=False) is False
    replacement.close.assert_called_once_with()
    assert provider.cleanup_succeeded is False
    assert [event["data"]["status"] for event in events] == ["started", "failed"]  # type: ignore[index]


def test_sample_policy_grants_broad_proc_not_proc_self() -> None:
    """OpenShell requires read access to /proc (not just /proc/self); guard against regression."""
    import yaml

    policy = yaml.safe_load(Path("configs/openshell/deep-researcher-research-policy.yaml").read_text(encoding="utf-8"))
    read_only = policy["filesystem_policy"]["read_only"]
    assert "/proc" in read_only
    assert "/proc/self" not in read_only
