# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strict OpenShell readiness capability checker."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "scripts" / "openshell" / "check_openshell_readiness.py"


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_openshell_readiness", _CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RpcError(Exception):
    def __init__(self, code: str) -> None:
        self._code = code

    def code(self) -> str:
        return self._code


class _Grpc(ModuleType):
    class StatusCode:
        NOT_FOUND = "not_found"

    Call = _RpcError
    RpcError = _RpcError


class _Stub:
    def __init__(self, *, status: int = 2) -> None:
        policy = SimpleNamespace(version=1)
        self.status = SimpleNamespace(
            active_version=1,
            revision=SimpleNamespace(
                version=1,
                status=status,
                load_error="",
                policy=policy,
                policy_hash="hash",
            ),
        )
        self.config = SimpleNamespace(version=1, policy_source=1, policy=policy, policy_hash="hash")

    def GetSandboxPolicyStatus(self, _request: object, *, timeout: float) -> object:  # noqa: N802
        assert getattr(_request, "workspace") == "research"
        del timeout
        return self.status

    def GetSandboxConfig(self, _request: object, *, timeout: float) -> object:  # noqa: N802
        del timeout
        return self.config


class _Client:
    instance: _Client

    def __init__(
        self,
        *,
        status: int = 2,
        labels: dict[str, str] | None = None,
        persist_request_labels: bool = True,
        version: str = "1.2.3",
    ) -> None:
        self.labels = labels or {"deep-researcher": "readiness-probe"}
        self.persist_request_labels = persist_request_labels
        self.version = version
        self.sandbox = SimpleNamespace(
            id="probe-id",
            name="probe-name",
            phase=2,
            current_policy_version=1,
            labels=self.labels,
        )
        self._stub = _Stub(status=status)
        self._stub.status.revision.policy = self._stub.config.policy
        self.deleted = False
        self.delete_calls = 0
        type(self).instance = self

    @classmethod
    def from_active_cluster(cls, *, cluster: str | None) -> _Client:
        del cluster
        return cls.instance

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def health(self) -> object:
        return SimpleNamespace(version=self.version)

    def create(self, *, workspace: str, spec: object, name: str, labels: dict[str, str]) -> object:
        assert workspace == "research"
        del spec
        self.sandbox.name = name
        if self.persist_request_labels:
            self.sandbox.labels = labels
        return self.sandbox

    def wait_ready(self, name: str, *, workspace: str, timeout_seconds: float) -> object:
        assert workspace == "research"
        del timeout_seconds
        assert name == self.sandbox.name
        return self.sandbox

    def list(self, *, workspace: str, label_selector: str) -> list[object]:
        assert workspace == "research"
        assert label_selector == "deep-researcher=readiness-probe"
        return [] if self.deleted else [self.sandbox]

    def get(self, name: str, *, workspace: str) -> object:
        assert workspace == "research"
        assert name == self.sandbox.name
        if self.deleted:
            raise _RpcError(_Grpc.StatusCode.NOT_FOUND)
        return self.sandbox

    def exec(self, sandbox_id: str, command: list[str]) -> object:
        assert sandbox_id == self.sandbox.id
        assert command
        return SimpleNamespace(exit_code=0, stdout="deep-researcher-openshell-ready")

    def delete(self, name: str, *, workspace: str) -> bool:
        assert workspace == "research"
        assert name == self.sandbox.name
        self.delete_calls += 1
        self.deleted = True
        return True

    def wait_deleted(self, name: str, *, workspace: str) -> None:
        assert workspace == "research"
        assert name == self.sandbox.name


class _NoLabelClient(_Client):
    def create(self, *, workspace: str, spec: object) -> object:
        assert workspace == "research"
        del spec
        return self.sandbox

    def list(self) -> list[object]:
        return [self.sandbox]


class _DeleteFailureClient(_Client):
    def delete(self, name: str, *, workspace: str) -> bool:
        assert workspace == "research"
        assert name == self.sandbox.name
        self.delete_calls += 1
        return False


class _MismatchedNameClient(_Client):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_name: str | None = None

    def create(self, *, workspace: str, spec: object, name: str, labels: dict[str, str]) -> object:
        assert workspace == "research"
        del spec, name
        self.sandbox.name = "gateway-returned-name"
        self.sandbox.labels = labels
        return self.sandbox

    def delete(self, name: str, *, workspace: str) -> bool:
        assert workspace == "research"
        self.deleted_name = name
        return super().delete(name, workspace=workspace)


@contextmanager
def _fake_runtime(client: _Client):
    openshell = ModuleType("openshell")
    sandbox_module = ModuleType("openshell.sandbox")
    sandbox_module.SandboxClient = type(client)  # type: ignore[attr-defined]
    proto_module = ModuleType("openshell._proto")
    proto_module.openshell_pb2 = SimpleNamespace(  # type: ignore[attr-defined]
        SANDBOX_PHASE_READY=2,
        POLICY_STATUS_UNSPECIFIED=0,
        POLICY_STATUS_PENDING=1,
        POLICY_STATUS_LOADED=2,
        POLICY_STATUS_FAILED=3,
        GetSandboxPolicyStatusRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    proto_module.sandbox_pb2 = SimpleNamespace(  # type: ignore[attr-defined]
        POLICY_SOURCE_UNSPECIFIED=0,
        POLICY_SOURCE_SANDBOX=1,
        GetSandboxConfigRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    grpc = _Grpc("grpc")
    with patch.dict(
        sys.modules,
        {
            "grpc": grpc,
            "openshell": openshell,
            "openshell._proto": proto_module,
            "openshell.sandbox": sandbox_module,
        },
    ):
        yield


def _config(checker: ModuleType, tmp_path: Path, **overrides: Any) -> object:
    values = {
        "gateway": "enterprise",
        "workspace": "research",
        "image": "deep-researcher:test",
        "policy": tmp_path / "policy.yaml",
        "openshell_bin": tmp_path / "openshell",
        "ready_timeout_seconds": 1.0,
        "policy_load_timeout_seconds": 0.01,
    }
    values.update(overrides)
    return checker.ReadinessConfig(**values)


@contextmanager
def _provider_helpers(client: _Client):
    prefix = "deep_researcher_agent.agents.deep_researcher.sandbox.providers.openshell"
    policy = SimpleNamespace(version=1)
    client._stub.config.policy = policy
    client._stub.status.revision.policy = policy
    with (
        patch(f"{prefix}._read_policy_data", return_value={"version": 1}),
        patch(f"{prefix}._parse_policy_proto", return_value=policy),
        patch(f"{prefix}._build_sandbox_spec", return_value="spec"),
    ):
        yield policy


def test_strict_probe_verifies_and_deletes_resource(checker: ModuleType, tmp_path: Path) -> None:
    client = _Client()
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
    ):
        versions = checker.run_check(_config(checker, tmp_path))

    assert versions == ("1.2.3", "1.2.3")
    assert client.delete_calls == 1
    assert client.deleted is True


def test_probe_rejects_version_mismatch_before_creation(checker: ModuleType, tmp_path: Path) -> None:
    client = _Client()
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.4"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="version_mismatch"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 0


@pytest.mark.parametrize("field", ["active_version", "current_policy_version"])
def test_probe_rejects_unreported_policy_versions(checker: ModuleType, tmp_path: Path, field: str) -> None:
    client = _Client()
    if field == "active_version":
        client._stub.status.active_version = 0
    else:
        client.sandbox.current_policy_version = 0
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="version_mismatch"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 1


def test_probe_accepts_equivalent_python_and_cargo_development_versions(
    checker: ModuleType,
    tmp_path: Path,
) -> None:
    client = _Client(version="0.0.79-dev.3+g616ff2f6")
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="0.0.79.dev3+g616ff2f6"),
        patch.object(checker, "_version_from_cli", return_value="0.0.79-dev.3+g616ff2f6"),
    ):
        versions = checker.run_check(_config(checker, tmp_path))

    assert versions == ("0.0.79.dev3+g616ff2f6", "0.0.79-dev.3+g616ff2f6")
    assert client.delete_calls == 1


def test_probe_rejects_different_development_commit(checker: ModuleType, tmp_path: Path) -> None:
    client = _Client(version="0.0.79-dev.3+g616ff2f7")
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="0.0.79.dev3+g616ff2f6"),
        patch.object(checker, "_version_from_cli", return_value="0.0.79-dev.3+g616ff2f6"),
        pytest.raises(checker.ReadinessError, match="version_mismatch"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 0


def test_probe_rejects_sdk_without_request_label_support(checker: ModuleType, tmp_path: Path) -> None:
    client = _NoLabelClient()
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="request_labels_unsupported"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 0


@pytest.mark.parametrize("method_name", ["create", "list", "wait_ready", "get", "delete", "wait_deleted"])
def test_probe_rejects_sdk_without_workspace_support(
    checker: ModuleType,
    tmp_path: Path,
    method_name: str,
) -> None:
    client = _Client()
    setattr(client, method_name, lambda: None)
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="request_labels_unsupported"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 0


def test_parser_treats_empty_workspace_as_default(
    checker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_OPENSHELL_WORKSPACE", "")

    args = checker._parser().parse_args(["--policy-file", "policy.yaml", "--openshell-bin", "openshell"])

    assert args.workspace == "default"


def test_probe_classifies_effective_policy_that_remains_pending(checker: ModuleType, tmp_path: Path) -> None:
    client = _Client(status=1)
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="policy_status_inconsistent"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 1


@pytest.mark.parametrize("surface", ["config", "revision"])
def test_probe_rejects_missing_authoritative_policy_hash(
    checker: ModuleType,
    tmp_path: Path,
    surface: str,
) -> None:
    client = _Client()
    target = client._stub.config if surface == "config" else client._stub.status.revision
    target.policy_hash = ""
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="policy_hash_missing"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 1


def test_probe_rejects_unequal_authoritative_policy_hashes(checker: ModuleType, tmp_path: Path) -> None:
    client = _Client()
    client._stub.status.revision.policy_hash = "other-hash"
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="policy_hash_mismatch"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 1


def test_probe_rejects_selector_metadata_mismatch_and_cleans_up(checker: ModuleType, tmp_path: Path) -> None:
    client = _Client(labels={"deep-researcher": "wrong"}, persist_request_labels=False)
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="selector_mismatch"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 1


def test_probe_cleans_up_sdk_returned_name_on_name_mismatch(checker: ModuleType, tmp_path: Path) -> None:
    client = _MismatchedNameClient()
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="probe_failed"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.deleted_name == "gateway-returned-name"
    assert client.deleted is True


def test_probe_reports_cleanup_failure(checker: ModuleType, tmp_path: Path) -> None:
    client = _DeleteFailureClient()
    with (
        _provider_helpers(client),
        _fake_runtime(client),
        patch.object(checker.importlib.metadata, "version", return_value="1.2.3"),
        patch.object(checker, "_version_from_cli", return_value="1.2.3"),
        pytest.raises(checker.ReadinessError, match="cleanup_failed"),
    ):
        checker.run_check(_config(checker, tmp_path))

    assert client.delete_calls == 1


def test_main_prints_only_sanitized_reason(checker: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(checker, "run_check", side_effect=checker.ReadinessError("selector_mismatch")):
        result = checker.main(
            [
                "--policy-file",
                "policy.yaml",
                "--openshell-bin",
                "openshell",
            ]
        )

    assert result == 1
    assert capsys.readouterr().err.strip() == "OpenShell readiness check failed: selector_mismatch"
