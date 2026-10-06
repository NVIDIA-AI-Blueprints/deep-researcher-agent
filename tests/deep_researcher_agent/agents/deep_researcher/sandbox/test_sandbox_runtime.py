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

"""Tests for the provider-neutral sandbox seam (registry, config, capabilities, base).

These run without a live Modal/OpenShell gateway: provider behavior is exercised
through small fakes, so only the framework logic (dispatch, fail-closed gate,
lazy creation, idempotency-gated retry, cleanup) is under test.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from deep_researcher_agent.agents.deep_researcher.sandbox import CapabilityError
from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxCapabilities
from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxConfig
from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxProvider
from deep_researcher_agent.agents.deep_researcher.sandbox import SandboxTerminatedError
from deep_researcher_agent.agents.deep_researcher.sandbox import create_sandbox_backend
from deep_researcher_agent.agents.deep_researcher.sandbox import register_sandbox_provider
from deep_researcher_agent.agents.deep_researcher.sandbox import registered_providers
from deep_researcher_agent.agents.deep_researcher.sandbox import verify_capabilities
from deep_researcher_agent.agents.deep_researcher.sandbox.config import job_scoped_artifact_dir
from deep_researcher_agent.agents.deep_researcher.sandbox.config import job_scoped_workdir


class _RecoverableError(Exception):
    """Stand-in for a provider's transient/stale-sandbox error."""


class _RegisteredFake(SandboxProvider):
    """Minimal registered provider with conservative (default) capabilities."""

    provider_name = "registered-fake"

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    def _create_session(self) -> Any:
        return MagicMock()

    def _prepare_workspace(self, session: Any) -> None:
        # These fakes assert exact execute call counts in the lock/retry/timeout tests;
        # per-job workspace prep is covered explicitly in TestWorkspacePreparation.
        return None


class _ScriptedProvider(SandboxProvider):
    """Provider that hands out caller-supplied sessions, for retry/lazy tests."""

    provider_name = "scripted"

    def __init__(self, config: SandboxConfig, job_id: str, sessions: list[Any]) -> None:
        super().__init__(config, job_id)
        self._sessions = sessions
        self.sessions_created = 0

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(supports_network_policy=True)

    def is_recoverable_error(self, exc: Exception) -> bool:
        return isinstance(exc, _RecoverableError)

    def _create_session(self) -> Any:
        session = self._sessions[self.sessions_created]
        self.sessions_created += 1
        return session

    def _prepare_workspace(self, session: Any) -> None:
        # See _RegisteredFake: keep exact execute call counts under test.
        return None


class _WorkspaceProvider(SandboxProvider):
    """Uses the default ``_prepare_workspace`` so its mkdir-on-create is observable."""

    provider_name = "workspace-fake"

    def __init__(self, config: SandboxConfig, job_id: str, session: Any) -> None:
        super().__init__(config, job_id)
        self._next = session

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(supports_network_policy=True)

    def _create_session(self) -> Any:
        return self._next


register_sandbox_provider("registered-fake", _RegisteredFake)


def _fake_config(**overrides: Any) -> SandboxConfig:
    base: dict[str, Any] = {"provider": "registered-fake", "block_network": False}
    base.update(overrides)
    return SandboxConfig(**base)


class TestRegistry:
    def test_builtin_providers_registered(self) -> None:
        assert "modal" in registered_providers()
        assert "openshell" in registered_providers()

    def test_create_unknown_provider_raises(self) -> None:
        config = _fake_config()
        object.__setattr__(config, "provider", "ghost")  # bypass field validation
        with pytest.raises(ValueError, match="Registered providers"):
            create_sandbox_backend(config, "job-1")

    def test_create_returns_provider_instance(self) -> None:
        backend = create_sandbox_backend(_fake_config(), "job-1")
        assert isinstance(backend, _RegisteredFake)


class TestSandboxConfig:
    def test_nested_modal_provider_settings(self) -> None:
        config = SandboxConfig(provider="modal", providers={"modal": {"image": "nested:tag"}})
        assert config.providers.modal.image == "nested:tag"

    def test_provider_normalized_lowercase(self) -> None:
        assert SandboxConfig(provider="MODAL").provider == "modal"

    def test_default_workdir(self) -> None:
        config = SandboxConfig()
        assert config.workdir == "/workspace"

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValueError, match="Registered providers"):
            SandboxConfig(provider="does-not-exist")


class TestCapabilityGate:
    def test_block_network_requires_capability(self) -> None:
        config = _fake_config(block_network=True)  # _RegisteredFake declares no network policy
        with pytest.raises(CapabilityError, match="block_network"):
            create_sandbox_backend(config, "job-1")

    def test_passes_when_network_unblocked(self) -> None:
        backend = create_sandbox_backend(_fake_config(block_network=False), "job-1")
        assert isinstance(backend, _RegisteredFake)

    def test_artifact_capture_requires_download(self) -> None:
        caps = SandboxCapabilities(supports_network_policy=True, supports_artifact_download=False)
        config = SandboxConfig(provider="registered-fake", artifact_capture={"enabled": True})
        with pytest.raises(CapabilityError, match="download"):
            verify_capabilities(config, caps)


class TestNetworkPolicy:
    def test_legacy_block_network_true_maps_to_blocked(self) -> None:
        config = SandboxConfig(provider="registered-fake", block_network=True)
        assert config.network.mode == "blocked"
        assert config.block_network is True

    def test_legacy_block_network_false_maps_to_open(self) -> None:
        config = SandboxConfig(provider="registered-fake", block_network=False)
        assert config.network.mode == "open"
        assert config.block_network is False

    def test_legacy_block_network_rejects_unknown_string(self) -> None:
        # A typo must fail loudly, not silently open egress on a network-blocked sandbox.
        with pytest.raises(ValidationError):
            SandboxConfig(provider="registered-fake", block_network="flase")

    def test_explicit_network_wins_over_legacy_block_network(self) -> None:
        config = SandboxConfig(provider="registered-fake", block_network=True, network={"mode": "open"})
        assert config.network.mode == "open"
        assert config.block_network is False

    def test_allowlist_requires_hosts(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            SandboxConfig(provider="registered-fake", network={"mode": "allowlist"})

    def test_allowlist_requires_capability(self) -> None:
        # _RegisteredFake declares neither network policy nor allowlist support.
        config = SandboxConfig(provider="registered-fake", network={"mode": "allowlist", "allow": ["pypi.org"]})
        with pytest.raises(CapabilityError, match="allowlist"):
            create_sandbox_backend(config, "job-1")

    def test_allowlist_passes_when_capability_declared(self) -> None:
        config = SandboxConfig(provider="registered-fake", network={"mode": "allowlist", "allow": ["pypi.org"]})
        verify_capabilities(config, SandboxCapabilities(supports_network_allowlist=True))


class TestResourceLimits:
    """Opt-in CPU/memory caps, gated fail-closed (SANDBOX-5)."""

    def test_unset_resources_run_on_any_provider(self) -> None:
        # Non-breaking: no limits requested -> no resource gate, regardless of capability.
        verify_capabilities(_fake_config(), SandboxCapabilities())

    def test_resource_limit_requires_capability(self) -> None:
        # A requested limit on a provider that can't enforce it must fail closed.
        config = _fake_config(resources={"cpu": 2})
        with pytest.raises(CapabilityError, match="resource limits"):
            verify_capabilities(config, SandboxCapabilities())

    def test_resource_limit_passes_when_declared(self) -> None:
        config = _fake_config(resources={"memory_mb": 2048})
        verify_capabilities(config, SandboxCapabilities(supports_resource_limits=True))

    def test_resource_limit_rejects_non_positive(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(provider="registered-fake", resources={"cpu": 0})


class TestEntryPointDiscovery:
    def test_entry_point_provider_is_discovered(self, monkeypatch: Any) -> None:
        from deep_researcher_agent.agents.deep_researcher.sandbox import registry

        class _EntryPointProvider(_RegisteredFake):
            provider_name = "ep-fake"

        class _FakeEntryPoint:
            name = "ep-fake"

            def load(self) -> type[SandboxProvider]:
                return _EntryPointProvider

        def _fake_entry_points(*, group: str) -> list[Any]:
            assert group == registry.SANDBOX_PROVIDER_ENTRY_POINT_GROUP
            return [_FakeEntryPoint()]

        monkeypatch.setattr(registry, "_entry_points_loaded", False)
        monkeypatch.setattr("importlib.metadata.entry_points", _fake_entry_points)
        registry._SANDBOX_PROVIDERS.pop("ep-fake", None)
        try:
            assert registry.is_registered("ep-fake")
            assert "ep-fake" in registered_providers()
        finally:
            registry._SANDBOX_PROVIDERS.pop("ep-fake", None)

    def test_broken_entry_point_is_skipped(self, monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
        from deep_researcher_agent.agents.deep_researcher.sandbox import registry

        class _BrokenEntryPoint:
            name = "broken"

            def load(self) -> type[SandboxProvider]:
                raise RuntimeError("credential=do-not-log")

        monkeypatch.setattr(registry, "_entry_points_loaded", False)
        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: [_BrokenEntryPoint()])
        # Must not raise; built-in resolution stays intact.
        with caplog.at_level(logging.WARNING):
            registry._load_entry_point_providers()
        assert "broken" not in registry._SANDBOX_PROVIDERS
        assert "modal" in registered_providers()
        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text


class TestProviderLifecycle:
    def test_event_emitter_failure_is_sanitized(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _RegisteredFake(_fake_config(), "job-1")

        def fail(_event: dict[str, object]) -> None:
            raise RuntimeError("credential=do-not-log")

        provider.set_event_emitter(fail)
        with caplog.at_level(logging.WARNING):
            provider._emit_event({"type": "test"})

        assert "event_emit_failed" in caplog.text
        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text

    def test_session_created_lazily(self) -> None:
        session = MagicMock()
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[session])
        assert provider.sessions_created == 0
        provider.execute("echo ok", timeout=5)
        assert provider.sessions_created == 1
        session.execute.assert_called_once_with("echo ok", timeout=5)

    def test_idempotent_download_retries_on_recoverable_error(self) -> None:
        first = MagicMock()
        first.download_files.side_effect = _RecoverableError("gone")
        second = MagicMock()
        second.download_files.return_value = ["ok"]
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[first, second])

        result = provider.download_files(["/workspace/a.png"])

        assert result == ["ok"]
        assert provider.sessions_created == 2  # recreated once

    def test_execute_does_not_retry_on_recoverable_error(self) -> None:
        first = MagicMock()
        first.execute.side_effect = _RecoverableError("gone")
        second = MagicMock()
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[first, second])

        with pytest.raises(_RecoverableError):
            provider.execute("echo ok")

        # Non-idempotent op must NOT silently recreate + re-run on a fresh empty sandbox.
        assert provider.sessions_created == 1

    def test_execute_timeout_clamped_to_config_limit(self) -> None:
        session = MagicMock()
        session.execute.return_value = "ok"
        provider = _ScriptedProvider(_fake_config(timeout=1200), "job-1", sessions=[session])

        provider.execute("echo ok", timeout=120000)  # e.g. a tool passing milliseconds

        session.execute.assert_called_once_with("echo ok", timeout=1200)

    def test_execute_timeout_passthrough_when_within_limit(self) -> None:
        session = MagicMock()
        session.execute.return_value = "ok"
        provider = _ScriptedProvider(_fake_config(timeout=1200), "job-1", sessions=[session])

        provider.execute("echo ok", timeout=30)

        session.execute.assert_called_once_with("echo ok", timeout=30)

    def test_close_releases_session(self) -> None:
        session = MagicMock()
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[session])
        provider.execute("echo ok")
        provider.close()
        session.close.assert_called_once()
        assert provider._session is None

    def test_session_close_failure_is_sanitized(self, caplog: pytest.LogCaptureFixture) -> None:
        session = MagicMock()
        session.close.side_effect = RuntimeError("credential=do-not-log")
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[session])
        provider._session = session

        with caplog.at_level(logging.WARNING):
            provider.close()

        assert provider.cleanup_failure_reason_codes == ("session_close_failed",)
        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text

    def test_terminate_releases_session_and_blocks_further_ops(self) -> None:
        session = MagicMock()
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[session])
        provider.execute("echo ok")

        provider.terminate()

        session.close.assert_called_once()
        assert provider._session is None
        # A terminated provider refuses new work instead of silently recreating.
        with pytest.raises(SandboxTerminatedError):
            provider.execute("echo again")
        assert provider.sessions_created == 1

    def test_terminate_preempts_in_flight_execute(self) -> None:
        # terminate() must not wait on the operation lock: while execute() is blocked in a
        # remote call, a concurrent terminate() closes the session out-of-band, which
        # interrupts the call. Without the two-lock split this test would deadlock/hang.
        import threading

        entered = threading.Event()
        released = threading.Event()

        session = MagicMock()

        def _blocking_execute(command: str, timeout: int | None = None) -> str:
            entered.set()
            if not released.wait(timeout=5):
                raise AssertionError("execute was not interrupted by terminate()")
            raise RuntimeError("session closed")

        session.execute.side_effect = _blocking_execute
        session.close.side_effect = lambda: released.set()
        provider = _ScriptedProvider(_fake_config(), "job-1", sessions=[session])

        result: dict[str, BaseException] = {}

        def _run_execute() -> None:
            try:
                provider.execute("sleep")
            except BaseException as exc:  # noqa: BLE001 - capture for assertion
                result["error"] = exc

        worker = threading.Thread(target=_run_execute)
        worker.start()
        assert entered.wait(timeout=5), "execute never started"

        provider.terminate()  # must return without waiting for the in-flight execute
        worker.join(timeout=5)

        assert not worker.is_alive(), "execute did not unblock after terminate()"
        session.close.assert_called_once()
        assert isinstance(result.get("error"), Exception)


class TestJobScopedPaths:
    """The per-job workspace helper isolates jobs within a shared/reused sandbox."""

    def test_workdir_and_artifact_dir_are_job_scoped(self) -> None:
        assert job_scoped_workdir("/sandbox", "abc") == "/sandbox/abc"
        assert job_scoped_artifact_dir("/sandbox", "abc") == "/sandbox/abc/deep-researcher-artifacts"

    def test_trailing_slash_normalized(self) -> None:
        assert job_scoped_workdir("/sandbox/", "abc") == "/sandbox/abc"

    def test_unsafe_job_id_cannot_escape_base(self) -> None:
        # A crafted id with path separators must collapse to a single safe segment so it
        # cannot move the workspace (and harvest confinement root) outside the base.
        result = job_scoped_workdir("/sandbox", "../../etc")
        assert result.startswith("/sandbox/")
        assert ".." not in result
        assert result.count("/") == 2


class TestWorkspacePreparation:
    """The provider base creates the per-job workspace on session start (mkdir -p)."""

    def test_provider_paths_are_job_scoped(self) -> None:
        provider = _RegisteredFake(_fake_config(workdir="/sandbox"), "job-xyz")
        assert provider.workdir == "/sandbox/job-xyz"
        assert provider.artifact_dir == "/sandbox/job-xyz/deep-researcher-artifacts"

    def test_workspace_created_on_session_start(self) -> None:
        session = MagicMock()
        session.execute.return_value = "ok"
        provider = _WorkspaceProvider(_fake_config(workdir="/sandbox"), "job-xyz", session)

        provider.execute("echo hi")

        # The first execute after creation is the idempotent mkdir -p of the per-job roots.
        mkdir_cmd = session.execute.call_args_list[0].args[0]
        assert mkdir_cmd.startswith("mkdir -p")
        assert "/sandbox/job-xyz" in mkdir_cmd
        assert "/sandbox/job-xyz/deep-researcher-artifacts" in mkdir_cmd

    def test_workspace_prep_failure_does_not_abort_the_call(self, caplog: pytest.LogCaptureFixture) -> None:
        # Best-effort: a mkdir failure is swallowed so it cannot break session creation;
        # a real filesystem problem resurfaces on the first actual write.
        session = MagicMock()
        session.execute.side_effect = [RuntimeError("credential=do-not-log"), "ok"]
        provider = _WorkspaceProvider(_fake_config(workdir="/sandbox"), "job-1", session)

        with caplog.at_level(logging.WARNING):
            assert provider.execute("echo hi") == "ok"

        assert "workspace_prepare_failed" in caplog.text
        assert "credential=do-not-log" not in caplog.text
