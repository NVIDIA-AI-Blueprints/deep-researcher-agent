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

"""Tests for DeepAgents runtime config, skill collection resolution, and backend wiring."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from threading import Event
from threading import Thread
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from deepagents.backends import CompositeBackend
from deepagents.backends import FilesystemBackend
from deepagents.backends import StateBackend
from pydantic import ValidationError

from aiq_agent.agents.deep_researcher.deepagents_runtime import BUILTIN_SKILL_SOURCE
from aiq_agent.agents.deep_researcher.deepagents_runtime import SHARED_ROUTE
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSkillsConfig
from aiq_agent.agents.deep_researcher.deepagents_runtime import _create_sandbox_backend
from aiq_agent.agents.deep_researcher.deepagents_runtime import discover_skill_collections
from aiq_agent.agents.deep_researcher.deepagents_runtime import resolve_skill_collections

SYNTHESIS_SKILL_SOURCE = f"{BUILTIN_SKILL_SOURCE}synthesis/"


class TestSkillCollections:
    """Public skill config uses collection names, not DeepAgents virtual paths."""

    def test_builtin_skill_collections_are_discovered(self) -> None:
        collections = discover_skill_collections()

        assert collections["research"] == "/skills/research/"
        assert collections["synthesis"] == "/skills/synthesis/"
        assert collections["visualization"] == "/skills/visualization/"

    def test_nested_skill_collections_are_discovered(self, tmp_path) -> None:
        skill_dir = tmp_path / "finance" / "earnings" / "quarterly-summary"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: quarterly-summary\n---\n", encoding="utf-8")

        assert discover_skill_collections(tmp_path) == {"finance/earnings": "/skills/finance/earnings/"}

    def test_resolve_skill_collections_maps_names_to_sources(self) -> None:
        assert resolve_skill_collections(("synthesis",)) == ("/skills/synthesis/",)

    def test_resolve_skill_collections_rejects_unknown_names(self) -> None:
        with pytest.raises(ValueError, match="Unknown deep research skill collection"):
            resolve_skill_collections(("typo",))

    def test_skills_config_forbids_old_fields(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DeepResearchSkillsConfig(enabled=True)


class TestDeepAgentsRuntimeRouting:
    """Runtime uses stock DeepAgents backends with only the routes required."""

    def test_baseline_uses_plain_state_backend(self) -> None:
        runtime = DeepAgentsRuntime()

        assert isinstance(runtime.backend, StateBackend)
        assert runtime.execution_enabled is False
        assert runtime.skills_enabled is False

    def test_skills_only_adds_skills_route(self) -> None:
        runtime = DeepAgentsRuntime(
            skills=DeepResearchSkillsConfig(agents={"writer-agent": ("synthesis",)}),
        )
        backend = runtime.backend

        assert isinstance(backend, CompositeBackend)
        assert isinstance(backend.default, StateBackend)
        assert set(backend.routes) == {BUILTIN_SKILL_SOURCE}
        assert isinstance(backend.routes[BUILTIN_SKILL_SOURCE], FilesystemBackend)
        assert runtime.skill_sources_for("writer-agent") == [SYNTHESIS_SKILL_SOURCE]

    def test_agent_has_chart_skill_tracks_visualization_collection(self) -> None:
        runtime = DeepAgentsRuntime(
            skills=DeepResearchSkillsConfig(
                agents={"writer-agent": ("visualization",), "researcher-agent": ("synthesis",)},
            ),
        )

        assert runtime.agent_has_chart_skill("writer-agent") is True
        assert runtime.agent_has_chart_skill("researcher-agent") is False
        assert runtime.agent_has_chart_skill("planner-agent") is False

    def test_sandbox_only_adds_shared_route(self) -> None:
        fake_sandbox = MagicMock()
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=fake_sandbox,
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig())
            backend = runtime.backend

        assert isinstance(backend, CompositeBackend)
        assert backend.default is fake_sandbox
        assert set(backend.routes) == {SHARED_ROUTE}
        assert runtime.execution_enabled is True
        assert runtime.skills_enabled is False

    def test_prepare_state_files_preserves_shared_paths_without_route(self) -> None:
        runtime = DeepAgentsRuntime()

        files = runtime.prepare_state_files({"/shared/original_report.md": "# Parent"})

        assert "/shared/original_report.md" in files
        assert files["/shared/original_report.md"]["content"] == "# Parent"
        assert "modified_at" in files["/shared/original_report.md"]

    def test_prepare_state_files_normalizes_shared_paths_for_route_backend(self) -> None:
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=MagicMock(),
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig())

        files = runtime.prepare_state_files(
            {
                "/shared/original_report.md": "# Parent",
                "/shared/source_summary.md": b"- src",
            }
        )

        assert "/original_report.md" in files
        assert "/source_summary.md" in files
        assert "/shared/original_report.md" not in files
        assert "/shared/source_summary.md" not in files
        assert files["/original_report.md"]["content"] == "# Parent"
        assert files["/source_summary.md"]["content"] == "- src"
        assert "modified_at" in files["/original_report.md"]

    def test_sandbox_and_skills_add_shared_and_skills_routes(self) -> None:
        fake_sandbox = MagicMock()
        skills = DeepResearchSkillsConfig(agents={"researcher-agent": ("research",)})
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=fake_sandbox,
        ):
            runtime = DeepAgentsRuntime(skills=skills, sandbox=DeepResearchSandboxConfig())
            backend = runtime.backend

        assert isinstance(backend, CompositeBackend)
        assert backend.default is fake_sandbox
        assert set(backend.routes) == {BUILTIN_SKILL_SOURCE, SHARED_ROUTE}
        assert isinstance(backend.routes[BUILTIN_SKILL_SOURCE], FilesystemBackend)
        assert isinstance(backend.routes[SHARED_ROUTE], StateBackend)
        assert runtime.skill_sources_for("researcher-agent") == ["/skills/research/"]

    def test_require_sandbox_collection_without_sandbox_raises(self) -> None:
        skills = DeepResearchSkillsConfig(
            agents={"researcher-agent": ("research",)},
            require_sandbox=("research",),
        )

        with pytest.raises(ValueError, match="require a sandbox"):
            DeepAgentsRuntime(skills=skills)

    def test_skills_config_rejects_unknown_agent_keys(self) -> None:
        with pytest.raises(ValidationError, match="Unknown deep research skill agent"):
            DeepResearchSkillsConfig(agents={"researcher": ("research",)})

    def test_require_sandbox_collection_with_sandbox_is_allowed(self) -> None:
        skills = DeepResearchSkillsConfig(
            agents={"researcher-agent": ("research",)},
            require_sandbox=("research",),
        )
        # Patch backend creation so the test does not require the optional OpenShell adapter
        # (the default provider) to be installed.
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=MagicMock(),
        ):
            runtime = DeepAgentsRuntime(skills=skills, sandbox=DeepResearchSandboxConfig())

        assert runtime.skill_sources_for("researcher-agent") == ["/skills/research/"]

    def test_require_sandbox_collection_with_sandbox_still_rejects_unknown_names(self) -> None:
        skills = DeepResearchSkillsConfig(
            agents={"researcher-agent": ("research",)},
            require_sandbox=("typo",),
        )

        with pytest.raises(ValueError, match="Unknown deep research skill collection"):
            DeepAgentsRuntime(skills=skills, sandbox=DeepResearchSandboxConfig())

    def test_deepagents_skill_scanner_finds_synthesis_skills_from_nested_source(self) -> None:
        from deepagents.middleware.skills import _list_skills

        runtime = DeepAgentsRuntime(skills=DeepResearchSkillsConfig(agents={"writer-agent": ("synthesis",)}))
        backend = runtime.backend

        top_level_skills = _list_skills(backend, BUILTIN_SKILL_SOURCE)
        synthesis_skills = _list_skills(backend, SYNTHESIS_SKILL_SOURCE)

        assert [skill["name"] for skill in top_level_skills] == []
        assert [skill["name"] for skill in synthesis_skills] == [
            "long-form-report-writer",
            "prediction-report-writer",
        ]
        assert [skill["path"] for skill in synthesis_skills] == [
            "/skills/synthesis/long-form-report-writer/SKILL.md",
            "/skills/synthesis/prediction-report-writer/SKILL.md",
        ]

    def test_deepagents_subagent_skills_key_adds_skills_middleware(self) -> None:
        from deepagents import create_deep_agent

        fake_graph = MagicMock()
        fake_graph.with_config.return_value = fake_graph
        create_agent_calls: list[dict[str, Any]] = []

        def fake_create_agent(*_args: Any, **kwargs: Any) -> MagicMock:
            create_agent_calls.append(kwargs)
            return fake_graph

        fake_model = MagicMock()
        profile = MagicMock()
        profile.tool_description_overrides = {}
        profile.excluded_tools = []
        profile.excluded_middleware = []
        profile.materialize_extra_middleware.return_value = []
        profile.general_purpose_subagent = MagicMock(enabled=False)
        profile.base_system_prompt = None
        profile.system_prompt_suffix = None

        with (
            patch("deepagents.graph.resolve_model", return_value=fake_model),
            patch("deepagents._models.resolve_model", return_value=fake_model),
            patch("deepagents.graph._harness_profile_for_model", return_value=profile),
            patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
            patch("deepagents.graph.create_agent", side_effect=fake_create_agent),
            patch("deepagents.middleware.subagents.create_agent", side_effect=fake_create_agent),
        ):
            create_deep_agent(
                model=fake_model,
                tools=[],
                subagents=[
                    {
                        "name": "writer-agent",
                        "description": "Writes the final answer.",
                        "system_prompt": "Write.",
                        "skills": [SYNTHESIS_SKILL_SOURCE],
                    }
                ],
                backend=StateBackend(),
                skills=[BUILTIN_SKILL_SOURCE],
            )

        writer_middleware = create_agent_calls[0]["middleware"]
        writer_skills = [m for m in writer_middleware if m.__class__.__name__ == "SkillsMiddleware"]
        assert len(writer_skills) == 1
        assert writer_skills[0].sources == [SYNTHESIS_SKILL_SOURCE]


class TestDeepAgentsRuntimeJobId:
    """job_id should drive the sandbox name; a missing one falls back to uuid."""

    def test_default_sandbox_provider_is_openshell(self) -> None:
        sandbox = DeepResearchSandboxConfig()

        assert sandbox.provider == "openshell"
        assert sandbox.workdir is None

    def test_explicit_job_id_is_kept(self) -> None:
        sandbox = DeepResearchSandboxConfig()
        with patch("aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend") as create_backend:
            runtime = DeepAgentsRuntime(sandbox=sandbox, job_id="job-abc-123")
            _ = runtime.backend

        create_backend.assert_called_once_with(sandbox, "job-abc-123")
        assert runtime.sandbox_backend is create_backend.return_value

    def test_missing_job_id_generates_uuid(self) -> None:
        sandbox_a = DeepResearchSandboxConfig()
        sandbox_b = DeepResearchSandboxConfig()
        with patch("aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend") as create_backend:
            runtime_a = DeepAgentsRuntime(sandbox=sandbox_a)
            runtime_b = DeepAgentsRuntime(sandbox=sandbox_b)
            _ = runtime_a.backend
            _ = runtime_b.backend

        job_id_a = create_backend.call_args_list[0].args[1]
        job_id_b = create_backend.call_args_list[1].args[1]
        assert len(job_id_a) == 36
        assert job_id_a != job_id_b

    def test_modal_sandbox_config_requires_provider_dependencies(self) -> None:
        def find_spec(module_name: str):
            if module_name == "langchain_modal":
                return None
            return object()

        with (
            patch(
                "aiq_agent.agents.deep_researcher.deepagents_runtime.importlib.util.find_spec", side_effect=find_spec
            ),
            pytest.raises(ImportError, match="langchain-modal"),
        ):
            _ = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig(provider="modal")).backend

    def test_public_openshell_config_maps_isolation_and_attestation(self) -> None:
        public = DeepResearchSandboxConfig(
            policy="policy.yaml",
            openshell_image="aiq:test",
            workspace="research",
            attest=True,
            expected_policy_version=3,
            policy_load_timeout_seconds=17,
            network="allowlist",
            network_allow=("api.github.com",),
        )

        with patch(
            "aiq_agent.agents.deep_researcher.sandbox.create_sandbox_backend",
            return_value="backend",
        ) as create:
            assert _create_sandbox_backend(public, "job-1") == "backend"

        resolved = create.call_args.args[0]
        assert resolved.network.mode == "allowlist"
        assert resolved.network.allow == ("api.github.com",)
        assert resolved.providers.openshell.image == "aiq:test"
        assert resolved.providers.openshell.workspace == "research"
        assert resolved.providers.openshell.attest is True
        assert resolved.providers.openshell.expected_policy_version == 3
        assert resolved.providers.openshell.policy_load_timeout_seconds == 17
        assert resolved.providers.openshell.delete_on_exit is True

    def test_public_allowlist_requires_hosts(self) -> None:
        with pytest.raises(ValueError, match="network_allow"):
            DeepResearchSandboxConfig(network="allowlist")

    @pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
    def test_public_ready_timeout_must_be_positive_and_finite(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="ready_timeout_seconds"):
            DeepResearchSandboxConfig(ready_timeout_seconds=timeout)


class TestDeepAgentsRuntimeArtifacts:
    """Terminal artifact harvesting is safe on normal and interrupted paths."""

    def test_final_harvest_logs_only_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = MagicMock()
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig())
        runtime.artifact_manager = MagicMock()
        runtime.artifact_manager.final_harvest.side_effect = RuntimeError("credential=do-not-log")

        with caplog.at_level(logging.WARNING):
            runtime.final_harvest()

        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text

    def test_normal_finalize_artifacts_harvests(self) -> None:
        provider = MagicMock()
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig())
        runtime.artifact_manager = MagicMock()

        assert runtime.finalize_artifacts(interrupted=False) is True
        assert runtime.finalize_artifacts(interrupted=False) is False
        runtime.artifact_manager.final_harvest.assert_called_once_with()

    @pytest.mark.parametrize(("lease_acquired", "expected"), [(True, True), (False, False)])
    def test_interrupted_finalize_harvests_only_when_provider_is_idle(
        self,
        lease_acquired: bool,
        expected: bool,
    ) -> None:
        provider = MagicMock()
        provider.try_operation_lease.return_value = nullcontext(lease_acquired)
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig())
        runtime.artifact_manager = MagicMock()

        assert runtime.finalize_artifacts(interrupted=True) is expected
        assert runtime.artifact_manager.final_harvest.call_count == int(lease_acquired)


class TestDeepAgentsRuntimeCleanup:
    """Terminal cleanup is idempotent and reports the provider's actual outcome."""

    def test_finalize_closes_once_and_emits_success(self) -> None:
        provider = MagicMock()
        provider.provider_name = "openshell"
        provider.physical_sandbox_name = "sandbox-1"
        provider.cleanup_succeeded = True
        events: list[dict[str, object]] = []
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(
                sandbox=DeepResearchSandboxConfig(),
                artifact_emit=events.append,
            )

        assert runtime.finalize(interrupted=False) is True
        assert runtime.finalize(interrupted=False) is True
        provider.close.assert_called_once_with()
        assert provider.terminate.call_count == 0
        assert [event["data"]["status"] for event in events] == ["started", "succeeded"]  # type: ignore[index]

    def test_finalize_without_provider_emits_no_cleanup_events(self) -> None:
        events: list[dict[str, object]] = []
        runtime = DeepAgentsRuntime(sandbox=None, artifact_emit=events.append)

        assert runtime.finalize(interrupted=False) is True
        assert runtime.finalize(interrupted=True) is True
        assert events == []

    def test_finalize_emits_failed_when_provider_observed_cleanup_error(self) -> None:
        provider = MagicMock()
        provider.provider_name = "openshell"
        provider.sandbox_name = "logical-job"
        provider.physical_sandbox_name = None
        provider.cleanup_succeeded = False
        events: list[dict[str, object]] = []
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(
                sandbox=DeepResearchSandboxConfig(),
                artifact_emit=events.append,
            )

        assert runtime.finalize(interrupted=True) is False
        provider.terminate.assert_called_once_with()
        assert [event["data"]["status"] for event in events] == ["started", "failed"]  # type: ignore[index]

    def test_finalize_logs_only_cleanup_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = MagicMock()
        provider.provider_name = "openshell"
        provider.physical_sandbox_name = "sandbox-1"
        provider.close.side_effect = RuntimeError("credential=do-not-log")
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(sandbox=DeepResearchSandboxConfig())

        with caplog.at_level(logging.WARNING):
            assert runtime.finalize(interrupted=False) is False

        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text

    def test_concurrent_finalize_waits_for_and_reuses_exact_result(self) -> None:
        provider = MagicMock()
        provider.provider_name = "openshell"
        provider.physical_sandbox_name = "sandbox-1"
        provider.cleanup_succeeded = True
        cleanup_started = Event()
        allow_cleanup = Event()
        second_caller_started = Event()
        events: list[dict[str, object]] = []
        results: list[bool] = []

        def close() -> None:
            cleanup_started.set()
            if not allow_cleanup.wait(timeout=2):
                raise AssertionError("cleanup was not released")
            provider.cleanup_succeeded = False

        provider.close.side_effect = close
        with patch(
            "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
            return_value=provider,
        ):
            runtime = DeepAgentsRuntime(
                sandbox=DeepResearchSandboxConfig(),
                artifact_emit=events.append,
            )

        first = Thread(target=lambda: results.append(runtime.finalize(interrupted=False)))

        def finalize_again() -> None:
            second_caller_started.set()
            results.append(runtime.finalize(interrupted=True))

        second = Thread(target=finalize_again)
        first.start()
        assert cleanup_started.wait(timeout=2)
        second.start()
        assert second_caller_started.wait(timeout=2)
        assert results == []

        allow_cleanup.set()
        first.join(timeout=2)
        second.join(timeout=2)

        assert not first.is_alive() and not second.is_alive()
        assert results == [False, False]
        provider.close.assert_called_once_with()
        provider.terminate.assert_not_called()
        assert [event["data"]["status"] for event in events] == ["started", "failed"]  # type: ignore[index]
