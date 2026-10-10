# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig

from aiq_harbor import runner
from aiq_harbor.agent import AiqHarborAgent


class FakeEnvironment:
    default_user: str | int | None = None

    def __init__(
        self,
        root: Path,
        *,
        fail_run: bool = False,
        cancel_run: bool = False,
        task_env: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        self.fail_run = fail_run
        self.cancel_run = cancel_run
        self.task_env_config = SimpleNamespace(env=task_env or {})
        self.exec_calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.exec_calls.append({"command": command, **kwargs})
        if "aiq_runner.py run" in command:
            if self.cancel_run:
                raise asyncio.CancelledError
            if self.fail_run:
                return SimpleNamespace(stdout="", stderr="workflow failed for tavily-secret", return_code=7)
        if "m.version" in command:
            return SimpleNamespace(stdout="2.2.0\n", stderr="", return_code=0)
        return SimpleNamespace(stdout="ok\n", stderr="", return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        target = self.root / target_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        self.uploads.append((source, target_path))


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yml"
    path.write_text("workflow:\n  _type: shallow_research_workflow\n", encoding="utf-8")
    return path


def _agent(tmp_path: Path, **kwargs: Any) -> AiqHarborAgent:
    return AiqHarborAgent(
        logs_dir=tmp_path / "logs",
        config_file=_config(tmp_path),
        extra_env={"NVIDIA_API_KEY": "nim-secret", "TAVILY_API_KEY": "tavily-secret"},  # pragma: allowlist secret
        **kwargs,
    )


def test_agent_factory_loads_custom_import_path(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = AgentConfig(
        import_path="aiq_harbor.agent:AiqHarborAgent",
        kwargs={
            "config_file": str(config_path),
        },
        env={"NVIDIA_API_KEY": "factory-secret"},  # pragma: allowlist secret
    )

    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path / "factory-logs")

    assert isinstance(agent, AiqHarborAgent)
    assert agent.name() == "aiq-harbor"
    assert agent.extra_env["NVIDIA_API_KEY"] == "factory-secret"  # pragma: allowlist secret


def test_checked_in_job_config_uses_migrated_agent() -> None:
    raw = yaml.safe_load(Path("configs/deepsearchqa.yaml").read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)
    profile_path = Path(config.agents[0].kwargs["config_file"])
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    research_llm = profile["llms"]["research_llm"]

    assert len(config.agents) == 1
    assert config.agents[0].import_path == "aiq_harbor.agent:AiqHarborAgent"
    assert profile_path == Path("configs/aiq/shallow_agent.yml")
    assert profile_path.is_file()
    assert research_llm["model_name"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert research_llm["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert research_llm["api_key"] == "${NVIDIA_API_KEY}"
    assert config.agents[0].env["NVIDIA_API_KEY"] == "${NVIDIA_API_KEY}"
    assert config.datasets[0].path == Path("datasets/deepsearchqa")


def test_deepresearch_bench_job_config_uses_direct_deep_research() -> None:
    raw = yaml.safe_load(Path("configs/deepresearch_bench_ii.yaml").read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)
    profile_path = Path(config.agents[0].kwargs["config_file"])
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    assert len(config.agents) == 1
    assert config.agents[0].import_path == "aiq_harbor.agent:AiqHarborAgent"
    assert config.agents[0].kwargs["output_file"] == "/workspace/report.md"
    assert profile_path == Path("configs/aiq/deep_agent.yml")
    assert profile["workflow"]["_type"] == "deep_research_workflow"
    assert profile["functions"]["deep_research_agent"]["tools"] == ["web_search_tool", "paper_search_tool"]
    assert config.timeout_multiplier == 1.0
    assert config.datasets[0].path == Path("datasets/deepresearch-bench-ii")


def test_setup_uses_preinstalled_runtime_and_uploaded_config(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    environment = FakeEnvironment(tmp_path / "remote")

    asyncio.run(agent.setup(environment))

    uploaded_config = tmp_path / "remote/installed-agent/aiq-agent-config.yml"
    assert uploaded_config.read_bytes() == agent._config_file.read_bytes()
    assert (tmp_path / "remote/installed-agent/aiq_runner.py").is_file()
    assert agent.version() == "2.2.0"

    commands = [call["command"] for call in environment.exec_calls]
    assert any("aiq_runner.py validate" in command for command in commands)
    assert all("pip install" not in command and "uv pip" not in command for command in commands)


def test_instruction_is_uploaded_without_shell_interpolation(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._validate_host_config()
    environment = FakeEnvironment(tmp_path / "remote")
    instruction = "line one\n'AIQEOF'\n$(touch /tmp/not-executed)\n`id`\n"

    asyncio.run(agent.run(instruction, environment, AgentContext()))

    uploaded = (tmp_path / "remote/installed-agent/instruction.txt").read_text(encoding="utf-8")
    run_command = next(call["command"] for call in environment.exec_calls if "aiq_runner.py run" in call["command"])
    assert uploaded == instruction
    assert instruction not in run_command
    assert "--instruction-file" in run_command
    assert "--events-output /logs/agent/aiq_events.jsonl" in run_command


@pytest.mark.parametrize("output_file", ["/workspace/report.md", "/workspace/answer.txt"])
def test_auto_output_file_uses_task_environment(tmp_path: Path, output_file: str) -> None:
    agent = _agent(tmp_path, output_file="auto")
    agent._validate_host_config()
    environment = FakeEnvironment(tmp_path / "remote", task_env={"AIQ_OUTPUT_FILE": output_file})

    asyncio.run(agent.run("prompt", environment, AgentContext()))

    run_command = next(call["command"] for call in environment.exec_calls if "aiq_runner.py run" in call["command"])
    assert f"--output-file {output_file}" in run_command


@pytest.mark.parametrize(
    "task_env",
    [
        {},
        {"AIQ_OUTPUT_FILE": "/tmp/report.md"},
        {"AIQ_OUTPUT_FILE": "/workspace/result.json"},
    ],
)
def test_auto_output_file_rejects_invalid_task_selection(tmp_path: Path, task_env: dict[str, str]) -> None:
    agent = _agent(tmp_path, output_file="auto")
    agent._validate_host_config()
    environment = FakeEnvironment(tmp_path / "remote", task_env=task_env)

    with pytest.raises(RuntimeError, match="AIQ_OUTPUT_FILE"):
        asyncio.run(agent.run("prompt", environment, AgentContext()))


def test_nonzero_runner_exit_is_classified_and_redacted(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._validate_host_config()
    environment = FakeEnvironment(tmp_path / "remote", fail_run=True)

    with pytest.raises(NonZeroAgentExitCodeError, match="workflow failed") as exc_info:
        asyncio.run(agent.run("prompt", environment, AgentContext()))

    assert "tavily-secret" not in str(exc_info.value)
    persisted = tmp_path / "remote/logs/agent/aiq-agent-stderr.txt"
    assert persisted.read_text(encoding="utf-8") == "workflow failed for ${TAVILY_API_KEY}"


def test_cancellation_terminates_runner(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._validate_host_config()
    environment = FakeEnvironment(tmp_path / "remote", cancel_run=True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.run("prompt", environment, AgentContext()))

    assert any("kill -TERM" in call["command"] for call in environment.exec_calls)


@pytest.mark.parametrize(
    "output_file",
    ["answer.txt", "/tmp/answer.txt", "/workspace/nested/answer.txt", "/workspace/output.json"],
)
def test_output_file_is_restricted(tmp_path: Path, output_file: str) -> None:
    with pytest.raises(ValueError, match="output_file"):
        _agent(tmp_path, output_file=output_file)


def test_populate_context_from_synced_sidecars(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.logs_dir.mkdir(parents=True)
    (agent.logs_dir / "aiq_job.json").write_text(
        json.dumps({"status": "completed", "output_chars": 3}),
        encoding="utf-8",
    )
    (agent.logs_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "final_metrics": {
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 3,
                    "total_cached_tokens": 2,
                    "total_cost_usd": 0.012,
                }
            }
        ),
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.metadata == {"aiq": {"status": "completed", "output_chars": 3}}
    assert context.n_input_tokens == 10
    assert context.n_output_tokens == 3
    assert context.n_cache_tokens == 2
    assert context.cost_usd == pytest.approx(0.012)


def test_runner_filters_duplicate_function_and_nested_tool_spans() -> None:
    def step(event_type: str, parent_id: str, function_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            event_type=SimpleNamespace(value=event_type),
            parent_id=parent_id,
            function_ancestry=SimpleNamespace(function_id=function_id),
        )

    steps = [
        step("LLM_END", "agent", "research"),
        step("TOOL_START", "agent", "research"),
        step("FUNCTION_START", "tool", "implementation"),
        step("TOOL_START", "tool", "implementation"),
        step("TOOL_END", "tool", "implementation"),
        step("FUNCTION_END", "tool", "implementation"),
        step("TOOL_END", "agent", "research"),
    ]

    filtered = runner._trajectory_steps_for_atif(steps)

    assert [item.event_type.value for item in filtered] == ["LLM_END", "TOOL_START", "TOOL_END"]


def test_live_progress_persists_counts_without_content(tmp_path: Path) -> None:
    state_path = tmp_path / "aiq_state.json"
    events_path = tmp_path / "aiq_events.jsonl"
    progress = runner._LiveProgress(
        state_path=state_path,
        events_path=events_path,
        session_id="session",
        started_at_epoch=runner.time.time(),
    )
    secret_content = "content-that-must-not-be-persisted"  # pragma: allowlist secret

    for event_type in ("LLM_START", "LLM_END"):
        progress.observe(
            SimpleNamespace(
                event_type=SimpleNamespace(value=event_type),
                event_timestamp=runner.time.time(),
                name="model",
                UUID="llm-call",
                parent_id="agent",
                function_ancestry=SimpleNamespace(function_id="research"),
                data=SimpleNamespace(input=secret_content, output=secret_content),
                metadata=SimpleNamespace(tool_inputs={"query": secret_content}),
            )
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    persisted = state_path.read_text(encoding="utf-8") + events_path.read_text(encoding="utf-8")
    assert state["calls"]["llm"] == {"started": 1, "completed": 1, "in_flight": 0}
    assert secret_content not in persisted


def test_runner_records_deep_profile_runtime_settings() -> None:
    assert {"parallel_tool_calls", "chat_template_kwargs"} <= set(runner._LLM_RUNTIME_FIELDS)
    assert {"paper_search", "deep_research_agent"} <= runner._FUNCTION_RUNTIME_FIELDS.keys()
    assert {"provider", "timeout", "max_results"} <= set(runner._FUNCTION_RUNTIME_FIELDS["paper_search"])
    assert {"orchestrator_llm", "tools", "enable_citation_verification"} <= set(
        runner._FUNCTION_RUNTIME_FIELDS["deep_research_agent"]
    )


def test_runner_parser_exposes_supported_commands() -> None:
    parser = runner.build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")

    assert set(subparsers.choices) == {"validate", "run"}
