# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT registration tests for request-scoped OpenShell Python."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aiq_agent.agents.data_science.sandboxed_python import register as register_module
from aiq_agent.agents.data_science.utils.analysis_runtime import begin_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import end_analysis_run
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig


def _builder_with(sandbox: object) -> MagicMock:
    builder = MagicMock()
    builder.get_function_config.return_value = sandbox
    return builder


@pytest.mark.asyncio
async def test_registration_creates_one_openshell_runner_per_analysis_run() -> None:
    config = register_module.SandboxedPythonConfig(sandbox="ds_python_sandbox", wall_timeout_seconds=60)
    builder = _builder_with(DeepResearchSandboxConfig(provider="openshell", network="blocked"))
    registration = register_module.sandboxed_python.__wrapped__(config, builder)
    function_info = await anext(registration)
    token = begin_analysis_run()
    runner = MagicMock()
    runner.execute = AsyncMock(return_value='{"status":"ok","result":"2"}')
    runner.aclose = AsyncMock()
    try:
        with (
            patch.object(register_module, "DeepAgentsRuntime") as runtime_cls,
            patch.object(register_module, "OpenShellPythonRunner", return_value=runner) as runner_cls,
        ):
            first = await function_info.single_fn("1 + 1")
            second = await function_info.single_fn("2 + 2")
    finally:
        await end_analysis_run(token)
        await registration.aclose()

    assert first == '{"status":"ok","result":"2"}'
    assert second == first
    runtime_cls.assert_called_once()
    assert runtime_cls.call_args.kwargs["sandbox"].provider == "openshell"
    assert runtime_cls.call_args.kwargs["job_id"].startswith("data-science-python-")
    runner_cls.assert_called_once()
    assert runner_cls.call_args.kwargs["limits"].max_memory_mb == 8_192
    assert runner_cls.call_args.kwargs["limits"].max_cpu_seconds == 600
    assert runner.execute.await_count == 2
    runner.execute.assert_any_await("1 + 1")
    runner.execute.assert_any_await("2 + 2")
    runner.aclose.assert_awaited_once()


@pytest.mark.parametrize(
    ("sandbox", "message"),
    [
        (DeepResearchSandboxConfig(provider="modal", network="blocked"), "OpenShell"),
        (
            DeepResearchSandboxConfig(provider="openshell", network="allowlist", network_allow=["example.com"]),
            "blocked",
        ),
        (
            DeepResearchSandboxConfig(
                provider="openshell",
                network="blocked",
                existing_sandbox_name="debug",
                allow_shared_sandbox=True,
            ),
            "fresh per-request",
        ),
    ],
)
@pytest.mark.asyncio
async def test_registration_rejects_nonisolated_sandbox_configuration(
    sandbox: DeepResearchSandboxConfig,
    message: str,
) -> None:
    config = register_module.SandboxedPythonConfig(sandbox="ds_python_sandbox")
    registration = register_module.sandboxed_python.__wrapped__(config, _builder_with(sandbox))

    with pytest.raises(ValueError, match=message):
        await anext(registration)


@pytest.mark.asyncio
async def test_registration_rejects_non_sandbox_configuration() -> None:
    config = register_module.SandboxedPythonConfig(sandbox="ds_python_sandbox")
    registration = register_module.sandboxed_python.__wrapped__(config, _builder_with(object()))

    with pytest.raises(TypeError, match="DeepResearchSandboxConfig"):
        await anext(registration)


def test_configuration_has_no_local_backend_switch() -> None:
    with pytest.raises(ValidationError, match="backend"):
        register_module.SandboxedPythonConfig(sandbox="ds_python_sandbox", backend="local")
