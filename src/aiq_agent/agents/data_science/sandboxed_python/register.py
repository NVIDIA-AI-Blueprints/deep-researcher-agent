# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register request-scoped sandboxed Python as a NAT function."""

from pydantic import ConfigDict
from pydantic import Field

from aiq_agent.agents.data_science.utils.analysis_runtime import get_analysis_run
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

from .session import OpenShellPythonRunner
from .session import PythonRunnerLimits


class SandboxedPythonConfig(FunctionBaseConfig, name="sandboxed_python"):
    """Configuration for isolated Python scripts in one OpenShell sandbox per DS request."""

    model_config = ConfigDict(extra="forbid")

    sandbox: FunctionRef = Field(
        ...,
        description="Reference to a blocked-network deep_research_sandbox function configured with OpenShell.",
    )
    wall_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_code_chars: int = Field(default=50_000, ge=1_000, le=500_000)
    max_output_chars: int = Field(default=50_000, ge=1_000, le=500_000)
    max_evidence_bytes: int = Field(default=20_000_000, ge=1_000_000, le=100_000_000)
    max_memory_mb: int = Field(default=8_192, ge=512, le=32_768)
    max_cpu_seconds: int = Field(default=600, ge=1, le=3_600)
    max_processes: int = Field(default=256, ge=1, le=1_024)
    max_open_files: int = Field(default=256, ge=32, le=4_096)
    max_file_bytes: int = Field(default=100_000_000, ge=1_000_000, le=1_000_000_000)


@register_function(config_type=SandboxedPythonConfig)
async def sandboxed_python(tool_config: SandboxedPythonConfig, builder: Builder):
    """Build a stateless Python tool backed only by request-owned OpenShell sandboxes."""

    sandbox_config = builder.get_function_config(tool_config.sandbox)
    if not isinstance(sandbox_config, DeepResearchSandboxConfig):
        raise TypeError(
            f"{tool_config.sandbox!r} must reference DeepResearchSandboxConfig, got {type(sandbox_config).__name__}"
        )
    if sandbox_config.provider.lower() != "openshell":
        raise ValueError("sandboxed_python requires an OpenShell sandbox provider")
    if sandbox_config.network_mode != "blocked":
        raise ValueError("sandboxed_python requires network: blocked")
    if sandbox_config.existing_sandbox_name or sandbox_config.sandbox_name or sandbox_config.allow_shared_sandbox:
        raise ValueError("sandboxed_python requires a fresh per-request OpenShell sandbox")

    limits = PythonRunnerLimits(
        wall_timeout_seconds=tool_config.wall_timeout_seconds,
        max_code_chars=tool_config.max_code_chars,
        max_output_chars=tool_config.max_output_chars,
        max_evidence_bytes=tool_config.max_evidence_bytes,
        max_memory_mb=tool_config.max_memory_mb,
        max_cpu_seconds=tool_config.max_cpu_seconds,
        max_processes=tool_config.max_processes,
        max_open_files=tool_config.max_open_files,
        max_file_bytes=tool_config.max_file_bytes,
    )

    async def _run(code: str) -> str:
        """Execute one self-contained Python script in a fresh namespace.

        Variables do not persist across calls. NumPy (`np`), pandas (`pd`), SciPy
        (`scipy`, `stats`), scikit-learn (`sklearn`), and statsmodels (`sm`) are
        preloaded every time. Successful agent-level structured-data results remain
        available through `list_analysis_results()`, `analysis_result(ref)`,
        `analysis_rows(ref)`, `analysis_sql(ref)`, and `analysis_latest()` in every
        invocation.

        This is an analysis tool, not a data-access tool. It has no configured
        provider client, source SQL connection, or benchmark database. Call the
        agent-level structured-data tools first, then analyze registered rows here.
        """

        run_state = get_analysis_run()
        if run_state is None:
            return '{"status":"error","error":"analysis_runtime_unavailable"}'
        runner = run_state.resources.get("sandboxed_python")
        if runner is None:
            runtime = DeepAgentsRuntime(
                sandbox=sandbox_config,
                job_id=f"data-science-python-{run_state.run_id}",
            )
            runner = OpenShellPythonRunner(
                runtime=runtime,
                host_manifest_path=run_state.manifest_path,
                host_evidence_root=run_state.root,
                limits=limits,
            )
            run_state.resources["sandboxed_python"] = runner
        return await runner.execute(code)

    yield FunctionInfo.from_fn(_run, description=_run.__doc__)
