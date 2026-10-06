# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts that keep chart generation aligned with artifact harvesting."""

from __future__ import annotations

from pathlib import Path

from deep_researcher_agent.common import render_prompt_template

_AGENT_ROOT = Path(__file__).parents[4] / "src" / "deep_researcher_agent" / "agents" / "deep_researcher"


def test_chart_skill_uses_runtime_argument_instead_of_executable_placeholder() -> None:
    skill = (_AGENT_ROOT / "skills" / "visualization" / "chart-generation" / "SKILL.md").read_text(encoding="utf-8")

    assert 'ARTIFACT_DIR = "<sandbox_artifact_dir>"' not in skill
    assert '"path": "<sandbox_artifact_dir>' not in skill
    assert "ARTIFACT_DIR = Path(sys.argv[1])" in skill
    assert 'ARTIFACT_DIR / "manifest.json"' in skill


def test_writer_provides_rendered_per_job_paths_as_chart_skill_context() -> None:
    prompt = (_AGENT_ROOT / "prompts" / "writer.j2").read_text(encoding="utf-8")
    rendered = render_prompt_template(
        prompt,
        current_datetime="2026-07-09",
        chart_skill_enabled=True,
        execution_enabled=True,
        parent_report_context_available=False,
        sandbox_workdir="/sandbox/job-123",
        sandbox_artifact_dir="/sandbox/job-123/deep-researcher-artifacts",
        user_info=None,
    )

    assert "/sandbox/job-123" in rendered
    assert "/sandbox/job-123/deep-researcher-artifacts" in rendered
    assert "<sandbox_workdir>" not in rendered
    assert "<sandbox_artifact_dir>" not in rendered
