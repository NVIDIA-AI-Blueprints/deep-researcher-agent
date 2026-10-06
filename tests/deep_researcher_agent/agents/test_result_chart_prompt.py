# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Guards the result-chart contract wherever the agents document it.

The inline ``chart`` spec used to be embedded in the deep-research writer prompt.
It now lives in the on-demand ``chart-generation`` skill, so the always-on writer
prompt stays short and both delivery modes are driven by the skill. The shallow
researcher keeps its own inline copy because the shallow path has no skill runtime.

These tests ensure the worked examples in every source that still documents the
contract (the shallow prompt and the chart skill) stay valid JSON that matches the
UI ResultChart zod schema, and that the writer prompt now delegates to the skill
instead of embedding the contract.
"""

import json
import re
from pathlib import Path

import pytest

from deep_researcher_agent.common import render_prompt_template

_AGENTS = Path(__file__).resolve().parents[3] / "src" / "deep_researcher_agent" / "agents"
_WRITER = _AGENTS / "deep_researcher" / "prompts" / "writer.j2"
_CHART_SKILL = _AGENTS / "deep_researcher" / "skills" / "visualization" / "chart-generation" / "SKILL.md"

# The inline ``chart`` contract now lives in exactly these two places: the shallow
# researcher prompt (no skill runtime) and the chart-generation skill (both the
# deep researcher and writer read it on demand).
_CONTRACT_SOURCES = {
    "shallow": _AGENTS / "shallow_researcher" / "prompts" / "researcher.j2",
    "skill": _CHART_SKILL,
}

# Kept in lockstep with the UI ResultChart schema in types.ts.
_CHART_TYPES = {"bar", "hbar", "line", "area", "grouped-bar", "delta"}
_CHART_COLORS = {"green", "blue", "amber", "red", "neutral"}
_VALUE_FORMATS = {"number", "compact", "percent", "currency"}
_KPI_TONES = {"default", "accent", "warn", "alarm"}

_CHART_BLOCK = re.compile(r"```chart\n(.*?)\n```", re.DOTALL)
_CAROUSEL_BLOCK = re.compile(r"```chart-carousel\n(.*?)\n```", re.DOTALL)


def _chart_examples(source_key: str) -> list[dict]:
    text = _CONTRACT_SOURCES[source_key].read_text()
    return [json.loads(block) for block in _CHART_BLOCK.findall(text)]


def _assert_kpis_match_schema(kpis: list) -> None:
    assert 1 <= len(kpis) <= 4
    for kpi in kpis:
        assert kpi["label"] and kpi["value"]
        if "tone" in kpi:
            assert kpi["tone"] in _KPI_TONES


def _assert_full_chart_matches_schema(spec: dict) -> None:
    """Mirror the ResultChart zod ChartSpecSchema shape."""
    assert spec["type"] in _CHART_TYPES
    assert spec["title"]
    assert spec["x"]["key"]
    if "y" in spec and spec["y"].get("format") is not None:
        assert spec["y"]["format"] in _VALUE_FORMATS
    assert 1 <= len(spec["series"]) <= 6
    for series in spec["series"]:
        assert series["key"]
        if series.get("color") is not None:
            assert series["color"] in _CHART_COLORS
    # A delta chart colors bars by sign with no legend, so it encodes one series.
    if spec["type"] == "delta":
        assert len(spec["series"]) == 1
    assert 1 <= len(spec["data"]) <= 60
    assert all(isinstance(row, dict) for row in spec["data"])
    if "kpis" in spec:
        _assert_kpis_match_schema(spec["kpis"])


@pytest.mark.parametrize("source_key", list(_CONTRACT_SOURCES))
def test_contract_source_defines_the_chart_contract(source_key: str) -> None:
    text = _CONTRACT_SOURCES[source_key].read_text()
    assert "chart-carousel" in text
    for chart_type in _CHART_TYPES:
        assert chart_type in text, f"{source_key} omits chart type {chart_type!r}"


@pytest.mark.parametrize("source_key", list(_CONTRACT_SOURCES))
def test_contract_source_carousel_examples_match_the_schema(source_key: str) -> None:
    text = _CONTRACT_SOURCES[source_key].read_text()
    blocks = _CAROUSEL_BLOCK.findall(text)
    assert blocks, f"{source_key} has no ```chart-carousel example"
    for block in blocks:
        carousel = json.loads(block)
        assert carousel["title"]
        assert 2 <= len(carousel["charts"]) <= 12
        for chart in carousel["charts"]:
            assert chart["type"] == "line"
            _assert_full_chart_matches_schema(chart)


@pytest.mark.parametrize("source_key", list(_CONTRACT_SOURCES))
def test_contract_source_chart_examples_match_the_schema(source_key: str) -> None:
    examples = _chart_examples(source_key)
    assert examples, f"{source_key} has no ```chart example"

    saw_full_chart = False
    saw_kpi_only = False
    for spec in examples:
        assert isinstance(spec, dict)
        assert spec["title"]
        if "type" in spec:
            saw_full_chart = True
            _assert_full_chart_matches_schema(spec)
        else:
            saw_kpi_only = True
            _assert_kpis_match_schema(spec["kpis"])

    assert saw_full_chart, f"{source_key} should show a full chart example"
    assert saw_kpi_only, f"{source_key} should show a KPI-only example"


def test_chart_skill_covers_both_delivery_modes() -> None:
    """The single skill drives sandbox (PNG artifact) and non-sandbox (inline spec) charts."""
    skill = _CHART_SKILL.read_text()
    assert "## Choose your mode" in skill
    # Sandbox mode markers.
    assert "artifact://" in skill
    assert "make_chart.py" in skill
    assert "matplotlib" in skill
    # Inline mode markers.
    assert "```chart" in skill
    assert "chart-carousel" in skill


_WRITER_RENDER_CONTEXT = {
    "current_datetime": "2026-01-01T00:00:00Z",
    "user_info": None,
    "parent_report_context_available": False,
    "sandbox_workdir": "/sandbox/workdir",
    "sandbox_artifact_dir": "/sandbox/artifacts",
}


def _render_writer(*, chart_skill_enabled: bool, execution_enabled: bool) -> str:
    return render_prompt_template(
        _WRITER.read_text(),
        chart_skill_enabled=chart_skill_enabled,
        execution_enabled=execution_enabled,
        **_WRITER_RENDER_CONTEXT,
    )


def test_writer_prompt_no_longer_embeds_the_inline_chart_contract() -> None:
    text = _WRITER.read_text()
    assert "## Presenting Data (Charts)" not in text
    assert "```chart" not in text


@pytest.mark.parametrize("execution_enabled", [True, False])
def test_writer_delegates_charts_to_the_skill_when_chart_skill_enabled(execution_enabled: bool) -> None:
    rendered = _render_writer(chart_skill_enabled=True, execution_enabled=execution_enabled)
    assert "## Figures and Charts" in rendered
    assert "chart-generation" in rendered
    # The contract is delivered on demand by the skill, never inlined here.
    assert "```chart" not in rendered


def test_writer_degrades_to_a_table_without_the_chart_skill() -> None:
    rendered = _render_writer(chart_skill_enabled=False, execution_enabled=False)
    assert "## Figures and Charts" in rendered
    assert "chart-generation" not in rendered
    assert "compact Markdown table" in rendered
    assert "```chart" not in rendered
