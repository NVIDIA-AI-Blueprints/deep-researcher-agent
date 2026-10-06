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

"""Tests for tokenomics report data builders.

Module under test: src/deep_researcher_agent/tokenomics/report/_report_builders.py
"""

from deep_researcher_agent.tokenomics.pricing import PricingRegistry
from deep_researcher_agent.tokenomics.profile import PHASE_ORCHESTRATOR
from deep_researcher_agent.tokenomics.profile import PhaseStats
from deep_researcher_agent.tokenomics.profile import RequestProfile
from deep_researcher_agent.tokenomics.report._report_builders import _build_comparison_data
from deep_researcher_agent.tokenomics.report._report_builders import _build_report_data


def _minimal_pricing() -> PricingRegistry:
    return PricingRegistry.from_dict(
        {
            "models": {
                "m1": {
                    "input_per_1m_tokens": 1.0,
                    "output_per_1m_tokens": 2.0,
                },
            },
            "tools": {"search": {"cost_per_call": 0.001}},
            "default": None,
        }
    )


def _minimal_profile(**overrides) -> RequestProfile:
    base = dict(
        request_index=0,
        question="hello",
        duration_s=3.0,
        phases=[
            PhaseStats(
                phase=PHASE_ORCHESTRATOR,
                model="m1",
                llm_calls=1,
                prompt_tokens=1000,
                cached_tokens=0,
                completion_tokens=100,
                cost_usd=0.002,
                cache_savings_usd=0.0,
            ),
        ],
        tool_calls={"search": 1},
        llm_call_events=[
            {
                "uuid": "llm-1",
                "isl": 1000,
                "osl": 100,
                "cached": 0,
                "reasoning": 0,
                "dur_s": 2.0,
                "tps": 50.0,
                "model": "m1",
                "phase": PHASE_ORCHESTRATOR,
                "call_idx": 0,
            },
        ],
        tool_call_events=[
            {"tool": "search", "dur_s": 0.5, "cost_usd": 0.001},
        ],
        total_llm_calls=1,
        total_prompt_tokens=1000,
        total_cached_tokens=0,
        total_completion_tokens=100,
        total_cost_usd=0.002,
        total_tool_cost_usd=0.001,
        total_cache_savings_usd=0.0,
    )
    base.update(overrides)
    return RequestProfile(**base)


def test_build_report_data_totals_and_per_query():
    pricing = _minimal_pricing()
    prof = _minimal_profile()
    rd = _build_report_data([prof], pricing, "/tmp/pricing.yml")

    assert rd["num_queries"] == 1
    assert rd["total_llm_calls"] == 1
    assert rd["total_prompt_tokens"] == 1000
    assert rd["total_completion_tokens"] == 100
    assert rd["llm_cost_usd"] == 0.002
    assert rd["tool_cost_usd"] == 0.001
    assert rd["total_cost_usd"] == 0.003
    assert rd["by_model"]["m1"] == 0.002
    assert "Orchestrator" in rd["by_phase"]
    assert rd["per_query"][0]["id"] == 0
    assert rd["per_query"][0]["question"] == "hello"
    assert rd["token_stats"]["by_model"]["m1"]["calls"] == 1
    assert rd["llm_latency"]["m1"]["count"] == 1
    assert rd["tool_latency"]["search"]["count"] == 1


def test_build_report_data_predicted_vs_actual():
    pricing = _minimal_pricing()
    prof = _minimal_profile()
    pred = {"llm-1": 99.0}
    rd = _build_report_data([prof], pricing, "/tmp/x.yml", predicted_osl_map=pred)
    pva = rd["token_stats"]["predicted_vs_actual"]
    assert len(pva) == 1
    assert pva[0]["predicted"] == 99.0
    assert pva[0]["actual"] == 100


def test_build_comparison_data_aligned_queries():
    pricing = _minimal_pricing()
    prof = _minimal_profile(request_index=1)
    a = _build_report_data([prof], pricing, "/tmp/a.yml")
    b = _build_report_data([prof], pricing, "/tmp/b.yml")
    a["label"] = "run_a"
    b["label"] = "run_b"
    b["total_cost_usd"] = a["total_cost_usd"] + 0.01

    cmp = _build_comparison_data([a, b])
    assert cmp["label_a"] == "run_a"
    assert cmp["label_b"] == "run_b"
    assert cmp["num_common_queries"] == 1
    assert cmp["num_queries_a"] == 1
    assert cmp["num_queries_b"] == 1
    assert cmp["cost_delta"] == 0.01
    assert len(cmp["per_query"]) == 1
    assert cmp["per_query"][0]["in_both"] is True
    assert cmp["per_query"][0]["cost_delta"] is not None


def test_build_comparison_data_union_when_ids_differ():
    pricing = _minimal_pricing()
    a = _build_report_data([_minimal_profile(request_index=1)], pricing, "/a.yml")
    b = _build_report_data([_minimal_profile(request_index=2)], pricing, "/b.yml")
    a["label"] = "a"
    b["label"] = "b"

    cmp = _build_comparison_data([a, b])
    assert cmp["num_common_queries"] == 0
    assert len(cmp["per_query"]) == 2
    both_flags = {row["in_both"] for row in cmp["per_query"]}
    assert both_flags == {False}
