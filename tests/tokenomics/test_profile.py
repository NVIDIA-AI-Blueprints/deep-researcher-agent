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

"""Tests for tokenomics profile dataclasses.

Module under test: src/deep_researcher_agent/tokenomics/profile.py
"""

from deep_researcher_agent.tokenomics.profile import PHASE_ORCHESTRATOR
from deep_researcher_agent.tokenomics.profile import PHASE_PLANNER
from deep_researcher_agent.tokenomics.profile import PHASE_RESEARCHER
from deep_researcher_agent.tokenomics.profile import PhaseStats
from deep_researcher_agent.tokenomics.profile import RequestProfile


def test_phase_stats_derived_fields():
    ps = PhaseStats(
        phase=PHASE_ORCHESTRATOR,
        model="m",
        prompt_tokens=100,
        cached_tokens=25,
        completion_tokens=50,
    )
    assert ps.uncached_tokens == 75
    assert ps.cache_hit_rate == 0.25
    assert ps.total_tokens == 150


def test_phase_stats_cache_hit_rate_zero_prompt():
    ps = PhaseStats(phase=PHASE_ORCHESTRATOR, model="m", prompt_tokens=0, cached_tokens=0)
    assert ps.cache_hit_rate == 0.0


def test_request_profile_grand_total_and_cache_rate():
    prof = RequestProfile(
        request_index=0,
        question="q",
        duration_s=1.0,
        total_cost_usd=10.0,
        total_tool_cost_usd=2.5,
        total_prompt_tokens=200,
        total_cached_tokens=50,
        total_completion_tokens=100,
        phases=[],
    )
    assert prof.grand_total_cost_usd == 12.5
    assert prof.cache_hit_rate == 0.25


def test_request_profile_phases_for_and_cost():
    prof = RequestProfile(
        request_index=0,
        question="q",
        duration_s=1.0,
        phases=[
            PhaseStats(phase=PHASE_PLANNER, model="a", cost_usd=1.0, prompt_tokens=10),
            PhaseStats(phase=PHASE_RESEARCHER, model="b", cost_usd=3.0, prompt_tokens=20),
            PhaseStats(phase=PHASE_RESEARCHER, model="c", cost_usd=2.0, prompt_tokens=30),
        ],
    )
    assert len(prof.phases_for(PHASE_RESEARCHER)) == 2
    assert prof.cost_for_phase(PHASE_PLANNER) == 1.0
    assert prof.cost_for_phase(PHASE_RESEARCHER) == 5.0


def test_request_profile_tokens_for_phase():
    prof = RequestProfile(
        request_index=0,
        question="q",
        duration_s=1.0,
        phases=[
            PhaseStats(
                phase=PHASE_RESEARCHER,
                model="b",
                prompt_tokens=100,
                cached_tokens=40,
                completion_tokens=60,
            ),
            PhaseStats(
                phase=PHASE_RESEARCHER,
                model="c",
                prompt_tokens=50,
                cached_tokens=10,
                completion_tokens=20,
            ),
        ],
    )
    p, c, o = prof.tokens_for_phase(PHASE_RESEARCHER)
    assert (p, c, o) == (150, 50, 80)


def test_request_profile_total_tool_calls():
    prof = RequestProfile(
        request_index=0,
        question="q",
        duration_s=1.0,
        tool_calls={"a": 2, "b": 5},
    )
    assert prof.total_tool_calls == 7
