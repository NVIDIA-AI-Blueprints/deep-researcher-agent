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

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

# Canonical phase names produced by atof_adapter.
PHASE_ORCHESTRATOR = "orchestrator"
PHASE_PLANNER = "planner-agent"
PHASE_RESEARCHER = "researcher-phase"

PHASE_ORDER = (PHASE_ORCHESTRATOR, PHASE_PLANNER, PHASE_RESEARCHER)


@dataclass
class PhaseStats:
    """
    Token and cost totals for one (phase, model) combination within a single
    workflow run.  Multiple models can contribute to the same phase (e.g.
    if the orchestrator LLM is swapped mid-run), so the primary grouping key
    is ``(phase, model)``.
    """

    phase: str
    model: str
    llm_calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_savings_usd: float = 0.0

    @property
    def uncached_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RequestProfile:
    """
    All tokenomics data for a single workflow run, pre-aggregated along the
    dimensions needed by the tokenomics HTML report.
    """

    request_index: int
    question: str
    duration_s: float

    # Aggregates across all phases
    total_cost_usd: float = 0.0
    total_tool_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    total_completion_tokens: int = 0
    total_cache_savings_usd: float = 0.0
    total_llm_calls: int = 0

    # One entry per (phase, model) pair — populated by atof_adapter
    phases: list[PhaseStats] = field(default_factory=list)

    # tool_name → invocation count
    tool_calls: dict[str, int] = field(default_factory=dict)

    # Individual LLM call observations (one dict per LLM_END event):
    # keys: isl, osl, cached, reasoning, dur_s, tps, model, phase, call_idx
    llm_call_events: list[dict] = field(default_factory=list)

    # Individual tool call observations (one dict per TOOL_END event):
    # keys: tool, dur_s
    tool_call_events: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def grand_total_cost_usd(self) -> float:
        """LLM token costs + tool API call costs combined."""
        return self.total_cost_usd + self.total_tool_cost_usd

    @property
    def cache_hit_rate(self) -> float:
        return self.total_cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0.0

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    def phases_for(self, phase: str) -> list[PhaseStats]:
        """Return all PhaseStats entries matching ``phase`` (may span models)."""
        return [p for p in self.phases if p.phase == phase]

    def cost_for_phase(self, phase: str) -> float:
        return sum(p.cost_usd for p in self.phases_for(phase))

    def tokens_for_phase(self, phase: str) -> tuple[int, int, int]:
        """Return (prompt, cached, completion) totals for a phase."""
        prompt = sum(p.prompt_tokens for p in self.phases_for(phase))
        cached = sum(p.cached_tokens for p in self.phases_for(phase))
        completion = sum(p.completion_tokens for p in self.phases_for(phase))
        return prompt, cached, completion
