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

"""Data aggregation helpers for the tokenomics report."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..pricing import PricingRegistry
from ..profile import PHASE_ORCHESTRATOR
from ..profile import PHASE_ORDER
from ..profile import PHASE_PLANNER
from ..profile import PHASE_RESEARCHER
from ..profile import RequestProfile
from ._report_stats import _latency_stats
from ._report_stats import _pct

PHASE_LABELS = {
    PHASE_ORCHESTRATOR: "Orchestrator",
    PHASE_PLANNER: "Planner",
    PHASE_RESEARCHER: "Researcher",
}


def _build_report_data(
    profiles: list[RequestProfile],
    pricing: PricingRegistry,
    config_path: str,
    predicted_osl_map: dict[str, float] | None = None,
) -> dict:
    # Flatten all per-call observations
    all_llm: list[dict] = []
    all_tool: list[dict] = []
    for prof in profiles:
        all_llm.extend(prof.llm_call_events)
        all_tool.extend(prof.tool_call_events)

    # ── Token stats by model ──────────────────────────────────────────────
    m_isls: dict[str, list] = defaultdict(list)
    m_osls: dict[str, list] = defaultdict(list)
    m_tps: dict[str, list] = defaultdict(list)
    m_tot: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "total_isl": 0,
            "total_osl": 0,
            "total_cached": 0,
            "total_reasoning": 0,
        }
    )
    for ev in all_llm:
        m = ev["model"]
        m_isls[m].append(ev["isl"])
        m_osls[m].append(ev["osl"])
        if ev["tps"] > 0:
            m_tps[m].append(ev["tps"])
        t = m_tot[m]
        t["calls"] += 1
        t["total_isl"] += ev["isl"]
        t["total_osl"] += ev["osl"]
        t["total_cached"] += ev["cached"]
        t["total_reasoning"] += ev["reasoning"]

    by_model_tokens: dict[str, dict] = {}
    for m, t in m_tot.items():
        isls = m_isls[m]
        osls = m_osls[m]
        tps_vals = m_tps[m]
        by_model_tokens[m] = {
            "calls": t["calls"],
            "total_isl": t["total_isl"],
            "total_osl": t["total_osl"],
            "total_cached": t["total_cached"],
            "total_reasoning": t["total_reasoning"],
            "isl_mean": round(sum(isls) / len(isls), 1) if isls else 0.0,
            "isl_p50": round(_pct(isls, 50), 1),
            "isl_p90": round(_pct(isls, 90), 1),
            "isl_p99": round(_pct(isls, 99), 1),
            "isl_max": max(isls) if isls else 0,
            "isl_min": min(isls) if isls else 0,
            "osl_mean": round(sum(osls) / len(osls), 1) if osls else 0.0,
            "osl_p50": round(_pct(osls, 50), 1),
            "osl_p90": round(_pct(osls, 90), 1),
            "osl_p99": round(_pct(osls, 99), 1),
            "osl_max": max(osls) if osls else 0,
            "cache_rate": t["total_cached"] / t["total_isl"] if t["total_isl"] > 0 else 0.0,
            "tps_mean": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0.0,
            "tps_p50": round(_pct(tps_vals, 50), 2) if tps_vals else 0.0,
            "tps_p90": round(_pct(tps_vals, 90), 2) if tps_vals else 0.0,
        }

    # ── Token stats by component (phase) ─────────────────────────────────
    ph_isls: dict[str, list] = defaultdict(list)
    ph_osls: dict[str, list] = defaultdict(list)
    ph_tot: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "total_isl": 0,
            "total_osl": 0,
            "total_cached": 0,
            "total_reasoning": 0,
        }
    )
    for ev in all_llm:
        ph = ev["phase"]
        ph_isls[ph].append(ev["isl"])
        ph_osls[ph].append(ev["osl"])
        t = ph_tot[ph]
        t["calls"] += 1
        t["total_isl"] += ev["isl"]
        t["total_osl"] += ev["osl"]
        t["total_cached"] += ev["cached"]
        t["total_reasoning"] += ev["reasoning"]

    by_component_tokens: dict[str, dict] = {}
    for ph in PHASE_ORDER:
        if ph not in ph_tot:
            continue
        t = ph_tot[ph]
        isls = ph_isls[ph]
        osls = ph_osls[ph]
        label = PHASE_LABELS.get(ph, ph)
        by_component_tokens[label] = {
            "calls": t["calls"],
            "total_isl": t["total_isl"],
            "total_osl": t["total_osl"],
            "total_cached": t["total_cached"],
            "total_reasoning": t["total_reasoning"],
            "isl_mean": round(sum(isls) / len(isls), 1) if isls else 0.0,
            "isl_p50": round(_pct(isls, 50), 1),
            "isl_p90": round(_pct(isls, 90), 1),
            "isl_p99": round(_pct(isls, 99), 1),
            "isl_max": max(isls) if isls else 0,
            "osl_mean": round(sum(osls) / len(osls), 1) if osls else 0.0,
            "osl_p50": round(_pct(osls, 50), 1),
            "osl_p90": round(_pct(osls, 90), 1),
            "osl_p99": round(_pct(osls, 99), 1),
            "osl_max": max(osls) if osls else 0,
            "cache_rate": t["total_cached"] / t["total_isl"] if t["total_isl"] > 0 else 0.0,
        }

    # ── ISL growth: avg ISL by sequential call index, per model ───────────
    growth_data: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for ev in all_llm:
        growth_data[ev["model"]][ev["call_idx"]].append(ev["isl"])

    isl_growth: dict[str, list[dict]] = {}
    for model in sorted(growth_data):
        idx_map = growth_data[model]
        isl_growth[model] = [
            {"idx": idx, "avg_isl": round(sum(v) / len(v), 1), "n": len(v)}
            for idx in sorted(idx_map)
            for v in [idx_map[idx]]
        ]

    # ── ISL vs latency sample ─────────────────────────────────────────────
    isl_latency_sample = [
        {"isl": ev["isl"], "dur_s": ev["dur_s"], "model": ev["model"], "osl": ev["osl"]}
        for ev in all_llm
        if ev["dur_s"] > 0
    ]

    # ── Sys-prompt estimate (min ISL per model) ───────────────────────────
    sys_prompt_est = {m: min(m_isls[m]) for m in m_isls if m_isls[m]}

    # ── LLM latency per model ─────────────────────────────────────────────
    m_durs: dict[str, list] = defaultdict(list)
    for ev in all_llm:
        if ev["dur_s"] > 0:
            m_durs[ev["model"]].append(ev["dur_s"])

    llm_latency = {m: _latency_stats(durs) for m, durs in m_durs.items()}

    # ── Tool latency per tool ─────────────────────────────────────────────
    t_durs: dict[str, list] = defaultdict(list)
    for ev in all_tool:
        if ev["dur_s"] > 0:
            t_durs[ev["tool"]].append(ev["dur_s"])

    tool_latency = {tool: _latency_stats(durs) for tool, durs in t_durs.items()}

    # ── Cost by model ─────────────────────────────────────────────────────
    by_model_cost: dict[str, float] = defaultdict(float)
    for prof in profiles:
        for ps in prof.phases:
            by_model_cost[ps.model] += ps.cost_usd

    # ── Cost by phase ─────────────────────────────────────────────────────
    by_phase_cost: dict[str, float] = {}
    for ph in PHASE_ORDER:
        total = sum(prof.cost_for_phase(ph) for prof in profiles)
        if total > 0:
            by_phase_cost[PHASE_LABELS.get(ph, ph)] = round(total, 6)

    # ── Per-query list ────────────────────────────────────────────────────
    per_query = []
    for prof in profiles:
        pq_by_phase = {}
        for ph in PHASE_ORDER:
            label = PHASE_LABELS.get(ph, ph)
            cost = prof.cost_for_phase(ph)
            if cost > 0:
                pq_by_phase[label] = round(cost, 6)
        per_query.append(
            {
                "id": prof.request_index,
                "question": prof.question,
                "cost_usd": round(prof.grand_total_cost_usd, 6),
                "llm_cost_usd": round(prof.total_cost_usd, 6),
                "tool_cost_usd": round(prof.total_tool_cost_usd, 6),
                "input_tokens": prof.total_prompt_tokens,
                "output_tokens": prof.total_completion_tokens,
                "cached_tokens": prof.total_cached_tokens,
                "entry_count": prof.total_llm_calls,
                "duration_s": round(prof.duration_s, 2),
                "by_phase": pq_by_phase,
            }
        )

    # ── Pricing snapshot ──────────────────────────────────────────────────
    pricing_snapshot: dict[str, dict] = {}
    for model in pricing.known_models():
        p = pricing.get(model)
        pricing_snapshot[model] = {
            "input_per_1m_tokens": p.input_per_1m_tokens,
            "cached_input_per_1m_tokens": p.cached_input_per_1m_tokens,
            "output_per_1m_tokens": p.output_per_1m_tokens,
        }
    if pricing._default is not None:
        pricing_snapshot["default"] = {
            "input_per_1m_tokens": pricing._default.input_per_1m_tokens,
            "cached_input_per_1m_tokens": pricing._default.cached_input_per_1m_tokens,
            "output_per_1m_tokens": pricing._default.output_per_1m_tokens,
        }

    # ── Tool cost aggregation ─────────────────────────────────────────────
    by_tool_cost: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total_cost_usd": 0.0})
    for ev in all_tool:
        entry = by_tool_cost[ev["tool"]]
        entry["calls"] += 1
        entry["total_cost_usd"] += ev.get("cost_usd", 0.0)
    by_tool_cost = {k: dict(v) for k, v in by_tool_cost.items()}

    # Tool pricing snapshot (only configured tools)
    tool_pricing_snapshot = {name: pricing.get_tool(name).cost_per_call for name in pricing.known_tools()}

    # ── Predicted vs actual OSL (from NOVA-Predicted-OSL in CSV) ─────────
    # NOTE: in current NAT traces, NOVA-Predicted-OSL is filled post-hoc with
    # the actual completion tokens, so predicted == actual on every call.
    # The list is populated here for forward-compatibility; the chart is hidden
    # when all errors are zero (trivially perfect, not informative).
    predicted_vs_actual: list[dict] = []
    if predicted_osl_map:
        for ev in all_llm:
            pred = predicted_osl_map.get(ev.get("uuid", ""))
            if pred is not None:
                predicted_vs_actual.append(
                    {
                        "model": ev["model"],
                        "predicted": pred,
                        "actual": ev["osl"],
                        "phase": ev["phase"],
                    }
                )

    total_llm_cost = sum(p.total_cost_usd for p in profiles)
    total_tool_cost = sum(p.total_tool_cost_usd for p in profiles)
    grand_total = total_llm_cost + total_tool_cost
    return {
        "label": Path(config_path).name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_queries": len(profiles),
        "total_cost_usd": round(grand_total, 6),
        "llm_cost_usd": round(total_llm_cost, 6),
        "tool_cost_usd": round(total_tool_cost, 6),
        "avg_cost_usd": round(grand_total / len(profiles), 6) if profiles else 0.0,
        "cache_savings_usd": round(sum(p.total_cache_savings_usd for p in profiles), 6),
        "total_prompt_tokens": sum(p.total_prompt_tokens for p in profiles),
        "total_cached_tokens": sum(p.total_cached_tokens for p in profiles),
        "total_completion_tokens": sum(p.total_completion_tokens for p in profiles),
        "total_llm_calls": sum(p.total_llm_calls for p in profiles),
        "per_query": per_query,
        "by_model": dict(by_model_cost),
        "by_phase": by_phase_cost,
        "by_tool": by_tool_cost,
        "phase_order": [PHASE_LABELS.get(ph, ph) for ph in PHASE_ORDER],
        "llm_latency": llm_latency,
        "tool_latency": tool_latency,
        "pricing_snapshot": pricing_snapshot,
        "tool_pricing_snapshot": tool_pricing_snapshot,
        "token_stats": {
            "by_model": by_model_tokens,
            "by_component": by_component_tokens,
            "isl_growth": isl_growth,
            "isl_latency_sample": isl_latency_sample,
            "sys_prompt_est": sys_prompt_est,
            "predicted_vs_actual": predicted_vs_actual,
        },
    }


def _build_comparison_data(run_datas: list[dict]) -> dict:
    """Return an A-vs-B comparison block to embed in the primary run's report_data.

    Only the first two runs are compared.  The per-query list is the UNION of
    both runs' query IDs so that queries unique to one run are still visible in
    the table (with null for the missing side).  The delta bar chart in the
    report only renders bars for queries present in both runs.
    """
    a = run_datas[0]
    b = run_datas[1]

    a_by_id = {q["id"]: q for q in a.get("per_query", [])}
    b_by_id = {q["id"]: q for q in b.get("per_query", [])}
    all_ids = sorted(set(a_by_id) | set(b_by_id))
    common_ids = set(a_by_id) & set(b_by_id)

    per_query_cmp = []
    for qid in all_ids:
        qa = a_by_id.get(qid)
        qb = b_by_id.get(qid)
        in_both = qa is not None and qb is not None

        cost_a = qa["cost_usd"] if qa else None
        cost_b = qb["cost_usd"] if qb else None
        if in_both:
            cost_delta: float | None = round(cost_b - cost_a, 6)  # type: ignore[operator]
            cost_pct: float | None = round((cost_delta / cost_a * 100) if cost_a else 0.0, 1)
        else:
            cost_delta = cost_pct = None

        per_query_cmp.append(
            {
                "id": qid,
                "question": (qa or qb).get("question", ""),  # type: ignore[union-attr]
                "cost_a": cost_a,
                "cost_b": cost_b,
                "cost_delta": cost_delta,
                "cost_pct": cost_pct,
                "isl_a": qa.get("input_tokens") if qa else None,
                "isl_b": qb.get("input_tokens") if qb else None,
                "osl_a": qa.get("output_tokens") if qa else None,
                "osl_b": qb.get("output_tokens") if qb else None,
                "duration_a": qa.get("duration_s") if qa else None,
                "duration_b": qb.get("duration_s") if qb else None,
                "llm_calls_a": qa.get("entry_count") if qa else None,
                "llm_calls_b": qb.get("entry_count") if qb else None,
                "in_both": in_both,
            }
        )

    cost_delta_total = b["total_cost_usd"] - a["total_cost_usd"]
    cost_pct_total = (cost_delta_total / a["total_cost_usd"] * 100) if a["total_cost_usd"] else 0.0
    prompt_a = a.get("total_prompt_tokens", 0)
    prompt_b = b.get("total_prompt_tokens", 0)

    return {
        "label_a": a["label"],
        "label_b": b["label"],
        "num_queries_a": a["num_queries"],
        "num_queries_b": b["num_queries"],
        "num_common_queries": len(common_ids),
        "total_cost_a": a["total_cost_usd"],
        "total_cost_b": b["total_cost_usd"],
        "llm_cost_a": a["llm_cost_usd"],
        "llm_cost_b": b["llm_cost_usd"],
        "total_llm_calls_a": a["total_llm_calls"],
        "total_llm_calls_b": b["total_llm_calls"],
        "cache_rate_a": round(a.get("total_cached_tokens", 0) / prompt_a, 4) if prompt_a else 0.0,
        "cache_rate_b": round(b.get("total_cached_tokens", 0) / prompt_b, 4) if prompt_b else 0.0,
        "cost_delta": round(cost_delta_total, 6),
        "cost_pct_change": round(cost_pct_total, 1),
        "by_model_a": a.get("by_model", {}),
        "by_model_b": b.get("by_model", {}),
        "by_phase_a": a.get("by_phase", {}),
        "by_phase_b": b.get("by_phase", {}),
        "by_tool_b": b.get("by_tool", {}),
        "llm_latency_b": b.get("llm_latency", {}),
        "tool_latency_b": b.get("tool_latency", {}),
        "token_stats_b": b.get("token_stats", {}),
        "per_query": per_query_cmp,
    }
