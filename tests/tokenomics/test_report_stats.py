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

"""Tests for tokenomics report statistics helpers.

Module under test: src/deep_researcher_agent/tokenomics/report/_report_stats.py
"""

from deep_researcher_agent.tokenomics.report._report_stats import _latency_stats
from deep_researcher_agent.tokenomics.report._report_stats import _load_csv_predictions
from deep_researcher_agent.tokenomics.report._report_stats import _pct


def test_pct_empty():
    assert _pct([], 50) == 0.0


def test_pct_single_value():
    assert _pct([7.0], 50) == 7.0
    assert _pct([7.0], 99) == 7.0


def test_pct_two_values_median():
    # k = 0.5 → linear interpolation between sorted[0] and sorted[1]
    assert _pct([10.0, 20.0], 50) == 15.0


def test_pct_sorted_order_irrelevant():
    assert _pct([30.0, 10.0, 20.0], 50) == 20.0


def test_latency_stats_empty():
    assert _latency_stats([]) == {
        "count": 0,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
        "p99_ms": 0.0,
        "max_ms": 0.0,
        "mean_ms": 0.0,
    }


def test_latency_stats_non_empty():
    out = _latency_stats([0.1, 0.2])
    assert out["count"] == 2
    assert out["mean_ms"] == 150.0
    assert out["max_ms"] == 200.0
    assert out["p50_ms"] == 150.0


def test_load_csv_predictions_missing_file(tmp_path):
    trace = tmp_path / "all_requests_profiler_traces.json"
    trace.write_text("[]")
    assert _load_csv_predictions(str(trace)) == {}


def test_load_csv_predictions_parses_llm_start_rows(tmp_path):
    sub = tmp_path / "run"
    sub.mkdir()
    trace = sub / "all_requests_profiler_traces.json"
    trace.write_text("[]")
    csv_file = sub / "standardized_data_all.csv"
    csv_file.write_text(
        "event_type,UUID,NOVA-Predicted-OSL\nLLM_START,u-1,12.5\nTOOL_START,u-2,99\nLLM_START,u-3,not_a_float\n"
    )
    got = _load_csv_predictions(str(trace))
    assert got == {"u-1": 12.5}
