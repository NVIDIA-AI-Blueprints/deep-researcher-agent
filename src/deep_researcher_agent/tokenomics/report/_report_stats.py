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

"""Pure statistical helpers for the tokenomics report. No project imports."""

from __future__ import annotations

import csv
from pathlib import Path


def _pct(data: list, p: float) -> float:
    """Return the p-th percentile of ``data`` (linear interpolation)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _latency_stats(durations_s: list[float]) -> dict:
    if not durations_s:
        return {"count": 0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    ms = [d * 1000.0 for d in durations_s]
    return {
        "count": len(ms),
        "p50_ms": round(_pct(ms, 50), 2),
        "p90_ms": round(_pct(ms, 90), 2),
        "p99_ms": round(_pct(ms, 99), 2),
        "max_ms": round(max(ms), 2),
        "mean_ms": round(sum(ms) / len(ms), 2),
    }


def _load_csv_predictions(trace_path: str) -> dict[str, float]:
    """
    Load NOVA-Predicted-OSL values from standardized_data_all.csv if it lives
    alongside the trace file.  Returns UUID → predicted_osl mapping.
    """
    csv_path = Path(trace_path).parent / "standardized_data_all.csv"
    if not csv_path.exists():
        return {}
    predictions: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") == "LLM_START" and row.get("NOVA-Predicted-OSL") and row.get("UUID"):
                try:
                    predictions[row["UUID"]] = float(row["NOVA-Predicted-OSL"])
                except ValueError:
                    pass
    return predictions
