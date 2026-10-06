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

"""
Generate a self-contained tokenomics HTML report from Relay ATOF JSONL.

Single-run
----------
python -m deep_researcher_agent.tokenomics.report \\
    --trace relay/deep-researcher-relay.atof.jsonl \\
    --config configs/config_tokenomics_pricing.yml \\
    [--output results/tokenomics_report.html]

Comparison (two or more runs)
------------------------------
python -m deep_researcher_agent.tokenomics.report \\
    --trace results/run_a/deep-researcher-relay.atof.jsonl \\
    --trace results/run_b/deep-researcher-relay.atof.jsonl \\
    --config configs/config_tokenomics_pricing.yml

Passing ``--trace`` more than once activates comparison mode: every tab
(Overview, Cost, Latency, Tokens, Efficiency, Per-Query) shows A-vs-B
comparison charts instead of single-run visualisations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from ..atof_adapter import parse_trace
from ..pricing import PricingRegistry
from ._report_builders import _build_comparison_data
from ._report_builders import _build_report_data
from ._report_stats import _load_csv_predictions
from ._report_template_comparison import render_html as _render_comparison
from ._report_template_single import render_html as _render_single


def generate_report(
    trace_path: str | list[str],
    config_path: str,
    output_path: str | None = None,
) -> str:
    """Generate a tokenomics HTML report.

    Parameters
    ----------
    trace_path:
        Path to a single Relay ATOF JSONL file, or a list of
        paths for comparison mode.  When more than one path is provided every
        tab (Overview, Cost, Latency, Tokens, Efficiency, Per-Query) shows
        A-vs-B comparison charts instead of single-run visualisations.
    config_path:
        Path to the eval config YAML (provides pricing).
    output_path:
        Destination HTML file.  Defaults to ``<first_trace_dir>/tokenomics_report.html``.
    """
    if isinstance(trace_path, str):
        trace_paths: list[str] = [trace_path]
    else:
        trace_paths = list(trace_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    pricing_raw = (config.get("tokenomics") or {}).get("pricing") or {}
    pricing = PricingRegistry.from_dict(pricing_raw)

    run_datas: list[dict] = []
    for tp in trace_paths:
        profiles = parse_trace(tp, pricing)
        if not profiles:
            print(f"WARNING: no request profiles parsed — check {tp}", file=sys.stderr)

        predicted_osl_map = _load_csv_predictions(tp)
        if predicted_osl_map:
            print(f"Loaded {len(predicted_osl_map)} NOVA-Predicted-OSL values from {Path(tp).name}.")

        rd = _build_report_data(profiles, pricing, config_path, predicted_osl_map)
        # In multi-run mode use the trace's parent directory name as the run label
        # so the comparison tab can distinguish "run_a" from "run_b".
        if len(trace_paths) > 1:
            rd["label"] = Path(tp).parent.name or Path(tp).stem
        run_datas.append(rd)

    primary = run_datas[0]
    if len(run_datas) >= 2:
        cmp = _build_comparison_data(run_datas)
        primary["comparison"] = cmp
        print(
            f"Comparison mode: "
            f"Run A ({cmp['label_a']}) = {cmp['num_queries_a']} queries, "
            f"Run B ({cmp['label_b']}) = {cmp['num_queries_b']} queries, "
            f"{cmp['num_common_queries']} aligned by query ID."
        )
        if cmp["num_common_queries"] == 0:
            print(
                "  WARNING: no overlapping query IDs — per-query deltas will be empty.\n"
                "  Make sure both runs use the same filter.allowlist.",
                file=sys.stderr,
            )
    else:
        primary["comparison"] = None

    html = _render_comparison(primary) if len(run_datas) >= 2 else _render_single(primary)

    if output_path is None:
        output_dir = (config.get("eval") or {}).get("general", {}).get("output_dir")
        if output_dir:
            output_path = str(Path(output_dir) / "tokenomics_report.html")
        else:
            output_path = str(Path(trace_paths[0]).parent / "tokenomics_report.html")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"Report written → {output_path}")
    return output_path
