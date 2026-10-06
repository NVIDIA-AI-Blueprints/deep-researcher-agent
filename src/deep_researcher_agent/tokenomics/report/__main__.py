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

"""CLI entry point: python -m deep_researcher_agent.tokenomics.report"""

from __future__ import annotations

import argparse

from . import generate_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a tokenomics HTML report from Relay ATOF JSONL.")
    parser.add_argument(
        "--trace",
        required=True,
        action="append",
        metavar="TRACE",
        help=(
            "Path to a Relay ATOF JSONL file.  "
            "Repeat the flag to compare multiple runs (e.g. --trace run_a/traces.json --trace run_b/traces.json)."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the eval config YAML")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <first_trace_dir>/tokenomics_report.html)",
    )
    args = parser.parse_args()
    generate_report(args.trace, args.config, args.output)
