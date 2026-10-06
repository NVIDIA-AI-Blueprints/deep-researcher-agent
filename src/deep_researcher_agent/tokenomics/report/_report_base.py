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

"""Shared HTML/CSS/JS constants reused by both single-run and comparison report templates."""

from __future__ import annotations

_CSS = """<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1f2937;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --blue: #58a6ff; --orange: #d29922;
    --purple: #bc8cff; --red: #f85149; --teal: #39d353;
    --nvidia: #76b900;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 16px;
    flex-wrap: wrap;
  }
  header h1 { font-size: 18px; font-weight: 600; color: var(--nvidia); }
  header .meta { color: var(--muted); font-size: 13px; }
  nav { background: var(--surface); border-bottom: 1px solid var(--border); display: flex; overflow-x: auto; }
  nav button {
    background: none; border: none; color: var(--muted); padding: 12px 20px;
    cursor: pointer; font-size: 13px; font-weight: 500;
    border-bottom: 2px solid transparent; white-space: nowrap;
    transition: color .15s, border-color .15s;
  }
  nav button:hover { color: var(--text); }
  nav button.active { color: var(--blue); border-bottom-color: var(--blue); }
  main { padding: 20px 24px; max-width: 1600px; margin: 0 auto; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; margin-bottom: 16px;
  }
  .card-header { padding: 10px 16px 8px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; }
  .card-sub { color: var(--muted); font-size: 11px; font-weight: 400; margin-top: 2px; }
  .card-body { padding: 4px; }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }
  .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .stat .label {
    color: var(--muted); font-size: 12px; margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: .5px;
  }
  .stat .value { font-size: 24px; font-weight: 700; }
  .stat .sub { color: var(--muted); font-size: 11px; margin-top: 4px; }
  .stat.green .value { color: var(--green); }
  .stat.blue .value { color: var(--blue); }
  .stat.orange .value { color: var(--orange); }
  .stat.purple .value { color: var(--purple); }
  .stat.red .value { color: var(--red); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    background: var(--surface2); padding: 10px 14px; text-align: left;
    color: var(--muted); font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: .5px;
    border-bottom: 1px solid var(--border);
  }
  td { padding: 9px 14px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface2); }
  .price-table td:nth-child(2), .price-table td:nth-child(3),
  .price-table td:nth-child(4) { color: var(--green); font-family: monospace; }
  @media (max-width: 900px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
</style>"""

_JS_LAYOUT_GLOBALS = """
// ── layout defaults ───────────────────────────────────────────────────────────
const LAYOUT_BASE = {
  paper_bgcolor: '#161b22',
  plot_bgcolor:  '#161b22',
  font: { color: '#e6edf3', size: 12, family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  colorway: ['#58a6ff','#3fb950','#d29922','#bc8cff','#f85149','#39d353','#76b900','#ff7b72','#ffa657'],
  xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: '#30363d' },
};
const CFG = { responsive: true, displayModeBar: false };
function L(extra) { return Object.assign({}, LAYOUT_BASE, extra); }

// ── helpers ───────────────────────────────────────────────────────────────────
function fmtK(v) {
  v = +v;
  return v >= 1e6 ? (v/1e6).toFixed(2)+'M' : v >= 1e3 ? (v/1e3).toFixed(1)+'k' : String(Math.round(v));
}
function fmt$(v, d=4) { return v == null ? 'N/A' : '$' + (+v).toFixed(d); }

const PALETTE = ['#58a6ff','#3fb950','#d29922','#bc8cff','#f85149','#39d353','#76b900','#ff7b72','#ffa657'];
const PHASE_COLORS = { Orchestrator: '#58a6ff', Planner: '#bc8cff', Researcher: '#3fb950' };
"""

_JS_TAB_SWITCHER = """
// ── tab switching ─────────────────────────────────────────────────────────────
let _rendered = {};
function showTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
  renderTab(id);
}
function renderTab(id) {
  if (_rendered[id]) return;
  _rendered[id] = true;
  if (id === 'overview')   renderOverview();
  if (id === 'cost')       renderCost();
  if (id === 'latency')    renderLatency();
  if (id === 'tokens')     renderTokens();
  if (id === 'efficiency') renderEfficiency();
  if (id === 'detail')     renderDetail();
}
"""

_HTML_TOP = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>"""

_HTML_MID = r"""</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
"""

_HTML_AFTER_CSS = r"""
</head>
<body>

<header>
  <h1>&#x26A1; Deep Researcher Agent Tokenomics Report</h1>
  <span class="meta" id="headerMeta"></span>
</header>

<nav>
  <button class="active" onclick="showTab('overview',this)">&#x1F4CA; Overview</button>
  <button onclick="showTab('cost',this)">&#x1F4B0; Cost</button>
  <button onclick="showTab('latency',this)">&#x23F1; Latency</button>
  <button onclick="showTab('tokens',this)">&#x1FA99; Tokens</button>
  <button onclick="showTab('efficiency',this)">&#x1F4D0; Efficiency</button>
  <button onclick="showTab('detail',this)">&#x1F4CB; Per-Query</button>
</nav>

<main>
"""

_JS_HEADER = r"""
</main>

<script>
// ── embedded data ─────────────────────────────────────────────────────────────
const DATA = __REPORT_DATA_JSON__;
"""

_JS_FOOTER = r"""
// ── INITIAL RENDER ────────────────────────────────────────────────────────────
renderTab('overview');
</script>
</body>
</html>
"""


def build_html(
    title: str,
    tab_html: str,
    js_data_extras: str,
    js_extra_globals: str,
    js_init: str,
    js_renders: str,
) -> str:
    """Assemble a complete self-contained HTML report page."""
    return (
        _HTML_TOP
        + title
        + _HTML_MID
        + _CSS
        + _HTML_AFTER_CSS
        + tab_html
        + _JS_HEADER
        + js_data_extras
        + _JS_LAYOUT_GLOBALS
        + js_extra_globals
        + _JS_TAB_SWITCHER
        + js_init
        + js_renders
        + _JS_FOOTER
    )
