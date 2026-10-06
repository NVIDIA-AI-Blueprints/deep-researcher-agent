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

"""HTML template and render helper for comparison-mode tokenomics reports. No project imports."""

from __future__ import annotations

import json

from ._report_base import build_html

_TAB_HTML = r"""
<!-- ── OVERVIEW ─────────────────────────────────────────────────────────── -->
<div id="tab-overview" class="tab-content active">
  <div class="stat-grid" id="overviewStats"></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F916; Cost by Model &mdash; Run A vs Run B
        <div class="card-sub">Same model in both bars = same routing; model only in one run = model swap.</div>
      </div>
      <div class="card-body"><div id="overviewModelBar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F3D7; Cost by Phase &mdash; Run A vs Run B
        <div class="card-sub">A phase cost shift often indicates fewer parallel calls or a changed routing policy.</div>
      </div>
      <div class="card-body"><div id="overviewPhaseBar"></div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F4CB; Per-Query Summary &mdash; Run A vs Run B</div>
    <div class="card-body" style="padding:0">
      <table id="overviewTable">
        <thead><tr>
          <th>Query #</th><th>Cost A</th><th>Cost B</th><th>&#x0394; Cost</th><th>&#x0394; %</th>
          <th>ISL A</th><th>ISL B</th><th>OSL A</th><th>OSL B</th>
          <th>Dur A (s)</th><th>Dur B (s)</th><th>Calls A</th><th>Calls B</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ── COST ──────────────────────────────────────────────────────────────── -->
<div id="tab-cost" class="tab-content">
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F916; Cost by Model &mdash; Run A vs Run B
        <div class="card-sub">Same model in both bars = same routing, different volumes or prompts.
          A model in only one run signals a model swap.</div>
      </div>
      <div class="card-body"><div id="costCmpModelBar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F3D7; Cost by Phase &mdash; Run A vs Run B
        <div class="card-sub">A phase cost shift (e.g. lower Researcher spend) often indicates fewer
          parallel search calls or a changed routing policy.</div>
      </div>
      <div class="card-body"><div id="costCmpPhaseBar"></div></div>
    </div>
  </div>
  <div id="costCmpToolCard" class="card">
    <div class="card-header">&#x1F50D; Tool Cost &mdash; Run A vs Run B
      <div class="card-sub">Compare per-tool API spend across the two runs.</div>
    </div>
    <div class="card-body"><div id="costCmpToolBar"></div></div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F4C8; Per-Query Cost Delta (B &minus; A)
      <div class="card-sub">Green bars = Run B is cheaper; red bars = Run B costs more.
        Only queries present in both runs are shown.</div>
    </div>
    <div class="card-body"><div id="costCmpDeltaBar"></div></div>
  </div>
</div>

<!-- ── LATENCY ───────────────────────────────────────────────────────────── -->
<div id="tab-latency" class="tab-content">
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F4CA; LLM p50 Latency &mdash; Run A vs Run B
        <div class="card-sub">Median LLM response time per model across both runs.</div>
      </div>
      <div class="card-body"><div id="latCmpLlmP50Bar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F4CA; LLM p90 Latency &mdash; Run A vs Run B
        <div class="card-sub">90th-percentile LLM response time per model across both runs.</div>
      </div>
      <div class="card-body"><div id="latCmpLlmP90Bar"></div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F50D; Tool p90 Latency &mdash; Run A vs Run B
      <div class="card-sub">90th-percentile tool latency across both runs.</div>
    </div>
    <div class="card-body"><div id="latCmpToolP90Bar"></div></div>
  </div>
</div>

<!-- ── TOKENS ────────────────────────────────────────────────────────────── -->
<div id="tab-tokens" class="tab-content">
  <div class="stat-grid" id="tokenStats"></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F4E5; ISL p50 &mdash; Run A vs Run B
        <div class="card-sub">Median prompt token count per model across both runs.</div>
      </div>
      <div class="card-body"><div id="tokenCmpIslP50Bar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F4E5; ISL p90 &mdash; Run A vs Run B
        <div class="card-sub">90th-percentile prompt tokens per model across both runs.</div>
      </div>
      <div class="card-body"><div id="tokenCmpIslP90Bar"></div></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F4E4; OSL p50 &mdash; Run A vs Run B
        <div class="card-sub">Median completion token count per model across both runs.</div>
      </div>
      <div class="card-body"><div id="tokenCmpOslP50Bar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F4E4; OSL p90 &mdash; Run A vs Run B
        <div class="card-sub">90th-percentile completion tokens per model across both runs.</div>
      </div>
      <div class="card-body"><div id="tokenCmpOslP90Bar"></div></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x26A1; TPS &mdash; Run A vs Run B
        <div class="card-sub">Completion tokens per second per model across both runs.</div>
      </div>
      <div class="card-body"><div id="tokenCmpTpsBar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F9DC; Cache Rate &mdash; Run A vs Run B
        <div class="card-sub">Fraction of prompt tokens served from cache per model.</div>
      </div>
      <div class="card-body"><div id="tokenCmpCacheBar"></div></div>
    </div>
  </div>
</div>

<!-- ── EFFICIENCY ─────────────────────────────────────────────────────────── -->
<div id="tab-efficiency" class="tab-content">
  <div class="card">
    <div class="card-header">&#x23F1;&#x1F4B0; Latency vs Cost &mdash; Run A vs Run B
      <div class="card-sub">Circles = Run A queries; diamonds = Run B queries. Top-right = slow and
        expensive in both runs.</div>
    </div>
    <div class="card-body"><div id="effCmpScatter"></div></div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F4B5; Cost per 1K Output Tokens &mdash; Run A vs Run B
      <div class="card-sub">Effective output cost per model. Lower = more efficient generation.</div>
    </div>
    <div class="card-body"><div id="effCmpCostPerKOslBar"></div></div>
  </div>
</div>

<!-- ── PER-QUERY DETAIL ──────────────────────────────────────────────────── -->
<div id="tab-detail" class="tab-content">
  <div class="card">
    <div class="card-header">&#x1F4CB; Per-Query Comparison &mdash; Run A vs Run B
      <div class="card-sub">All queries from both runs.
        Dimmed rows = query only in one run (&mdash; in missing columns).
        Coloured rows = queries in both runs, with cost delta.</div>
    </div>
    <div class="card-body" style="padding:0;overflow-x:auto">
      <table id="detailCmpTable">
        <thead><tr>
          <th>Query #</th><th>Question</th>
          <th>Cost A</th><th>Cost B</th><th>&#x0394; Cost</th><th>&#x0394; %</th>
          <th>ISL A</th><th>ISL B</th><th>OSL A</th><th>OSL B</th>
          <th>Dur A (s)</th><th>Dur B (s)</th>
          <th>Calls A</th><th>Calls B</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>
"""

_JS_DATA_EXTRAS = r"""
const cmp  = DATA.comparison;
"""

_JS_EXTRA_GLOBALS = r"""
const fmt$N = v => v == null ? '\u2014' : fmt$(v);
const fmtKN = v => v == null ? '\u2014' : fmtK(v);
const fmtTN = v => v == null ? '\u2014' : (+v).toFixed(1);
const fmtCN = v => v == null ? '\u2014' : String(v);

// ── grouped A-vs-B bar helper ─────────────────────────────────────────────────
function _cmpBar(divId, keysA, keysB, valA, valB, la, lb, ytitle, h) {
  const allKeys = [...new Set([...keysA, ...keysB])];
  const mapA = Object.fromEntries(keysA.map((k,i) => [k, valA[i]]));
  const mapB = Object.fromEntries(keysB.map((k,i) => [k, valB[i]]));
  Plotly.newPlot(divId, [
    { type:'bar', name:la, x:allKeys, y:allKeys.map(k=>mapA[k]||0), marker:{color:'#58a6ff'} },
    { type:'bar', name:lb, x:allKeys, y:allKeys.map(k=>mapB[k]||0), marker:{color:'#3fb950'} },
  ], L({ height:h||300, barmode:'group', yaxis:{title:ytitle},
         xaxis:{automargin:true,tickangle:-25}, margin:{t:20,r:20,b:90,l:70} }), CFG);
}
"""

_JS_INIT = r"""
// ── INIT ─────────────────────────────────────────────────────────────────────
const la = cmp.label_a;
const lb = cmp.label_b;
document.getElementById('headerMeta').textContent =
  la + ' \u2194 ' + lb + ' \u2022 ' + DATA.generated_at
  + ' \u2022 ' + cmp.num_common_queries + ' aligned queries';
"""

_JS_RENDERS = r"""
// ── OVERVIEW ─────────────────────────────────────────────────────────────────
function renderOverview() {
  const costSaved  = -cmp.cost_delta;
  const savingsPct = -cmp.cost_pct_change;
  const callsDelta = cmp.total_llm_calls_b - cmp.total_llm_calls_a;
  const ds = v => (v >= 0 ? '+' : '') + v;
  document.getElementById('overviewStats').innerHTML = `
    <div class="stat orange"><div class="label">Run A Total Cost</div>
      <div class="value">${fmt$(cmp.total_cost_a,2)}</div>
      <div class="sub">${cmp.label_a}</div></div>
    <div class="stat orange"><div class="label">Run B Total Cost</div>
      <div class="value">${fmt$(cmp.total_cost_b,2)}</div>
      <div class="sub">${cmp.label_b}</div></div>
    <div class="stat ${costSaved>=0?'green':'red'}"><div class="label">Cost Delta (B \u2212 A)</div>
      <div class="value">${costSaved>=0?'\u2212':'+'}${fmt$(Math.abs(cmp.cost_delta),2)}</div>
      <div class="sub">${
        savingsPct>=0?savingsPct.toFixed(1)+'% cheaper':(-savingsPct).toFixed(1)+'% costlier'
      }</div></div>
    <div class="stat blue"><div class="label">Aligned / A / B</div>
      <div class="value">${cmp.num_common_queries}</div>
      <div class="sub">of ${cmp.num_queries_a} A &amp; ${cmp.num_queries_b} B</div></div>
    <div class="stat purple"><div class="label">LLM Calls A \u2192 B</div>
      <div class="value">${cmp.total_llm_calls_a} \u2192 ${cmp.total_llm_calls_b}</div>
      <div class="sub">${ds(callsDelta)} calls</div></div>
    <div class="stat green"><div class="label">Cache Rate A / B</div>
      <div class="value">${(cmp.cache_rate_a*100).toFixed(1)}% / ${(cmp.cache_rate_b*100).toFixed(1)}%</div></div>
  `;

  // Cost by model A-vs-B
  const allMods = [...new Set([...Object.keys(cmp.by_model_a||{}), ...Object.keys(cmp.by_model_b||{})])];
  const filtMods = allMods.filter(m => (cmp.by_model_a[m]||0) > 0.0001 || (cmp.by_model_b[m]||0) > 0.0001);
  _cmpBar('overviewModelBar',
    filtMods, filtMods,
    filtMods.map(m=>cmp.by_model_a[m]||0), filtMods.map(m=>cmp.by_model_b[m]||0),
    la, lb, 'Cost (USD)', 280);

  // Cost by phase A-vs-B
  const allPhs = [...new Set([...Object.keys(cmp.by_phase_a||{}), ...Object.keys(cmp.by_phase_b||{})])];
  _cmpBar('overviewPhaseBar',
    allPhs, allPhs,
    allPhs.map(p=>cmp.by_phase_a[p]||0), allPhs.map(p=>cmp.by_phase_b[p]||0),
    la, lb, 'Cost (USD)', 280);

  // Per-query comparison table
  const pq = cmp.per_query || [];
  document.querySelector('#overviewTable tbody').innerHTML = pq.map(q => {
    const hasBoth = q.in_both;
    const dc   = !hasBoth ? '#8b949e' : (q.cost_delta <= 0 ? '#3fb950' : '#f85149');
    const dTxt = !hasBoth ? '\u2014'
               : (q.cost_delta<0?'\u2212':q.cost_delta>0?'+':'')+'$'+Math.abs(q.cost_delta).toFixed(4);
    const pStr = !hasBoth ? '\u2014' : (q.cost_pct<=0?'':'+') + q.cost_pct.toFixed(1) + '%';
    return `<tr${!hasBoth?' style="opacity:.65"':''}>
      <td><strong>${q.id}</strong></td>
      <td style="color:#d29922">${fmt$N(q.cost_a)}</td>
      <td style="color:#d29922">${fmt$N(q.cost_b)}</td>
      <td style="color:${dc};font-weight:600">${dTxt}</td>
      <td style="color:${dc}">${pStr}</td>
      <td>${fmtKN(q.isl_a)}</td><td>${fmtKN(q.isl_b)}</td>
      <td>${fmtKN(q.osl_a)}</td><td>${fmtKN(q.osl_b)}</td>
      <td>${fmtTN(q.duration_a)}</td><td>${fmtTN(q.duration_b)}</td>
      <td>${fmtCN(q.llm_calls_a)}</td><td>${fmtCN(q.llm_calls_b)}</td>
    </tr>`;
  }).join('');
}

// ── COST ─────────────────────────────────────────────────────────────────────
function renderCost() {
  // Cost by model A-vs-B
  const allMods = [...new Set([...Object.keys(cmp.by_model_a||{}), ...Object.keys(cmp.by_model_b||{})])];
  const filtMods = allMods.filter(m => (cmp.by_model_a[m]||0) > 0.0001 || (cmp.by_model_b[m]||0) > 0.0001);
  _cmpBar('costCmpModelBar',
    filtMods, filtMods,
    filtMods.map(m=>cmp.by_model_a[m]||0), filtMods.map(m=>cmp.by_model_b[m]||0),
    la, lb, 'Cost (USD)', 300);

  // Cost by phase A-vs-B
  const allPhs = [...new Set([...Object.keys(cmp.by_phase_a||{}), ...Object.keys(cmp.by_phase_b||{})])];
  _cmpBar('costCmpPhaseBar',
    allPhs, allPhs,
    allPhs.map(p=>cmp.by_phase_a[p]||0), allPhs.map(p=>cmp.by_phase_b[p]||0),
    la, lb, 'Cost (USD)', 300);

  // Tool cost A-vs-B (if available)
  const toolA = Object.keys((DATA.by_tool)||{}).filter(t => (DATA.by_tool[t].total_cost_usd||0) > 0);
  const toolBData = cmp.by_tool_b || {};
  const toolB = Object.keys(toolBData).filter(t => (toolBData[t].total_cost_usd||0) > 0);
  const allTools = [...new Set([...toolA, ...toolB])];
  if (allTools.length > 0) {
    _cmpBar('costCmpToolBar',
      allTools, allTools,
      allTools.map(t => (DATA.by_tool[t]||{}).total_cost_usd||0),
      allTools.map(t => (toolBData[t]||{}).total_cost_usd||0),
      la, lb, 'Cost (USD)', 280);
  } else {
    const tc = document.getElementById('costCmpToolCard');
    if (tc) tc.style.display = 'none';
  }

  // Per-query cost delta bar
  const pqBoth = (cmp.per_query||[]).filter(q => q.in_both);
  if (pqBoth.length) {
    Plotly.newPlot('costCmpDeltaBar', [{
      type:'bar', x:pqBoth.map(q=>'Q'+q.id), y:pqBoth.map(q=>q.cost_delta),
      text:pqBoth.map(q=>(q.cost_delta<=0?'':'+')+'$'+(+q.cost_delta).toFixed(4)),
      textposition:'outside',
      marker:{ color:pqBoth.map(q=>q.cost_delta<=0?'#3fb950':'#f85149') },
      hovertemplate:'Q%{x}: $%{y:.4f}<extra></extra>',
    }], L({ height:320,
            yaxis:{title:'Cost delta: B \u2212 A (USD)', zerolinecolor:'#8b949e', zeroline:true},
            xaxis:{automargin:true, tickangle:-45},
            margin:{t:20,r:20,b:110,l:70}, showlegend:false }), CFG);
  } else {
    document.getElementById('costCmpDeltaBar').innerHTML =
      '<p style="padding:40px;color:var(--muted);text-align:center">No queries aligned between the two runs.</p>';
  }
}

// ── LATENCY ──────────────────────────────────────────────────────────────────
function renderLatency() {
  // LLM p50 A-vs-B
  const llmA = DATA.llm_latency || {};
  const llmB = cmp.llm_latency_b || {};
  const allLlmModels = [...new Set([...Object.keys(llmA), ...Object.keys(llmB)])];
  _cmpBar('latCmpLlmP50Bar',
    allLlmModels, allLlmModels,
    allLlmModels.map(m => (llmA[m]||{}).p50_ms||0).map(v=>v/1000),
    allLlmModels.map(m => (llmB[m]||{}).p50_ms||0).map(v=>v/1000),
    la, lb, 'Seconds (p50)', 300);

  // LLM p90 A-vs-B
  _cmpBar('latCmpLlmP90Bar',
    allLlmModels, allLlmModels,
    allLlmModels.map(m => (llmA[m]||{}).p90_ms||0).map(v=>v/1000),
    allLlmModels.map(m => (llmB[m]||{}).p90_ms||0).map(v=>v/1000),
    la, lb, 'Seconds (p90)', 300);

  // Tool p90 A-vs-B
  const toolA = DATA.tool_latency || {};
  const toolB = cmp.tool_latency_b || {};
  const allTools = [...new Set([...Object.keys(toolA), ...Object.keys(toolB)])]
    .filter(t => ((toolA[t]||{}).p90_ms||0) > 10 || ((toolB[t]||{}).p90_ms||0) > 10);
  if (allTools.length) {
    _cmpBar('latCmpToolP90Bar',
      allTools, allTools,
      allTools.map(t => (toolA[t]||{}).p90_ms||0).map(v=>v/1000),
      allTools.map(t => (toolB[t]||{}).p90_ms||0).map(v=>v/1000),
      la, lb, 'Seconds (p90)', 300);
  } else {
    document.getElementById('latCmpToolP90Bar').innerHTML =
      '<p style="padding:40px;color:var(--muted);text-align:center">No significant tool latency data</p>';
  }
}

// ── TOKENS ───────────────────────────────────────────────────────────────────
function renderTokens() {
  const ts  = DATA.token_stats || {};
  const bm  = ts.by_model || {};
  const tsB = cmp.token_stats_b || {};
  const bmB = tsB.by_model || {};

  const models  = Object.keys(bm);
  const modelsB = Object.keys(bmB);
  const allModels = [...new Set([...models, ...modelsB])];

  // Token totals for Run A
  const totalPrompt  = models.reduce((s,m) => s + (bm[m].total_isl||0), 0);
  const totalComp    = models.reduce((s,m) => s + (bm[m].total_osl||0), 0);
  const totalCached  = models.reduce((s,m) => s + (bm[m].total_cached||0), 0);
  const totalCalls   = models.reduce((s,m) => s + (bm[m].calls||0), 0);
  const cacheRate    = totalPrompt > 0 ? (totalCached/totalPrompt*100).toFixed(1) : '0';

  // Token totals for Run B
  const totalPromptB = modelsB.reduce((s,m) => s + (bmB[m].total_isl||0), 0);
  const totalCompB   = modelsB.reduce((s,m) => s + (bmB[m].total_osl||0), 0);
  const totalCachedB = modelsB.reduce((s,m) => s + (bmB[m].total_cached||0), 0);
  const totalCallsB  = modelsB.reduce((s,m) => s + (bmB[m].calls||0), 0);
  const cacheRateB   = totalPromptB > 0 ? (totalCachedB/totalPromptB*100).toFixed(1) : '0';

  document.getElementById('tokenStats').innerHTML = `
    <div class="stat blue"><div class="label">LLM Calls A / B</div>
      <div class="value">${fmtK(totalCalls)} / ${fmtK(totalCallsB)}</div></div>
    <div class="stat orange"><div class="label">Total Prompt A / B</div>
      <div class="value">${fmtK(totalPrompt)} / ${fmtK(totalPromptB)}</div>
      <div class="sub">ISL tokens</div></div>
    <div class="stat green"><div class="label">Total Completion A / B</div>
      <div class="value">${fmtK(totalComp)} / ${fmtK(totalCompB)}</div>
      <div class="sub">OSL tokens</div></div>
    <div class="stat purple"><div class="label">Cache Rate A / B</div>
      <div class="value">${cacheRate}% / ${cacheRateB}%</div></div>
  `;

  // ISL p50 A-vs-B
  _cmpBar('tokenCmpIslP50Bar', allModels, allModels,
    allModels.map(m=>(bm[m]||{}).isl_p50||0), allModels.map(m=>(bmB[m]||{}).isl_p50||0),
    la, lb, 'ISL p50 (tokens)', 280);

  // ISL p90 A-vs-B
  _cmpBar('tokenCmpIslP90Bar', allModels, allModels,
    allModels.map(m=>(bm[m]||{}).isl_p90||0), allModels.map(m=>(bmB[m]||{}).isl_p90||0),
    la, lb, 'ISL p90 (tokens)', 280);

  // OSL p50 A-vs-B
  _cmpBar('tokenCmpOslP50Bar', allModels, allModels,
    allModels.map(m=>(bm[m]||{}).osl_p50||0), allModels.map(m=>(bmB[m]||{}).osl_p50||0),
    la, lb, 'OSL p50 (tokens)', 280);

  // OSL p90 A-vs-B
  _cmpBar('tokenCmpOslP90Bar', allModels, allModels,
    allModels.map(m=>(bm[m]||{}).osl_p90||0), allModels.map(m=>(bmB[m]||{}).osl_p90||0),
    la, lb, 'OSL p90 (tokens)', 280);

  // TPS A-vs-B
  _cmpBar('tokenCmpTpsBar', allModels, allModels,
    allModels.map(m=>(bm[m]||{}).tps_mean||0), allModels.map(m=>(bmB[m]||{}).tps_mean||0),
    la, lb, 'Completion tokens / second', 280);

  // Cache rate A-vs-B
  _cmpBar('tokenCmpCacheBar', allModels, allModels,
    allModels.map(m=>((bm[m]||{}).cache_rate||0)*100),
    allModels.map(m=>((bmB[m]||{}).cache_rate||0)*100),
    la, lb, 'Cache rate (%)', 280);
}

// ── EFFICIENCY ────────────────────────────────────────────────────────────────
function renderEfficiency() {
  const bm = (DATA.token_stats || {}).by_model || {};
  const models = Object.keys(bm);

  // Scatter: run A circles, run B diamonds
  const pqA = DATA.per_query || [];
  const pqBData = cmp.per_query || [];
  const scatterTraces = [];
  if (pqA.length) {
    scatterTraces.push({
      type: 'scatter', mode: 'markers+text',
      name: la,
      x: pqA.map(q => q.duration_s||0),
      y: pqA.map(q => q.cost_usd||0),
      text: pqA.map(q => 'Q'+q.id),
      textposition: 'top center',
      textfont: {size:9, color:'#8b949e'},
      marker: { symbol:'circle', color:'#58a6ff', size:9, opacity:.8 },
      hovertemplate: 'Run A Q%{text}<br>%{x:.1f}s  $%{y:.4f}<extra></extra>',
    });
  }
  const pqBBoth = pqBData.filter(q => q.in_both && q.duration_b != null && q.cost_b != null);
  if (pqBBoth.length) {
    scatterTraces.push({
      type: 'scatter', mode: 'markers+text',
      name: lb,
      x: pqBBoth.map(q => q.duration_b||0),
      y: pqBBoth.map(q => q.cost_b||0),
      text: pqBBoth.map(q => 'Q'+q.id),
      textposition: 'top center',
      textfont: {size:9, color:'#8b949e'},
      marker: { symbol:'diamond', color:'#3fb950', size:9, opacity:.8 },
      hovertemplate: 'Run B Q%{text}<br>%{x:.1f}s  $%{y:.4f}<extra></extra>',
    });
  }
  Plotly.newPlot('effCmpScatter', scatterTraces,
    L({ height:380, xaxis:{title:'Workflow duration (s)'},
        yaxis:{title:'Total cost (USD)'},
        margin:{t:20,r:20,b:60,l:70} }), CFG);

  // Cost per 1K OSL A-vs-B
  const bmB = (cmp.token_stats_b||{}).by_model || {};
  const allMods = [...new Set([...models, ...Object.keys(bmB)])];
  const cpkA = allMods.map(m => bm[m] && bm[m].total_osl > 0 ? (DATA.by_model[m]||0)/(bm[m].total_osl/1000) : 0);
  const cpkB = allMods.map(
    m => bmB[m] && bmB[m].total_osl > 0 ? ((cmp.by_model_b||{})[m]||0)/(bmB[m].total_osl/1000) : 0);
  _cmpBar('effCmpCostPerKOslBar',
    allMods, allMods, cpkA, cpkB,
    la, lb, '$ per 1K completion tokens', 300);
}

// ── PER-QUERY DETAIL ──────────────────────────────────────────────────────────
function renderDetail() {
  const pq = cmp.per_query || [];
  const tbody = document.querySelector('#detailCmpTable tbody');
  tbody.innerHTML = pq.map(q => {
    const hasBoth = q.in_both;
    const dc   = !hasBoth ? '#8b949e' : (q.cost_delta <= 0 ? '#3fb950' : '#f85149');
    const dTxt = !hasBoth ? '\u2014'
               : (q.cost_delta<0?'\u2212':q.cost_delta>0?'+':'')+'$'+Math.abs(q.cost_delta).toFixed(4);
    const pStr = !hasBoth ? '\u2014' : (q.cost_pct<=0?'':'+') + q.cost_pct.toFixed(1) + '%';
    const qtxt = (q.question||'').substring(0,80) + (q.question&&q.question.length>80?'\u2026':'');
    return `<tr${!hasBoth?' style="opacity:.65"':''}>
      <td><strong>${q.id}</strong></td>
      <td style="color:#8b949e;max-width:200px;word-break:break-word;font-size:12px">${qtxt||'\u2014'}</td>
      <td style="color:#d29922">${fmt$N(q.cost_a)}</td>
      <td style="color:#d29922">${fmt$N(q.cost_b)}</td>
      <td style="color:${dc};font-weight:600">${dTxt}</td>
      <td style="color:${dc}">${pStr}</td>
      <td>${fmtKN(q.isl_a)}</td>
      <td>${fmtKN(q.isl_b)}</td>
      <td>${fmtKN(q.osl_a)}</td>
      <td>${fmtKN(q.osl_b)}</td>
      <td>${fmtTN(q.duration_a)}</td>
      <td>${fmtTN(q.duration_b)}</td>
      <td>${fmtCN(q.llm_calls_a)}</td>
      <td>${fmtCN(q.llm_calls_b)}</td>
    </tr>`;
  }).join('');
}
"""

_HTML = build_html(
    title="Deep Researcher Agent Tokenomics Report \u2014 Comparison",
    tab_html=_TAB_HTML,
    js_data_extras=_JS_DATA_EXTRAS,
    js_extra_globals=_JS_EXTRA_GLOBALS,
    js_init=_JS_INIT,
    js_renders=_JS_RENDERS,
)


def render_html(report_data: dict) -> str:
    return _HTML.replace("__REPORT_DATA_JSON__", json.dumps(report_data, ensure_ascii=False))
