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

"""HTML template and render helper for single-run tokenomics reports. No project imports."""

from __future__ import annotations

import json

from ._report_base import build_html

_TAB_HTML = r"""
<!-- ── OVERVIEW ─────────────────────────────────────────────────────────── -->
<div id="tab-overview" class="tab-content active">
  <div class="stat-grid" id="overviewStats"></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F916; Cost by Model
        <div class="card-sub" id="overviewModelBarSub">Which model is consuming most of the budget?</div>
      </div>
      <div class="card-body"><div id="overviewModelBar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F3D7; Cost by Phase
        <div class="card-sub" id="overviewPhaseBarSub">Orchestrator = reasoning overhead;
          Researcher = parallel search calls. High Researcher share means many tool-heavy sub-tasks.</div>
      </div>
      <div class="card-body"><div id="overviewPhaseBar"></div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header" id="overviewTableHeader">&#x1F4CB; Per-Query Summary</div>
    <div class="card-body" style="padding:0">
      <table id="overviewTable">
        <thead><tr id="overviewTableHead">
          <th>Query #</th><th>Cost ($)</th><th>Prompt (ISL)</th><th>Completion (OSL)</th>
          <th>Cached</th><th>Cache %</th><th>LLM Calls</th><th>Duration (s)</th>
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
      <div class="card-header">&#x1F967; Cost Split by Model
        <div class="card-sub">Hover for exact values. A single dominant slice means one model drives
          nearly all spend.</div>
      </div>
      <div class="card-body"><div id="costPie"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F3D7; Cost by Phase
        <div class="card-sub">Total spend per phase summed across all queries. Orchestrator dominance
          is normal; unexpectedly high Researcher cost suggests overly broad search loops.</div>
      </div>
      <div class="card-body"><div id="costPhaseBar"></div></div>
    </div>
  </div>
  <div id="toolCostCard" class="card">
    <div class="card-header">&#x1F50D; Tool API Cost by Tool
      <div class="card-sub">Per-call cost &#xD7; invocation count for each tool. These charges are separate
        from LLM token costs. High search costs relative to LLM costs suggest reducing max_results or
        switching to a cheaper search provider.</div>
    </div>
    <div class="card-body"><div id="toolCostBar"></div></div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F4E6; Cost by Phase per Query
      <div class="card-sub">Spot outlier queries and identify which phase drove the extra cost. Uniform
        bars = consistent workload; spikes = difficult queries.</div>
    </div>
    <div class="card-body"><div id="costPerQueryStack"></div></div>
  </div>
</div>

<!-- ── LATENCY ───────────────────────────────────────────────────────────── -->
<div id="tab-latency" class="tab-content">
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F4CA; LLM Latency Percentiles by Model
        <div class="card-sub">A large gap between p50 and p99 means occasional very long completions &mdash;
          usually caused by high OSL. If p50 is already slow, the bottleneck is network or server load.
        </div>
      </div>
      <div class="card-body"><div id="llmLatencyBar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F50D; Tool Latency Percentiles
        <div class="card-sub">Search/web tools typically run 3&ndash;8 s. p90 above 10 s signals a retrieval
          bottleneck that adds directly to total query time.</div>
      </div>
      <div class="card-body"><div id="toolLatencyBar"></div></div>
    </div>
  </div>
</div>

<!-- ── TOKENS ────────────────────────────────────────────────────────────── -->
<div id="tab-tokens" class="tab-content">
  <div class="stat-grid" id="tokenStats"></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F4E5; ISL (Input Sequence Length) &mdash; p50 / p90 / p99 by Model
        <div class="card-sub">Prompt token counts sent to each model. A rising p99 vs p50 means some calls
          hit much larger contexts &mdash; check ISL Growth below to see when.</div>
      </div>
      <div class="card-body"><div id="islBar"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F4E4; OSL (Output Sequence Length) &mdash; p50 / p90 / p99 by Model
        <div class="card-sub">Completion token counts. High p99 OSL means some calls produce very long
          reasoning chains or verbose outputs, which directly drives both cost and latency.</div>
      </div>
      <div class="card-body"><div id="oslBar"></div></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F4C8; Context Accumulation &mdash; Avg ISL by Call Index
        <div class="card-sub">How prompt size grows over sequential LLM calls within a query. An upward
          slope means the model is accumulating conversation history. A plateau suggests caching or a
          fresh-start pattern. The dashed line is the estimated system-prompt floor (minimum ISL
          observed).</div>
      </div>
      <div class="card-body"><div id="islGrowth"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x26A1; Throughput &mdash; Completion Tokens / Second
        <div class="card-sub">Inference speed per model. Low TPS with small OSL often indicates network
          round-trip overhead rather than slow generation. Compare models to spot which is the throughput
          bottleneck.</div>
      </div>
      <div class="card-body"><div id="tpsBar"></div></div>
    </div>
  </div>
  <div id="predVsActualCard" class="card">
    <div class="card-header">&#x1F52E; NOVA-Predicted vs Actual OSL
      <div class="card-sub">Each dot is one LLM call. Points on the diagonal line = perfect prediction.
        Points above = model generated more than predicted (underestimate). Points below = model generated
        less (overestimate). Tight clustering around the diagonal means NAT's routing hints are accurate.
      </div>
    </div>
    <div class="card-body"><div id="predVsActualScatter"></div></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x1F9DC; Token Budget &mdash; Cached vs Uncached vs Completion
        <div class="card-sub">Green = tokens served from cache (billed at the cheaper cached rate). Grey
          = uncached prompt tokens (full price). Blue = completion tokens (most expensive per token).
          Maximise green to reduce cost.</div>
      </div>
      <div class="card-body"><div id="cacheBreakdown"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F517; ISL vs Latency &mdash; Is Prompt Size the Bottleneck?
        <div class="card-sub">Each dot is one LLM call. A diagonal trend means longer prompts take longer
          (prompt-bound). A flat cloud means latency is driven by output length or server capacity, not
          context size.</div>
      </div>
      <div class="card-body"><div id="islLatencyScatter"></div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F3D7; Token Mix by Phase
      <div class="card-sub">Total tokens consumed across Orchestrator / Planner / Researcher phases.
        Cached (green) vs uncached (grey) prompt tokens show how well each phase leverages the prompt
        cache. Reasoning tokens (purple) are non-billed thinking tokens where applicable.</div>
    </div>
    <div class="card-body"><div id="componentTokenStack"></div></div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F4CB; Token Summary Table (by model)</div>
    <div class="card-body" style="padding:0;overflow-x:auto">
      <table id="tokenTable">
        <thead><tr>
          <th>Model</th><th>Calls</th>
          <th>Avg ISL</th><th>p90 ISL</th><th>Max ISL</th>
          <th>Avg OSL</th><th>p90 OSL</th><th>Max OSL</th>
          <th>Total Prompt</th><th>Total Completion</th>
          <th>Total Cached</th><th>Cache Rate</th>
          <th>Avg TPS</th><th>Sys Prompt Est.</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ── EFFICIENCY ─────────────────────────────────────────────────────────── -->
<div id="tab-efficiency" class="tab-content">
  <div class="card">
    <div class="card-header">&#x23F1;&#x1F4B0; Latency vs Cost per Query
      <div class="card-sub">Each dot is one query. Queries in the top-right are both slow and expensive &mdash;
        highest priority for optimization. A diagonal cluster means slow queries are inherently costlier
        (more LLM calls). Outliers far from the cluster are worth investigating individually.</div>
    </div>
    <div class="card-body"><div id="latCostScatter"></div></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-header">&#x26A1;&#x1F4C9; TPS vs ISL &mdash; Does Throughput Drop as Context Grows?
        <div class="card-sub">Each dot is one LLM call. A downward slope means longer prompts hurt
          inference speed (prompt-bound). A flat cloud means generation speed is independent of context
          size (compute-bound). Use this to decide whether KV-cache optimizations would help.</div>
      </div>
      <div class="card-body"><div id="tpsIslScatter"></div></div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F4B5; Effective Cost per 1K Output Tokens by Model
        <div class="card-sub">Total spend divided by total completion tokens generated &mdash; the true output
          cost. A model with cheaper listed pricing may still be more expensive here if it generates more
          tokens to answer the same question.</div>
      </div>
      <div class="card-body"><div id="costPerKOslBar"></div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header">&#x1F3AF; Model Efficiency &mdash; Output Cost vs p90 Latency
      <div class="card-sub">Each point is a model. Bottom-left is ideal: cheap output AND fast. Use this
        to compare model trade-offs when evaluating alternatives. Bubble size = total LLM call count.
      </div>
    </div>
    <div class="card-body"><div id="modelEfficiencyScatter"></div></div>
  </div>
</div>

<!-- ── PER-QUERY DETAIL ──────────────────────────────────────────────────── -->
<div id="tab-detail" class="tab-content">
  <div class="card">
    <div class="card-header">&#x1F4CB; Per-Query Token &amp; Cost Detail</div>
    <div class="card-body" style="padding:0;overflow-x:auto">
      <table id="detailTable">
        <thead><tr>
          <th>Query #</th><th>Cost ($)</th>
          <th>Prompt (ISL)</th><th>Completion (OSL)</th><th>Cached</th>
          <th>ISL:OSL</th><th>LLM Calls</th><th>Duration (s)</th>
          <th>Question</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>
"""

_JS_INIT = r"""
// ── INIT ─────────────────────────────────────────────────────────────────────
document.getElementById('headerMeta').textContent =
  DATA.label + ' \u2022 ' + DATA.generated_at + ' \u2022 ' + DATA.num_queries + ' queries';
"""

_JS_RENDERS = r"""
// ── OVERVIEW ─────────────────────────────────────────────────────────────────
function renderOverview() {
  const d = DATA;
  const cr = d.total_prompt_tokens > 0
    ? (d.total_cached_tokens / d.total_prompt_tokens * 100).toFixed(1) : '0';
  document.getElementById('overviewStats').innerHTML = `
    <div class="stat blue"><div class="label">Queries</div>
      <div class="value">${d.num_queries}</div></div>
    <div class="stat orange"><div class="label">Total Cost</div>
      <div class="value">${fmt$(d.total_cost_usd,2)}</div>
      <div class="sub">${fmt$(d.avg_cost_usd,4)}/query avg</div></div>
    <div class="stat orange"><div class="label">LLM Cost</div>
      <div class="value">${fmt$(d.llm_cost_usd,2)}</div>
      <div class="sub">token charges</div></div>
    <div class="stat orange"><div class="label">Tool API Cost</div>
      <div class="value">${fmt$(d.tool_cost_usd,2)}</div>
      <div class="sub">search / external APIs</div></div>
    <div class="stat green"><div class="label">Cache Savings</div>
      <div class="value">${fmt$(d.cache_savings_usd,2)}</div>
      <div class="sub">${cr}% cache rate</div></div>
    <div class="stat orange"><div class="label">Total Prompt</div>
      <div class="value">${fmtK(d.total_prompt_tokens)}</div>
      <div class="sub">ISL tokens</div></div>
    <div class="stat blue"><div class="label">Total Completion</div>
      <div class="value">${fmtK(d.total_completion_tokens)}</div>
      <div class="sub">OSL tokens</div></div>
    <div class="stat purple"><div class="label">Total LLM Calls</div>
      <div class="value">${d.total_llm_calls}</div></div>
  `;

  // Cost by model bar
  const mods = Object.keys(d.by_model).filter(m => d.by_model[m] > 0.0001);
  Plotly.newPlot('overviewModelBar', [{
    type: 'bar', x: mods, y: mods.map(m => d.by_model[m]),
    text: mods.map(m => fmt$(d.by_model[m],3)), textposition: 'outside',
    marker: { color: PALETTE },
  }], L({ height: 280, yaxis: {title:'Cost (USD)'}, xaxis: {automargin:true,tickangle:-25},
           margin: {t:20,r:20,b:90,l:70}, showlegend: false }), CFG);

  // Cost by phase horizontal bar
  const phases = Object.keys(d.by_phase);
  Plotly.newPlot('overviewPhaseBar', [{
    type: 'bar', orientation: 'h',
    y: phases, x: phases.map(p => d.by_phase[p]),
    text: phases.map(p => fmt$(d.by_phase[p],3)), textposition: 'outside',
    marker: { color: '#bc8cff' },
  }], L({ height: 280, xaxis: {title:'Cost (USD)'}, yaxis: {automargin:true},
           margin: {t:20,r:80,b:50,l:140} }), CFG);

  // Per-query table
  const tbody = document.querySelector('#overviewTable tbody');
  tbody.innerHTML = d.per_query.map(q => {
    const cr2 = q.input_tokens > 0 ? (q.cached_tokens/q.input_tokens*100).toFixed(1)+'%' : '\u2014';
    return `<tr>
      <td><strong>${q.id}</strong></td>
      <td style="color:#d29922">${fmt$(q.cost_usd)}</td>
      <td>${(q.input_tokens||0).toLocaleString()}</td>
      <td>${(q.output_tokens||0).toLocaleString()}</td>
      <td style="color:#39d353">${(q.cached_tokens||0).toLocaleString()}</td>
      <td style="color:#39d353">${cr2}</td>
      <td>${q.entry_count||0}</td>
      <td>${(q.duration_s||0).toFixed(1)}</td>
    </tr>`;
  }).join('');
}

// ── COST ─────────────────────────────────────────────────────────────────────
function renderCost() {
  const d = DATA;
  const mods = Object.keys(d.by_model).filter(m => d.by_model[m] > 0.0001);

  // Donut by model
  Plotly.newPlot('costPie', [{
    type: 'pie', labels: mods, values: mods.map(m => d.by_model[m]),
    hole: .45, textfont: { color: '#e6edf3' },
    marker: { colors: PALETTE },
  }], L({ height: 320, showlegend: true, margin: {t:20,r:120,b:20,l:20} }), CFG);

  // Horizontal bar by phase
  const phases = Object.keys(d.by_phase);
  Plotly.newPlot('costPhaseBar', [{
    type: 'bar', orientation: 'h',
    y: phases, x: phases.map(p => d.by_phase[p]),
    text: phases.map(p => fmt$(d.by_phase[p],3)), textposition: 'outside',
    marker: { color: '#bc8cff' },
  }], L({ height: 320, xaxis:{title:'Cost (USD)'}, yaxis:{automargin:true},
           margin:{t:20,r:80,b:50,l:140} }), CFG);

  // Tool cost bar
  const toolData = d.by_tool || {};
  const toolCard = document.getElementById('toolCostCard');
  const toolNames = Object.keys(toolData).filter(t => toolData[t].total_cost_usd > 0 || toolData[t].calls > 0);
  if (toolNames.length > 0) {
    const toolCosts = toolNames.map(t => toolData[t].total_cost_usd);
    const toolCalls = toolNames.map(t => toolData[t].calls);
    const hasCost = toolCosts.some(c => c > 0);
    if (hasCost) {
      Plotly.newPlot('toolCostBar', [{
        type: 'bar', x: toolNames, y: toolCosts,
        text: toolNames.map((t,i) => fmt$(toolCosts[i],3) + ' (' + toolCalls[i] + ' calls)'),
        textposition: 'outside',
        marker: { color: '#39d353' },
      }], L({ height: 280, yaxis:{title:'Cost (USD)'}, xaxis:{automargin:true,tickangle:-25},
              margin:{t:20,r:20,b:90,l:70}, showlegend:false }), CFG);
    } else {
      // Show call counts even when all costs are $0 (tools not priced)
      Plotly.newPlot('toolCostBar', [{
        type: 'bar', x: toolNames, y: toolCalls,
        text: toolCalls.map(c => c + ' calls'), textposition: 'outside',
        marker: { color: '#58a6ff' },
      }], L({ height: 280, yaxis:{title:'Call Count'}, xaxis:{automargin:true,tickangle:-25},
              margin:{t:20,r:20,b:90,l:70}, showlegend:false }), CFG);
      if (toolCard) {
        const sub = toolCard.querySelector('.card-sub');
        if (sub) {
          sub.textContent =
            'Tool call counts shown (no cost data \u2014 add tool prices to '
            + 'tokenomics.pricing.tools in the config to see cost breakdown).';
        }
      }
    }
  } else {
    if (toolCard) toolCard.style.display = 'none';
  }

  // Stacked bar: cost by phase per query
  const stackTraces = (d.phase_order||[]).map(ph => ({
    type: 'bar', name: ph,
    x: d.per_query.map(q => 'Q' + q.id),
    y: d.per_query.map(q => (q.by_phase||{})[ph]||0),
    marker: { color: PHASE_COLORS[ph]||'#8b949e' },
  }));
  Plotly.newPlot('costPerQueryStack', stackTraces,
    L({ height: 300, barmode: 'stack', yaxis:{title:'Cost (USD)'},
        xaxis:{automargin:true,tickangle:-25}, margin:{t:20,r:20,b:90,l:70} }), CFG);
}

// ── LATENCY ──────────────────────────────────────────────────────────────────
function renderLatency() {
  const d = DATA;

  // LLM percentile bars
  const llmE = Object.entries(d.llm_latency||{}).sort((a,b) => b[1].p90_ms - a[1].p90_ms);
  if (llmE.length) {
    const names = llmE.map(e => e[0]);
    Plotly.newPlot('llmLatencyBar', [
      { type:'bar', name:'p50', x:names, y:llmE.map(e=>e[1].p50_ms/1000), marker:{color:'#3fb950'} },
      { type:'bar', name:'p90', x:names, y:llmE.map(e=>e[1].p90_ms/1000), marker:{color:'#58a6ff'} },
      { type:'bar', name:'p99', x:names, y:llmE.map(e=>e[1].p99_ms/1000), marker:{color:'#f85149'} },
    ], L({ height:320, barmode:'group', yaxis:{title:'Seconds'}, xaxis:{automargin:true,tickangle:-30},
           margin:{t:20,r:20,b:100,l:60} }), CFG);
  } else {
    document.getElementById('llmLatencyBar').innerHTML =
      '<p style="padding:40px;color:var(--muted);text-align:center">'
      + 'No LLM latency data (missing span_event_timestamp?)</p>';
  }

  // Tool percentile bars (skip near-zero tools)
  const toolE = Object.entries(d.tool_latency||{})
    .filter(([k,v]) => v.p90_ms > 10)
    .sort((a,b) => b[1].p90_ms - a[1].p90_ms)
    .slice(0, 12);
  if (toolE.length) {
    const tnames = toolE.map(e => e[0]);
    Plotly.newPlot('toolLatencyBar', [
      { type:'bar', name:'p50', x:tnames, y:toolE.map(e=>e[1].p50_ms/1000), marker:{color:'#3fb950'} },
      { type:'bar', name:'p90', x:tnames, y:toolE.map(e=>e[1].p90_ms/1000), marker:{color:'#58a6ff'} },
      { type:'bar', name:'p99', x:tnames, y:toolE.map(e=>e[1].p99_ms/1000), marker:{color:'#f85149'} },
    ], L({ height:320, barmode:'group', yaxis:{title:'Seconds'}, xaxis:{automargin:true,tickangle:-30},
           margin:{t:20,r:20,b:100,l:60} }), CFG);
  } else {
    document.getElementById('toolLatencyBar').innerHTML =
      '<p style="padding:40px;color:var(--muted);text-align:center">No significant tool latency data</p>';
  }
}

// ── TOKENS ───────────────────────────────────────────────────────────────────
function renderTokens() {
  const ts  = DATA.token_stats || {};
  const bm  = ts.by_model || {};
  const bc  = ts.by_component || {};
  const spl = ts.isl_latency_sample || [];
  const grw = ts.isl_growth || {};
  const sys = ts.sys_prompt_est || {};

  const models = Object.keys(bm);
  const colorOf = m => PALETTE[models.indexOf(m) % PALETTE.length];

  // Stat grid
  const totalPrompt = models.reduce((s,m) => s + (bm[m].total_isl||0), 0);
  const totalComp   = models.reduce((s,m) => s + (bm[m].total_osl||0), 0);
  const totalCached = models.reduce((s,m) => s + (bm[m].total_cached||0), 0);
  const totalCalls  = models.reduce((s,m) => s + (bm[m].calls||0), 0);
  const cacheRate   = totalPrompt > 0 ? (totalCached/totalPrompt*100).toFixed(1) : '0';

  document.getElementById('tokenStats').innerHTML = `
    <div class="stat blue"><div class="label">Total LLM Calls</div>
      <div class="value">${fmtK(totalCalls)}</div></div>
    <div class="stat orange"><div class="label">Total Prompt</div>
      <div class="value">${fmtK(totalPrompt)}</div>
      <div class="sub">ISL tokens</div></div>
    <div class="stat green"><div class="label">Total Completion</div>
      <div class="value">${fmtK(totalComp)}</div>
      <div class="sub">OSL tokens</div></div>
    <div class="stat purple"><div class="label">Total Cached</div>
      <div class="value">${fmtK(totalCached)}</div>
      <div class="sub">${cacheRate}% cache rate</div></div>
    <div class="stat blue"><div class="label">ISL:OSL Ratio</div>
      <div class="value">${totalComp > 0 ? (totalPrompt/totalComp).toFixed(1) : '\u2014'}:1</div></div>
  `;

  // ISL p50/p90/p99 by model
  Plotly.newPlot('islBar', [
    { type:'bar', name:'p50', x:models, y:models.map(m=>bm[m].isl_p50||0), marker:{color:'#3fb950'} },
    { type:'bar', name:'p90', x:models, y:models.map(m=>bm[m].isl_p90||0), marker:{color:'#58a6ff'} },
    { type:'bar', name:'p99', x:models, y:models.map(m=>bm[m].isl_p99||0), marker:{color:'#f85149'} },
  ], L({ height:300, barmode:'group', yaxis:{title:'Tokens'}, xaxis:{automargin:true,tickangle:-25},
    margin:{t:20,r:20,b:90,l:70},
    annotations: models.map(m => ({
      x:m, y:bm[m].isl_max||0, text:'max '+fmtK(bm[m].isl_max||0),
      showarrow:false, font:{size:9,color:'#8b949e'}, yshift:4,
    }))
  }), CFG);

  // OSL p50/p90/p99 by model
  Plotly.newPlot('oslBar', [
    { type:'bar', name:'p50', x:models, y:models.map(m=>bm[m].osl_p50||0), marker:{color:'#3fb950'} },
    { type:'bar', name:'p90', x:models, y:models.map(m=>bm[m].osl_p90||0), marker:{color:'#58a6ff'} },
    { type:'bar', name:'p99', x:models, y:models.map(m=>bm[m].osl_p99||0), marker:{color:'#f85149'} },
  ], L({ height:300, barmode:'group', yaxis:{title:'Tokens'}, xaxis:{automargin:true,tickangle:-25},
    margin:{t:20,r:20,b:90,l:70} }), CFG);

  // ISL growth (context accumulation)
  const growthTraces = Object.entries(grw).map(([model, pts], i) => ({
    type:'scatter', mode:'lines+markers', name: model,
    x: pts.map(p=>p.idx), y: pts.map(p=>p.avg_isl),
    line: { color: colorOf(model), width: 2 },
    marker: { size: 5, color: colorOf(model) },
    hovertemplate: model + '<br>Call #%{x}<br>Avg ISL: %{y:,.0f} tokens<extra></extra>',
  }));
  // Dashed sys-prompt estimate lines
  Object.entries(sys).forEach(([model, minIsl], i) => {
    const maxIdx = Math.max(...((grw[model]||[{idx:0}]).map(p=>p.idx)), 10);
    growthTraces.push({
      type:'scatter', mode:'lines', name: model+' sys-prompt est.',
      x: [0, maxIdx], y: [minIsl, minIsl],
      line: { color: colorOf(model), width: 1, dash: 'dot' },
      hovertemplate: 'Sys-prompt lower bound: ' + fmtK(minIsl) + ' tokens<extra></extra>',
    });
  });
  Plotly.newPlot('islGrowth', growthTraces,
    L({ height:320, xaxis:{title:'Call index within query', dtick:5}, yaxis:{title:'Avg ISL (tokens)'},
        margin:{t:20,r:20,b:60,l:80},
        annotations:[{text:'Dashed = system-prompt lower bound (min ISL observed)',
          x:0.01, y:0.97, xref:'paper', yref:'paper', showarrow:false,
          font:{color:'#8b949e',size:10}}]
    }), CFG);

  // TPS bar
  const tpsSorted = models.map(m=>({m, tps:bm[m].tps_mean||0})).sort((a,b)=>b.tps-a.tps);
  Plotly.newPlot('tpsBar', [{
    type:'bar', x:tpsSorted.map(d=>d.m), y:tpsSorted.map(d=>d.tps),
    text:tpsSorted.map(d=>d.tps.toFixed(1)+' tok/s'), textposition:'outside',
    marker:{color: tpsSorted.map((_,i) => PALETTE[i%PALETTE.length])},
  }], L({ height:300, yaxis:{title:'Completion tokens / second'},
    xaxis:{automargin:true,tickangle:-25}, margin:{t:20,r:20,b:90,l:70}, showlegend:false }), CFG);

  // Cache breakdown stacked bar
  Plotly.newPlot('cacheBreakdown', [
    {
      type:'bar', name:'Cached prompt', x:models,
      y:models.map(m=>bm[m].total_cached||0), marker:{color:'#39d353'},
    },
    {
      type:'bar', name:'Uncached prompt', x:models,
      y:models.map(m=>(bm[m].total_isl||0)-(bm[m].total_cached||0)),
      marker:{color:'#30363d'},
    },
    {
      type:'bar', name:'Completion', x:models,
      y:models.map(m=>bm[m].total_osl||0), marker:{color:'#58a6ff'},
    },
  ], L({ height:320, barmode:'stack', yaxis:{title:'Tokens'},
    xaxis:{automargin:true,tickangle:-25}, margin:{t:20,r:20,b:90,l:80} }), CFG);

  // ISL vs Latency scatter
  const modelsSeen = [...new Set(spl.map(p=>p.model))];
  const scatterTraces = modelsSeen.map(m => {
    const pts = spl.filter(p=>p.model===m);
    return {
      type:'scatter', mode:'markers', name:m,
      x: pts.map(p=>p.isl), y: pts.map(p=>p.dur_s),
      marker: { color: colorOf(m), size: 5, opacity: .55 },
      hovertemplate: 'ISL: %{x:,}<br>Latency: %{y:.2f}s<extra>'+m+'</extra>',
    };
  });
  Plotly.newPlot('islLatencyScatter', scatterTraces,
    L({ height:320, xaxis:{title:'Prompt tokens (ISL)'}, yaxis:{title:'Latency (s)'},
        margin:{t:20,r:20,b:60,l:70} }), CFG);

  // Component token stacked bar (by phase)
  const comps = Object.keys(bc);
  Plotly.newPlot('componentTokenStack', [
    {
      type:'bar', name:'Prompt (uncached)', x:comps,
      y:comps.map(c=>(bc[c].total_isl||0)-(bc[c].total_cached||0)),
      marker:{color:'#30363d'},
    },
    {
      type:'bar', name:'Prompt (cached)', x:comps,
      y:comps.map(c=>bc[c].total_cached||0), marker:{color:'#39d353'},
    },
    {
      type:'bar', name:'Completion', x:comps,
      y:comps.map(c=>bc[c].total_osl||0), marker:{color:'#58a6ff'},
    },
    {
      type:'bar', name:'Reasoning', x:comps,
      y:comps.map(c=>bc[c].total_reasoning||0), marker:{color:'#bc8cff'},
    },
  ], L({ height:320, barmode:'stack', yaxis:{title:'Tokens'},
    xaxis:{automargin:true,tickangle:-25}, margin:{t:20,r:20,b:90,l:80} }), CFG);

  // Predicted vs Actual OSL scatter
  const pva = ts.predicted_vs_actual || [];
  const predCard = document.getElementById('predVsActualCard');
  // Hide when all predicted == actual (post-hoc filled, no predictive signal)
  const hasRealPredictions = pva.some(p => p.predicted !== p.actual);
  if (pva.length === 0 || !hasRealPredictions) {
    if (predCard) predCard.style.display = 'none';
  } else {
    const pvaModels = [...new Set(pva.map(p => p.model))];
    const pvaTraces = pvaModels.map(m => {
      const pts = pva.filter(p => p.model === m);
      const errs = pts.map(p => p.actual - p.predicted);
      const pct = pts.map(p => p.predicted > 0 ? ((p.actual - p.predicted) / p.predicted * 100) : 0);
      return {
        type: 'scatter', mode: 'markers', name: m,
        x: pts.map(p => p.predicted), y: pts.map(p => p.actual),
        customdata: pts.map((p, i) => [errs[i], pct[i].toFixed(1)]),
        marker: { color: colorOf(m), size: 5, opacity: .65 },
        hovertemplate:
          'Predicted: %{x:,}<br>Actual: %{y:,}<br>Error: %{customdata[0]:+,} '
          + '(%{customdata[1]}%)<extra>' + m + '</extra>',
      };
    });
    // Perfect-prediction diagonal
    const allVals = pva.flatMap(p => [p.predicted, p.actual]);
    const axMax = Math.max(...allVals) * 1.05;
    pvaTraces.push({
      type: 'scatter', mode: 'lines', name: 'Perfect prediction',
      x: [0, axMax], y: [0, axMax],
      line: { color: '#8b949e', width: 1, dash: 'dot' },
      hoverinfo: 'skip',
    });
    Plotly.newPlot('predVsActualScatter', pvaTraces,
      L({ height: 360, xaxis: {title: 'NOVA Predicted OSL (tokens)', range: [0, axMax]},
          yaxis: {title: 'Actual OSL (tokens)', range: [0, axMax]},
          margin: {t:20,r:20,b:60,l:80} }), CFG);
  }

  // Token summary table
  const tbody = document.querySelector('#tokenTable tbody');
  tbody.innerHTML = models.map(m => {
    const s = bm[m];
    const est = sys[m];
    return `<tr>
      <td><strong>${m}</strong></td>
      <td>${(s.calls||0).toLocaleString()}</td>
      <td>${fmtK(s.isl_mean||0)}</td>
      <td style="color:#58a6ff">${fmtK(s.isl_p90||0)}</td>
      <td style="color:#f85149">${fmtK(s.isl_max||0)}</td>
      <td>${fmtK(s.osl_mean||0)}</td>
      <td style="color:#58a6ff">${fmtK(s.osl_p90||0)}</td>
      <td style="color:#f85149">${fmtK(s.osl_max||0)}</td>
      <td style="color:#8b949e">${fmtK(s.total_isl||0)}</td>
      <td style="color:#8b949e">${fmtK(s.total_osl||0)}</td>
      <td style="color:#39d353">${fmtK(s.total_cached||0)}</td>
      <td style="color:#39d353">${((s.cache_rate||0)*100).toFixed(1)}%</td>
      <td>${(s.tps_mean||0).toFixed(1)} tok/s</td>
      <td style="color:#d29922;font-style:italic"
        title="Min ISL observed \u2014 lower bound on system-prompt size">~${
        est != null ? fmtK(est) : 'N/A'}</td>
    </tr>`;
  }).join('');
}

// ── EFFICIENCY ────────────────────────────────────────────────────────────────
function renderEfficiency() {
  const d = DATA;
  const bm = (d.token_stats || {}).by_model || {};
  const ll = d.llm_latency || {};
  const spl = (d.token_stats || {}).isl_latency_sample || [];
  const models = Object.keys(bm);

  // Per-query latency vs cost scatter
  const pq = d.per_query || [];
  const costs = pq.map(q => q.cost_usd || 0);
  const durs  = pq.map(q => q.duration_s || 0);
  Plotly.newPlot('latCostScatter', [{
    type: 'scatter', mode: 'markers+text',
    x: durs, y: costs,
    text: pq.map(q => 'Q' + q.id),
    textposition: 'top center',
    textfont: { size: 10, color: '#8b949e' },
    marker: {
      color: costs,
      colorscale: 'Viridis',
      size: 10, opacity: .8,
      colorbar: { title: 'Cost ($)', thickness: 12, len: .7 },
    },
    hovertemplate: 'Query %{text}<br>Duration: %{x:.1f}s<br>Cost: $%{y:.4f}<extra></extra>',
  }], L({ height: 380, xaxis: {title: 'Workflow duration (s)'},
          yaxis: {title: 'Total cost (USD)'},
          margin: {t:20,r:80,b:60,l:70}, showlegend: false }), CFG);

  // TPS vs ISL scatter (from isl_latency_sample)
  const modelsSeen = [...new Set(spl.map(p => p.model))];
  const tpsIslTraces = modelsSeen.map(m => {
    const pts = spl.filter(p => p.model === m && p.dur_s > 0 && p.osl > 0);
    return {
      type: 'scatter', mode: 'markers', name: m,
      x: pts.map(p => p.isl),
      y: pts.map(p => p.osl / p.dur_s),
      marker: { color: PALETTE[models.indexOf(m) % PALETTE.length], size: 5, opacity: .55 },
      hovertemplate: 'ISL: %{x:,}<br>TPS: %{y:.1f}<extra>' + m + '</extra>',
    };
  });
  Plotly.newPlot('tpsIslScatter', tpsIslTraces,
    L({ height: 340, xaxis: {title: 'Prompt tokens (ISL)'},
        yaxis: {title: 'Completion tokens / second (TPS)'},
        margin: {t:20,r:20,b:60,l:70} }), CFG);

  // Effective cost per 1K output tokens
  const cpk = models.map(m => ({
    m,
    val: bm[m].total_osl > 0 ? (d.by_model[m] || 0) / (bm[m].total_osl / 1000) : 0,
  })).sort((a, b) => b.val - a.val);
  Plotly.newPlot('costPerKOslBar', [{
    type: 'bar', x: cpk.map(d => d.m), y: cpk.map(d => d.val),
    text: cpk.map(d => '$' + d.val.toFixed(4)), textposition: 'outside',
    marker: { color: cpk.map((_, i) => PALETTE[i % PALETTE.length]) },
  }], L({ height: 300, yaxis: {title: '$ per 1K completion tokens'},
          xaxis: {automargin: true, tickangle: -25},
          margin: {t:20,r:20,b:90,l:80}, showlegend: false }), CFG);

  // Model efficiency: output cost vs p90 latency bubble
  const effModels = models.filter(m => bm[m].total_osl > 0 && ll[m]);
  if (effModels.length > 0) {
    const cpkMap = Object.fromEntries(cpk.map(d => [d.m, d.val]));
    Plotly.newPlot('modelEfficiencyScatter', [{
      type: 'scatter', mode: 'markers+text',
      x: effModels.map(m => (ll[m].p90_ms || 0) / 1000),
      y: effModels.map(m => cpkMap[m] || 0),
      text: effModels.map(m => m.split('/').pop()),
      textposition: 'top center',
      textfont: { size: 11 },
      marker: {
        size: effModels.map(m => Math.max(14, Math.min(50, (bm[m].calls || 0) / 5))),
        color: effModels.map((_, i) => PALETTE[i % PALETTE.length]),
        opacity: .8, line: {width: 1, color: '#30363d'},
      },
      hovertemplate: effModels.map(m =>
        '<b>' + m + '</b><br>p90 latency: ' + ((ll[m].p90_ms||0)/1000).toFixed(1) + 's<br>' +
        'Cost/1K out: $' + (cpkMap[m]||0).toFixed(4) + '<br>Calls: ' + (bm[m].calls||0) + '<extra></extra>'),
    }], L({ height: 380, xaxis: {title: 'p90 LLM latency (s)'},
            yaxis: {title: '$ per 1K completion tokens'},
            margin: {t:20,r:20,b:60,l:80}, showlegend: false,
            annotations: [{text: 'Bubble size = call count. Bottom-left = cheapest + fastest.',
              x: .01, y: .99, xref: 'paper', yref: 'paper', showarrow: false,
              font: {color: '#8b949e', size: 10}}] }), CFG);
  } else {
    document.getElementById('modelEfficiencyScatter').innerHTML =
      '<p style="padding:40px;color:var(--muted);text-align:center">Not enough model diversity for comparison</p>';
  }
}

// ── PER-QUERY DETAIL ──────────────────────────────────────────────────────────
function renderDetail() {
  const tbody = document.querySelector('#detailTable tbody');
  tbody.innerHTML = DATA.per_query.map(q => {
    const isl = q.input_tokens||0, osl = q.output_tokens||0;
    const ratio = osl > 0 ? (isl/osl).toFixed(1)+':1' : '\u2014';
    const qtxt = q.question ? q.question.substring(0,120)+(q.question.length>120?'\u2026':'') : '\u2014';
    return `<tr>
      <td><strong>${q.id}</strong></td>
      <td style="color:#d29922">${fmt$(q.cost_usd)}</td>
      <td>${isl.toLocaleString()}</td>
      <td>${osl.toLocaleString()}</td>
      <td style="color:#39d353">${(q.cached_tokens||0).toLocaleString()}</td>
      <td>${ratio}</td>
      <td>${q.entry_count||0}</td>
      <td>${(q.duration_s||0).toFixed(1)}</td>
      <td style="color:#8b949e;max-width:400px;word-break:break-word">${qtxt}</td>
    </tr>`;
  }).join('');
}
"""

_HTML = build_html(
    title="Deep Researcher Agent Tokenomics Report",
    tab_html=_TAB_HTML,
    js_data_extras="",
    js_extra_globals="",
    js_init=_JS_INIT,
    js_renders=_JS_RENDERS,
)


def render_html(report_data: dict) -> str:
    return _HTML.replace("__REPORT_DATA_JSON__", json.dumps(report_data, ensure_ascii=False))
