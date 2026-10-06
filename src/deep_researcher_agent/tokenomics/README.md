# Deep Researcher Agent Tokenomics

Post-eval analysis module for the Deep Research Agent. Parses NeMo Relay ATOF JSONL, attributes costs and token counts to workflow phases (Orchestrator / Planner / Researcher), and renders a self-contained interactive HTML report.

---

## Background

### Subagent attribution

The adapter follows Relay `parent_uuid` ancestry. LLM calls below real
`planner-agent` and `researcher-agent` scopes are attributed directly to those
phases; other calls are attributed to the orchestrator. No timing-window
inference or NAT callback renaming is required.

---

## File structure

```
src/deep_researcher_agent/tokenomics/
├── pricing.py       # PricingRegistry — maps model names to per-token prices
├── profile.py       # RequestProfile, PhaseStats — structured data classes
├── atof_adapter.py  # parse_trace() — Relay ATOF JSONL → list[RequestProfile]
└── report.py        # generate_report() — builds and renders HTML dashboard
```

---

## Pricing configuration

Pricing lives in a YAML file under `tokenomics.pricing`. Prices are in **USD per 1 million tokens**.


```yaml
# frontends/benchmarks/deepresearch_bench/configs/config_tokenomics_pricing.yml

tokenomics:
  pricing:
    models:
      "nvidia/nemotron-3-ultra-550b-a55b":
        # NVIDIA-hosted access for this example; not self-hosting cost.
        input_per_1m_tokens: 0.00
        output_per_1m_tokens: 0.00
    tools:
      # Tavily pay-as-you-go rates. Monthly plans have a lower effective rate.
      "web_search_tool":
        cost_per_call: 0.008
      "advanced_web_search_tool":
        cost_per_call: 0.016
      # Serper Starter; change for the selected tier/provider.
      "paper_search":
        cost_per_call: 0.001
```

Model name lookup is: exact match → substring match → default. This means a key of `"nemotron-3-ultra"` will match a trace model name of `"nvidia/nemotron-3-ultra-550b-a55b"`.

Tool name lookup tries exact names before substring matching. Unknown tools
default to $0/call, so free internal tools can be omitted. Configure only the
provider-facing tool scope to avoid charging both a wrapper and its underlying
API call.

---

## Generating a report

Run after a Relay-instrumented workflow completes. The default ATOF sink writes `relay/deep-researcher-relay.atof.jsonl`.

```bash
PYTHONPATH=src python -m deep_researcher_agent.tokenomics.report \
    --trace relay/deep-researcher-relay.atof.jsonl \
    --config frontends/benchmarks/deepresearch_bench/configs/config_tokenomics_pricing.yml \
    [--output path/to/report.html]
```

If `--output` is omitted, the report is written to `<trace_dir>/tokenomics_report.html`.

If `standardized_data_all.csv` exists in the same directory as the trace, it is automatically loaded to enrich the report with any additional NOVA metadata fields.

---

## Report tabs

### 📊 Overview
Top-level stat cards (total cost, cache savings, token totals, LLM call count) plus a per-query summary table and cost split by model and phase.

### 💰 Cost
- **Cost split by model** — donut chart of budget allocation
- **Cost by phase** — which of Orchestrator / Planner / Researcher drove most spend
- **Cost by phase per query (stacked bar)** — spots outlier queries and which phase drove the spike
- **Per-query cost histogram** — shape of cost distribution (shown only when ≥ 10 queries; wide right tail = high query difficulty variance)

### ⏱ Latency
- **LLM latency p50/p90/p99 by model** — a large gap between p50 and p99 means occasional very long completions; if p50 is already slow the bottleneck is network or server load
- **Tool latency p50/p90/p99** — search/web tools typically 3–8 s; p90 > 10 s is a retrieval bottleneck

### 🪙 Tokens
The most detailed tab. All statistics are across individual LLM call observations (not per-request aggregates), so distributions are meaningful even for small query sets.

| Chart | What to look for |
|-------|-----------------|
| **ISL p50/p90/p99 by model** | Rising p99 vs p50 = some calls hit much larger contexts |
| **OSL p50/p90/p99 by model** | High p99 OSL = long reasoning chains or verbose outputs driving latency and cost |
| **Context accumulation (ISL by call index)** | Upward slope = history building up; plateau = caching or fresh-start; dashed line = estimated system-prompt floor |
| **Throughput (TPS by model)** | Low TPS with small OSL = network overhead, not slow generation |
| **Token budget (cache breakdown)** | Green = cached (cheaper); grey = uncached; blue = completion. Maximise green. |
| **ISL vs latency scatter** | Diagonal trend = prompt-bound; flat cloud = compute-bound |
| **Token mix by phase** | Which phase consumes tokens and how much is cached per phase |
| **Predicted vs Actual OSL** | Shown only when `NOVA-Predicted-OSL` contains real pre-call estimates (hidden when post-hoc filled) |

### 📐 Efficiency
Latency/cost joint analysis:

| Chart | What to look for |
|-------|-----------------|
| **Latency vs cost per query** | Top-right outliers are slow *and* expensive — highest-priority targets for optimization |
| **TPS vs ISL** | Downward slope = prompt-bound inference; KV-cache optimizations would help |
| **Effective cost per 1K output tokens** | True output cost after accounting for actual generation volume |
| **Model efficiency bubble** | Each bubble = one model; bottom-left = cheapest + fastest. Use for model selection trade-off analysis. |

### 🏷 Pricing
Configured prices visualised as bar charts (input and output $/1M) plus a full pricing table.

### 📋 Per-Query
Full per-query table: cost, ISL, OSL, cached tokens, ISL:OSL ratio, LLM call count, workflow duration, and the question text.

---

## Python API

```python
from deep_researcher_agent.tokenomics import parse_trace, PricingRegistry

# Load pricing from the tokenomics YAML (not the nat eval config)
import yaml
with open("frontends/benchmarks/deepresearch_bench/configs/config_tokenomics_pricing.yml") as f:
    config = yaml.safe_load(f)
pricing = PricingRegistry.from_dict(config["tokenomics"]["pricing"])

# Parse trace → one RequestProfile per query
profiles = parse_trace("relay/deep-researcher-relay.atof.jsonl", pricing)

for prof in profiles:
    print(f"Query {prof.request_index}: ${prof.total_cost_usd:.4f}, "
          f"{prof.total_prompt_tokens:,} ISL, {prof.total_completion_tokens:,} OSL, "
          f"{prof.cache_hit_rate:.1%} cache hit")

    for ps in prof.phases:
        print(f"  {ps.phase} / {ps.model}: {ps.llm_calls} calls, ${ps.cost_usd:.4f}")
```

### Key data classes

**`RequestProfile`** — one per query
- `request_index`, `question`, `duration_s`
- `total_cost_usd`, `total_cache_savings_usd`
- `total_prompt_tokens`, `total_cached_tokens`, `total_completion_tokens`
- `phases: list[PhaseStats]` — per `(phase, model)` pair
- `tool_calls: dict[str, int]` — tool name → invocation count
- `llm_call_events: list[dict]` — per-call observations with `isl`, `osl`, `cached`, `dur_s`, `tps`, `model`, `phase`, `call_idx`, `uuid`
- `tool_call_events: list[dict]` — per-call observations with `tool`, `dur_s`

**`PhaseStats`** — one per `(phase, model)` pair within a request
- `phase` — one of `"orchestrator"`, `"planner-agent"`, `"researcher-phase"`
- `model`, `llm_calls`, `prompt_tokens`, `cached_tokens`, `completion_tokens`
- `cost_usd`, `cache_savings_usd`
- Properties: `cache_hit_rate`, `uncached_tokens`, `total_tokens`

**`PricingRegistry`**
- `PricingRegistry.from_dict(raw_dict)` — construct from the `tokenomics.pricing` config dict
- `registry.get(model_name) -> ModelPrice` — exact → substring → default lookup
- `registry.known_models() -> list[str]`
