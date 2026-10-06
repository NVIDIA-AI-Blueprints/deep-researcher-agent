<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Observability with NeMo Relay

Deep Researcher Agent uses [NeMo Relay](https://docs.nvidia.com/nemo/relay) as its observability runtime.

Relay gives Deep Researcher Agent users four complementary views:

- **Developer logs** show agent, LLM, and tool activity in the terminal.
- **ATOF JSONL** provides a durable, machine-readable event stream for debugging
  and post-processing.
- **OpenTelemetry (OTEL)** exports the same scope tree to backends such as
  Phoenix.
- **Pricing enrichment** attaches model cost data when a configured catalog
  matches the observed model.

Logging, ATOF export, full observability payloads, and redaction are enabled by
default. OTEL export is opt-in so a default Deep Researcher Agent installation does not attempt
to contact an observability server. Pricing is enabled with no catalog sources;
token usage is still recorded, but monetary cost is not estimated until you
configure prices.

## Installation

NeMo Relay and the LangChain, LangGraph, and Deep Agents integrations are
installed with Deep Researcher Agent:

```bash
./scripts/setup.sh
```

For an existing development checkout, synchronize the environment:

```bash
uv sync
```

Verify the installed version:

```bash
uv run python -c 'from importlib.metadata import version; print(version("nemo-relay"))'
```

Deep Researcher Agent supports the Relay version range pinned in `pyproject.toml`.

## Configuration walkthrough

Relay is configured under the top-level Deep Researcher Agent workflow:

```yaml
workflow:
  _type: chat_deepresearcher_agent
  relay:
    logging: true
    observability:
      enable_full_payloads: true
      atof:
        enabled: true
        output_directory: ./relay
        filename: deep-researcher-relay.atof.jsonl
        mode: append
      opentelemetry:
        enabled: false
    redaction:
      enabled: true
```

Most users can omit this block and use the defaults. Add only the settings that
you intend to change.

| Setting | Default | Purpose |
|---|---:|---|
| `logging` | `true` | Register Deep Researcher Agent's Relay console subscriber. |
| `observability.enable_full_payloads` | `true` | Preserve supported inputs, outputs, metadata, and annotated usage for sanitization and export. |
| `observability.atof.enabled` | `true` | Write Relay events to ATOF JSONL. |
| `observability.atof.mode` | `append` | Preserve events across turns and async jobs. Use `overwrite` only for a single isolated run. |
| `observability.opentelemetry.enabled` | `false` | Export Relay scopes over OTEL when explicitly enabled. |
| `redaction.enabled` | `true` | Sanitize supported sensitive values before logs and exporters receive them. |
| `pricing.sources` | `[]` | Model pricing catalogs used to enrich Relay usage. |

Relay configuration is strict. Unknown fields, invalid endpoint URLs, and an
invalid source shape fail workflow validation instead of being ignored.

### Change the ATOF output path

Set `output_directory` and `filename` independently:

```yaml
workflow:
  relay:
    observability:
      atof:
        enabled: true
        output_directory: ./observability/traces
        filename: deep-researcher-development.atof.jsonl
        mode: append
```

Relative directories are resolved from the working directory where Deep Researcher Agent is
started. Use an absolute directory for containers, services, or async workers
when their working directories might differ:

```yaml
output_directory: /var/lib/deep-researcher/relay
```

The resulting file is
`/var/lib/deep-researcher/relay/deep-researcher-development.atof.jsonl`. Ensure every worker can write
to the directory. Keep `mode: append` when multiple user turns or async jobs
share a file; choose a unique filename instead of `overwrite` when you need
per-run isolation.

## Inspect ATOF traces

By default, Deep Researcher Agent appends events to:

```text
relay/deep-researcher-relay.atof.jsonl
```

Each line is one JSON event. Use `jq` to inspect it:

```bash
# Follow new events while Deep Researcher Agent runs.
tail -f relay/deep-researcher-relay.atof.jsonl | jq -c .

# Count scope starts by category.
jq -s '
  [.[] | select(.kind == "scope" and .scope_category == "start")]
  | group_by(.category)
  | map({category: .[0].category, count: length})
' relay/deep-researcher-relay.atof.jsonl

# Show LLM usage recorded on completed LLM scopes.
jq -c '
  select(.category == "llm" and .scope_category == "end")
  | {
      model: .category_profile.annotated_response.model,
      usage: .category_profile.annotated_response.usage,
      status: .metadata["otel.status_code"]
    }
' relay/deep-researcher-relay.atof.jsonl

# Find scopes that do not have exactly one start and one end.
jq -s '
  [.[] | select(.kind == "scope")]
  | sort_by(.uuid)
  | group_by(.uuid)
  | map({
      uuid: .[0].uuid,
      name: .[0].name,
      starts: map(select(.scope_category == "start")) | length,
      ends: map(select(.scope_category == "end")) | length
    })
  | map(select(.starts != 1 or .ends != 1))
' relay/deep-researcher-relay.atof.jsonl
```

An empty final result from the balance check means every recorded scope closed
exactly once. ATOF `mode: append` is important for web and async-job testing:
multiple worker processes can initialize exporters, and `overwrite` can replace
events written by an earlier job.

## Relay logging subscriber

Deep Researcher Agent registers a process-wide Relay subscriber when `workflow.relay.logging` is
enabled. It reads the sanitized Relay lifecycle stream and renders developer logs:

```text
[Chain Start] shallow_research_agent
[AGENT] model-name
[Tool Calls] 1 tool(s) requested
  → web_search_tool
[Tokens] prompt=1882, completion=42, model=model-name
[Tool Start] web_search_tool
[Tool Result] chars=8472 ref=sha256:...
[Chain End] shallow_research_agent
```

The subscriber does not instrument the workflow itself. Relay's maintained
framework integrations and Deep Researcher Agent's semantic agent/tool scopes produce events;
the subscriber only formats those events. This keeps console logging aligned
with ATOF and OTEL rather than maintaining a second callback-based trace.

Raw prompts, responses, tool arguments, and tool results are not printed.
Instead, the subscriber logs sizes and stable content references after Relay
redaction. Set the normal console log level under `general.telemetry.logging`.

## Export Relay traces to Phoenix

[Phoenix](https://docs.arize.com/phoenix) provides a local UI for inspecting the
Relay hierarchy, latency, model inputs and outputs, tool calls, token usage, and
errors.

Start Phoenix:

```bash
uvx --from arize-phoenix phoenix serve
```

Phoenix is available at [http://localhost:6006](http://localhost:6006). OTEL is
commented out in the default Deep Researcher Agent configs. Uncomment or add this Relay block:

```yaml
workflow:
  relay:
    observability:
      opentelemetry:
        enabled: true
        endpoints:
          - type: openinference
            endpoint: http://localhost:6006/v1/traces
            service_name: deep-researcher-relay
            resource_attributes:
              openinference.project.name: deep-researcher-relay
              deployment.environment: development
```

The `openinference` projection gives Phoenix semantic LLM, agent, and tool span
attributes and the corresponding UI icons. `openinference.project.name`
selects the Phoenix project. Use a distinct project name for each Deep Researcher Agent
environment that you want to compare independently.

Relay also supports `full` and `gen_ai` OTEL projections. Use `full` when the
destination needs the richest Relay-native attributes, and `gen_ai` when the
destination expects OpenTelemetry GenAI semantic conventions. Phoenix users
should normally use `openinference`.

### Troubleshoot missing Phoenix traces

If a trace does not appear in the expected Phoenix project:

1. Check that Phoenix is listening at the configured endpoint.
2. Confirm `opentelemetry.enabled: true` and inspect the Deep Researcher Agent log for export
   errors.
3. Look in Phoenix's `default` project and any project configured globally on
   the machine.
4. Inspect `~/.config/nemo-relay/plugins.toml`. NeMo Relay automatically
   discovers user-level plugin configuration. If Relay was already configured
   for another application or coding agent, that exporter can send the Deep Researcher Agent
   trace to its globally configured Phoenix project instead of the project you
   are currently viewing.
5. Compare the Phoenix trace with the local ATOF file. If ATOF contains the
   scopes, instrumentation worked and the remaining issue is OTEL destination,
   project selection, export, or batching.

Keep personal Relay configuration when it is needed by other applications.
Use an Deep Researcher Agent-specific project in the workflow configuration and account for all
discovered exporters when validating where telemetry is sent.

## Read Deep Researcher Agent traces

Deep Researcher Agent creates one root trace for each user turn. Multiple turns in the same
conversation have different trace IDs and share the Phoenix `session.id`, so
the session view groups them without merging their execution trees.

An async deep-research job runs in a separate Relay trace because it executes
outside the request task, often in another Dask worker process. Job metadata
links the background trace to the submitted job and originating request.

A typical deep-research trace is structured as follows:

```text
<workflow>
└── chat_deepresearcher_agent
    ├── intent_classifier
    │   └── LLM
    ├── clarifier_agent
    │   ├── LLM
    │   └── tool
    └── deep_research_agent
        ├── planner-agent
        │   └── LLM
        ├── researcher-agent
        │   ├── LLM
        │   └── tool
        └── writer-agent
            └── LLM
```

Use the tree in this order:

1. Start at the root and check its terminal status and duration.
2. Find the slowest agent, LLM, or tool child.
3. Inspect LLM spans for model, token usage, response status, and sanitized
   input/output attributes.
4. Inspect tool spans for tool name, duration, sanitized arguments/results, and
   errors.
5. For parallel researchers, compare sibling spans rather than adding their
   wall-clock durations.
6. For a failed async job, search by `deep-researcher.job.id` and confirm the root scope has
   one start, one end, and an `ERROR` terminal status.

Internal graph-routing nodes are represented as decision metadata/events where
possible rather than noisy agent spans. Framework-generated names can still
appear when the underlying integration exposes a real execution boundary.

## Redaction and privacy

Relay redaction runs before the Deep Researcher Agent logging subscriber, ATOF sink, and OTEL
exporter. The default detectors cover common credentials and personal data.
Deep Researcher Agent can also request privacy-mode sanitization for supported `data` and
`category_profile` payloads through request privacy context.

Redaction reduces accidental disclosure; it is not a substitute for auditing
the destination's access controls, retention, and data policy. Validate every
configured exported attribute with synthetic sensitive values before enabling
full payloads in a production environment.

## Pricing and cost analysis

Default Deep Researcher Agent configs omit the Relay pricing block. The resulting empty source
list records token usage without claiming a monetary cost. Pricing depends on
the provider, deployment, contract, region, cache policy, and date.

Use the dedicated example when you want model cost enrichment:

```bash
nat serve \
  --config_file configs/nemo_relay/config_web_default_with_pricing.yml \
  --port 8000
```

That config loads `configs/nemo_relay/relay_pricing_catalog.json`:

```yaml
pricing:
  enabled: true
  sources:
    - type: file
      path: configs/nemo_relay/relay_pricing_catalog.json
```

Relay matches the observed provider/model name against the catalog and adds
cost information to the annotated LLM usage. Review and date every rate before
using it operationally. A zero-dollar hosted API rate does not mean that a
self-hosted deployment has no infrastructure cost.

Relay's pricing catalog covers model usage. Deep Researcher Agent's tokenomics report also
supports per-call prices for external tools such as web search. After a run,
generate the report with:

```bash
PYTHONPATH=src python -m deep_researcher_agent.tokenomics.report \
  --trace relay/deep-researcher-relay.atof.jsonl \
  --config frontends/benchmarks/deepresearch_bench/configs/config_tokenomics_pricing.yml
```

The report uses Relay-attributed model cost when present and the report pricing
configuration as a fallback. It also calculates configured tool API charges
and writes a self-contained HTML report. See [Profiling and Cost
Analysis](../profiling/index.md) for report fields, phase attribution, and
pricing maintenance.

## Validate the configuration

Validate an edited workflow before starting Deep Researcher Agent:

```bash
uv run python .agents/skills/deep-researcher-configure-workflow/scripts/validate_config.py \
  configs/config_web_default_llamaindex.yml
```

Then start Deep Researcher Agent, run one shallow turn and one deep-research job, and verify all
three views that you enabled:

- console logs show agent, LLM, and tool lifecycle activity;
- ATOF contains balanced scopes and annotated LLM usage;
- Phoenix shows the expected project, per-turn traces, shared session grouping,
  and an independent async-job trace.
