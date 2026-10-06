<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ResultChart

Renders agent-authored chart specs as themed, accessible inline SVG charts, with
a "Show data" table and CSV export. The agent writes a small JSON spec in a
fenced code block in its answer; this module parses, validates, normalizes, and
draws it. There is no code execution and no charting-library dependency.

## How it is wired

`MarkdownRenderer` routes two fenced languages to this module:

- ` ```chart ` -> a single chart (or a KPI card)
- ` ```chart-carousel ` -> a pageable set of related line charts

```tsx
import { ChartBlock, fenceBareSpecs } from '@/shared/components/ResultChart'

// In MarkdownRenderer, chart/chart-carousel code fences become:
<ChartBlock raw={codeContent} fallback={fallback} />
```

`fenceBareSpecs(markdown)` also wraps any standalone chart-spec JSON line the
agent forgot to fence, so it still renders instead of showing raw JSON. A
malformed spec falls back to a normal code block, so nothing breaks.

## The chart-spec contract

The agent emits JSON inside the fence. `ChartBlock` picks the renderer by shape:
a full spec draws a chart (or a KPI card when it conveys no comparison), a
KPI-only block draws headline tiles, and a carousel draws pageable line charts.

### Chart spec

```json
{
  "type": "bar",
  "title": "GPUs by Model",
  "subtitle": "Active fleet inventory",
  "x": { "key": "model" },
  "y": { "label": "GPUs", "format": "compact" },
  "series": [{ "key": "gpus", "color": "green" }],
  "data": [
    { "model": "H100", "gpus": 4200 },
    { "model": "A100", "gpus": 3100 },
    { "model": "L40S", "gpus": 1800 }
  ]
}
```

| Field | Notes |
| --- | --- |
| `type` | `bar`, `hbar`, `line`, `area`, `grouped-bar`, `delta` |
| `title` | required; `subtitle` optional |
| `x` | `{ key, label? }`; `key` must resolve on the data rows |
| `y` | `{ label?, format? }`; `format` is `number`, `compact`, `percent`, or `currency` |
| `series` | 1 to 6 of `{ key, label?, color? }`; `color` is `green`, `blue`, `amber`, `red`, or `neutral` |
| `data` | 1 to 60 row objects keyed by `x.key` and each `series.key` |
| `kpis` | optional headline tiles (see below) |

`delta` diverges around zero and colors gains green / losses red by sign;
`grouped-bar` draws a legend for its multiple series.

### KPI-only spec

For a single value or a one-entity result where a chart would compare nothing:

```json
{
  "title": "Fleet Health",
  "kpis": [
    { "label": "Total GPUs", "value": "10,000", "sub": "across 4 models", "tone": "accent" },
    { "label": "At-Risk Nodes", "value": "23", "tone": "warn" }
  ]
}
```

`tone` is `default`, `accent`, `warn`, or `alarm`.

### Chart carousel

```json
{
  "title": "Revenue Forecast: Baseline vs Prediction",
  "charts": [
    { "type": "line", "title": "Baseline vs Prediction", "x": { "key": "month" }, "y": { "format": "currency" },
      "series": [{ "key": "baseline", "color": "neutral" }, { "key": "prediction", "color": "green" }],
      "data": [ { "month": "Jan", "baseline": 5200000, "prediction": 5300000 } ] },
    { "type": "line", "title": "Units Shipped", "x": { "key": "month" }, "y": { "format": "compact" },
      "series": [{ "key": "units", "color": "blue" }],
      "data": [ { "month": "Jan", "units": 1800 } ] }
  ]
}
```

A carousel holds 2 to 12 line charts and pages between them.

## Module layout

| File | Responsibility |
| --- | --- |
| `ChartBlock.tsx` | entry: parse a raw fence, pick chart / KPI / carousel, or fall back |
| `types.ts` | the zod spec schemas and exported types |
| `parse.ts` | parse + validate specs; `fenceBareSpecs`, `toNumber` |
| `normalize.ts` | safe axis defaults, top-N truncation, KPI degeneration |
| `scale.ts`, `geometry.ts` | domains, formatting, plot geometry |
| `renderers/` | per-type primitive marks (axis, bar, hbar, line) |
| `ResultChart.tsx` | the SVG shell: maps marks, tooltips, gradient fills, toolbar |
| `ResultChartCarousel.tsx` | the pageable carousel |
| `ChartKpi.tsx` | KPI cards and tiles |
| `ChartToolbar.tsx`, `ChartDataTable.tsx` | Show-data toggle and CSV export |

## Safety and accessibility

- Magnitude bars start at zero; large numbers are compacted on the axis.
- Rankings past the row cap show a disclosed "top N of M" note, never a silent drop.
- Missing points render as gaps, never an implied zero.
- Colors are a fixed enum and all text is escaped React content, so an
  agent-authored spec has no injection surface.
- The CSV export neutralizes spreadsheet formula injection while preserving real
  negative values.

## Portability

The module is self-contained apart from two deep-researcher integration points: the five
toolbar/carousel icons from `@/adapters/ui/icons`, and the `.result-chart*` CSS
rules in `src/app/globals.css`. To reuse it elsewhere, inline those two and the
module renders standalone.

## Tests

`*.spec.ts(x)` cover the module at 100% statements, branches, functions, and
lines (enforced by a scoped Vitest threshold; global coverage is unaffected).
