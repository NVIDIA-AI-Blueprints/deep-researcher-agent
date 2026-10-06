---
name: forecast-analysis
description: >
  Use this skill during research for lightweight forecast evidence analysis: base-rate checks, prediction-market anchors, scenario ranges, directional factor summaries, implied probabilities, and monitoring indicators. Triggers: "forecast", "prediction", "probability", "odds", "base rate", "scenario", "Polymarket", "prediction market", "will happen", "market-implied", "confidence interval", "price target", "expected value". Outputs: forecast evidence notes or compact forecast inputs returned in your ResearchNotes for writer synthesis.
---

# Forecast Analysis Skill

Use this skill when a researcher worker needs to prepare forecast evidence, not when the writer is drafting the final answer. The goal is to make the forecast inputs explicit, auditable, and easy for synthesis to use.

## Required Execution Standard

1. Separate evidence into anchors, supporting factors, opposing factors, uncertainty drivers, and monitoring indicators.
2. Prefer explicit anchors from sources, such as prediction-market prices, base rates, latest measured values, official projections, analyst forecasts, or recent trend data.
3. Use `execute` for any arithmetic: probability conversion, expected value, weighted scenario averages, interval arithmetic, or base-rate adjustments.
4. Do not invent a final forecast when the assigned ResearchQuery only asks for evidence. Capture what the evidence implies and preserve uncertainty.
5. Include compact forecast inputs in your `ResearchNotes` only after any calculation has succeeded.
6. In `ResearchNotes`, cite original source IDs for every forecast anchor and material factor.

## Forecast Evidence Map

Use this shape for the saved artifact when useful:

```json
{
  "forecast_question": "...",
  "target_variable": "probability | value | date | threshold | option",
  "anchors": [
    {
      "label": "Prediction market price",
      "value": "62%",
      "source_ref": "source id or URL",
      "timestamp_or_date": "..."
    }
  ],
  "supporting_factors": ["..."],
  "opposing_factors": ["..."],
  "uncertainties": ["..."],
  "monitoring_indicators": ["..."],
  "calculation_notes": "..."
}
```

## Lightweight Scenario Template

Use `execute` for scenario arithmetic:

```python
scenarios = [
    {"name": "upside", "probability": 0.25, "value": 80},
    {"name": "base", "probability": 0.50, "value": 55},
    {"name": "downside", "probability": 0.25, "value": 30},
]

probability_sum = sum(item["probability"] for item in scenarios)
expected_value = sum(item["probability"] * item["value"] for item in scenarios)

print(f"Probability sum: {probability_sum:.3f}")
print(f"Scenario-weighted expected value: {expected_value:.2f}")
for item in scenarios:
    contribution = item["probability"] * item["value"]
    print(f"- {item['name']}: contribution {contribution:.2f}")
```

## ResearchNotes Guidance

When returning `ResearchNotes`, include:
- one finding for the main anchor,
- one finding for supporting factors,
- one finding for opposing factors or caveats,
- a gap if there is no recent anchor or if source data is stale.

Do not treat prediction-market prices as guaranteed truth. They are evidence of market-implied expectations at a point in time and should be labeled as such.
