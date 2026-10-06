---
name: lightweight-calculation
description: >
  Use this skill for small deterministic calculations during research when pandas/table analysis is unnecessary. Triggers: "calculate", "arithmetic", "unit conversion", "percentage point", "expected value", "weighted average", "range", "ratio", "sanity check", "implied value", "probability conversion". Outputs: concise calculation notes, JSON snippets, or Markdown bullets returned in your ResearchNotes for later synthesis.
---

# Lightweight Calculation Skill

Use this skill when the research task needs a small reproducible calculation but does not need full table normalization. Keep the calculation narrow and source-grounded.

## Required Execution Standard

1. Identify the exact input values and their source references.
2. Use `execute` with a short Python script for arithmetic, ratios, probability conversion, expected value, weighted averages, confidence/range arithmetic, or unit conversion.
3. Do not hand-compute values in prose when the arithmetic affects a finding.
4. Use `/workspace` for sandbox-local files. Sandbox code cannot read or write `/shared/` directly.
5. Return the final result in your `ResearchNotes` after the successful `execute` call (not via `write_file`); `run_research_batch` persists your returned notes to `/shared/`.
6. State assumptions, rounding rules, missing inputs, and source references.

## Execution Pattern

1. Gather the relevant figures from source-tool output or research notes.
2. Run a compact Python calculation with explicit variables.
3. Inspect output and fix any code issue before using the result.
4. Include a short calculation summary (Markdown or JSON) in your `ResearchNotes` for synthesis.
5. Cite the original source IDs in the eventual `ResearchFinding`; the calculation artifact is supporting work, not a substitute for sources.

## Python Template

```python
from decimal import Decimal, ROUND_HALF_UP

inputs = {
    "market_probability": Decimal("0.62"),
    "payout_if_yes": Decimal("1.00"),
    "price": Decimal("0.62"),
}

expected_value = inputs["market_probability"] * inputs["payout_if_yes"] - inputs["price"]
percentage = (inputs["market_probability"] * Decimal("100")).quantize(
    Decimal("0.1"),
    rounding=ROUND_HALF_UP,
)

print(f"Implied probability: {percentage}%")
print(f"Expected value per $1 payout contract: {expected_value:.3f}")
print("Assumptions: probability and price are current source values.")
```

## Output Guidance

Keep the saved artifact short:

```markdown
# Calculation Check: [topic]

- Inputs: ...
- Formula: ...
- Result: ...
- Rounding: ...
- Caveats: ...
```
