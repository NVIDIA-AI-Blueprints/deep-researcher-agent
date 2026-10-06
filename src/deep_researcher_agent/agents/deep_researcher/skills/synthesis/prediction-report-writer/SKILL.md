---
name: prediction-report-writer
description: >
  Use this skill when the final answer strategy calls for a prediction, forecast, probability estimate, price target, expected value, threshold outcome, scenario outlook, or prediction-style research report. This skill takes priority over general long-form writing skills for forecasts, stock-price outlooks, probabilities, odds, price targets, expected values, and next-period value estimates, even when the requested answer is concise. Triggers: "prediction", "forecast", "probability", "odds", "confidence interval", "price target", "expected value", "will X happen", "stock price", "market outlook", "scenario analysis", "threshold", "single forecast", "predict". Outputs: A cited Markdown prediction report with a clear forecast, confidence interval or range, executive summary, supporting and opposing factors, uncertainties, recommendations or implications, methodology note, and sources.
---

# Prediction Report Writer Skill

Write a clear, actionable prediction report from the planned answer strategy and ResearchNotes artifacts. This skill adapts prediction-system synthesis patterns for Deep Researcher Agent's broader research tasks.

Use this skill for probabilistic predictions, point forecasts, ranges, price targets, threshold forecasts, event likelihoods, and forecast-style decision reports. This skill is for final synthesis only; do not perform new research.

Use this skill instead of `long-form-report-writer` whenever `answer_strategy.answer_type` is `prediction` or the user asks for a forecast, stock price, price target, probability, odds, expected value, or next-period value. Only combine it with long-form guidance when the user explicitly asks for a long-form prediction report.

## Required Input Review

Before writing, read:
1. `/shared/plan.json`
2. Every `ResearchNotes` JSON file under `/shared/` that supports the plan
3. The compact result of `get_verified_sources`

Use `think` to build a prediction synthesis outline before drafting:
- Identify the original prediction question.
- Identify the target variable: probability, price, value, date, winner, threshold bucket, yes/no outcome, ranking, or option selection.
- Extract all anchors from the research notes, such as latest known values, base rates, market prices, expert forecasts, recent trend data, and official measurements.
- Extract directional evidence supporting the prediction.
- Extract directional evidence opposing the prediction.
- Use each note's `ResearchNotes.evidence_judgment` when present to prioritize notes: high-score/high-confidence notes should anchor the forecast, medium notes can support or nuance it, and low-score or low-confidence notes should mainly inform gaps, caveats, conflicts, or weak-evidence warnings.
- List major uncertainties that could change the forecast.
- Decide the final forecast before drafting, then write the report around it.

## Forecast Discipline

- Commit to a view. Do not say only that the outcome is uncertain.
- Use the strongest available anchor as the starting point:
  - For probabilities: use prediction markets, base rates, polls, odds, expert consensus, or historical frequency when available.
  - For stock/market/economic values: use the latest known value and recent trend data before applying an adjustment.
  - For threshold or multiple-choice predictions: identify the forecasted value first, then select the matching option or bucket.
  - For ranked or categorical outcomes: rank candidates using evidence strength and recent continuity.
- If research notes include an ensemble result, aggregated probability, market consensus, or model estimate, use it as the starting point. Adjust only when the cited evidence strongly warrants it, and explain the adjustment.
- If there is no explicit ensemble, synthesize from the research evidence and say what anchor or base rate you used.
- Express uncertainty through a confidence interval, forecast range, or scenario probabilities rather than vague hedging.
- Avoid language like "it is hard to say" unless immediately followed by a concrete best estimate and uncertainty range.

## Required Report Content

The final Markdown answer should include these elements unless the user's requested shape is explicitly shorter:

1. **Direct Forecast**
   - Put the final probability, value, range, selected option, or prediction at the top.
   - For probabilistic questions, include a probability and confidence interval such as `42% (confidence interval: 35%-55%)`.
   - For numeric forecasts, include a point estimate and plausible range.
   - For multiple-choice or threshold questions, state the selected option and the underlying forecast value when possible.

2. **Executive Summary**
   - One concise paragraph explaining the forecast and why.

3. **Supporting Factors**
   - Three to five cited bullet points or short paragraphs with evidence that supports the forecast.

4. **Opposing Factors**
   - Two to four cited bullet points or short paragraphs with evidence against the forecast or reasons the forecast could be too high/low.

5. **Key Uncertainties**
   - Two to four major unknowns that could change the outcome.

6. **Recommendations or Implications**
   - Two to four actionable recommendations, implications, monitoring points, or decision considerations for someone acting on the forecast.

7. **Methodology Note**
   - Briefly explain how the forecast was synthesized from evidence. Do not mention agents, prompts, tools, or internal files.

8. **Sources**
   - Include a compact sources section with every cited source.

## Output Shape Guidance

Use clear Markdown. A good default structure is:

```markdown
# [Prediction Title]

**Forecast:** [direct probability/value/selection]
**Confidence interval / range:** [range]
**Confidence:** [low/medium/high]

## Executive Summary
...

## Supporting Factors
- ...

## Opposing Factors
- ...

## Key Uncertainties
- ...

## Recommendations / Implications
- ...

## Methodology Note
...

## Sources
[1] ...
```

For a very short requested answer, keep the direct forecast first, then include only the minimum rationale and sources needed to make it trustworthy.

For table-oriented forecast requests, include a table with scenario, forecast, probability or range, evidence, and caveats.

## Citation Rules

- Every material factual claim must have a normal numeric citation like `[1]`.
- Cite anchors, latest values, market odds, polls, expert forecasts, official data, and each major supporting/opposing factor.
- Do not place bare URLs in the report body.
- Before drafting, call `get_verified_sources` in its default compact mode and build the final citation map only from that output. ResearchNotes can guide what evidence to use, but ResearchNotes titles, archive names, publisher names, source labels, and note-local citation numbers are not valid final citations unless the exact same URL or citation key appears in `get_verified_sources`.
- Call `get_verified_sources(mode="full")` only if a source locator from ResearchNotes is missing from the compact output and is materially needed.
- Each citation number must map to exactly one verified URL or one exact verified citation key from `get_verified_sources`. Do not group multiple publishers, tools, URLs, archive names, collections, documents, or source names under one citation number.
- Do not write unverifiable aggregate labels such as `[1] CNN; Yahoo Finance; Barchart`, `[2] Factory Inspection Reports`, or `[3] National Archives correspondence`.
- For claims supported by multiple sources, use adjacent citations like `[1][2]`, with each number pointing to its own source line.
- The Sources section must include one line per cited source in the form `[N] Title: https://...` for URL sources, or `[N] exact-document-or-tool-citation-key` for URL-less verified sources.
- Every Sources line must contain one full URL or exact citation key copied from `get_verified_sources`. A descriptive source label by itself is invalid.
- Do not invent, recall, paraphrase, shorten, summarize, or "prettify" URLs, citation keys, source titles, market data, model outputs, or ensemble probabilities.
- If a forecast-relevant claim cannot be mapped to a `get_verified_sources` entry, remove it or state it as an uncertainty/gap without a citation.
- Do not use edit loops or search-and-replace passes to repair citation numbering. Deterministic post-processing verifies and sanitizes citations after `/shared/output.md` is written.

## Calibration And Confidence

Reflect confidence as:
- `high`: multiple strong, recent, mutually reinforcing sources or a reliable market/official anchor.
- `medium`: reasonable evidence but meaningful disagreement, stale data, or only one strong anchor.
- `low`: sparse evidence, weak sources, large uncertainty, or highly volatile target.

Reflect uncertainty in the Markdown itself:
- Probability: include a confidence interval.
- Numeric value: include a plausible range.
- Choice/threshold: include the forecast value and why it falls into the chosen bucket.
- Recommendation: include monitoring indicators that would change the view.

## Final Verification

Before finishing:
1. Confirm the final forecast is explicit and appears near the top.
2. Confirm supporting and opposing evidence are both represented unless no opposing evidence was found; if none was found, say so briefly as a caveat.
3. Confirm key uncertainties are concrete and actionable.
4. Confirm every material factual claim has an inline citation.
5. Confirm the Sources section includes every cited source and that each source line contains one full verified URL or one exact citation key copied from `get_verified_sources`; descriptive source labels alone are invalid.
6. Confirm no internal files, agents, prompts, or tool names are mentioned.
7. Confirm internal `evidence_judgment` scores and rationales are not exposed unless the user explicitly asked for methodology.
8. Write the complete Markdown answer to `/shared/output.md`.
9. Return only the short completion marker `Wrote /shared/output.md`; do not echo the full Markdown.
