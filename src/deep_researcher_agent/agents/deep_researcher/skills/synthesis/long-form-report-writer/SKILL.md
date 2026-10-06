---
name: long-form-report-writer
description: >
  Use this skill when the final answer strategy calls for a long-form report, publication-quality research writeup, comprehensive analysis, deep dive, whitepaper-style narrative, or detailed cited Markdown report. Triggers: "long_form_report", "long-form report", "comprehensive report", "publication-ready", "deep dive", "in-depth analysis", "whitepaper", "research report", "detailed report", "full report". Outputs: A polished, cited Markdown report with coherent structure, analytical depth, source-grounded claims, limitations, and a sources section.
---

# Long-Form Report Writer Skill

Write a publication-quality, well-cited Markdown report from the planned answer strategy and ResearchNotes artifacts. This skill is for final synthesis only; do not perform new research.

## When To Use

Use this skill when `answer_strategy.answer_type` is `long_form_report`, or when the user's request clearly asks for a comprehensive report, deep analysis, publication-ready writeup, or whitepaper-style answer.

Do not use this skill for brief answers, primary table outputs, data extraction, multiple-choice selection, or numeric predictions unless the answer strategy also requires a full report narrative.

If `answer_strategy.answer_type` is `prediction`, or the user asks for a forecast, stock price, price target, probability, odds, expected value, or next-period value, this skill is not the controlling skill. If `prediction-report-writer` is available, read and follow that skill instead.

## Long-Form Planning

Before writing, complete the general cross-synthesis pass from the base writer prompt. Then use `think` to turn that synthesis map into a report plan:
- Choose the report title, main sections, and subsection flow from the `answer_strategy`, the user's request, and the evidence.
- Use each note's `ResearchNotes.evidence_judgment` when present to prioritize notes: high-score/high-confidence notes should anchor the report, medium notes can support or nuance it, and low-score or low-confidence notes should mainly inform gaps, caveats, conflicts, or weak-evidence warnings.
- Decide which components need detailed prose, which need compact summary, and which need explicit caveat or gap treatment.
- Treat `required_components` as a coverage checklist, not a mechanical table of contents. Combine, split, or reorder components when that produces a more coherent reader-facing argument.
- Decide where report-specific elements such as tables, equations, timelines, case-study boxes, or comparison matrices materially improve the reader's understanding.
- Decide how to order the report so it moves from fundamentals to implications without exposing internal workflow steps.

## Report Writing

Write a comprehensive, in-depth report following the answer strategy and satisfying the constraints. If you need to recall research findings during writing, re-read the relevant research-note JSON files before drafting.

For broad explanatory reports, target 3000-5000+ words unless the user or plan asks for a shorter answer. Each section should have detailed paragraphs, not just a few sentences. Ensure all aspects of the user's query are addressed. Provide analytical depth: explain mechanisms and causes, not just surface descriptions. Acknowledge trade-offs, limitations, uncertainty, or open questions where the evidence warrants.

Create a coherent narrative that integrates information across sources and sections. Identify core concepts that appear across multiple notes, recognize complementary findings that build a fuller picture, prioritize recent and high-quality evidence, and connect concrete details to the report's larger argument. Do not produce a sequence of short, isolated bullet points or one-fact paragraphs.

## Report Presentation

- Use clear Markdown headings: `#` for the title, `##` for main sections, and `###` for subsections when useful.
- Write in developed paragraphs for readability; avoid excessive bullet points.
- Use bullets only for genuinely list-like material such as taxonomies, compact takeaways, risks, or checklists, and surround them with prose that explains their significance.
- No self-referential language such as "I found" or "I researched".
- Each paragraph should be substantive and detailed.
- Structure from fundamentals to complex concepts.
- Provide contextual explanation of specialized terms.
- When appropriate, prefer clear language over jargon; explain technical terms when used.
- Use tables, equations, or code blocks when they materially improve the report.
- Keep the report human-facing; do not expose the internal planning workflow.

## Content To Never Include

Never include:
- References to agents, constraints, workflow, prompts, tools, or internal files.
- Methodology sections or meta-commentary about how the report was produced.
- Statements like "the user requested" or "this report satisfies".
- Word counts or length statements.
- Internal workflow tool names such as `get_verified_sources`, `write_file`, `read_file`, `think`, `task`, or `run_research_batch`.
- Descriptions of how sources were gathered, verified, or validated.

Exception: in the Sources section, a verified data-source tool name may appear when the source has no URL or document locator. Preserve the raw tool/source name exactly as shown by verified sources.

The report must read as if written by a professional human researcher.

## Citation Guidelines

- Number sources sequentially with inline citations like `[1]`, `[2]`, etc.
- Place citations immediately following the relevant claim.
- Citations are required for every material source-derived factual claim, number, date, sourced comparison, source-specific caveat, or analytical conclusion grounded in evidence.
- For arithmetic, rankings, ranges, means, medians, or summary statistics computed solely from already cited values in the report, cite the underlying source data where those values appear. Then state that the derived statistics are computed from that cited table or cited values; do not attach source numbers to every computed-statistic line.
- In a summary-statistics block, cite only source-derived inputs, source-specific caveats, or any newly introduced external value. Lines such as count, mean, median, range, min, or max do not need their own citation when they are computed from the immediately preceding cited table or cited value set.
- Never place bare URLs or hyperlinks in the report body; use only `[N]` citations inline. URLs belong exclusively in the Sources section.
- For information supported by multiple sources, use adjacent citations like `[1][4][7]`.
- Include the full source list at the end of the document.
- Build the final citation map from the default compact `get_verified_sources` output. ResearchNotes can guide what evidence to use, but ResearchNotes titles, archive names, publisher names, source labels, and note-local citation numbers are not valid final citations unless the exact same URL or citation key appears in `get_verified_sources`.
- Call `get_verified_sources(mode="full")` only if a source locator from ResearchNotes is missing from the compact output and is materially needed.
- Assign each unique verified URL or verified citation key a single citation number across all findings. Each citation number must map to exactly one verified URL or one exact verified citation key from `get_verified_sources`.
- Do not group multiple publishers, tools, URLs, archive names, collections, documents, or source names under one citation number. Do not write unverifiable aggregate labels
- Number sources sequentially without gaps.
- For internal documents, cite the filename/page locator only when the exact locator appears in `get_verified_sources`.
- For URL-less structured tool sources, cite the raw tool/source name exactly as shown by `get_verified_sources`; do not invent friendly titles.
- Do not invent, recall, paraphrase, shorten, summarize, or "prettify" URLs or citation keys. Use only source locators copied from `get_verified_sources`.
- If a claim cannot be mapped to a `get_verified_sources` entry, remove it or state it as an evidence gap without pretending it is cited.

### Sources Section Format

Use this format:

```markdown
## Sources
[1] Source Title: https://example.com/source
[2] internal-report.pdf, p.15
[3] mcp_time__get_current_time
```

## Final Verification

Before finishing:
1. Confirm the report follows the plan constraints.
2. Confirm every required answer component is covered or explicitly named as a gap.
3. Confirm every material factual claim has an inline citation.
4. Confirm the Sources section includes every cited source and no uncited filler. Each source line must contain one full verified URL or one exact citation key copied from `get_verified_sources`; descriptive source labels alone are invalid.
5. Confirm the report does not mention internal files, agents, prompts, or tools.
6. Confirm internal `evidence_judgment` scores and rationales are not exposed unless the user explicitly asked for methodology.
7. Write the complete Markdown answer to `/shared/output.md`.
8. Return only the short completion marker `Wrote /shared/output.md`; do not echo the full Markdown.
9. Do not use edit loops or search-and-replace passes to repair citation numbering. Deterministic post-processing verifies and sanitizes citations after `/shared/output.md` is written.
