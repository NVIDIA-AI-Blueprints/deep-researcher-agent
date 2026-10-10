# DeepResearch Bench II Harbor Adapter

## Overview

This adapter converts DeepResearch Bench II into Harbor task format.
[DeepResearch Bench II](https://github.com/imlrz/DeepResearch-Bench-II)
contains 132 multilingual long-form research prompts. Each example includes a
prompt, language, theme, hidden expert-report-derived rubrics for information
recall, analysis, and presentation, and blocked-source restrictions.

## Features

- Downloads `tasks_and_rubrics.jsonl` from a pinned upstream revision by
  default.
- Supports `--tasks-jsonl` for local/offline dataset generation.
- Converts every source row in order and generates stable task IDs:
  `deepresearch-bench-ii-001` through `deepresearch-bench-ii-132`.
- Validates source IDs, consecutive source indices, required fields, and all
  three rubric dimensions.
- Uses `aiq/<task-id>` task names by default for AIQ Harbor evaluation runs.
- Preserves source-prompt fidelity by rendering `instruction.md` as only the
  normalized benchmark prompt, with no wrapper or output-path instruction.
- Keeps rubrics and blocked-source metadata under `tests/`, where only the
  verifier can read them.
- Implements the official scoring path for information recall, analysis, and
  presentation.

## Metrics

- `reward`: Fraction of all scored rubric items satisfied without
  blocked-source evidence; used as the primary Harbor score.
- `info_recall`: Fraction of information-recall rubric items scored as
  satisfied.
- `analysis`: Fraction of analysis rubric items scored as satisfied.
- `presentation`: Fraction of presentation rubric items scored as satisfied.
- `blocked_rate`: Fraction of all scored rubric items supported by a blocked
  source and therefore assigned a score of `-1`.
- `rubric_item_count`: Number of rubric items scored across all three
  dimensions.
- `grader_valid`: Binary indicator that the official scorer returned a valid
  result.
- `grader_tokens`: Total grader tokens reported across chunked scoring calls;
  present only when available.

## Authors

Original benchmark authors listed in generated task metadata:

Ruizhe Li, Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao.
