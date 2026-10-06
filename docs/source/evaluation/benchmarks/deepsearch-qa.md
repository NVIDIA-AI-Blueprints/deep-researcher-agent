<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# DeepSearchQA Evaluation for Deep Researcher Agent Deep Researcher

This directory contains the evaluation setup for running [DeepSearchQA](https://www.kaggle.com/datasets/deepmind/deepsearchqa) benchmark from DeepMind on the deep researcher agent.

## Overview

DeepSearchQA is a question-answering benchmark designed to evaluate deep research capabilities. It requires models to search for information and provide accurate answers to complex questions across diverse categories.

### Dataset Statistics

- **Total Problems**: 900
- **Answer Types**: Single Answer, Set Answer
- **Categories**: Politics & Government, Media & Entertainment, Education, Geography, Health, Science, Finance & Economics, Sports, Travel, History, Other

### Evaluation Methodology

Uses the official DeepMind LLM-as-judge methodology from the [starter code](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code):

- **Correctness Details**: Per-answer-component correctness assessment
- **Excessive Answers**: Detection of answers not in the ground truth
- **Metrics**:
  - **Precision**: Correct answers / (Correct + Excessive answers)
  - **Recall**: Correct answers / Expected answers
  - **F1 Score**: Harmonic mean of precision and recall
  - **Accuracy**: Percentage of problems with all correct answers and no excessive answers

## Prerequisites

### Dataset Setup

The dataset is not included in the repository. Download it before running evaluation:

1. Download from [Kaggle - DeepSearchQA](https://www.kaggle.com/datasets/deepmind/deepsearchqa)
2. Place `DSQA-full.csv` in `frontends/benchmarks/deepsearch_qa/data/`

### Judge model and API key

The evaluator uses an LLM judge to score answers. The default config (`config_deepsearch_qa.yml`) uses **OpenAI GPT-4o** as the judge.

1. **Choose a judge model** (for example, OpenAI GPT-4o or Gemini 2.5 Flash).
2. **Obtain an API key** for the provider you chose.
3. **Set the key** in `deploy/.env` (for example, `OPENAI_API_KEY=your_key`) or export it.
4. To use a different judge (for example, Gemini), add a Gemini LLM under `llms:` in the config and set `eval.evaluators.deepsearchqa.llm_name` to that LLM name.

### Other API keys

Set in `deploy/.env`: `NVIDIA_API_KEY` (agent), `TAVILY_API_KEY` (web search).

## Quick Start

```bash
dotenv -f deploy/.env run nat eval --config_file frontends/benchmarks/deepsearch_qa/configs/config_deepsearch_qa.yml
```

Results are written to `frontends/benchmarks/deepsearch_qa/results` (or the `output_dir` in the config).


## Scoring

- **100 points**: All expected answers correct, no excessive answers
- **75 points**: All expected answers correct, but has excessive answers
- **0-50 points**: Partial correctness (scaled by proportion correct)
- **0 points**: No correct answers or empty response

## References

- [DeepSearchQA Dataset](https://www.kaggle.com/datasets/deepmind/deepsearchqa)
- [DeepSearchQA Starter Code](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code)
- [NeMo Agent Toolkit Evaluation Framework](https://github.com/NVIDIA/GenerativeAIExamples)

---

## Configuration Files

| Config | Description |
|--------|-------------|
| `configs/config_deepsearch_qa.yml` | Default: Nemotron Ultra on NVIDIA API Catalog, OpenAI judge. Use for quickstart. |
