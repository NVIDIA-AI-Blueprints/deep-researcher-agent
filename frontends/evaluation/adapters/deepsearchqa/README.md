# DeepSearchQA Harbor Adapter

## Overview

This adapter converts Google DeepMind's DeepSearchQA dataset into Harbor task format. [DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) contains 900 English web-research question-answering prompts across 17 fields. Each example includes a user problem, problem category, hidden gold answer, and hidden answer type (`Single Answer` or `Set Answer`).

## Features

- Downloads `DSQA-full.csv` from Hugging Face by default.
- Supports `--csv-path` for local/offline CSV generation.
- Converts every source row in order and generates stable task IDs. The
  official 900-row source produces `deepsearchqa-0001` through
  `deepsearchqa-0900`.
- Validates that required official columns are present and non-empty.
- Uses `aiq/<task-id>` task names by default for AIQ Harbor evaluation runs.
- Preserves source-prompt fidelity by rendering `instruction.md` as only the
  normalized DeepSearchQA `problem` text, with no wrapper or output-path
  instruction.
- Implements the official Gemini 2.5 Flash autorater prompt from the starter [notebook](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code).

## Metrics

- `reward`: Primary Harbor score; equal to `f1_score`.
- `fully_correct`: Binary indicator that every expected answer was identified
  and no excessive answers were returned.
- `fully_incorrect`: Binary indicator that none of the expected answers were
  identified.
- `correct_with_excessive_answers`: Binary indicator that every expected
  answer was identified but at least one excessive answer was returned.
- `precision`: Fraction of returned answers that are expected answers.
- `recall`: Fraction of expected answers that were identified.
- `f1_score`: Harmonic mean of precision and recall.
- `grader_valid`: Binary indicator that the grader response was parsed and
  evaluated successfully.

## Authors

Original benchmark authors listed in generated task metadata:

Nikita Gupta, Riju Chatterjee, Lukas Haas, Connie Tao, Andrew Wang, Chang Liu, Hidekazu Oiwa, Elena Gribovskaya, Jan Ackermann, John Blitzer, Sasha Goldshtein, and Dipanjan Das.
