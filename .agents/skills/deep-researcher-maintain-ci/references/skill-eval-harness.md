<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Skill-eval harness

Authoritative sources: `.github/skill-eval/README.md` and
`.github/workflows/skills-eval.yml`.

## What it is

`.github/skill-eval/` is the product-level Agent Skill evaluation harness that
gates skill changes. It contains `skills_eval_agent.py`, `adapters/`,
`verifiers/`, and its own `README.md` / `AGENTS.md`.

## How it works

The `skills-eval.yml` workflow runs in stages:

1. **`detect-changes`** — path-gates the run (the trigger has no `paths:` filter;
   this job decides whether to proceed).
2. **`generate-datasets`** ("Validate skill-eval specs") — finds specs under
   `skills/<skill>/evals/*-product.json`, validates their required fields, and
   generates datasets via the matching adapter under
   `.github/skill-eval/adapters/<skill>/`. **No Harbor credentials needed** — this
   is the stage you can reason about and reproduce locally.
3. **`harbor-eval`** — runs the Harbor trials on the self-hosted `deep-researcher-eval`
   runner, then a deterministic verifier checks results against a reward threshold.

The first supported skill is `deep-researcher-research`
(`skills/deep-researcher-research/evals/*-product.json`).

## Changing the harness safely

- Keep the `detect-changes` path gate working; do not switch it to a trigger
  `paths:` filter (the workflow comment explains why).
- `harbor-eval` requires the self-hosted runner + credentials; `generate-datasets`
  does not, so validate spec/adapter/verifier shape there (locally) and rely on
  the mirrored CI run for the full sweep.
- When adding a skill to the harness, add its `*-product.json` spec and a matching
  adapter; mirror the existing `deep-researcher-research` adapter.

## Validation

Defer to `.github/skill-eval/README.md` for running the harness. For local sanity,
confirm your `*-product.json` specs parse and follow the README's required fields
(this is what `generate-datasets` checks), and after editing the workflow run:

```bash
uv run pre-commit run --files .github/workflows/skills-eval.yml
```

Expected: the specs parse, and the workflow edit passes pre-commit (YAML check).
The full evaluation runs on the self-hosted runner via the mirrored CI.
