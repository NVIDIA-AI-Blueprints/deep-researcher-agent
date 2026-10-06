<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validation

Run from the repo root. Start with the narrowest command and broaden only if the
change crosses shared boundaries.

## 1. Install the package editable

```bash
uv pip install -e ./sources/my_data_source
```

Expected: the package installs and its `nat.plugins` entry point is registered.

## 2. Run the package tests

```bash
uv run pytest sources/my_data_source/tests
```

Model tests on `sources/google_scholar_paper_search/tests/test_paper_search.py`:
unit-test the client with `unittest.mock` (no live network), and cover both the
configured path and the missing-secret stub path.

## 3. Lint

```bash
uv run ruff check sources/my_data_source
uv run ruff format --check sources/my_data_source
```

Expected: no lint or formatting failures (Ruff line length 120, target 3.11).

## 4. Smoke-check the registration (optional)

Start the backend with a config that registers the source and confirm it appears:

```bash
./scripts/start_cli.sh
curl -s http://localhost:8000/v1/data_sources
```

Expected: the new source `id` appears in the returned list.

## Evidence to capture

For the PR (see `deep-researcher-prepare-pr`), record:

- The list of created/changed files.
- The `pytest` transcript with pass counts.
- The `ruff check` result.
- Confirmation the source is registered in the `data_source_registry` config.

Do not print secret values in any captured output.
