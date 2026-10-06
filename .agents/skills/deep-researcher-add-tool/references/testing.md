<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing a tool

Authoritative source: `docs/source/extending/adding-a-tool.md` (Step 8). Model
tests on `sources/google_scholar_paper_search/tests/test_paper_search.py`.

## Unit tests

Test the client (in `my_client.py`) directly with mocked I/O — no live network.
Cover both the success path and graceful empty/error handling.

```python
import pytest
from unittest.mock import AsyncMock, patch

from my_tool.my_client import MyClient


@pytest.mark.asyncio
async def test_returns_results():
    client = MyClient(api_key="test-key", max_results=5)
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"results": [{"url": "https://example.com", "content": "Example"}]},
            raise_for_status=lambda: None,
        )
        result = await client.run("test query")
    assert "Example" in result


@pytest.mark.asyncio
async def test_handles_no_results():
    client = MyClient(api_key="test-key")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"results": []},
            raise_for_status=lambda: None,
        )
        result = await client.run("nonexistent")
    assert "No results found" in result
```

Also cover the missing-secret stub path: with no API key set, the registered
function should yield the stub that returns a clear error string rather than
raising.

## Validation commands

Run from the repo root:

```bash
uv pip install -e ./sources/my_tool
uv run pytest sources/my_tool/tests
uv run ruff check sources/my_tool
uv run ruff format --check sources/my_tool
```

Expected: the package installs, tests pass, and Ruff reports no lint or format
failures (line length 120, target 3.11).

## Integration check (optional)

With a real API key set, run the tool through a config that references it:

```bash
.venv/bin/nat run --config_file configs/my_config.yml --input "test query"
```

## Evidence to capture

For the PR (see `deep-researcher-prepare-pr`), record the created/changed files, the
`pytest` transcript with pass counts, and the `ruff` results. Never print secret
values in captured output.
