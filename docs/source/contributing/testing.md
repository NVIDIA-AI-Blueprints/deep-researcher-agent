<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Testing

To run tests:

```bash
uv sync --group dev
uv run pytest
uv run pytest --cov=src/deep_researcher_agent --cov-report=html
uv run pytest tests/path/to/test_file.py

# MCP uses its own project, environment, and lockfile.
uv sync --project mcp --extra dev
uv run --project mcp --extra dev pytest mcp/tests
```

Use mocks for external services; mark slow/integration tests with `@pytest.mark.slow` / `@pytest.mark.integration` as needed.

Refer to each benchmark's README for details. The [Customization guide](../customization/index.md) has a short section on adding eval harnesses.

## Debugging

- **Relay logging:** enabled by `workflow.relay.logging`; inspect the console subscriber and configured ATOF/OTEL destinations.
- **Phoenix tracing:** Start `uvx --from arize-phoenix phoenix serve`, run the agent with Phoenix tracing enabled in config, then open `http://localhost:6006`.
- **Common issues:** Import errors -- ensure `uv pip install -e .`; auth -- check env vars; tool not found -- check config; pre-commit cache -- `pre-commit clean`.
