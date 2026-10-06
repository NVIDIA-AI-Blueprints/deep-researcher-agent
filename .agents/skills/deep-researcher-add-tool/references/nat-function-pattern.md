<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NAT function pattern

Authoritative source: `docs/source/extending/adding-a-tool.md`. Model new tools
on `sources/google_scholar_paper_search/` (has tests) or `sources/tavily_web_search/`
(minimal).

## Package layout

```text
sources/my_tool/
  pyproject.toml
  README.md
  src/
    __init__.py
    register.py        # config class + NAT registration
    my_client.py       # NAT-independent implementation
  tests/
    __init__.py
    conftest.py
    test_my_tool.py
```

Keep the client (`my_client.py`) free of NAT imports so it is unit-testable on
its own; `register.py` adapts it into a NAT function.

## Config class and registration

In `src/register.py`, define a `FunctionBaseConfig` subclass with a stable
`name=` (this is the YAML `_type`), resolve secrets via `SecretStr`, and yield a
`FunctionInfo` from an async `@register_function`:

```python
import os

from pydantic import Field, SecretStr
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class MyToolConfig(FunctionBaseConfig, name="my_tool"):
    """Config for the my_tool function."""

    max_results: int = Field(default=10, description="Max results to return")
    my_api_key: SecretStr | None = Field(default=None, description="API key")


@register_function(config_type=MyToolConfig)
async def my_tool(tool_config: MyToolConfig, builder: Builder):
    api_key = os.environ.get("MY_API_KEY") or (
        tool_config.my_api_key.get_secret_value() if tool_config.my_api_key else None
    )
    if not api_key:
        async def _stub(query: str) -> str:
            """Tool unavailable - missing MY_API_KEY."""
            return "Error: my_tool is unavailable because MY_API_KEY is not set."

        yield FunctionInfo.from_fn(_stub, description=_stub.__doc__)
        return

    async def _run(query: str) -> str:
        """Describe WHAT the tool does, WHEN to use it, and WHAT it returns."""
        ...  # call the client in my_client.py

    yield FunctionInfo.from_fn(_run, description=_run.__doc__)
```

Notes:

- The function docstring becomes the `description` the LLM sees — make it
  specific so the agent calls the tool at the right time.
- Tools must not raise; catch errors and return an error string.
- For results with URLs, use the `<Document href="...">...</Document>` format so
  the agent can cite sources.

## pyproject.toml entry point

```toml
[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools >= 64", "setuptools-scm>=8"]

[tool.setuptools]
packages = ["my_tool"]
package-dir = {"my_tool" = "src"}

[project]
name = "my-tool"
version = "1.0.0"
description = "NAT-based <what it does> tool"
readme = "README.md"
requires-python = ">=3.11,<3.14"
license = {text = "Apache-2.0"}
dependencies = [
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

# REQUIRED so NAT discovers the registration at import time. The left-hand key
# is the YAML `_type` name; the right-hand value points at the register module.
[project.entry-points."nat.plugins"]
my_tool = "my_tool.register"
```

## Wire the tool into an agent (YAML)

Unlike a data source, a plain tool is referenced **directly** in an agent's
`tools` list — there is no `data_source_registry` entry and no UI toggle:

```yaml
functions:
  my_search:
    _type: my_tool         # matches the config class name= / entry-point key
    max_results: 10

  shallow_research_agent:
    _type: shallow_research_agent
    llm: research_llm
    tools:
      - my_search
```

If you want the tool to appear as a toggleable UI source instead, stop here and
use `deep-researcher-add-data-source`, which adds the registry entry on top of this pattern.
