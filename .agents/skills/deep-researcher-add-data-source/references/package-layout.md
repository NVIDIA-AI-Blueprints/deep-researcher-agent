<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Data source package layout

Model new sources on `sources/google_scholar_paper_search/`. The layout is:

```text
sources/my_data_source/
  pyproject.toml
  README.md
  src/
    __init__.py
    register.py        # config class + NAT registration
    search.py          # NAT-independent client/tool implementation
  tests/
    __init__.py
    conftest.py
    test_search.py
```

Keep the client (`search.py`) free of NAT imports so it is unit-testable on its
own; `register.py` adapts it into a NAT function.

## pyproject.toml

```toml
[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools >= 64", "setuptools-scm>=8"]

[tool.setuptools]
packages = ["my_data_source"]
package-dir = {"my_data_source" = "src"}

[project]
name = "my-data-source"
version = "1.0.0"
description = "NAT-based <what it searches> data source"
readme = "README.md"
requires-python = ">=3.11,<3.14"
license = {text = "Apache-2.0"}
dependencies = [
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

# REQUIRED so NAT discovers the registration. The left-hand key is the YAML
# `_type` name; the right-hand value points at the register module.
[project.entry-points."nat.plugins"]
my_data_source = "my_data_source.register"
```

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


class MyDataSourceConfig(FunctionBaseConfig, name="my_data_source"):
    """Config for the my_data_source retrieval tool."""

    max_results: int = Field(default=10, description="Max results to return")
    my_api_key: SecretStr | None = Field(default=None, description="API key")


@register_function(config_type=MyDataSourceConfig)
async def my_data_source(tool_config: MyDataSourceConfig, builder: Builder):
    api_key = os.environ.get("MY_API_KEY") or (
        tool_config.my_api_key.get_secret_value() if tool_config.my_api_key else None
    )
    if not api_key:
        async def _stub(query: str) -> str:
            """Tool unavailable - missing MY_API_KEY."""
            return "Error: my_data_source is unavailable because MY_API_KEY is not set."

        yield FunctionInfo.from_fn(_stub, description=_stub.__doc__)
        return

    async def _search(query: str) -> str:
        """Searches <domain> and returns structured, citation-rich results."""
        ...  # call the client in search.py

    yield FunctionInfo.from_fn(_search, description=_search.__doc__)
```

The tool's docstring becomes the description the agent sees — make it specific
and return well-structured, citable output.
