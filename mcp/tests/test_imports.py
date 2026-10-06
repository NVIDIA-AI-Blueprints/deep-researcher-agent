# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package and module-boundary tests for the MCP component."""

from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from mcp.server import fastmcp

import deep_researcher_mcp
import mcp


@pytest.mark.parametrize(
    "module_name",
    [
        "deep_researcher_mcp.checkpoint_todos",
        "deep_researcher_mcp.db_url",
        "deep_researcher_mcp.jobs",
        "deep_researcher_mcp.job_store",
        "deep_researcher_mcp.server",
        "deep_researcher_mcp.workflow_runner",
    ],
)
def test_public_module_imports(module_name: str) -> None:
    assert import_module(module_name).__name__ == module_name


def test_package_does_not_shadow_protocol_package() -> None:
    component_root = Path(__file__).resolve().parents[1]
    component_source_root = component_root / "src"

    assert deep_researcher_mcp.__name__ == "deep_researcher_mcp"
    assert mcp.__name__ == "mcp"
    assert Path(deep_researcher_mcp.__file__).parent.name == "deep_researcher_mcp"
    assert not (component_root / "__init__.py").exists()
    assert component_source_root not in Path(mcp.__file__).parents
    assert component_source_root not in Path(fastmcp.__file__).parents


def test_package_version_present() -> None:
    assert deep_researcher_mcp.__version__ == "0.1.0"


def test_console_script_entry_point_resolves_to_public_server_main() -> None:
    entry_point = next(
        candidate
        for candidate in entry_points(group="console_scripts")
        if candidate.name == "deep-researcher-mcp-server"
    )

    assert entry_point.value == "deep_researcher_mcp.server:main"
    assert entry_point.load() is import_module("deep_researcher_mcp.server").main


def test_component_runtime_surface_is_explicitly_allowlisted() -> None:
    component_root = Path(deep_researcher_mcp.__file__).parents[2]
    source_root = component_root / "src" / "deep_researcher_mcp"
    assert {path.name for path in source_root.glob("*.py")} == {
        "__init__.py",
        "checkpoint_todos.py",
        "db_url.py",
        "jobs.py",
        "job_store.py",
        "server.py",
        "workflow_runner.py",
    }
