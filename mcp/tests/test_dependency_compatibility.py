# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused NAT 1.8 and MCP 1.28.1 compatibility contracts."""

from __future__ import annotations

import inspect
import os
from importlib.metadata import version
from pathlib import Path

import pytest
from mcp.shared.version import LATEST_PROTOCOL_VERSION
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from deep_researcher_mcp.workflow_runner import WorkflowRunner
from nat.builder.context import Context
from nat.runtime.loader import load_workflow
from nat.runtime.session import SessionManager

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARITY_DOCUMENT = _REPO_ROOT / "mcp" / "REFERENCE_PARITY.md"


def test_runtime_dependency_versions_are_the_validated_compatibility_baseline() -> None:
    assert version("mcp") == "1.28.1"
    assert version("nvidia-nat") == "1.8.0"
    assert version("nvidia-nat-core") == "1.8.0"
    assert version("deep-researcher-agent") == "2.2.0"
    assert version("cryptography") == "50.0.0"


def test_mcp_protocol_version_is_explicitly_supported() -> None:
    assert LATEST_PROTOCOL_VERSION == "2025-11-25"
    assert SUPPORTED_PROTOCOL_VERSIONS == [
        "2024-11-05",
        "2025-03-26",
        "2025-06-18",
        "2025-11-25",
    ]


def test_nat_18_api_surface_used_by_workflow_runner_is_present() -> None:
    assert list(inspect.signature(load_workflow).parameters) == ["config_file", "max_concurrency"]
    assert isinstance(SessionManager.shared_builder, property)
    assert "conversation_id" in inspect.signature(SessionManager.session).parameters
    assert callable(Context.scope)


def test_reference_snapshot_and_intentional_deviations_are_recorded() -> None:
    text = _PARITY_DOCUMENT.read_text()

    assert (
        "81eba67fadd56e64b58a84b700b202841f8636c93c6cbf63752507c8bf5ca96a"  # pragma: allowlist secret
        in text
    )

    for documented_deviation in (
        "Malformed or noncanonical capability IDs",
        "Authentication and principal",
        "Transport path and health",
        "Lifecycle ownership",
        "Startup and environment",
        "Public workflow surface",
        "Certificates",
        "Public design decisions",
    ):
        assert documented_deviation in text


@pytest.mark.asyncio
async def test_nat_18_loads_the_real_public_mcp_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_url = os.getenv("DEEP_RESEARCHER_MCP_TEST_DB_URL")
    if not postgres_url:
        pytest.skip("set DEEP_RESEARCHER_MCP_TEST_DB_URL to load the real NAT MCP workflow")
    monkeypatch.setenv("DEEP_RESEARCHER_CHECKPOINT_DB", postgres_url)
    monkeypatch.setenv("NVIDIA_API_KEY", "not-a-real-key")  # pragma: allowlist secret
    monkeypatch.setenv("TAVILY_API_KEY", "not-a-real-key")  # pragma: allowlist secret
    runner = WorkflowRunner(_REPO_ROOT / "configs" / "config_mcp.yml")

    await runner.start()
    try:
        assert runner._session_manager is not None
        intent_classifier = await runner._session_manager.shared_builder.get_function("intent_classifier")
        assert callable(intent_classifier.ainvoke)
    finally:
        await runner.stop()
