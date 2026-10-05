# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for request-local analysis artifacts and cleanup."""

import asyncio
from pathlib import Path

import pytest

from aiq_agent.agents.data_science.utils.analysis_runtime import begin_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import end_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import get_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import register_structured_result


@pytest.mark.asyncio
async def test_manifest_write_failure_does_not_advertise_unusable_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = Path.replace

    def _replace(path: Path, target: str | Path) -> Path:
        if path.name == "structured-results.tmp":
            raise OSError("manifest unavailable")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", _replace)
    token = begin_analysis_run()
    state = get_analysis_run()
    assert state is not None
    try:
        reference = register_structured_result(
            provider="ontology",
            tool_name="ontology__text_to_sql",
            question="Persist rows",
            database_name="example",
            payload={"rows": [{"value": 7}]},
        )

        assert reference is None
        assert state.structured_results == []
        assert list(state.root.glob("structured_*.json")) == []
        assert not state.manifest_path.exists()
    finally:
        await end_analysis_run(token)


@pytest.mark.asyncio
async def test_non_finite_payload_does_not_create_analysis_reference() -> None:
    token = begin_analysis_run()
    state = get_analysis_run()
    assert state is not None
    try:
        reference = register_structured_result(
            provider="ontology",
            tool_name="ontology__text_to_sql",
            question="Non-finite value",
            database_name="example",
            payload={"rows": [{"value": float("nan")}]},
        )

        assert reference is None
        assert state.structured_results == []
        assert list(state.root.glob("structured_*.json")) == []
    finally:
        await end_analysis_run(token)


@pytest.mark.asyncio
async def test_cleanup_removes_artifacts_when_resource_close_is_cancelled() -> None:
    class CancelledResource:
        async def aclose(self) -> None:
            raise asyncio.CancelledError

    token = begin_analysis_run()
    state = get_analysis_run()
    assert state is not None
    root = Path(state.root)
    state.resources["cancelled"] = CancelledResource()

    with pytest.raises(asyncio.CancelledError):
        await end_analysis_run(token)

    assert not root.exists()
    assert get_analysis_run() is None
