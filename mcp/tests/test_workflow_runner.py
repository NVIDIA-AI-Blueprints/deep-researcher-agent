# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT 1.8 workflow-runner compatibility tests."""

import logging
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from deep_researcher_agent.agents.chat_researcher.models import RESEARCH_WORKFLOW_FAILURE_ERROR
from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherResponse
from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState
from deep_researcher_agent.agents.chat_researcher.models import WorkflowFailure
from deep_researcher_agent.agents.chat_researcher.models import WorkflowOutcome
from deep_researcher_agent.agents.chat_researcher.models import WorkflowSuccess
from deep_researcher_agent.common import _create_chat_response
from deep_researcher_agent.common.logging_utils import log_identifier_ref
from deep_researcher_mcp import workflow_runner as workflow_runner_module
from deep_researcher_mcp.workflow_runner import WorkflowRunner
from nat.builder.context import Context


def _workflow_response(content: str, outcome: WorkflowOutcome) -> ChatResearcherResponse:
    response = _create_chat_response(content)
    return ChatResearcherResponse(**response.model_dump(), workflow_outcome=outcome)


def test_run_query_requires_explicit_conversation_id(tmp_path) -> None:
    runner = WorkflowRunner(tmp_path / "config.yml")

    with pytest.raises(TypeError, match="conversation_id"):
        runner.run_query("query")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_classify_invokes_target_intent_function(tmp_path) -> None:
    observed: dict[str, object] = {}

    class _IntentFunction:
        async def ainvoke(self, state: ChatResearcherState) -> dict[str, str]:
            observed["state"] = state
            return {"classification": "ok"}

    class _Builder:
        async def get_function(self, name: str) -> _IntentFunction:
            observed["function_name"] = name
            return _IntentFunction()

    runner = WorkflowRunner(tmp_path / "config.yml")
    runner._session_manager = SimpleNamespace(shared_builder=_Builder())

    assert await runner.classify("What is CUDA?") == {"classification": "ok"}
    assert observed["function_name"] == "intent_classifier"
    state = observed["state"]
    assert isinstance(state, ChatResearcherState)
    assert isinstance(state.messages[-1], HumanMessage)
    assert state.messages[-1].content == "What is CUDA?"


@pytest.mark.asyncio
async def test_start_and_stop_own_one_nat_workflow_lifecycle(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, str]] = []
    session_manager = SimpleNamespace()

    @asynccontextmanager
    async def fake_load_workflow(config_file: str):
        events.append(("enter", config_file))
        try:
            yield session_manager
        finally:
            events.append(("exit", config_file))

    monkeypatch.setattr(workflow_runner_module, "load_workflow", fake_load_workflow)
    runner = WorkflowRunner(tmp_path / "config.yml")

    await runner.start()
    await runner.start()
    assert runner._session_manager is session_manager

    await runner.stop()
    await runner.stop()

    assert events == [
        ("enter", str(tmp_path / "config.yml")),
        ("exit", str(tmp_path / "config.yml")),
    ]


@pytest.mark.parametrize(
    ("response_content", "workflow_outcome"),
    [
        pytest.param("research answer", WorkflowSuccess(result="research answer"), id="success"),
        pytest.param(
            "Please try again.",
            WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR),
            id="failed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_query_returns_structured_outcome_and_restores_context(
    tmp_path, caplog, response_content: str, workflow_outcome: WorkflowOutcome
) -> None:
    observed: dict[str, str | None] = {}
    job_id = str(uuid.uuid4())

    class _Result:
        async def result(self, to_type: type[ChatResearcherResponse]) -> ChatResearcherResponse:
            observed["result_type"] = to_type.__name__
            observed["result_context"] = Context.get().conversation_id
            return _workflow_response(response_content, workflow_outcome)

    class _RunContext:
        async def __aenter__(self) -> _Result:
            return _Result()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _Session:
        def run(self, query: str) -> _RunContext:
            observed["query"] = query
            observed["run_context"] = Context.get().conversation_id
            return _RunContext()

    class _SessionContext:
        async def __aenter__(self) -> _Session:
            return _Session()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _SessionManager:
        def session(self, *, conversation_id: str) -> _SessionContext:
            observed["session_id"] = conversation_id
            observed["session_context"] = Context.get().conversation_id
            return _SessionContext()

    runner = WorkflowRunner(tmp_path / "config.yml")
    runner._session_manager = _SessionManager()  # type: ignore[assignment]
    caplog.set_level(logging.INFO, logger="deep_researcher_mcp.workflow_runner")

    with Context.scope(conversation_id="outer"):
        assert await runner.run_query("query", conversation_id=job_id) == workflow_outcome
        assert Context.get().conversation_id == "outer"

    assert observed == {
        "session_id": job_id,
        "session_context": job_id,
        "query": "query",
        "run_context": job_id,
        "result_type": "ChatResearcherResponse",
        "result_context": job_id,
    }
    assert job_id not in caplog.text
    assert log_identifier_ref(job_id) in caplog.text


@pytest.mark.asyncio
async def test_workflow_runner_closes_only_owned_checkpointers(monkeypatch, tmp_path) -> None:
    from deep_researcher_agent import common as deep_researcher_common

    closed: list[str] = []

    class _Connection:
        async def close(self) -> None:
            closed.append("connection")

    class _Pool:
        def close(self) -> None:
            closed.append("pool")

    preexisting = object()
    monkeypatch.setattr(
        deep_researcher_common,
        "_checkpointers",
        {"preexisting": preexisting, "owned": SimpleNamespace(conn=_Connection())},
    )
    monkeypatch.setattr(deep_researcher_common, "_postgres_pools", {"owned": _Pool()})

    runner = WorkflowRunner(tmp_path / "config.yml")
    runner._owned_checkpointer_keys = {"owned"}

    await runner._close_owned_checkpointers()

    assert closed == ["connection", "pool"]
    assert deep_researcher_common._checkpointers == {"preexisting": preexisting}
    assert deep_researcher_common._postgres_pools == {}
    assert runner._owned_checkpointer_keys == set()


@pytest.mark.asyncio
async def test_close_owned_checkpointers_warns_when_expected_caches_missing(monkeypatch, tmp_path, caplog) -> None:
    """If deep_researcher_agent renames/removes the private caches, cleanup must warn, not silently no-op."""
    from deep_researcher_agent import common as deep_researcher_common

    monkeypatch.delattr(deep_researcher_common, "_checkpointers", raising=False)
    monkeypatch.delattr(deep_researcher_common, "_postgres_pools", raising=False)

    runner = WorkflowRunner(tmp_path / "config.yml")
    runner._owned_checkpointer_keys = {"owned"}

    caplog.set_level(logging.WARNING, logger="deep_researcher_mcp.workflow_runner")
    await runner._close_owned_checkpointers()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a warning when the checkpointer caches are unavailable"
    assert "_checkpointers" in caplog.text and "_postgres_pools" in caplog.text
    assert "may leak" in caplog.text
    assert runner._owned_checkpointer_keys == set()


@pytest.mark.asyncio
async def test_close_owned_checkpointers_warns_even_without_recorded_owned_keys(monkeypatch, tmp_path, caplog) -> None:
    """The rename case: start() records no owned keys because the snapshot read the old
    name, so the warning must fire even though _owned_checkpointer_keys is empty."""
    from deep_researcher_agent import common as deep_researcher_common

    monkeypatch.delattr(deep_researcher_common, "_checkpointers", raising=False)
    monkeypatch.delattr(deep_researcher_common, "_postgres_pools", raising=False)

    runner = WorkflowRunner(tmp_path / "config.yml")
    assert runner._owned_checkpointer_keys == set()

    caplog.set_level(logging.WARNING, logger="deep_researcher_mcp.workflow_runner")
    await runner._close_owned_checkpointers()

    assert [r for r in caplog.records if r.levelno == logging.WARNING], "rename must still warn"


@pytest.mark.asyncio
async def test_close_owned_checkpointers_silent_when_caches_present_but_unowned(monkeypatch, tmp_path, caplog) -> None:
    """Present-but-empty caches with nothing owned is normal — must not warn (no false positive)."""
    from deep_researcher_agent import common as deep_researcher_common

    monkeypatch.setattr(deep_researcher_common, "_checkpointers", {})
    monkeypatch.setattr(deep_researcher_common, "_postgres_pools", {})

    runner = WorkflowRunner(tmp_path / "config.yml")
    assert runner._owned_checkpointer_keys == set()

    caplog.set_level(logging.WARNING, logger="deep_researcher_mcp.workflow_runner")
    await runner._close_owned_checkpointers()

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
