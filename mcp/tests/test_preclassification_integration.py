# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration guard for DeepResearcher-001: the caller's depth decision is reused by the real workflow.

Loads the real public MCP workflow and drives its ``intent_classifier`` node through the
genuine ``nat.Function.ainvoke`` wrapper (not a hand-rolled stand-in), with the intent LLM
replaced by a counting stub. This closes the one link the design left to the implementation:
that the request-scoped ``preclassified_depth`` ContextVar propagates through NAT's session/
function machinery into the in-graph node.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState
from deep_researcher_agent.agents.chat_researcher.preclassification import preclassified_depth
from deep_researcher_mcp.jobs import _extract_depth
from deep_researcher_mcp.jobs import _extract_intent
from deep_researcher_mcp.workflow_runner import WorkflowRunner
from nat.builder.workflow_builder import WorkflowBuilder

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INTENT_LLM_NAME = "nemotron_lightning_intent_llm"


class _CountingIntentLLM:
    """Duck-typed stand-in for the intent LLM.

    Counts invocations and always classifies ``deep`` so that a preset of ``shallow``
    is provably a *different* decision than the model would make on its own.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages, config=None):  # noqa: ANN001 - duck-typed LangChain shim
        del messages, config
        self.calls += 1
        return AIMessage(content='{"intent": "research", "research_depth": "deep"}')


@pytest.mark.asyncio
async def test_preclassified_depth_overrides_intent_llm_through_real_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intent_classifier node reuses a caller-supplied depth instead of re-classifying.

    Through the real NAT function wrapper:

    * with no hook set, the node invokes the intent LLM and adopts its depth (``deep``);
    * with the hook set to a *different* depth (``shallow``), the node short-circuits — the
      intent LLM is not invoked a second time and the preset depth wins.

    That is exactly the DeepResearcher-001 contract: the classification made once in ``submit()`` is the
    one the workflow executes, rather than a fresh temperature-``0.5`` decision that can differ.
    """
    postgres_url = os.getenv("DEEP_RESEARCHER_MCP_TEST_DB_URL")
    if not postgres_url:
        pytest.skip("set DEEP_RESEARCHER_MCP_TEST_DB_URL to load the real NAT MCP workflow")

    monkeypatch.setenv("DEEP_RESEARCHER_CHECKPOINT_DB", postgres_url)
    monkeypatch.setenv("NVIDIA_API_KEY", "not-a-real-key")  # pragma: allowlist secret
    monkeypatch.setenv("TAVILY_API_KEY", "not-a-real-key")  # pragma: allowlist secret

    stub = _CountingIntentLLM()
    original_get_llm = WorkflowBuilder.get_llm

    async def _patched_get_llm(self, llm_name, wrapper_type):  # noqa: ANN001 - matches NAT signature
        if str(llm_name) == _INTENT_LLM_NAME:
            return stub
        return await original_get_llm(self, llm_name, wrapper_type)

    monkeypatch.setattr(WorkflowBuilder, "get_llm", _patched_get_llm)

    runner = WorkflowRunner(_REPO_ROOT / "configs" / "config_mcp.yml")
    await runner.start()
    try:
        assert runner._session_manager is not None
        intent_fn = await runner._session_manager.shared_builder.get_function("intent_classifier")
        state = ChatResearcherState(messages=[HumanMessage(content="Who founded NVIDIA and in what year?")])

        # No hook: the real node classifies via the (stubbed) intent LLM and takes its answer.
        baseline = await intent_fn.ainvoke(state)
        assert stub.calls == 1
        assert _extract_intent(baseline) == "research"
        assert _extract_depth(baseline) == "deep"

        # Hook set to a *different* depth: the node reuses it and does not re-invoke the LLM.
        with preclassified_depth("shallow"):
            reused = await intent_fn.ainvoke(state)
        assert stub.calls == 1
        assert _extract_intent(reused) == "research"
        assert _extract_depth(reused) == "shallow"
    finally:
        await runner.stop()
