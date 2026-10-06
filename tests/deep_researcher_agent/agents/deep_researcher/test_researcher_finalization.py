# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the researcher model-call budget and its reserved finalization phase."""

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool

from deep_researcher_agent.agents.deep_researcher.custom_middleware import RESEARCHER_FINALIZATION_MODEL_CALLS
from deep_researcher_agent.agents.deep_researcher.custom_middleware import ResearcherBudgetExhaustedError
from deep_researcher_agent.agents.deep_researcher.custom_middleware import ResearcherFinalizationMiddleware
from deep_researcher_agent.agents.deep_researcher.custom_middleware import StructuredResponseTextFallbackMiddleware
from deep_researcher_agent.agents.deep_researcher.factory import build_researcher_runnable
from deep_researcher_agent.agents.deep_researcher.models import ResearchNotes
from deep_researcher_agent.agents.deep_researcher.models import ResearchQuery
from deep_researcher_agent.agents.deep_researcher.tools.research import _run_research_query
from deep_researcher_agent.common.logging_utils import log_content_metadata

RESEARCH_NOTES_TOOL = "ResearchNotes"


@tool
def think(thought: str) -> str:
    """Record a thought."""
    return "Thought recorded."


def _query(text: str = "current NVIDIA data center revenue") -> ResearchQuery:
    return ResearchQuery(
        query=text,
        preferred_tools=["web_search"],
        target_components=["revenue_anchor"],
        rationale="Needed for the revenue component.",
    )


def _notes_payload() -> dict:
    return {
        "query_topic": "revenue",
        "target_components": ["revenue_anchor"],
        "summary": "Partial synthesis from the evidence gathered before the budget ran out.",
        "findings": [],
        "gaps": [],
        "sources": [],
        "narrative_notes": "Truncated.",
        "language": "en",
    }


def _request(*, calls_made: int, tools: list | None = None) -> ModelRequest:
    """Build a model request that reports ``calls_made`` completed model turns."""
    return ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="")]),
        messages=[HumanMessage(content="Batch research invocation.")],
        system_prompt="rendered researcher prompt",
        tools=list(tools if tools is not None else [think]),
        response_format=ResearchNotes,
        state={"messages": [], "run_model_call_count": calls_made},
    )


def _finalized_response() -> ModelResponse:
    """A finalization turn the model cooperated with, so the guard lets the response through."""
    return ModelResponse(
        result=[AIMessage(content="")],
        structured_response=ResearchNotes.model_validate(_notes_payload()),
    )


def _raise(exc: Exception):
    """Return a zero-argument callable that raises ``exc``, for scripting a worker failure."""

    def _fail():
        raise exc

    return _fail


def _note_for_worker(behavior, *, query: ResearchQuery | None = None) -> ResearchNotes:
    """Run ``_run_research_query`` against a worker whose ainvoke does ``behavior``."""
    query = query or _query()
    runnable = MagicMock()

    async def _ainvoke(*_args, **_kwargs):
        return behavior()

    runnable.ainvoke = _ainvoke
    return asyncio.run(
        _run_research_query(
            query=query,
            researcher_runnable=runnable,
            runtime=None,
            callbacks=[],
            semaphore=asyncio.Semaphore(1),
        )
    )


class _ResearcherLoopFakeChatModel(FakeMessagesListChatModel):
    """Fake researcher that keeps calling ``think`` until only the output tool is bound.

    Modelling the *bound tool list* rather than a scripted response sequence is the point: it
    is what proves the finalization turn leaves the model no move except returning
    ``ResearchNotes``, without depending on the model choosing to cooperate.
    """

    bound_tools: list[list[str]] = []
    finalize_when_forced: bool = True

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Record the tool names bound for this turn and keep the model itself unbound."""
        self.bound_tools.append([getattr(item, "name", None) or str(item) for item in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        turn = len(self.bound_tools)
        forced = self.bound_tools[-1] == [RESEARCH_NOTES_TOOL] if self.bound_tools else False
        if forced and self.finalize_when_forced:
            call = {"name": RESEARCH_NOTES_TOOL, "args": _notes_payload(), "id": f"call-{turn}"}
        else:
            call = {"name": think.name, "args": {"thought": "keep looking"}, "id": f"call-{turn}"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


class _ProseOnForcedTurnFakeChatModel(FakeMessagesListChatModel):
    """Fake researcher that answers the reserved finalization turn in prose, not the output tool.

    ``ToolStrategy`` sets ``tool_choice="any"``, but that is a request rather than a guarantee:
    providers here do skip the structured-output tool, which is the whole reason
    ``StructuredResponseTextFallbackMiddleware`` exists. A worker that emits no tool call on its
    reserved turn is the one shape the tool-call refusal path cannot reach, because there is no
    tool call to refuse and no further graph iteration to check the limit before.
    """

    bound_tools: list[list[str]] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Record the tool names bound for this turn and keep the model itself unbound."""
        self.bound_tools.append([getattr(item, "name", None) or str(item) for item in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        forced = self.bound_tools[-1] == [RESEARCH_NOTES_TOOL] if self.bound_tools else False
        if forced:
            message = AIMessage(content="Here is a prose summary of what I found. It is not JSON.")
        else:
            turn = len(self.bound_tools)
            message = AIMessage(
                content="",
                tool_calls=[{"name": think.name, "args": {"thought": "keep looking"}, "id": f"call-{turn}"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _drive_researcher(agent) -> tuple[dict | None, Exception | None]:
    """Run one researcher worker to completion, tolerating either exit mechanism.

    Whether an exhausted worker ends by returning or by raising is the implementation's choice.
    These tests pin the note ``run_research_batch`` records, not the mechanism that produced it.
    """
    try:
        return agent.invoke({"messages": [HumanMessage(content="Batch research invocation.")]}), None
    except Exception as exc:  # noqa: BLE001 - the exit mechanism is deliberately not asserted
        return None, exc


def _note_for_researcher_run(agent) -> ResearchNotes:
    """Return the note ``run_research_batch`` would record for one researcher worker."""
    outcome, exc = _drive_researcher(agent)
    return _note_for_worker(_raise(exc) if exc is not None else (lambda: outcome))


def _researcher_agent(*, max_model_calls: int, finalize_when_forced: bool = True, model=None):
    """Build a researcher-shaped agent around the budget middleware pair."""
    model = model or _ResearcherLoopFakeChatModel(
        responses=[AIMessage(content="")],
        finalize_when_forced=finalize_when_forced,
    )
    agent = create_agent(
        model=model,
        tools=[think],
        system_prompt="rendered researcher prompt",
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max_model_calls + RESEARCHER_FINALIZATION_MODEL_CALLS,
                exit_behavior="error",
            ),
            ResearcherFinalizationMiddleware(max_model_calls=max_model_calls),
            StructuredResponseTextFallbackMiddleware(ResearchNotes),
        ],
        response_format=ResearchNotes,
    )
    return agent, model


class TestResearcherFinalizationMiddleware:
    """One budget, spent on research, then a single forced finalization turn."""

    def test_leaves_the_request_untouched_while_budget_remains(self):
        """A worker inside its budget researches exactly as it would with no guard installed."""
        middleware = ResearcherFinalizationMiddleware(max_model_calls=3)
        request = _request(calls_made=2)
        seen = []

        middleware.wrap_model_call(request, lambda req: seen.append(req) or "response")

        assert seen[0] is request
        assert [item.name for item in seen[0].tools] == [think.name]
        assert seen[0].messages == request.messages

    def test_withdraws_tools_and_forces_structured_output_at_the_budget(self, caplog):
        """The finalization turn leaves ResearchNotes as the only move the model has."""
        middleware = ResearcherFinalizationMiddleware(max_model_calls=3)
        request = _request(calls_made=3)
        seen = []

        middleware.wrap_model_call(request, lambda req: seen.append(req) or _finalized_response())

        finalization = seen[0]
        assert finalization.tools == []
        assert isinstance(finalization.response_format, ToolStrategy)
        assert finalization.response_format.schema is ResearchNotes
        # The system prompt is the researcher template verbatim; the nudge is an extra user turn.
        assert finalization.system_message.content == "rendered researcher prompt"
        assert "Return your ResearchNotes now" in finalization.messages[-1].content
        assert finalization.messages[:-1] == request.messages
        assert "entering finalization" in caplog.text

    @pytest.mark.asyncio
    async def test_async_path_finalizes_identically(self):
        """``ainvoke`` is the only path researcher workers use, so it must behave the same."""
        middleware = ResearcherFinalizationMiddleware(max_model_calls=1)
        seen = []

        async def handler(req):
            seen.append(req)
            return _finalized_response()

        await middleware.awrap_model_call(_request(calls_made=1), handler)

        assert seen[0].tools == []
        assert isinstance(seen[0].response_format, ToolStrategy)


class TestResearcherBudgetWiring:
    """The finalization middleware and the built-in call limit are installed as a pair."""

    def test_researcher_runnable_installs_the_pair_with_the_reserved_calls(self):
        """The limit is the budget plus the finalization calls, so finalizing is always possible."""
        with (
            patch(
                "deep_researcher_agent.agents.deep_researcher.factory.create_summarization_middleware",
                return_value=MagicMock(name="summarization"),
            ),
            patch("deep_researcher_agent.agents.deep_researcher.factory.create_agent") as create,
        ):
            build_researcher_runnable(
                researcher_model=MagicMock(),
                researcher_tools=[think],
                system_prompt="rendered researcher prompt",
                researcher_middleware=[],
                max_researcher_model_calls=12,
            )

        middleware = create.call_args.kwargs["middleware"]
        call_limit = next(item for item in middleware if isinstance(item, ModelCallLimitMiddleware))
        finalization = next(item for item in middleware if isinstance(item, ResearcherFinalizationMiddleware))
        fallback = next(item for item in middleware if isinstance(item, StructuredResponseTextFallbackMiddleware))
        assert call_limit.run_limit == 12 + RESEARCHER_FINALIZATION_MODEL_CALLS
        assert call_limit.exit_behavior == "error"
        assert finalization.max_model_calls == 12
        # Outside the text fallback, whose corrective call would otherwise re-enter finalization
        # within a single counted turn and hand the worker a second chance.
        assert middleware.index(finalization) < middleware.index(fallback)
        assert middleware.index(call_limit) < middleware.index(finalization)


class TestResearcherWorkerTermination:
    """A looping worker always terminates, and always yields exactly one ResearchNotes."""

    def test_finalization_turn_ends_the_worker_with_its_own_notes(self):
        """The forced turn produces the researcher's synthesis, not a code-built placeholder."""
        agent, model = _researcher_agent(max_model_calls=4)

        result = agent.invoke({"messages": [HumanMessage(content="Batch research invocation.")]})

        assert isinstance(result["structured_response"], ResearchNotes)
        assert result["structured_response"].summary.startswith("Partial synthesis")
        # Four research turns, then the one reserved finalization turn.
        assert len(model.bound_tools) == 5
        assert model.bound_tools[3] == [think.name, RESEARCH_NOTES_TOOL]
        assert model.bound_tools[4] == [RESEARCH_NOTES_TOOL]

    def test_worker_gets_exactly_one_finalization_turn(self):
        """A model that refuses to finalize is stopped, not given a second finalization prompt."""
        agent, model = _researcher_agent(max_model_calls=4, finalize_when_forced=False)

        with pytest.raises(ModelCallLimitExceededError):
            agent.invoke({"messages": [HumanMessage(content="Batch research invocation.")]})

        assert len(model.bound_tools) == 4 + RESEARCHER_FINALIZATION_MODEL_CALLS
        assert model.bound_tools[-1] == [RESEARCH_NOTES_TOOL]

    def test_tool_calls_made_after_the_budget_never_execute(self):
        """Emptying the model binding is not enough: the ToolNode still holds every tool."""
        agent, model = _researcher_agent(max_model_calls=2, finalize_when_forced=False)
        executed = []

        original = think.func

        def _record(thought: str) -> str:
            executed.append(thought)
            return original(thought)

        think.func = _record
        try:
            with pytest.raises(ModelCallLimitExceededError):
                agent.invoke({"messages": [HumanMessage(content="Batch research invocation.")]})
        finally:
            think.func = original

        # Two research turns ran think; the finalization turn asked for it and was refused.
        assert len(model.bound_tools) == 3
        assert executed == ["keep looking", "keep looking"]

    def test_last_research_turn_may_still_use_its_tools(self):
        """The refusal must not reach back into the turn that spent the final unit of budget."""
        middleware = ResearcherFinalizationMiddleware(max_model_calls=3)
        request = MagicMock()
        request.tool_call = {"name": think.name, "args": {"thought": "x"}, "id": "tc1"}
        request.state = {"run_model_call_count": 3}

        assert middleware.wrap_tool_call(request, lambda _req: "executed") == "executed"

        request.state = {"run_model_call_count": 4}
        refused = middleware.wrap_tool_call(request, lambda _req: "executed")
        assert refused.status == "error"
        assert "was not executed" in refused.content

    def test_a_prose_only_finalization_turn_still_ends_the_worker(self):
        """A worker that emits no tool call on its reserved turn must still stop there."""
        model = _ProseOnForcedTurnFakeChatModel(responses=[AIMessage(content="")])
        agent, _ = _researcher_agent(max_model_calls=3, model=model)

        _drive_researcher(agent)

        # Three research turns, then the one reserved finalization turn - and nothing after it.
        assert model.bound_tools == [
            [think.name, RESEARCH_NOTES_TOOL],
            [think.name, RESEARCH_NOTES_TOOL],
            [think.name, RESEARCH_NOTES_TOOL],
            [RESEARCH_NOTES_TOOL],
        ]

    def test_a_prose_only_finalization_turn_is_attributed_to_the_budget(self):
        """The budget is what stopped this worker, so the note must say so.

        The refusal path covers a worker that keeps calling tools past the budget. This is the
        other half: the model spends its reserved turn on prose, LangChain exits normally because
        there is no tool call to route, and ``ModelCallLimitMiddleware`` never gets another
        ``before_model`` in which to notice. The note must not blame a contract failure for a
        worker the budget stopped - that misattribution is exactly what the typed exhaustion
        branch exists to prevent, and it is what the documented
        ``Researcher worker exhausted its model-call budget`` log line is meant to surface.
        """
        model = _ProseOnForcedTurnFakeChatModel(responses=[AIMessage(content="")])
        agent, _ = _researcher_agent(max_model_calls=3, model=model)

        note = _note_for_researcher_run(agent)

        assert "model-call budget was exhausted" in note.summary
        assert "model-call budget was exhausted" in note.evidence_judgment.rationale
        assert note.findings == []
        assert note.evidence_judgment.relevance_score == 0

    def test_batch_records_an_exhausted_note_when_the_budget_ran_out(self, caplog):
        """The exhaustion path is typed, so the note can name the budget as the cause."""
        sensitive_query = "customer-content-identifier-12345"
        note = _note_for_worker(
            _raise(ModelCallLimitExceededError(0, 6, None, 6)),
            query=_query(sensitive_query),
        )

        assert note.findings == []
        assert "model-call budget was exhausted" in note.summary
        assert note.evidence_judgment.confidence == "low"
        assert note.evidence_judgment.relevance_score == 0
        assert sensitive_query not in caplog.text
        assert "customer-content-identifier-12345" not in caplog.text
        assert log_content_metadata(sensitive_query) in caplog.text

    def test_a_contract_failure_raises_rather_than_becoming_a_note(self):
        """Only the budget yields a code-built note; anything else stays a per-item failure.

        A worker that ends without notes while it still had budget left has broken its contract.
        ``run_research_batch`` surfaces that to the orchestrator with an invitation to resubmit
        the failed queries, which is a retry a fabricated empty note would silently spend.
        """
        with pytest.raises(ValueError, match="did not return structured ResearchNotes"):
            _note_for_worker(lambda: {"messages": [AIMessage(content="I am done.")]})

    def test_a_prose_only_finalization_note_is_not_a_contract_failure(self):
        """The budget path is the one exception, and it is reached by a typed error."""
        note = _note_for_worker(_raise(ResearcherBudgetExhaustedError(3, 3)))

        assert "model-call budget was exhausted" in note.summary
        assert note.findings == []

    def test_typed_exhaustion_error_reports_the_actual_call_count(self):
        """Diagnostics distinguish the configured limit from the observed count."""
        error = ResearcherBudgetExhaustedError(4, 3)

        assert "after 4 calls" in str(error)

    def test_truncated_notes_carry_the_query_and_an_explicit_gap(self):
        """Whatever stopped the worker, the writer sees the unsupported components, not silence."""
        query = _query()
        note = _note_for_worker(_raise(ModelCallLimitExceededError(0, 6, None, 6)), query=query)

        assert note.target_components == query.target_components
        assert query.query in note.gaps[0].description
        assert note.gaps[0].suggested_follow_up_queries == [query.query]
