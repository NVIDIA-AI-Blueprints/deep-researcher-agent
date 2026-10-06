# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for single-classification, code-owned catalog routing."""

import asyncio
import json
from time import monotonic

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel
from pydantic import ValidationError

from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState
from deep_researcher_agent.agents.chat_researcher.nodes.context_aware_intent_router import ContextAwareIntentRouter
from deep_researcher_agent.agents.chat_researcher.nodes.context_aware_intent_router import EntryClassification
from deep_researcher_agent.agents.chat_researcher.nodes.context_aware_intent_router import RoutingProtocolError

CATALOG_CANDIDATE = {
    "label": "ColumnAttribute",
    "attribute": "recognized_revenue",
    "term": "Revenue",
    "id": "attr:revenue",
}


class CatalogToolInput(BaseModel):
    question: str
    database_name: str | None = None
    max_results: int = 10
    max_distance: float = 0.75


class FakeClassifier:
    def __init__(self, *results):
        self.results = iter(results)
        self.contexts = []
        self.calls = 0

    async def ainvoke(self, inputs, *, config, context):
        del inputs, config
        self.calls += 1
        self.contexts.append(context)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeCatalogTool:
    def __init__(self, *results):
        self.results = iter(results)
        self.calls = []

    async def ainvoke(self, payload, config=None):
        del config
        self.calls.append(payload)
        return next(self.results)


class SlowRetryClassifier:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, inputs, *, config, context):
        del inputs, config, context
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0.4)
            raise RuntimeError("[500]")
        await asyncio.sleep(10)


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def _classification(
    interaction="new_research",
    *,
    catalog_action="skip",
    catalog_question=None,
    research_depth="shallow",
    meta_response=None,
    reasoning="The request needs bounded research.",
):
    if interaction != "new_research" and catalog_action == "skip":
        catalog_action = None
    if interaction in {"meta", "report_ask", "report_cosmetic_edit"} and research_depth == "shallow":
        research_depth = None
    if interaction == "meta" and meta_response is None:
        meta_response = "Hello."
    if interaction == "report_delta_research":
        research_depth = "deep"
    return EntryClassification(
        interaction=interaction,
        catalog_action=catalog_action,
        catalog_question=catalog_question,
        research_depth=research_depth,
        meta_response=meta_response,
        reasoning=reasoning,
    )


def _catalog_result(*, coverage=0.8, candidates=None, request_id="catalog-1"):
    if candidates is None:
        candidates = [CATALOG_CANDIDATE] if coverage >= 0.6 else []
    return json.dumps({"request_id": request_id, "coverage": coverage, "candidates": candidates})


def _router(classification, *catalog_results):
    router = ContextAwareIntentRouter.__new__(ContextAwareIntentRouter)
    router.classifier = FakeClassifier({"structured_response": classification})
    router.catalog_tool = FakeCatalogTool(*catalog_results)
    router.catalog_input_schema = CatalogToolInput
    router.catalog_source_id = "gsf"
    router.max_catalog_results = 10
    router.catalog_confidence_threshold = 0.6
    router.catalog_max_distance = 0.75
    router.callbacks = []
    router.llm_timeout = 5
    router.classifier_max_attempts = 2
    router.classifier_retry_delay_seconds = 0
    return router


def _state(query="query", **kwargs):
    return ChatResearcherState(messages=[HumanMessage(content=query)], **kwargs)


def test_classification_schema_has_one_catalog_applicability_field():
    schema = EntryClassification.model_json_schema()
    assert "structured_data_required" not in schema["properties"]
    assert set(schema["required"]) == {
        "interaction",
        "catalog_action",
        "catalog_question",
        "research_depth",
        "meta_response",
        "reasoning",
    }


def test_classification_enforces_interaction_fields():
    with pytest.raises(ValidationError, match="catalog action"):
        EntryClassification(
            interaction="new_research",
            catalog_action=None,
            catalog_question=None,
            research_depth="shallow",
            meta_response=None,
            reasoning="Research.",
        )
    with pytest.raises(ValidationError, match="only valid for catalog search"):
        _classification(catalog_question="query")


def test_router_classifier_has_no_catalog_tool(monkeypatch):
    captured = {}

    @tool
    async def catalog(question: str) -> str:
        """Search the catalog."""
        return question

    monkeypatch.setattr(
        "deep_researcher_agent.agents.chat_researcher.nodes.context_aware_intent_router.create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    ContextAwareIntentRouter(object(), catalog, "Query: {{ query }}")

    assert captured["tools"] == []
    assert captured["response_format"].schema is EntryClassification


@pytest.mark.asyncio
async def test_compiled_router_uses_one_llm_call_then_scoped_python_catalog_call():
    catalog_calls = []

    @tool
    async def catalog(
        question: str,
        max_results: int,
        max_distance: float,
        database_name: str | None = None,
    ) -> str:
        """Search the catalog."""
        catalog_calls.append((question, max_results, max_distance, database_name))
        return _catalog_result()

    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EntryClassification",
                        "args": _classification(
                            catalog_action="search",
                            catalog_question="Show revenue",
                        ).model_dump(),
                        "id": "classification-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    router = ContextAwareIntentRouter(model, catalog, "Query: {{ query }}")

    result = await router.run(_state("Show revenue", database_name="finance_prod"))

    assert model.calls == 1
    assert catalog_calls == [("Show revenue", 10, 0.75, "finance_prod")]
    assert result["user_intent"].target == "hybrid_research"


@pytest.mark.asyncio
async def test_transient_classifier_500_is_retried_once():
    router = _router(_classification())
    router.classifier = FakeClassifier(
        Exception("[500] {'message': 'Internal server error', 'code': 500}"),
        {"structured_response": _classification()},
    )

    result = await router.run(_state())

    assert result["user_intent"].target == "new_research"
    assert router.classifier.calls == 2


@pytest.mark.asyncio
async def test_unscoped_public_request_skips_catalog():
    router = _router(_classification(research_depth="deep"))

    result = await router.run(_state("Write a public market report"))

    assert result["user_intent"].target == "new_research"
    assert result["depth_decision"].decision == "deep"
    assert router.catalog_tool.calls == []


@pytest.mark.asyncio
async def test_unscoped_disabled_catalog_preserves_classic_fallback():
    router = _router(_classification(catalog_action="search", catalog_question="query", research_depth="deep"))

    result = await router.run(_state(data_sources=["web_search"]))

    assert result["user_intent"].target == "new_research"
    assert result["depth_decision"].decision == "deep"
    assert router.catalog_tool.calls == []


@pytest.mark.asyncio
async def test_scoped_disabled_catalog_fails_instead_of_using_web():
    router = _router(_classification(catalog_action="skip", research_depth="deep"))

    with pytest.raises(RoutingProtocolError, match="Database-scoped research requires the catalog source"):
        await router.run(_state(database_name="finance_prod", data_sources=["web_search"]))

    assert router.catalog_tool.calls == []


@pytest.mark.asyncio
async def test_scoped_classifier_skip_is_overridden_to_catalog_and_hybrid():
    router = _router(_classification(catalog_action="skip"), _catalog_result(coverage=0.1, candidates=[]))

    result = await router.run(_state("Show internal trends", database_name="finance_prod"))

    assert result["user_intent"].target == "hybrid_research"
    assert router.catalog_tool.calls[0]["question"] == "Show internal trends"
    assert router.catalog_tool.calls[0]["database_name"] == "finance_prod"


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["shallow", "deep"])
async def test_unscoped_below_threshold_preserves_fallback_depth(depth):
    router = _router(
        _classification(catalog_action="search", catalog_question="query", research_depth=depth),
        _catalog_result(coverage=0.59),
    )

    result = await router.run(_state())

    assert result["user_intent"].target == "new_research"
    assert result["depth_decision"].decision == depth


@pytest.mark.asyncio
async def test_threshold_coverage_routes_to_hybrid():
    router = _router(
        _classification(catalog_action="search", catalog_question="query"),
        _catalog_result(coverage=0.6),
    )

    result = await router.run(_state())

    assert result["user_intent"].target == "hybrid_research"


@pytest.mark.asyncio
async def test_scoped_empty_catalog_candidates_remain_hybrid():
    router = _router(
        _classification(catalog_action="search", catalog_question="query"),
        _catalog_result(coverage=0.0, candidates=[]),
    )

    result = await router.run(_state(database_name="finance_prod"))

    assert result["user_intent"].target == "hybrid_research"
    assert result["catalog_context"].candidates == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question,coverage",
    [
        (
            "Return all available columns and metadata for the Basketball Stars app from the playstore table, "
            "including app name, category, rating, reviews count, installs, type, price, content rating, genres, "
            "last updated date, current version, and android version.",
            0.5,
        ),
        (
            "Show the distribution of apps across install tiers: count the number of distinct apps and total reviews "
            "for each distinct Installs value, ordered by total reviews descending.",
            0.3333,
        ),
    ],
)
async def test_scoped_fda_requests_ignore_low_catalog_coverage(question, coverage):
    router = _router(
        _classification(catalog_action="search", catalog_question=question),
        _catalog_result(coverage=coverage, candidates=[CATALOG_CANDIDATE]),
    )

    result = await router.run(_state(question, database_name="app_store"))

    assert result["user_intent"].target == "hybrid_research"
    assert router.catalog_tool.calls[0]["database_name"] == "app_store"


@pytest.mark.asyncio
async def test_unscoped_high_coverage_without_candidates_is_rejected():
    router = _router(
        _classification(catalog_action="search", catalog_question="query"),
        _catalog_result(coverage=0.8, candidates=[]),
    )
    with pytest.raises(RoutingProtocolError, match="without candidates"):
        await router.run(_state())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,match",
    [
        (json.dumps({"status": "error", "code": "forbidden", "message": "denied"}), "failed"),
        ("not-json", "invalid JSON"),
        (json.dumps({"coverage": 0.8}), "invalid response"),
    ],
)
async def test_catalog_failures_do_not_silently_fall_back(payload, match):
    router = _router(_classification(catalog_action="search", catalog_question="query"), payload)
    with pytest.raises(RoutingProtocolError, match=match):
        await router.run(_state())


@pytest.mark.asyncio
async def test_catalog_tool_message_error_is_rejected():
    payload = ToolMessage(content="private provider detail", tool_call_id="catalog-1", status="error")
    router = _router(_classification(catalog_action="search", catalog_question="query"), payload)

    with pytest.raises(RoutingProtocolError, match="failed"):
        await router.run(_state())


@pytest.mark.asyncio
async def test_catalog_tool_message_artifact_takes_precedence_over_content():
    payload = ToolMessage(
        content="not-json",
        artifact={"request_id": "catalog-1", "coverage": 0.8, "candidates": [CATALOG_CANDIDATE]},
        tool_call_id="catalog-1",
    )
    router = _router(_classification(catalog_action="search", catalog_question="query"), payload)

    result = await router.run(_state())

    assert result["user_intent"].target == "hybrid_research"
    assert result["catalog_request_id"] == "catalog-1"


@pytest.mark.asyncio
async def test_classifier_timeout_is_shared_across_retries():
    router = _router(_classification())
    classifier = SlowRetryClassifier()
    router.classifier = classifier
    router.llm_timeout = 0.5
    router.classifier_retry_delay_seconds = 0

    started = monotonic()
    with pytest.raises(TimeoutError):
        await router.run(_state())
    elapsed = monotonic() - started

    assert elapsed < 0.75
    assert classifier.calls == 2


@pytest.mark.asyncio
async def test_python_owns_bounds_and_exact_database_scope():
    router = _router(
        _classification(catalog_action="search", catalog_question="query"),
        _catalog_result(coverage=0.4),
    )
    router.max_catalog_results = 7
    router.catalog_max_distance = 0.5

    result = await router.run(_state(database_name="finance_prod"))

    assert result["user_intent"].target == "hybrid_research"
    assert router.catalog_tool.calls == [
        {
            "question": "query",
            "database_name": "finance_prod",
            "max_results": 7,
            "max_distance": 0.5,
        }
    ]
    assert router.classifier.contexts[0]["database_scope_provided"] is True
    assert "database_name" not in router.classifier.contexts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("proposed", [None, "", "rewritten question"])
async def test_unsafe_catalog_span_falls_back_to_full_query(proposed):
    query = "Calculate recognized sales from internal order records"
    router = _router(
        _classification(catalog_action="search", catalog_question=proposed),
        _catalog_result(),
    )

    await router.run(_state(query))

    assert router.catalog_tool.calls[0]["question"] == query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interaction,expected_target",
    [("meta", "meta"), ("report_ask", "report"), ("report_cosmetic_edit", "report")],
)
async def test_meta_and_active_report_routes_ignore_incidental_scope(interaction, expected_target):
    router = _router(_classification(interaction))

    result = await router.run(_state(active_report_job_id="report-1", database_name="finance_prod"))

    assert result["user_intent"].target == expected_target
    assert router.catalog_tool.calls == []
