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

"""Tests for deep researcher structured response contracts."""

import pytest
from pydantic import ValidationError

from deep_researcher_agent.agents.deep_researcher.models import AnswerStrategy
from deep_researcher_agent.agents.deep_researcher.models import Constraint
from deep_researcher_agent.agents.deep_researcher.models import EvidenceJudgment
from deep_researcher_agent.agents.deep_researcher.models import ResearchNotes
from deep_researcher_agent.agents.deep_researcher.models import ResearchPlan
from deep_researcher_agent.agents.deep_researcher.models import ResearchQuery
from deep_researcher_agent.agents.deep_researcher.models import SourceRoutingPlan
from deep_researcher_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_QUERIES


def _answer_strategy() -> dict:
    return {
        "answer_type": "comparison",
        "title": "CUDA and OpenCL Trade-offs",
        "required_components": [
            {
                "id": "programming_model",
                "name": "Programming model",
                "description": "Compare kernel, memory, and execution models.",
            }
        ],
    }


def _task_analysis() -> dict:
    return {
        "user_intent": "Understand CUDA and OpenCL trade-offs.",
        "explicit_requirements": ["Compare CUDA and OpenCL"],
        "implicit_requirements": ["Cover ecosystem and portability"],
        "out_of_scope": ["General GPU purchasing advice"],
        "language": "English",
    }


def _research_query(**overrides) -> dict:
    payload = {
        "query": "CUDA OpenCL portability",
        "subqueries": [],
        "preferred_tools": ["web_search_tool"],
        "fallback_tools": [],
        "target_components": ["programming_model"],
        "rationale": "Supports the comparison component.",
    }
    payload.update(overrides)
    return payload


def test_research_plan_contract_validates_expected_shape():
    plan = ResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": _answer_strategy(),
            "constraints": [
                {
                    "category": "content",
                    "constraint": "Compare portability, performance, and ecosystem maturity.",
                    "rationale": "These dimensions determine practical adoption.",
                }
            ],
            "queries": [
                {
                    "query": "CUDA OpenCL portability performance ecosystem comparison",
                    "subqueries": ["CUDA OpenCL portability", "CUDA OpenCL benchmark comparison"],
                    "preferred_tools": ["web_search_tool"],
                    "fallback_tools": [],
                    "target_components": ["programming_model"],
                    "rationale": "Supports the comparison component.",
                }
            ],
        }
    )

    assert plan.answer_strategy.required_components[0].id == "programming_model"
    assert plan.constraints[0].category == "content"
    assert plan.queries[0].target_components == ["programming_model"]
    assert plan.queries[0].subqueries == ["CUDA OpenCL portability", "CUDA OpenCL benchmark comparison"]
    assert plan.queries[0].preferred_tools == ["web_search_tool"]
    assert plan.queries[0].fallback_tools == []


def test_research_plan_contract_accepts_prediction_answer_type():
    answer_strategy = _answer_strategy()
    answer_strategy["answer_type"] = "prediction"
    answer_strategy["title"] = "Election Forecast"

    plan = ResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": answer_strategy,
            "constraints": [],
            "queries": [
                {
                    "query": "Example election forecast evidence",
                    "subqueries": [],
                    "preferred_tools": ["polymarket_search_tool"],
                    "fallback_tools": [],
                    "target_components": ["programming_model"],
                    "rationale": "Supports the forecast evidence component.",
                }
            ],
        }
    )

    assert plan.answer_strategy.answer_type == "prediction"
    assert plan.queries[0].preferred_tools == ["polymarket_search_tool"]


def test_research_query_text_boundaries_are_enforced_per_nested_item():
    """Per-item bounds prevent deeply nested plan strings from evading aggregate quotas."""
    accepted = ResearchQuery.model_validate(
        _research_query(
            query="q" * 4096,
            subqueries=["s" * 2048],
        )
    )

    assert len(accepted.query) == 4096
    assert len(accepted.subqueries[0]) == 2048

    with pytest.raises(ValidationError):
        ResearchQuery.model_validate(_research_query(query="q" * 4097))
    with pytest.raises(ValidationError):
        ResearchQuery.model_validate(_research_query(subqueries=["s" * 2049]))


def test_research_plan_query_count_matches_shared_security_ceiling():
    """The structured schema matches the immutable per-job query-count ceiling."""
    base = {
        "task_analysis": _task_analysis(),
        "answer_strategy": _answer_strategy(),
        "constraints": [],
    }

    accepted = ResearchPlan.model_validate(
        {**base, "queries": [_research_query() for _ in range(DEFAULT_MAX_RESEARCH_QUERIES)]}
    )

    assert len(accepted.queries) == DEFAULT_MAX_RESEARCH_QUERIES
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {**base, "queries": [_research_query() for _ in range(DEFAULT_MAX_RESEARCH_QUERIES + 1)]}
        )


def test_reduced_answer_strategy_contract_validates():
    strategy = AnswerStrategy.model_validate(_answer_strategy())

    assert strategy.answer_type == "comparison"
    assert strategy.title == "CUDA and OpenCL Trade-offs"
    assert strategy.required_components[0].id == "programming_model"


def test_constraint_contract_rejects_verification_field():
    with pytest.raises(ValidationError):
        Constraint.model_validate(
            {
                "category": "content",
                "constraint": "Compare portability, performance, and ecosystem maturity.",
                "rationale": "These dimensions determine practical adoption.",
                "verification": "Each dimension appears in the final answer.",
            }
        )


def test_research_notes_contract_validates_expected_shape():
    notes = ResearchNotes.model_validate(
        {
            "query_topic": "CUDA vs OpenCL portability",
            "target_components": ["programming_model"],
            "summary": "CUDA is NVIDIA-specific while OpenCL targets cross-vendor portability.",
            "findings": [
                {
                    "claim": "OpenCL is designed for cross-vendor heterogeneous compute.",
                    "evidence": "The source describes OpenCL as an open standard for heterogeneous platforms.",
                    "source_ids": [1],
                    "confidence": "high",
                    "caveats": ["Portability does not guarantee equal performance across vendors."],
                }
            ],
            "gaps": [
                {
                    "description": "Recent benchmark coverage is sparse.",
                    "impact": "Limits quantitative comparison.",
                    "suggested_follow_up_queries": ["CUDA OpenCL benchmark 2026"],
                }
            ],
            "sources": [
                {
                    "id": 1,
                    "title": "OpenCL Overview",
                    "source_type": "url",
                    "locator": "https://example.test/opencl",
                }
            ],
            "narrative_notes": "OpenCL offers broader portability, while CUDA typically has deeper vendor tooling.",
            "language": "English",
        }
    )

    assert notes.target_components == ["programming_model"]
    assert notes.findings[0].source_ids == [1]
    assert notes.sources[0].source_type == "url"
    assert notes.sources[0].locator == "https://example.test/opencl"
    assert notes.evidence_judgment is None


def test_research_notes_contract_accepts_evidence_judgment():
    notes = ResearchNotes.model_validate(
        {
            "query_topic": "CUDA vs OpenCL portability",
            "target_components": ["programming_model"],
            "summary": "CUDA is NVIDIA-specific while OpenCL targets portability.",
            "findings": [],
            "gaps": [],
            "sources": [],
            "narrative_notes": "OpenCL offers broader portability.",
            "language": "English",
            "evidence_judgment": {
                "relevance_score": 85,
                "confidence": "high",
                "rationale": "Directly supports the programming model component.",
            },
        }
    )

    assert notes.evidence_judgment is not None
    assert notes.evidence_judgment.relevance_score == 85
    assert notes.evidence_judgment.confidence == "high"


def test_evidence_judgment_contract_rejects_invalid_score():
    with pytest.raises(ValidationError):
        EvidenceJudgment.model_validate(
            {
                "relevance_score": 101,
                "confidence": "high",
                "rationale": "Score must stay within the configured range.",
            }
        )


def test_source_routing_plan_contract_validates_expected_shape():
    route = SourceRoutingPlan.model_validate(
        {
            "domain_id": "current_news",
            "domain_name": "Current News",
            "routing_reason": "The user asks for recent developments.",
            "recommendations": [
                {
                    "source_id": "news_search",
                    "tool_names": ["duckduckgo_news_search_tool"],
                    "priority": 1,
                    "rationale": "Best fit for recent news.",
                }
            ],
            "fallback_sources": [
                {
                    "source_id": "web_search",
                    "tool_names": ["web_search_tool"],
                    "priority": 2,
                    "rationale": "Broad web fallback.",
                }
            ],
            "planner_guidance": "Use news_search first, then web_search if coverage is weak.",
        }
    )

    assert route.domain_id == "current_news"
    assert route.recommendations[0].tool_names == ["duckduckgo_news_search_tool"]


def test_subagent_contracts_reject_extra_fields_and_old_plan_shape():
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "answer_strategy": _answer_strategy(),
                "constraints": [],
                "queries": [],
                "unexpected": "value",
            }
        )

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "report_title": "Title",
                "report_toc": [],
                "constraints": [],
                "queries": [],
            }
        )

    with pytest.raises(ValidationError):
        ResearchNotes.model_validate(
            {
                "query_topic": "CUDA vs OpenCL portability",
                "target_sections": ["Programming Model Differences"],
                "summary": "Old field should fail.",
                "findings": [],
                "gaps": [],
                "sources": [],
                "narrative_notes": "",
                "language": "English",
            }
        )

    for removed_field, value in (
        ("assembly_instruction", "Synthesize evidence into a comparison."),
        ("selection_mode", "none"),
        ("expected_count", None),
        ("options", []),
    ):
        old_strategy = _answer_strategy()
        old_strategy[removed_field] = value
        with pytest.raises(ValidationError):
            AnswerStrategy.model_validate(old_strategy)
