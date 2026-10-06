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

"""Structured response contracts for deep researcher planning, research, and synthesis."""

from typing import Annotated
from typing import ClassVar
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints

from ..resource_limits import DEFAULT_MAX_RESEARCH_QUERIES

ResearchQueryText = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
ResearchSubqueryText = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
ToolName = Annotated[str, StringConstraints(min_length=1, max_length=256)]
ComponentId = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class _StrictContract(BaseModel):
    """Base model for structured response schemas."""

    model_config: ClassVar[ConfigDict] = {"extra": "forbid"}


class TaskAnalysis(_StrictContract):
    """Planner analysis of the user's research request."""

    user_intent: str = Field(description="Brief statement of what the user wants to achieve.")
    explicit_requirements: list[str] = Field(description="Requirements explicitly stated by the user.")
    implicit_requirements: list[str] = Field(description="Requirements implied by the request.")
    out_of_scope: list[str] = Field(description="Tangential topics that should be excluded from the report.")
    language: str = Field(description="Language to use for the plan, notes, and final report.")


class AnswerComponent(_StrictContract):
    """Required evidence or synthesis component for the final answer."""

    id: str = Field(description="Stable component identifier, such as 'latest_price_anchor'.")
    name: str = Field(description="Short human-readable component name.")
    description: str = Field(description="What the writer must cover for this component.")


class AnswerStrategy(_StrictContract):
    """Planner guidance for the final answer shape and synthesis logic."""

    answer_type: Literal[
        "long_form_report",
        "brief_answer",
        "table",
        "comparison",
        "prediction",
        "multiple_choice",
        "data_extraction",
        "custom",
    ] = Field(description="The intended final output shape.")
    title: str = Field(description="Concise human-facing title for the final output.")
    required_components: list[AnswerComponent] = Field(
        description="Evidence and synthesis components that must be covered in the final answer."
    )


class Constraint(_StrictContract):
    """Lightweight final-answer requirement."""

    category: Literal["content", "source", "structure", "depth", "format", "exclusion"] = Field(
        description="Constraint category."
    )
    constraint: str = Field(description="Specific, actionable constraint text.")
    rationale: str = Field(description="Why this constraint exists.")


class SourceRecommendation(_StrictContract):
    """A source-router recommendation for the planner."""

    source_id: str = Field(description="Configured data source ID to use.")
    tool_names: list[str] = Field(description="Exact available source tool names under this source.")
    priority: int = Field(ge=1, le=3, description="Priority rank for this source: 1 is highest, 3 is lowest.")
    rationale: str = Field(description="Why this source should support the request.")


class SourceRoutingPlan(_StrictContract):
    """Advisory source route produced before planning."""

    domain_id: str = Field(description="Best-fit configured domain route for this request.")
    domain_name: str = Field(description="Human-readable domain name.")
    routing_reason: str = Field(description="Why this domain/source route fits the user request.")
    recommendations: list[SourceRecommendation] = Field(description="Primary source recommendations.")
    fallback_sources: list[SourceRecommendation] = Field(description="Fallback sources if primary sources are weak.")
    planner_guidance: str = Field(description="Concise instructions the planner should apply when writing queries.")


class ResearchQuery(_StrictContract):
    """Self-contained research query for a researcher worker."""

    query: ResearchQueryText = Field(
        description="Specific, self-contained search or document query.",
    )
    subqueries: list[ResearchSubqueryText] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Optional ordered concrete search angles for distinct facets unlikely to be covered by the main query. "
            "Prefer leaving this empty for focused queries and creating separate ResearchQuery items for independent "
            "evidence needs."
        ),
    )
    preferred_tools: list[ToolName] = Field(
        min_length=1,
        max_length=16,
        description=(
            "Ordered exact available source tool names to prioritize for this query. "
            "The first item is the primary tool the researcher should use first."
        ),
    )
    fallback_tools: list[ToolName] = Field(
        default_factory=list,
        max_length=16,
        description="Ordered exact available source tool names to use for corroboration or gaps.",
    )
    target_components: list[ComponentId] = Field(
        max_length=32,
        description="Answer components this query is intended to support.",
    )
    rationale: str = Field(max_length=4096, description="Why this query is needed.")


class ResearchPlan(_StrictContract):
    """Structured plan produced by the planner subagent."""

    task_analysis: TaskAnalysis = Field(description="Planner analysis of the user's request.")
    answer_strategy: AnswerStrategy = Field(description="Final answer shape and synthesis strategy.")
    constraints: list[Constraint] = Field(description="Lightweight requirements for the final answer.")
    queries: list[ResearchQuery] = Field(
        max_length=DEFAULT_MAX_RESEARCH_QUERIES,
        description="Queries for researcher workers to execute.",
    )


class ResearchSource(_StrictContract):
    """Source used by a researcher worker."""

    id: int = Field(description="Integer source identifier used by findings in this note.")
    title: str = Field(description="Source title or document name.")
    source_type: Literal["url", "internal_document", "tool"] = Field(
        description="Kind of source referenced by locator."
    )
    locator: str = Field(
        description=(
            "URL for web sources, document/page citation for internal documents, "
            "or raw tool name for URL-less structured tool results."
        )
    )


class ResearchFinding(_StrictContract):
    """Atomic finding captured from one or more sources."""

    claim: str = Field(description="Concise factual claim or analytical conclusion.")
    evidence: str = Field(description="Detailed supporting evidence, including dates, figures, names, and context.")
    source_ids: list[int] = Field(description="IDs from the sources list that support this finding.")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in the finding.")
    caveats: list[str] = Field(description="Limitations, disagreements, or context needed to use this finding.")


class ResearchGap(_StrictContract):
    """Information gap identified during research."""

    description: str = Field(description="Missing or weakly supported information.")
    impact: str = Field(description="Why the gap matters for the final report.")
    suggested_follow_up_queries: list[str] = Field(description="Queries that could close the gap.")


class EvidenceJudgment(_StrictContract):
    """Post-research judgment attached to a research note."""

    relevance_score: int = Field(
        ge=0,
        le=100,
        description="How useful this note is for the final answer, from 0 to 100.",
    )
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in this judgment.")
    rationale: str = Field(description="Concise explanation of the relevance score and confidence.")


class ResearchNotes(_StrictContract):
    """Structured notes produced by a researcher worker."""

    query_topic: str = Field(description="Short topic label for this research note.")
    target_components: list[str] = Field(description="Answer components these notes support.")
    summary: str = Field(description="Brief synthesis of the research results.")
    findings: list[ResearchFinding] = Field(description="Detailed findings supported by cited sources.")
    gaps: list[ResearchGap] = Field(description="Open gaps or weak spots discovered during research.")
    sources: list[ResearchSource] = Field(description="Every source used by these notes.")
    narrative_notes: str = Field(description="Detailed synthesis preserving nuance for final answer writing.")
    language: str = Field(description="Language used in these research notes.")
    evidence_judgment: EvidenceJudgment | None = Field(
        default=None,
        description="Researcher self-assessment of this note's usefulness for final synthesis.",
    )
