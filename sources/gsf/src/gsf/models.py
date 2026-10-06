# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, NAT-independent contracts for GSF capabilities."""

from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints

DatabaseName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
    ),
]


class GSFRequest(BaseModel):
    """Base model for data sent from Deep Researcher Agent to GSF."""

    model_config = ConfigDict(extra="forbid")


class GSFResponse(BaseModel):
    """Base model for the validated subset of data returned by GSF."""

    model_config = ConfigDict(extra="ignore")


class CatalogSearchRequest(GSFRequest):
    """Find semantic candidates relevant to an enterprise-data question."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: DatabaseName | None = None
    max_results: int = Field(default=10, ge=1, le=100)
    max_distance: float = Field(default=0.75, gt=0)


class CatalogCandidate(GSFResponse):
    """A semantic candidate returned by GSF entity-coverage search."""

    label: str
    attribute: str
    term: str
    id: str


class CatalogSearchResponse(GSFResponse):
    """Coverage and ranked semantic candidates returned by GSF."""

    request_id: str | None = None
    coverage: float | None = Field(default=None, ge=0, le=1)
    candidates: list[CatalogCandidate]
    uncovered_entities: list[str] | None = None
    truncated: bool = False


class ResultColumn(GSFResponse):
    """A column in a bounded SQL result."""

    name: str
    data_type: str | None = None


class SemanticContext(GSFResponse):
    """Semantic provenance used to produce a SQL query."""

    metrics: list[dict[str, Any]] = Field(default_factory=list)
    grain: str | None = None
    units: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)


class TextToSQLRequest(GSFRequest):
    """Generate and execute validated SQL with bounded results."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: DatabaseName | None = None
    max_rows: int = Field(default=1_000, ge=1)


class TextToPQLRequest(GSFRequest):
    """Run a natural-language question through GSF's PQL prediction path."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: DatabaseName | None = Field(
        default=None,
        description="Optional benchmark-only database selector; normal Deep Researcher Agent calls leave this unset.",
    )
    max_rows: int = Field(default=1_000, ge=1)


class TextToSQLResponse(GSFResponse):
    """Validated SQL, bounded rows, and semantic provenance returned by GSF."""

    request_id: str | None = None
    thoughts: str | None = None
    sql: str
    columns: list[ResultColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    custom_analyses_used: list[Any] | None = None
    objects_used: list[str] | None = None
    joins_used: list[dict[str, Any]] | None = None
    semantic_context: SemanticContext | None = None
    validation_attempts: list[dict[str, Any]] | None = None
    assumptions: list[str] | None = None
    warnings: list[str] | None = None
    timings: dict[str, int | float] | None = None


class TextToPQLResponse(GSFResponse):
    """PQL, bounded prediction results, and diagnostic context returned by GSF."""

    request_id: str | None = None
    response: str | None = None
    thoughts: str | None = None
    pql: str | None = None
    columns: list[ResultColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    custom_analyses_used: list[Any] | None = None
    objects_used: list[str] | None = None
    semantic_context: SemanticContext | None = None
    assumptions: list[str] | None = None
    warnings: list[str] | None = None
    timings: dict[str, int | float] | None = None


class QueryContextRequest(GSFRequest):
    """Build compact, authorized context for a later SQL-generation step."""

    question: str = Field(min_length=1, max_length=4_096)
    database_name: DatabaseName | None = None
    object_ids: list[str] = Field(default_factory=list)
    token_budget: int | None = Field(default=None, ge=1)
