# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog routing contracts."""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class CatalogCandidate(BaseModel):
    """A ranked semantic candidate returned by GSF entity-coverage search."""

    model_config = ConfigDict(extra="forbid")

    label: str
    attribute: str
    term: str
    id: str


class CatalogRoutingResponse(BaseModel):
    """Validated GSF catalog result used to select the research workflow."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    coverage: float = Field(ge=0, le=1)
    candidates: list[CatalogCandidate]
    uncovered_entities: list[str] | None = None
    truncated: bool = False
