# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for catalog-routing contracts."""

import pytest
from pydantic import ValidationError

from deep_researcher_agent.agents.chat_researcher.models import CatalogRoutingResponse


def test_catalog_response_accepts_current_gsf_shape() -> None:
    response = CatalogRoutingResponse.model_validate(
        {
            "request_id": "gsf-catalog-request-1",
            "coverage": 0.5,
            "candidates": [
                {
                    "label": "ColumnAttribute",
                    "attribute": "recognized_revenue",
                    "term": "Revenue",
                    "id": "attr:revenue",
                }
            ],
            "uncovered_entities": ["region"],
            "truncated": False,
        }
    )

    assert response.coverage == 0.5
    assert response.candidates[0].id == "attr:revenue"
    assert response.uncovered_entities == ["region"]


def test_catalog_response_allows_missing_request_id() -> None:
    response = CatalogRoutingResponse.model_validate(
        {
            "coverage": 0.5,
            "candidates": [],
            "uncovered_entities": None,
            "truncated": False,
        }
    )

    assert response.request_id is None


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "request-1", "candidates": []},
        {"request_id": "request-1", "coverage": "full", "candidates": []},
        {
            "request_id": "request-1",
            "coverage": 0.5,
            "candidates": [],
            "can_answer": True,
        },
    ],
)
def test_catalog_response_rejects_missing_or_legacy_routing_fields(payload: dict) -> None:
    with pytest.raises(ValidationError):
        CatalogRoutingResponse.model_validate(payload)
