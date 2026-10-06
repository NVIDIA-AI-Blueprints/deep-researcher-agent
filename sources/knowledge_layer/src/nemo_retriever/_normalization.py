# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure NeMo Retriever response normalization shared by adapter modes."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from deep_researcher_agent.knowledge import Chunk
from deep_researcher_agent.knowledge import ContentType
from deep_researcher_agent.knowledge.schema import FileStatus

from ._models import QueryHitWire

_PHYSICAL_STORAGE_KEYS = frozenset(
    {
        "database_uri",
        "lance_uri",
        "lancedb_uri",
        "physical_table",
        "table_name",
        "table_path",
        "vdb_uri",
    }
)
UNSUPPORTED_FILTERS_ERROR = "NeMo Retriever metadata filters are not supported by the query contract"
_SUCCESS_STATUSES = frozenset({"completed", "indexed", "ready", "success", "succeeded"})
_FAILED_STATUSES = frozenset({"failed", "error"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def strict_bool(value: Any, *, name: str) -> bool:
    """Parse booleans without treating arbitrary non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"{name} must be a boolean")


def status_to_file_status(status: str) -> FileStatus:
    """Map NRL terminal and transient document states to the Deep Researcher Agent contract."""
    normalized = status.lower()
    if normalized in _SUCCESS_STATUSES:
        return FileStatus.SUCCESS
    if normalized in _FAILED_STATUSES:
        return FileStatus.FAILED
    if normalized == "pending":
        return FileStatus.UPLOADING
    return FileStatus.INGESTING


def scrub_metadata(value: Any) -> Any:
    """Remove backend-owned physical storage selectors from public metadata."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _PHYSICAL_STORAGE_KEYS or "lancedb" in normalized:
                continue
            result[str(key)] = scrub_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [scrub_metadata(item) for item in value]
    return value


def normalize_content_type(value: str | None) -> ContentType:
    """Map NRL content labels to Deep Researcher Agent's four public content types."""
    normalized = (value or "").strip().lower()
    if "table" in normalized:
        return ContentType.TABLE
    if "chart" in normalized or "plot" in normalized or "graph" in normalized:
        return ContentType.CHART
    if any(token in normalized for token in ("image", "figure", "graphic")):
        return ContentType.IMAGE
    return ContentType.TEXT


def _safe_structured_data(metadata: dict[str, Any]) -> str | None:
    for key in ("structured_data", "table_content", "chart_data"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return None


def _bbox(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def normalize_query_hit(raw_result: Any) -> Chunk:
    """Validate one NRL public query hit and preserve its native distance."""
    hit = raw_result if isinstance(raw_result, QueryHitWire) else QueryHitWire.model_validate(raw_result)
    metadata = scrub_metadata(hit.metadata)
    bounding_box = _bbox(hit.bbox if hit.bbox is not None else hit.bbox_xyxy_norm)
    metadata.update(
        {
            "document_id": hit.document_id,
            "source": scrub_metadata(hit.source),
            "source_id": hit.source_id,
            "bounding_box": bounding_box,
        }
    )
    content_type = normalize_content_type(hit.content_type)
    page_number = hit.page_number
    citation = f"{hit.filename}, p.{page_number}" if page_number else hit.filename
    image_storage_uri = hit.stored_image_uri or None
    image_url = _http_url(image_storage_uri)
    if image_url is None and content_type == ContentType.IMAGE:
        image_url = _http_url(hit.source)
    return Chunk(
        chunk_id=hit.chunk_id,
        content=hit.text or "",
        score=0.0,
        distance=hit.distance,
        file_name=hit.filename,
        page_number=page_number,
        display_citation=citation,
        content_type=content_type,
        content_subtype=hit.content_type if hit.content_type and hit.content_type != content_type.value else None,
        structured_data=_safe_structured_data(metadata),
        image_storage_uri=image_storage_uri,
        image_url=image_url,
        metadata=metadata,
    )


__all__ = [
    "UNSUPPORTED_FILTERS_ERROR",
    "normalize_query_hit",
    "scrub_metadata",
    "status_to_file_status",
    "strict_bool",
]
