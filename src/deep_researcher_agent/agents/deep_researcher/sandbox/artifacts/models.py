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

"""Durable artifact record.

This is the metadata record only. Bytes are stored out-of-band by the
``ArtifactStore`` and fetched server-side on demand; they never enter the agent's
context or conversation history. Reports reference artifacts by ``artifact_id``
(``artifact://<id>``), keeping prompt cost independent of artifact size.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel
from pydantic import Field


class ArtifactKind(StrEnum):
    """High-level artifact category used for rendering and grouping."""

    IMAGE = "image"
    TABLE = "table"
    DATASET = "dataset"
    NOTEBOOK = "notebook"
    DOCUMENT = "document"
    TEXT = "text"
    ARCHIVE = "archive"
    OTHER = "other"


class ArtifactStatus(StrEnum):
    """Lifecycle status of an artifact record."""

    PENDING = "pending"
    AVAILABLE = "available"
    REJECTED = "rejected"
    DELETED = "deleted"


class ArtifactProvenance(BaseModel):
    """Reproducibility metadata for a generated artifact."""

    command: str | None = Field(default=None, description="Command that produced the artifact")
    script_sha256: str | None = Field(default=None, description="Digest of the generating script")
    input_file_hashes: dict[str, str] = Field(default_factory=dict, description="Path -> sha256 of input files")
    package_snapshot: tuple[str, ...] = Field(default=(), description="Installed package versions at run time")


class Artifact(BaseModel):
    """Durable record for a single generated artifact (metadata, not bytes)."""

    artifact_id: str = Field(..., max_length=64, description="Stable ID (UUID, optionally suffixed with a digest)")
    job_id: str = Field(..., max_length=64, description="Owning async job (retention + authorization boundary)")
    kind: ArtifactKind = Field(default=ArtifactKind.OTHER)
    mime_type: str = Field(..., description="MIME type validated from bytes, not just filename")
    filename: str = Field(..., description="User-facing filename")
    sandbox_path: str = Field(..., description="Original path inside the sandbox")
    storage_uri: str = Field(..., description="Durable location of the bytes (local path or object-store URI)")
    sha256: str = Field(..., min_length=64, max_length=64, description="Content digest for integrity and deduplication")
    size_bytes: int = Field(..., ge=0, description="Byte size for quota and UI display")
    title: str | None = Field(default=None, description="Optional display title")
    caption: str | None = Field(default=None, description="Optional report caption")
    inline: bool = Field(default=False, description="Whether the report may embed it inline")
    workflow: str | None = Field(default=None, description="orchestrator | planner-agent | researcher-agent | skill")
    source_tool_call_id: str | None = Field(default=None, description="Tool call that created or registered it")
    provenance: ArtifactProvenance = Field(default_factory=ArtifactProvenance)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Event ordering timestamp")
    status: ArtifactStatus = Field(default=ArtifactStatus.PENDING)

    def to_sse_payload(self, content_url: str) -> dict[str, object]:
        """Build the canonical ``artifact.update`` SSE payload for live UI updates.

        Args:
            content_url: Authenticated URL where the bytes can be fetched.

        Returns:
            A JSON-serializable payload (no bytes) matching the API/UI event contract.
        """
        return {
            "type": "artifact.update",
            "name": self.filename,
            "data": {
                "type": "file",
                "url": content_url,
                "content_url": content_url,
                "file_path": self.filename,
                "artifact_id": self.artifact_id,
                "job_id": self.job_id,
                "kind": self.kind.value,
                "mime_type": self.mime_type,
                "size_bytes": self.size_bytes,
                "sha256": self.sha256,
                "title": self.title,
                "caption": self.caption,
                "inline": self.inline,
            },
        }
