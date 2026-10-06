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
"""
Universal Schema for Knowledge Layer.

This module defines the "Golden Record" - strict Pydantic models that all
adapters must output. This ensures agents always see a consistent format
regardless of the underlying backend (LlamaIndex, Foundational RAG, and so on).

Schema Rules (enforced by all adapters):
1. Four Pillars: content_type MUST be exactly "text", "table", "chart", or "image"
2. Display Citation: display_citation MUST be populated with human-readable string
3. Visual Safety: content MUST NEVER be None (use empty string for visuals)
4. Data vs View: Raw data in structured_data, renderable image in image_url
5. Link Rot: image_url MUST be presigned URL, not internal S3 path
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class ContentType(StrEnum):
    """
    The Four Pillars - strict content categorization.

    All adapters MUST map their internal types to one of these four categories.
    The frontend relies on this for component switching.
    """

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"


class Chunk(BaseModel):
    """
    The Atomic Unit of Knowledge (The 'Golden Record').

    This schema unifies data from ANY backend (NV-Ingest, LlamaIndex, and so on)
    so the Agent always sees a consistent format.
    """

    # --- 1. Core Data (The Agent Reads This) ---
    chunk_id: str = Field(..., description="Unique ID for citation tracking.")
    content: str = Field(
        ...,
        description="The main text. If this is a visual/table, this MUST be the summary or caption.",
    )
    score: float = Field(0.0, ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0).")
    distance: float | None = Field(
        None,
        allow_inf_nan=False,
        description=(
            "Optional finite backend-native vector distance; lower values are closer, there is no fixed range, and "
            "values are not necessarily comparable across backends or metrics."
        ),
    )

    # --- 2. Strict Citation Contract (For UI Reliability) ---
    file_name: str = Field(..., description="Original filename (for example, 'Q3_Report.pdf').")
    page_number: int | None = Field(
        None,
        ge=1,
        description="The page number (1-based). None if not applicable (for example, JSON/TXT).",
    )
    display_citation: str = Field(
        ...,
        description="User-facing citation label. Adapters MUST populate this.",
    )

    # --- 3. The Four Pillars (Strictly Typed) ---
    content_type: ContentType = Field(
        ...,
        description="The semantic category of this chunk.",
    )
    content_subtype: str | None = Field(
        None,
        description="Granular subtype (for example, 'bar_chart', 'pie_chart').",
    )

    # --- 4. The 'Deep' Payload (Optional) ---
    structured_data: str | None = Field(
        None,
        description="Raw code/data representation for Code Interpreter analysis.",
    )

    # --- 5. Visual Assets (Optional) ---
    image_storage_uri: str | None = Field(
        None,
        description="Internal S3/MinIO URI for system access.",
    )
    image_url: str | None = Field(
        None,
        description="Presigned HTTP URL for frontend display.",
    )

    # --- 6. Extra Metadata ---
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Passthrough for extra backend-specific metadata.",
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_content_string(cls, data: Any) -> Any:
        """Safety check: Ensure content is never None, even if backend returns None."""
        if isinstance(data, dict):
            if data.get("content") is None:
                data["content"] = ""
        return data


class RetrievalResult(BaseModel):
    """The container object returned by the retrieve_documents tool."""

    chunks: list[Chunk] = Field(default_factory=list)
    total_tokens: int = Field(0, ge=0, description="Estimated token count for context management.")
    query: str = Field(..., description="The original query that produced these results.")
    backend: str = Field(..., description="The backend that produced these results.")
    success: bool = Field(default=True, description="Whether retrieval succeeded.")
    error_message: str | None = Field(default=None, description="Error details if retrieval failed.")


class JobState(StrEnum):
    """Ingestion job states for async processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileStatus(StrEnum):
    """File processing status for UI tracking."""

    UPLOADING = "uploading"
    INGESTING = "ingesting"
    SUCCESS = "success"
    FAILED = "failed"


class CollectionInfo(BaseModel):
    """
    Metadata about a collection/index.

    Used by UI to display available sources/collections.
    """

    name: str = Field(..., description="Unique collection name.")
    description: str | None = Field(None, description="Human-readable description.")
    file_count: int = Field(0, ge=0, description="Number of files in this collection.")
    chunk_count: int = Field(0, ge=0, description="Total number of chunks/vectors.")
    created_at: datetime | None = Field(None, description="When the collection was created.")
    updated_at: datetime | None = Field(None, description="Last modification time.")
    backend: str = Field(..., description="Backend that manages this collection.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific metadata (e.g., embedding model, vector dimensions).",
    )


class FileInfo(BaseModel):
    """
    Metadata about a file within a collection.

    Used by UI to display files and their processing status.
    """

    file_id: str = Field(..., description="Unique file identifier.")
    file_name: str = Field(..., description="Original filename.")
    collection_name: str = Field(..., description="Collection this file belongs to.")
    status: FileStatus = Field(default=FileStatus.UPLOADING, description="Current processing status.")
    file_size: int | None = Field(None, ge=0, description="File size in bytes.")
    chunk_count: int = Field(0, ge=0, description="Number of chunks created from this file.")
    uploaded_at: datetime | None = Field(None, description="When the file was uploaded.")
    ingested_at: datetime | None = Field(None, description="When ingestion completed.")
    expiration_date: datetime | None = Field(None, description="When the file will be auto-deleted.")
    error_message: str | None = Field(None, description="Error message if processing failed.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-specific metadata (e.g., page count, content types).",
    )


class FileProgress(BaseModel):
    """Progress tracking for individual files within an ingestion job."""

    file_id: str = Field(default="", description="Unique identifier for the file.")
    file_name: str = Field(..., description="Name of the file being processed.")
    status: FileStatus = Field(default=FileStatus.UPLOADING, description="Current processing status.")
    progress_percent: float = Field(0.0, ge=0.0, le=100.0, description="Processing progress (0-100).")
    error_message: str | None = Field(default=None, description="Error message if processing failed.")
    chunks_created: int = Field(0, ge=0, description="Number of chunks created from this file.")


class IngestionJobStatus(BaseModel):
    """
    Status model for async ingestion jobs.

    This follows a polling pattern where the frontend submits a job and then
    polls this status endpoint every 2 seconds until completion.
    """

    job_id: str = Field(..., description="Unique identifier for this ingestion job.")
    status: JobState = Field(default=JobState.PENDING, description="Overall job status.")
    submitted_at: datetime = Field(..., description="When the job was submitted.")
    started_at: datetime | None = Field(None, description="When processing started.")
    completed_at: datetime | None = Field(None, description="When processing completed.")
    total_files: int = Field(0, ge=0, description="Total number of files to process.")
    processed_files: int = Field(0, ge=0, description="Number of files processed so far.")
    file_details: list[FileProgress] = Field(
        default_factory=list,
        description="Detailed progress for each file.",
    )
    collection_name: str = Field(..., description="Target collection/index name.")
    backend: str = Field(..., description="Ingestion backend used.")
    error_message: str | None = Field(None, description="Error message if job failed.")

    # Extra metadata for backend-specific info (e.g., milvus_uri, results_count)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific metadata. Contains db info for retrieval.",
    )

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100.0

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state (completed or failed)."""
        return self.status in (JobState.COMPLETED, JobState.FAILED)

    @property
    def is_success(self) -> bool:
        """Check if job completed successfully (at least some files processed)."""
        return self.status == JobState.COMPLETED and self.processed_files > 0


class AvailableDocument(BaseModel):
    """Represents a user-uploaded document with optional summary.

    This model provides context about available documents to research agents
    so they can prioritize internal document searches.

    Attributes:
        file_name: The name of the uploaded file.
        summary: Optional one-sentence summary of the document content.
    """

    file_name: str
    summary: str | None = None
