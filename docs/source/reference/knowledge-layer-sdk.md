<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Knowledge Layer SDK Reference

> **Audience:** This document is for developers building custom Knowledge Layer backend adapters.
> For setup, configuration, and usage, refer to the [User Guide](../customization/knowledge-layer.md).

A pluggable abstraction for document ingestion and retrieval. This document provides everything needed to implement a custom storage backend adapter.

> **Integration Note:** This specification is designed to be implementation-agnostic. The base classes (`BaseRetriever`, `BaseIngestor`), schemas (`Chunk`, `RetrievalResult`, etc.), and factory functions (`register_retriever`, `register_ingestor`, `get_retriever`, `get_ingestor`) are provided by the host application. Import paths shown in examples (for example, `from knowledge_layer.factory import ...`) are placeholders--the actual paths will be provided during integration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Schemas](#data-schemas)
  - [Chunk (The Golden Record)](#chunk-the-golden-record)
  - [RetrievalResult](#retrievalresult)
  - [CollectionInfo](#collectioninfo)
  - [FileInfo](#fileinfo)
  - [FileProgress](#fileprogress)
  - [IngestionJobStatus](#ingestionjobstatus)
  - [Enums](#enums)
- [BaseRetriever Interface](#baseretriever-interface)
  - [Constructor](#retriever-constructor)
  - [retrieve()](#retrieve)
  - [normalize()](#normalize)
  - [backend_name](#retriever-backend_name)
  - [health_check()](#retriever-health_check)
- [BaseIngestor Interface](#baseingestor-interface)
  - [Constructor](#ingestor-constructor)
  - [Job Management](#job-management)
  - [Collection Management](#collection-management)
  - [File Management](#file-management)
  - [Optional Methods](#optional-methods)
- [Factory Registration](#factory-registration)
- [Document Summaries](#document-summaries)
- [Error Handling](#error-handling)
- [Complete Implementation Example](#complete-implementation-example)
- [Testing Your Implementation](#testing-your-implementation)
- [Checklist](#implementation-checklist)
- [HTTP API Integration](#http-api-integration)

---

## Overview

The Knowledge Layer provides a unified interface for document storage, ingestion, and semantic retrieval. It uses the **Adapter Pattern** to support multiple backends (vector databases, RAG services, etc.) through a common API.

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **Retriever** | Searches documents and returns ranked results |
| **Ingestor** | Handles document upload, chunking, embedding, and storage |
| **Collection** | A logical grouping of documents (like a database table) |
| **Chunk** | The atomic unit of retrieved content - the "Golden Record" |
| **Job** | Async ingestion task with status tracking |

**What You'll Implement:**

1. `BaseRetriever` - Query your vector store and normalize results to `Chunk` objects
2. `BaseIngestor` - Manage collections, files, and async ingestion jobs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
│                   (Agents, APIs, UIs, etc.)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Factory Layer                           │
│         get_retriever("your_backend") → YourRetriever           │
│         get_ingestor("your_backend")  → YourIngestor            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Abstract Base Classes                      │
│              BaseRetriever          BaseIngestor                │
│              ─────────────          ────────────                │
│              retrieve()             submit_job()                │
│              normalize()            get_job_status()            │
│              health_check()         create_collection()         │
│                                     upload_file()               │
│                                     list_files()                │
│                                     ...                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Your Backend Adapter                         │
│                                                                 │
│   @register_retriever("your_backend")                           │
│   class YourRetriever(BaseRetriever): ...                       │
│                                                                 │
│   @register_ingestor("your_backend")                            │
│   class YourIngestor(BaseIngestor): ...                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Your Storage Backend                          │
│        (Milvus, Pinecone, Weaviate, Qdrant, etc.)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Schemas

All adapters must convert their native formats to these universal schemas. The schemas are [Pydantic](https://docs.pydantic.dev/) models.

### Chunk (The Golden Record)

The atomic unit of retrieved content. **All adapters MUST output this exact schema.**

```python
from pydantic import BaseModel, Field
from typing import Any
from enum import StrEnum

class ContentType(StrEnum):
    """The Four Pillars - strict content categorization."""
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"

class Chunk(BaseModel):
    """
    The Atomic Unit of Knowledge.

    This schema unifies data from ANY backend so applications always
    see a consistent format.
    """

    # ─── Core Content (Required) ───────────────────────────────────

    chunk_id: str = Field(
        ...,
        description="Unique identifier for citation tracking. "
                    "Used to deduplicate and reference specific chunks."
    )

    content: str = Field(
        ...,
        description="The main text content. For visuals (tables, charts, images), "
                    "this MUST be the summary, caption, or textual representation. "
                    "NEVER None - use empty string if no content."
    )

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity/relevance score normalized to 0.0-1.0 range. "
                    "Higher is more relevant."
    )

    # ─── Citation Contract (Required) ──────────────────────────────

    file_name: str = Field(
        ...,
        description="Original filename (e.g., 'Q3_Report.pdf'). "
                    "Used for source attribution."
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based page number. None if not applicable "
                    "(e.g., for JSON, TXT, or non-paginated content)."
    )

    display_citation: str = Field(
        ...,
        description="User-facing citation label. Adapters MUST populate this. "
                    "Examples: 'report.pdf, p.15', 'data.json', 'image_001.png'"
    )

    # ─── Content Typing (Required) ─────────────────────────────────

    content_type: ContentType = Field(
        ...,
        description="The semantic category. MUST be one of: text, table, chart, image. "
                    "Frontend uses this for component switching."
    )

    content_subtype: str | None = Field(
        default=None,
        description="Granular subtype for specialized rendering. "
                    "Examples: 'bar_chart', 'pie_chart', 'markdown_table', 'photograph'"
    )

    # ─── Rich Payload (Optional) ───────────────────────────────────

    structured_data: str | None = Field(
        default=None,
        description="Raw structured data for programmatic access. "
                    "For tables: CSV or JSON. For charts: data series. "
                    "Enables Code Interpreter analysis."
    )

    # ─── Visual Assets (Optional) ──────────────────────────────────

    image_storage_uri: str | None = Field(
        default=None,
        description="Internal storage URI (S3, MinIO, etc.) for system access. "
                    "NOT for frontend display."
    )

    image_url: str | None = Field(
        default=None,
        description="Presigned HTTP URL for frontend display. "
                    "MUST be accessible via browser. Has expiration."
    )

    # ─── Metadata Passthrough ──────────────────────────────────────

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific metadata passthrough. "
                    "Include anything useful: embeddings, timestamps, etc."
    )
```

**Schema Rules (enforced by all adapters):**

| Rule | Description |
|------|-------------|
| **Four Pillars** | `content_type` MUST be exactly `text`, `table`, `chart`, or `image` |
| **Display Citation** | `display_citation` MUST be populated with human-readable string |
| **Visual Safety** | `content` MUST NEVER be None (use empty string for visuals) |
| **Data vs View** | Raw data in `structured_data`, renderable image in `image_url` |
| **Link Rot** | `image_url` MUST be presigned/accessible URL, not internal path |

---

### RetrievalResult

Container for search results returned by `retrieve()`.

```python
class RetrievalResult(BaseModel):
    """Container returned by retriever.retrieve()."""

    chunks: list[Chunk] = Field(
        default_factory=list,
        description="List of retrieved chunks, ordered by relevance (highest first)."
    )

    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Estimated token count for context window management. "
                    "Sum of tokens across all chunk contents."
    )

    query: str = Field(
        ...,
        description="The original query that produced these results. "
                    "Useful for logging and debugging."
    )

    backend: str = Field(
        ...,
        description="Backend name that produced these results "
                    "(e.g., 'llamaindex', 'pinecone')."
    )

    success: bool = Field(
        default=True,
        description="Whether retrieval succeeded. False indicates an error occurred."
    )

    error_message: str | None = Field(
        default=None,
        description="Error details if success=False. "
                    "Should be user-friendly, not stack traces."
    )
```

**Usage Pattern:**

```python
# Successful retrieval
return RetrievalResult(
    chunks=[chunk1, chunk2, chunk3],
    total_tokens=1500,
    query=query,
    backend=self.backend_name,
    success=True
)

# Failed retrieval (connection error, etc.)
return RetrievalResult(
    chunks=[],
    total_tokens=0,
    query=query,
    backend=self.backend_name,
    success=False,
    error_message="Cannot connect to vector database: connection refused"
)
```

---

### CollectionInfo

Metadata about a collection/index.

```python
from datetime import datetime

class CollectionInfo(BaseModel):
    """Metadata about a collection/index."""

    name: str = Field(
        ...,
        description="Unique collection identifier. "
                    "Convention: lowercase, underscores allowed."
    )

    description: str | None = Field(
        default=None,
        description="Human-readable description of the collection's purpose."
    )

    file_count: int = Field(
        default=0,
        ge=0,
        description="Number of source files in this collection."
    )

    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks/vectors stored."
    )

    created_at: datetime | None = Field(
        default=None,
        description="When the collection was created. UTC timezone."
    )

    updated_at: datetime | None = Field(
        default=None,
        description="Last modification time (file added/removed). "
                    "Used for TTL cleanup. UTC timezone."
    )

    backend: str = Field(
        ...,
        description="Backend that manages this collection."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific metadata. Examples: "
                    "embedding_model, vector_dimensions, index_type."
    )
```

---

### FileInfo

Metadata about a file within a collection.

```python
class FileStatus(StrEnum):
    """File processing lifecycle states."""
    UPLOADING = "uploading"   # File being uploaded to storage
    INGESTING = "ingesting"   # Chunking, embedding, indexing in progress
    SUCCESS = "success"       # Successfully processed
    FAILED = "failed"         # Processing failed

class FileInfo(BaseModel):
    """Metadata about a file within a collection."""

    file_id: str = Field(
        ...,
        description="Unique file identifier. Can be UUID or filename."
    )

    file_name: str = Field(
        ...,
        description="Original filename as uploaded."
    )

    collection_name: str = Field(
        ...,
        description="Collection this file belongs to."
    )

    status: FileStatus = Field(
        default=FileStatus.UPLOADING,
        description="Current processing status."
    )

    file_size: int | None = Field(
        default=None,
        ge=0,
        description="File size in bytes."
    )

    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of chunks created from this file."
    )

    uploaded_at: datetime | None = Field(
        default=None,
        description="When the file was uploaded."
    )

    ingested_at: datetime | None = Field(
        default=None,
        description="When ingestion completed (success or failure)."
    )

    expiration_date: datetime | None = Field(
        default=None,
        description="When the file will be auto-deleted (TTL)."
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if status=FAILED."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-specific metadata: page_count, content_types, etc."
    )
```

**State Transitions:**

```
UPLOADING ──► INGESTING ──► SUCCESS
                  │
                  └──────► FAILED
```

---

### FileProgress

Progress tracking for individual files within a batch job.

```python
class FileProgress(BaseModel):
    """Progress tracking for a single file in an ingestion job."""

    file_id: str = Field(
        default="",
        description="Unique identifier for the file."
    )

    file_name: str = Field(
        ...,
        description="Name of the file being processed."
    )

    status: FileStatus = Field(
        default=FileStatus.UPLOADING,
        description="Current processing status."
    )

    progress_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Processing progress 0-100. "
                    "Optional - set to 0 if not trackable."
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if status=FAILED. "
                    "IMPORTANT: UI displays this to users."
    )

    chunks_created: int = Field(
        default=0,
        ge=0,
        description="Number of chunks created from this file."
    )
```

---

### IngestionJobStatus

Status model for async ingestion jobs.

```python
class JobState(StrEnum):
    """Overall job lifecycle states."""
    PENDING = "pending"       # Job queued, not started
    PROCESSING = "processing" # At least one file being processed
    COMPLETED = "completed"   # All files processed (some may have failed)
    FAILED = "failed"         # Job-level failure (e.g., all files failed)

class IngestionJobStatus(BaseModel):
    """
    Status model for async ingestion jobs.

    Supports polling pattern: submit job → poll status → handle completion.
    """

    job_id: str = Field(
        ...,
        description="Unique job identifier returned by submit_job()."
    )

    status: JobState = Field(
        default=JobState.PENDING,
        description="Overall job status."
    )

    submitted_at: datetime = Field(
        ...,
        description="When the job was submitted."
    )

    started_at: datetime | None = Field(
        default=None,
        description="When processing actually started."
    )

    completed_at: datetime | None = Field(
        default=None,
        description="When processing completed (success or failure)."
    )

    total_files: int = Field(
        default=0,
        ge=0,
        description="Total number of files in this job."
    )

    processed_files: int = Field(
        default=0,
        ge=0,
        description="Number of files processed (success + failed)."
    )

    file_details: list[FileProgress] = Field(
        default_factory=list,
        description="Detailed progress for each file."
    )

    collection_name: str = Field(
        ...,
        description="Target collection for ingested documents."
    )

    backend: str = Field(
        ...,
        description="Ingestion backend used."
    )

    error_message: str | None = Field(
        default=None,
        description="Job-level error message if status=FAILED."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific metadata: task_ids, URIs, etc."
    )

    # ─── Computed Properties ───────────────────────────────────────

    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100.0

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (JobState.COMPLETED, JobState.FAILED)

    @property
    def is_success(self) -> bool:
        """Check if job completed with at least some successes."""
        return self.status == JobState.COMPLETED and self.processed_files > 0
```

**Job State Transitions:**

```
PENDING ──► PROCESSING ──► COMPLETED (all files done, at least one succeeded)
                 │
                 └──────► FAILED (all files failed OR job-level error)
```

---

### Enums

```python
from enum import StrEnum

class ContentType(StrEnum):
    """Content type categories - the 'Four Pillars'."""
    TEXT = "text"    # Regular text content
    TABLE = "table"  # Tabular data (CSV, HTML tables, etc.)
    CHART = "chart"  # Visualizations (bar charts, line graphs, etc.)
    IMAGE = "image"  # Photographs, diagrams, screenshots

class FileStatus(StrEnum):
    """File processing states."""
    UPLOADING = "uploading"
    INGESTING = "ingesting"
    SUCCESS = "success"
    FAILED = "failed"

class JobState(StrEnum):
    """Job processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

---

## BaseRetriever Interface

The retriever searches documents and returns ranked results.

### Retriever Constructor

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseRetriever(ABC):
    """Abstract base class for retrieval adapters."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the retriever with configuration.

        Args:
            config: Backend-specific configuration dictionary.
                    Common keys:
                    - endpoint: API endpoint URL
                    - api_key: Authentication key
                    - timeout: Request timeout in seconds
                    - persist_dir: Local storage path (for embedded DBs)

        Example:
            config = {
                "endpoint": "http://localhost:19530",
                "api_key": "your-api-key",  # pragma: allowlist secret
                "timeout": 30
            }
        """
        self.config = config or {}
```

---

### retrieve()

**The primary method** - searches documents and returns normalized results.

```python
@abstractmethod
async def retrieve(
    self,
    query: str,
    collection_name: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> RetrievalResult:
    """
    Retrieve documents matching the query.

    This is the core retrieval method. It must:
    1. Convert the query to embeddings (or use backend's built-in)
    2. Search the specified collection
    3. Normalize results to Chunk objects using normalize()
    4. Return a RetrievalResult container

    Args:
        query: Natural language search query.
               Example: "What are the key findings in Q3?"

        collection_name: Target collection/index name.
                        Example: "financial_reports"

        top_k: Maximum number of results to return.
               Default: 10. Typical range: 3-20.

        filters: Optional metadata filters. Format is backend-specific.
                 Common patterns:
                 - Dict: {"category": "finance", "year": 2024}
                 - String expression: "category == 'finance' AND year >= 2024"

    Returns:
        RetrievalResult containing:
        - chunks: List of Chunk objects, ordered by relevance (highest first)
        - total_tokens: Estimated token count for context management
        - query: Original query (for logging)
        - backend: Your backend name
        - success: True if retrieval succeeded
        - error_message: Error details if success=False

    Raises:
        This method should NOT raise exceptions. Instead, return a
        RetrievalResult with success=False and error_message set.

    Example Implementation:
        async def retrieve(self, query, collection_name, top_k=10, filters=None):
            try:
                # 1. Call your backend
                raw_results = await self.client.search(
                    collection=collection_name,
                    query_vector=self._embed(query),
                    limit=top_k,
                    filter=filters
                )

                # 2. Normalize results
                chunks = [self.normalize(r) for r in raw_results]

                # 3. Return success
                return RetrievalResult(
                    chunks=chunks,
                    total_tokens=sum(len(c.content.split()) for c in chunks),
                    query=query,
                    backend=self.backend_name,
                    success=True
                )

            except ConnectionError as e:
                return RetrievalResult(
                    chunks=[],
                    total_tokens=0,
                    query=query,
                    backend=self.backend_name,
                    success=False,
                    error_message=f"Connection failed: {e}"
                )
    """
```

---

### normalize()

Converts backend-specific results to the universal Chunk schema.

```python
@abstractmethod
def normalize(self, raw_result: Any) -> Chunk:
    """
    Convert a backend-specific result into a universal Chunk.

    This is the core of the Adapter Pattern. Each backend has different
    result formats - this method translates them to the common schema.

    Args:
        raw_result: The raw result object from your backend.
                   Type varies by backend (dict, object, tuple, etc.)

    Returns:
        A normalized Chunk object with all required fields populated.

    Required Field Mappings:
        Your backend field          →  Chunk field
        ─────────────────────────────────────────────
        unique id / hash            →  chunk_id
        text content                →  content
        similarity score (0-1)      →  score
        source filename             →  file_name
        page number (1-based)       →  page_number
        "filename, p.X" format      →  display_citation
        text/table/chart/image      →  content_type

    Example Implementation:
        def normalize(self, raw_result: dict, idx: int = 0) -> Chunk:
            # Extract fields from your backend's format
            doc_id = raw_result.get("id", f"chunk_{idx}")
            text = raw_result.get("text", "")
            score = raw_result.get("score", 0.0)

            # Normalize score to 0-1 range if needed
            if score > 1.0:
                score = score / 100.0  # Convert percentage

            # Extract metadata
            metadata = raw_result.get("metadata", {})
            file_name = metadata.get("source", "unknown")
            page = metadata.get("page_number")

            # Build citation
            if page and page > 0:
                citation = f"{file_name}, p.{page}"
            else:
                citation = file_name

            # Determine content type
            doc_type = metadata.get("type", "text").lower()
            if doc_type in ("table", "csv"):
                content_type = ContentType.TABLE
            elif doc_type in ("chart", "graph", "plot"):
                content_type = ContentType.CHART
            elif doc_type in ("image", "figure", "photo"):
                content_type = ContentType.IMAGE
            else:
                content_type = ContentType.TEXT

            return Chunk(
                chunk_id=doc_id,
                content=text,
                score=score,
                file_name=file_name,
                page_number=page,
                display_citation=citation,
                content_type=content_type,
                metadata=metadata
            )
    """
```

---

### Retriever backend_name

Property that returns the backend identifier.

```python
@property
@abstractmethod
def backend_name(self) -> str:
    """
    Return the name of this backend.

    This MUST match the name used in @register_retriever("name").
    Used for logging, metrics, and result attribution.

    Returns:
        Backend name string (e.g., "pinecone", "milvus", "qdrant")

    Example:
        @property
        def backend_name(self) -> str:
            return "pinecone"
    """
```

---

### Retriever health_check()

Optional health check method.

```python
async def health_check(self) -> bool:
    """
    Check if the backend is healthy and reachable.

    Override this to implement actual health checking.
    Default implementation returns True.

    Returns:
        True if healthy, False otherwise.

    Example:
        async def health_check(self) -> bool:
            try:
                response = await self.client.ping()
                return response.status == "ok"
            except Exception:
                return False
    """
    return True
```

---

## BaseIngestor Interface

The ingestor handles document upload, processing, and collection management.

### Ingestor Constructor

```python
class BaseIngestor(ABC):
    """Abstract base class for ingestion adapters."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the ingestor with configuration.

        Args:
            config: Backend-specific configuration dictionary.
                    Common keys:
                    - endpoint: API endpoint URL
                    - api_key: Authentication key
                    - timeout: Request timeout in seconds
                    - chunk_size: Default chunk size for text splitting
                    - chunk_overlap: Overlap between chunks
                    - persist_dir: Local storage path

        Implementation Notes:
            - Initialize your backend client here
            - Set up job tracking data structures
            - Start background tasks if needed (e.g., TTL cleanup)

        Example:
            def __init__(self, config):
                super().__init__(config)
                self.endpoint = self.config.get("endpoint", "http://localhost:8000")
                self.chunk_size = self.config.get("chunk_size", 512)
                self._jobs: dict[str, IngestionJobStatus] = {}  # Job tracking
        """
        self.config = config or {}
```

---

### Job Management

#### submit_job()

```python
@abstractmethod
def submit_job(
    self,
    file_paths: list[str],
    collection_name: str,
    config: dict[str, Any] | None = None,
) -> str:
    """
    Submit files for ingestion (non-blocking).

    This method MUST return immediately with a job ID. Actual processing
    happens asynchronously (background thread, queue, external service).

    Args:
        file_paths: List of local file paths to ingest.
                   Supported formats typically: PDF, TXT, DOCX, MD, JSON, CSV
                   Example: ["/tmp/report.pdf", "/tmp/data.csv"]

        collection_name: Target collection. Must exist (call create_collection first).

        config: Optional ingestion configuration:
                - chunk_size: Override default chunk size
                - chunk_overlap: Override default overlap
                - cleanup_files: Delete temp files after processing (default: False)
                - original_filenames: Original names for temp files
                - extract_tables: Enable table extraction (PDF)
                - extract_images: Enable image extraction (PDF)

    Returns:
        job_id (str): Unique identifier for status polling.
                     Typically a UUID.

    Implementation Pattern:
        def submit_job(self, file_paths, collection_name, config=None):
            job_id = str(uuid.uuid4())
            config = config or {}

            # Create initial job status
            file_details = [
                FileProgress(
                    file_id=str(uuid.uuid4()),
                    file_name=Path(fp).name,
                    status=FileStatus.UPLOADING
                )
                for fp in file_paths
            ]

            job = IngestionJobStatus(
                job_id=job_id,
                status=JobState.PENDING,
                submitted_at=datetime.utcnow(),
                total_files=len(file_paths),
                collection_name=collection_name,
                backend=self.backend_name,
                file_details=file_details
            )
            self._jobs[job_id] = job

            # Start async processing
            thread = threading.Thread(
                target=self._process_files,
                args=(job_id, file_paths, collection_name, config),
                daemon=True
            )
            thread.start()

            return job_id
    """
```

#### get_job_status()

```python
@abstractmethod
def get_job_status(self, job_id: str) -> IngestionJobStatus:
    """
    Get current status of an ingestion job.

    Called by polling loop to check progress. Must return current state
    even if job is complete or failed.

    Args:
        job_id: The job ID returned from submit_job().

    Returns:
        IngestionJobStatus with current state. If job_id not found,
        return a FAILED status with appropriate error message.

    Implementation Notes:
        1. Look up job in your tracking store
        2. If using external service, poll for updates
        3. Update file_details with per-file progress
        4. Transition job state based on file states:
           - All files done (success or failed) → COMPLETED or FAILED
           - At least one file processing → PROCESSING
           - No files started → PENDING

    Error Message Handling (IMPORTANT for UI):
        The UI displays FileProgress.error_message to users. Your
        implementation MUST extract and set meaningful error messages:

        if backend_response.get("state") == "failed":
            error_msg = (
                backend_response.get("error")
                or backend_response.get("message")
                or backend_response.get("result", {}).get("error")
                or "Unknown error"
            )
            file_progress.status = FileStatus.FAILED
            file_progress.error_message = error_msg

    Example:
        def get_job_status(self, job_id):
            if job_id not in self._jobs:
                return IngestionJobStatus(
                    job_id=job_id,
                    status=JobState.FAILED,
                    submitted_at=datetime.utcnow(),
                    collection_name="unknown",
                    backend=self.backend_name,
                    error_message=f"Job {job_id} not found"
                )
            return self._jobs[job_id]
    """
```

---

### Collection Management

#### create_collection()

```python
@abstractmethod
def create_collection(
    self,
    name: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CollectionInfo:
    """
    Create a new collection/index.

    Args:
        name: Unique collection name.
              Convention: lowercase, underscores, no spaces.
              Example: "financial_reports_2024"

        description: Human-readable description.
                    Example: "Q1-Q4 financial reports and earnings calls"

        metadata: Backend-specific configuration:
                 - embedding_dimension: Vector dimensions (default varies)
                 - distance_metric: "cosine", "euclidean", "dot"
                 - index_type: Backend-specific index configuration

    Returns:
        CollectionInfo with created collection details.

    Raises:
        May raise exception if collection already exists (backend-dependent).
        Some implementations return existing collection instead.

    Example:
        def create_collection(self, name, description=None, metadata=None):
            metadata = metadata or {}

            # Create in backend
            self.client.create_collection(
                name=name,
                dimension=metadata.get("embedding_dimension", 1536),
                metric=metadata.get("distance_metric", "cosine")
            )

            return CollectionInfo(
                name=name,
                description=description,
                file_count=0,
                chunk_count=0,
                created_at=datetime.utcnow(),
                backend=self.backend_name,
                metadata=metadata
            )
    """
```

#### delete_collection()

```python
@abstractmethod
def delete_collection(self, name: str) -> bool:
    """
    Delete a collection and all its contents.

    This is a destructive operation - all documents and vectors are removed.

    Args:
        name: Collection name to delete.

    Returns:
        True if deleted successfully, False otherwise.
        Returns False if collection doesn't exist (not an error).

    Example:
        def delete_collection(self, name):
            try:
                self.client.drop_collection(name)
                return True
            except CollectionNotFoundError:
                return False
            except Exception as e:
                logger.error(f"Failed to delete {name}: {e}")
                return False
    """
```

#### list_collections()

```python
@abstractmethod
def list_collections(self) -> list[CollectionInfo]:
    """
    List all available collections.

    Returns:
        List of CollectionInfo objects. Empty list if none exist.

    Implementation Notes:
        - Include file_count and chunk_count if available
        - Include timestamps for TTL cleanup support
        - Handle pagination if backend has many collections

    Example:
        def list_collections(self):
            collections = []
            for col in self.client.list_collections():
                stats = self.client.get_collection_stats(col.name)
                collections.append(CollectionInfo(
                    name=col.name,
                    description=col.description,
                    file_count=stats.get("file_count", 0),
                    chunk_count=stats.get("vector_count", 0),
                    created_at=col.created_at,
                    updated_at=col.updated_at,
                    backend=self.backend_name
                ))
            return collections
    """
```

#### get_collection()

```python
@abstractmethod
def get_collection(self, name: str) -> CollectionInfo | None:
    """
    Get metadata for a specific collection.

    Args:
        name: Collection name.

    Returns:
        CollectionInfo if found, None otherwise.

    Example:
        def get_collection(self, name):
            try:
                col = self.client.get_collection(name)
                return CollectionInfo(
                    name=col.name,
                    description=col.description,
                    chunk_count=col.count(),
                    backend=self.backend_name
                )
            except CollectionNotFoundError:
                return None
    """
```

---

### File Management

#### upload_file()

```python
@abstractmethod
def upload_file(
    self,
    file_path: str,
    collection_name: str,
    metadata: dict[str, Any] | None = None,
) -> FileInfo:
    """
    Upload a single file to a collection.

    This is a convenience method that wraps submit_job for single files.
    Returns immediately - actual ingestion is async.

    Args:
        file_path: Local path to the file.
        collection_name: Target collection (must exist).
        metadata: Optional file metadata (passed to submit_job config).

    Returns:
        FileInfo with initial upload status.

    Example:
        def upload_file(self, file_path, collection_name, metadata=None):
            job_id = self.submit_job([file_path], collection_name, config=metadata)
            job_status = self.get_job_status(job_id)

            if job_status.file_details:
                return FileInfo(
                    file_id=job_status.file_details[0].file_id,
                    file_name=Path(file_path).name,
                    collection_name=collection_name,
                    status=FileStatus.INGESTING,
                    metadata={"job_id": job_id}
                )
    """
```

#### delete_file()

```python
@abstractmethod
def delete_file(self, file_id: str, collection_name: str) -> bool:
    """
    Delete a file and its chunks from a collection.

    Args:
        file_id: File identifier. Can be:
                - The file_id from FileInfo
                - The original filename
                - Backend-specific document ID

        collection_name: Collection containing the file.

    Returns:
        True if deleted successfully, False otherwise.

    Implementation Notes:
        - Delete all chunks/vectors with matching file_id or file_name
        - Handle temp file naming patterns (e.g., "tmp12345678_original.pdf")
        - Return False if file not found (not an error)

    Example:
        def delete_file(self, file_id, collection_name):
            try:
                # Delete by metadata filter
                self.client.delete(
                    collection=collection_name,
                    filter={"file_name": file_id}
                )
                return True
            except Exception as e:
                logger.error(f"Failed to delete {file_id}: {e}")
                return False
    """
```

#### delete_files()

```python
def delete_files(
    self,
    file_ids: list[str],
    collection_name: str,
) -> dict[str, Any]:
    """
    Delete multiple files from a collection (batch delete).

    Default implementation calls delete_file() for each. Override for
    optimized batch deletion.

    Args:
        file_ids: List of file IDs to delete.
        collection_name: Collection containing the files.

    Returns:
        Dict with:
        - successful: list[str] - IDs of deleted files
        - failed: list[dict] - [{file_id, error}, ...] for failures
        - total_deleted: int - count of successful deletions
        - message: str - summary message

    Example Response:
        {
            "successful": ["file1.pdf", "file2.pdf"],
            "failed": [{"file_id": "file3.pdf", "error": "Not found"}],
            "total_deleted": 2,
            "message": "Deleted 2 of 3 files"
        }
    """
    # Default implementation - override for batch optimization
    successful = []
    failed = []

    for file_id in file_ids:
        try:
            if self.delete_file(file_id, collection_name):
                successful.append(file_id)
            else:
                failed.append({"file_id": file_id, "error": "Not found or already deleted"})
        except Exception as e:
            failed.append({"file_id": file_id, "error": str(e)})

    return {
        "message": f"Deleted {len(successful)} of {len(file_ids)} files",
        "successful": successful,
        "failed": failed,
        "total_deleted": len(successful),
    }
```

#### list_files()

```python
@abstractmethod
def list_files(self, collection_name: str) -> list[FileInfo]:
    """
    List all files in a collection.

    Args:
        collection_name: Collection to list files from.

    Returns:
        List of FileInfo objects. Empty list if collection empty.

    Implementation Notes:
        - Group chunks by file_name metadata to count files
        - Calculate chunk_count per file
        - Include status (SUCCESS for listed files - they're ingested)

    Example:
        def list_files(self, collection_name):
            # Get unique files from chunk metadata
            results = self.client.query(
                collection=collection_name,
                output_fields=["file_name"],
                limit=10000
            )

            # Group by file_name
            file_counts = Counter(r["file_name"] for r in results)

            return [
                FileInfo(
                    file_id=name,
                    file_name=name,
                    collection_name=collection_name,
                    status=FileStatus.SUCCESS,
                    chunk_count=count
                )
                for name, count in file_counts.items()
            ]
    """
```

#### get_file_status()

```python
@abstractmethod
def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
    """
    Get current status of a file.

    Args:
        file_id: File identifier (from FileInfo.file_id or original filename).
        collection_name: Collection containing the file.

    Returns:
        FileInfo if found, None otherwise.

    Implementation Notes:
        - Check job tracking first (for in-progress files)
        - Fall back to listing files (for completed files)
        - Update status based on job progress if still processing
    """
```

---

### Optional Methods

These have default implementations but can be overridden.

#### health_check()

```python
async def health_check(self) -> bool:
    """
    Check if the backend is healthy and reachable.

    Returns:
        True if healthy, False otherwise.

    Example:
        async def health_check(self):
            try:
                # Ping your backend
                response = await self.client.health()
                return response.get("status") == "healthy"
            except Exception:
                return False
    """
    return True
```

#### select_sources() / get_selected_sources()

```python
def select_sources(self, source_names: list[str]) -> bool:
    """
    Select which collections to use for multi-collection queries.

    Optional - only implement if your backend supports querying
    across multiple collections simultaneously.

    Args:
        source_names: List of collection names to activate.

    Returns:
        True if selection succeeded.

    Raises:
        NotImplementedError if not supported.
    """
    raise NotImplementedError(
        f"{self.backend_name} does not support source selection"
    )

def get_selected_sources(self) -> list[str]:
    """Return currently selected collections."""
    raise NotImplementedError(
        f"{self.backend_name} does not support source selection"
    )
```

#### generate_summary()

```python
def generate_summary(self, text_content: str, file_name: str) -> str | None:
    """
    Generate a short summary of the document content.

    Override in adapters to enable summarization. Default returns None.

    Args:
        text_content: Combined text from first and last chunks.
        file_name: Original filename for context.

    Returns:
        One-sentence summary or None if not implemented.

    Example:
        def generate_summary(self, text_content, file_name):
            prompt = f"Summarize this document ({file_name}) in one sentence: {text_content[:1000]}"
            response = self.llm.invoke(prompt)
            return response.content
    """
    return None
```

---

## Factory Registration

Backends register themselves using decorators when their module is imported. The host application provides the factory and registration decorators.

```python
# NOTE: The factory module is provided by the host application.
# Your adapter imports from wherever the host application specifies.
# Common pattern: from knowledge_layer.factory import register_retriever, register_ingestor

@register_retriever("your_backend")  # ← Registration name
class YourRetriever(BaseRetriever):
    ...

@register_ingestor("your_backend")   # ← Same name
class YourIngestor(BaseIngestor):
    ...
```

**Important:** The adapter module must be imported for registration to occur.

```python
# In __init__.py - import triggers registration
from .adapter import YourRetriever, YourIngestor

__all__ = ["YourRetriever", "YourIngestor"]
```

**Usage:**

```python
# 1. Import to register
from your_backend import YourRetriever, YourIngestor

# 2. Get via factory (import path provided by host application)
from knowledge_layer.factory import get_retriever, get_ingestor

retriever = get_retriever("your_backend", config={...})
ingestor = get_ingestor("your_backend", config={...})
```

---

## Document Summaries

The Knowledge Layer includes a backend-agnostic summary system that generates one-sentence document summaries during ingestion. Summaries are stored in a persistent database and exposed to agents through their system prompts.

### Summary Flow

```
Ingestion -> Text extraction -> LLM summary call -> register_summary() -> Summary DB
                                                                              |
Agent startup -> get_available_documents_async(collection) -> System prompt injection
```

### Summary API

The summary system uses three functions in `deep_researcher_agent.knowledge.factory`:

| Function | Purpose |
|----------|---------|
| `register_summary(collection, filename, summary)` | Store a summary after ingestion |
| `unregister_summary(collection, filename)` | Remove summary on file deletion |
| `get_available_documents_async(collection)` | Retrieve summaries for agent prompt injection |

### Implementing Summaries in Your Adapter

Call `register_summary()` after successful ingestion:

```python
from deep_researcher_agent.knowledge import register_summary

# In your ingestion worker, after processing a file:
if self.generate_summary_enabled and summary_text:
    register_summary(collection_name, file_name, summary_text)
```

Call `unregister_summary()` when deleting files:

```python
from deep_researcher_agent.knowledge import unregister_summary

def delete_file(self, file_id, collection_name):
    # ... delete chunks from your backend ...
    unregister_summary(collection_name, file_name)
    return True
```

### Summary Storage

Summaries are persisted in a database configured through `summary_db`:

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    generate_summary: true
    summary_model: summary_llm
    summary_db: ${DEEP_RESEARCHER_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

The following drivers are supported:

| Driver | Use Case | Example |
|--------|----------|---------|
| SQLite | Local development | `sqlite+aiosqlite:///./summaries.db` |
| PostgreSQL | Production | `postgresql+psycopg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}` |

The summary store uses SQLAlchemy (`summary_store.py`) and can share the same PostgreSQL instance as the jobs database.

### How Agents Consume Summaries

When documents have summaries, agents see them in their system prompt through the `available_documents` Jinja2 variable:

```
## Uploaded Documents

The user has uploaded the following documents to the knowledge base:

- **report.pdf**: Q3 financial results showing 15% revenue growth.
- **roadmap.pptx**: 2024 product development timeline including AI features.

When the query relates to these documents, prioritize searching them before using external tools.
```

Both LlamaIndex and Foundational RAG adapters call `register_summary()` and `unregister_summary()`, ensuring consistent behavior regardless of backend choice.

---

## Error Handling

### Error Response Patterns

**Never raise exceptions from retrieve()**. Return error state instead:

```python
async def retrieve(self, query, collection_name, top_k=10, filters=None):
    try:
        # ... retrieval logic ...
        return RetrievalResult(chunks=chunks, success=True, ...)

    except ConnectionError as e:
        return RetrievalResult(
            chunks=[],
            total_tokens=0,
            query=query,
            backend=self.backend_name,
            success=False,
            error_message=f"Cannot connect to backend: {str(e)[:100]}"
        )

    except TimeoutError:
        return RetrievalResult(
            chunks=[],
            total_tokens=0,
            query=query,
            backend=self.backend_name,
            success=False,
            error_message=f"Request timed out after {self.timeout}s"
        )

    except Exception as e:
        logger.exception("Unexpected retrieval error")
        return RetrievalResult(
            chunks=[],
            total_tokens=0,
            query=query,
            backend=self.backend_name,
            success=False,
            error_message=f"Retrieval failed: {str(e)[:100]}"
        )
```

### Common Error Scenarios

| Scenario | How to Handle |
|----------|---------------|
| Collection not found | Return empty RetrievalResult with `success=False` |
| Connection refused | Return error with clear message about connectivity |
| Authentication failed | Return error mentioning API key/credentials |
| Timeout | Return error with timeout duration |
| Rate limited | Return error suggesting retry |
| Invalid query | Return error describing the issue |

### Job Error Handling

```python
def get_job_status(self, job_id):
    job = self._jobs.get(job_id)
    if not job:
        return IngestionJobStatus(
            job_id=job_id,
            status=JobState.FAILED,
            submitted_at=datetime.utcnow(),
            collection_name="unknown",
            backend=self.backend_name,
            error_message=f"Job {job_id} not found"
        )

    # Update file statuses from backend
    for file_progress in job.file_details:
        if file_progress.status == FileStatus.INGESTING:
            backend_status = self._check_backend_status(file_progress.file_id)

            if backend_status["state"] == "success":
                file_progress.status = FileStatus.SUCCESS
                file_progress.chunks_created = backend_status.get("chunks", 0)

            elif backend_status["state"] == "failed":
                file_progress.status = FileStatus.FAILED
                # CRITICAL: Extract and set error message for UI
                file_progress.error_message = (
                    backend_status.get("error")
                    or backend_status.get("message")
                    or "Ingestion failed"
                )

    # Update job state
    success_count = sum(1 for f in job.file_details if f.status == FileStatus.SUCCESS)
    failed_count = sum(1 for f in job.file_details if f.status == FileStatus.FAILED)

    if success_count + failed_count == job.total_files:
        job.status = JobState.FAILED if failed_count == job.total_files else JobState.COMPLETED
        job.completed_at = datetime.utcnow()

    return job
```

---

## Complete Implementation Example

Here's a complete, minimal implementation you can use as a template:

```python
"""
Example Knowledge Layer Backend Adapter

This is a complete, working example that you can use as a template.
Replace the TODO sections with your backend-specific logic.

INTEGRATION NOTE:
-----------------
The base classes, schemas, and factory decorators are provided by the host
application. When integrating, you'll receive the import paths to use.
The imports below use placeholder paths - replace with actual paths provided.
"""

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Base classes - provided by the host application
# Replace these imports with the actual paths provided during integration
from knowledge_layer.base import BaseIngestor, BaseRetriever
from knowledge_layer.factory import register_ingestor, register_retriever
from knowledge_layer.schema import (
    Chunk,
    CollectionInfo,
    ContentType,
    FileInfo,
    FileProgress,
    FileStatus,
    IngestionJobStatus,
    JobState,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_TIMEOUT = 30


# ============================================================================
# Retriever Implementation
# ============================================================================

@register_retriever("example_backend")
class ExampleRetriever(BaseRetriever):
    """
    Example retriever implementation.

    Replace TODO sections with your backend client calls.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Configuration
        self.endpoint = self.config.get("endpoint", DEFAULT_ENDPOINT)
        self.api_key = self.config.get("api_key", "")
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)

        # TODO: Initialize your backend client
        # self.client = YourVectorDBClient(
        #     endpoint=self.endpoint,
        #     api_key=self.api_key
        # )

        logger.info(f"ExampleRetriever initialized: endpoint={self.endpoint}")

    @property
    def backend_name(self) -> str:
        return "example_backend"

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Search documents and return normalized results."""
        try:
            logger.info(f"Retrieving: query='{query[:50]}...', collection={collection_name}")

            # TODO: Replace with your backend search
            # raw_results = await self.client.search(
            #     collection=collection_name,
            #     query=query,  # or query_vector if you embed separately
            #     limit=top_k,
            #     filter=filters
            # )

            # Example mock results for testing
            raw_results = [
                {
                    "id": "chunk_001",
                    "text": f"This is a sample result for query: {query}",
                    "score": 0.95,
                    "metadata": {
                        "file_name": "sample.pdf",
                        "page_number": 1,
                        "type": "text"
                    }
                }
            ]

            # Normalize results to Chunk objects
            chunks = [self.normalize(r, idx) for idx, r in enumerate(raw_results)]

            return RetrievalResult(
                chunks=chunks,
                total_tokens=sum(len(c.content.split()) for c in chunks),
                query=query,
                backend=self.backend_name,
                success=True,
            )

        except ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            return RetrievalResult(
                chunks=[],
                total_tokens=0,
                query=query,
                backend=self.backend_name,
                success=False,
                error_message=f"Cannot connect to backend: {str(e)[:100]}",
            )

        except Exception as e:
            logger.exception("Retrieval failed")
            return RetrievalResult(
                chunks=[],
                total_tokens=0,
                query=query,
                backend=self.backend_name,
                success=False,
                error_message=f"Retrieval error: {str(e)[:100]}",
            )

    def normalize(self, raw_result: dict[str, Any], idx: int = 0) -> Chunk:
        """Convert backend result to universal Chunk schema."""
        metadata = raw_result.get("metadata", {})

        # Extract fields
        chunk_id = raw_result.get("id", f"chunk_{idx}")
        content = raw_result.get("text", "")
        score = float(raw_result.get("score", 0.0))

        # Normalize score to 0-1 if needed
        if score > 1.0:
            score = min(score / 100.0, 1.0)

        file_name = metadata.get("file_name", "unknown")
        page_number = metadata.get("page_number")

        # Build citation
        if page_number and page_number > 0:
            display_citation = f"{file_name}, p.{page_number}"
        else:
            display_citation = file_name

        # Determine content type
        type_str = metadata.get("type", "text").lower()
        if type_str in ("table", "csv"):
            content_type = ContentType.TABLE
        elif type_str in ("chart", "graph"):
            content_type = ContentType.CHART
        elif type_str in ("image", "figure"):
            content_type = ContentType.IMAGE
        else:
            content_type = ContentType.TEXT

        return Chunk(
            chunk_id=chunk_id,
            content=content,
            score=score,
            file_name=file_name,
            page_number=page_number,
            display_citation=display_citation,
            content_type=content_type,
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        """Check backend connectivity."""
        try:
            # TODO: Replace with your health check
            # return await self.client.ping()
            return True
        except Exception:
            return False


# ============================================================================
# Ingestor Implementation
# ============================================================================

@register_ingestor("example_backend")
class ExampleIngestor(BaseIngestor):
    """
    Example ingestor implementation.

    Replace TODO sections with your backend client calls.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Configuration
        self.endpoint = self.config.get("endpoint", DEFAULT_ENDPOINT)
        self.api_key = self.config.get("api_key", "")
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.chunk_size = self.config.get("chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)

        # Job tracking (in-memory)
        self._jobs: dict[str, IngestionJobStatus] = {}
        self._collections: dict[str, CollectionInfo] = {}

        # TODO: Initialize your backend client
        # self.client = YourVectorDBClient(...)

        logger.info(f"ExampleIngestor initialized: endpoint={self.endpoint}")

    @property
    def backend_name(self) -> str:
        return "example_backend"

    # ─── Job Management ────────────────────────────────────────────

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Submit files for async ingestion."""
        job_id = str(uuid.uuid4())
        config = config or {}
        original_filenames = config.get("original_filenames", [])

        # Create file progress entries
        file_details = []
        for i, fp in enumerate(file_paths):
            file_name = original_filenames[i] if i < len(original_filenames) else Path(fp).name
            file_details.append(
                FileProgress(
                    file_id=str(uuid.uuid4()),
                    file_name=file_name,
                    status=FileStatus.UPLOADING,
                )
            )

        # Create job status
        job = IngestionJobStatus(
            job_id=job_id,
            status=JobState.PENDING,
            submitted_at=datetime.utcnow(),
            total_files=len(file_paths),
            processed_files=0,
            file_details=file_details,
            collection_name=collection_name,
            backend=self.backend_name,
        )
        self._jobs[job_id] = job

        # Start async processing
        thread = threading.Thread(
            target=self._process_files,
            args=(job_id, file_paths, collection_name, config),
            daemon=True,
        )
        thread.start()

        logger.info(f"Submitted job {job_id} with {len(file_paths)} files")
        return job_id

    def _process_files(
        self,
        job_id: str,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any],
    ):
        """Background thread for file processing."""
        job = self._jobs[job_id]
        job.status = JobState.PROCESSING
        job.started_at = datetime.utcnow()

        original_filenames = config.get("original_filenames", [])

        for i, file_path in enumerate(file_paths):
            file_detail = job.file_details[i]
            file_name = original_filenames[i] if i < len(original_filenames) else Path(file_path).name

            try:
                file_detail.status = FileStatus.INGESTING

                # TODO: Replace with your ingestion logic
                # 1. Read file
                # 2. Chunk text
                # 3. Generate embeddings
                # 4. Store in your backend
                #
                # chunks = self._chunk_file(file_path)
                # embeddings = self._embed_chunks(chunks)
                # self.client.insert(collection_name, chunks, embeddings)

                # Simulate processing
                import time
                time.sleep(1)

                file_detail.status = FileStatus.SUCCESS
                file_detail.chunks_created = 10  # Example chunk count
                logger.info(f"Processed {file_name}")

            except Exception as e:
                file_detail.status = FileStatus.FAILED
                file_detail.error_message = str(e)
                logger.error(f"Failed to process {file_name}: {e}")

            job.processed_files = i + 1

        # Update job final status
        failed_count = sum(1 for f in job.file_details if f.status == FileStatus.FAILED)
        job.status = JobState.FAILED if failed_count == job.total_files else JobState.COMPLETED
        job.completed_at = datetime.utcnow()

        logger.info(f"Job {job_id} completed: {job.status}")

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        """Get current job status."""
        if job_id not in self._jobs:
            return IngestionJobStatus(
                job_id=job_id,
                status=JobState.FAILED,
                submitted_at=datetime.utcnow(),
                collection_name="unknown",
                backend=self.backend_name,
                error_message=f"Job {job_id} not found",
            )
        return self._jobs[job_id]

    # ─── Collection Management ─────────────────────────────────────

    def create_collection(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        """Create a new collection."""
        metadata = metadata or {}

        # TODO: Create collection in your backend
        # self.client.create_collection(
        #     name=name,
        #     dimension=metadata.get("embedding_dimension", 1536)
        # )

        info = CollectionInfo(
            name=name,
            description=description,
            file_count=0,
            chunk_count=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            backend=self.backend_name,
            metadata=metadata,
        )
        self._collections[name] = info

        logger.info(f"Created collection: {name}")
        return info

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name not in self._collections:
            return False

        # TODO: Delete from your backend
        # self.client.drop_collection(name)

        del self._collections[name]
        logger.info(f"Deleted collection: {name}")
        return True

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections."""
        # TODO: Fetch from your backend
        # return [self._to_collection_info(c) for c in self.client.list_collections()]
        return list(self._collections.values())

    def get_collection(self, name: str) -> CollectionInfo | None:
        """Get collection metadata."""
        return self._collections.get(name)

    # ─── File Management ───────────────────────────────────────────

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        """Upload a single file."""
        job_id = self.submit_job([file_path], collection_name, config=metadata)
        job = self.get_job_status(job_id)

        if job.file_details:
            fd = job.file_details[0]
            return FileInfo(
                file_id=fd.file_id,
                file_name=fd.file_name,
                collection_name=collection_name,
                status=fd.status,
                metadata={"job_id": job_id},
            )

        return FileInfo(
            file_id=job_id,
            file_name=Path(file_path).name,
            collection_name=collection_name,
            status=FileStatus.FAILED,
            error_message="Failed to create job",
        )

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        """Delete a file's chunks."""
        # TODO: Delete from your backend
        # self.client.delete(
        #     collection=collection_name,
        #     filter={"file_name": file_id}
        # )
        logger.info(f"Deleted file {file_id} from {collection_name}")
        return True

    def list_files(self, collection_name: str) -> list[FileInfo]:
        """List files in a collection."""
        # TODO: Query your backend for unique file_names
        # results = self.client.query(collection_name, output_fields=["file_name"])
        # unique_files = set(r["file_name"] for r in results)
        return []

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        """Get file status."""
        # Check jobs first
        for job in self._jobs.values():
            for fd in job.file_details:
                if fd.file_id == file_id or fd.file_name == file_id:
                    return FileInfo(
                        file_id=fd.file_id,
                        file_name=fd.file_name,
                        collection_name=collection_name,
                        status=fd.status,
                        chunk_count=fd.chunks_created,
                        error_message=fd.error_message,
                    )

        # Check collection files
        files = self.list_files(collection_name)
        for f in files:
            if f.file_id == file_id or f.file_name == file_id:
                return f

        return None

    async def health_check(self) -> bool:
        """Check backend connectivity."""
        try:
            # TODO: Replace with your health check
            return True
        except Exception:
            return False
```

---

## Testing Your Implementation

### Basic Functionality Test

```python
import asyncio
import time

async def test_backend():
    # Import your adapter (triggers registration)
    from your_backend import YourRetriever, YourIngestor
    # Factory import path provided by host application
    from knowledge_layer.factory import get_retriever, get_ingestor

    # Get instances
    config = {"endpoint": "http://localhost:8000"}
    ingestor = get_ingestor("your_backend", config)
    retriever = get_retriever("your_backend", config)

    # Test health
    print("Testing health check...")
    assert await ingestor.health_check(), "Ingestor health check failed"
    assert await retriever.health_check(), "Retriever health check failed"
    print("Health checks passed")

    # Test collection management
    print("Testing collection management...")
    collection = ingestor.create_collection("test_collection", "Test collection")
    assert collection.name == "test_collection"

    collections = ingestor.list_collections()
    assert any(c.name == "test_collection" for c in collections)
    print("Collection management passed")

    # Test ingestion
    print("Testing ingestion...")
    job_id = ingestor.submit_job(["/path/to/test.pdf"], "test_collection")

    # Poll for completion
    for _ in range(30):
        status = ingestor.get_job_status(job_id)
        print(f"  Job status: {status.status}, progress: {status.progress_percent:.0f}%")
        if status.is_terminal:
            break
        time.sleep(1)

    assert status.status == "completed", f"Job failed: {status.error_message}"
    print("Ingestion passed")

    # Test retrieval
    print("Testing retrieval...")
    result = await retriever.retrieve("test query", "test_collection", top_k=5)
    assert result.success, f"Retrieval failed: {result.error_message}"
    print(f"Retrieval passed ({len(result.chunks)} chunks)")

    # Cleanup
    ingestor.delete_collection("test_collection")
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_backend())
```

### Schema Validation Test

```python
# Schema imports - path provided by host application
from knowledge_layer.schema import Chunk, ContentType

def test_chunk_normalization():
    """Test that your normalize() produces valid Chunks."""
    retriever = YourRetriever(config={})

    # Test with various backend result formats
    test_cases = [
        {"id": "1", "text": "Hello", "score": 0.9, "metadata": {"file_name": "test.pdf"}},
        {"id": "2", "text": "", "score": 0.5, "metadata": {"file_name": "image.png", "type": "image"}},
    ]

    for raw in test_cases:
        chunk = retriever.normalize(raw)

        # Validate required fields
        assert chunk.chunk_id, "chunk_id is required"
        assert chunk.content is not None, "content must not be None"
        assert 0 <= chunk.score <= 1, "score must be 0-1"
        assert chunk.file_name, "file_name is required"
        assert chunk.display_citation, "display_citation is required"
        assert chunk.content_type in ContentType, "content_type must be valid"

        print(f"Chunk valid: {chunk.chunk_id}")
```

---

## Implementation Checklist

Use this checklist to ensure your implementation is complete:

### BaseRetriever

- [ ] `__init__()` - Initialize backend client with config
- [ ] `backend_name` property - Returns registration name
- [ ] `retrieve()` - Searches and returns `RetrievalResult`
- [ ] `normalize()` - Converts backend results to `Chunk`
- [ ] `health_check()` - Optional, returns True if healthy

### BaseIngestor

- [ ] `__init__()` - Initialize backend client and job tracking
- [ ] `backend_name` property - Returns registration name

**Job Management:**
- [ ] `submit_job()` - Submits async job, returns job_id
- [ ] `get_job_status()` - Returns `IngestionJobStatus`

**Collection Management:**
- [ ] `create_collection()` - Creates collection, returns `CollectionInfo`
- [ ] `delete_collection()` - Deletes collection, returns bool
- [ ] `list_collections()` - Returns list of `CollectionInfo`
- [ ] `get_collection()` - Returns `CollectionInfo` or None

**File Management:**
- [ ] `upload_file()` - Uploads single file, returns `FileInfo`
- [ ] `delete_file()` - Deletes file, returns bool
- [ ] `list_files()` - Returns list of `FileInfo`
- [ ] `get_file_status()` - Returns `FileInfo` or None
- [ ] `health_check()` - Optional, returns True if healthy

### Registration

- [ ] `@register_retriever("name")` decorator on retriever class
- [ ] `@register_ingestor("name")` decorator on ingestor class
- [ ] Classes exported in `__init__.py`

### Error Handling

- [ ] `retrieve()` returns error state (not exceptions)
- [ ] `get_job_status()` handles unknown job_id
- [ ] `FileProgress.error_message` populated on failures
- [ ] Meaningful error messages for UI display

### Testing

- [ ] Health check works
- [ ] Collection CRUD works
- [ ] File upload and status polling works
- [ ] Retrieval returns valid Chunk objects
- [ ] Error cases handled gracefully

---

## HTTP API Integration

> **Note:** This section is provided for reference only. You do not need to implement or integrate the HTTP layer yourself--the host application handles this automatically once your adapter is registered. This documentation helps you understand how your adapter methods are called in production.

Once your adapter is registered, the host application exposes it through HTTP endpoints. This section describes how your adapter methods map to REST API calls that clients (like Deep Researcher Agent) consume.

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Client Application (Deep Researcher Agent)                   │
│                    (React/Web UI, CLI, etc.)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI HTTP Layer                             │
│                                                                     │
│   POST /v1/collections          → ingestor.create_collection()      │
│   GET  /v1/collections          → ingestor.list_collections()       │
│   GET  /v1/collections/{name}   → ingestor.get_collection()         │
│   DELETE /v1/collections/{name} → ingestor.delete_collection()      │
│                                                                     │
│   POST /v1/collections/{name}/documents → ingestor.submit_job()     │
│   GET  /v1/collections/{name}/documents → ingestor.list_files()     │
│   DELETE /v1/collections/{name}/documents → ingestor.delete_files() │
│                                                                     │
│   GET  /v1/documents/{job_id}/status → ingestor.get_job_status()    │
│   GET  /v1/knowledge/health          → ingestor.health_check()      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Your Backend Adapter                         │
│              (BaseIngestor / BaseRetriever)                     │
└─────────────────────────────────────────────────────────────────┘
```

### How the Ingestor is Obtained

The HTTP layer obtains your ingestor through the **active ingestor pattern**:

1. During application startup, the `knowledge_retrieval` function is configured
2. This triggers import of your adapter module (registering it with the factory)
3. The factory creates your ingestor singleton and sets it as "active"
4. HTTP routes retrieve the active ingestor using `get_active_ingestor()`

```python
# Simplified flow (internal to host application)

# 1. Application startup configures knowledge_retrieval
#    This imports your adapter and registers it

# 2. Factory creates singleton and activates it
ingestor = get_ingestor("your_backend", config={...})
set_active_ingestor(ingestor)

# 3. HTTP routes use the active ingestor
def _get_active_ingestor():
    return get_active_ingestor()  # Returns your singleton

# 4. Routes call your methods
@app.get("/v1/collections")
async def list_collections():
    ingestor = _get_active_ingestor()
    return ingestor.list_collections()
```

### REST Endpoint Reference

#### Collections API

| Method | Endpoint | Adapter Method | Description |
|--------|----------|----------------|-------------|
| `POST` | `/v1/collections` | `create_collection()` | Create a new collection |
| `GET` | `/v1/collections` | `list_collections()` | List all collections |
| `GET` | `/v1/collections/{name}` | `get_collection()` | Get collection details |
| `DELETE` | `/v1/collections/{name}` | `delete_collection()` | Delete a collection |

**Create Collection Request:**

```text
POST /v1/collections
{
  "name": "my_documents",
  "description": "My document collection",
  "metadata": {}
}
```

**Response:** Returns `CollectionInfo` from your adapter.

---

#### Documents API

| Method | Endpoint | Adapter Method | Description |
|--------|----------|----------------|-------------|
| `POST` | `/v1/collections/{name}/documents` | `submit_job()` | Upload files (returns job_id) |
| `GET` | `/v1/collections/{name}/documents` | `list_files()` | List files in collection |
| `DELETE` | `/v1/collections/{name}/documents` | `delete_files()` | Delete files by ID |

**Upload Documents Request:**

```text
POST /v1/collections/my_documents/documents
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf]
```

**Response:**

```json
{
  "job_id": "abc123-def456",
  "file_ids": ["file-001", "file-002"],
  "message": "Ingestion job submitted for 2 file(s)"
}
```

The HTTP layer:

1. Saves uploaded files to temporary locations
2. Calls `ingestor.submit_job(temp_paths, collection_name, config)`
3. Returns the job_id for status polling
4. Your adapter is responsible for cleaning up temp files after processing

---

#### Job Status API

| Method | Endpoint | Adapter Method | Description |
|--------|----------|----------------|-------------|
| `GET` | `/v1/documents/{job_id}/status` | `get_job_status()` | Poll ingestion progress |

**Response:** Returns `IngestionJobStatus` from your adapter.

```json
{
  "job_id": "abc123-def456",
  "status": "processing",
  "submitted_at": "2025-01-15T10:30:00Z",
  "total_files": 2,
  "processed_files": 1,
  "file_details": [
    {
      "file_id": "file-001",
      "file_name": "report.pdf",
      "status": "success",
      "chunks_created": 45
    },
    {
      "file_id": "file-002",
      "file_name": "data.csv",
      "status": "ingesting",
      "progress_percent": 50.0
    }
  ],
  "collection_name": "my_documents",
  "backend": "your_backend"
}
```

**Polling Pattern:**

Clients poll this endpoint until `is_terminal` is true:

```javascript
// Client-side polling (simplified)
async function waitForIngestion(jobId) {
  while (true) {
    const status = await fetch(`/v1/documents/${jobId}/status`);
    const data = await status.json();

    if (data.status === 'completed' || data.status === 'failed') {
      return data;
    }

    await sleep(1000);  // Poll every second
  }
}
```

---

#### Health API

| Method | Endpoint | Adapter Method | Description |
|--------|----------|----------------|-------------|
| `GET` | `/v1/knowledge/health` | `health_check()` | Check backend connectivity |

**Response:**

```json
{
  "status": "healthy",
  "backend": "your_backend"
}
```

### Error Response Format

HTTP errors are returned as JSON with appropriate status codes:

```json
{
  "detail": "Collection 'nonexistent' not found"
}
```

| HTTP Status | Meaning |
|-------------|---------|
| `200` | Success |
| `201` | Created (POST /collections) |
| `202` | Accepted (POST /documents - job submitted) |
| `400` | Bad request (invalid input) |
| `404` | Not found (collection/job doesn't exist) |
| `500` | Server error (adapter exception) |
| `503` | Service unavailable (health check failed) |

### Important Implementation Notes

1. **Your adapter methods are called synchronously** by the HTTP layer (except `health_check()` which is async). Ensure your methods don't block for extended periods.

2. **File uploads go through temp files.** The HTTP layer writes uploaded files to temporary paths and passes those paths to `submit_job()`. Set `cleanup_files: True` in the config to have your adapter delete them after processing.

3. **Original filenames are preserved.** The HTTP layer passes `original_filenames` in the config dict so your `file_details` can display user-friendly names instead of temp file paths.

4. **Error messages surface to users.** The `FileProgress.error_message` field is displayed directly in the UI. Make these messages user-friendly, not stack traces.

5. **The ingestor is a singleton.** The same instance handles all requests, so it must be thread-safe if you use background threads for ingestion.

---

## Questions?

If you're implementing a backend adapter and have questions:

1. Review the existing implementations:
   - `sources/knowledge_layer/src/llamaindex/adapter.py` - Local ChromaDB backend example
   - `sources/knowledge_layer/src/foundational_rag/adapter.py` - HTTP-based RAG service example

2. The complete schema definitions are included in this document (refer to [Data Schemas](#data-schemas))

3. Test with the validation scripts above before integration

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [User Guide](../customization/knowledge-layer.md) | Setup, configuration, Web UI, YAML config examples |
| Foundational RAG Setup (`sources/knowledge_layer/src/foundational_rag/README.md`) | Production deployment with NVIDIA RAG Blueprint |
| Knowledge Layer README (`sources/knowledge_layer/README.md`) | Quick reference and installation |
