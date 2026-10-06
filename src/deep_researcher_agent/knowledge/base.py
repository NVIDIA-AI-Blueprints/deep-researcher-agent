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
Base adapter interfaces for the Knowledge Layer.

This module defines abstract base classes that all retrieval and ingestion
adapters must implement. The Adapter Pattern allows new backends to be added
without modifying core code.
"""

import logging
import threading
import time
from abc import ABC
from abc import abstractmethod
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

from .schema import Chunk
from .schema import CollectionInfo
from .schema import FileInfo
from .schema import IngestionJobStatus
from .schema import RetrievalResult

logger = logging.getLogger(__name__)


class IngestionBatchTooLargeError(RuntimeError):
    """Raised when one ingestion batch exceeds the backend's admission limit."""

    def __init__(self) -> None:
        super().__init__("Upload batch exceeds the configured ingestion capacity")


class IngestionCapacityError(RuntimeError):
    """Raised when temporary backend load prevents immediate ingestion admission."""

    def __init__(self) -> None:
        super().__init__("Document ingestion is temporarily at capacity; retry later")


class TTLCleanupMixin:
    """
    Mixin that provides TTL-based collection cleanup.

    This mixin adds background thread-based cleanup of collections that haven't
    been updated within a configurable time period. Collections without an
    updated_at timestamp are skipped.

    Requires the class to implement:
        - list_collections() -> list[CollectionInfo]
        - delete_collection(name: str) -> bool
        - backend_name: str (for logging)

    Usage:
        class MyIngestor(TTLCleanupMixin, BaseIngestor):
            def __init__(self, config):
                super().__init__(config)
                # Start cleanup with 24-hour TTL, checking every hour
                self._start_ttl_cleanup_task(ttl_hours=24, interval_seconds=3600)
    """

    _ttl_hours: float
    _cleanup_interval_seconds: int

    def _start_ttl_cleanup_task(self, ttl_hours: float, interval_seconds: int) -> None:
        """
        Start the background thread for TTL-based collection cleanup.

        Args:
            ttl_hours: Hours before a collection is considered expired.
            interval_seconds: Seconds between cleanup runs.
        """
        self._ttl_hours = ttl_hours
        self._cleanup_interval_seconds = interval_seconds

        thread = threading.Thread(
            target=self._ttl_cleanup_loop,
            daemon=True,
            name=f"{self.backend_name}-ttl-cleanup",
        )
        thread.start()
        logger.info(
            f"Started collection TTL cleanup task for {self.backend_name} "
            f"(runs every {interval_seconds}s, TTL={ttl_hours}h)"
        )

    def _ttl_cleanup_loop(self) -> None:
        """Background loop that periodically cleans up expired collections."""
        while True:
            try:
                time.sleep(self._cleanup_interval_seconds)
                self._cleanup_expired_collections()
            except Exception as e:
                logger.error(f"TTL cleanup loop error for {self.backend_name}: {e}")

    def _cleanup_expired_collections(self) -> None:
        """Check all collections and delete those that have expired."""
        try:
            collections = self.list_collections()
            logger.info(f"TTL cleanup ({self.backend_name}): checking {len(collections)} collections for expiration")
            now = datetime.now(UTC)
            ttl_threshold = now - timedelta(hours=self._ttl_hours)

            for collection in collections:
                if collection.updated_at is None:
                    logger.debug(f"TTL cleanup ({self.backend_name}): skipping '{collection.name}' - no timestamp")
                    continue

                # Ensure updated_at is timezone-aware for comparison
                updated_at = collection.updated_at
                if updated_at.tzinfo is None:
                    # Assume naive datetime is UTC
                    updated_at = updated_at.replace(tzinfo=UTC)

                logger.debug(
                    f"TTL cleanup ({self.backend_name}): '{collection.name}' "
                    f"updated_at={collection.updated_at}, threshold={ttl_threshold}, "
                    f"expired={updated_at < ttl_threshold}"
                )

                if updated_at < ttl_threshold:
                    logger.info(
                        f"Collection '{collection.name}' expired (last indexed: {collection.updated_at}), deleting..."
                    )
                    if self.delete_collection(collection.name):
                        logger.info(
                            f"TTL cleanup ({self.backend_name}): deleted expired collection '{collection.name}'"
                        )
                    else:
                        logger.warning(
                            f"TTL cleanup ({self.backend_name}): failed to delete collection '{collection.name}'"
                        )

        except Exception as e:
            logger.error(f"Failed to cleanup expired collections for {self.backend_name}: {e}")


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval adapters.

    All retrieval backends must implement this interface. The key responsibility
    is converting backend-specific results into the Universal Chunk schema.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the retriever with configuration.

        Args:
            config: Backend-specific configuration dictionary.
        """
        self.config = config or {}

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

        Args:
            query: The search query string.
            collection_name: Target collection/index name.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            RetrievalResult containing normalized Chunks.
        """

    @abstractmethod
    def normalize(self, raw_result: Any) -> Chunk:
        """
        Convert a backend-specific result into a universal Chunk.

        This is the core of the Adapter Pattern - each backend must translate
        its native format to the Universal Schema.

        Args:
            raw_result: The raw result from the backend.

        Returns:
            A normalized Chunk object.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend for logging/tracking."""

    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and reachable.

        Returns:
            True if healthy, False otherwise.
        """
        return True  # NOSONAR


class BaseIngestor(ABC):
    """
    Abstract base class for ingestion adapters.

    All ingestion backends must implement this interface. Ingestion is
    inherently asynchronous - submit_job returns immediately with a job ID,
    and get_job_status is used for polling.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the ingestor with configuration.

        Args:
            config: Backend-specific configuration dictionary.
        """
        self.config = config or {}

    @abstractmethod
    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """
        Non-blocking job submission.

        This method should return immediately with a job ID. The actual
        processing happens asynchronously in the background.

        Args:
            file_paths: List of file paths (local or S3 URIs) to ingest.
            collection_name: Target collection/index name.
            config: Optional ingestion configuration (chunking, extraction, and so on).

        Returns:
            job_id (str) immediately.
        """

    @abstractmethod
    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        """
        Get the current status of an ingestion job.

        This is called by the polling loop to check progress.

        Args:
            job_id: The job ID returned from submit_job.

        Returns:
            IngestionJobStatus with current state.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend for logging/tracking."""

    # =========================================================================
    # Collection Management
    # =========================================================================

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
            description: Human-readable description.
            metadata: Backend-specific configuration.

        Returns:
            CollectionInfo with the created collection details.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection and all its contents.

        Args:
            name: Collection name to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """

    @abstractmethod
    def list_collections(self) -> list[CollectionInfo]:
        """
        List all available collections.

        Returns:
            List of CollectionInfo objects.
        """

    @abstractmethod
    def get_collection(self, name: str) -> CollectionInfo | None:
        """
        Get metadata for a specific collection.

        Args:
            name: Collection name.

        Returns:
            CollectionInfo if found, None otherwise.
        """

    # =========================================================================
    # File Management
    # =========================================================================

    @abstractmethod
    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        """
        Upload a file to a collection.

        This should handle the file upload and return immediately.
        Actual ingestion may happen asynchronously via submit_job.

        Args:
            file_path: Local path to the file.
            collection_name: Target collection.
            metadata: Optional file metadata.

        Returns:
            FileInfo with upload status.
        """

    @abstractmethod
    def delete_file(self, file_id: str, collection_name: str) -> bool:
        """
        Delete a file and its chunks from a collection.

        Args:
            file_id: File identifier (can be filename or internal ID).
            collection_name: Collection containing the file.

        Returns:
            True if deleted successfully, False otherwise.
        """

    def delete_files(
        self,
        file_ids: list[str],
        collection_name: str,
    ) -> dict[str, Any]:
        """
        Delete multiple files from a collection (batch delete).

        Follows NVIDIA RAG Blueprint pattern for batch file deletion.

        Args:
            file_ids: List of file IDs to delete.
            collection_name: Collection containing the files.

        Returns:
            Dict with 'successful' (list), 'failed' (list), 'total_deleted' (int).
        """
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

    @abstractmethod
    def list_files(self, collection_name: str) -> list[FileInfo]:
        """
        List all files in a collection.

        Args:
            collection_name: Collection to list files from.

        Returns:
            List of FileInfo objects.
        """

    @abstractmethod
    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        """
        Get the current status of a file.

        Args:
            file_id: File identifier.
            collection_name: Collection containing the file.

        Returns:
            FileInfo if found, None otherwise.
        """

    # =========================================================================
    # Optional: Source Selection (for multi-collection queries)
    # =========================================================================

    def select_sources(self, source_names: list[str]) -> bool:
        """
        Select which collections/sources to use for retrieval.

        This is optional - backends that don't support multi-source
        queries can leave the default implementation.

        Args:
            source_names: List of collection names to activate.

        Returns:
            True if selection succeeded, False otherwise.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support source selection. Override this method if your backend supports it."
        )

    def get_selected_sources(self) -> list[str]:
        """
        Get the currently selected sources.

        Returns:
            List of selected collection names.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support source selection. Override this method if your backend supports it."
        )

    # =========================================================================
    # Optional: Document Summarization
    # =========================================================================

    def generate_summary(self, text_content: str, file_name: str) -> str | None:
        """
        Generate a short summary of the document content.

        Override in adapters to enable summarization. Default returns None.

        Args:
            text_content: Combined text from first and last chunks.
            file_name: Original filename for context.

        Returns:
            One-sentence summary or None if not implemented.
        """
        return None

    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and reachable.

        Returns:
            True if healthy, False otherwise.
        """
        return True  # NOSONAR
