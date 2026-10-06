# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental in-process NeMo Retriever backend for the Deep Researcher Agent Knowledge Layer."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from deep_researcher_agent.knowledge import BaseIngestor
from deep_researcher_agent.knowledge import BaseRetriever
from deep_researcher_agent.knowledge import Chunk
from deep_researcher_agent.knowledge import IngestionJobStatus
from deep_researcher_agent.knowledge import RetrievalResult
from deep_researcher_agent.knowledge import register_ingestor
from deep_researcher_agent.knowledge import register_retriever
from deep_researcher_agent.knowledge.schema import CollectionInfo
from deep_researcher_agent.knowledge.schema import FileInfo

from ._local_client import LocalRuntimeHandle
from ._local_client import NemoRetrieverLocalError
from ._local_client import acquire_local_runtime
from ._models import QueryHitWire
from ._normalization import UNSUPPORTED_FILTERS_ERROR
from ._normalization import normalize_query_hit

logger = logging.getLogger(__name__)

_BACKEND_NAME = "nemo_retriever_local"


class _LocalAdapter:
    """Shared lifecycle and runtime setup for both registered adapter roles."""

    _runtime_handle: LocalRuntimeHandle

    def _initialize_local_runtime(self) -> None:
        self._runtime_handle = acquire_local_runtime(self.config)
        self._runtime = self._runtime_handle.runtime

    def close(self) -> None:
        """Release this adapter's process-lifetime runtime reference."""
        try:
            self._runtime_handle.close()
        except Exception as error:
            raise self._public_exception(error) from error

    def _public_exception(self, error: Exception) -> Exception:
        """Preserve safe validation types while containing backend details."""
        message = self._runtime.public_error(error)
        if isinstance(error, NemoRetrieverLocalError):
            if message == str(error):
                return error
            return type(error)(message)
        if isinstance(error, ValueError):
            return error if message == str(error) else ValueError(message)
        return NemoRetrieverLocalError(message)

    def _runtime_call(self, operation: Any, /, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            public = self._public_exception(error)
            if public is error:
                raise
            raise public from error


@register_retriever(_BACKEND_NAME)
class NemoRetrieverLocalRetriever(_LocalAdapter, BaseRetriever):
    """Query embedded LanceDB through NeMo Retriever's scoped VDB operator."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._initialize_local_runtime()

    @property
    def backend_name(self) -> str:
        return _BACKEND_NAME

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        if filters:
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=[],
                success=False,
                error_message=UNSUPPORTED_FILTERS_ERROR,
            )
        try:
            hits = await asyncio.to_thread(self._runtime.query, query, collection_name, top_k)
            chunks = [self.normalize(hit) for hit in hits]
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=chunks,
                total_tokens=sum(max(1, len(chunk.content) // 4) for chunk in chunks if chunk.content),
                success=True,
            )
        except Exception as error:
            message = self._runtime.public_error(error)
            logger.warning("Embedded NeMo Retriever query failed: %s", message)
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=[],
                success=False,
                error_message=message,
            )

    def normalize(self, raw_result: Any) -> Chunk:
        hit = raw_result if isinstance(raw_result, QueryHitWire) else QueryHitWire.model_validate(raw_result)
        return normalize_query_hit(hit)

    async def health_check(self) -> bool:
        return await asyncio.to_thread(self._runtime.health_check)


@register_ingestor(_BACKEND_NAME)
class NemoRetrieverLocalIngestor(_LocalAdapter, BaseIngestor):
    """Manage local NRL collections and bounded background ingestion."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._initialize_local_runtime()

    @property
    def backend_name(self) -> str:
        return _BACKEND_NAME

    def create_collection(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        return self._runtime_call(self._runtime.create_collection, name, description, metadata)

    def delete_collection(self, name: str) -> bool:
        return self._runtime_call(self._runtime.delete_collection, name)

    def list_collections(self) -> list[CollectionInfo]:
        return self._runtime_call(self._runtime.list_collections)

    def get_collection(self, name: str) -> CollectionInfo | None:
        return self._runtime_call(self._runtime.get_collection, name)

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        return self._runtime_call(self._runtime.submit_job, file_paths, collection_name, config)

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        return self._runtime_call(self._runtime.get_job_status, job_id)

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        return self._runtime_call(self._runtime.upload_file, file_path, collection_name, metadata)

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        return self._runtime_call(self._runtime.delete_file, file_id, collection_name)

    def list_files(self, collection_name: str) -> list[FileInfo]:
        return self._runtime_call(self._runtime.list_files, collection_name)

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        return self._runtime_call(self._runtime.get_file_status, file_id, collection_name)

    async def health_check(self) -> bool:
        return await asyncio.to_thread(self._runtime.health_check)


__all__ = ["NemoRetrieverLocalIngestor", "NemoRetrieverLocalRetriever"]
