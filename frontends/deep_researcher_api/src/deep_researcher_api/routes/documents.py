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

"""Document management endpoints."""

import asyncio
import logging
from functools import partial
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile

from deep_researcher_agent.fastapi_extensions.upload_security import UPLOAD_ENDPOINT_DESCRIPTION
from deep_researcher_agent.fastapi_extensions.upload_security import UploadValidationError
from deep_researcher_agent.fastapi_extensions.upload_security import get_upload_limits
from deep_researcher_agent.fastapi_extensions.upload_security import submit_validated_upload_batch
from deep_researcher_agent.fastapi_extensions.upload_security import validate_upload_count
from deep_researcher_agent.fastapi_extensions.upload_security import validated_upload_batch
from deep_researcher_agent.knowledge.base import BaseIngestor
from deep_researcher_agent.knowledge.base import IngestionBatchTooLargeError
from deep_researcher_agent.knowledge.base import IngestionCapacityError
from deep_researcher_agent.knowledge.schema import FileInfo
from deep_researcher_agent.knowledge.schema import IngestionJobStatus

from ..models.requests import DeleteFilesRequest
from ..models.requests import UploadResponse
from .collections import _require_ingestor

logger = logging.getLogger(__name__)


def add_document_routes(router: APIRouter):
    """Add document management routes to the FastAPI app."""

    @router.post(
        "/v1/collections/{collection_name}/documents",
        response_model=UploadResponse,
        status_code=202,
        tags=["documents"],
        summary="Upload documents to a collection",
        description=UPLOAD_ENDPOINT_DESCRIPTION,
        responses={
            400: {"description": "No files provided"},
            404: {"description": "Collection not found"},
            413: {"description": "Upload size, file-count, or ingestion-capacity limit exceeded"},
            415: {"description": "Unsupported, malformed, or mismatched file content"},
            503: {"description": "Document ingestion is temporarily at capacity"},
            500: {"description": "Ingestion failed"},
        },
    )
    async def upload_documents(
        collection_name: str,
        files: list[UploadFile] = File(..., description="Files to upload"),
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> UploadResponse:
        """
        Upload documents to a collection.

        Returns a job ID for polling the ingestion status.
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        limits = get_upload_limits()
        try:
            validate_upload_count(len(files), limits)
        except UploadValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        # Verify collection exists
        collection = await asyncio.to_thread(ingestor.get_collection, collection_name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        try:
            async with validated_upload_batch(files, limits=limits) as batch:
                job_id = await submit_validated_upload_batch(
                    partial(
                        ingestor.submit_job,
                        batch.temp_paths,
                        collection_name,
                        config={
                            "cleanup_files": True,
                            "original_filenames": batch.original_filenames,
                        },
                    ),
                    batch,
                )

                job_status = await asyncio.to_thread(ingestor.get_job_status, job_id)
                file_ids = [fd.file_id for fd in job_status.file_details]
                logger.info("Submitted ingestion job %s for %d file(s)", job_id, len(files))
                return UploadResponse(
                    job_id=job_id,
                    file_ids=file_ids,
                    message=f"Ingestion job submitted for {len(files)} file(s)",
                )

        except UploadValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except IngestionBatchTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except IngestionCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to submit document ingestion job (error_type=%s)",
                type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail="Failed to submit ingestion job") from exc

    @router.get(
        "/v1/collections/{collection_name}/documents",
        response_model=list[FileInfo],
        tags=["documents"],
        summary="List documents in a collection",
    )
    async def list_documents(
        collection_name: str,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> list[FileInfo]:
        """List all documents in a collection."""
        # Verify collection exists
        collection = await asyncio.to_thread(ingestor.get_collection, collection_name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        try:
            return await asyncio.to_thread(ingestor.list_files, collection_name)
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "/v1/collections/{collection_name}/documents",
        tags=["documents"],
        summary="Delete files from a collection",
    )
    async def delete_files(
        collection_name: str,
        request: DeleteFilesRequest,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> dict[str, Any]:
        """Delete files from a collection by ID."""
        # Verify collection exists
        collection = await asyncio.to_thread(ingestor.get_collection, collection_name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        if not request.file_ids:
            return {
                "message": "No file IDs provided",
                "successful": [],
                "failed": [],
                "total_deleted": 0,
            }

        try:
            result = await asyncio.to_thread(ingestor.delete_files, request.file_ids, collection_name)
            total_deleted = result.get("total_deleted", 0)
            failed = result.get("failed", [])

            if failed:
                result["message"] = "Some files could not be deleted"
            elif total_deleted == 0:
                result["message"] = "No matching files found"
            else:
                result["message"] = f"Successfully deleted {total_deleted} file(s)"

            return result
        except Exception as e:
            logger.error(f"Failed to delete files from {collection_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/documents/{job_id}/status",
        response_model=IngestionJobStatus,
        tags=["documents"],
        summary="Get ingestion job status",
    )
    async def get_job_status(
        job_id: str,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> IngestionJobStatus:
        """Get the status of an ingestion job."""
        try:
            status = await asyncio.to_thread(ingestor.get_job_status, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

            return status
        except HTTPException:
            raise
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code in {404, 410}:
                detail = f"Job '{job_id}' not found" if status_code == 404 else f"Job '{job_id}' expired"
                raise HTTPException(status_code=status_code, detail=detail) from e
            logger.error(f"Failed to get job status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
