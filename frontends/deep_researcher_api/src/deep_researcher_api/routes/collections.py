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

"""Collection management endpoints."""

import asyncio
import logging

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from deep_researcher_agent.knowledge.base import BaseIngestor
from deep_researcher_agent.knowledge.schema import CollectionInfo

from ..models.requests import CreateCollectionRequest

logger = logging.getLogger(__name__)


def _require_ingestor() -> BaseIngestor:
    from deep_researcher_agent.knowledge.factory import get_active_ingestor

    ingestor = get_active_ingestor()
    if ingestor is None:
        raise HTTPException(status_code=503, detail="Knowledge API not configured")
    return ingestor


def add_collection_routes(router: APIRouter):
    """Add collection management routes to the FastAPI app."""

    @router.post(
        "/v1/collections",
        response_model=CollectionInfo,
        status_code=201,
        tags=["collections"],
        summary="Create a new collection",
    )
    async def create_collection(
        request: CreateCollectionRequest,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> CollectionInfo:
        """Create a new collection for storing documents."""
        try:
            return await asyncio.to_thread(
                ingestor.create_collection,
                name=request.name,
                description=request.description,
                metadata=request.metadata,
            )
        except Exception as e:
            logger.error(f"Failed to create collection '{request.name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/collections",
        response_model=list[CollectionInfo],
        tags=["collections"],
        summary="List all collections",
    )
    async def list_collections(
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> list[CollectionInfo]:
        """List all available collections."""
        try:
            return await asyncio.to_thread(ingestor.list_collections)
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/collections/{name}",
        response_model=CollectionInfo,
        tags=["collections"],
        summary="Get collection details",
    )
    async def get_collection(
        name: str,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> CollectionInfo:
        """Get details for a specific collection."""
        collection = await asyncio.to_thread(ingestor.get_collection, name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return collection

    @router.delete(
        "/v1/collections/{name}",
        tags=["collections"],
        summary="Delete a collection",
    )
    async def delete_collection(
        name: str,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> dict:
        """Delete a collection and all its contents."""
        try:
            success = await asyncio.to_thread(ingestor.delete_collection, name)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to delete collection '{name}'")
            return {"success": True, "collection": name}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/knowledge/health",
        tags=["health"],
        summary="Check knowledge backend health",
    )
    async def health_check(
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> dict:
        """Check if the knowledge backend is healthy and reachable."""
        try:
            healthy = await ingestor.health_check()
            if not healthy:
                raise HTTPException(status_code=503, detail="Knowledge backend unhealthy")
            return {"status": "healthy", "backend": ingestor.backend_name}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))
