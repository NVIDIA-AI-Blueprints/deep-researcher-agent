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
NAT plugin registration for Knowledge API endpoints.

This module provides FastAPI routes for managing collections and documents
via the knowledge layer adapters. Configuration follows the same pattern
as sources/knowledge_layer/src/register.py.
"""

import logging
import os

from fastapi import FastAPI
from pydantic import Field
from pydantic import model_validator

from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_front_end
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase
from nat.utils.type_utils import override

from .routes.collections import add_collection_routes
from .routes.documents import add_document_routes

logger = logging.getLogger(__name__)


class KnowledgeAPIConfig(FastApiFrontEndConfig, name="deep_researcher_frontend"):
    """Configuration for Knowledge API endpoints."""

    backend: str = Field(
        default="llamaindex",
        description="Knowledge backend: 'llamaindex' or 'foundational_rag'",
    )
    # LlamaIndex-specific
    chroma_dir: str = Field(
        default="/tmp/chroma_data",
        description="Directory for ChromaDB persistence (LlamaIndex only)",
    )
    # Foundational RAG-specific
    rag_url: str = Field(
        default="http://localhost:8081/v1",
        description="RAG query server URL (foundational_rag only)",
    )
    ingest_url: str = Field(
        default="http://localhost:8082/v1",
        description="RAG ingestion server URL (foundational_rag only)",
    )
    timeout: int = Field(
        default=120,
        description="Request timeout in seconds (foundational_rag only)",
    )

    @model_validator(mode="after")
    def validate_backend_config(self):
        """Validate and warn about unused backend-specific config options."""
        backend = self.backend.lower()

        if backend == "llamaindex":
            if self.rag_url != "http://localhost:8081/v1":
                logger.warning("rag_url is ignored for llamaindex backend")
            if self.ingest_url != "http://localhost:8082/v1":
                logger.warning("ingest_url is ignored for llamaindex backend")

        elif backend == "foundational_rag":
            if self.chroma_dir != "/tmp/chroma_data":
                logger.warning("chroma_dir is ignored for foundational_rag backend")

        return self


def _get_ingestor(config: KnowledgeAPIConfig):
    """
    Get ingestor instance based on config.

    Uses the factory singleton to avoid creating duplicate instances
    (and duplicate TTL cleanup threads). Falls back to direct instantiation
    only if the factory is not available.
    """
    backend = config.backend.lower()

    if backend == "llamaindex":
        os.environ.setdefault("DEEP_RESEARCHER_CHROMA_DIR", config.chroma_dir)
        os.environ.setdefault("KNOWLEDGE_INGESTOR_BACKEND", "llamaindex")

        # Use factory singleton to share instance with knowledge_retrieval plugin
        from deep_researcher_agent.knowledge.factory import get_ingestor

        ingestor = get_ingestor("llamaindex", {"persist_dir": config.chroma_dir})
        logger.info(f"Initialized LlamaIndex ingestor via factory (chroma_dir={config.chroma_dir})")
        return ingestor

    elif backend == "foundational_rag":
        os.environ.setdefault("KNOWLEDGE_INGESTOR_BACKEND", "foundational_rag")

        from deep_researcher_agent.knowledge.factory import get_ingestor

        ingestor = get_ingestor(
            "foundational_rag",
            {
                "rag_url": config.rag_url,
                "ingest_url": config.ingest_url,
                "timeout": config.timeout,
            },
        )
        logger.info(f"Initialized Foundational RAG ingestor (ingest_url={config.ingest_url})")
        return ingestor

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'llamaindex' or 'foundational_rag'.")


class KnowledgeAPIWorker(FastApiFrontEndPluginWorker):
    """Worker that adds knowledge API routes to the FastAPI app."""

    @override
    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):
        await super().add_routes(app, builder)

        # front_end_config is set by parent class from config.general.front_end
        # It's our KnowledgeAPIConfig instance
        knowledge_config: KnowledgeAPIConfig = self.front_end_config  # type: ignore[assignment]

        # Get configured ingestor (singleton for job state preservation)
        ingestor = _get_ingestor(knowledge_config)

        # Add routes
        add_collection_routes(app, ingestor)
        add_document_routes(app, ingestor)

        logger.info(f"Knowledge API routes registered (backend={knowledge_config.backend})")


class KnowledgeAPIPlugin(FastApiFrontEndPlugin):
    """Plugin that adds knowledge management endpoints to the FastAPI server."""

    def __init__(self, full_config: Config, config: KnowledgeAPIConfig):
        super().__init__(full_config=full_config)
        self.config = config

    @override
    def get_worker_class(self) -> type[FastApiFrontEndPluginWorkerBase]:
        return KnowledgeAPIWorker


@register_front_end(config_type=KnowledgeAPIConfig)
async def register_knowledge_api(config: KnowledgeAPIConfig, full_config: Config):
    """Register Knowledge API with NAT framework."""
    yield KnowledgeAPIPlugin(full_config=full_config, config=config)
