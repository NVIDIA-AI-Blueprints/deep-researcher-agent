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
Knowledge Layer - Universal RAG interfaces and schemas.

This module defines the foundational abstractions for knowledge ingestion and retrieval.
All backend adapters (LlamaIndex, Foundational RAG, etc.) must implement these interfaces
and output data conforming to these schemas.

Architecture:
    src/deep_researcher_agent/knowledge/      # Core abstractions
    ├── schema.py                # Data models (Chunk, RetrievalResult, etc.)
    ├── base.py                  # Abstract interfaces (BaseRetriever, BaseIngestor)
    └── factory.py               # Registry + factory pattern

    sources/knowledge_layer/src/ # Backend implementations
    ├── llamaindex/              # ChromaDB + NVIDIA embeddings
    ├── nvingest/                # Milvus + NV-Ingest pipeline
    └── foundational_rag/        # Hosted RAG Blueprint

Usage:
    from deep_researcher_agent.knowledge import Chunk, BaseRetriever, get_retriever
"""

from .base import BaseIngestor
from .base import BaseRetriever
from .factory import clear_active_ingestor
from .factory import clear_all_summaries
from .factory import clear_collection_summaries
from .factory import configure_summary_db
from .factory import get_active_ingestor
from .factory import get_available_documents
from .factory import get_available_documents_async
from .factory import get_ingestor
from .factory import get_retriever
from .factory import list_summary_collections
from .factory import register_ingestor
from .factory import register_retriever
from .factory import register_summary
from .factory import set_active_ingestor
from .factory import unregister_summary
from .schema import AvailableDocument
from .schema import Chunk
from .schema import ContentType
from .schema import FileProgress
from .schema import IngestionJobStatus
from .schema import JobState
from .schema import RetrievalResult

__all__ = [
    # Schema
    "AvailableDocument",
    "Chunk",
    "ContentType",
    "FileProgress",
    "IngestionJobStatus",
    "JobState",
    "RetrievalResult",
    # Base classes
    "BaseRetriever",
    "BaseIngestor",
    # Factory
    "get_retriever",
    "get_ingestor",
    "register_retriever",
    "register_ingestor",
    # Active ingestor (for Knowledge API)
    "get_active_ingestor",
    "set_active_ingestor",
    "clear_active_ingestor",
    # Summary Registry (SQLAlchemy-backed, backend-agnostic)
    "configure_summary_db",
    "register_summary",
    "unregister_summary",
    "get_available_documents",
    "get_available_documents_async",
    "list_summary_collections",
    "clear_collection_summaries",
    "clear_all_summaries",
]
