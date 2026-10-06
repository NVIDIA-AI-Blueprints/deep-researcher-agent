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
Factory pattern for Knowledge Layer adapters.

This module provides a registry-based factory for creating retriever and
ingestor instances. New backends are registered using decorators and can
be instantiated by name at runtime.

Configuration:
    The default backend is configured via environment variables at startup:
    - KNOWLEDGE_RETRIEVER_BACKEND: Default retriever (e.g., "llamaindex")
    - KNOWLEDGE_INGESTOR_BACKEND: Default ingestor (e.g., "llamaindex")

    This ensures the backend is set up BEFORE any API calls are made.
    Users don't need to specify the backend in each request.
"""

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from .base import BaseIngestor
from .base import BaseRetriever

if TYPE_CHECKING:
    from .schema import AvailableDocument
    from .summary_store import SummaryStore

logger = logging.getLogger(__name__)

# =============================================================================
# Default Backend Configuration (Set at Startup)
# =============================================================================
# These are read from environment variables at module load time.
# The API layer uses these defaults - users don't need to specify per-request.
#
# NOTE: Backend-specific configuration (milvus_uri, chroma_path, etc.) should
# be defined in each adapter, NOT here. The factory is backend-agnostic.

# @environment_variable KNOWLEDGE_RETRIEVER_BACKEND
# @category Knowledge Layer
# @type str
# @default llamaindex
# @required false
# Default retriever backend. Set at startup by knowledge_retrieval function.
DEFAULT_RETRIEVER_BACKEND = os.environ.get("KNOWLEDGE_RETRIEVER_BACKEND", "llamaindex")

# @environment_variable KNOWLEDGE_INGESTOR_BACKEND
# @category Knowledge Layer
# @type str
# @default llamaindex
# @required false
# Default ingestor backend. Set at startup by knowledge_retrieval function.
DEFAULT_INGESTOR_BACKEND = os.environ.get("KNOWLEDGE_INGESTOR_BACKEND", "llamaindex")

# =============================================================================
# Registry
# =============================================================================

# Registry for retriever adapters (class registry)
_RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {}

# Registry for ingestor adapters (class registry)
_INGESTOR_REGISTRY: dict[str, type[BaseIngestor]] = {}

# Singleton instances (cached for job state persistence)
_RETRIEVER_INSTANCES: dict[str, BaseRetriever] = {}
_INGESTOR_INSTANCES: dict[str, BaseIngestor] = {}

# Active ingestor for the Knowledge API (set by knowledge_retrieval function)
_ACTIVE_INGESTOR: BaseIngestor | None = None


def register_retriever(name: str) -> Callable[[type[BaseRetriever]], type[BaseRetriever]]:
    """
    Decorator to register a retriever adapter.

    Usage:
        @register_retriever("llamaindex")
        class LlamaIndexRetriever(BaseRetriever):
            ...

    Args:
        name: The name to register this adapter under.

    Returns:
        Decorator function.
    """

    def decorator(cls: type[BaseRetriever]) -> type[BaseRetriever]:
        if name in _RETRIEVER_REGISTRY:
            logger.warning(f"Overwriting existing retriever adapter: {name}")
        _RETRIEVER_REGISTRY[name] = cls
        logger.debug(f"Registered retriever adapter: {name}")
        return cls

    return decorator


def register_ingestor(name: str) -> Callable[[type[BaseIngestor]], type[BaseIngestor]]:
    """
    Decorator to register an ingestor adapter.

    Usage:
        @register_ingestor("llamaindex")
        class LlamaIndexIngestor(BaseIngestor):
            ...

    Args:
        name: The name to register this adapter under.

    Returns:
        Decorator function.
    """

    def decorator(cls: type[BaseIngestor]) -> type[BaseIngestor]:
        if name in _INGESTOR_REGISTRY:
            logger.warning(f"Overwriting existing ingestor adapter: {name}")
        _INGESTOR_REGISTRY[name] = cls
        logger.debug(f"Registered ingestor adapter: {name}")
        return cls

    return decorator


def get_retriever(
    backend: str | None = None,
    config: dict[str, Any] | None = None,
) -> BaseRetriever:
    """
    Factory function to get a configured retriever adapter.

    Configuration Precedence (highest to lowest):
        1. Explicit ``backend`` parameter passed to this function
        2. KNOWLEDGE_RETRIEVER_BACKEND environment variable
        3. Default: "llamaindex"

    Args:
        backend: The backend name ('llamaindex' or 'foundational_rag').
                 If None, uses the environment variable or default.
        config: Backend-specific configuration. Passed directly to the adapter.
                Each adapter defines its own defaults internally.

    Returns:
        Configured retriever adapter instance.

    Raises:
        ValueError: If backend is not registered.

    Example:
        >>> retriever = get_retriever("llamaindex", {"persist_dir": "/data/chroma"})
        >>> result = await retriever.retrieve("What is RAG?", "my_collection")
    """
    # Use configured default if not specified
    backend = backend or DEFAULT_RETRIEVER_BACKEND

    if backend not in _RETRIEVER_REGISTRY:
        available = list(_RETRIEVER_REGISTRY.keys())
        raise ValueError(f"Unknown retriever backend: {backend}. Available backends: {available}")

    # Pass config directly to adapter - each adapter handles its own defaults
    adapter_cls = _RETRIEVER_REGISTRY[backend]
    return adapter_cls(config=config or {})


def get_ingestor(
    backend: str | None = None,
    config: dict[str, Any] | None = None,
) -> BaseIngestor:
    """
    Factory function to get a configured ingestor adapter.

    Uses singleton pattern to preserve job state across requests.

    Configuration Precedence (highest to lowest):
        1. Explicit ``backend`` parameter passed to this function
        2. KNOWLEDGE_INGESTOR_BACKEND environment variable
        3. Default: "llamaindex"

    Args:
        backend: The backend name ('llamaindex' or 'foundational_rag').
                 If None, uses the environment variable or default.
        config: Backend-specific configuration. Passed directly to the adapter.
                Each adapter defines its own defaults internally.
                Note: config is only used on first instantiation (singleton).

    Returns:
        Configured ingestor adapter instance (singleton per backend).

    Raises:
        ValueError: If backend is not registered.

    Example:
        >>> ingestor = get_ingestor("llamaindex", {"persist_dir": "/data/chroma"})
        >>> job_id = ingestor.submit_job(["/path/to/file.pdf"], "my_collection")
    """
    # Use configured default if not specified
    backend = backend or DEFAULT_INGESTOR_BACKEND

    if backend not in _INGESTOR_REGISTRY:
        available = list(_INGESTOR_REGISTRY.keys())
        raise ValueError(f"Unknown ingestor backend: {backend}. Available backends: {available}")

    # Return cached instance if available (singleton pattern for job persistence)
    if backend in _INGESTOR_INSTANCES:
        if config:
            logger.debug(f"Returning cached {backend} ingestor (config parameter ignored)")
        return _INGESTOR_INSTANCES[backend]

    # Pass config directly to adapter - each adapter handles its own defaults
    adapter_cls = _INGESTOR_REGISTRY[backend]
    instance = adapter_cls(config=config or {})

    # Cache the instance
    _INGESTOR_INSTANCES[backend] = instance
    logger.info(f"Created singleton ingestor instance for backend: {backend}")

    return instance


def list_retrievers() -> list[str]:
    """Return list of registered retriever backends."""
    return list(_RETRIEVER_REGISTRY.keys())


def list_ingestors() -> list[str]:
    """Return list of registered ingestor backends."""
    return list(_INGESTOR_REGISTRY.keys())


def set_active_ingestor(ingestor: BaseIngestor) -> None:
    """
    Set the active ingestor for the Knowledge API.

    Called by the knowledge_retrieval function during initialization to make
    the configured ingestor available to API routes.

    Args:
        ingestor: The ingestor instance to activate.
    """
    global _ACTIVE_INGESTOR
    _ACTIVE_INGESTOR = ingestor
    logger.info("Set active ingestor: %s", ingestor.backend_name)


def get_active_ingestor() -> BaseIngestor | None:
    """
    Get the active ingestor for the Knowledge API.

    Returns:
        The active BaseIngestor instance, or None if not configured.
    """
    return _ACTIVE_INGESTOR


def clear_active_ingestor() -> None:
    """Clear the active ingestor (for testing)."""
    global _ACTIVE_INGESTOR
    _ACTIVE_INGESTOR = None


def release_ingestor(backend: str, ingestor: BaseIngestor) -> bool:
    """Remove one expected cached ingestor during a backend lifecycle finalizer.

    The identity check prevents an older finalizer from evicting a replacement
    instance created for the same backend.
    """
    if _INGESTOR_INSTANCES.get(backend) is not ingestor:
        return False
    del _INGESTOR_INSTANCES[backend]
    logger.info("Released singleton ingestor instance for backend: %s", backend)
    return True


# =============================================================================
# Summary Registry (SQLAlchemy-backed, Backend-Agnostic)
# =============================================================================
# Persistent storage for document summaries using configurable SQLite/PostgreSQL.
# Backends call register_summary() after ingestion; agents call
# get_available_documents() for prompt context.

_summary_store: "SummaryStore | None" = None

# Default DB URL (used if configure_summary_db not called)
_DEFAULT_SUMMARY_DB = "sqlite+aiosqlite:///./summaries.db"


def configure_summary_db(db_url: str) -> None:
    """Initialize summary store with given DB URL."""
    global _summary_store
    from .summary_store import SummaryStore
    from .summary_store import _redact_db_url

    _summary_store = SummaryStore(db_url)
    logger.info("Summary store configured: %s", _redact_db_url(db_url))


def _get_summary_store() -> "SummaryStore":
    """Get or create the summary store (lazy init with default)."""
    global _summary_store
    if _summary_store is None:
        from .summary_store import SummaryStore

        _summary_store = SummaryStore(_DEFAULT_SUMMARY_DB)
        logger.info("Summary store initialized with default: %s", _DEFAULT_SUMMARY_DB)
    return _summary_store


def register_summary(collection: str, filename: str, summary: str | None, upsert: bool = True) -> None:
    """Store a summary, optionally preserving an existing filename's summary."""
    if not summary:
        return
    _get_summary_store().register(collection, filename, summary, upsert=upsert)


def get_available_documents(collection: str) -> list["AvailableDocument"]:
    """Get documents with summaries (sync)."""
    return _get_summary_store().get_all(collection)


async def get_available_documents_async(collection: str) -> list["AvailableDocument"]:
    """Get documents with summaries (async)."""
    return await _get_summary_store().get_all_async(collection)


def list_summary_collections() -> list[str]:
    """Get every collection that currently holds summaries."""
    return _get_summary_store().list_collections()


def unregister_summary(collection: str, filename: str) -> None:
    """Delete a file's summary."""
    _get_summary_store().unregister(collection, filename)


def clear_collection_summaries(collection: str) -> None:
    """Delete all summaries in a collection."""
    _get_summary_store().clear_collection(collection)


def clear_all_summaries() -> None:
    """Delete all summaries."""
    _get_summary_store().clear_all()


def is_retriever_registered(name: str) -> bool:
    """Check if a retriever backend is registered."""
    return name in _RETRIEVER_REGISTRY


def is_ingestor_registered(name: str) -> bool:
    """Check if an ingestor backend is registered."""
    return name in _INGESTOR_REGISTRY


# =============================================================================
# Configuration Helpers
# =============================================================================


def get_default_retriever_backend() -> str:
    """Get the configured default retriever backend."""
    return DEFAULT_RETRIEVER_BACKEND


def get_default_ingestor_backend() -> str:
    """Get the configured default ingestor backend."""
    return DEFAULT_INGESTOR_BACKEND


def get_knowledge_layer_config() -> dict[str, Any]:
    """
    Get the complete knowledge layer configuration.

    Useful for debugging and displaying current setup.
    Note: Backend-specific config is defined in each adapter, not here.
    """
    return {
        "retriever": {
            "default_backend": DEFAULT_RETRIEVER_BACKEND,
            "available_backends": list(_RETRIEVER_REGISTRY.keys()),
        },
        "ingestor": {
            "default_backend": DEFAULT_INGESTOR_BACKEND,
            "available_backends": list(_INGESTOR_REGISTRY.keys()),
        },
    }
