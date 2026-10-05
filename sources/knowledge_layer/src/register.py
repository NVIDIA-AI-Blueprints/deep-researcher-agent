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

"""NAT function for knowledge retrieval.

This function provides direct library access to the knowledge layer,
allowing agents to search ingested documents without an external API server.

The retriever is instantiated once and reused for all queries.
"""

import logging
import os
from typing import Literal

from pydantic import Field
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic import model_validator

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import OptionalSecretStr
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


def _url_from_env(name: str, default: str | None = None) -> HttpUrl | None:
    value = os.environ.get(name) or default
    return HttpUrl(value) if value else None


def _secret_from_env(name: str) -> SecretStr | None:
    value = os.environ.get(name)
    return SecretStr(value) if value else None


# Type-safe backend selection - Pydantic validates at config load time
BackendType = Literal[
    "llamaindex",
    "foundational_rag",
    "opensearch",
    "azure_ai_search",
    "nemo_retriever",
    "nemo_retriever_local",
]
OpenSearchAuthType = Literal["none", "basic", "sigv4"]
OpenSearchAwsService = Literal["aoss", "es"]
OpenSearchIngestionMode = Literal["local", "dask", "auto"]
OpenSearchDaskFileTransfer = Literal["bytes", "paths"]


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None and value != "" else default


class KnowledgeRetrievalConfig(FunctionBaseConfig, name="knowledge_retrieval"):
    """Configuration for knowledge retrieval function."""

    backend: BackendType = Field(default="llamaindex", description="Knowledge backend to use")
    collection_name: str = Field(default="default", description="Name of the collection/index to search")
    top_k: int = Field(default=5, description="Number of results to return")
    backend_config: dict[str, object] = Field(
        default_factory=dict,
        description="Backend-owned configuration passed unchanged to the selected knowledge adapter.",
    )
    # Summarization options (applies to all backends)
    generate_summary: bool = Field(
        default=False, description="Generate one-sentence summary for each ingested document"
    )
    summary_model: str | None = Field(
        default=None,
        description="Required when generate_summary=true: LLM reference from llms: section",
    )
    summary_db: str = Field(
        default="sqlite+aiosqlite:///./summaries.db",
        description="Database URL for document summaries (SQLite or PostgreSQL)",
    )
    # LlamaIndex-specific options
    chroma_dir: str = Field(
        default="/tmp/chroma_data", description="Directory for ChromaDB persistence (LlamaIndex only)"
    )
    # Foundational RAG (hosted RAG Blueprint) options
    rag_url: str = Field(default="http://localhost:8081/v1", description="RAG query server URL (foundational_rag only)")
    ingest_url: str = Field(
        default="http://localhost:8082/v1", description="RAG ingestion server URL (foundational_rag only)"
    )
    timeout: int = Field(default=120, description="Request timeout in seconds (foundational_rag only)")
    verify_ssl: bool = Field(
        default=True, description="Verify SSL certificates (foundational_rag only). Set false for self-signed certs."
    )
    # OpenSearch-specific options
    opensearch_url: str = Field(
        default_factory=lambda: _env_value("OPENSEARCH_URL", default="http://localhost:9200"),
        description="OpenSearch endpoint URL (OpenSearch only).",
    )
    opensearch_auth_type: OpenSearchAuthType = Field(
        default_factory=lambda: _env_value("OPENSEARCH_AUTH_TYPE", default="none"),
        description="OpenSearch auth mode: none, basic, or sigv4.",
    )
    opensearch_username: str | None = Field(
        default_factory=lambda: _env_value("OPENSEARCH_USERNAME"),
        description="Username for OpenSearch basic auth. Falls back to OPENSEARCH_USERNAME.",
    )
    opensearch_password: SecretStr | None = Field(
        default_factory=lambda: SecretStr(pw) if (pw := _env_value("OPENSEARCH_PASSWORD")) is not None else None,
        description="Password for OpenSearch basic auth. Falls back to OPENSEARCH_PASSWORD.",
    )
    opensearch_verify_certs: bool = Field(
        default_factory=lambda: _env_bool("OPENSEARCH_VERIFY_CERTS", True),
        description="Verify OpenSearch TLS certificates. Set false only for trusted development clusters.",
    )
    opensearch_ca_certs: str | None = Field(
        default_factory=lambda: _env_value("OPENSEARCH_CA_CERTS"),
        description="Path to a custom CA bundle for OpenSearch TLS verification.",
    )
    opensearch_aws_region: str = Field(
        default_factory=lambda: _env_value("AWS_REGION", "AWS_DEFAULT_REGION", default="us-east-1"),
        description="AWS region for OpenSearch SigV4 auth.",
    )
    opensearch_aws_service: OpenSearchAwsService = Field(
        default_factory=lambda: _env_value("OPENSEARCH_AWS_SERVICE", default="aoss"),
        description="SigV4 service name: aoss for Amazon OpenSearch Serverless, es for Amazon OpenSearch Service.",
    )
    opensearch_index_prefix: str = Field(
        default_factory=lambda: _env_value("OPENSEARCH_INDEX_PREFIX", default="deep-researcher"),
        description="Prefix for OpenSearch collection indexes.",
    )
    opensearch_vector_field: str = Field(
        default_factory=lambda: _env_value("OPENSEARCH_VECTOR_FIELD", default="embedding"),
        description="Vector field name in OpenSearch documents.",
    )
    opensearch_text_field: str = Field(
        default_factory=lambda: _env_value("OPENSEARCH_TEXT_FIELD", default="content"),
        description="Text field name in OpenSearch documents.",
    )
    opensearch_embedding_dim: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_EMBEDDING_DIM", 2048),
        gt=0,
        description="Embedding vector dimension for OpenSearch knn_vector mappings.",
    )
    opensearch_engine: str = Field(
        default_factory=lambda: _env_value("OPENSEARCH_ENGINE", default="faiss"),
        description="OpenSearch k-NN engine.",
    )
    opensearch_space_type: str = Field(
        default_factory=lambda: _env_value("OPENSEARCH_SPACE_TYPE", default="cosinesimil"),
        description="OpenSearch k-NN space type.",
    )
    opensearch_m: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_M", 16),
        gt=0,
        description="HNSW m parameter for OpenSearch indexes.",
    )
    opensearch_ef_construction: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_EF_CONSTRUCTION", 512),
        gt=0,
        description="HNSW ef_construction parameter for OpenSearch indexes.",
    )
    opensearch_ef_search: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_EF_SEARCH", 512),
        gt=0,
        description="OpenSearch ef_search query parameter.",
    )
    opensearch_timeout: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_TIMEOUT", 120),
        gt=0,
        description="OpenSearch request timeout in seconds.",
    )
    opensearch_max_retries: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_MAX_RETRIES", 3),
        ge=0,
        description="OpenSearch client max retries.",
    )
    opensearch_bulk_batch_size: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_BULK_BATCH_SIZE", 100),
        gt=0,
        description="Number of documents per OpenSearch bulk indexing request.",
    )
    opensearch_embedding_batch_size: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_EMBEDDING_BATCH_SIZE", 16),
        gt=0,
        description="Number of texts per embedding request for OpenSearch ingestion.",
    )
    opensearch_chunk_size: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_CHUNK_SIZE", 1024),
        gt=0,
        description="Approximate words per OpenSearch text chunk.",
    )
    opensearch_chunk_overlap: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_CHUNK_OVERLAP", 128),
        ge=0,
        description="Approximate overlapping words between OpenSearch text chunks.",
    )
    opensearch_allow_document_ids: bool | None = Field(
        default_factory=lambda: _env_optional_bool("OPENSEARCH_ALLOW_DOCUMENT_IDS"),
        description="Whether to set explicit document IDs in bulk index requests. Defaults off for AOSS.",
    )
    opensearch_bulk_refresh: bool | None = Field(
        default_factory=lambda: _env_optional_bool("OPENSEARCH_BULK_REFRESH"),
        description="Refresh policy for OpenSearch bulk writes. Defaults off for AOSS.",
    )
    opensearch_aoss_delete_max_batches: int = Field(
        default_factory=lambda: _env_int("OPENSEARCH_AOSS_DELETE_MAX_BATCHES", 100),
        gt=0,
        description="Maximum search/delete batches for AOSS file deletion.",
    )
    opensearch_aoss_delete_backoff_seconds: float = Field(
        default_factory=lambda: _env_float("OPENSEARCH_AOSS_DELETE_BACKOFF_SECONDS", 0.25),
        ge=0,
        description="Backoff between AOSS delete batches to account for eventual search visibility.",
    )
    opensearch_ingestion_mode: OpenSearchIngestionMode = Field(
        default_factory=lambda: _env_value("OPENSEARCH_INGESTION_MODE", default="local"),
        description="OpenSearch ingestion execution mode: local, dask, or auto.",
    )
    opensearch_dask_scheduler_address: str | None = Field(
        default_factory=lambda: _env_value("OPENSEARCH_DASK_SCHEDULER_ADDRESS", "NAT_DASK_SCHEDULER_ADDRESS"),
        description="Dask scheduler address for OpenSearch distributed ingestion.",
    )
    opensearch_dask_file_transfer: OpenSearchDaskFileTransfer = Field(
        default_factory=lambda: _env_value("OPENSEARCH_DASK_FILE_TRANSFER", default="bytes"),
        description="How Dask ingestion workers receive files: bytes or paths.",
    )
    embed_model: str = Field(
        default_factory=lambda: _env_value("DEEP_RESEARCHER_EMBED_MODEL", default="nvidia/nemotron-3-embed-1b"),
        description="Embedding model for OpenSearch and Azure AI Search ingestion and retrieval.",
    )
    embed_base_url: str = Field(
        default_factory=lambda: _env_value(
            "DEEP_RESEARCHER_EMBED_BASE_URL", default="https://integrate.api.nvidia.com/v1"
        ),
        description="OpenAI-compatible embeddings endpoint base URL.",
    )
    # Azure AI Search options
    azure_search_endpoint: HttpUrl | None = Field(
        default_factory=lambda: _url_from_env("AZURE_SEARCH_ENDPOINT"),
        description="Azure AI Search service URL; defaults to AZURE_SEARCH_ENDPOINT",
    )
    azure_search_api_key: OptionalSecretStr = Field(
        default_factory=lambda: _secret_from_env("AZURE_SEARCH_API_KEY"),
        description="Optional Azure AI Search admin key; defaults to AZURE_SEARCH_API_KEY",
    )
    azure_search_index_prefix: str = Field(
        default_factory=lambda: _env_value("DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX", default="deep-researcher"),
        min_length=1,
        description="Unique deployment namespace for the shared Deep Researcher Agent index",
    )
    embed_dim: int = Field(
        default_factory=lambda: _env_int("DEEP_RESEARCHER_EMBED_DIM", 2048),
        gt=0,
        description="Embedding dimensions; defaults to DEEP_RESEARCHER_EMBED_DIM and must match existing indexes",
    )

    @model_validator(mode="after")
    def validate_backend_config(self):
        """Validate and warn about unused backend-specific config options."""
        backend = self.backend.lower()

        # Validate summary configuration
        if self.generate_summary and not self.summary_model:
            raise ValueError(
                "generate_summary=true requires summary_model to be set. "
                "Configure summary_model to reference an LLM from the llms: section."
            )

        if backend == "llamaindex":
            # LlamaIndex uses chroma_dir, warn if RAG-specific options are set
            if self.rag_url != "http://localhost:8081/v1":
                logger.warning("rag_url is ignored for llamaindex backend")
            if self.ingest_url != "http://localhost:8082/v1":
                logger.warning("ingest_url is ignored for llamaindex backend")
            if self.opensearch_url != "http://localhost:9200":
                logger.warning("opensearch_url is ignored for llamaindex backend")

        elif backend == "foundational_rag":
            # Foundational RAG uses rag_url/ingest_url, warn if others are set
            if self.chroma_dir != "/tmp/chroma_data":
                logger.warning("chroma_dir is ignored for foundational_rag backend")
            if self.opensearch_url != "http://localhost:9200":
                logger.warning("opensearch_url is ignored for foundational_rag backend")
            if not self.verify_ssl:
                logger.warning("SSL verification disabled for foundational_rag. Use only in trusted environments.")

        elif backend == "opensearch":
            if self.chroma_dir != "/tmp/chroma_data":
                logger.warning("chroma_dir is ignored for opensearch backend")
            if self.rag_url != "http://localhost:8081/v1":
                logger.warning("rag_url is ignored for opensearch backend")
            if self.ingest_url != "http://localhost:8082/v1":
                logger.warning("ingest_url is ignored for opensearch backend")
            if self.opensearch_auth_type == "basic":
                has_username = self.opensearch_username or os.environ.get("OPENSEARCH_USERNAME")
                has_password = self.opensearch_password or os.environ.get("OPENSEARCH_PASSWORD")
                if not has_username or not has_password:
                    logger.warning(
                        "OpenSearch basic auth selected but username/password are not fully configured. "
                        "Set opensearch_username/opensearch_password or OPENSEARCH_USERNAME/OPENSEARCH_PASSWORD."
                    )
            if not self.opensearch_verify_certs:
                logger.warning("TLS verification disabled for opensearch. Use only in trusted environments.")
        elif backend == "azure_ai_search":
            if self.azure_search_endpoint is None:
                raise ValueError("azure_ai_search requires azure_search_endpoint")
        elif backend == "nemo_retriever":
            from knowledge_layer.nemo_retriever.adapter import normalize_backend_config

            self.backend_config = normalize_backend_config(self.backend_config)
        elif backend == "nemo_retriever_local":
            from knowledge_layer.nemo_retriever._local_client import normalize_backend_config

            self.backend_config = normalize_backend_config(self.backend_config)
        return self


def _setup_backend(config: KnowledgeRetrievalConfig, summary_llm_obj=None) -> tuple[str, dict]:
    """
    Import the backend adapter and build its configuration.

    Importing the adapter module triggers the @register_retriever/@register_ingestor
    decorators, which register the adapter classes with the factory.

    Args:
        config: Knowledge retrieval configuration
        summary_llm_obj: Optional resolved LLM object for summarization

    Returns:
        Tuple of (backend_name, backend_config_dict)
    """
    backend = config.backend.lower()

    # Summary config: LLM object if resolved, else adapters use default NVIDIA model
    summary_config = {
        "generate_summary": config.generate_summary,
        "summary_llm": summary_llm_obj,
    }

    if backend == "llamaindex":
        import knowledge_layer.llamaindex.adapter  # noqa: F401

        os.environ.setdefault("DEEP_RESEARCHER_CHROMA_DIR", config.chroma_dir)
        backend_config = {
            "persist_dir": config.chroma_dir,
            **summary_config,
        }

    elif backend == "foundational_rag":
        import knowledge_layer.foundational_rag.adapter  # noqa: F401

        backend_config = {
            "rag_url": config.rag_url,
            "ingest_url": config.ingest_url,
            "timeout": config.timeout,
            "verify_ssl": config.verify_ssl,
            **summary_config,
        }

    elif backend == "opensearch":
        import knowledge_layer.opensearch.adapter  # noqa: F401

        os.environ.setdefault("OPENSEARCH_URL", config.opensearch_url)
        backend_config = {
            "endpoint": config.opensearch_url,
            "auth_type": config.opensearch_auth_type,
            "username": config.opensearch_username,
            "password": (config.opensearch_password.get_secret_value() if config.opensearch_password else None),
            "verify_certs": config.opensearch_verify_certs,
            "ca_certs": config.opensearch_ca_certs,
            "aws_region": config.opensearch_aws_region,
            "aws_service": config.opensearch_aws_service,
            "index_prefix": config.opensearch_index_prefix,
            "vector_field": config.opensearch_vector_field,
            "text_field": config.opensearch_text_field,
            "embedding_dim": config.opensearch_embedding_dim,
            "engine": config.opensearch_engine,
            "space_type": config.opensearch_space_type,
            "m": config.opensearch_m,
            "ef_construction": config.opensearch_ef_construction,
            "ef_search": config.opensearch_ef_search,
            "timeout": config.opensearch_timeout,
            "max_retries": config.opensearch_max_retries,
            "bulk_batch_size": config.opensearch_bulk_batch_size,
            "embedding_batch_size": config.opensearch_embedding_batch_size,
            "chunk_size": config.opensearch_chunk_size,
            "chunk_overlap": config.opensearch_chunk_overlap,
            "allow_document_ids": config.opensearch_allow_document_ids,
            "bulk_refresh": config.opensearch_bulk_refresh,
            "aoss_delete_max_batches": config.opensearch_aoss_delete_max_batches,
            "aoss_delete_backoff_seconds": config.opensearch_aoss_delete_backoff_seconds,
            "ingestion_mode": config.opensearch_ingestion_mode,
            "dask_scheduler_address": config.opensearch_dask_scheduler_address,
            "dask_file_transfer": config.opensearch_dask_file_transfer,
            "embed_model": config.embed_model,
            "embed_base_url": config.embed_base_url,
            **summary_config,
        }

    elif backend == "azure_ai_search":
        import knowledge_layer.azure_ai_search.adapter  # noqa: F401

        backend_config = {
            "endpoint": str(config.azure_search_endpoint),
            "api_key": config.azure_search_api_key,
            "index_prefix": config.azure_search_index_prefix,
            "embed_base_url": str(config.embed_base_url),
            "embed_model": config.embed_model,
            "embed_dim": config.embed_dim,
            "collection_name": config.collection_name,
            "cleanup_files": False,
            **summary_config,
        }

    elif backend == "nemo_retriever":
        import knowledge_layer.nemo_retriever.adapter  # noqa: F401

        backend_config = {
            **config.backend_config,
            **summary_config,
        }

    elif backend == "nemo_retriever_local":
        import knowledge_layer.nemo_retriever.local_adapter  # noqa: F401

        backend_config = {
            **config.backend_config,
            **summary_config,
        }

    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'llamaindex', 'foundational_rag', 'opensearch', "
            "'azure_ai_search', 'nemo_retriever', or 'nemo_retriever_local'."
        )

    os.environ["KNOWLEDGE_RETRIEVER_BACKEND"] = backend
    os.environ["KNOWLEDGE_INGESTOR_BACKEND"] = backend

    return backend, backend_config


def _get_retriever(config: KnowledgeRetrievalConfig):
    """Get the retriever singleton from the factory."""
    from deep_researcher_agent.knowledge.factory import get_retriever

    backend, backend_config = _setup_backend(config)
    retriever = get_retriever(backend, backend_config)
    logger.info("Initialized %s retriever", backend)
    return retriever


def _initialize_ingestor(config: KnowledgeRetrievalConfig, summary_llm_obj=None):
    """
    Initialize and activate the ingestor for the Knowledge API.

    Called during function registration to:
    1. Create the ingestor singleton via the factory
    2. Set it as the active ingestor for API routes to use

    Args:
        config: Knowledge retrieval configuration
        summary_llm_obj: Optional resolved LLM object for summarization
    """
    from deep_researcher_agent.knowledge.factory import get_ingestor
    from deep_researcher_agent.knowledge.factory import set_active_ingestor

    backend, backend_config = _setup_backend(config, summary_llm_obj)
    ingestor = get_ingestor(backend, backend_config)
    set_active_ingestor(ingestor)
    logger.info("Activated %s ingestor for Knowledge API", backend)
    return ingestor


def _format_results(retrieval_result, query: str) -> str:
    """
    Format retrieval results for LLM consumption.

    Returns a structured string that provides context for the agent.
    The format includes explicit citation fields so the LLM knows exactly
    what to use in its References section.
    """
    # Check for retrieval errors and surface them to the agent
    if not retrieval_result.success:
        error_msg = retrieval_result.error_message or "Unknown error"
        return f"Knowledge retrieval failed: {error_msg}\n\nQuery: '{query}'"

    if not retrieval_result.chunks:
        return f"No relevant documents found for query: '{query}'"

    lines = [f"Found {len(retrieval_result.chunks)} relevant document(s):\n"]

    for i, chunk in enumerate(retrieval_result.chunks, 1):
        # Build citation string: "filename, p.X" or just "filename"
        if chunk.page_number and chunk.page_number > 0:
            citation = f"{chunk.file_name}, p.{chunk.page_number}"
        else:
            citation = chunk.file_name

        # Header with source info
        lines.append(f"--- Result {i} ---")
        lines.append(f"Source: {chunk.file_name}")
        if chunk.page_number and chunk.page_number > 0:
            lines.append(f"Page: {chunk.page_number}")
        lines.append(f"Citation: {citation}")
        lines.append(f"Content Type: {chunk.content_type.value}")
        if chunk.distance is not None:
            lines.append(f"Vector Distance: {chunk.distance:.4g} (lower is closer)")
        else:
            lines.append(f"Relevance Score: {chunk.score:.2f}")
        lines.append("")

        # Content (truncate if very long)
        content = chunk.content
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


@register_function(config_type=KnowledgeRetrievalConfig)
async def knowledge_retrieval(config: KnowledgeRetrievalConfig, _builder: Builder):
    """
    Knowledge retrieval function for searching ingested documents.

    This function provides semantic search over documents that have been
    previously ingested into the knowledge layer. It supports multiple
    backends (LlamaIndex, Foundational RAG, OpenSearch, Azure AI Search) and returns formatted results
    suitable for LLM consumption.

    The retriever and ingestor are initialized once when the function is
    created and reused for all subsequent queries. The ingestor singleton
    is also made available to the Knowledge API routes via the factory.
    """
    # Resolve summary LLM if specified (enterprise approach)
    summary_llm_obj = None
    if config.summary_model and config.generate_summary:
        from nat.builder.framework_enum import LLMFrameworkEnum

        summary_llm_obj = await _builder.get_llm(config.summary_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        logger.info("Resolved summary model: %s", config.summary_model)

    # Initialize summary DB with configured URL
    from deep_researcher_agent.knowledge.factory import configure_summary_db

    configure_summary_db(config.summary_db)

    retriever = _get_retriever(config)
    try:
        ingestor = _initialize_ingestor(config, summary_llm_obj)
    except Exception:
        # The local retriever acquires a reference to the locked embedded
        # runtime. If ingestor construction fails, NAT never reaches the yield
        # finalizer, so release that reference here before propagating startup.
        if config.backend == "nemo_retriever_local":
            close = getattr(retriever, "close", None)
            if callable(close):
                close()
        raise

    collection = config.collection_name
    top_k = config.top_k

    logger.info(
        "Knowledge retrieval initialized: backend=%s, collection=%s, top_k=%d", config.backend, collection, top_k
    )

    async def search(query: str) -> str:
        """Search for documents relevant to the query.

        Args:
            query (str): Natural language query describing what information you need.

        Returns:
            str: Formatted string containing relevant document excerpts with citations.
        """
        # Determine collection: prefer session context (UI) over config default
        try:
            ctx = Context.get()
            session_collection = ctx.conversation_id if ctx else None
            target_collection = session_collection or collection
        except Exception:
            target_collection = collection

        logger.info(f"Knowledge search: query='{query[:100]}...' collection={target_collection}")

        try:
            # Call the retriever
            result = await retriever.retrieve(
                query=query,
                collection_name=target_collection,
                top_k=top_k,
            )

            # Format for LLM
            formatted = _format_results(result, query)
            logger.info(f"Knowledge search returned {len(result.chunks)} chunks")
            # Debug: Log what we're returning to the LLM
            logger.debug(f"Formatted result for LLM:\n{formatted[:500]}...")
            return formatted

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"Error searching knowledge base: {str(e)}"

    try:
        # Yield the function info for NAT registration.
        yield FunctionInfo.from_fn(
            search,
            description=(
                "Search the knowledge base for relevant documents. "
                "Use this to find information from ingested PDFs, documents, and other files. "
                f"Returns up to {top_k} relevant excerpts with citations."
            ),
        )
    finally:
        if config.backend in {"nemo_retriever", "nemo_retriever_local"}:
            from deep_researcher_agent.knowledge.factory import clear_active_ingestor
            from deep_researcher_agent.knowledge.factory import get_active_ingestor
            from deep_researcher_agent.knowledge.factory import release_ingestor

            if get_active_ingestor() is ingestor:
                clear_active_ingestor()
            release_ingestor(config.backend, ingestor)
            for component in (retriever, ingestor):
                close = getattr(component, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        logger.warning(
                            "Failed to close NeMo Retriever component (backend=%s, component=%s, error_type=%s)",
                            config.backend,
                            type(component).__name__,
                            type(exc).__name__,
                        )
