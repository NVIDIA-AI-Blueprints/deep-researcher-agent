# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.core.exceptions import HttpResponseError
from azure.core.exceptions import ResourceNotFoundError
from azure.core.exceptions import ServiceRequestError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import HnswAlgorithmConfiguration
from azure.search.documents.indexes.models import HnswParameters
from azure.search.documents.indexes.models import SearchableField
from azure.search.documents.indexes.models import SearchField
from azure.search.documents.indexes.models import SearchFieldDataType
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes.models import SimpleField
from azure.search.documents.indexes.models import VectorSearch
from azure.search.documents.indexes.models import VectorSearchAlgorithmMetric
from azure.search.documents.indexes.models import VectorSearchProfile
from azure.search.documents.models import VectorizedQuery

from deep_researcher_agent.knowledge import BaseIngestor
from deep_researcher_agent.knowledge import BaseRetriever
from deep_researcher_agent.knowledge import Chunk
from deep_researcher_agent.knowledge import ContentType
from deep_researcher_agent.knowledge import FileProgress
from deep_researcher_agent.knowledge import IngestionJobStatus
from deep_researcher_agent.knowledge import JobState
from deep_researcher_agent.knowledge import RetrievalResult
from deep_researcher_agent.knowledge import clear_collection_summaries
from deep_researcher_agent.knowledge import register_ingestor
from deep_researcher_agent.knowledge import register_retriever
from deep_researcher_agent.knowledge import register_summary
from deep_researcher_agent.knowledge import unregister_summary
from deep_researcher_agent.knowledge.base import CollectionInfo
from deep_researcher_agent.knowledge.base import FileInfo
from deep_researcher_agent.knowledge.base import TTLCleanupMixin
from deep_researcher_agent.knowledge.schema import FileStatus

logger = logging.getLogger(__name__)

_BACKEND_NAME = "azure_ai_search"
_SCHEMA_VERSION = 1
_MARKER_PREFIX = "deep-researcher.azure_ai_search:"
_MARKER_MAX_CHARS = 4000
_MAX_INDEX_NAME_LENGTH = 128
_MAX_BATCH_ACTIONS = 1000
_MAX_BATCH_BYTES = 16 * 1024 * 1024
_PAGE_SIZE = 1000
_DELETE_ATTEMPTS = 3
_CONSISTENCY_ATTEMPTS = 20
_CONSISTENCY_DELAY_SECONDS = 0.25
_CONSISTENCY_STABLE_READS = 3
_CHUNK_SIZE = 1024
_CHUNK_OVERLAP = 128
_SUMMARY_MAX_CHARS = 1000
_RECORD_COLLECTION = "collection"
_RECORD_FILE = "file"
_RECORD_CHUNK = "chunk"
_COLLECTION_ACTIVE = "active"
_COLLECTION_DELETING = "deleting"

COLLECTION_TTL_HOURS = float(os.environ.get("DEEP_RESEARCHER_COLLECTION_TTL_HOURS", "24"))
TTL_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("DEEP_RESEARCHER_TTL_CLEANUP_INTERVAL_SECONDS", "3600"))


def _coerce_config(config: dict[str, Any] | None) -> SimpleNamespace:
    """Apply adapter defaults so direct factory usage matches YAML usage."""
    values: dict[str, Any] = {
        "endpoint": os.environ.get("AZURE_SEARCH_ENDPOINT"),
        "api_key": os.environ.get("AZURE_SEARCH_API_KEY"),
        "embed_base_url": os.environ.get("DEEP_RESEARCHER_EMBED_BASE_URL") or "https://integrate.api.nvidia.com/v1",
        "embed_model": os.environ.get("DEEP_RESEARCHER_EMBED_MODEL", "nvidia/nemotron-3-embed-1b"),
        "embed_dim": int(os.environ.get("DEEP_RESEARCHER_EMBED_DIM", "2048")),
        "collection_name": "default",
        "cleanup_files": False,
        "generate_summary": False,
        "summary_llm": None,
        "index_prefix": os.environ.get("DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX", "deep-researcher"),
        "start_ttl_cleanup": True,
    }
    values.update({key: value for key, value in (config or {}).items() if key in values})
    if not values["endpoint"]:
        raise ValueError("Azure AI Search configuration requires `endpoint`")
    return SimpleNamespace(**values)


def _build_search_credential(cfg: SimpleNamespace):
    """Pick the Azure SDK credential without exposing secret values."""
    api_key = _secret_value(cfg.api_key)
    return AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _sanitize_index_part(value: str, fallback: str = "default") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or fallback


def _index_name_for_config(prefix: str, embed_model: str, embed_dim: int) -> str:
    """Map one deployment and embedding schema to one physical Azure index."""
    suffix = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{_SCHEMA_VERSION}\0{prefix}\0{embed_model}\0{embed_dim}",
    ).hex[:12]
    tail = f"knowledge-v{_SCHEMA_VERSION}-{suffix}"
    available = _MAX_INDEX_NAME_LENGTH - len(tail) - 1
    prefix_part = _sanitize_index_part(prefix, "deep-researcher")[:available].rstrip("-") or "deep-researcher"
    return f"{prefix_part}-{tail}"


def _validate_index_name(name: str) -> None:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?", name) or "--" in name:
        raise ValueError(
            f"Invalid Azure AI Search index name {name!r}. Names must use lowercase letters, digits, and single "
            "hyphens; be 2-128 characters; and start and end with a letter or digit."
        )


def _encode_marker(marker: dict[str, Any]) -> str:
    encoded = _MARKER_PREFIX + json.dumps(marker, separators=(",", ":"), sort_keys=True)
    if len(encoded) > _MARKER_MAX_CHARS:
        raise ValueError(f"Azure AI Search ownership marker exceeds {_MARKER_MAX_CHARS} characters")
    return encoded


def _decode_marker(description: str | None) -> dict[str, Any] | None:
    if not description or not description.startswith(_MARKER_PREFIX):
        return None
    try:
        marker = json.loads(description[len(_MARKER_PREFIX) :])
    except (TypeError, ValueError):
        return None
    return marker if isinstance(marker, dict) else None


def _new_marker(cfg: SimpleNamespace) -> dict[str, Any]:
    marker = {
        "backend": _BACKEND_NAME,
        "schema_version": _SCHEMA_VERSION,
        "index_prefix": cfg.index_prefix,
        "embedding_model": cfg.embed_model,
        "embedding_dim": cfg.embed_dim,
        "created_at": _utc_now().isoformat(),
    }
    _encode_marker(marker)
    return marker


def _record_id(record_type: str, collection_name: str, file_id: str | None = None) -> str:
    value = f"{record_type}\0{collection_name}\0{file_id or ''}"
    return f"{record_type}-{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


def _chunk_id(file_id: str, index: int) -> str:
    return f"chunk-{file_id}-{index:08d}"


def _odata_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _and_filter(*filters: str | None) -> str | None:
    values = [f"({value})" for value in filters if value]
    return " and ".join(values) or None


def _record_filter(
    record_type: str,
    collection_name: str | None = None,
    *,
    file_id: str | None = None,
    file_name: str | None = None,
    status: str | None = None,
) -> str:
    return (
        _and_filter(
            f"record_type eq {_odata_literal(record_type)}",
            f"collection_id eq {_odata_literal(collection_name)}" if collection_name is not None else None,
            f"file_id eq {_odata_literal(file_id)}" if file_id is not None else None,
            f"file_name eq {_odata_literal(file_name)}" if file_name is not None else None,
            f"status eq {_odata_literal(status)}" if status is not None else None,
        )
        or ""
    )


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed Azure AI Search metadata JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _iter_index_batches(
    documents: list[dict[str, Any]],
    action: str,
    max_actions: int = _MAX_BATCH_ACTIONS,
    max_bytes: int = _MAX_BATCH_BYTES,
) -> Iterator[list[dict[str, Any]]]:
    """Batch actions below both Azure's count and serialized payload limits."""
    batch: list[dict[str, Any]] = []
    batch_bytes = len(b'{"value":[]}')
    for document in documents:
        payload = {"@search.action": action, **document}
        action_bytes = (
            len(json.dumps(payload, default=_json_default, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            + 1
        )
        if action_bytes + len(b'{"value":[]}') > max_bytes:
            raise ValueError(f"Azure AI Search {action} action exceeds the 16 MiB request limit")
        if batch and (len(batch) >= max_actions or batch_bytes + action_bytes > max_bytes):
            yield batch
            batch = []
            batch_bytes = len(b'{"value":[]}')
        batch.append(document)
        batch_bytes += action_bytes
    if batch:
        yield batch


def _indexing_outcome(results: list[Any], expected: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    expected_keys = [str(document["id"]) for document in expected]
    by_key = {str(getattr(result, "key", "")): result for result in results}
    succeeded: list[str] = []
    failures: list[str] = []
    for key in expected_keys:
        result = by_key.get(key)
        if result is not None and getattr(result, "succeeded", False):
            succeeded.append(key)
        else:
            message = getattr(result, "error_message", None) if result is not None else "missing result"
            failures.append(f"{key}: {message or 'rejected'}")
    return succeeded, failures


def _build_index_schema(name: str, embed_dim: int, description: str | None = None) -> SearchIndex:
    return SearchIndex(
        name=name,
        description=description,
        fields=[
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="record_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
            SimpleField(name="collection_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="chunk", analyzer_name="standard.lucene"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embed_dim,
                vector_search_profile_name="hnsw-profile",
            ),
            SimpleField(name="file_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="file_name", filterable=True, sortable=True),
            SimpleField(name="status", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="error_message", type=SearchFieldDataType.String),
            SimpleField(name="description", type=SearchFieldDataType.String),
            SimpleField(name="summary", type=SearchFieldDataType.String),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="chunk_count", type=SearchFieldDataType.Int32),
            SimpleField(name="file_size", type=SearchFieldDataType.Int64, filterable=True),
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="updated_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="uploaded_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="ingested_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="metadata", type=SearchFieldDataType.String),
        ],
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-default",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
            ],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-default")],
        ),
    )


def _validate_index_schema(index: SearchIndex, cfg: SimpleNamespace) -> dict[str, Any]:
    marker = _decode_marker(index.description)
    if marker is None:
        raise RuntimeError(f"Azure AI Search index {index.name!r} is not owned by Deep Researcher Agent")
    if marker.get("backend") != _BACKEND_NAME or marker.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(
            f"Azure AI Search index {index.name!r} has an incompatible Deep Researcher Agent ownership marker"
        )
    if (
        marker.get("index_prefix") != cfg.index_prefix
        or marker.get("embedding_dim") != cfg.embed_dim
        or marker.get("embedding_model") != cfg.embed_model
    ):
        raise RuntimeError(f"Azure AI Search index {index.name!r} ownership or embedding configuration does not match")

    fields = {field.name: field for field in index.fields}
    required_types = {
        "id": SearchFieldDataType.String,
        "record_type": SearchFieldDataType.String,
        "collection_id": SearchFieldDataType.String,
        "chunk": SearchFieldDataType.String,
        "embedding": SearchFieldDataType.Collection(SearchFieldDataType.Single),
        "file_id": SearchFieldDataType.String,
        "file_name": SearchFieldDataType.String,
        "status": SearchFieldDataType.String,
        "error_message": SearchFieldDataType.String,
        "description": SearchFieldDataType.String,
        "summary": SearchFieldDataType.String,
        "page_number": SearchFieldDataType.Int32,
        "chunk_index": SearchFieldDataType.Int32,
        "chunk_count": SearchFieldDataType.Int32,
        "file_size": SearchFieldDataType.Int64,
        "created_at": SearchFieldDataType.DateTimeOffset,
        "updated_at": SearchFieldDataType.DateTimeOffset,
        "uploaded_at": SearchFieldDataType.DateTimeOffset,
        "ingested_at": SearchFieldDataType.DateTimeOffset,
        "metadata": SearchFieldDataType.String,
    }
    missing = [name for name in required_types if name not in fields]
    mismatched = [
        name for name, field_type in required_types.items() if name in fields and fields[name].type != field_type
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"Azure AI Search index {index.name!r} schema mismatch: missing={missing}, wrong_type={mismatched}"
        )
    if not fields["id"].key or not fields["id"].filterable or not fields["id"].sortable:
        raise RuntimeError(f"Azure AI Search index {index.name!r} requires id to be key/filterable/sortable")
    if not fields["record_type"].filterable or not fields["record_type"].facetable:
        raise RuntimeError(f"Azure AI Search index {index.name!r} requires facetable record_type")
    if not fields["collection_id"].filterable or not fields["file_id"].filterable:
        raise RuntimeError(f"Azure AI Search index {index.name!r} requires filterable identity fields")
    embedding = fields["embedding"]
    if embedding.vector_search_dimensions != cfg.embed_dim or embedding.vector_search_profile_name != "hnsw-profile":
        raise RuntimeError(f"Azure AI Search index {index.name!r} vector profile or dimensions do not match")
    profile_names = {profile.name for profile in (index.vector_search.profiles if index.vector_search else [])}
    if "hnsw-profile" not in profile_names:
        raise RuntimeError(f"Azure AI Search index {index.name!r} is missing hnsw-profile")
    return marker


class _AzureIndexMixin:
    cfg: SimpleNamespace
    _credential: Any
    _embedding: Any
    _index_client: SearchIndexClient
    _search_client: SearchClient | None
    _index_validated: bool

    def _initialize_azure(self, config: dict[str, Any]) -> None:
        self.cfg = _coerce_config(config)
        self._credential = _build_search_credential(self.cfg)
        self._index_client = SearchIndexClient(endpoint=str(self.cfg.endpoint), credential=self._credential)
        self._search_client = None
        self._index_validated = False
        self._embedding = None

    @property
    def embedding(self):
        if self._embedding is None:
            from llama_index.embeddings.nvidia import NVIDIAEmbedding

            self._embedding = NVIDIAEmbedding(
                model=self.cfg.embed_model,
                base_url=str(self.cfg.embed_base_url),
            )
        return self._embedding

    def _physical_index_name(self) -> str:
        name = _index_name_for_config(self.cfg.index_prefix, self.cfg.embed_model, self.cfg.embed_dim)
        _validate_index_name(name)
        return name

    def _get_search_client(self) -> SearchClient:
        if self._search_client is None:
            self._search_client = self._index_client.get_search_client(self._physical_index_name())
        return self._search_client

    def _get_owned_index(self) -> tuple[SearchIndex, dict[str, Any]]:
        index = self._index_client.get_index(self._physical_index_name())
        return index, _validate_index_schema(index, self.cfg)

    def _ensure_index(self) -> tuple[SearchIndex, dict[str, Any]]:
        try:
            index, marker = self._get_owned_index()
        except ResourceNotFoundError:
            marker = _new_marker(self.cfg)
            schema = _build_index_schema(self._physical_index_name(), self.cfg.embed_dim, _encode_marker(marker))
            try:
                index = self._index_client.create_index(schema)
            except Exception as create_error:  # noqa: BLE001
                try:
                    index = self._index_client.get_index(self._physical_index_name())
                except ResourceNotFoundError:
                    raise create_error
            marker = _validate_index_schema(index, self.cfg)
        self._index_validated = True
        return index, marker

    def _get_validated_search_client(self) -> SearchClient:
        if not self._index_validated:
            self._get_owned_index()
            self._index_validated = True
        return self._get_search_client()

    def _get_document(self, document_id: str) -> dict[str, Any] | None:
        try:
            return dict(self._get_validated_search_client().get_document(key=document_id))
        except ResourceNotFoundError:
            return None

    def _get_collection_manifest(self, collection_name: str) -> dict[str, Any] | None:
        document = self._get_document(_record_id(_RECORD_COLLECTION, collection_name))
        if (
            document
            and document.get("record_type") == _RECORD_COLLECTION
            and document.get("collection_id") == collection_name
        ):
            return document
        return None


@register_retriever(_BACKEND_NAME)
class AzureAISearchRetriever(_AzureIndexMixin, BaseRetriever):
    """Hybrid retriever backed by one owned Azure index."""

    backend_name = _BACKEND_NAME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._initialize_azure(self.config)

    def _get_client(self, collection_name: str) -> SearchClient:
        if self._get_collection_manifest(collection_name) is None:
            raise ResourceNotFoundError(f"Collection {collection_name!r} not found")
        return self._get_validated_search_client()

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        return await asyncio.to_thread(self._retrieve_sync, query, collection_name, top_k, filters)

    def _retrieve_sync(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> RetrievalResult:
        if filters:
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=[],
                success=False,
                error_message="Azure AI Search metadata filters are not supported",
            )
        try:
            client = self._get_client(collection_name)
            query_vector = self.embedding.get_query_embedding(query)
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max(top_k * 3, 20),
                fields="embedding",
            )
            search_params: dict[str, Any] = {
                "search_text": query,
                "vector_queries": [vector_query],
                "filter": _record_filter(_RECORD_CHUNK, collection_name),
                "top": top_k,
                "select": ["id", "chunk", "file_id", "file_name", "page_number", "metadata"],
            }
            chunks = [self.normalize(hit) for hit in client.search(**search_params)]
            return RetrievalResult(query=query, backend=_BACKEND_NAME, chunks=chunks, success=True)
        except ResourceNotFoundError:
            message = f"Deep Researcher Agent Azure AI Search collection {collection_name!r} not found"
        except ClientAuthenticationError as error:
            message = f"AI Search authentication failed: {error!s}"
        except ServiceRequestError as error:
            message = f"AI Search service unavailable: {error!s}"
        except HttpResponseError as error:
            message = f"AI Search request failed: {error.status_code} {error.reason or error!s}"
        except Exception as error:  # noqa: BLE001
            message = f"Unexpected error during retrieval: {error!s}"
            logger.exception("Azure AI Search retrieval failed")
        return RetrievalResult(query=query, backend=_BACKEND_NAME, chunks=[], success=False, error_message=message)

    def normalize(self, raw_result: Any) -> Chunk:
        chunk_id = raw_result.get("id") or str(uuid.uuid4())
        content = raw_result.get("chunk") or ""
        file_name = raw_result.get("file_name") or "unknown"
        page_number = _coerce_page_number(raw_result.get("page_number"))
        search_score = raw_result.get("@search.score") or 0.0
        score = float(search_score)
        score = min(max(score, 0.0), 1.0)
        metadata = _parse_metadata(raw_result.get("metadata"))
        if file_id := raw_result.get("file_id"):
            metadata["file_id"] = file_id
        return Chunk(
            chunk_id=str(chunk_id),
            content=str(content),
            score=score,
            file_name=str(file_name),
            page_number=page_number,
            display_citation=f"{file_name}, p.{page_number}" if page_number else str(file_name),
            content_type=ContentType.TEXT,
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        def _check() -> bool:
            list(self._index_client.list_index_names())
            return True

        try:
            return await asyncio.to_thread(_check)
        except Exception:  # noqa: BLE001
            logger.exception("Azure AI Search retriever health_check failed")
            return False


@register_ingestor(_BACKEND_NAME)
class AzureAISearchIngestor(TTLCleanupMixin, _AzureIndexMixin, BaseIngestor):
    """Parse, embed, and persist documents in owned Azure AI Search indexes."""

    backend_name = _BACKEND_NAME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._initialize_azure(self.config)
        self._splitter = None
        self._summary_llm = self.cfg.summary_llm
        self._jobs_lock = threading.RLock()
        self._jobs: dict[str, IngestionJobStatus] = {}
        self._files: dict[str, FileInfo] = {}
        self._deleted_collections: set[str] = set()
        self._deleted_files: set[tuple[str, str]] = set()
        if self.cfg.start_ttl_cleanup:
            self._start_ttl_cleanup_task(COLLECTION_TTL_HOURS, TTL_CLEANUP_INTERVAL_SECONDS)

    @property
    def splitter(self):
        if self._splitter is None:
            from llama_index.core.node_parser import SentenceSplitter

            self._splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
        return self._splitter

    def _write_document(self, document: dict[str, Any]) -> None:
        self._upload_documents(self._get_validated_search_client(), [document])

    def _wait_for_search_state(
        self,
        filter_text: str,
        *,
        present: bool,
        label: str,
        stable_reads: int = 1,
    ) -> None:
        client = self._get_validated_search_client()
        matching_reads = 0
        for attempt in range(_CONSISTENCY_ATTEMPTS):
            found = bool(list(client.search(search_text="*", filter=filter_text, select=["id"], top=1)))
            if found == present:
                matching_reads += 1
                if matching_reads >= stable_reads:
                    return
            else:
                matching_reads = 0
            if attempt + 1 < _CONSISTENCY_ATTEMPTS:
                time.sleep(_CONSISTENCY_DELAY_SECONDS)
        state = "visible" if present else "absent"
        raise RuntimeError(f"Timed out waiting for {label} to become {state} in Azure AI Search")

    def _wait_for_search_count(
        self,
        filter_text: str,
        *,
        expected_count: int,
        label: str,
        stable_reads: int = 1,
    ) -> None:
        client = self._get_validated_search_client()
        visible_count = 0
        matching_reads = 0
        for attempt in range(_CONSISTENCY_ATTEMPTS):
            visible_count = sum(1 for _ in self._iter_documents(client, filter_text=filter_text, select=["id"]))
            if visible_count == expected_count:
                matching_reads += 1
                if matching_reads >= stable_reads:
                    return
            else:
                matching_reads = 0
            if attempt + 1 < _CONSISTENCY_ATTEMPTS:
                time.sleep(_CONSISTENCY_DELAY_SECONDS)
        raise RuntimeError(
            f"Timed out waiting for {label} to expose {expected_count} document(s) in Azure AI Search; "
            f"found {visible_count}"
        )

    def _write_collection_manifest(
        self,
        collection_name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        status: str = _COLLECTION_ACTIVE,
    ) -> dict[str, Any]:
        self._ensure_index()
        existing = self._get_collection_manifest(collection_name) or {}
        if status == _COLLECTION_ACTIVE and existing.get("status") == _COLLECTION_DELETING:
            raise ValueError(f"Collection {collection_name!r} is being deleted")
        now = _utc_now()
        existing_metadata = _parse_metadata(existing.get("metadata"))
        merged_metadata = {**existing_metadata, **(metadata or {})}
        document = {
            "id": _record_id(_RECORD_COLLECTION, collection_name),
            "record_type": _RECORD_COLLECTION,
            "collection_id": collection_name,
            "status": status,
            "description": description if description is not None else existing.get("description"),
            "metadata": json.dumps(merged_metadata, separators=(",", ":"), sort_keys=True),
            "created_at": _parse_timestamp(existing.get("created_at")) or now,
            "updated_at": now,
        }
        self._write_document(document)
        if status == _COLLECTION_ACTIVE:
            self._deleted_collections.discard(collection_name)
        return document

    def _update_collection_timestamp(self, collection_name: str) -> None:
        manifest = self._get_collection_manifest(collection_name)
        if manifest is None or manifest.get("status") != _COLLECTION_ACTIVE:
            raise ResourceNotFoundError(f"Collection {collection_name!r} not found")
        manifest["updated_at"] = _utc_now()
        self._write_document(manifest)

    def _get_file_manifest(self, file_id: str, collection_name: str) -> dict[str, Any] | None:
        documents = list(
            self._get_validated_search_client().search(
                search_text="*",
                filter=_record_filter(_RECORD_FILE, collection_name, file_id=file_id),
                top=1,
            )
        )
        return dict(documents[0]) if documents else None

    def _write_file_manifest(self, info: FileInfo, *, summary: str | None = None) -> dict[str, Any]:
        metadata = {key: value for key, value in info.metadata.items() if key not in {"job_id", "summary"}}
        document = {
            "id": _record_id(_RECORD_FILE, info.collection_name, info.file_id),
            "record_type": _RECORD_FILE,
            "collection_id": info.collection_name,
            "file_id": info.file_id,
            "file_name": info.file_name,
            "status": info.status.value,
            "error_message": info.error_message,
            "summary": summary or info.metadata.get("summary"),
            "chunk_count": info.chunk_count,
            "file_size": info.file_size,
            "uploaded_at": info.uploaded_at,
            "ingested_at": info.ingested_at,
            "metadata": json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        }
        self._write_document(document)
        self._deleted_files.discard((info.collection_name, info.file_id))
        return document

    @staticmethod
    def _file_info_from_manifest(document: dict[str, Any]) -> FileInfo:
        metadata = _parse_metadata(document.get("metadata"))
        if summary := document.get("summary"):
            metadata["summary"] = summary
        try:
            status = FileStatus(document.get("status"))
        except ValueError:
            status = FileStatus.FAILED
        return FileInfo(
            file_id=str(document.get("file_id") or ""),
            file_name=str(document.get("file_name") or "unknown"),
            collection_name=str(document.get("collection_id") or ""),
            status=status,
            error_message=document.get("error_message"),
            file_size=document.get("file_size"),
            chunk_count=int(document.get("chunk_count") or 0),
            uploaded_at=_parse_timestamp(document.get("uploaded_at")),
            ingested_at=_parse_timestamp(document.get("ingested_at")),
            metadata=metadata,
        )

    def _collection_counts(self, collection_name: str) -> tuple[int, int]:
        results = self._get_validated_search_client().search(
            search_text="*",
            filter=f"collection_id eq {_odata_literal(collection_name)}",
            facets=["record_type,count:0"],
            top=0,
        )
        facets = results.get_facets() or {}
        counts = {str(item.get("value")): int(item.get("count") or 0) for item in facets.get("record_type", [])}
        return counts.get(_RECORD_FILE, 0), counts.get(_RECORD_CHUNK, 0)

    def _collection_info(self, manifest: dict[str, Any]) -> CollectionInfo:
        name = str(manifest["collection_id"])
        file_count, chunk_count = self._collection_counts(name)
        return CollectionInfo(
            name=name,
            description=manifest.get("description"),
            file_count=file_count,
            chunk_count=chunk_count,
            backend=_BACKEND_NAME,
            metadata={
                **_parse_metadata(manifest.get("metadata")),
                "index_name": self._physical_index_name(),
                "embedding_model": self.cfg.embed_model,
                "embedding_dim": self.cfg.embed_dim,
            },
            created_at=_parse_timestamp(manifest.get("created_at")),
            updated_at=_parse_timestamp(manifest.get("updated_at")),
        )

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        job_config = {**self.config, **(config or {})}
        original_filenames = _resolve_filenames(file_paths, job_config.get("original_filenames"))
        validated = [(path, original_filenames[index]) for index, path in enumerate(file_paths) if Path(path).is_file()]
        file_metadata = job_config.get("metadata") or {}
        json.dumps(file_metadata, separators=(",", ":"), sort_keys=True)

        if not validated:
            with self._jobs_lock:
                self._jobs[job_id] = IngestionJobStatus(
                    job_id=job_id,
                    status=JobState.FAILED,
                    collection_name=collection_name,
                    backend=_BACKEND_NAME,
                    submitted_at=_utc_now(),
                    completed_at=_utc_now(),
                    total_files=len(file_paths),
                    error_message="No valid file paths provided",
                )
            return job_id

        submitted_at = _utc_now()
        file_progress: list[FileProgress] = []
        files: dict[str, FileInfo] = {}
        for path, file_name in validated:
            file_id = str(uuid.uuid4())
            file_progress.append(FileProgress(file_id=file_id, file_name=file_name, status=FileStatus.UPLOADING))
            files[file_id] = FileInfo(
                file_id=file_id,
                file_name=file_name,
                collection_name=collection_name,
                status=FileStatus.UPLOADING,
                file_size=Path(path).stat().st_size,
                uploaded_at=submitted_at,
                metadata={**file_metadata, "job_id": job_id},
            )

        with self._jobs_lock:
            collection = self._get_collection_manifest(collection_name)
            if collection is not None and collection.get("status") != _COLLECTION_ACTIVE:
                raise ValueError(f"Collection {collection_name!r} is being deleted")
            self._files.update(files)
            self._jobs[job_id] = IngestionJobStatus(
                job_id=job_id,
                status=JobState.PENDING,
                collection_name=collection_name,
                backend=_BACKEND_NAME,
                submitted_at=submitted_at,
                total_files=len(validated),
                file_details=file_progress,
            )

        try:
            threading.Thread(
                target=self._process_job,
                args=(job_id, [path for path, _ in validated], collection_name, job_config),
                daemon=True,
                name=f"deep-researcher-azure-search-ingest-{job_id[:8]}",
            ).start()
        except Exception as error:  # noqa: BLE001
            message = f"Failed to start ingestion worker: {self._translate_error(error)}"
            self._fail_job_setup(job_id, message)
            if job_config.get("cleanup_files", self.cfg.cleanup_files):
                self._cleanup_paths([path for path, _ in validated])
            logger.exception("Failed to start Azure AI Search ingestion job %s", job_id)
        return job_id

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job.model_copy(deep=True)
        return IngestionJobStatus(
            job_id=job_id,
            status=JobState.FAILED,
            collection_name="",
            backend=_BACKEND_NAME,
            submitted_at=_utc_now(),
            completed_at=_utc_now(),
            error_message="Job not found",
        )

    def _wait_for_job_visibility(self, job_id: str) -> None:
        job = self.get_job_status(job_id)
        expected: dict[str, tuple[str, int]] = {}
        for detail in job.file_details:
            if detail.status not in {FileStatus.SUCCESS, FileStatus.FAILED}:
                raise RuntimeError(f"File {detail.file_id!r} did not reach a terminal ingestion state")
            expected[detail.file_id] = (detail.status.value, detail.chunks_created)

        client = self._get_validated_search_client()
        file_ids = ",".join(expected)
        filter_text = _and_filter(
            f"collection_id eq {_odata_literal(job.collection_name)}",
            f"search.in(file_id, {_odata_literal(file_ids)}, ',')",
        )
        matching_reads = 0
        for attempt in range(_CONSISTENCY_ATTEMPTS):
            manifests: dict[str, str] = {}
            chunk_counts: dict[str, int] = {}
            for document in self._iter_documents(
                client,
                filter_text=filter_text,
                select=["id", "record_type", "file_id", "status"],
            ):
                file_id = str(document.get("file_id") or "")
                if document.get("record_type") == _RECORD_FILE:
                    manifests[file_id] = str(document.get("status") or "")
                elif document.get("record_type") == _RECORD_CHUNK:
                    chunk_counts[file_id] = chunk_counts.get(file_id, 0) + 1
            if all(
                manifests.get(file_id) == status and chunk_counts.get(file_id, 0) == chunk_count
                for file_id, (status, chunk_count) in expected.items()
            ):
                matching_reads += 1
                if matching_reads >= _CONSISTENCY_STABLE_READS:
                    return
            else:
                matching_reads = 0
            if attempt + 1 < _CONSISTENCY_ATTEMPTS:
                time.sleep(_CONSISTENCY_DELAY_SECONDS)
        raise RuntimeError(f"Timed out waiting for job {job_id!r} manifests and chunks in Azure AI Search")

    def _process_job(
        self,
        job_id: str,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any],
    ) -> None:
        cleanup = bool(config.get("cleanup_files", self.cfg.cleanup_files))
        self._update_job(job_id, status=JobState.PROCESSING, started_at=_utc_now())
        client: SearchClient | None = None
        try:
            self._ensure_index()
            client = self._get_validated_search_client()
            collection = self._get_collection_manifest(collection_name)
            if collection is None:
                self._write_collection_manifest(collection_name)
            elif collection.get("status") != _COLLECTION_ACTIVE:
                raise ValueError(f"Collection {collection_name!r} is being deleted")

            job = self.get_job_status(job_id)
            for detail in job.file_details:
                self._write_file_manifest(self._files[detail.file_id])
        except Exception as error:  # noqa: BLE001
            message = self._translate_error(error)
            if client is not None:
                try:
                    job = self.get_job_status(job_id)
                    self._delete_document_ids(
                        client,
                        [_record_id(_RECORD_FILE, collection_name, detail.file_id) for detail in job.file_details],
                    )
                except Exception as rollback_error:  # noqa: BLE001
                    message = f"{message}; manifest rollback failed: {self._translate_error(rollback_error)}"
            self._fail_job_setup(job_id, message)
            if cleanup:
                self._cleanup_paths(file_paths)
            logger.exception("Failed to initialize Azure AI Search ingestion job %s", job_id)
            return

        failed = 0
        for index, path in enumerate(file_paths):
            job = self.get_job_status(job_id)
            detail = job.file_details[index]
            tracked = self._files[detail.file_id]
            self._update_file_progress(job_id, index, status=FileStatus.INGESTING)
            chunk_count = 0
            try:
                chunk_count, summary, ingested_at = self._process_file(
                    path=path,
                    collection_name=collection_name,
                    file_id=detail.file_id,
                    file_name=detail.file_name,
                    file_size=tracked.file_size or 0,
                    uploaded_at=tracked.uploaded_at or _utc_now(),
                    metadata={key: value for key, value in tracked.metadata.items() if key != "job_id"},
                )
                self._update_file_progress(
                    job_id,
                    index,
                    status=FileStatus.SUCCESS,
                    progress_percent=100.0,
                    chunks_created=chunk_count,
                    summary=summary,
                    ingested_at=ingested_at,
                )
            except Exception as error:  # noqa: BLE001
                failed += 1
                message = self._translate_error(error)
                if chunk_count:
                    try:
                        self._delete_file_documents(detail.file_id, collection_name, chunk_count)
                    except Exception as rollback_error:  # noqa: BLE001
                        message = f"{message}; chunk rollback failed: {self._translate_error(rollback_error)}"
                self._update_file_progress(
                    job_id,
                    index,
                    status=FileStatus.FAILED,
                    progress_percent=0.0,
                    chunks_created=0,
                    error_message=message,
                )
                logger.exception("Failed to ingest %s", detail.file_name)
            finally:
                if cleanup:
                    self._cleanup_paths([path])
                self._update_job(job_id, processed_files=index + 1)

        try:
            self._wait_for_job_visibility(job_id)
        except Exception as error:  # noqa: BLE001
            message = f"Failed to finalize ingestion: {self._translate_error(error)}"
            for index, detail in enumerate(self.get_job_status(job_id).file_details):
                if detail.status != FileStatus.SUCCESS:
                    continue
                try:
                    self.delete_file(detail.file_id, collection_name)
                except Exception as rollback_error:  # noqa: BLE001
                    message = f"{message}; rollback failed: {self._translate_error(rollback_error)}"
                try:
                    self._update_file_progress(
                        job_id,
                        index,
                        status=FileStatus.FAILED,
                        progress_percent=0.0,
                        chunks_created=0,
                        error_message=message,
                    )
                except Exception as rollback_error:  # noqa: BLE001
                    message = f"{message}; failed-status update failed: {self._translate_error(rollback_error)}"
            self._fail_job(job_id, message)
            logger.exception("Failed to finalize Azure AI Search ingestion job %s", job_id)
        else:
            if failed == len(file_paths):
                self._fail_job(job_id, f"All {failed} file(s) failed to ingest")
            else:
                self._update_job(job_id, status=JobState.COMPLETED, completed_at=_utc_now())

    def _process_file(
        self,
        *,
        path: str,
        collection_name: str,
        file_id: str,
        file_name: str,
        file_size: int,
        uploaded_at: datetime,
        metadata: dict[str, Any],
    ) -> tuple[int, str | None, datetime]:
        from llama_index.core import SimpleDirectoryReader

        documents = SimpleDirectoryReader(input_files=[path]).load_data()
        if not documents:
            raise ValueError(f"No content extracted from {file_name}")
        nodes = self.splitter.get_nodes_from_documents(documents)
        if not nodes:
            raise ValueError(f"Chunking produced 0 chunks for {file_name}")
        texts = [node.get_content() for node in nodes]
        embeddings = self.embedding.get_text_embedding_batch(texts)
        if any(len(vector) != self.cfg.embed_dim for vector in embeddings):
            raise ValueError(f"Embedding dimensions do not match configured embed_dim={self.cfg.embed_dim}")

        ingested_at = _utc_now()
        encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        search_documents: list[dict[str, Any]] = []
        for chunk_index, (node, vector) in enumerate(zip(nodes, embeddings, strict=True)):
            search_documents.append(
                {
                    "id": _chunk_id(file_id, chunk_index),
                    "record_type": _RECORD_CHUNK,
                    "collection_id": collection_name,
                    "chunk": node.get_content(),
                    "embedding": list(vector),
                    "file_id": file_id,
                    "file_name": file_name,
                    "page_number": _coerce_page_number(node.metadata.get("page_label")),
                    "chunk_index": chunk_index,
                    "file_size": file_size,
                    "uploaded_at": uploaded_at,
                    "ingested_at": ingested_at,
                    "metadata": encoded_metadata,
                }
            )

        chunk_ids = [str(document["id"]) for document in search_documents]
        client = self._get_validated_search_client()
        try:
            self._upload_documents(client, search_documents)
            collection = self._get_collection_manifest(collection_name)
            if collection is None or collection.get("status") != _COLLECTION_ACTIVE:
                raise RuntimeError(f"Collection {collection_name!r} became unavailable during ingestion")

            summary = self._generate_summary("\n".join(texts), file_name) if self.cfg.generate_summary else None
            self._update_collection_timestamp(collection_name)
        except Exception as processing_error:  # noqa: BLE001
            try:
                self._delete_document_ids(client, chunk_ids)
            except Exception as rollback_error:  # noqa: BLE001
                raise RuntimeError(
                    f"{processing_error}; chunk rollback failed: {self._translate_error(rollback_error)}"
                ) from processing_error
            raise
        return len(search_documents), summary, ingested_at

    def _upload_documents(self, client: SearchClient, documents: list[dict[str, Any]]) -> None:
        uploaded_ids: list[str] = []
        try:
            for batch in _iter_index_batches(documents, "upload"):
                results = client.upload_documents(documents=batch)
                succeeded, failures = _indexing_outcome(results, batch)
                uploaded_ids.extend(succeeded)
                if failures:
                    raise RuntimeError(f"Azure AI Search rejected upload actions: {'; '.join(failures)}")
        except Exception as upload_error:  # noqa: BLE001
            if uploaded_ids:
                try:
                    self._delete_document_ids(client, uploaded_ids)
                except Exception as rollback_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"Upload failed ({upload_error}); rollback failed ({rollback_error})"
                    ) from upload_error
            raise

    def _delete_document_ids(self, client: SearchClient, document_ids: list[str]) -> None:
        for batch in _iter_index_batches([{"id": item} for item in document_ids], "delete"):
            pending = batch
            failures: list[str] = []
            for _attempt in range(_DELETE_ATTEMPTS):
                results = client.delete_documents(documents=pending)
                _succeeded, failures = _indexing_outcome(results, pending)
                if not failures:
                    break
                failed_keys = {failure.split(":", 1)[0] for failure in failures}
                pending = [document for document in pending if str(document["id"]) in failed_keys]
            if failures:
                raise RuntimeError(f"Azure AI Search rejected delete actions: {'; '.join(failures)}")

    def _iter_documents(
        self,
        client: SearchClient,
        *,
        filter_text: str | None = None,
        select: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        last_id: str | None = None
        while True:
            cursor_filter = f"id gt {_odata_literal(last_id)}" if last_id is not None else None
            page = list(
                client.search(
                    search_text="*",
                    filter=_and_filter(filter_text, cursor_filter),
                    select=select,
                    order_by=["id asc"],
                    top=_PAGE_SIZE,
                )
            )
            if not page:
                return
            yield from page
            next_id = str(page[-1].get("id") or "")
            if len(page) < _PAGE_SIZE:
                return
            if not next_id or next_id == last_id:
                raise RuntimeError("Azure AI Search pagination did not advance")
            last_id = next_id

    def _stable_documents(
        self,
        client: SearchClient,
        *,
        filter_text: str,
        select: list[str] | None = None,
        label: str,
    ) -> list[dict[str, Any]]:
        observed_documents: dict[str, dict[str, Any]] = {}
        previous_ids: set[str] | None = None
        matching_reads = 0
        for attempt in range(_CONSISTENCY_ATTEMPTS):
            documents = list(self._iter_documents(client, filter_text=filter_text, select=select))
            document_ids = {str(document["id"]) for document in documents if document.get("id")}
            observed_documents.update((str(document["id"]), document) for document in documents if document.get("id"))
            if document_ids == previous_ids:
                matching_reads += 1
            else:
                previous_ids = document_ids
                matching_reads = 1
            if attempt + 1 < _CONSISTENCY_ATTEMPTS:
                time.sleep(_CONSISTENCY_DELAY_SECONDS)
        if matching_reads < _CONSISTENCY_STABLE_READS:
            raise RuntimeError(f"Timed out waiting for stable {label} in Azure AI Search")
        return list(observed_documents.values())

    def _delete_file_documents(
        self,
        file_id: str,
        collection_name: str,
        expected_count: int | None = None,
    ) -> int:
        client = self._get_validated_search_client()
        if expected_count is None:
            filter_text = _record_filter(_RECORD_CHUNK, collection_name, file_id=file_id)
            ids = [
                str(hit["id"])
                for hit in self._iter_documents(client, filter_text=filter_text, select=["id"])
                if hit.get("id")
            ]
        else:
            ids = [_chunk_id(file_id, index) for index in range(expected_count)]
        if ids:
            self._delete_document_ids(client, ids)
        return len(ids)

    def _generate_summary(self, text: str, file_name: str) -> str | None:
        if self._summary_llm is None:
            return None
        snippet = text[:_SUMMARY_MAX_CHARS]
        prompt = f"Summarise the following document ({file_name}) in one sentence (max 30 words):\n\n{snippet}"
        try:
            response = self._summary_llm.invoke(prompt)
            summary = getattr(response, "content", None) or str(response)
            return summary.strip() or None
        except Exception:  # noqa: BLE001
            logger.exception("Summary generation failed for %s", file_name)
            return None

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                self._jobs[job_id] = job.model_copy(update=fields)

    def _fail_job(self, job_id: str, error_message: str) -> None:
        self._update_job(job_id, status=JobState.FAILED, error_message=error_message, completed_at=_utc_now())

    def _fail_job_setup(self, job_id: str, error_message: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            details = [
                detail.model_copy(update={"status": FileStatus.FAILED, "error_message": error_message})
                for detail in job.file_details
            ]
            for detail in details:
                if tracked := self._files.get(detail.file_id):
                    tracked.status = FileStatus.FAILED
                    tracked.error_message = error_message
                self._deleted_files.add((job.collection_name, detail.file_id))
            self._jobs[job_id] = job.model_copy(
                update={
                    "status": JobState.FAILED,
                    "processed_files": job.total_files,
                    "file_details": details,
                    "error_message": error_message,
                    "completed_at": _utc_now(),
                }
            )

    def _update_file_progress(self, job_id: str, index: int, **fields: Any) -> None:
        summary = fields.pop("summary", None)
        ingested_at = fields.pop("ingested_at", None)
        snapshot: FileInfo | None = None
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or index >= len(job.file_details):
                return
            details = list(job.file_details)
            details[index] = details[index].model_copy(update=fields)
            tracked = self._files.get(details[index].file_id)
            if tracked:
                tracked.status = details[index].status
                tracked.error_message = details[index].error_message
                tracked.chunk_count = details[index].chunks_created
                if tracked.status == FileStatus.SUCCESS:
                    tracked.ingested_at = ingested_at or _utc_now()
                    if summary:
                        tracked.metadata["summary"] = summary
                else:
                    tracked.ingested_at = None
                    tracked.metadata.pop("summary", None)
                snapshot = tracked.model_copy(deep=True)
            self._jobs[job_id] = job.model_copy(update={"file_details": details})
        if snapshot and (collection := self._get_collection_manifest(snapshot.collection_name)):
            if collection.get("status") == _COLLECTION_ACTIVE:
                self._write_file_manifest(snapshot, summary=summary)
                if snapshot.status == FileStatus.SUCCESS and summary:
                    register_summary(snapshot.collection_name, snapshot.file_name, summary)

    @staticmethod
    def _cleanup_paths(paths: list[str]) -> None:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def _translate_error(error: Exception) -> str:
        if isinstance(error, ResourceNotFoundError):
            return f"AI Search index not found: {error!s}"
        if isinstance(error, ClientAuthenticationError):
            return f"AI Search authentication failed: {error!s}"
        if isinstance(error, ServiceRequestError):
            return f"AI Search service unavailable: {error!s}"
        if isinstance(error, HttpResponseError):
            return f"AI Search request failed ({error.status_code}): {error.reason or error!s}"
        return str(error)

    def create_collection(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        manifest = self._write_collection_manifest(name, description, metadata)
        self._wait_for_search_state(
            _record_filter(_RECORD_COLLECTION, name, status=_COLLECTION_ACTIVE),
            present=True,
            label=f"collection {name!r}",
        )
        return self._collection_info(manifest)

    def delete_collection(self, name: str) -> bool:
        manifest = self._get_collection_manifest(name)
        if manifest is None:
            return False
        client = self._get_validated_search_client()
        with self._jobs_lock:
            if any(
                info.collection_name == name
                and info.status in {FileStatus.UPLOADING, FileStatus.INGESTING}
                and (name, file_id) not in self._deleted_files
                for file_id, info in self._files.items()
            ):
                raise ValueError(f"Cannot delete collection {name!r} while files are ingesting")
            self._write_collection_manifest(name, status=_COLLECTION_DELETING)
            tracked_files = {
                file_id: info.model_copy(deep=True)
                for file_id, info in self._files.items()
                if info.collection_name == name
            }

        content_filter = _and_filter(
            f"collection_id eq {_odata_literal(name)}",
            f"record_type ne {_odata_literal(_RECORD_COLLECTION)}",
        )
        child_documents = self._stable_documents(
            client,
            filter_text=content_filter or "",
            select=[
                "id",
                "record_type",
                "file_id",
                "file_name",
                "status",
                "error_message",
                "summary",
                "chunk_count",
                "file_size",
                "uploaded_at",
                "ingested_at",
                "metadata",
            ],
            label=f"collection {name!r} child documents",
        )
        files = {
            str(document["file_id"]): self._file_info_from_manifest(document)
            for document in child_documents
            if document.get("record_type") == _RECORD_FILE and document.get("file_id")
        }
        files.update(tracked_files)
        if any(
            info.status in {FileStatus.UPLOADING, FileStatus.INGESTING}
            for file_id, info in files.items()
            if (name, file_id) not in self._deleted_files
        ):
            if manifest.get("status") == _COLLECTION_ACTIVE:
                manifest["updated_at"] = _utc_now()
                self._write_document(manifest)
            raise ValueError(f"Cannot delete collection {name!r} while files are ingesting")

        document_ids = {str(document["id"]) for document in child_documents if document.get("id")}
        for info in files.values():
            document_ids.add(_record_id(_RECORD_FILE, name, info.file_id))
            if info.status == FileStatus.SUCCESS:
                document_ids.update(_chunk_id(info.file_id, index) for index in range(info.chunk_count))
        if document_ids:
            self._delete_document_ids(client, sorted(document_ids))
        self._wait_for_search_count(
            content_filter or "",
            expected_count=0,
            label=f"collection {name!r} child documents",
            stable_reads=_CONSISTENCY_STABLE_READS,
        )
        self._delete_document_ids(client, [str(manifest["id"])])
        with self._jobs_lock:
            self._deleted_collections.add(name)
            self._files = {file_id: info for file_id, info in self._files.items() if info.collection_name != name}
        if self.cfg.generate_summary:
            clear_collection_summaries(name)
        return True

    def list_collections(self) -> list[CollectionInfo]:
        try:
            manifests = self._iter_documents(
                self._get_validated_search_client(),
                filter_text=_record_filter(_RECORD_COLLECTION, status=_COLLECTION_ACTIVE),
            )
            return [
                self._collection_info(manifest)
                for manifest in manifests
                if manifest.get("collection_id") not in self._deleted_collections
            ]
        except ResourceNotFoundError:
            return []

    def get_collection(self, name: str) -> CollectionInfo | None:
        if name in self._deleted_collections:
            return None
        manifest = self._get_collection_manifest(name)
        if manifest is None or manifest.get("status") != _COLLECTION_ACTIVE:
            return None
        return self._collection_info(manifest)

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        path = Path(file_path)
        if not path.is_file():
            return FileInfo(
                file_id=str(uuid.uuid4()),
                file_name=path.name,
                collection_name=collection_name,
                status=FileStatus.FAILED,
                error_message=f"File not found: {file_path}",
            )
        job_id = self.submit_job(
            [file_path],
            collection_name,
            {"original_filenames": [path.name], "metadata": metadata or {}},
        )
        with self._jobs_lock:
            file_id = self._jobs[job_id].file_details[0].file_id
            info = self._files[file_id].model_copy(deep=True)
            info.metadata["job_id"] = job_id
            return info

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        if (collection_name, file_id) in self._deleted_files:
            return False
        with self._jobs_lock:
            tracked = self._files.get(file_id)
            info = tracked.model_copy(deep=True) if tracked and tracked.collection_name == collection_name else None
        if info is None:
            info = self.get_file_status(file_id, collection_name)
        if info is None:
            return False
        if info.status in {FileStatus.UPLOADING, FileStatus.INGESTING}:
            raise ValueError(f"Cannot delete file {file_id!r} while it is ingesting")

        expected_count = info.chunk_count if info.status == FileStatus.SUCCESS and info.chunk_count > 0 else None
        self._delete_file_documents(file_id, collection_name, expected_count)
        self._delete_document_ids(
            self._get_validated_search_client(),
            [_record_id(_RECORD_FILE, collection_name, file_id)],
        )
        self._wait_for_search_count(
            _and_filter(
                f"collection_id eq {_odata_literal(collection_name)}",
                f"file_id eq {_odata_literal(file_id)}",
            )
            or "",
            expected_count=0,
            label=f"file {file_id!r} manifest and chunks",
            stable_reads=_CONSISTENCY_STABLE_READS,
        )

        if self.cfg.generate_summary:
            remaining = [
                item
                for item in self.list_files(collection_name)
                if item.file_name == info.file_name and item.status == FileStatus.SUCCESS
            ]
            newest = max(remaining, key=lambda item: item.ingested_at or datetime.min.replace(tzinfo=UTC), default=None)
            if newest and (summary := newest.metadata.get("summary")):
                register_summary(collection_name, info.file_name, str(summary))
            else:
                unregister_summary(collection_name, info.file_name)
        self._update_collection_timestamp(collection_name)
        with self._jobs_lock:
            self._deleted_files.add((collection_name, file_id))
            self._files.pop(file_id, None)
        return True

    def list_files(self, collection_name: str) -> list[FileInfo]:
        if collection_name in self._deleted_collections:
            return []
        try:
            collection = self._get_collection_manifest(collection_name)
            if collection is None:
                return []
            manifests = self._iter_documents(
                self._get_validated_search_client(),
                filter_text=_record_filter(_RECORD_FILE, collection_name),
            )
            files = [
                self._file_info_from_manifest(manifest)
                for manifest in manifests
                if (collection_name, str(manifest.get("file_id") or "")) not in self._deleted_files
            ]
        except ResourceNotFoundError:
            return []
        except Exception:  # noqa: BLE001
            logger.exception("Failed to list files for %r", collection_name)
            return []
        return sorted(files, key=lambda item: (item.file_name, item.file_id))

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        if (collection_name, file_id) in self._deleted_files:
            return None
        manifest = self._get_file_manifest(file_id, collection_name)
        return self._file_info_from_manifest(manifest) if manifest else None

    async def health_check(self) -> bool:
        def _check() -> bool:
            list(self._index_client.list_index_names())
            return True

        try:
            return await asyncio.to_thread(_check)
        except Exception:  # noqa: BLE001
            logger.exception("Azure AI Search ingestor health_check failed")
            return False


def _resolve_filenames(file_paths: list[str], raw: Any) -> list[str]:
    if isinstance(raw, dict):
        return [raw.get(path) or Path(path).name for path in file_paths]
    if isinstance(raw, list):
        return [
            raw[index] if index < len(raw) and raw[index] else Path(path).name for index, path in enumerate(file_paths)
        ]
    return [Path(path).name for path in file_paths]


def _coerce_page_number(page_label: Any) -> int | None:
    if page_label is None:
        return None
    try:
        page_number = int(str(page_label))
        return page_number if page_number > 0 else None
    except (TypeError, ValueError):
        return None
