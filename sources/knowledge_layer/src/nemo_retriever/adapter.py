# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Researcher Agent Knowledge Layer adapter for a separately deployed NeMo Retriever service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

from pydantic import SecretStr
from pydantic import ValidationError

from deep_researcher_agent.knowledge import BaseIngestor
from deep_researcher_agent.knowledge import BaseRetriever
from deep_researcher_agent.knowledge import Chunk
from deep_researcher_agent.knowledge import FileProgress
from deep_researcher_agent.knowledge import IngestionJobStatus
from deep_researcher_agent.knowledge import JobState
from deep_researcher_agent.knowledge import RetrievalResult
from deep_researcher_agent.knowledge import clear_collection_summaries
from deep_researcher_agent.knowledge import get_available_documents
from deep_researcher_agent.knowledge import list_summary_collections
from deep_researcher_agent.knowledge import register_ingestor
from deep_researcher_agent.knowledge import register_retriever
from deep_researcher_agent.knowledge import register_summary
from deep_researcher_agent.knowledge import unregister_summary
from deep_researcher_agent.knowledge.base import IngestionBatchTooLargeError
from deep_researcher_agent.knowledge.base import IngestionCapacityError
from deep_researcher_agent.knowledge.base import TTLCleanupMixin
from deep_researcher_agent.knowledge.schema import CollectionInfo
from deep_researcher_agent.knowledge.schema import FileInfo
from deep_researcher_agent.knowledge.schema import FileStatus

from ._models import CollectionDeleteWire
from ._models import CollectionPageWire
from ._models import CollectionWire
from ._models import DocumentDeleteWire
from ._models import DocumentPageWire
from ._models import DocumentWire
from ._models import JobAggregateWire
from ._models import JobCreatedWire
from ._models import JobDocumentsPageWire
from ._models import JobDocumentWire
from ._models import QueryHitWire
from ._models import QueryResponseWire
from ._models import UploadAcceptedWire
from ._normalization import UNSUPPORTED_FILTERS_ERROR
from ._normalization import normalize_query_hit
from ._normalization import scrub_metadata as _scrub_metadata
from ._normalization import status_to_file_status
from ._normalization import strict_bool
from ._transport import NemoRetrieverError
from ._transport import NemoRetrieverHTTPError
from ._transport import _NRLTransport

logger = logging.getLogger(__name__)

_BACKEND_NAME = "nemo_retriever"
_RESOURCE_PAGE_SIZE = 100
_JOB_PAGE_SIZE = 1000
_PUBLIC_DOCUMENT_ERROR = "NeMo Retriever document ingestion failed"
_SERVICE_CONFIG_KEYS = frozenset(
    {
        "base_url",
        "api_token",
        "scope",
        "connect_timeout_s",
        "request_timeout_s",
        "max_retries",
        "max_concurrency",
        "max_queued_uploads",
        "verify_ssl",
        "ca_bundle",
        "collection_ttl_hours",
    }
)
_SUCCESS_STATUSES = frozenset({"completed", "indexed", "ready", "success", "succeeded"})
_FAILED_STATUSES = frozenset({"failed", "error"})
_PLACEHOLDER_SUMMARY = "No Summary Available"
# Shared with the other knowledge backends so one setting paces every TTL cleanup thread.
TTL_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("DEEP_RESEARCHER_TTL_CLEANUP_INTERVAL_SECONDS", "3600"))


@dataclass(frozen=True)
class _Settings:
    base_url: str
    api_token: SecretStr | None
    scope: str
    connect_timeout_s: float
    request_timeout_s: float
    max_retries: int
    max_concurrency: int
    max_queued_uploads: int
    verify_ssl: bool
    ca_bundle: str | None
    collection_ttl_hours: float
    warm_start: bool
    start_ttl_cleanup: bool


@dataclass(frozen=True)
class _FileDescriptor:
    position: int
    path: Path
    filename: str
    file_size: int
    content_sha256: str
    manifest_entry_id: str


@dataclass
class _UploadBatchState:
    """Process-local state for multipart uploads not yet visible to NRL."""

    descriptors: tuple[_FileDescriptor, ...]
    accepted_by_position: dict[int, UploadAcceptedWire] = field(default_factory=dict)
    failed_by_position: dict[int, str] = field(default_factory=dict)
    remaining: int = 0
    done: threading.Event = field(default_factory=threading.Event)


_SHARED_TRANSPORTS: dict[tuple[Any, ...], _NRLTransport] = {}
_SHARED_TRANSPORTS_LOCK = threading.Lock()


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    normalized = str(value).strip()
    return normalized or None


def _config_value(config: dict[str, Any], key: str, env_name: str, default: Any = None) -> Any:
    """Resolve an adapter-owned option without centralizing backend fields."""
    if key in config and config[key] not in (None, ""):
        return config[key]
    value = os.environ.get(env_name)
    return value if value not in (None, "") else default


def _settings(config: dict[str, Any]) -> _Settings:
    base_url = str(_config_value(config, "base_url", "NRL_BASE_URL", "http://127.0.0.1:7670")).strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("nrl_base_url must be an absolute HTTP(S) URL")
    scope = str(_config_value(config, "scope", "NRL_SCOPE", "")).strip()
    if not scope:
        raise ValueError("nemo_retriever requires an explicit nrl_scope")
    connect_timeout_s = float(_config_value(config, "connect_timeout_s", "NRL_CONNECT_TIMEOUT_S", 30))
    request_timeout_s = float(_config_value(config, "request_timeout_s", "NRL_REQUEST_TIMEOUT_S", 300))
    max_retries = int(_config_value(config, "max_retries", "NRL_MAX_RETRIES", 5))
    max_concurrency = int(_config_value(config, "max_concurrency", "NRL_MAX_CONCURRENCY", 8))
    max_queued_uploads = int(_config_value(config, "max_queued_uploads", "NRL_MAX_QUEUED_UPLOADS", 128))
    collection_ttl_hours = float(_config_value(config, "collection_ttl_hours", "NRL_COLLECTION_TTL_HOURS", 24))
    if connect_timeout_s <= 0 or request_timeout_s <= 0:
        raise ValueError("NeMo Retriever timeouts must be greater than zero")
    if max_retries < 0:
        raise ValueError("nrl_max_retries must be zero or greater")
    if max_concurrency < 1:
        raise ValueError("nrl_max_concurrency must be at least one")
    if max_queued_uploads < 0:
        raise ValueError("nrl_max_queued_uploads must be zero or greater")
    if collection_ttl_hours <= 0:
        raise ValueError("nrl_collection_ttl_hours must be greater than zero")
    return _Settings(
        base_url=base_url,
        api_token=(
            SecretStr(api_token)
            if (api_token := _secret_value(_config_value(config, "api_token", "NRL_API_TOKEN")))
            else None
        ),
        scope=scope,
        connect_timeout_s=connect_timeout_s,
        request_timeout_s=request_timeout_s,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        max_queued_uploads=max_queued_uploads,
        verify_ssl=strict_bool(
            _config_value(config, "verify_ssl", "NRL_VERIFY_SSL", True),
            name="nrl_verify_ssl" if "verify_ssl" in config else "NRL_VERIFY_SSL",
        ),
        ca_bundle=(str(value) if (value := _config_value(config, "ca_bundle", "NRL_CA_BUNDLE")) is not None else None),
        collection_ttl_hours=collection_ttl_hours,
        warm_start=bool(config.get("warm_start", True)),
        start_ttl_cleanup=bool(config.get("start_ttl_cleanup", True)),
    )


def normalize_backend_config(config: dict[str, object]) -> dict[str, object]:
    """Validate public service options and retain secrets as ``SecretStr``."""
    if unsupported := sorted(set(config).difference(_SERVICE_CONFIG_KEYS)):
        raise ValueError(f"Unsupported nemo_retriever backend_config option(s): {', '.join(unsupported)}")
    settings = _settings(config)
    return {
        "base_url": settings.base_url,
        "api_token": settings.api_token,
        "scope": settings.scope,
        "connect_timeout_s": settings.connect_timeout_s,
        "request_timeout_s": settings.request_timeout_s,
        "max_retries": settings.max_retries,
        "max_concurrency": settings.max_concurrency,
        "max_queued_uploads": settings.max_queued_uploads,
        "verify_ssl": settings.verify_ssl,
        "ca_bundle": settings.ca_bundle,
        "collection_ttl_hours": settings.collection_ttl_hours,
    }


def _transport_for(config: dict[str, Any], settings: _Settings) -> _NRLTransport:
    injected = config.get("_transport")
    if injected is not None:
        return injected
    api_token = _secret_value(settings.api_token)
    token_fingerprint = hashlib.sha256((api_token or "").encode()).hexdigest()
    key = (
        settings.base_url,
        settings.scope,
        token_fingerprint,
        settings.connect_timeout_s,
        settings.request_timeout_s,
        settings.max_retries,
        settings.verify_ssl,
        settings.ca_bundle,
    )
    with _SHARED_TRANSPORTS_LOCK:
        transport = _SHARED_TRANSPORTS.get(key)
        if transport is None:
            transport = _NRLTransport(
                base_url=settings.base_url,
                scope=settings.scope,
                api_token=api_token,
                connect_timeout_s=settings.connect_timeout_s,
                request_timeout_s=settings.request_timeout_s,
                max_retries=settings.max_retries,
                verify_ssl=settings.verify_ssl,
                ca_bundle=settings.ca_bundle,
            )
            _SHARED_TRANSPORTS[key] = transport
        return transport


def _wire(model: Any, payload: Any, operation: str) -> Any:
    try:
        return model.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as error:
        raise NemoRetrieverError(
            f"NeMo Retriever {operation} response did not match the public API contract"
        ) from error


def _resource_path(value: str) -> str:
    return quote(value, safe="")


def _parse_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _public_document_error(value: str | None) -> str | None:
    """Contain producer exception text at the Deep Researcher Agent public API boundary."""
    return _PUBLIC_DOCUMENT_ERROR if value else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptors(file_paths: list[str], original_filenames: list[str]) -> list[_FileDescriptor]:
    if not file_paths:
        raise ValueError("At least one file is required for NeMo Retriever ingestion")
    descriptors: list[_FileDescriptor] = []
    for position, raw_path in enumerate(file_paths):
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"Ingestion file does not exist or is not a regular file: {path}")
        supplied_name = original_filenames[position] if position < len(original_filenames) else path.name
        filename = Path(str(supplied_name)).name or path.name
        content_sha256 = _sha256_file(path)
        manifest_entry_id = hashlib.sha256(f"{position}\0{filename}\0{content_sha256}".encode()).hexdigest()
        descriptors.append(
            _FileDescriptor(
                position=position,
                path=path,
                filename=filename,
                file_size=path.stat().st_size,
                content_sha256=content_sha256,
                manifest_entry_id=manifest_entry_id,
            )
        )
    return descriptors


def _idempotency_key(
    collection_name: str,
    descriptors: list[_FileDescriptor],
    metadata: dict[str, Any],
) -> str:
    canonical = {
        "collection_name": collection_name,
        "documents": [
            {
                "position": item.position,
                "filename": item.filename,
                "content_sha256": item.content_sha256,
            }
            for item in descriptors
        ],
        "metadata": metadata,
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"deep-researcher-{hashlib.sha256(encoded).hexdigest()}"


def _collection_info(item: CollectionWire) -> CollectionInfo:
    metadata = _scrub_metadata(item.metadata)
    metadata.update({"nrl_status": item.status, "expires_at": item.expires_at.isoformat() if item.expires_at else None})
    return CollectionInfo(
        name=item.name,
        description=item.description,
        backend=_BACKEND_NAME,
        created_at=_parse_timestamp(item.created_at),
        updated_at=_parse_timestamp(item.updated_at),
        metadata=metadata,
    )


def _collection_expiration(info: CollectionInfo) -> datetime | None:
    """Read back the expiration that ``_collection_info`` carries as an ISO string in metadata."""
    raw = info.metadata.get("expires_at")
    if not raw:
        return None
    try:
        return _parse_timestamp(datetime.fromisoformat(str(raw)))
    except ValueError:
        logger.warning("NeMo Retriever reported a collection expiration that could not be parsed")
        return None


def _document_info(item: DocumentWire) -> FileInfo:
    status = status_to_file_status(item.status)
    return FileInfo(
        file_id=item.document_id,
        file_name=item.filename,
        collection_name=item.collection_name,
        status=status,
        chunk_count=item.chunk_count,
        uploaded_at=_parse_timestamp(item.created_at),
        ingested_at=_parse_timestamp(item.updated_at) if status == FileStatus.SUCCESS else None,
        error_message=_public_document_error(item.error),
        metadata={
            "content_sha256": item.content_sha256,
            "document_version": item.document_version,
            "job_id": item.job_id,
            "nrl_status": item.status,
        },
    )


@register_retriever(_BACKEND_NAME)
class NemoRetrieverRetriever(BaseRetriever):
    """Retrieve citation-ready chunks through NeMo Retriever's public REST API."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._settings = _settings(self.config)
        self._transport = _transport_for(self.config, self._settings)

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
            payload = await self._transport.arequest_json(
                "POST",
                "/v1/query",
                operation="query",
                retryable=True,
                json={"query": query, "collection_name": collection_name, "top_k": top_k},
            )
            response = _wire(QueryResponseWire, payload, "query")
            if len(response.results) != 1:
                raise NemoRetrieverError("NeMo Retriever query returned an unexpected number of result sets")
            chunks = [self.normalize(hit) for hit in response.results[0].hits]
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=chunks,
                total_tokens=sum(max(1, len(chunk.content) // 4) for chunk in chunks if chunk.content),
                success=True,
            )
        except NemoRetrieverError as error:
            logger.warning("NeMo Retriever query failed: %s", error)
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=[],
                success=False,
                error_message=str(error),
            )

    def normalize(self, raw_result: Any) -> Chunk:
        hit = raw_result if isinstance(raw_result, QueryHitWire) else _wire(QueryHitWire, raw_result, "query hit")
        return normalize_query_hit(hit)

    async def health_check(self) -> bool:
        try:
            payload = await self._transport.arequest_json("GET", "/v1/health", operation="health check")
            return isinstance(payload, dict) and payload.get("status") in {"ok", "healthy"}
        except NemoRetrieverError:
            return False


@register_ingestor(_BACKEND_NAME)
class NemoRetrieverIngestor(TTLCleanupMixin, BaseIngestor):
    """Manage NRL collections and accepted ingestion jobs through REST."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._settings = _settings(self.config)
        self._transport = _transport_for(self.config, self._settings)
        self._tracking_lock = threading.Lock()
        self._accepted_by_job: dict[str, list[UploadAcceptedWire]] = {}
        self._file_jobs: dict[str, str] = {}
        self._file_sizes: dict[str, int] = {}
        self._job_collections: dict[str, str] = {}
        self._upload_batches: dict[str, _UploadBatchState] = {}
        # Needed because we want the summary to show the filename, but we work with document_ids
        self._document_id_to_filename: dict[str, str] = {}
        self._summarized: defaultdict[str, set[str]] = defaultdict(set)
        self._collection_expirations: dict[str, datetime] = {}
        self._submission_condition = threading.Condition(self._tracking_lock)
        self._outstanding_uploads = 0
        self._active_submissions = 0
        self._upload_executor = ThreadPoolExecutor(
            max_workers=self._settings.max_concurrency,
            thread_name_prefix="nrl-upload",
        )
        self._close_lock = threading.Lock()
        self._closed = False
        if self._settings.warm_start:
            self._start_warm_start_task()
        if self._settings.start_ttl_cleanup:
            self._start_ttl_cleanup_task(self._settings.collection_ttl_hours, TTL_CLEANUP_INTERVAL_SECONDS)

    def _begin_submission(self, requested: int) -> None:
        """Atomically reserve a complete batch before creating an upstream job."""
        with self._submission_condition:
            if self._closed:
                raise NemoRetrieverError("The NeMo Retriever service adapter is closed")
            outstanding = self._outstanding_uploads
            limit = self._settings.max_concurrency + self._settings.max_queued_uploads
            if requested > limit:
                logger.warning(
                    "NeMo Retriever upload admission rejected (requested=%d, outstanding=%d, limit=%d)",
                    requested,
                    outstanding,
                    limit,
                )
                raise IngestionBatchTooLargeError()
            if outstanding + requested > limit:
                logger.warning(
                    "NeMo Retriever upload admission rejected (requested=%d, outstanding=%d, limit=%d)",
                    requested,
                    outstanding,
                    limit,
                )
                raise IngestionCapacityError()
            self._outstanding_uploads += requested
            self._active_submissions += 1

    def _end_submission(self) -> None:
        with self._submission_condition:
            self._active_submissions -= 1
            self._submission_condition.notify_all()

    def _release_upload_reservations(self, count: int) -> None:
        with self._submission_condition:
            self._outstanding_uploads -= count

    def _start_warm_start_task(self) -> None:
        """Reconcile on a daemon thread so startup neither blocks on NRL nor fails with it."""
        thread = threading.Thread(
            target=self._warm_start,
            daemon=True,
            name=f"{self.backend_name}-warm-start",
        )
        thread.start()

    def _warm_start(self) -> None:
        """Reconcile with NRL, retire collections that expired while down, and warm the caches."""
        try:
            self._reconcile_summaries()
            self._cleanup_expired_collections()
        except Exception:
            logger.exception("NeMo Retriever warm start did not complete")

    def _cleanup_expired_collections(self) -> None:
        """Drop state only after NRL confirms a locally due collection is gone.

        NRL expires the collection itself, so cleanup never issues a delete. A locally cached
        deadline is only a signal to check NRL: the collection could have had its expiration
        extended after the last create, update, or reconciliation. Deep Researcher Agent drops its summaries and
        cached state only once NRL no longer returns the collection.

        It also replaces the mixin's idle-time policy: NRL is given an absolute expiration at
        creation and enforces it, so inferring one from ``updated_at`` would push Deep Researcher Agent's view of a
        collection past the deadline the service will actually act on.
        """
        now = datetime.now(UTC)
        with self._tracking_lock:
            due = [name for name, expires_at in self._collection_expirations.items() if expires_at <= now]
        for name in due:
            try:
                collection = self.get_collection(name)
                if collection is not None:
                    nrl_expires_at = _collection_expiration(collection)
                    if nrl_expires_at is None or nrl_expires_at > now:
                        self._track_expiration(name, nrl_expires_at)
                        continue
            except NemoRetrieverError:
                logger.warning("Skipped cleanup for a due collection because NeMo Retriever is unavailable")
                continue

            self._forget_collection(name)
            logger.info("Dropped summaries for an expired NeMo Retriever collection")

    def _track_expiration(self, name: str, expires_at: datetime | None) -> None:
        """Record when NRL will expire a collection, or stop tracking one that no longer expires."""
        with self._tracking_lock:
            if expires_at is None:
                self._collection_expirations.pop(name, None)
            else:
                self._collection_expirations[name] = expires_at

    def _forget_collection(self, name: str) -> None:
        """Drop a collection's summaries and the adapter state that tracked its documents."""
        clear_collection_summaries(name)
        with self._tracking_lock:
            self._collection_expirations.pop(name, None)
            for document_id in self._summarized.pop(name, set()):
                self._document_id_to_filename.pop(document_id, None)

    def _reconcile_summaries(self) -> None:
        """Update the summary store to match the in-scope collections and documents NRL holds.

        NRL owns document lifetime, including server-side collection expiration, so it is the
        authority here: documents it serves must have a summary row, and rows it no longer backs
        are dropped. A collection is cleared only once NRL positively reports it absent, so a
        transport failure leaves the store untouched instead of deleting live summaries. The
        expirations NRL reports are adopted as well, which is what lets TTL cleanup act on
        collections this process did not create itself.
        """
        collections = self.list_collections()
        for collection in collections:
            self._track_expiration(collection.name, _collection_expiration(collection))
        live_collections = {collection.name for collection in collections}
        for name in live_collections | set(list_summary_collections()):
            try:
                if name in live_collections:
                    self._reconcile_collection(name)
                # collections in the summary store that we don't find in NRL are left as-is
                # because they may be owned by another scope
            except NemoRetrieverError:
                logger.warning("Skipped summary reconciliation for one collection (NeMo Retriever unavailable)")

    def _reconcile_collection(self, collection_name: str) -> None:
        """Reconcile one collection that NRL still holds against its stored summaries.

        The store is read before NRL so that a document ingested mid-reconcile is treated as new
        rather than as an orphan: an extra idempotent write is harmless, a wrong delete is not.

        Every document NRL lists gets its filename mapped, because a delete can arrive for one that
        is still ingesting. Only successfully ingested documents count as summarized.
        """
        stored = {document.file_name for document in get_available_documents(collection_name)}
        documents = self.list_files(collection_name)
        known: set[str] = set()
        for document in documents:
            filename = document.file_name
            known.add(filename)
            with self._tracking_lock:
                self._document_id_to_filename[document.file_id] = filename
            if document.status != FileStatus.SUCCESS:
                continue
            if filename not in stored:
                self._register_summary(collection_name, document.status, document.file_id, filename, None)
            else:
                with self._tracking_lock:
                    self._summarized[collection_name].add(document.file_id)
        for orphan in stored - known:
            unregister_summary(collection_name, orphan)

    @property
    def backend_name(self) -> str:
        return _BACKEND_NAME

    def create_collection(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        expires_at = datetime.now(UTC) + timedelta(hours=self._settings.collection_ttl_hours)
        payload = self._transport.request_json(
            "POST",
            "/v1/collections",
            operation="create collection",
            json={
                "name": name,
                "description": description,
                "metadata": _scrub_metadata(metadata or {}),
                "expires_at": expires_at.isoformat(),
            },
        )
        info = _collection_info(_wire(CollectionWire, payload, "create collection"))
        self._track_expiration(name, _collection_expiration(info) or expires_at)
        return info

    def update_collection(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> CollectionInfo:
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = _scrub_metadata(metadata)
        if expires_at is not None:
            body["expires_at"] = expires_at.isoformat()
        payload = self._transport.request_json(
            "PATCH",
            f"/v1/collections/{_resource_path(name)}",
            operation="update collection",
            json=body,
        )
        info = _collection_info(_wire(CollectionWire, payload, "update collection"))
        self._track_expiration(name, _collection_expiration(info) or expires_at)
        return info

    def delete_collection(self, name: str) -> bool:
        payload = self._transport.request_json(
            "DELETE",
            f"/v1/collections/{_resource_path(name)}",
            operation="delete collection",
            retryable=True,
            params={"if_exists": "true"},
        )
        result = _wire(CollectionDeleteWire, payload, "delete collection")
        self._forget_collection(name)
        return result.deleted or not result.existed or result.cleanup_pending

    def list_collections(self) -> list[CollectionInfo]:
        items: list[CollectionInfo] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": _RESOURCE_PAGE_SIZE}
            if token:
                params["continuation_token"] = token
            payload = self._transport.request_json(
                "GET", "/v1/collections", operation="list collections", params=params
            )
            page = _wire(CollectionPageWire, payload, "list collections")
            items.extend(_collection_info(item) for item in page.items)
            if not page.next_token:
                return items
            if page.next_token in seen_tokens:
                raise NemoRetrieverError("NeMo Retriever collection pagination repeated a continuation token")
            seen_tokens.add(page.next_token)
            token = page.next_token

    def get_collection(self, name: str) -> CollectionInfo | None:
        try:
            payload = self._transport.request_json(
                "GET", f"/v1/collections/{_resource_path(name)}", operation="get collection"
            )
        except NemoRetrieverHTTPError as error:
            if error.status_code == 404:
                return None
            raise
        return _collection_info(_wire(CollectionWire, payload, "get collection"))

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        job_config = dict(config or {})
        originals = [str(value) for value in job_config.get("original_filenames", [])]
        cleanup_files = strict_bool(job_config.get("cleanup_files", False), name="cleanup_files")
        descriptors = _descriptors(file_paths, originals)
        file_metadata = _scrub_metadata(job_config.get("metadata") or {})
        manifest = [
            {
                "manifest_entry_id": item.manifest_entry_id,
                "filename": item.filename,
                "content_sha256": item.content_sha256,
            }
            for item in descriptors
        ]
        idempotency_key = str(
            job_config.get("idempotency_key") or _idempotency_key(collection_name, descriptors, file_metadata)
        )
        self._begin_submission(len(descriptors))
        reservations_owned_by_submission = True
        try:
            created_payload = self._transport.request_json(
                "POST",
                "/v1/ingest/job",
                operation="job creation",
                retryable=True,
                compatibility_route=True,
                json={
                    "expected_documents": len(descriptors),
                    "collection_name": collection_name,
                    "operation": "append",
                    "retain_results": False,
                    "idempotency_key": idempotency_key,
                    "document_manifest": manifest,
                },
            )
            created = _wire(JobCreatedWire, created_payload, "job creation")

            batch = _UploadBatchState(descriptors=tuple(descriptors), remaining=len(descriptors))
            with self._tracking_lock:
                self._upload_batches[created.job_id] = batch
                self._job_collections[created.job_id] = collection_name

            reservations_owned_by_submission = False
            scheduled = 0
            try:
                for descriptor in batch.descriptors:
                    future = self._upload_executor.submit(
                        self._upload_document,
                        created.job_id,
                        descriptor,
                        file_metadata,
                    )
                    scheduled += 1
                    future.add_done_callback(
                        lambda completed, item=descriptor: self._finish_upload(
                            created.job_id,
                            batch,
                            item,
                            cleanup_files,
                            completed,
                        )
                    )
            except Exception as error:
                logger.warning(
                    "NeMo Retriever upload scheduling failed (job_id=%s, error_type=%s)",
                    created.job_id,
                    type(error).__name__,
                )
                for descriptor in batch.descriptors[scheduled:]:
                    self._finish_upload(created.job_id, batch, descriptor, cleanup_files, None)
            return created.job_id
        except Exception:
            if reservations_owned_by_submission:
                self._release_upload_reservations(len(descriptors))
            raise
        finally:
            self._end_submission()

    def _upload_document(
        self,
        job_id: str,
        descriptor: _FileDescriptor,
        file_metadata: dict[str, Any],
    ) -> UploadAcceptedWire:
        metadata: dict[str, Any] = {"filename": descriptor.filename}
        if file_metadata:
            metadata["metadata"] = file_metadata
        payload = self._transport.upload_document(
            job_id=job_id,
            file_path=descriptor.path,
            filename=descriptor.filename,
            manifest_entry_id=descriptor.manifest_entry_id,
            metadata=json.dumps(metadata, separators=(",", ":"), sort_keys=True),
            retryable=True,
        )
        return _wire(UploadAcceptedWire, payload, "job document upload")

    def _finish_upload(
        self,
        job_id: str,
        batch: _UploadBatchState,
        descriptor: _FileDescriptor,
        cleanup_file: bool,
        future: Future[UploadAcceptedWire] | None,
    ) -> None:
        accepted: UploadAcceptedWire | None = None
        try:
            if future is not None:
                accepted = future.result()
        except Exception as error:
            logger.warning(
                "NeMo Retriever document upload failed (filename=%s, error_type=%s)",
                descriptor.filename,
                type(error).__name__,
            )
        finally:
            if cleanup_file:
                try:
                    descriptor.path.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(
                        "Failed to remove temporary upload (filename=%s, error_type=%s)",
                        descriptor.filename,
                        type(error).__name__,
                    )

        uploads_done = False
        with self._tracking_lock:
            if accepted is None:
                batch.failed_by_position[descriptor.position] = _PUBLIC_DOCUMENT_ERROR
            else:
                batch.accepted_by_position[descriptor.position] = accepted
                self._file_jobs[accepted.document_id] = job_id
                self._file_sizes[accepted.document_id] = descriptor.file_size
            batch.remaining -= 1
            self._outstanding_uploads -= 1
            if batch.remaining == 0:
                accepted_items = [batch.accepted_by_position[index] for index in sorted(batch.accepted_by_position)]
                self._accepted_by_job[job_id] = accepted_items
                uploads_done = True

        if uploads_done:
            batch.done.set()

    def close(self) -> None:
        """Finish accepted uploads and release the adapter-owned worker pool."""
        with self._close_lock:
            with self._submission_condition:
                if self._closed:
                    return
                self._closed = True
                self._submission_condition.notify_all()
                while self._active_submissions:
                    self._submission_condition.wait()
            self._upload_executor.shutdown(wait=True, cancel_futures=False)

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        payload = self._transport.request_json(
            "GET", f"/v1/ingest/job/{_resource_path(job_id)}", operation="job status"
        )
        aggregate = _wire(JobAggregateWire, payload, "job status")
        documents = self._list_job_documents(job_id)
        with self._tracking_lock:
            batch = self._upload_batches.get(job_id)
            uploads_done = batch.done.is_set() if batch else True
            upload_failures = dict(batch.failed_by_position) if batch else {}
            accepted_uploads = dict(batch.accepted_by_position) if batch else {}
        state = {
            "pending": JobState.PENDING,
            "processing": JobState.PROCESSING,
            "completed": JobState.COMPLETED,
            "failed": JobState.FAILED,
            "partial_success": JobState.COMPLETED,
        }.get(aggregate.status.lower(), JobState.PROCESSING)
        upstream_details: dict[str, FileProgress] = {}
        attempt_ids: dict[str, str] = {}
        collection_name = aggregate.collection_name
        if not collection_name:
            with self._tracking_lock:
                collection_name = self._job_collections.get(job_id, None)
        for item in documents:
            status = status_to_file_status(item.status)
            attempt_ids[item.document_id] = item.attempt_id
            # Always register at least the filename. We may want to gate this behind the generate_summary flag later.
            # TODO: Get the summary for the document
            self._register_summary(
                collection_name,
                status,
                item.document_id,
                item.filename,
            )
            upstream_details[item.document_id] = FileProgress(
                file_id=item.document_id,
                file_name=item.filename or item.document_id,
                status=status,
                progress_percent=100.0 if status in {FileStatus.SUCCESS, FileStatus.FAILED} else 50.0,
                error_message=_public_document_error(item.error),
                chunks_created=max(0, item.result_rows or 0),
            )

        file_details: list[FileProgress] = []
        if batch:
            for descriptor in batch.descriptors:
                if message := upload_failures.get(descriptor.position):
                    detail = FileProgress(
                        file_id=descriptor.manifest_entry_id,
                        file_name=descriptor.filename,
                        status=FileStatus.FAILED,
                        progress_percent=100.0,
                        error_message=message,
                    )
                elif accepted := accepted_uploads.get(descriptor.position):
                    detail = upstream_details.pop(
                        accepted.document_id,
                        FileProgress(
                            file_id=accepted.document_id,
                            file_name=descriptor.filename,
                            status=status_to_file_status(accepted.status),
                            progress_percent=25.0,
                        ),
                    )
                else:
                    detail = FileProgress(
                        file_id=descriptor.manifest_entry_id,
                        file_name=descriptor.filename,
                        status=FileStatus.UPLOADING,
                        progress_percent=0.0,
                    )
                file_details.append(detail)

            file_details.extend(upstream_details.values())

            if not uploads_done:
                state = JobState.PROCESSING if documents else JobState.PENDING
        else:
            file_details.extend(upstream_details.values())

        terminal = sum(item.status in {FileStatus.SUCCESS, FileStatus.FAILED} for item in file_details)
        if batch and uploads_done and upload_failures:
            state = (
                JobState.FAILED
                if aggregate.status.lower() == "failed" or terminal >= aggregate.expected_documents
                else JobState.PROCESSING
            )
        error_message = None
        if upload_failures and state == JobState.FAILED:
            error_message = "NeMo Retriever rejected one or more document uploads"
        elif aggregate.status.lower() == "failed":
            error_message = "NeMo Retriever ingestion job failed"
        elif aggregate.status.lower() == "partial_success":
            error_message = "NeMo Retriever ingestion job completed with one or more failed documents"
        return IngestionJobStatus(
            job_id=aggregate.job_id,
            status=state,
            submitted_at=_parse_timestamp(aggregate.created_at) or datetime.now(UTC),
            started_at=_parse_timestamp(aggregate.started_at),
            completed_at=(
                _parse_timestamp(aggregate.finalized_at) if state in {JobState.COMPLETED, JobState.FAILED} else None
            ),
            total_files=aggregate.expected_documents,
            processed_files=terminal,
            file_details=file_details,
            collection_name=collection_name or "",
            backend=_BACKEND_NAME,
            error_message=error_message,
            metadata={
                "nrl_status": aggregate.status,
                "counts": dict(aggregate.counts),
                "operation": aggregate.operation,
                "trace_id": aggregate.trace_id,
                "attempt_ids": attempt_ids,
                "uploads_complete": uploads_done,
                "upload_failures": len(upload_failures),
            },
        )

    def _list_job_documents(self, job_id: str) -> list[JobDocumentWire]:
        items: list[JobDocumentWire] = []
        offset = 0
        while True:
            payload = self._transport.request_json(
                "GET",
                f"/v1/ingest/job/{_resource_path(job_id)}/documents",
                operation="job document listing",
                params={"offset": offset, "limit": _JOB_PAGE_SIZE},
            )
            page = _wire(JobDocumentsPageWire, payload, "job document listing")
            items.extend(page.items)
            offset += len(page.items)
            if offset >= page.total_filtered:
                return items
            if not page.items:
                raise NemoRetrieverError("NeMo Retriever job document pagination made no progress")

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        job_id = self.submit_job(
            [file_path],
            collection_name,
            config={"original_filenames": [Path(file_path).name], "metadata": metadata or {}},
        )
        with self._tracking_lock:
            batch = self._upload_batches.get(job_id)
        if batch is None or not batch.done.wait(timeout=self._settings.request_timeout_s):
            raise NemoRetrieverError("Timed out waiting for NeMo Retriever to accept the document upload")
        with self._tracking_lock:
            accepted = list(self._accepted_by_job.get(job_id, []))
            failed = bool(batch.failed_by_position)
        if failed:
            raise NemoRetrieverError("NeMo Retriever rejected the document upload")
        if not accepted:
            raise NemoRetrieverError("NeMo Retriever accepted the job without a document identifier")
        item = accepted[0]
        return FileInfo(
            file_id=item.document_id,
            file_name=Path(file_path).name,
            collection_name=collection_name,
            status=status_to_file_status(item.status),
            file_size=Path(file_path).stat().st_size,
            uploaded_at=_parse_timestamp(item.created_at),
            metadata={
                "job_id": job_id,
                "attempt_id": item.attempt_id,
                "content_sha256": item.content_sha256,
                "nrl_status": item.status,
            },
        )

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        payload = self._transport.request_json(
            "DELETE",
            f"/v1/collections/{_resource_path(collection_name)}/documents/{_resource_path(file_id)}",
            operation="delete document",
            retryable=True,
            params={"if_exists": "true"},
        )
        result = _wire(DocumentDeleteWire, payload, "delete document")
        with self._tracking_lock:
            summary_key = self._document_id_to_filename.get(file_id, file_id)
        unregister_summary(collection_name, summary_key)
        with self._tracking_lock:
            self._document_id_to_filename.pop(file_id, None)
            self._summarized[collection_name].discard(file_id)
        return result.deleted or not result.existed or result.cleanup_pending

    def list_files(self, collection_name: str) -> list[FileInfo]:
        items: list[FileInfo] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": _RESOURCE_PAGE_SIZE}
            if token:
                params["continuation_token"] = token
            payload = self._transport.request_json(
                "GET",
                f"/v1/collections/{_resource_path(collection_name)}/documents",
                operation="list documents",
                params=params,
            )
            page = _wire(DocumentPageWire, payload, "list documents")
            items.extend(_document_info(item) for item in page.items)
            if not page.next_token:
                return items
            if page.next_token in seen_tokens:
                raise NemoRetrieverError("NeMo Retriever document pagination repeated a continuation token")
            seen_tokens.add(page.next_token)
            token = page.next_token

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        try:
            payload = self._transport.request_json(
                "GET",
                f"/v1/collections/{_resource_path(collection_name)}/documents/{_resource_path(file_id)}",
                operation="get document",
            )
            return _document_info(_wire(DocumentWire, payload, "get document"))
        except NemoRetrieverHTTPError as error:
            if error.status_code != 404:
                raise
        with self._tracking_lock:
            job_id = self._file_jobs.get(file_id)
            file_size = self._file_sizes.get(file_id)
        if not job_id:
            return None
        for item in self._list_job_documents(job_id):
            if item.document_id != file_id:
                continue
            status = status_to_file_status(item.status)
            return FileInfo(
                file_id=item.document_id,
                file_name=item.filename or item.document_id,
                collection_name=item.collection_name or collection_name,
                status=status,
                file_size=file_size,
                chunk_count=max(0, item.result_rows or 0),
                uploaded_at=_parse_timestamp(item.submitted_at),
                ingested_at=_parse_timestamp(item.completed_at) if status == FileStatus.SUCCESS else None,
                error_message=_public_document_error(item.error),
                metadata={
                    "job_id": item.job_id,
                    "attempt_id": item.attempt_id,
                    "content_sha256": item.content_sha256,
                    "nrl_status": item.status,
                },
            )
        return None

    async def health_check(self) -> bool:
        try:
            payload = await self._transport.arequest_json("GET", "/v1/health", operation="health check")
            return isinstance(payload, dict) and payload.get("status") in {"ok", "healthy"}
        except NemoRetrieverError:
            return False

    def _register_summary(
        self,
        collection_name: str | None,
        status: FileStatus,
        document_id: str,
        filename: str | None,
        summary: str | None = None,
    ) -> None:
        """Register a summary for a document if it has been successfully ingested."""
        if not filename or not collection_name:
            return
        if status == FileStatus.SUCCESS and document_id not in self._summarized[collection_name]:
            has_summary = summary is not None
            summary = summary or _PLACEHOLDER_SUMMARY
            logger.info(f"registering summary for {filename}")
            with self._tracking_lock:
                self._document_id_to_filename[document_id] = filename
                self._summarized[collection_name].add(document_id)
            register_summary(collection_name, filename, summary, upsert=has_summary)
