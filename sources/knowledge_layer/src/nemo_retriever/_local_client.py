# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local orchestration for embedded NeMo Retriever and LanceDB."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from deep_researcher_agent.knowledge import FileProgress
from deep_researcher_agent.knowledge import IngestionJobStatus
from deep_researcher_agent.knowledge import JobState
from deep_researcher_agent.knowledge.schema import CollectionInfo
from deep_researcher_agent.knowledge.schema import FileInfo
from deep_researcher_agent.knowledge.schema import FileStatus

from ._normalization import scrub_metadata
from ._normalization import status_to_file_status
from ._normalization import strict_bool

logger = logging.getLogger(__name__)

_BACKEND_NAME = "nemo_retriever_local"
_ADAPTER_MARKER = "deep_researcher_nemo_retriever_local"
_ADAPTER_SCHEMA_VERSION = 1
_PAGE_SIZE = 100
_RECONCILIATION_INTERVAL_S = 60.0
_WORK_QUEUE_SIZE = 128
_DEFAULT_TABLE_NAME = "nemo_retriever"
_TERMINAL_FILE_STATUSES = frozenset({FileStatus.SUCCESS, FileStatus.FAILED})
_PHYSICAL_TABLE_PATTERN = re.compile(r"\bnrl_[0-9a-f]{40}\b")
_LOCAL_CONFIG_KEYS = frozenset(
    {
        "data_dir",
        "scope",
        "profile",
        "page_elements_invoke_url",
        "ocr_invoke_url",
        "table_structure_invoke_url",
        "embed_invoke_url",
        "embed_model_name",
        "embed_model_provider_prefix",
        "inference_api_key",
        "collection_ttl_hours",
    }
)


class NemoRetrieverLocalError(RuntimeError):
    """Base error for the embedded adapter."""


class NemoRetrieverLocalDependencyError(NemoRetrieverLocalError):
    """The isolated local-backend dependencies are unavailable."""


class NemoRetrieverLocalOwnershipError(NemoRetrieverLocalError):
    """A collection was not created by this compatible adapter configuration."""


class NemoRetrieverLocalLockError(NemoRetrieverLocalError):
    """Another process owns the selected local data directory."""


@dataclass(frozen=True)
class LocalSettings:
    """Validated non-secret and secret runtime settings."""

    data_dir: Path
    scope: str
    profile: str
    page_elements_invoke_url: str | None
    ocr_invoke_url: str | None
    table_structure_invoke_url: str | None
    embed_invoke_url: str | None
    embed_model_name: str | None
    embed_model_provider_prefix: str | None
    inference_api_key: SecretStr | None
    collection_ttl_hours: float = 24.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LocalSettings:
        raw_dir = str(
            _config_value(config, "data_dir", "NRL_LOCAL_DATA_DIR", ".deep-researcher-data/nemo_retriever")
        ).strip()
        if not raw_dir:
            raise ValueError("nrl_local_data_dir must not be empty")
        scope = str(_config_value(config, "scope", "NRL_SCOPE", "local")).strip()
        if not scope:
            raise ValueError("nemo_retriever_local requires an explicit nrl_scope")
        profile = str(_config_value(config, "profile", "NRL_LOCAL_PROFILE", "auto")).strip().lower()
        if profile not in {"auto", "fast-text"}:
            raise ValueError("nrl_local_profile must be either 'auto' or 'fast-text'")
        collection_ttl_hours = float(_config_value(config, "collection_ttl_hours", "NRL_COLLECTION_TTL_HOURS", 24.0))
        if collection_ttl_hours <= 0:
            raise ValueError("nrl_collection_ttl_hours must be greater than zero")
        return cls(
            data_dir=Path(raw_dir).expanduser().resolve(),
            scope=scope,
            profile=profile,
            page_elements_invoke_url=_optional_text(
                _config_value(config, "page_elements_invoke_url", "NRL_PAGE_ELEMENTS_INVOKE_URL")
            ),
            ocr_invoke_url=_optional_text(_config_value(config, "ocr_invoke_url", "NRL_OCR_INVOKE_URL")),
            table_structure_invoke_url=_optional_text(
                _config_value(config, "table_structure_invoke_url", "NRL_TABLE_STRUCTURE_INVOKE_URL")
            ),
            embed_invoke_url=_optional_text(_config_value(config, "embed_invoke_url", "NRL_EMBED_INVOKE_URL")),
            embed_model_name=_optional_text(_config_value(config, "embed_model_name", "NRL_EMBED_MODEL_NAME")),
            embed_model_provider_prefix=_optional_text(
                _config_value(config, "embed_model_provider_prefix", "NRL_EMBED_MODEL_PROVIDER_PREFIX")
            ),
            inference_api_key=(
                SecretStr(api_key)
                if (api_key := _secret_value(_config_value(config, "inference_api_key", "NRL_INFERENCE_API_KEY")))
                else None
            ),
            collection_ttl_hours=collection_ttl_hours,
        )

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        """Values that must agree for callers sharing one runtime."""
        secret_fingerprint = hashlib.sha256((_secret_value(self.inference_api_key) or "").encode()).hexdigest()
        return (
            self.scope,
            self.profile,
            self.page_elements_invoke_url,
            self.ocr_invoke_url,
            self.table_structure_invoke_url,
            self.embed_invoke_url,
            self.embed_model_name,
            self.embed_model_provider_prefix,
            secret_fingerprint,
            self.collection_ttl_hours,
        )


def normalize_backend_config(config: dict[str, object]) -> dict[str, object]:
    """Validate public local options and retain inference credentials as ``SecretStr``."""
    if unsupported := sorted(set(config).difference(_LOCAL_CONFIG_KEYS)):
        raise ValueError(f"Unsupported nemo_retriever_local backend_config option(s): {', '.join(unsupported)}")
    settings = LocalSettings.from_config(config)
    return {
        "data_dir": str(settings.data_dir),
        "scope": settings.scope,
        "profile": settings.profile,
        "page_elements_invoke_url": settings.page_elements_invoke_url,
        "ocr_invoke_url": settings.ocr_invoke_url,
        "table_structure_invoke_url": settings.table_structure_invoke_url,
        "embed_invoke_url": settings.embed_invoke_url,
        "embed_model_name": settings.embed_model_name,
        "embed_model_provider_prefix": settings.embed_model_provider_prefix,
        "inference_api_key": settings.inference_api_key,
        "collection_ttl_hours": settings.collection_ttl_hours,
    }


@dataclass(frozen=True)
class _NRLBindings:
    create_ingestor: Any
    LanceDB: Any
    IngestVdbOperator: Any
    RetrieveVdbOperator: Any
    CollectionWriteContext: Any
    CollectionCreateRequest: Any
    IngestOperation: Any
    VDBInvalidRequest: type[Exception]
    VDBResourceNotFound: type[Exception]
    VDBResourceConflict: type[Exception]
    IngestPlanRequest: Any
    IngestSourceOptions: Any
    IngestRuntimeOptions: Any
    IngestExtractOptions: Any
    IngestEmbedOptions: Any
    resolve_ingest_plan: Any
    resolve_embed_model: Any
    resolve_remote_api_key: Any
    infer_microservice: Any
    default_embed_endpoint: str
    to_client_vdb_records: Any
    apply_sidecar_metadata_to_client_batches: Any
    pandas: Any
    FileLock: Any
    FileLockTimeout: type[Exception]


@dataclass
class _StagedFile:
    document_id: str
    document_version: str
    content_sha256: str
    filename: str
    file_size: int
    path: Path
    metadata: dict[str, Any]


@dataclass
class _Job:
    job_id: str
    collection_name: str
    submitted_at: datetime
    files: list[FileProgress]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _WorkItem:
    job_id: str
    collection_name: str
    staged: _StagedFile


def _load_nrl_bindings() -> _NRLBindings:
    """Import the isolated local stack only when this backend is selected."""
    try:
        import pandas as pd
        from filelock import FileLock
        from filelock import Timeout as FileLockTimeout
        from nemo_retriever import create_ingestor
        from nemo_retriever.common.remote_auth import resolve_remote_api_key
        from nemo_retriever.common.schemas.collections import CollectionCreateRequest
        from nemo_retriever.common.schemas.collections import IngestOperation
        from nemo_retriever.common.vdb.adt_vdb import CollectionWriteContext
        from nemo_retriever.common.vdb.adt_vdb import VDBInvalidRequest
        from nemo_retriever.common.vdb.adt_vdb import VDBResourceConflict
        from nemo_retriever.common.vdb.adt_vdb import VDBResourceNotFound
        from nemo_retriever.common.vdb.lancedb import LanceDB
        from nemo_retriever.common.vdb.records import to_client_vdb_records
        from nemo_retriever.common.vdb.sidecar_metadata import apply_sidecar_metadata_to_client_batches
        from nemo_retriever.ingest.plan import IngestEmbedOptions
        from nemo_retriever.ingest.plan import IngestExtractOptions
        from nemo_retriever.ingest.plan import IngestPlanRequest
        from nemo_retriever.ingest.plan import IngestRuntimeOptions
        from nemo_retriever.ingest.plan import IngestSourceOptions
        from nemo_retriever.ingest.plan import resolve_ingest_plan
        from nemo_retriever.models import resolve_embed_model
        from nemo_retriever.models.nim.util import infer_microservice
        from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor
        from nemo_retriever.operators.vdb import IngestVdbOperator
        from nemo_retriever.operators.vdb import RetrieveVdbOperator
    except ImportError as error:
        raise NemoRetrieverLocalDependencyError(
            "nemo_retriever_local requires the isolated Python 3.12 environment. "
            "Start Deep Researcher Agent with `uv run --project environments/nemo_retriever_local nat serve ...`."
        ) from error
    try:
        default_embed_endpoint = _BatchEmbedCPUActor.DEFAULT_EMBED_INVOKE_URL
    except AttributeError as error:
        raise NemoRetrieverLocalDependencyError(
            "nemo_retriever_local requires the pinned NeMo Retriever revision. "
            "Recreate the isolated environment with "
            "`uv sync --project environments/nemo_retriever_local --frozen`."
        ) from error
    return _NRLBindings(
        create_ingestor=create_ingestor,
        LanceDB=LanceDB,
        IngestVdbOperator=IngestVdbOperator,
        RetrieveVdbOperator=RetrieveVdbOperator,
        CollectionWriteContext=CollectionWriteContext,
        CollectionCreateRequest=CollectionCreateRequest,
        IngestOperation=IngestOperation,
        VDBInvalidRequest=VDBInvalidRequest,
        VDBResourceNotFound=VDBResourceNotFound,
        VDBResourceConflict=VDBResourceConflict,
        IngestPlanRequest=IngestPlanRequest,
        IngestSourceOptions=IngestSourceOptions,
        IngestRuntimeOptions=IngestRuntimeOptions,
        IngestExtractOptions=IngestExtractOptions,
        IngestEmbedOptions=IngestEmbedOptions,
        resolve_ingest_plan=resolve_ingest_plan,
        resolve_embed_model=resolve_embed_model,
        resolve_remote_api_key=resolve_remote_api_key,
        infer_microservice=infer_microservice,
        # NRL does not yet expose its remote embedding default through a public
        # resolver. Keep this compatibility import isolated at the pinned SHA.
        default_embed_endpoint=default_embed_endpoint,
        to_client_vdb_records=to_client_vdb_records,
        apply_sidecar_metadata_to_client_batches=apply_sidecar_metadata_to_client_batches,
        pandas=pd,
        FileLock=FileLock,
        FileLockTimeout=FileLockTimeout,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _config_value(config: dict[str, Any], key: str, env_name: str, default: Any = None) -> Any:
    """Resolve a local-adapter option while keeping its environment contract local."""
    if key in config and config[key] not in (None, ""):
        return config[key]
    value = os.environ.get(env_name)
    return value if value not in (None, "") else default


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return _optional_text(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _is_active_collection(item: dict[str, Any]) -> bool:
    if str(item.get("status") or "").lower() != "active":
        return False
    expires_at = _timestamp(item.get("expires_at"))
    return expires_at is None or expires_at > datetime.now(UTC)


def _model_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    raise TypeError(f"Unexpected NeMo Retriever value: {type(value).__name__}")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        logger.debug("Could not tighten permissions on local NeMo Retriever directory %s", path)


class LocalRuntime:
    """One embedded NRL/LanceDB runtime for one canonical data directory."""

    def __init__(self, settings: LocalSettings, bindings: _NRLBindings | None = None):
        self.settings = settings
        self.bindings = bindings or _load_nrl_bindings()
        self._state_lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._active_writes: set[tuple[str, str]] = set()
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue(maxsize=_WORK_QUEUE_SIZE)
        self._stop = threading.Event()
        self._closed = False
        self._lock_file: Any | None = None
        self._worker: threading.Thread | None = None
        self._reconciler: threading.Thread | None = None
        _safe_mkdir(settings.data_dir)
        self._acquire_process_lock()
        worker_started = False
        reconciler_started = False
        try:
            # Staging belongs to process-local jobs. Only the process holding
            # the directory lock may discard leftovers from an interrupted run.
            shutil.rmtree(self._staging_root, ignore_errors=True)
            _safe_mkdir(self._staging_root)
            self._embedding_model = str(self.bindings.resolve_embed_model(settings.embed_model_name))
            self._embedding_endpoint = settings.embed_invoke_url or self.bindings.default_embed_endpoint
            resolved_api_key = self.bindings.resolve_remote_api_key(_secret_value(settings.inference_api_key))
            self._inference_api_key = SecretStr(resolved_api_key) if resolved_api_key else None
            self.vdb = self.bindings.LanceDB(
                uri=str(settings.data_dir / "lancedb"),
                table_name=_DEFAULT_TABLE_NAME,
                vector_dim=None,
                overwrite=False,
                build_index=False,
                _service_table_schema=True,
                expiration_cleanup_enabled=True,
            )
            self.ingest_operator = self.bindings.IngestVdbOperator(vdb=self.vdb)
            self.retrieve_operator = self.bindings.RetrieveVdbOperator(vdb=self.vdb)
            self.vdb.reconcile_collections()
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="nemo-retriever-local-worker",
            )
            self._reconciler = threading.Thread(
                target=self._reconciliation_loop,
                daemon=True,
                name="nemo-retriever-local-reconciler",
            )
            self._worker.start()
            worker_started = True
            self._reconciler.start()
            reconciler_started = True
        except Exception:
            self._stop.set()
            if worker_started:
                self._queue.put_nowait(None)
            if reconciler_started:
                self._reconciler.join()
            if worker_started:
                self._worker.join()
            shutil.rmtree(self._staging_root, ignore_errors=True)
            self._release_process_lock()
            raise

    @property
    def _staging_root(self) -> Path:
        return self.settings.data_dir / ".staging"

    @property
    def ownership_marker(self) -> dict[str, Any]:
        return {
            "adapter_schema_version": _ADAPTER_SCHEMA_VERSION,
            "embedding_model": self._embedding_model,
            "provider_prefix": self.settings.embed_model_provider_prefix,
            "profile": self.settings.profile,
        }

    def _acquire_process_lock(self) -> None:
        lock_path = self.settings.data_dir / ".deep-researcher-nemo-retriever.lock"
        # NAT may finalize a function on a different thread than the one that
        # initialized it. Keep lock ownership process-scoped rather than tied
        # to filelock's default thread-local context.
        lock = self.bindings.FileLock(str(lock_path), thread_local=False)
        try:
            lock.acquire(timeout=0)
        except self.bindings.FileLockTimeout as error:
            raise NemoRetrieverLocalLockError(
                f"The NeMo Retriever local data directory {self.settings.data_dir} is already open by another "
                "process. Stop that Deep Researcher Agent process or choose a different nrl_local_data_dir."
            ) from error
        self._lock_file = lock

    def _release_process_lock(self) -> None:
        lock, self._lock_file = self._lock_file, None
        if lock is not None:
            lock.release()

    def _reconciliation_loop(self) -> None:
        while not self._stop.wait(_RECONCILIATION_INTERVAL_S):
            try:
                self.vdb.reconcile_collections()
            except Exception as error:
                logger.warning(
                    "Embedded NeMo Retriever collection reconciliation failed: %s",
                    self.public_error(error),
                )

    def close(self) -> None:
        """Stop background work and release the process-lifetime lock."""
        # Serialize the terminal transition with the final accepted enqueue.
        # Staging happens outside this lock, but submit_job rechecks open state
        # while holding it and cleans any rejected private copy.
        with self._submit_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                self._stop.set()
            worker = self._worker
            if worker is not None:
                # Queued pre-write jobs intentionally do not survive shutdown.
                # Drain their private staging files, then wait only for the item
                # already being processed before releasing the directory lock.
                while True:
                    try:
                        work = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if work is not None:
                            with self._state_lock:
                                self._active_writes.discard((work.collection_name, work.staged.document_id))
                            work.staged.path.unlink(missing_ok=True)
                    finally:
                        self._queue.task_done()
                self._queue.put_nowait(None)
        reconciler = self._reconciler
        if reconciler is not None:
            reconciler.join()
        if worker is not None:
            worker.join()
        shutil.rmtree(self._staging_root, ignore_errors=True)
        self._release_process_lock()

    def _ensure_open(self) -> None:
        if self._closed:
            raise NemoRetrieverLocalError("The embedded NeMo Retriever runtime is closed")

    def _assert_owned_collection(self, name: str) -> dict[str, Any]:
        raw = self.vdb.get_collection(scope=self.settings.scope, collection_name=name)
        item = _model_dict(raw)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        marker = metadata.get(_ADAPTER_MARKER)
        if marker != self.ownership_marker:
            raise NemoRetrieverLocalOwnershipError(
                f"Collection {name!r} was created by a different adapter, extraction profile, or embedding model. "
                "Use a matching configuration or a new collection name."
            )
        return item

    def create_collection(
        self,
        name: str,
        description: str | None,
        metadata: dict[str, Any] | None,
    ) -> CollectionInfo:
        self._ensure_open()
        public_metadata = scrub_metadata(metadata or {})
        public_metadata[_ADAPTER_MARKER] = self.ownership_marker
        request = self.bindings.CollectionCreateRequest(
            name=name,
            description=description,
            metadata=public_metadata,
            expires_at=(datetime.now(UTC) + timedelta(hours=self.settings.collection_ttl_hours)).isoformat(),
        )
        try:
            item = _model_dict(self.vdb.create_collection(scope=self.settings.scope, request=request))
        except self.bindings.VDBResourceConflict:
            item = self._assert_owned_collection(name)
        return self._collection_info(item)

    def get_collection(self, name: str) -> CollectionInfo | None:
        self._ensure_open()
        try:
            item = self._assert_owned_collection(name)
            if not _is_active_collection(item):
                return None
            return self._collection_info(item)
        except (self.bindings.VDBInvalidRequest, self.bindings.VDBResourceNotFound):
            return None

    def list_collections(self) -> list[CollectionInfo]:
        self._ensure_open()
        items: list[CollectionInfo] = []
        token: str | None = None
        seen: set[str] = set()
        while True:
            page = self.vdb.list_collections(
                scope=self.settings.scope,
                limit=_PAGE_SIZE,
                continuation_token=token,
            )
            payload = _model_dict(page)
            for raw in payload.get("items", []):
                item = _model_dict(raw)
                if not _is_active_collection(item):
                    continue
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                if metadata.get(_ADAPTER_MARKER) != self.ownership_marker:
                    raise NemoRetrieverLocalOwnershipError(
                        f"Collection {item.get('name')!r} is incompatible with this local adapter configuration"
                    )
                try:
                    items.append(self._collection_info(item))
                except (self.bindings.VDBInvalidRequest, self.bindings.VDBResourceNotFound):
                    # Reconciliation can transition an item after the catalog
                    # page was read but before its document counts are loaded.
                    continue
            token = _optional_text(payload.get("next_token"))
            if token is None:
                return items
            if token in seen:
                raise NemoRetrieverLocalError("NeMo Retriever collection pagination repeated a continuation token")
            seen.add(token)

    def delete_collection(self, name: str) -> bool:
        with self._submit_lock:
            self._ensure_open()
            try:
                self._assert_owned_collection(name)
            except self.bindings.VDBResourceNotFound:
                return True
            with self._state_lock:
                if any(collection_name == name for collection_name, _document_id in self._active_writes):
                    raise NemoRetrieverLocalError(
                        f"Collection {name!r} still has documents being ingested; "
                        "retry deletion after all jobs are terminal"
                    )
            result = _model_dict(
                self.vdb.delete_collection(
                    scope=self.settings.scope,
                    collection_name=name,
                    if_exists=True,
                )
            )
        return bool(result.get("deleted") or not result.get("existed", True) or result.get("cleanup_pending"))

    def _collection_info(self, item: dict[str, Any]) -> CollectionInfo:
        name = str(item["name"])
        files = self.list_files(name, validate_ownership=False)
        return CollectionInfo(
            name=name,
            description=item.get("description"),
            file_count=len(files),
            chunk_count=sum(file.chunk_count for file in files),
            created_at=_timestamp(item.get("created_at")),
            updated_at=_timestamp(item.get("updated_at")),
            backend=_BACKEND_NAME,
            metadata=scrub_metadata(item.get("metadata") or {}),
        )

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None,
    ) -> str:
        """Stage uploads and enqueue bounded one-file work without extracting inline."""
        self._ensure_open()
        self._assert_owned_collection(collection_name)
        job_config = dict(config or {})
        original_filenames = [str(value) for value in job_config.get("original_filenames", [])]
        cleanup_files = strict_bool(job_config.get("cleanup_files", False), name="cleanup_files")
        metadata = scrub_metadata(job_config.get("metadata") or {})
        if not file_paths:
            raise ValueError("At least one file is required for NeMo Retriever ingestion")
        job_id = uuid.uuid4().hex
        job_dir = self._staging_root / job_id
        _safe_mkdir(job_dir)
        staged_files: list[_StagedFile] = []
        source_paths: list[Path] = []
        document_ids: set[str] = set()
        try:
            for position, raw_path in enumerate(file_paths):
                source = Path(raw_path)
                if not source.is_file():
                    raise ValueError(f"Ingestion file does not exist or is not a regular file: {source}")
                supplied = original_filenames[position] if position < len(original_filenames) else source.name
                filename = Path(supplied).name or source.name
                # The upstream planner validates the manifest by extension;
                # preserve that file-family signal while buffers() retains the
                # exact user-facing basename for extraction and citations.
                staged_path = job_dir / f"{position:04d}{Path(filename).suffix.lower()}"
                shutil.copyfile(source, staged_path)
                content_sha256 = _sha256_file(staged_path)
                # Keep the manifest identity independent of request batching so
                # retrying one failed file outside its original batch targets
                # the same NRL document and chunk identities.
                manifest_entry = hashlib.sha256(f"{filename}\0{content_sha256}".encode()).hexdigest()
                document_id = hashlib.sha256(
                    f"{self.settings.scope}\0{collection_name}\0{manifest_entry}\0{content_sha256}".encode()
                ).hexdigest()
                if document_id in document_ids:
                    raise ValueError(
                        f"The upload batch contains the same logical document more than once: {filename!r}"
                    )
                document_ids.add(document_id)
                try:
                    staged_path.chmod(0o600)
                except OSError:
                    logger.debug("Could not tighten permissions on staged upload %s", staged_path)
                staged_files.append(
                    _StagedFile(
                        document_id=document_id,
                        document_version=content_sha256,
                        content_sha256=content_sha256,
                        filename=filename,
                        file_size=staged_path.stat().st_size,
                        path=staged_path,
                        metadata=dict(metadata),
                    )
                )
                source_paths.append(source)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        now = datetime.now(UTC)
        job = _Job(
            job_id=job_id,
            collection_name=collection_name,
            submitted_at=now,
            files=[
                FileProgress(
                    file_id=item.document_id,
                    file_name=item.filename,
                    status=FileStatus.UPLOADING,
                    progress_percent=0.0,
                )
                for item in staged_files
            ],
        )
        try:
            with self._submit_lock:
                if self._closed:
                    raise NemoRetrieverLocalError("The embedded NeMo Retriever runtime is closed")
                # Collection deletion uses the same admission lock. Recheck
                # after staging so a delete that won the race cannot be
                # followed by a queued write into a missing collection.
                self._assert_owned_collection(collection_name)
                if self._queue.maxsize - self._queue.qsize() < len(staged_files):
                    raise NemoRetrieverLocalError("The NeMo Retriever local ingestion queue is full; retry later")
                with self._state_lock:
                    if any((collection_name, item.document_id) in self._active_writes for item in staged_files):
                        raise NemoRetrieverLocalError(
                            "One or more documents are already queued for NeMo Retriever ingestion; "
                            "retry after completion"
                        )
                    self._jobs[job_id] = job
                    self._active_writes.update((collection_name, item.document_id) for item in staged_files)
                for item in staged_files:
                    self._queue.put_nowait(_WorkItem(job_id=job_id, collection_name=collection_name, staged=item))
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            with self._state_lock:
                self._jobs.pop(job_id, None)
                for item in staged_files:
                    self._active_writes.discard((collection_name, item.document_id))
            raise
        if cleanup_files:
            for source in source_paths:
                try:
                    source.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove a temporary upload file after local staging")
        return job_id

    def _worker_loop(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work is None:
                    return
                try:
                    self._process_file(work)
                except Exception as error:
                    # One corrupt process-local item must not stop the sole
                    # consumer and leave every later job pending forever.
                    logger.warning(
                        "Embedded NeMo Retriever worker discarded an invalid item (error_type=%s)",
                        type(error).__name__,
                    )
                    try:
                        self._release_work_item(work)
                    except Exception as cleanup_error:
                        logger.warning(
                            "Embedded NeMo Retriever worker cleanup failed (error_type=%s)",
                            type(cleanup_error).__name__,
                        )
            finally:
                self._queue.task_done()

    def _process_file(self, work: _WorkItem) -> None:
        job: _Job | None = None
        progress: FileProgress | None = None
        now = datetime.now(UTC)
        try:
            with self._state_lock:
                job = self._jobs.get(work.job_id)
                if job is None:
                    logger.warning("Discarding stale embedded NeMo Retriever work for an unknown job")
                    return
                progress = next(
                    (item for item in job.files if item.file_id == work.staged.document_id),
                    None,
                )
                if progress is None:
                    # Submission records progress and queues work atomically;
                    # preserve any other entries because they own valid items.
                    logger.warning("Discarding embedded NeMo Retriever work with no matching file progress")
                    return
                job.started_at = job.started_at or now
                progress.status = FileStatus.INGESTING
                progress.progress_percent = 10.0
            dataframe = self._extract_and_embed(work.staged)
            ingest_data = self._apply_file_metadata(dataframe, work.staged)
            result = self.ingest_operator.run(
                ingest_data,
                collection_context=self.bindings.CollectionWriteContext(
                    scope=self.settings.scope,
                    collection_name=work.collection_name,
                    document_id=work.staged.document_id,
                    document_version=work.staged.document_version,
                    content_sha256=work.staged.content_sha256,
                    filename=work.staged.filename,
                    job_id=work.job_id,
                    operation=self.bindings.IngestOperation.APPEND,
                ),
            )
            written = max(0, int(getattr(result, "written", 0)))
            if written == 0:
                raise NemoRetrieverLocalError(
                    f"NeMo Retriever extracted no chunks from {work.staged.filename!r}; the document was not committed"
                )
            with self._state_lock:
                progress.status = FileStatus.SUCCESS
                progress.progress_percent = 100.0
                progress.chunks_created = written
        except Exception as error:
            logger.warning(
                "Embedded NeMo Retriever ingestion failed for %s: %s",
                work.staged.filename,
                self.public_error(error),
            )
            with self._state_lock:
                if progress is not None:
                    progress.status = FileStatus.FAILED
                    progress.progress_percent = 100.0
                    progress.error_message = self.public_error(error)
        finally:
            self._release_work_item(work, job)

    def _release_work_item(self, work: _WorkItem, job: _Job | None = None) -> None:
        """Release private staging and active-write state for one queued file."""
        try:
            work.staged.path.unlink(missing_ok=True)
            work.staged.path.parent.rmdir()
        except OSError:
            logger.debug("Staging directory still contains queued work for job %s", work.job_id)
        with self._state_lock:
            if job is not None:
                self._finalize_job_if_terminal(job)
            self._active_writes.discard((work.collection_name, work.staged.document_id))

    def _extract_and_embed(self, staged: _StagedFile) -> Any:
        inference_api_key = _secret_value(self._inference_api_key)
        plan = self.bindings.resolve_ingest_plan(
            self.bindings.IngestPlanRequest(
                source=self.bindings.IngestSourceOptions(
                    documents=[str(staged.path)],
                    profile=self.settings.profile,
                    input_type="auto",
                ),
                runtime=self.bindings.IngestRuntimeOptions(run_mode="inprocess"),
                extract=self.bindings.IngestExtractOptions(
                    page_elements_invoke_url=self.settings.page_elements_invoke_url,
                    ocr_invoke_url=self.settings.ocr_invoke_url,
                    table_structure_invoke_url=self.settings.table_structure_invoke_url,
                    extract_api_key=inference_api_key,
                ),
                embed=self.bindings.IngestEmbedOptions(
                    embed_invoke_url=self._embedding_endpoint,
                    embed_model_name=self._embedding_model,
                    embed_model_provider_prefix=self.settings.embed_model_provider_prefix,
                    embed_api_key=inference_api_key,
                ),
            )
        )
        extract_call_kwargs = plan.extract_call_kwargs()
        if plan.split_config is not None:
            extract_call_kwargs["split_config"] = plan.split_config
        ingestor = self.bindings.create_ingestor(**plan.create_kwargs)
        return (
            ingestor.buffers((staged.filename, BytesIO(staged.path.read_bytes())))
            .extract(plan.extract_params, **extract_call_kwargs)
            .embed(plan.embed_params)
            .ingest()
        )

    def _apply_file_metadata(self, dataframe: Any, staged: _StagedFile) -> Any:
        if not staged.metadata:
            return dataframe
        records = self.bindings.to_client_vdb_records(dataframe)
        fields = list(staged.metadata)
        join_field = "__deep_researcher_source"
        while join_field in staged.metadata:
            join_field = f"_{join_field}"
        sidecar = self.bindings.pandas.DataFrame([{**staged.metadata, join_field: staged.filename}])
        return self.bindings.apply_sidecar_metadata_to_client_batches(
            records,
            meta_df=sidecar,
            meta_source_field=join_field,
            meta_fields=fields,
            join_key="auto",
        )

    def public_error(self, error: Exception) -> str:
        """Return a bounded error without credentials or NRL storage selectors."""
        message = str(error).strip() or type(error).__name__
        sensitive_values = (
            _secret_value(self._inference_api_key),
            str(self.settings.data_dir),
            self._embedding_endpoint,
            self.settings.embed_invoke_url,
            self.settings.page_elements_invoke_url,
            self.settings.ocr_invoke_url,
            self.settings.table_structure_invoke_url,
        )
        for value in sensitive_values:
            if value:
                message = message.replace(value, "[redacted]")
        message = _PHYSICAL_TABLE_PATTERN.sub("[redacted-table]", message)
        return message[:1024]

    def _finalize_job_if_terminal(self, job: _Job) -> None:
        terminal = [item for item in job.files if item.status in _TERMINAL_FILE_STATUSES]
        if len(terminal) != len(job.files):
            return
        job.completed_at = datetime.now(UTC)
        failures = [item for item in terminal if item.status == FileStatus.FAILED]
        if len(failures) == len(job.files):
            job.error_message = "All files failed NeMo Retriever ingestion"
        elif failures:
            job.error_message = "NeMo Retriever ingestion completed with one or more failed files"

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return IngestionJobStatus(
                    job_id=job_id,
                    status=JobState.FAILED,
                    submitted_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    collection_name="unknown",
                    backend=_BACKEND_NAME,
                    error_message="Job ID not found; local job history does not survive process restart",
                )
            files = [item.model_copy(deep=True) for item in job.files]
            terminal = [item for item in files if item.status in _TERMINAL_FILE_STATUSES]
            successes = [item for item in files if item.status == FileStatus.SUCCESS]
            if len(terminal) == len(files):
                state = JobState.COMPLETED if successes else JobState.FAILED
            elif job.started_at is not None:
                state = JobState.PROCESSING
            else:
                state = JobState.PENDING
            return IngestionJobStatus(
                job_id=job.job_id,
                status=state,
                submitted_at=job.submitted_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                total_files=len(files),
                processed_files=len(terminal),
                file_details=files,
                collection_name=job.collection_name,
                backend=_BACKEND_NAME,
                error_message=job.error_message,
                metadata={"profile": self.settings.profile},
            )

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None,
    ) -> FileInfo:
        job_id = self.submit_job(
            [file_path],
            collection_name,
            {"original_filenames": [Path(file_path).name], "metadata": metadata or {}},
        )
        status = self.get_job_status(job_id)
        progress = status.file_details[0]
        return FileInfo(
            file_id=progress.file_id,
            file_name=progress.file_name,
            collection_name=collection_name,
            status=progress.status,
            file_size=Path(file_path).stat().st_size,
            uploaded_at=status.submitted_at,
            metadata={"job_id": job_id},
        )

    def list_files(self, collection_name: str, *, validate_ownership: bool = True) -> list[FileInfo]:
        self._ensure_open()
        if validate_ownership:
            self._assert_owned_collection(collection_name)
        items: list[FileInfo] = []
        token: str | None = None
        seen: set[str] = set()
        while True:
            page = self.vdb.list_documents(
                scope=self.settings.scope,
                collection_name=collection_name,
                limit=_PAGE_SIZE,
                continuation_token=token,
            )
            payload = _model_dict(page)
            items.extend(self._file_info(_model_dict(item)) for item in payload.get("items", []))
            token = _optional_text(payload.get("next_token"))
            if token is None:
                return items
            if token in seen:
                raise NemoRetrieverLocalError("NeMo Retriever document pagination repeated a continuation token")
            seen.add(token)

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        self._ensure_open()
        self._assert_owned_collection(collection_name)
        try:
            item = self.vdb.get_document(
                scope=self.settings.scope,
                collection_name=collection_name,
                document_id=file_id,
            )
        except self.bindings.VDBResourceNotFound:
            with self._state_lock:
                for job in self._jobs.values():
                    if job.collection_name != collection_name:
                        continue
                    for progress in job.files:
                        if progress.file_id == file_id:
                            return FileInfo(
                                file_id=file_id,
                                file_name=progress.file_name,
                                collection_name=collection_name,
                                status=progress.status,
                                chunk_count=progress.chunks_created,
                                uploaded_at=job.submitted_at,
                                ingested_at=job.completed_at if progress.status == FileStatus.SUCCESS else None,
                                error_message=progress.error_message,
                                metadata={"job_id": job.job_id},
                            )
            return None
        return self._file_info(_model_dict(item))

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        # Serialize deletion with submit_job so a retry cannot pass the active
        # write check while this document is being removed.
        with self._submit_lock:
            self._ensure_open()
            self._assert_owned_collection(collection_name)
            with self._state_lock:
                if (collection_name, file_id) in self._active_writes:
                    raise NemoRetrieverLocalError(
                        f"Document {file_id!r} is still being ingested; retry deletion after the job is terminal"
                    )
            result = self.vdb.delete_document(
                scope=self.settings.scope,
                collection_name=collection_name,
                document_id=file_id,
                if_exists=True,
            )
        payload = _model_dict(result)
        return bool(payload.get("deleted") or not payload.get("existed", True) or payload.get("cleanup_pending"))

    def _file_info(self, item: dict[str, Any]) -> FileInfo:
        status = status_to_file_status(str(item.get("status") or ""))
        raw_error = _optional_text(item.get("error"))
        return FileInfo(
            file_id=str(item["document_id"]),
            file_name=str(item["filename"]),
            collection_name=str(item["collection_name"]),
            status=status,
            chunk_count=max(0, int(item.get("chunk_count") or 0)),
            uploaded_at=_timestamp(item.get("created_at")),
            ingested_at=_timestamp(item.get("updated_at")) if status == FileStatus.SUCCESS else None,
            error_message=self.public_error(NemoRetrieverLocalError(raw_error)) if raw_error else None,
            metadata={
                "content_sha256": item.get("content_sha256"),
                "document_version": item.get("document_version"),
                "job_id": item.get("job_id"),
                "nrl_status": item.get("status"),
            },
        )

    def query(self, query: str, collection_name: str, top_k: int) -> list[dict[str, Any]]:
        self._ensure_open()
        self._assert_owned_collection(collection_name)
        vectors = self.bindings.infer_microservice(
            [query],
            model_name=self._embedding_model,
            embedding_endpoint=self._embedding_endpoint,
            nvidia_api_key=_secret_value(self._inference_api_key),
            input_type="query",
            model_provider_prefix=self.settings.embed_model_provider_prefix,
            grpc=False,
        )
        result = self.retrieve_operator.run(
            vectors,
            scope=self.settings.scope,
            collection_name=collection_name,
            query_texts=[query],
            top_k=top_k,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise NemoRetrieverLocalError("NeMo Retriever collection query returned an invalid result contract")
        hits_by_query, _strategies = result
        if not isinstance(hits_by_query, list) or len(hits_by_query) != 1:
            raise NemoRetrieverLocalError("NeMo Retriever query returned an unexpected number of result sets")
        return list(hits_by_query[0])

    def health_check(self) -> bool:
        if self._closed:
            return False
        try:
            health = self.vdb.health()
        except Exception:
            return False
        if not isinstance(health, dict):
            return False
        catalog = health.get("catalog")
        return not isinstance(catalog, dict) or catalog.get("healthy", True) is True


@dataclass
class _RuntimeEntry:
    runtime: LocalRuntime
    compatibility_key: tuple[Any, ...]
    references: int = 0


_RUNTIMES: dict[Path, _RuntimeEntry] = {}
_RUNTIMES_CLOSING: dict[Path, threading.Event] = {}
_RUNTIMES_LOCK = threading.Lock()


class LocalRuntimeHandle:
    """Reference-counted lifecycle handle used by NAT function finalizers."""

    def __init__(self, key: Path, runtime: LocalRuntime):
        self._key = key
        self.runtime = runtime
        self._closed = False
        self._close_lock = threading.Lock()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            runtime: LocalRuntime | None = None
            closing: threading.Event | None = None
            with _RUNTIMES_LOCK:
                entry = _RUNTIMES.get(self._key)
                if entry is None:
                    return
                entry.references -= 1
                if entry.references == 0:
                    runtime = entry.runtime
                    del _RUNTIMES[self._key]
                    closing = threading.Event()
                    _RUNTIMES_CLOSING[self._key] = closing
            if runtime is not None:
                assert closing is not None
                try:
                    runtime.close()
                finally:
                    with _RUNTIMES_LOCK:
                        if _RUNTIMES_CLOSING.get(self._key) is closing:
                            del _RUNTIMES_CLOSING[self._key]
                        closing.set()


def acquire_local_runtime(config: dict[str, Any]) -> LocalRuntimeHandle:
    """Share one runtime per canonical data directory within this process."""
    settings = LocalSettings.from_config(config)
    key = settings.data_dir
    bindings = config.get("_bindings")
    while True:
        with _RUNTIMES_LOCK:
            closing = _RUNTIMES_CLOSING.get(key)
            if closing is None:
                entry = _RUNTIMES.get(key)
                if entry is None:
                    runtime = LocalRuntime(settings, bindings=bindings)
                    entry = _RuntimeEntry(runtime=runtime, compatibility_key=settings.compatibility_key)
                    _RUNTIMES[key] = entry
                elif entry.compatibility_key != settings.compatibility_key:
                    raise NemoRetrieverLocalError(
                        f"The NeMo Retriever local data directory {key} is already open with different settings"
                    )
                entry.references += 1
                return LocalRuntimeHandle(key, entry.runtime)
        # The previous runtime still owns the process file lock. Wait without
        # blocking acquisitions for unrelated data directories, then retry.
        closing.wait()


__all__ = [
    "LocalRuntime",
    "LocalRuntimeHandle",
    "LocalSettings",
    "NemoRetrieverLocalDependencyError",
    "NemoRetrieverLocalError",
    "NemoRetrieverLocalLockError",
    "NemoRetrieverLocalOwnershipError",
    "acquire_local_runtime",
]
