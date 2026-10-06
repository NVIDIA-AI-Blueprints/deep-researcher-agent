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

"""Durable artifact storage.

Artifact metadata is stored in an ``artifacts`` table on the shared job ``db_url``
(the same database used by the job/event stores), keyed by ``artifact_id``. Artifact
bytes are stored either in the table's ``content`` BLOB column or in the configured
S3-compatible object store. The Dask worker (writer) and API process (reader) use the
same configured storage provider, so no shared filesystem is required.

- Metadata columns are queryable and listed in the UI/CLI.
- SQL BLOB storage remains the default for local development.
- Production deployments can store bytes in S3-compatible object storage without
  changing callers.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from ..logging_utils import log_sandbox_failure
from .blob_store import ArtifactBlobStore
from .blob_store import SqlArtifactBlobStore
from .models import Artifact
from .models import ArtifactProvenance
from .models import ArtifactStatus

logger = logging.getLogger(__name__)


def _normalize_db_url(db_url: str) -> str:
    """Normalize a SQLAlchemy URL to a sync driver (mirrors the event store).

    Artifacts use sync SQLAlchemy; the async API path wraps reads in an executor.
    """
    if db_url.startswith("postgresql") or db_url.startswith("postgres"):
        base = db_url.replace("+asyncpg", "").replace("+psycopg2", "").replace("+psycopg", "")
        base = base.replace("postgres://", "postgresql://")
        return base.replace("postgresql://", "postgresql+psycopg://")
    if db_url.startswith("sqlite"):
        return db_url.replace("+aiosqlite", "")
    return db_url


class ArtifactStore(ABC):
    """Pluggable durable store for artifact metadata and bytes."""

    @abstractmethod
    def put(self, artifact: Artifact, data: bytes) -> Artifact:
        """Persist bytes and metadata, returning the stored (possibly deduped) record."""

    @abstractmethod
    def open_bytes(self, job_id: str, artifact_id: str) -> Iterator[bytes]:
        """Stream an artifact's bytes for the content endpoint or PDF embedding."""

    @abstractmethod
    def get(self, job_id: str, artifact_id: str) -> Artifact | None:
        """Return artifact metadata, or ``None`` if not found for this job."""

    @abstractmethod
    def find_by_digest(self, job_id: str, sha256: str) -> Artifact | None:
        """Return an existing artifact with the same content digest for this job."""

    @abstractmethod
    def list(self, job_id: str) -> list[Artifact]:
        """List all artifacts owned by a job."""

    @abstractmethod
    def delete_job(self, job_id: str) -> int:
        """Delete all artifacts for a job (retention/expiry). Returns count removed."""

    @abstractmethod
    def cleanup_old_artifacts(self, retention_seconds: int) -> int:
        """Delete artifacts older than the retention period. Returns count removed."""

    @abstractmethod
    def validate(self) -> None:
        """Raise when the configured artifact storage is unavailable."""


class SqlArtifactStore(ArtifactStore):
    """SQL-backed metadata store (SQLite/Postgres) on the shared job ``db_url``.

    Bytes use the configured blob store. Dedup is per-job by content digest so
    harvest is idempotent across job retries.
    """

    _engines: dict[str, Any] = {}
    _initialized: set[str] = set()
    _lock = threading.Lock()

    def __init__(
        self,
        db_url: str = "sqlite+aiosqlite:///./jobs.db",
        *,
        blob_store: ArtifactBlobStore | None = None,
    ) -> None:
        """Bind to the shared job database and ensure the artifacts table exists.

        Args:
            db_url: SQLAlchemy URL of the shared job/event database.
        """
        self.db_url = db_url
        self._engine = self._get_engine(db_url)
        self._ensure_table()
        self._blob_store = blob_store or SqlArtifactBlobStore(self._engine)

    @classmethod
    def _get_engine(cls, db_url: str) -> Any:
        """Return a process-wide engine for the URL, creating it once (thread-safe)."""
        from sqlalchemy import create_engine
        from sqlalchemy import event

        with cls._lock:
            if db_url in cls._engines:
                return cls._engines[db_url]
            normalized = _normalize_db_url(db_url)
            connect_args = {"check_same_thread": False, "timeout": 30} if normalized.startswith("sqlite") else {}
            engine = create_engine(normalized, pool_pre_ping=True, pool_recycle=1800, connect_args=connect_args)
            if normalized.startswith("sqlite"):
                # WAL improves concurrency between the worker writer and API reader.
                @event.listens_for(engine, "connect")
                def _set_wal(dbapi_conn: Any, _record: Any) -> None:  # pragma: no cover - driver callback
                    """Enable SQLite WAL mode on connect for reader/writer concurrency."""
                    cursor = dbapi_conn.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.close()

            cls._engines[db_url] = engine
            return engine

    def _ensure_table(self) -> None:
        """Create the ``artifacts`` table on first use for this database URL."""
        if self.db_url in SqlArtifactStore._initialized:
            return
        from sqlalchemy import BigInteger
        from sqlalchemy import Boolean
        from sqlalchemy import Column
        from sqlalchemy import DateTime
        from sqlalchemy import Index
        from sqlalchemy import LargeBinary
        from sqlalchemy import MetaData
        from sqlalchemy import String
        from sqlalchemy import Table
        from sqlalchemy import Text
        from sqlalchemy import inspect
        from sqlalchemy.sql import func

        metadata = MetaData()
        Table(
            "artifacts",
            metadata,
            Column("artifact_id", String(64), primary_key=True),
            Column("job_id", String(64), nullable=False, index=True),
            Column("kind", String(32), nullable=False),
            Column("mime_type", String(128), nullable=False),
            Column("filename", String(512), nullable=False),
            Column("sandbox_path", Text, nullable=False),
            Column("storage_uri", Text, nullable=False),
            Column("sha256", String(64), nullable=False),
            Column("size_bytes", BigInteger, nullable=False),
            Column("title", Text, nullable=True),
            Column("caption", Text, nullable=True),
            Column("inline", Boolean, nullable=False, default=False),
            Column("workflow", String(64), nullable=True),
            Column("source_tool_call_id", String(128), nullable=True),
            Column("provenance", Text, nullable=True),
            Column("status", String(16), nullable=False),
            Column("content", LargeBinary, nullable=True),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Index("idx_artifacts_job_sha", "job_id", "sha256"),
        )
        inspector = inspect(self._engine)
        if not inspector.has_table("artifacts"):
            metadata.create_all(self._engine)
            logger.info("Created artifacts table (backend=%s)", self._engine.dialect.name)
        SqlArtifactStore._initialized.add(self.db_url)

    def put(self, artifact: Artifact, data: bytes) -> Artifact:
        """Persist bytes and metadata, deduping per job by content digest.

        Args:
            artifact: Metadata for the artifact to store.
            data: Raw artifact bytes.

        Returns:
            The stored record, or the existing record on a digest dedup hit.
        """
        existing = self.find_by_digest(artifact.job_id, artifact.sha256)
        if existing is not None:
            logger.debug("Artifact dedup hit for job=%s sha=%s", artifact.job_id, artifact.sha256[:12])
            return existing

        prepared = artifact.model_copy(
            update={
                # Logical location only — never embed db_url (it may carry credentials).
                "storage_uri": self._blob_store.make_uri(artifact),
            }
        )
        try:
            content = self._blob_store.put(prepared, data)
            stored = prepared.model_copy(update={"status": ArtifactStatus.AVAILABLE})
            self._insert_metadata(stored, content=content)
        except Exception as exc:
            log_sandbox_failure(
                logger,
                operation="artifact_store",
                reason_code="artifact_storage_failed",
                exc=exc,
                sandbox=prepared.job_id,
            )
            try:
                self._blob_store.delete(prepared)
            except Exception as cleanup_exc:
                log_sandbox_failure(
                    logger,
                    operation="artifact_store_rollback",
                    reason_code="artifact_rollback_failed",
                    exc=cleanup_exc,
                    sandbox=prepared.job_id,
                )
            raise

        return stored

    def _insert_metadata(self, artifact: Artifact, *, content: bytes | None) -> None:
        """Insert one artifact metadata row and optional SQL-backed content."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO artifacts (artifact_id, job_id, kind, mime_type, filename, sandbox_path, "
                    "storage_uri, sha256, size_bytes, title, caption, inline, workflow, source_tool_call_id, "
                    "provenance, status, content) VALUES (:artifact_id, :job_id, :kind, :mime_type, :filename, "
                    ":sandbox_path, :storage_uri, :sha256, :size_bytes, :title, :caption, :inline, :workflow, "
                    ":source_tool_call_id, :provenance, :status, :content)"
                ),
                {
                    "artifact_id": artifact.artifact_id,
                    "job_id": artifact.job_id,
                    "kind": artifact.kind.value,
                    "mime_type": artifact.mime_type,
                    "filename": artifact.filename,
                    "sandbox_path": artifact.sandbox_path,
                    "storage_uri": artifact.storage_uri,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                    "title": artifact.title,
                    "caption": artifact.caption,
                    "inline": artifact.inline,
                    "workflow": artifact.workflow,
                    "source_tool_call_id": artifact.source_tool_call_id,
                    "provenance": artifact.provenance.model_dump_json(),
                    "status": artifact.status.value,
                    "content": content,
                },
            )
            conn.commit()

    def open_bytes(self, job_id: str, artifact_id: str) -> Iterator[bytes]:
        """Yield an artifact's bytes from the configured provider, or nothing if not found."""
        artifact = self.get(job_id, artifact_id)
        if artifact is None:
            return
        yield from self._blob_store.open_bytes(artifact)

    def get(self, job_id: str, artifact_id: str) -> Artifact | None:
        """Return artifact metadata for the job, or ``None`` if not found."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        f"SELECT {_META_COLUMNS} FROM artifacts WHERE job_id = :job_id AND artifact_id = :artifact_id"
                    ),
                    {"job_id": job_id, "artifact_id": artifact_id},
                )
                .mappings()
                .fetchone()
            )
        return _row_to_artifact(row) if row else None

    def find_by_digest(self, job_id: str, sha256: str) -> Artifact | None:
        """Return an existing artifact with the same digest for the job, if any."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(f"SELECT {_META_COLUMNS} FROM artifacts WHERE job_id = :job_id AND sha256 = :sha256 LIMIT 1"),
                    {"job_id": job_id, "sha256": sha256},
                )
                .mappings()
                .fetchone()
            )
        return _row_to_artifact(row) if row else None

    def list(self, job_id: str) -> list[Artifact]:
        """Return all artifacts owned by the job, ordered by creation time."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(f"SELECT {_META_COLUMNS} FROM artifacts WHERE job_id = :job_id ORDER BY created_at"),
                    {"job_id": job_id},
                )
                .mappings()
                .fetchall()
            )
        return [_row_to_artifact(row) for row in rows]

    def delete_job(self, job_id: str) -> int:
        """Delete a job's bytes first and remove metadata after each success."""
        return sum(self._delete_artifact(artifact) for artifact in self.list(job_id))

    def cleanup_old_artifacts(self, retention_seconds: int) -> int:
        """Delete artifacts older than the retention window and return the count.

        A non-positive retention is refused (returns 0) to avoid deleting everything.
        """
        from sqlalchemy import text

        # A non-positive retention would make the cutoff "now or later" and delete everything.
        if retention_seconds <= 0:
            logger.warning("Refusing artifact cleanup with non-positive retention_seconds=%s", retention_seconds)
            return 0
        is_postgres = _normalize_db_url(self.db_url).startswith("postgresql")
        with self._engine.connect() as conn:
            if is_postgres:
                rows = (
                    conn.execute(
                        text(
                            f"SELECT {_META_COLUMNS} FROM artifacts "
                            "WHERE created_at < NOW() - :seconds * INTERVAL '1 second'"
                        ),
                        {"seconds": retention_seconds},
                    )
                    .mappings()
                    .fetchall()
                )
            else:
                rows = (
                    conn.execute(
                        text(f"SELECT {_META_COLUMNS} FROM artifacts WHERE created_at < datetime('now', :interval)"),
                        {"interval": f"-{retention_seconds} seconds"},
                    )
                    .mappings()
                    .fetchall()
                )
        return sum(self._delete_artifact(_row_to_artifact(row)) for row in rows)

    def validate(self) -> None:
        """Validate the configured byte storage."""
        self._blob_store.validate()

    def _delete_artifact(self, artifact: Artifact) -> int:
        """Delete one artifact's bytes and then its metadata row."""
        from sqlalchemy import text

        try:
            self._blob_store.delete(artifact)
        except Exception as exc:
            log_sandbox_failure(
                logger,
                operation="artifact_byte_delete",
                reason_code="artifact_delete_failed",
                exc=exc,
                sandbox=artifact.job_id,
            )
            return 0
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("DELETE FROM artifacts WHERE job_id = :job_id AND artifact_id = :artifact_id"),
                    {"job_id": artifact.job_id, "artifact_id": artifact.artifact_id},
                )
                conn.commit()
                return result.rowcount
        except Exception as exc:
            log_sandbox_failure(
                logger,
                operation="artifact_metadata_delete",
                reason_code="artifact_metadata_delete_failed",
                exc=exc,
                sandbox=artifact.job_id,
            )
            return 0


# Backwards-friendly alias; local development and single-node use the same SQL store.
LocalArtifactStore = SqlArtifactStore

_META_COLUMNS = (
    "artifact_id, job_id, kind, mime_type, filename, sandbox_path, storage_uri, sha256, size_bytes, "
    "title, caption, inline, workflow, source_tool_call_id, provenance, status, created_at"
)


def _row_to_artifact(row: Any) -> Artifact:
    """Build an ``Artifact`` from a metadata row, tolerating bad provenance JSON."""
    provenance = ArtifactProvenance()
    raw = row.get("provenance")
    if raw:
        try:
            provenance = ArtifactProvenance.model_validate(json.loads(raw))
        except (ValueError, TypeError):
            logger.warning("Failed to parse artifact provenance for %s", row.get("artifact_id"))
    return Artifact(
        artifact_id=row["artifact_id"],
        job_id=row["job_id"],
        kind=row["kind"],
        mime_type=row["mime_type"],
        filename=row["filename"],
        sandbox_path=row["sandbox_path"],
        storage_uri=row["storage_uri"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        title=row.get("title"),
        caption=row.get("caption"),
        inline=bool(row.get("inline")),
        workflow=row.get("workflow"),
        source_tool_call_id=row.get("source_tool_call_id"),
        provenance=provenance,
        status=row["status"],
        created_at=row["created_at"],
    )
