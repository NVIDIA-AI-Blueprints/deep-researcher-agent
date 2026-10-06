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

"""Event store for real-time SSE streaming.

Uses async SQLAlchemy with psycopg (psycopg3) for PostgreSQL - the same driver
used by LangGraph checkpointer, reducing dependency footprint.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from typing import Any

from deep_researcher_agent.common.logging_utils import log_content_metadata

from .crypto import ContentEncryptionError

logger = logging.getLogger(__name__)


class SQLAlchemyPoolFilter(logging.Filter):
    """
    Filter to suppress expected CancelledError exceptions from SQLAlchemy pool.

    These occur normally when SSE clients disconnect and async tasks are cancelled.
    The errors are benign but noisy in logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            msg = str(record.getMessage())
            if "CancelledError" in msg or "Exception terminating connection" in msg:
                return False
        return True


def configure_sqlalchemy_logging():
    """Apply filter to suppress expected async cancellation errors."""
    pool_logger = logging.getLogger("sqlalchemy.pool")
    pool_logger.addFilter(SQLAlchemyPoolFilter())


configure_sqlalchemy_logging()

ENGINE_CACHE_TTL_SECONDS = 3600
ENGINE_CACHE_MAX_SIZE = 10


def _normalize_db_url(db_url: str, async_mode: bool = True) -> str:
    """
    Normalize database URL to use consistent drivers.

    For PostgreSQL: Uses psycopg (psycopg3) for both sync and async
    For SQLite: Uses aiosqlite for async, standard sqlite for sync
    """
    if db_url.startswith("postgresql") or db_url.startswith("postgres"):
        base_url = db_url.replace("+asyncpg", "").replace("+psycopg2", "").replace("+psycopg", "")
        if not base_url.startswith("postgresql://"):
            base_url = base_url.replace("postgres://", "postgresql://")
        return (
            f"{base_url.replace('postgresql://', 'postgresql+psycopg://')}"
            if async_mode
            else base_url.replace("postgresql://", "postgresql+psycopg://")
        )
    elif db_url.startswith("sqlite"):
        base_url = db_url.replace("+aiosqlite", "")
        return base_url.replace("sqlite:///", "sqlite+aiosqlite:///") if async_mode else base_url
    return db_url


class EventStore:
    """
    Event store for real-time SSE streaming using SQLAlchemy.

    Uses SQLAlchemy with psycopg (psycopg3) for PostgreSQL, consolidating
    on the same driver used by LangGraph checkpointer.

    Features:
    - Automatic SQLite/PostgreSQL support based on db_url
    - Connection pooling with TTL-based cache management
    - Both sync and async operations supported

    PostgreSQL deployments use LISTEN/NOTIFY for real-time push-based events.
    SQLite deployments use polling (SQLite doesn't support pub-sub).
    """

    _async_engine_cache: dict[str, tuple[Any, float]] = {}
    _sync_engine_cache: dict[str, tuple[Any, float]] = {}
    _cache_lock = threading.Lock()
    _tables_initialized: set[str] = set()

    def __init__(
        self,
        db_url: str = "sqlite+aiosqlite:///./jobs.db",
        job_id: str | None = None,
        content_cipher: Any | None = None,
    ):
        self.db_url = db_url
        self.job_id = job_id
        self._content_cipher = content_cipher
        self._fail_closed = bool(content_cipher is not None and content_cipher.manager.config.encrypted)
        self._is_postgres = db_url.startswith("postgresql")
        self._sync_engine = self._get_or_create_sync_engine(db_url)
        self._ensure_table_sync()

    def _prepare_event_for_storage(self, event: dict) -> dict:
        """Encrypt sensitive event fields before persistence when content encryption is enabled."""

        if self.job_id is None or self._content_cipher is None:
            return event
        if event.get("type") != "artifact.update":
            return event
        data = event.get("data")
        if not isinstance(data, dict) or data.get("type") not in {"output", "file"} or "content" not in data:
            return event

        from .crypto import encrypt_event_field
        from .crypto import is_encrypted_event_field

        if is_encrypted_event_field(data["content"]):
            return event

        encrypted_event = copy.deepcopy(event)
        encrypted_event["data"]["content"] = encrypt_event_field(
            self.job_id,
            "artifact.update.data.content",
            encrypted_event["data"].get("content"),
            self._content_cipher,
        )
        return encrypted_event

    @classmethod
    def _prepare_event_for_return(cls, job_id: str, event: dict) -> dict:
        """Decrypt sensitive event fields before returning events to API/SSE callers."""

        if event.get("type") != "artifact.update":
            return event
        data = event.get("data")
        if not isinstance(data, dict) or data.get("type") not in {"output", "file"} or "content" not in data:
            return event

        from .crypto import decrypt_event_field
        from .crypto import is_encrypted_event_field

        if not is_encrypted_event_field(data["content"]):
            return event

        decrypted_event = copy.deepcopy(event)
        decrypted_event["data"]["content"] = decrypt_event_field(
            job_id,
            "artifact.update.data.content",
            decrypted_event["data"].get("content"),
        )
        return decrypted_event

    @classmethod
    def _get_or_create_sync_engine(cls, db_url: str):
        """Get or create a sync SQLAlchemy engine with TTL-based caching."""
        with cls._cache_lock:
            cls._cleanup_stale_engines(cls._sync_engine_cache)

            if db_url in cls._sync_engine_cache:
                engine, _ = cls._sync_engine_cache[db_url]
                cls._sync_engine_cache[db_url] = (engine, time.monotonic())
                return engine

            from sqlalchemy import create_engine

            normalized_url = _normalize_db_url(db_url, async_mode=False)
            connect_args = {}
            if normalized_url.startswith("sqlite"):
                connect_args = {"check_same_thread": False, "timeout": 30}
            else:
                try:
                    import psycopg  # noqa: F401
                except ImportError as e:
                    raise ImportError(
                        "PostgreSQL support requires psycopg (v3). "
                        "Install with: pip install 'deep-researcher-api[postgres]' or pip install 'psycopg[binary]'"
                    ) from e

            engine = create_engine(
                normalized_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_recycle=1800,
                connect_args=connect_args,
                hide_parameters=True,
            )
            cls._sync_engine_cache[db_url] = (engine, time.monotonic())
            logger.debug("Created sync engine for %s", db_url[:50])
            return engine

    @classmethod
    def _get_or_create_async_engine(cls, db_url: str):
        """Get or create an async SQLAlchemy engine with TTL-based caching."""
        with cls._cache_lock:
            cls._cleanup_stale_engines(cls._async_engine_cache)

            if db_url in cls._async_engine_cache:
                engine, _ = cls._async_engine_cache[db_url]
                cls._async_engine_cache[db_url] = (engine, time.monotonic())
                return engine

            from sqlalchemy.ext.asyncio import create_async_engine

            normalized_url = _normalize_db_url(db_url, async_mode=True)

            engine = create_async_engine(
                normalized_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_recycle=1800,
                hide_parameters=True,
            )
            cls._async_engine_cache[db_url] = (engine, time.monotonic())
            logger.debug("Created async engine for %s", db_url[:50])
            return engine

    @classmethod
    def _cleanup_stale_engines(cls, cache: dict[str, tuple[Any, float]]):
        """Remove engines that haven't been used recently."""
        now = time.monotonic()
        stale_keys = [key for key, (_, last_used) in cache.items() if now - last_used > ENGINE_CACHE_TTL_SECONDS]
        for key in stale_keys:
            engine, _ = cache.pop(key, (None, 0))
            if engine:
                cls._dispose_engine(engine, log_debug_key=key)

        if len(cache) > ENGINE_CACHE_MAX_SIZE:
            sorted_entries = sorted(cache.items(), key=lambda x: x[1][1])
            for key, (engine, _) in sorted_entries[: len(sorted_entries) - ENGINE_CACHE_MAX_SIZE]:
                cache.pop(key, None)
                if engine:
                    cls._dispose_engine(engine, log_debug_key=key)

    @classmethod
    def _dispose_engine(cls, engine: Any, log_debug_key: str | None = None) -> None:
        """Dispose a sync or async SQLAlchemy engine safely."""
        try:
            dispose_result = engine.dispose()
        except (RuntimeError, OSError):
            return
        except Exception:
            logger.exception("Failed to dispose engine")
            return

        if log_debug_key:
            logger.debug("Disposed stale engine for %s", log_debug_key[:50])

        if asyncio.iscoroutine(dispose_result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    asyncio.run(dispose_result)
                except (RuntimeError, OSError):
                    pass
            else:
                loop.create_task(dispose_result)

    @classmethod
    async def dispose_all_engines_async(cls):
        """Dispose all cached engines (async-safe shutdown)."""
        async_engines: list[Any] = []
        with cls._cache_lock:
            for _, (engine, _) in list(cls._sync_engine_cache.items()):
                try:
                    engine.dispose()
                except (RuntimeError, OSError):
                    pass
            cls._sync_engine_cache.clear()

            async_engines = [engine for _, (engine, _) in list(cls._async_engine_cache.items())]
            cls._async_engine_cache.clear()
            cls._tables_initialized.clear()

        for engine in async_engines:
            try:
                await engine.dispose()
            except (RuntimeError, OSError):
                pass

    @classmethod
    def dispose_all_engines(cls):
        """Dispose all cached engines (for shutdown). Prefer async in event loops."""
        async_engines: list[Any] = []
        with cls._cache_lock:
            for _, (engine, _) in list(cls._sync_engine_cache.items()):
                try:
                    engine.dispose()
                except (RuntimeError, OSError):
                    pass
            cls._sync_engine_cache.clear()

            async_engines = [engine for _, (engine, _) in list(cls._async_engine_cache.items())]
            cls._async_engine_cache.clear()
            cls._tables_initialized.clear()

        if not async_engines:
            return

        for engine in async_engines:
            cls._dispose_engine(engine)

    def _ensure_table_sync(self):
        """Create events table if it doesn't exist (sync version)."""
        if self.db_url in EventStore._tables_initialized:
            return

        from sqlalchemy import Column
        from sqlalchemy import DateTime
        from sqlalchemy import Index
        from sqlalchemy import Integer
        from sqlalchemy import MetaData
        from sqlalchemy import String
        from sqlalchemy import Table
        from sqlalchemy import Text
        from sqlalchemy import inspect
        from sqlalchemy.sql import func

        metadata = MetaData()

        Table(
            "job_events",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("job_id", String(64), nullable=False, index=True),
            Column("event_type", String(64), nullable=False),
            Column("event_data", Text, nullable=True),
            Column("created_at", DateTime, server_default=func.now()),
            Index("idx_job_events_job_id_id", "job_id", "id"),
        )

        inspector = inspect(self._sync_engine)
        if not inspector.has_table("job_events"):
            metadata.create_all(self._sync_engine)
            logger.info("Created job_events table in %s", self.db_url[:50])

        EventStore._tables_initialized.add(self.db_url)

    @classmethod
    async def _ensure_table_async(cls, db_url: str):
        """Ensure job_events table exists (async version)."""
        if db_url in cls._tables_initialized:
            return

        from sqlalchemy import Column
        from sqlalchemy import DateTime
        from sqlalchemy import Index
        from sqlalchemy import Integer
        from sqlalchemy import MetaData
        from sqlalchemy import String
        from sqlalchemy import Table
        from sqlalchemy import Text
        from sqlalchemy.sql import func

        engine = cls._get_or_create_async_engine(db_url)
        metadata = MetaData()

        Table(
            "job_events",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("job_id", String(64), nullable=False, index=True),
            Column("event_type", String(64), nullable=False),
            Column("event_data", Text, nullable=True),
            Column("created_at", DateTime, server_default=func.now()),
            Index("idx_job_events_job_id_id", "job_id", "id"),
        )

        async with engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: metadata.create_all(sync_conn))

        cls._tables_initialized.add(db_url)
        logger.info("Created job_events table (async) in %s", db_url[:50])

    def store(self, event: dict):
        """
        Store an event and notify listeners (PostgreSQL only).

        For PostgreSQL deployments, issues NOTIFY on a job-specific channel
        to enable real-time push-based SSE streaming with sub-10ms latency.
        SQLite deployments rely on polling.
        """
        import json

        from sqlalchemy import text

        event_type = event.get("type", "unknown")
        stored_event = self._prepare_event_for_storage(event)
        event_json = json.dumps(stored_event)

        try:
            with self._sync_engine.connect() as conn:
                if self._is_postgres:
                    result = conn.execute(
                        text(
                            "INSERT INTO job_events (job_id, event_type, event_data) "
                            "VALUES (:job_id, :event_type, :event_data) RETURNING id"
                        ),
                        {
                            "job_id": self.job_id,
                            "event_type": event_type,
                            "event_data": event_json,
                        },
                    )
                    event_id = result.scalar()

                    channel = f"job_events_{self.job_id.replace('-', '_')}"
                    payload = json.dumps({"id": event_id, "type": event_type})
                    conn.execute(text("SELECT pg_notify(:channel, :payload)"), {"channel": channel, "payload": payload})
                else:
                    conn.execute(
                        text(
                            "INSERT INTO job_events (job_id, event_type, event_data) "
                            "VALUES (:job_id, :event_type, :event_data)"
                        ),
                        {
                            "job_id": self.job_id,
                            "event_type": event_type,
                            "event_data": event_json,
                        },
                    )

                conn.commit()
                logger.debug("Stored event %s for job %s", event_type, self.job_id)
        except Exception as e:
            logger.warning(
                "Failed to store event %s for job %s (error_type=%s error_%s)",
                event_type,
                self.job_id,
                e.__class__.__name__,
                log_content_metadata(e),
            )
            if self._fail_closed:
                raise

    def store_batch(self, events: list[dict]):
        """
        Store multiple events in a single transaction.

        Reduces DB round-trips by batching INSERTs into one COMMIT.
        For PostgreSQL, issues pg_notify for each event in the batch.
        """
        if not events:
            return

        import json

        from sqlalchemy import text

        rows = []
        for event in events:
            stored_event = self._prepare_event_for_storage(event)
            rows.append(
                {
                    "job_id": self.job_id,
                    "event_type": stored_event.get("type", "unknown"),
                    "event_data": json.dumps(stored_event),
                }
            )

        try:
            with self._sync_engine.connect() as conn:
                if self._is_postgres:
                    channel = f"job_events_{self.job_id.replace('-', '_')}"
                    for row in rows:
                        result = conn.execute(
                            text(
                                "INSERT INTO job_events (job_id, event_type, event_data) "
                                "VALUES (:job_id, :event_type, :event_data) RETURNING id"
                            ),
                            row,
                        )
                        event_id = result.scalar()
                        payload = json.dumps({"id": event_id, "type": row["event_type"]})
                        conn.execute(
                            text("SELECT pg_notify(:channel, :payload)"),
                            {"channel": channel, "payload": payload},
                        )
                else:
                    conn.execute(
                        text(
                            "INSERT INTO job_events (job_id, event_type, event_data) "
                            "VALUES (:job_id, :event_type, :event_data)"
                        ),
                        rows,
                    )

                conn.commit()
                logger.debug("Stored batch of %d events for job %s", len(events), self.job_id)
        except Exception as e:
            logger.warning(
                "Failed to store batch of %d events for job %s (error_type=%s error_%s)",
                len(events),
                self.job_id,
                e.__class__.__name__,
                log_content_metadata(e),
            )
            if self._fail_closed:
                raise

    @classmethod
    def _ensure_table_exists(cls, db_url: str):
        """Ensure job_events table exists (class method for use without instance)."""
        if db_url in cls._tables_initialized:
            return

        from sqlalchemy import Column
        from sqlalchemy import DateTime
        from sqlalchemy import Index
        from sqlalchemy import Integer
        from sqlalchemy import MetaData
        from sqlalchemy import String
        from sqlalchemy import Table
        from sqlalchemy import Text
        from sqlalchemy import inspect
        from sqlalchemy.sql import func

        engine = cls._get_or_create_sync_engine(db_url)
        inspector = inspect(engine)

        if not inspector.has_table("job_events"):
            metadata = MetaData()
            Table(
                "job_events",
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("job_id", String(64), nullable=False, index=True),
                Column("event_type", String(64), nullable=False),
                Column("event_data", Text, nullable=True),
                Column("created_at", DateTime, server_default=func.now()),
                Index("idx_job_events_job_id_id", "job_id", "id"),
            )
            metadata.create_all(engine)
            logger.info("Created job_events table in %s", db_url[:50])

        cls._tables_initialized.add(db_url)

    @classmethod
    def _prepare_event_rows_for_return(cls, job_id: str, rows: list[Any]) -> list[dict]:
        """Deserialize and decrypt event rows in sync code, suitable for thread offload."""

        import json

        events = []
        for row in rows:
            try:
                event = cls._prepare_event_for_return(job_id, json.loads(row[1]))
                event["_id"] = row[0]
                events.append(event)
            except json.JSONDecodeError:
                logger.warning("Malformed event data for job %s, event id %d", job_id, row[0])
            except ContentEncryptionError as e:
                logger.warning(
                    "Encrypted event data failed for job %s, event id %d: %s",
                    job_id,
                    row[0],
                    e.__class__.__name__,
                )
                raise
        return events

    @classmethod
    def get_events(cls, db_url: str, job_id: str, after_id: int = 0, limit: int = 100) -> list[dict]:
        """
        Retrieve events for SSE streaming (sync).

        Args:
            db_url: Database URL
            job_id: Job ID to fetch events for
            after_id: Only return events with id > after_id
            limit: Maximum events to return

        Returns:
            List of event dicts with '_id' field for cursor tracking
        """
        from sqlalchemy import text

        try:
            cls._ensure_table_exists(db_url)
            engine = cls._get_or_create_sync_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT id, event_data FROM job_events "
                        "WHERE job_id = :job_id AND id > :after_id "
                        "ORDER BY id LIMIT :limit"
                    ),
                    {"job_id": job_id, "after_id": after_id, "limit": limit},
                )
                return cls._prepare_event_rows_for_return(job_id, list(result))
        except ContentEncryptionError:
            raise
        except Exception as e:
            logger.warning("Failed to get events for job %s: %s", job_id, e)
            return []

    @classmethod
    async def get_events_async(cls, db_url: str, job_id: str, after_id: int = 0, limit: int = 100) -> list[dict]:
        """
        Async version of get_events for FastAPI SSE routes.

        Uses native async SQLAlchemy with psycopg for true async I/O.
        """
        from sqlalchemy import text

        try:
            await cls._ensure_table_async(db_url)
            engine = cls._get_or_create_async_engine(db_url)
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(
                        "SELECT id, event_data FROM job_events "
                        "WHERE job_id = :job_id AND id > :after_id "
                        "ORDER BY id LIMIT :limit"
                    ),
                    {"job_id": job_id, "after_id": after_id, "limit": limit},
                )
                return await asyncio.to_thread(cls._prepare_event_rows_for_return, job_id, list(result))
        except ContentEncryptionError:
            raise
        except Exception as e:
            logger.warning("Failed to get events async for job %s: %s", job_id, e)
            return await asyncio.get_running_loop().run_in_executor(
                None, cls.get_events, db_url, job_id, after_id, limit
            )

    @classmethod
    def get_event_by_id(cls, db_url: str, event_id: int) -> dict | None:
        """
        Retrieve a single event by its ID.

        Used by PostgreSQL pub-sub SSE generator to fetch event details
        after receiving a NOTIFY notification.

        Args:
            db_url: Database URL
            event_id: Event ID to fetch

        Returns:
            Event dict with '_id' field, or None if not found
        """
        import json

        from sqlalchemy import text

        try:
            cls._ensure_table_exists(db_url)
            engine = cls._get_or_create_sync_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT id, job_id, event_data FROM job_events WHERE id = :event_id"),
                    {"event_id": event_id},
                )
                row = result.fetchone()
                if row:
                    try:
                        event = cls._prepare_event_for_return(row[1], json.loads(row[2]))
                        event["_id"] = row[0]
                        return event
                    except json.JSONDecodeError:
                        logger.warning("Malformed event data for event id %d", event_id)
                    except ContentEncryptionError as e:
                        logger.warning(
                            "Encrypted event data failed for job %s, event id %d: %s",
                            row[1],
                            event_id,
                            e.__class__.__name__,
                        )
                        raise
                return None
        except ContentEncryptionError:
            raise
        except Exception as e:
            logger.warning("Failed to get event %d: %s", event_id, e)
            return None

    @classmethod
    async def get_event_by_id_async(cls, db_url: str, event_id: int) -> dict | None:
        """
        Async version of get_event_by_id for PostgreSQL pub-sub SSE generator.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.get_event_by_id, db_url, event_id)

    @classmethod
    def is_postgres(cls, db_url: str) -> bool:
        """Check if the database URL is for PostgreSQL."""
        return db_url.startswith("postgresql")

    # -------------------------------------------------------------------------
    # Event Cleanup Methods
    #
    # Two cleanup strategies are available:
    #
    # 1. cleanup_job_events(job_id) - Delete events for a specific job.
    #    Use when a job is explicitly deleted or for targeted cleanup.
    #
    # 2. cleanup_old_events(retention_seconds) - Delete all events older than
    #    the retention period. Used by the background cleanup task in plugin.py
    #    to prevent unbounded table growth. Runs periodically based on
    #    expiry_seconds from config.
    #
    # NAT's JobStore handles job metadata expiry separately. These methods
    # handle our custom job_events table which NAT doesn't know about.
    # -------------------------------------------------------------------------

    @classmethod
    def cleanup_job_events(cls, db_url: str, job_id: str) -> int:
        """
        Delete all events for a specific job.

        Use for targeted cleanup when a job is deleted or no longer needed.
        For automatic time-based cleanup, see cleanup_old_events().

        Args:
            db_url: Database connection URL
            job_id: Job ID to delete events for

        Returns:
            Number of deleted events
        """
        from sqlalchemy import text

        try:
            engine = cls._get_or_create_sync_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("DELETE FROM job_events WHERE job_id = :job_id"),
                    {"job_id": job_id},
                )
                conn.commit()
                return result.rowcount
        except Exception as e:
            logger.warning("Failed to cleanup events for job %s: %s", job_id, e)
            return 0

    @classmethod
    def cleanup_old_events(cls, db_url: str, retention_seconds: int = 86400) -> int:
        """
        Delete all events older than retention period (global cleanup).

        Called periodically by the background task in plugin.py to prevent
        unbounded table growth. Uses expiry_seconds from front_end config.

        Args:
            db_url: Database connection URL
            retention_seconds: Delete events older than this (default: 86400 = 24h)

        Returns:
            Number of deleted events
        """
        from sqlalchemy import text

        try:
            engine = cls._get_or_create_sync_engine(db_url)
            is_postgres = db_url.startswith("postgresql")

            with engine.connect() as conn:
                if is_postgres:
                    result = conn.execute(
                        text("DELETE FROM job_events WHERE created_at < NOW() - :seconds * INTERVAL '1 second'"),
                        {"seconds": retention_seconds},
                    )
                else:
                    result = conn.execute(
                        text("DELETE FROM job_events WHERE created_at < datetime('now', :interval)"),
                        {"interval": f"-{retention_seconds} seconds"},
                    )
                conn.commit()
                deleted = result.rowcount
                if deleted > 0:
                    logger.info("Cleaned up %d events older than %d seconds", deleted, retention_seconds)
                return deleted
        except Exception as e:
            logger.warning("Failed to cleanup old events: %s", e)
            return 0

    @classmethod
    async def cleanup_old_events_async(cls, db_url: str, retention_seconds: int = 86400) -> int:
        """Async wrapper for cleanup_old_events(), used by background task."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.cleanup_old_events, db_url, retention_seconds)


class BatchingEventStore:
    """
    Wraps EventStore with time-based and size-based batching.

    Buffers events and flushes them in a single transaction when either:
    - The buffer reaches MAX_BATCH_SIZE events, or
    - FLUSH_INTERVAL_MS milliseconds have passed since the first buffered event.

    This reduces DB round-trips from N to N/MAX_BATCH_SIZE while keeping
    maximum event delivery latency under FLUSH_INTERVAL_MS.

    The wrapper exposes the same store() interface as EventStore, so it
    can be used as a drop-in replacement in AgentEventCallback.
    """

    FLUSH_INTERVAL_MS = 200
    MAX_BATCH_SIZE = 10

    def __init__(self, event_store: EventStore):
        self._store = event_store
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._flush_error: Exception | None = None

    @property
    def job_id(self) -> str | None:
        """Expose job_id from the underlying EventStore."""
        return self._store.job_id

    def store(self, event: dict):
        """Buffer an event; flush when batch is full or timer fires."""
        with self._lock:
            self._raise_flush_error_locked()
            self._buffer.append(event)
            if len(self._buffer) >= self.MAX_BATCH_SIZE:
                self._flush_locked()
            elif self._timer is None:
                self._timer = threading.Timer(self.FLUSH_INTERVAL_MS / 1000, self._flush_from_timer)
                self._timer.daemon = True
                self._timer.start()

    def _raise_flush_error_locked(self):
        """Raise a previously captured background flush failure."""
        if self._flush_error is not None:
            raise self._flush_error

    def _flush_locked(self):
        """Flush while already holding the lock."""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self._raise_flush_error_locked()
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        try:
            self._store.store_batch(batch)
        except Exception as exc:
            if self._flush_error is None:
                self._flush_error = exc
            raise

    def _flush_from_timer(self):
        """Capture timer-thread failures for the owning job to observe."""
        with self._lock:
            try:
                self._flush_locked()
            except Exception as exc:
                logger.warning(
                    "Background event flush failed for job %s exception=%s",
                    self.job_id,
                    exc.__class__.__name__,
                )

    def flush(self):
        """Force flush all buffered events. Call before job completion."""
        with self._lock:
            self._flush_locked()
