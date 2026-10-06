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

"""A SQLite-backed NAT ``ObjectStore`` for serviceless, cross-process storage.

NAT ships only ``in_memory`` (process-local) in core, plus ``redis``/``s3``/
``mysql`` in separate packages — all of which are network services. The per-user
MCP token store, however, is written by the API process (OAuth ``/connect`` and
``/callback``) and read by a *separate* Dask worker process at job time, so an
in-memory store can't bridge that gap and a network service can't run "with no
deployment".

A file on local disk is the one option that is both serviceless **and** visible
across processes: both processes open the same SQLite file (WAL mode), exactly
how this project's job/checkpoint/summary stores already span the same two
processes locally. The same class points at a path locally and is swapped for
``redis`` in deployment via the config's env-selectable ``_type``.

This implements NAT's ``ObjectStore`` interface only; the token serialization,
refresh, and expiry logic stay in NAT's ``ObjectStoreTokenStorage`` /
``mcp_oauth2`` on top — nothing here is token-specific.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import aiosqlite
from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    key         TEXT PRIMARY KEY,
    data        BLOB NOT NULL,
    content_type TEXT,
    metadata    TEXT,
    expires_at  REAL
)
"""


def _restrict_file(path: str) -> None:
    """chmod ``path`` to owner-only (0600) if it exists; best-effort."""
    try:
        if os.path.exists(path):
            os.chmod(path, 0o600)
    except OSError as exc:  # e.g. unsupported on the platform — don't crash startup
        logger.warning("Could not restrict permissions on %s: %s", path, exc)


def _ensure_private_file(path: str) -> None:
    """Ensure the token DB file exists with owner-only (0600) permissions.

    Creating it ourselves with ``O_CREAT`` + ``0o600`` closes the window where
    sqlite would otherwise create it at the umask default (0644), and we chmod
    unconditionally so an existing 0644 file is tightened on next open.
    """
    if not os.path.exists(path):
        try:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        except OSError as exc:
            logger.warning("Could not pre-create token DB %s with private perms: %s", path, exc)
            return
    _restrict_file(path)


class DeepResearcherSqliteObjectStoreConfig(ObjectStoreBaseConfig, name="deep_researcher_sqlite"):
    """SQLite-file object store — serviceless and shared across processes.

    Suited to local/single-host runs where standing up Redis/S3/MySQL is
    undesirable. For multi-replica deployments use a networked object store
    (e.g. ``redis``) instead, since a SQLite file is only shared by processes
    that can see the same path.
    """

    db_path: str = Field(
        default="./mcp_tokens.db",
        description="Path to the SQLite database file (created on first use).",
    )
    bucket_name: str | None = Field(
        default=None,
        description="Optional key prefix so multiple logical buckets can share one file.",
    )
    ttl: int | None = Field(
        default=None,
        description="TTL in seconds for stored objects (None = no expiration).",
    )


class DeepResearcherSqliteObjectStore(ObjectStore):
    """ObjectStore backed by a single SQLite file.

    Cross-process visibility relies on WAL mode + a busy timeout: each process
    opens its own connection to the same file, and committed writes from one
    process are readable by the others.
    """

    def __init__(self, db_path: str, bucket_name: str | None = None, ttl: int | None = None) -> None:
        self._db_path = db_path
        self._prefix = f"{bucket_name}/" if bucket_name else ""
        self._ttl = ttl
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # ── connection / helpers ──
    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            # The serialized objects hold plaintext access/refresh tokens, so the
            # file must never be group/world-readable. Create it 0600 BEFORE sqlite
            # opens it (default umask 022 would otherwise yield 0644), and tighten
            # an existing file too.
            _ensure_private_file(self._db_path)
            db = await aiosqlite.connect(self._db_path)
            # WAL allows concurrent readers across processes alongside one writer;
            # busy_timeout waits out a peer's write lock instead of failing fast.
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute(_SCHEMA)
            await db.commit()
            # WAL/SHM sidecars are created by sqlite on first write and also carry
            # token bytes; restrict them once they exist.
            for suffix in ("-wal", "-shm"):
                _restrict_file(self._db_path + suffix)
            self._db = db
        return self._db

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _expiry(self) -> float | None:
        return time.time() + self._ttl if self._ttl else None

    @staticmethod
    def _expired(expires_at: float | None) -> bool:
        return expires_at is not None and expires_at <= time.time()

    def _row_to_item(self, row: tuple) -> ObjectStoreItem:
        data, content_type, metadata = row[0], row[1], row[2]
        return ObjectStoreItem(
            data=data,
            content_type=content_type,
            metadata=json.loads(metadata) if metadata else None,
        )

    # ── ObjectStore interface ──
    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        k = self._k(key)
        async with self._lock:
            db = await self._conn()
            # An expired row is logically absent, so let a fresh put replace it.
            async with db.execute("SELECT expires_at FROM objects WHERE key = ?", (k,)) as cur:
                existing = await cur.fetchone()
            if existing is not None and not self._expired(existing[0]):
                raise KeyAlreadyExistsError(key)
            await db.execute(
                "INSERT OR REPLACE INTO objects (key, data, content_type, metadata, expires_at) VALUES (?, ?, ?, ?, ?)",
                (k, item.data, item.content_type, json.dumps(item.metadata) if item.metadata else None, self._expiry()),
            )
            await db.commit()

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        k = self._k(key)
        async with self._lock:
            db = await self._conn()
            await db.execute(
                "INSERT OR REPLACE INTO objects (key, data, content_type, metadata, expires_at) VALUES (?, ?, ?, ?, ?)",
                (k, item.data, item.content_type, json.dumps(item.metadata) if item.metadata else None, self._expiry()),
            )
            await db.commit()

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        k = self._k(key)
        async with self._lock:
            db = await self._conn()
            async with db.execute(
                "SELECT data, content_type, metadata, expires_at FROM objects WHERE key = ?", (k,)
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise NoSuchKeyError(key)
            if self._expired(row[3]):
                await db.execute("DELETE FROM objects WHERE key = ?", (k,))
                await db.commit()
                raise NoSuchKeyError(key)
            return self._row_to_item(row)

    @override
    async def delete_object(self, key: str) -> None:
        k = self._k(key)
        async with self._lock:
            db = await self._conn()
            cur = await db.execute("DELETE FROM objects WHERE key = ?", (k,))
            await db.commit()
            if cur.rowcount == 0:
                raise NoSuchKeyError(key)

    async def aclose(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


@register_object_store(config_type=DeepResearcherSqliteObjectStoreConfig)
async def deep_researcher_sqlite_object_store(config: DeepResearcherSqliteObjectStoreConfig, builder: Builder):
    store = DeepResearcherSqliteObjectStore(db_path=config.db_path, bucket_name=config.bucket_name, ttl=config.ttl)
    # Log the ABSOLUTE path: the API and worker must resolve db_path to the same
    # file. With a relative path that only holds if they share a working dir — a
    # cwd divergence otherwise creates two files and silently loses tokens. The
    # absolute path here makes such a mismatch visible in each process's logs.
    logger.info("SQLite object store initialized at %s", os.path.abspath(config.db_path))
    try:
        yield store
    finally:
        await store.aclose()
