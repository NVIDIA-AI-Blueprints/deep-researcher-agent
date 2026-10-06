# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
import sys

import pytest

from deep_researcher_api.mcp_auth.sqlite_object_store import DeepResearcherSqliteObjectStore
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.models import ObjectStoreItem


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "tokens.db")


def _item(data: bytes = b'{"tok": 1}', **kw) -> ObjectStoreItem:
    return ObjectStoreItem(data=data, content_type=kw.get("content_type"), metadata=kw.get("metadata"))


async def test_cross_connection_round_trip(db_path):
    """A second connection (proxy for the worker process) reads what the first wrote.

    This is the property Redis exists for: the API process writes the token and a
    *separate* Dask worker process reads it at job time. Two independent
    connections to the same file stand in for those two processes.
    """
    writer = DeepResearcherSqliteObjectStore(db_path, bucket_name="mcp-tokens")
    await writer.put_object("alice", _item(b"secret", metadata={"u": "alice"}))

    reader = DeepResearcherSqliteObjectStore(db_path, bucket_name="mcp-tokens")
    got = await reader.get_object("alice")
    assert got.data == b"secret"
    assert got.metadata == {"u": "alice"}

    await writer.aclose()
    await reader.aclose()


async def test_put_rejects_existing_key(db_path):
    store = DeepResearcherSqliteObjectStore(db_path)
    await store.put_object("k", _item())
    with pytest.raises(KeyAlreadyExistsError):
        await store.put_object("k", _item())
    await store.aclose()


async def test_upsert_overwrites(db_path):
    store = DeepResearcherSqliteObjectStore(db_path)
    await store.put_object("k", _item(b"v1"))
    await store.upsert_object("k", _item(b"v2"))
    assert (await store.get_object("k")).data == b"v2"
    await store.aclose()


async def test_get_and_delete_missing_raise(db_path):
    store = DeepResearcherSqliteObjectStore(db_path)
    with pytest.raises(NoSuchKeyError):
        await store.get_object("nope")
    with pytest.raises(NoSuchKeyError):
        await store.delete_object("nope")
    await store.aclose()


async def test_delete_removes(db_path):
    store = DeepResearcherSqliteObjectStore(db_path)
    await store.put_object("k", _item())
    await store.delete_object("k")
    with pytest.raises(NoSuchKeyError):
        await store.get_object("k")
    await store.aclose()


async def test_expired_object_is_absent(db_path):
    # Negative TTL => already expired on write; reads must treat it as missing.
    store = DeepResearcherSqliteObjectStore(db_path, ttl=-1)
    await store.upsert_object("k", _item())
    with pytest.raises(NoSuchKeyError):
        await store.get_object("k")
    # And an expired key does not block a fresh put.
    fresh = DeepResearcherSqliteObjectStore(db_path)
    await fresh.put_object("k", _item(b"new"))
    assert (await fresh.get_object("k")).data == b"new"
    await store.aclose()
    await fresh.aclose()


async def test_bucket_prefix_isolates_keys(db_path):
    a = DeepResearcherSqliteObjectStore(db_path, bucket_name="a")
    b = DeepResearcherSqliteObjectStore(db_path, bucket_name="b")
    await a.put_object("same", _item(b"in-a"))
    await b.put_object("same", _item(b"in-b"))
    assert (await a.get_object("same")).data == b"in-a"
    assert (await b.get_object("same")).data == b"in-b"
    await a.aclose()
    await b.aclose()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes not meaningful on Windows")
async def test_db_file_is_owner_only(db_path):
    """The token DB (and its WAL/SHM sidecars) must not be group/world-readable.

    The stored objects contain plaintext access/refresh tokens, so a 0644 file
    would let other local users read the credential store.
    """
    store = DeepResearcherSqliteObjectStore(db_path, bucket_name="mcp-tokens")
    # Force the connection (and thus file creation + WAL sidecars) to happen.
    await store.put_object("alice", _item(b"secret"))

    for path in (db_path, db_path + "-wal", db_path + "-shm"):
        if os.path.exists(path):
            mode = stat.S_IMODE(os.stat(path).st_mode)
            assert mode & 0o077 == 0, f"{path} is accessible to group/other: {oct(mode)}"

    await store.aclose()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes not meaningful on Windows")
async def test_existing_loose_db_is_tightened(db_path):
    """An already-present 0644 DB file is chmod'd to 0600 on next open."""
    # Simulate a pre-existing world-readable file left by an older build.
    with open(db_path, "wb"):
        pass
    os.chmod(db_path, 0o644)

    store = DeepResearcherSqliteObjectStore(db_path)
    await store.put_object("k", _item())
    assert stat.S_IMODE(os.stat(db_path).st_mode) & 0o077 == 0
    await store.aclose()


async def test_registered_with_nat():
    """The store is discoverable by NAT under _type: deep_researcher_sqlite."""
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    discover_and_register_plugins(PluginTypes.ALL)
    from nat.cli.type_registry import GlobalTypeRegistry

    names = [getattr(i, "local_name", "") for i in GlobalTypeRegistry.get().get_registered_object_stores()]
    assert "deep_researcher_sqlite" in names
