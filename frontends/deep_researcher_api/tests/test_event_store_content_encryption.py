# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import pytest
from sqlalchemy import text

from deep_researcher_api.jobs import crypto
from deep_researcher_api.jobs.event_store import BatchingEventStore
from deep_researcher_api.jobs.event_store import EventStore


def _static_key() -> str:
    return base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")


def _other_static_key() -> str:
    return base64.urlsafe_b64encode(bytes(reversed(range(32)))).decode("ascii")


@pytest.fixture(autouse=True)
def clean_encryption_env(monkeypatch):
    for name in (
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS",
        "DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS",
        "VAULT_ADDR",
        "VAULT_NAMESPACE",
        "VAULT_TRANSIT_MOUNT",
        "VAULT_ROLE_ID",
        "VAULT_SECRET_ID",
        "DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY",
        "VAULT_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    EventStore._tables_initialized.clear()
    crypto.reset_content_encryption_manager_for_tests()
    yield
    EventStore._tables_initialized.clear()
    crypto.reset_content_encryption_manager_for_tests()


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path / 'events.db'}"


def _enable_static_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION", "key")
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _static_key())
    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID", "test-key")
    crypto.reset_content_encryption_manager_for_tests()


def _raw_event_data(db_url: str) -> dict:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT event_data FROM job_events ORDER BY id DESC LIMIT 1")).fetchone()
    assert row is not None
    return json.loads(row[0])


def _latest_event_id(db_url: str) -> int:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id FROM job_events ORDER BY id DESC LIMIT 1")).fetchone()
    assert row is not None
    return row[0]


def test_output_artifact_content_is_encrypted_at_rest_and_decrypted_on_read(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)

    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
                "output_category": "final_report",
            },
        }
    )

    raw_event = _raw_event_data(db_url)
    raw_content = raw_event["data"]["content"]
    assert raw_event["data"]["type"] == "output"
    assert raw_event["data"]["output_category"] == "final_report"
    assert raw_content[crypto.ENCRYPTED_FIELD_MARKER] is True
    assert raw_content[crypto.ENCRYPTED_FIELD_VALUE].startswith(crypto.ENVELOPE_PREFIX)
    assert "secret report" not in json.dumps(raw_event)

    events = EventStore.get_events(db_url, "job-1")
    assert events[0]["data"]["content"] == "secret report"


def test_file_artifact_content_is_encrypted_in_batch(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)

    store.store_batch(
        [
            {
                "type": "artifact.update",
                "data": {
                    "type": "file",
                    "content": "secret file content",
                    "file_path": "/shared/output.md",
                },
            }
        ]
    )

    raw_event = _raw_event_data(db_url)
    assert raw_event["data"]["content"][crypto.ENCRYPTED_FIELD_VALUE].startswith(crypto.ENVELOPE_PREFIX)
    assert "secret file content" not in json.dumps(raw_event)

    events = EventStore.get_events(db_url, "job-1")
    assert events[0]["data"]["content"] == "secret file content"


@pytest.mark.parametrize("use_batch", [False, True])
def test_encrypted_event_persistence_failure_is_raised(monkeypatch, db_url, use_batch):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)

    def fail_connect():
        raise RuntimeError("transient database failure")

    monkeypatch.setattr(store._sync_engine, "connect", fail_connect)
    event = {"type": "test.event", "data": {"value": "test"}}

    with pytest.raises(RuntimeError, match="transient database failure"):
        if use_batch:
            store.store_batch([event])
        else:
            store.store(event)


@pytest.mark.parametrize("use_batch", [False, True])
def test_off_mode_event_persistence_failure_remains_best_effort(monkeypatch, db_url, use_batch):
    store = EventStore(db_url, "job-1", content_cipher=crypto.create_job_content_cipher("job-1"))

    def fail_connect():
        raise RuntimeError("transient database failure")

    monkeypatch.setattr(store._sync_engine, "connect", fail_connect)
    event = {"type": "test.event", "data": {"value": "test"}}

    if use_batch:
        store.store_batch([event])
    else:
        store.store(event)


def test_background_encrypted_batch_failure_is_raised_by_foreground_flush(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    raw_store = EventStore(db_url, "job-1", content_cipher=cipher)
    flush_attempted = threading.Event()
    attempts = 0
    original_connect = raw_store._sync_engine.connect

    def fail_once_connect():
        nonlocal attempts
        attempts += 1
        flush_attempted.set()
        if attempts == 1:
            raise RuntimeError("transient database failure")
        return original_connect()

    monkeypatch.setattr(raw_store._sync_engine, "connect", fail_once_connect)
    store = BatchingEventStore(raw_store)
    store.FLUSH_INTERVAL_MS = 1

    store.store(
        {
            "type": "artifact.update",
            "data": {"type": "output", "content": "secret report"},
        }
    )

    assert flush_attempted.wait(timeout=1)
    with pytest.raises(RuntimeError, match="transient database failure"):
        store.flush()
    with pytest.raises(RuntimeError, match="transient database failure"):
        store.store({"type": "test.event", "data": {"value": "later"}})
    assert attempts == 1


def test_non_sensitive_artifact_content_remains_plaintext(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)

    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "citation_source",
                "content": "https://example.com/source",
                "url": "https://example.com/source",
            },
        }
    )

    raw_event = _raw_event_data(db_url)
    assert raw_event["data"]["content"] == "https://example.com/source"


def test_plaintext_historical_event_rows_still_read_in_encrypted_mode(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    store = EventStore(db_url, "job-1")

    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "historical plaintext",
            },
        }
    )

    events = EventStore.get_events(db_url, "job-1")
    assert events[0]["data"]["content"] == "historical plaintext"


@pytest.mark.asyncio
async def test_async_event_reads_decrypt_content(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "async secret report",
            },
        }
    )

    events = await EventStore.get_events_async(db_url, "job-1")

    assert events[0]["data"]["content"] == "async secret report"


@pytest.mark.asyncio
async def test_async_event_read_decrypts_off_event_loop(monkeypatch, db_url):
    loop_thread = threading.get_ident()
    decrypt_thread = None
    store = EventStore(db_url, "job-1")
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": {
                    crypto.ENCRYPTED_FIELD_MARKER: True,
                    crypto.ENCRYPTED_FIELD_VALUE: f"{crypto.ENVELOPE_PREFIX}fake",
                },
            },
        }
    )

    def fake_decrypt_event_field(_job_id, _field_path, _stored_value):
        nonlocal decrypt_thread
        decrypt_thread = threading.get_ident()
        return "decrypted report"

    monkeypatch.setattr(crypto, "decrypt_event_field", fake_decrypt_event_field)

    events = await EventStore.get_events_async(db_url, "job-1")

    assert events[0]["data"]["content"] == "decrypted report"
    assert decrypt_thread is not None
    assert decrypt_thread != loop_thread


@pytest.mark.asyncio
async def test_async_event_read_slow_decrypt_does_not_block_event_loop(monkeypatch, db_url):
    store = EventStore(db_url, "job-1")
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": {
                    crypto.ENCRYPTED_FIELD_MARKER: True,
                    crypto.ENCRYPTED_FIELD_VALUE: f"{crypto.ENVELOPE_PREFIX}fake",
                },
            },
        }
    )

    def slow_decrypt_event_field(_job_id, _field_path, _stored_value):
        time.sleep(0.1)
        return "decrypted report"

    monkeypatch.setattr(crypto, "decrypt_event_field", slow_decrypt_event_field)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while ticks < 3:
            await asyncio.sleep(0.01)
            ticks += 1

    read_task = asyncio.create_task(EventStore.get_events_async(db_url, "job-1"))
    ticker_task = asyncio.create_task(ticker())

    events = await read_task
    await ticker_task

    assert events[0]["data"]["content"] == "decrypted report"
    assert ticks >= 3


def test_concurrent_event_reads_decrypt_consistently(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)
    for index in range(10):
        store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "output",
                    "content": f"secret report {index}",
                },
            }
        )
    event_id = _latest_event_id(db_url)

    def read_events() -> tuple[list[str], str]:
        events = EventStore.get_events(db_url, "job-1", 0, 100)
        event = EventStore.get_event_by_id(db_url, event_id)
        assert event is not None
        return [item["data"]["content"] for item in events], event["data"]["content"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_events) for _ in range(32)]
        for future in as_completed(futures):
            contents, latest_content = future.result()
            assert contents == [f"secret report {index}" for index in range(10)]
            assert latest_content == "secret report 9"


def test_encrypted_event_wrong_key_raises_instead_of_returning_empty(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )

    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _other_static_key())
    crypto.reset_content_encryption_manager_for_tests()

    with pytest.raises(crypto.ContentEncryptionInvalidData):
        EventStore.get_events(db_url, "job-1")


def test_encrypted_event_by_id_wrong_key_raises_instead_of_returning_none(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )
    event_id = _latest_event_id(db_url)

    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _other_static_key())
    crypto.reset_content_encryption_manager_for_tests()

    with pytest.raises(crypto.ContentEncryptionInvalidData):
        EventStore.get_event_by_id(db_url, event_id)


@pytest.mark.asyncio
async def test_async_encrypted_event_wrong_key_raises_instead_of_fallback(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )

    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _other_static_key())
    crypto.reset_content_encryption_manager_for_tests()

    with pytest.raises(crypto.ContentEncryptionInvalidData):
        await EventStore.get_events_async(db_url, "job-1")


@pytest.mark.asyncio
async def test_async_encrypted_event_by_id_wrong_key_raises_instead_of_returning_none(monkeypatch, db_url):
    _enable_static_key(monkeypatch)
    cipher = crypto.create_job_content_cipher("job-1")
    store = EventStore(db_url, "job-1", content_cipher=cipher)
    store.store(
        {
            "type": "artifact.update",
            "data": {
                "type": "output",
                "content": "secret report",
            },
        }
    )
    event_id = _latest_event_id(db_url)

    monkeypatch.setenv("DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY", _other_static_key())
    crypto.reset_content_encryption_manager_for_tests()

    with pytest.raises(crypto.ContentEncryptionInvalidData):
        await EventStore.get_event_by_id_async(db_url, event_id)
