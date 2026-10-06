# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock

import httpx
import knowledge_layer.nemo_retriever._transport as transport_module
import knowledge_layer.nemo_retriever.adapter as adapter_module
import pytest
from knowledge_layer.nemo_retriever._transport import NemoRetrieverCompatibilityError
from knowledge_layer.nemo_retriever._transport import NemoRetrieverError
from knowledge_layer.nemo_retriever._transport import NemoRetrieverHTTPError
from knowledge_layer.nemo_retriever._transport import NemoRetrieverTransportError
from knowledge_layer.nemo_retriever._transport import _NRLTransport
from knowledge_layer.nemo_retriever.adapter import NemoRetrieverIngestor
from knowledge_layer.nemo_retriever.adapter import NemoRetrieverRetriever
from knowledge_layer.register import KnowledgeRetrievalConfig
from knowledge_layer.register import _format_results
from knowledge_layer.register import _setup_backend
from pydantic import SecretStr
from pydantic import ValidationError

from deep_researcher_agent.knowledge import BaseIngestor
from deep_researcher_agent.knowledge import BaseRetriever
from deep_researcher_agent.knowledge import Chunk
from deep_researcher_agent.knowledge import ContentType
from deep_researcher_agent.knowledge import JobState
from deep_researcher_agent.knowledge.base import IngestionBatchTooLargeError
from deep_researcher_agent.knowledge.base import IngestionCapacityError
from deep_researcher_agent.knowledge.factory import is_ingestor_registered
from deep_researcher_agent.knowledge.factory import is_retriever_registered
from deep_researcher_agent.knowledge.schema import FileStatus

NOW = "2026-07-17T12:00:00+00:00"


def _response(request: httpx.Request, status: int, payload: Any = None, **kwargs: Any) -> httpx.Response:
    if payload is None:
        return httpx.Response(status, request=request, **kwargs)
    return httpx.Response(status, request=request, json=payload, **kwargs)


class FakeNRL:
    def __init__(self, *, paginate: bool = False):
        self.paginate = paginate
        self.requests: list[httpx.Request] = []
        self.create_job_bodies: list[dict[str, Any]] = []
        self.query_hits: list[dict[str, Any]] = []
        self.manifest: list[dict[str, str]] = []
        self.documents: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.collections = [
            {
                "name": "first",
                "scope": "workspace-123",
                "status": "active",
                "description": "one",
                "metadata": {"owner": "deep-researcher", "table_name": "physical-secret"},
                "created_at": NOW,
                "updated_at": NOW,
                "expires_at": None,
            },
            {
                "name": "second",
                "scope": "workspace-123",
                "status": "active",
                "description": "two",
                "metadata": {},
                "created_at": NOW,
                "updated_at": NOW,
                "expires_at": None,
            },
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
        path = request.url.path
        method = request.method

        if path == "/v1/health":
            return _response(request, 200, {"status": "ok", "mode": "standalone"})
        if path == "/v1/query":
            return _response(request, 200, {"results": [{"hits": self.query_hits}]})
        if path == "/v1/collections" and method == "POST":
            body = json.loads(request.content)
            return _response(
                request,
                201,
                {
                    "name": body["name"],
                    "scope": "workspace-123",
                    "status": "active",
                    "description": body.get("description"),
                    "metadata": body.get("metadata", {}),
                    "created_at": NOW,
                    "updated_at": NOW,
                    "expires_at": body.get("expires_at"),
                },
            )
        if path == "/v1/collections" and method == "GET":
            token = request.url.params.get("continuation_token")
            if self.paginate and token is None:
                return _response(request, 200, {"items": self.collections[:1], "next_token": "page-2"})
            return _response(request, 200, {"items": self.collections[1:] if self.paginate else self.collections})
        if path == "/v1/collections/test" and method == "GET":
            return _response(request, 200, {**self.collections[0], "name": "test"})
        if path == "/v1/collections/test" and method == "DELETE":
            return _response(
                request,
                200,
                {
                    "name": "test",
                    "scope": "workspace-123",
                    "existed": True,
                    "deleted": True,
                    "status": "deleted",
                    "cleanup_pending": False,
                },
            )
        if path == "/v1/ingest/job" and method == "POST":
            body = json.loads(request.content)
            with self._lock:
                self.create_job_bodies.append(body)
                self.manifest = body["document_manifest"]
                status_code = 201 if len(self.create_job_bodies) == 1 else 200
            return _response(
                request,
                status_code,
                {
                    "job_id": "job-1",
                    "expected_documents": body["expected_documents"],
                    "status": "pending",
                    "created_at": NOW,
                    "collection_name": body["collection_name"],
                    "operation": "append",
                },
            )
        if path == "/v1/ingest/job/job-1/document" and method == "POST":
            body = request.content.decode(errors="replace")
            with self._lock:
                entry = next(item for item in self.manifest if item["manifest_entry_id"] in body)
                position = self.manifest.index(entry)
                stable_id = f"stable-doc-{position}"
                accepted = {
                    "document_id": stable_id,
                    "attempt_id": f"attempt-{position}",
                    "job_id": "job-1",
                    "content_sha256": entry["content_sha256"],
                    "status": "pending",
                    "created_at": NOW,
                }
                self.documents[stable_id] = {
                    "document_id": stable_id,
                    "attempt_id": f"attempt-{position}",
                    "job_id": "job-1",
                    "status": "completed",
                    "submitted_at": NOW,
                    "started_at": NOW,
                    "completed_at": NOW,
                    "filename": entry["filename"],
                    "result_rows": position + 2,
                    "error": None,
                    "collection_name": "test",
                    "content_sha256": entry["content_sha256"],
                }
            return _response(request, 202, accepted)
        if path == "/v1/ingest/job/job-1" and method == "GET":
            return _response(
                request,
                200,
                {
                    "job_id": "job-1",
                    "expected_documents": len(self.manifest),
                    "status": "completed",
                    "created_at": NOW,
                    "started_at": NOW,
                    "finalized_at": NOW,
                    "counts": {"completed": len(self.documents)},
                    "document_ids": [item["attempt_id"] for item in self.documents.values()],
                    "collection_name": "test",
                    "operation": "append",
                },
            )
        if path == "/v1/ingest/job/job-1/documents" and method == "GET":
            items = list(self.documents.values())
            return _response(
                request,
                200,
                {
                    "job_id": "job-1",
                    "total": len(items),
                    "total_filtered": len(items),
                    "offset": int(request.url.params.get("offset", 0)),
                    "limit": int(request.url.params.get("limit", 1000)),
                    "items": items,
                },
            )
        if path == "/v1/collections/test/documents" and method == "GET":
            items = [
                {
                    "document_id": item["document_id"],
                    "collection_name": "test",
                    "scope": "workspace-123",
                    "filename": item["filename"],
                    "content_sha256": item["content_sha256"],
                    "document_version": "v1",
                    "status": item["status"],
                    "chunk_count": item["result_rows"],
                    "job_id": "job-1",
                    "created_at": NOW,
                    "updated_at": NOW,
                    "error": item["error"],
                }
                for item in self.documents.values()
            ]
            token = request.url.params.get("continuation_token")
            if self.paginate and token is None and len(items) > 1:
                return _response(request, 200, {"items": items[:1], "next_token": "docs-2"})
            return _response(request, 200, {"items": items[1:] if self.paginate and token else items})
        if path.startswith("/v1/collections/test/documents/"):
            document_id = path.rsplit("/", 1)[-1]
            if method == "DELETE":
                existed = document_id in self.documents
                self.documents.pop(document_id, None)
                return _response(
                    request,
                    200,
                    {
                        "document_id": document_id,
                        "collection_name": "test",
                        "scope": "workspace-123",
                        "existed": existed,
                        "deleted": existed,
                        "status": "deleted",
                        "cleanup_pending": False,
                    },
                )
            item = self.documents.get(document_id)
            if item is None:
                return _response(request, 404, {"detail": "not found"})
            return _response(
                request,
                200,
                {
                    "document_id": document_id,
                    "collection_name": "test",
                    "scope": "workspace-123",
                    "filename": item["filename"],
                    "content_sha256": item["content_sha256"],
                    "document_version": "v1",
                    "status": item["status"],
                    "chunk_count": item["result_rows"],
                    "job_id": "job-1",
                    "created_at": NOW,
                    "updated_at": NOW,
                    "error": item["error"],
                },
            )
        return _response(request, 404, {"detail": f"unhandled {method} {path}"})


def _adapter_config(handler: Any, *, token: str = "super-secret", retries: int = 0) -> dict[str, Any]:
    mock_transport = httpx.MockTransport(handler)
    transport = _NRLTransport(
        base_url="https://nrl.example.test",
        scope="workspace-123",
        api_token=token,
        connect_timeout_s=1,
        request_timeout_s=5,
        max_retries=retries,
        verify_ssl=True,
        ca_bundle=None,
        client=httpx.Client(transport=mock_transport),
    )
    return {
        "base_url": "https://nrl.example.test",
        "scope": "workspace-123",
        "api_token": SecretStr(token),
        "max_retries": retries,
        "max_concurrency": 2,
        "max_queued_uploads": 128,
        "_transport": transport,
        "warm_start": False,
    }


def _wait_for_uploads(ingestor: NemoRetrieverIngestor, job_id: str) -> None:
    with ingestor._tracking_lock:
        batch = ingestor._upload_batches[job_id]
    assert batch.done.wait(timeout=2), "background NRL uploads did not finish"


def test_backend_registration_config_validation_and_secret_redaction():
    assert is_retriever_registered("nemo_retriever")
    assert is_ingestor_registered("nemo_retriever")
    assert issubclass(NemoRetrieverRetriever, BaseRetriever)
    assert issubclass(NemoRetrieverIngestor, BaseIngestor)

    with pytest.raises(ValueError, match="explicit nrl_scope"):
        KnowledgeRetrievalConfig(backend="nemo_retriever", backend_config={"scope": ""})
    with pytest.raises(ValueError, match="Unsupported nemo_retriever backend_config option.*typo"):
        KnowledgeRetrievalConfig(backend="nemo_retriever", backend_config={"scope": "scope", "typo": True})

    secret = "must-not-appear"  # pragma: allowlist secret
    config = KnowledgeRetrievalConfig(
        backend="nemo_retriever",
        backend_config={"scope": "workspace-123", "api_token": secret},
    )
    assert secret not in repr(config)
    assert isinstance(config.backend_config["api_token"], SecretStr)
    backend, backend_config = _setup_backend(config)
    assert backend == "nemo_retriever"
    assert isinstance(backend_config["api_token"], SecretStr)
    assert secret not in repr(backend_config)


def test_service_environment_and_explicit_backend_config_are_equivalent(monkeypatch):
    values = {
        "NRL_BASE_URL": "https://nrl.example.test/",
        "NRL_API_TOKEN": "environment-secret",
        "NRL_SCOPE": "workspace-123",
        "NRL_CONNECT_TIMEOUT_S": "12",
        "NRL_REQUEST_TIMEOUT_S": "34",
        "NRL_MAX_RETRIES": "2",
        "NRL_MAX_CONCURRENCY": "3",
        "NRL_MAX_QUEUED_UPLOADS": "4",
        "NRL_VERIFY_SSL": "false",
        "NRL_COLLECTION_TTL_HOURS": "48",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    from_environment = adapter_module._settings({})
    explicit = adapter_module._settings(
        {
            "base_url": values["NRL_BASE_URL"],
            "api_token": SecretStr(values["NRL_API_TOKEN"]),
            "scope": values["NRL_SCOPE"],
            "connect_timeout_s": values["NRL_CONNECT_TIMEOUT_S"],
            "request_timeout_s": values["NRL_REQUEST_TIMEOUT_S"],
            "max_retries": values["NRL_MAX_RETRIES"],
            "max_concurrency": values["NRL_MAX_CONCURRENCY"],
            "max_queued_uploads": values["NRL_MAX_QUEUED_UPLOADS"],
            "verify_ssl": values["NRL_VERIFY_SSL"],
            "collection_ttl_hours": values["NRL_COLLECTION_TTL_HOURS"],
        }
    )
    public_config = KnowledgeRetrievalConfig(backend="nemo_retriever")

    assert from_environment == explicit
    assert adapter_module._settings(public_config.backend_config) == from_environment
    assert values["NRL_API_TOKEN"] not in repr(public_config)
    assert "environment-secret" not in repr(from_environment)


def test_service_upload_queue_bound_accepts_zero_and_rejects_negative() -> None:
    settings = adapter_module._settings(
        {
            "base_url": "https://nrl.example.test",
            "scope": "workspace-123",
            "max_concurrency": 1,
            "max_queued_uploads": 0,
        }
    )
    assert settings.max_queued_uploads == 0

    with pytest.raises(ValueError, match="nrl_max_queued_uploads must be zero or greater"):
        adapter_module._settings(
            {
                "base_url": "https://nrl.example.test",
                "scope": "workspace-123",
                "max_queued_uploads": -1,
            }
        )


def test_headers_collection_and_document_pagination(tmp_path):
    fake = FakeNRL(paginate=True)
    config = _adapter_config(fake)
    ingestor = NemoRetrieverIngestor(config)

    collections = ingestor.list_collections()
    assert [item.name for item in collections] == ["first", "second"]
    assert "table_name" not in collections[0].metadata

    first = tmp_path / "one.txt"
    second = tmp_path / "two.html"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("<p>beta</p>", encoding="utf-8")
    job_id = ingestor.submit_job([str(first), str(second)], "test")
    _wait_for_uploads(ingestor, job_id)
    files = ingestor.list_files("test")
    ingestor.get_job_status("job-1")
    assert {item.file_id for item in files} == {"stable-doc-0", "stable-doc-1"}

    assert all(request.headers["X-NRL-Scope"] == "workspace-123" for request in fake.requests)
    assert all(request.headers["Authorization"] == "Bearer super-secret" for request in fake.requests)
    resource_list_requests = [
        request
        for request in fake.requests
        if request.method == "GET" and request.url.path in {"/v1/collections", "/v1/collections/test/documents"}
    ]
    job_document_requests = [
        request
        for request in fake.requests
        if request.method == "GET" and request.url.path == "/v1/ingest/job/job-1/documents"
    ]
    assert all(request.url.params["limit"] == "100" for request in resource_list_requests)
    assert all(request.url.params["limit"] == "1000" for request in job_document_requests)


def test_deterministic_manifest_idempotency_and_status_mapping(tmp_path):
    fake = FakeNRL()
    ingestor = NemoRetrieverIngestor(_adapter_config(fake))
    one = tmp_path / "upload-a"
    two = tmp_path / "upload-b"
    one.write_text("same bytes", encoding="utf-8")
    two.write_text("other bytes", encoding="utf-8")

    job_id = ingestor.submit_job(
        [str(one), str(two)],
        "test",
        {"original_filenames": ["report.txt", "page.html"]},
    )
    replay_id = ingestor.submit_job(
        [str(one), str(two)],
        "test",
        {"original_filenames": ["report.txt", "page.html"]},
    )
    _wait_for_uploads(ingestor, replay_id)

    assert job_id == replay_id == "job-1"
    assert fake.create_job_bodies[0]["idempotency_key"] == fake.create_job_bodies[1]["idempotency_key"]
    expected_sha = hashlib.sha256(one.read_bytes()).hexdigest()
    entry = fake.create_job_bodies[0]["document_manifest"][0]
    assert entry["filename"] == "report.txt"
    assert entry["content_sha256"] == expected_sha
    assert entry["manifest_entry_id"] == hashlib.sha256(f"0\0report.txt\0{expected_sha}".encode()).hexdigest()

    status = ingestor.get_job_status(job_id)
    assert status.status == JobState.COMPLETED
    assert status.processed_files == 2
    assert {item.file_id for item in status.file_details} == {"stable-doc-0", "stable-doc-1"}
    assert status.metadata["attempt_ids"] == {
        "stable-doc-0": "attempt-0",
        "stable-doc-1": "attempt-1",
    }
    assert all(item.status == FileStatus.SUCCESS for item in status.file_details)


def test_default_idempotency_changes_only_when_public_metadata_changes(tmp_path):
    fake = FakeNRL()
    ingestor = NemoRetrieverIngestor(_adapter_config(fake))
    upload = tmp_path / "report.txt"
    upload.write_text("same content", encoding="utf-8")

    first = ingestor.submit_job([str(upload)], "test", {"metadata": {"department": "finance"}})
    _wait_for_uploads(ingestor, first)
    replay = ingestor.submit_job([str(upload)], "test", {"metadata": {"department": "finance"}})
    _wait_for_uploads(ingestor, replay)
    changed = ingestor.submit_job([str(upload)], "test", {"metadata": {"department": "legal"}})
    _wait_for_uploads(ingestor, changed)

    keys = [body["idempotency_key"] for body in fake.create_job_bodies]
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


def test_submit_job_returns_before_multipart_upload_and_reports_pending(tmp_path):
    fake = FakeNRL()
    upload_started = threading.Event()
    release_upload = threading.Event()

    def delayed(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/document"):
            upload_started.set()
            assert release_upload.wait(timeout=2)
        return fake(request)

    ingestor = NemoRetrieverIngestor(_adapter_config(delayed))
    upload = tmp_path / "report.txt"
    upload.write_text("content", encoding="utf-8")

    try:
        job_id = ingestor.submit_job([str(upload)], "test")
        assert job_id == "job-1"
        assert upload_started.wait(timeout=1)
        pending = ingestor.get_job_status(job_id)
        assert pending.status == JobState.PENDING
        assert pending.completed_at is None
        assert len(pending.file_details) == 1
        assert pending.file_details[0].file_name == "report.txt"
        assert pending.file_details[0].status == FileStatus.UPLOADING
        assert pending.file_details[0].file_id == fake.manifest[0]["manifest_entry_id"]
    finally:
        release_upload.set()

    _wait_for_uploads(ingestor, job_id)
    assert ingestor.get_job_status(job_id).status == JobState.COMPLETED


def test_service_upload_admission_is_atomic_and_capacity_returns_after_completion(tmp_path, caplog):
    fake = FakeNRL()
    upload_started = threading.Event()
    release_upload = threading.Event()

    def delayed(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/document"):
            upload_started.set()
            assert release_upload.wait(timeout=2)
        return fake(request)

    config = _adapter_config(delayed)
    config.update({"max_concurrency": 1, "max_queued_uploads": 1})
    ingestor = NemoRetrieverIngestor(config)
    files = [tmp_path / f"report-{index}.txt" for index in range(3)]
    for path in files:
        path.write_text("content", encoding="utf-8")

    try:
        job_id = ingestor.submit_job([str(files[0]), str(files[1])], "test")
        assert upload_started.wait(timeout=1)
        with pytest.raises(IngestionCapacityError, match="temporarily at capacity"):
            ingestor.submit_job([str(files[2])], "test")

        assert len(fake.create_job_bodies) == 1
        assert ingestor._outstanding_uploads == 2
        assert "requested=1, outstanding=2, limit=2" in caplog.text
        assert str(tmp_path) not in caplog.text

        release_upload.set()
        _wait_for_uploads(ingestor, job_id)
        assert ingestor._outstanding_uploads == 0

        retry_id = ingestor.submit_job([str(files[2])], "test")
        _wait_for_uploads(ingestor, retry_id)
        assert len(fake.create_job_bodies) == 2
        assert ingestor._outstanding_uploads == 0
    finally:
        release_upload.set()
        ingestor.close()


def test_service_upload_admission_rejects_oversized_batch_before_job_creation(tmp_path):
    fake = FakeNRL()
    config = _adapter_config(fake)
    config.update({"max_concurrency": 1, "max_queued_uploads": 1})
    ingestor = NemoRetrieverIngestor(config)
    files = [tmp_path / f"report-{index}.txt" for index in range(3)]
    for path in files:
        path.write_text("content", encoding="utf-8")

    try:
        with pytest.raises(IngestionBatchTooLargeError, match="exceeds the configured ingestion capacity"):
            ingestor.submit_job([str(path) for path in files], "test")
        assert fake.create_job_bodies == []
        assert ingestor._outstanding_uploads == 0
    finally:
        ingestor.close()


def test_service_upload_job_creation_and_scheduling_failures_release_capacity(tmp_path, monkeypatch):
    fake = FakeNRL()
    fail_job_creation = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fail_job_creation
        if fail_job_creation and request.method == "POST" and request.url.path == "/v1/ingest/job":
            fail_job_creation = False
            return _response(request, 503, {"detail": "unavailable"})
        return fake(request)

    config = _adapter_config(handler)
    config.update({"max_concurrency": 1, "max_queued_uploads": 0})
    ingestor = NemoRetrieverIngestor(config)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"
    for path in (first, second, third):
        path.write_text("content", encoding="utf-8")

    try:
        with pytest.raises(NemoRetrieverHTTPError):
            ingestor.submit_job([str(first)], "test")
        assert ingestor._outstanding_uploads == 0

        original_submit = ingestor._upload_executor.submit
        monkeypatch.setattr(ingestor._upload_executor, "submit", Mock(side_effect=RuntimeError("closed")))
        job_id = ingestor.submit_job([str(second)], "test", {"cleanup_files": True})
        _wait_for_uploads(ingestor, job_id)
        assert ingestor._outstanding_uploads == 0
        assert not second.exists()

        monkeypatch.setattr(ingestor._upload_executor, "submit", original_submit)
        retry_id = ingestor.submit_job([str(third)], "test")
        _wait_for_uploads(ingestor, retry_id)
        assert ingestor._outstanding_uploads == 0
    finally:
        ingestor.close()


def test_service_upload_executor_preserves_configured_concurrency(tmp_path):
    fake = FakeNRL()
    state_lock = threading.Lock()
    two_active = threading.Event()
    release_uploads = threading.Event()
    active = 0
    maximum_active = 0

    def measured(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        if request.method != "POST" or not request.url.path.endswith("/document"):
            return fake(request)
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_active.set()
        try:
            assert release_uploads.wait(timeout=2)
            return fake(request)
        finally:
            with state_lock:
                active -= 1

    config = _adapter_config(measured)
    config.update({"max_concurrency": 2, "max_queued_uploads": 2})
    ingestor = NemoRetrieverIngestor(config)
    files = [tmp_path / f"report-{index}.txt" for index in range(4)]
    for path in files:
        path.write_text("content", encoding="utf-8")

    try:
        job_id = ingestor.submit_job([str(path) for path in files], "test")
        assert two_active.wait(timeout=1)
        assert maximum_active == 2
        release_uploads.set()
        _wait_for_uploads(ingestor, job_id)
        assert maximum_active == 2
        assert ingestor._outstanding_uploads == 0
    finally:
        release_uploads.set()
        ingestor.close()


def test_service_shutdown_waits_for_submission_before_closing_executor(tmp_path):
    fake = FakeNRL()
    job_creation_started = threading.Event()
    release_job_creation = threading.Event()

    def delayed_job_creation(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/ingest/job":
            job_creation_started.set()
            assert release_job_creation.wait(timeout=2)
        return fake(request)

    config = _adapter_config(delayed_job_creation)
    config.update({"max_concurrency": 1, "max_queued_uploads": 0})
    ingestor = NemoRetrieverIngestor(config)
    upload = tmp_path / "report.txt"
    upload.write_text("content", encoding="utf-8")
    submitted_jobs: list[str] = []
    submission_errors: list[Exception] = []

    def submit() -> None:
        try:
            submitted_jobs.append(ingestor.submit_job([str(upload)], "test"))
        except Exception as error:  # pragma: no cover - asserted below
            submission_errors.append(error)

    submit_thread = threading.Thread(target=submit)
    close_thread = threading.Thread(target=ingestor.close)
    submit_thread.start()
    assert job_creation_started.wait(timeout=1)
    close_thread.start()
    with ingestor._submission_condition:
        assert ingestor._submission_condition.wait_for(lambda: ingestor._closed, timeout=1)
        assert ingestor._active_submissions == 1

    release_job_creation.set()
    submit_thread.join(timeout=2)
    close_thread.join(timeout=2)

    assert not submit_thread.is_alive()
    assert not close_thread.is_alive()
    assert submission_errors == []
    assert submitted_jobs == ["job-1"]
    assert ingestor._outstanding_uploads == 0


def test_background_upload_failure_is_terminal_and_secret_safe(tmp_path):
    fake = FakeNRL()
    private_path = str(tmp_path / "private")

    def rejected(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/document"):
            raise httpx.ConnectError(
                f"Bearer super-secret failed at {private_path}",
                request=request,
            )
        return fake(request)

    ingestor = NemoRetrieverIngestor(_adapter_config(rejected))
    upload = tmp_path / "report.txt"
    upload.write_text("content", encoding="utf-8")
    job_id = ingestor.submit_job([str(upload)], "test", {"cleanup_files": True})
    _wait_for_uploads(ingestor, job_id)

    status = ingestor.get_job_status(job_id)
    assert status.status == JobState.FAILED
    assert status.file_details[0].status == FileStatus.FAILED
    assert status.file_details[0].error_message == "NeMo Retriever document ingestion failed"
    assert "super-secret" not in repr(status)
    assert private_path not in repr(status)
    assert not upload.exists()


def test_query_mapping_citations_content_types_and_image_safety():
    fake = FakeNRL()
    fake.query_hits = [
        {
            "chunk_id": "text-1",
            "document_id": "doc-1",
            "text": "Text body",
            "distance": -0.1,
            "filename": "report.pdf",
            "page_number": 2,
            "content_type": "text",
            "source": "report.pdf",
            "source_id": "source-1",
            "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"table_name": "hidden", "section": "intro"},
        },
        {
            "chunk_id": "table-1",
            "document_id": "doc-1",
            "text": "Table caption",
            "distance": 0.2,
            "filename": "report.pdf",
            "page_number": 3,
            "content_type": "structured_table",
            "metadata": {"structured_data": {"columns": ["a"]}},
        },
        {
            "chunk_id": "chart-1",
            "document_id": "doc-1",
            "text": "Chart caption",
            "distance": 0.3,
            "filename": "report.pdf",
            "page_number": 4,
            "content_type": "bar_chart",
            "metadata": {},
        },
        {
            "chunk_id": "image-1",
            "document_id": "doc-1",
            "text": "Image caption",
            "distance": 0.4,
            "filename": "report.pdf",
            "page_number": 5,
            "content_type": "image",
            "stored_image_uri": "s3://private-bucket/image.png",
            "metadata": {"lancedb_uri": "/data/private"},
        },
        {
            "chunk_id": "image-2",
            "document_id": "doc-1",
            "text": "Public image",
            "distance": 0.5,
            "filename": "report.pdf",
            "page_number": None,
            "content_type": "figure",
            "stored_image_uri": "https://images.example.test/signed.png",
            "metadata": {},
        },
        {
            "chunk_id": "unknown-1",
            "document_id": "doc-1",
            "text": "Unknown type",
            "distance": 0.6,
            "filename": "notes.txt",
            "page_number": None,
            "content_type": "novel-modality",
            "metadata": {},
            "source": {"table_name": "hidden", "label": "logical source"},
        },
    ]
    retriever = NemoRetrieverRetriever(_adapter_config(fake))
    result = asyncio.run(retriever.retrieve("findings", "test", top_k=6))

    assert result.success
    assert [chunk.chunk_id for chunk in result.chunks] == [
        "text-1",
        "table-1",
        "chart-1",
        "image-1",
        "image-2",
        "unknown-1",
    ]
    assert [chunk.distance for chunk in result.chunks] == pytest.approx([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert all(chunk.score == 0.0 for chunk in result.chunks)
    assert [chunk.content_type for chunk in result.chunks] == [
        ContentType.TEXT,
        ContentType.TABLE,
        ContentType.CHART,
        ContentType.IMAGE,
        ContentType.IMAGE,
        ContentType.TEXT,
    ]
    assert result.chunks[0].display_citation == "report.pdf, p.2"
    assert result.chunks[0].metadata["document_id"] == "doc-1"
    assert result.chunks[0].metadata["bounding_box"] == [0.1, 0.2, 0.3, 0.4]
    assert "table_name" not in result.chunks[0].metadata
    assert result.chunks[1].structured_data == '{"columns": ["a"]}'
    assert result.chunks[3].image_storage_uri == "s3://private-bucket/image.png"
    assert result.chunks[3].image_url is None
    assert "lancedb_uri" not in result.chunks[3].metadata
    assert result.chunks[4].image_url == "https://images.example.test/signed.png"
    assert result.chunks[5].metadata["source"] == {"label": "logical source"}
    formatted = _format_results(result, "findings")
    assert "Vector Distance: -0.1 (lower is closer)" in formatted
    assert "Relevance Score:" not in formatted


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), float("-inf")])
def test_invalid_query_distances_are_rejected(distance):
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk-1",
            content="body",
            distance=distance,
            file_name="report.pdf",
            display_citation="report.pdf",
            content_type=ContentType.TEXT,
        )

    retriever = NemoRetrieverRetriever(_adapter_config(FakeNRL()))
    with pytest.raises(NemoRetrieverError, match="public API contract"):
        retriever.normalize(
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "body",
                "distance": distance,
                "filename": "report.pdf",
            }
        )


def test_filters_empty_results_and_malformed_response():
    fake = FakeNRL()
    retriever = NemoRetrieverRetriever(_adapter_config(fake))
    filtered = asyncio.run(retriever.retrieve("q", "test", filters={"author": "Kyle"}))
    assert not filtered.success
    assert "filters are not supported" in (filtered.error_message or "")
    assert not any(request.url.path == "/v1/query" for request in fake.requests)

    empty = asyncio.run(retriever.retrieve("q", "test"))
    assert empty.success
    assert empty.chunks == []

    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"not-json")

    malformed_retriever = NemoRetrieverRetriever(_adapter_config(malformed))
    result = asyncio.run(malformed_retriever.retrieve("q", "test"))
    assert not result.success
    assert "malformed JSON" in (result.error_message or "")


def test_async_operations_are_safe_across_event_loops():
    fake = FakeNRL()
    retriever = NemoRetrieverRetriever(_adapter_config(fake))

    assert asyncio.run(retriever.health_check())
    first = asyncio.run(retriever.retrieve("first query", "test"))
    second = asyncio.run(retriever.retrieve("second query", "test"))
    assert first.success
    assert second.success


def test_rest_retrievers_do_not_close_the_process_shared_transport(monkeypatch):
    shared_transport = SimpleNamespace(
        arequest_json=AsyncMock(return_value={"status": "ok"}),
        close=Mock(),
    )
    constructor = Mock(return_value=shared_transport)
    monkeypatch.setattr(adapter_module, "_NRLTransport", constructor)
    adapter_module._SHARED_TRANSPORTS.clear()
    try:
        config = {"base_url": "https://nrl.example.test", "scope": "workspace-123"}
        first = NemoRetrieverRetriever(config)
        second = NemoRetrieverRetriever(config)

        cleanup = getattr(first, "close", None)
        if callable(cleanup):
            cleanup()

        assert first._transport is second._transport
        assert asyncio.run(second.health_check()) is True
        assert constructor.call_count == 1
        shared_transport.close.assert_not_called()
    finally:
        adapter_module._SHARED_TRANSPORTS.clear()


def test_injected_client_and_request_headers_are_not_mutated():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request, 200, {"status": "ok"})

    client = httpx.Client(
        headers={"X-Caller-Default": "preserve"},
        transport=httpx.MockTransport(handler),
    )
    original_client_headers = dict(client.headers)
    request_headers = {"X-Request-Header": "preserve"}
    transport = _NRLTransport(
        base_url="https://nrl.example.test",
        scope="workspace-123",
        api_token="super-secret",
        connect_timeout_s=1,
        request_timeout_s=5,
        max_retries=0,
        verify_ssl=True,
        ca_bundle=None,
        client=client,
    )

    assert dict(client.headers) == original_client_headers
    transport.request_json("GET", "/v1/health", operation="health check", headers=request_headers)

    assert dict(client.headers) == original_client_headers
    assert request_headers == {"X-Request-Header": "preserve"}
    assert requests[0].headers["X-Caller-Default"] == "preserve"
    assert requests[0].headers["X-Request-Header"] == "preserve"
    assert requests[0].headers["X-NRL-Scope"] == "workspace-123"
    assert requests[0].headers["Authorization"] == "Bearer super-secret"


@pytest.mark.parametrize("retry_after", ["3600", "inf", "nan"])
def test_retry_after_is_bounded(monkeypatch: pytest.MonkeyPatch, retry_after: str):
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(request, 429, headers={"Retry-After": retry_after})
        return _response(request, 200, {"status": "ok"})

    monkeypatch.setattr(transport_module, "time", SimpleNamespace(sleep=delays.append))
    transport = _adapter_config(handler, retries=1)["_transport"]

    transport.request_json("GET", "/v1/health", operation="health check")

    assert calls == 2
    assert len(delays) == 1
    assert 0 <= delays[0] <= 30.0


def test_writes_retry_only_when_explicitly_safe(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls in {1, 2}:
            return _response(request, 503, {"detail": "retry"}, headers={"Retry-After": "0"})
        return _response(request, 200, {"status": "ok"})

    monkeypatch.setattr(transport_module, "time", SimpleNamespace(sleep=lambda _: None))
    transport = _adapter_config(handler, retries=1)["_transport"]

    with pytest.raises(NemoRetrieverHTTPError) as error:
        transport.request_json("POST", "/unsafe", operation="unsafe write", json={})
    assert error.value.status_code == 503
    assert calls == 1

    result = transport.request_json(
        "POST",
        "/idempotent",
        operation="idempotent write",
        retryable=True,
        json={},
    )
    assert result == {"status": "ok"}
    assert calls == 3


def test_adapter_boolean_settings_are_strict():
    false_config = _adapter_config(FakeNRL())
    false_config["verify_ssl"] = "false"
    ingestor = NemoRetrieverIngestor(false_config)
    assert ingestor._settings.verify_ssl is False

    invalid_config = _adapter_config(FakeNRL())
    invalid_config["verify_ssl"] = "definitely"
    with pytest.raises(ValueError, match="nrl_verify_ssl must be a boolean"):
        NemoRetrieverIngestor(invalid_config)


@pytest.mark.parametrize("status", [401, 403, 404, 409, 422])
def test_non_retryable_http_errors(status):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(request, status, {"detail": "rejected"})

    ingestor = NemoRetrieverIngestor(_adapter_config(handler, retries=3))
    with pytest.raises(NemoRetrieverHTTPError) as error:
        ingestor.create_collection("test")
    assert error.value.status_code == status
    assert calls == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_http_errors(status):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(request, status, {"detail": "retry"}, headers={"Retry-After": "0"})
        return _response(request, 200, {"status": "ok"})

    config = _adapter_config(handler, retries=1)
    assert asyncio.run(NemoRetrieverRetriever(config).health_check())
    assert calls == 2


def test_timeout_secret_redaction_and_job_api_mismatch(tmp_path):
    secret = "token-that-must-not-leak"  # pragma: allowlist secret

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"Bearer {secret}", request=request)

    transport = _adapter_config(timeout_handler, token=secret)["_transport"]
    with pytest.raises(NemoRetrieverTransportError) as error:
        transport.request_json("GET", "/v1/health", operation="health check")
    assert secret not in str(error.value)

    for status in (404, 410):

        def mismatch(request: httpx.Request, status_code: int = status) -> httpx.Response:
            return _response(request, status_code, {"detail": "missing"})

        ingestor = NemoRetrieverIngestor(_adapter_config(mismatch))
        with pytest.raises(NemoRetrieverHTTPError) as error:
            ingestor.get_job_status("missing-job")
        assert type(error.value) is NemoRetrieverHTTPError
        assert error.value.status_code == status

        upload = tmp_path / f"upload-{status}.txt"
        upload.write_text("content", encoding="utf-8")
        with pytest.raises(NemoRetrieverCompatibilityError, match="compatible collection-management API versions"):
            ingestor.submit_job([str(upload)], "test")


def test_partial_success_and_failed_file_status_mapping():
    fake = FakeNRL()
    fake.manifest = [
        {"manifest_entry_id": "a" * 64, "filename": "ok.txt", "content_sha256": "b" * 64},
        {"manifest_entry_id": "c" * 64, "filename": "bad.txt", "content_sha256": "d" * 64},
    ]
    fake.documents = {
        "stable-ok": {
            "document_id": "stable-ok",
            "attempt_id": "attempt-ok",
            "job_id": "job-1",
            "status": "completed",
            "submitted_at": NOW,
            "completed_at": NOW,
            "filename": "ok.txt",
            "result_rows": 2,
            "error": None,
            "collection_name": "test",
            "content_sha256": "b" * 64,
        },
        "stable-bad": {
            "document_id": "stable-bad",
            "attempt_id": "attempt-bad",
            "job_id": "job-1",
            "status": "failed",
            "submitted_at": NOW,
            "completed_at": NOW,
            "filename": "bad.txt",
            "result_rows": 0,
            "error": "extract failed",
            "collection_name": "test",
            "content_sha256": "d" * 64,
        },
    }

    original = fake.__call__

    def partial(request: httpx.Request) -> httpx.Response:
        response = original(request)
        if request.url.path == "/v1/ingest/job/job-1" and response.status_code == 200:
            body = response.json()
            body["status"] = "partial_success"
            return _response(request, 200, body)
        return response

    status = NemoRetrieverIngestor(_adapter_config(partial)).get_job_status("job-1")
    assert status.status == JobState.COMPLETED
    assert status.error_message
    assert [item.status for item in status.file_details] == [FileStatus.SUCCESS, FileStatus.FAILED]
    assert status.file_details[1].error_message == "NeMo Retriever document ingestion failed"
    assert "extract failed" not in status.file_details[1].error_message
    assert status.submitted_at == datetime.fromisoformat(NOW).astimezone(UTC)


def test_worker_errors_are_contained_in_every_public_document_model(tmp_path):
    fake = FakeNRL()
    force_direct_404 = False

    def handler(request: httpx.Request) -> httpx.Response:
        if force_direct_404 and request.method == "GET" and request.url.path.endswith("/stable-doc-0"):
            return _response(request, 404, {"detail": "not found"})
        return fake(request)

    ingestor = NemoRetrieverIngestor(_adapter_config(handler))
    upload = tmp_path / "report.txt"
    upload.write_text("content", encoding="utf-8")
    job_id = ingestor.submit_job([str(upload)], "test")
    _wait_for_uploads(ingestor, job_id)

    raw_error = (
        "Bearer super-secret failed at https://nrl.example.test/v1 "
        f"using {tmp_path}/private-ca.pem and nrl_{'a' * 40}"
    )  # pragma: allowlist secret
    fake.documents["stable-doc-0"]["status"] = "failed"
    fake.documents["stable-doc-0"]["error"] = raw_error

    job_error = ingestor.get_job_status("job-1").file_details[0].error_message
    list_error = ingestor.list_files("test")[0].error_message
    direct_error = ingestor.get_file_status("stable-doc-0", "test").error_message
    force_direct_404 = True
    fallback_error = ingestor.get_file_status("stable-doc-0", "test").error_message

    assert {job_error, list_error, direct_error, fallback_error} == {"NeMo Retriever document ingestion failed"}
    for public_error in (job_error, list_error, direct_error, fallback_error):
        assert "super-secret" not in public_error
        assert "nrl.example.test" not in public_error
        assert str(tmp_path) not in public_error
        assert f"nrl_{'a' * 40}" not in public_error
