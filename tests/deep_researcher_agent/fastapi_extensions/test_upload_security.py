# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded, content-validated document uploads."""

from __future__ import annotations

import asyncio
import os
import struct
import threading
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.routing import APIRoute
from starlette.datastructures import Headers

from deep_researcher_agent.fastapi_extensions import upload_security as upload_security_module
from deep_researcher_agent.fastapi_extensions.routes.documents import add_document_routes as add_legacy_document_routes
from deep_researcher_agent.fastapi_extensions.upload_security import UPLOAD_ENDPOINT_DESCRIPTION
from deep_researcher_agent.fastapi_extensions.upload_security import UploadLimits
from deep_researcher_agent.fastapi_extensions.upload_security import UploadValidationError
from deep_researcher_agent.fastapi_extensions.upload_security import ValidatedUploadBatch
from deep_researcher_agent.fastapi_extensions.upload_security import get_upload_limits
from deep_researcher_agent.fastapi_extensions.upload_security import save_validated_upload
from deep_researcher_agent.fastapi_extensions.upload_security import submit_validated_upload_batch
from deep_researcher_agent.fastapi_extensions.upload_security import validate_upload_count
from deep_researcher_agent.knowledge.base import IngestionBatchTooLargeError
from deep_researcher_agent.knowledge.base import IngestionCapacityError
from deep_researcher_api.routes.documents import add_document_routes as add_unified_document_routes

_DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class _NonSeekableZipSink:
    """Write-only sink that makes zipfile emit data descriptors."""

    def __init__(self) -> None:
        self._buffer = BytesIO()

    def write(self, content: bytes) -> int:
        return self._buffer.write(content)

    def flush(self) -> None:
        pass

    def getvalue(self) -> bytes:
        return self._buffer.getvalue()


def _upload(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def _limits(
    *,
    max_files: int = 2,
    max_file_bytes: int = 1024,
    max_total_bytes: int = 2048,
    accepted_extensions: frozenset[str] = frozenset({".txt", ".pdf"}),
) -> UploadLimits:
    return UploadLimits(
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        accepted_extensions=accepted_extensions,
    )


def _office_document(
    *,
    required_member: str = "word/document.xml",
    extra_entries: dict[str, bytes] | None = None,
) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr(required_member, b"<document/>")
        for name, content in (extra_entries or {}).items():
            archive.writestr(name, content)
    return output.getvalue()


def _streaming_office_document() -> bytes:
    output = _NonSeekableZipSink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("word/document.xml", b"<document/>")
    content = output.getvalue()
    flag_bits = struct.unpack_from("<H", content, 6)[0]
    assert flag_bits & 0x8
    assert b"PK\x07\x08" in content
    return content


def _forge_first_data_descriptor_crc(content: bytes, *, crc: int) -> bytes:
    forged = bytearray(content)
    descriptor_offset = forged.find(b"PK\x07\x08")
    assert descriptor_offset >= 0
    struct.pack_into("<I", forged, descriptor_offset + 4, crc)
    return bytes(forged)


def _mark_first_zip_entry_encrypted(content: bytes) -> bytes:
    encrypted = bytearray(content)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        header_offset = encrypted.find(signature)
        assert header_offset >= 0
        field_offset = header_offset + flag_offset
        flags = int.from_bytes(encrypted[field_offset : field_offset + 2], "little") | 0x1
        encrypted[field_offset : field_offset + 2] = flags.to_bytes(2, "little")
    return bytes(encrypted)


def _forge_central_entry_size(content: bytes, entry_name: str, *, uncompressed_size: int) -> bytes:
    forged = bytearray(content)
    offset = 0
    while (header_offset := forged.find(b"PK\x01\x02", offset)) >= 0:
        filename_length, extra_length, comment_length = struct.unpack_from("<HHH", forged, header_offset + 28)
        filename_start = header_offset + 46
        filename = bytes(forged[filename_start : filename_start + filename_length]).decode("utf-8")
        if filename == entry_name:
            struct.pack_into("<I", forged, header_offset + 24, uncompressed_size)
            return bytes(forged)
        offset = filename_start + filename_length + extra_length + comment_length
    raise AssertionError(f"central-directory entry not found: {entry_name}")


def _forge_entry_crc(content: bytes, entry_name: str, *, crc: int) -> bytes:
    forged = bytearray(content)
    local_offset = 0
    while (header_offset := forged.find(b"PK\x03\x04", local_offset)) >= 0:
        filename_length, extra_length = struct.unpack_from("<HH", forged, header_offset + 26)
        filename_start = header_offset + 30
        filename = bytes(forged[filename_start : filename_start + filename_length]).decode("utf-8")
        if filename == entry_name:
            struct.pack_into("<I", forged, header_offset + 14, crc)
            break
        local_offset = filename_start + filename_length + extra_length
    else:
        raise AssertionError(f"local entry not found: {entry_name}")

    central_offset = 0
    while (header_offset := forged.find(b"PK\x01\x02", central_offset)) >= 0:
        filename_length, extra_length, comment_length = struct.unpack_from("<HHH", forged, header_offset + 28)
        filename_start = header_offset + 46
        filename = bytes(forged[filename_start : filename_start + filename_length]).decode("utf-8")
        if filename == entry_name:
            struct.pack_into("<I", forged, header_offset + 16, crc)
            return bytes(forged)
        central_offset = filename_start + filename_length + extra_length + comment_length
    raise AssertionError(f"central-directory entry not found: {entry_name}")


def _reverse_central_directory_entries(content: bytes) -> bytes:
    """Reorder central records without changing their stable local-header offsets."""
    eocd_offset = content.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    central_size, central_offset = struct.unpack_from("<II", content, eocd_offset + 12)
    central_end = central_offset + central_size
    records: list[bytes] = []
    offset = central_offset
    while offset < central_end:
        assert content[offset : offset + 4] == b"PK\x01\x02"
        filename_length, extra_length, comment_length = struct.unpack_from("<HHH", content, offset + 28)
        record_end = offset + 46 + filename_length + extra_length + comment_length
        records.append(content[offset:record_end])
        offset = record_end
    assert offset == central_end

    reordered = bytearray(content)
    reordered[central_offset:central_end] = b"".join(reversed(records))
    return bytes(reordered)


def _upload_route(register_routes: Callable[[APIRouter], None]) -> APIRoute:
    router = APIRouter()
    register_routes(router)
    return next(
        route
        for route in router.routes
        if isinstance(route, APIRoute)
        and route.path == "/v1/collections/{collection_name}/documents"
        and "POST" in route.methods
    )


def _upload_endpoint(register_routes: Callable[[APIRouter], None]) -> Callable:
    return _upload_route(register_routes).endpoint


def _job_status_endpoint(register_routes: Callable[[APIRouter], None]) -> Callable:
    router = APIRouter()
    register_routes(router)
    return next(
        route.endpoint
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == "/v1/documents/{job_id}/status"
    )


class _FailingIngestor:
    def get_collection(self, collection_name: str) -> object:
        return object()

    def submit_job(self, *args, **kwargs) -> str:
        raise RuntimeError("secret database connection details")


class _SuccessfulIngestor:
    def __init__(self) -> None:
        self.submitted_paths: list[str] = []

    def get_collection(self, collection_name: str) -> object:
        return object()

    def submit_job(self, paths, *args, **kwargs) -> str:
        self.submitted_paths = list(paths)
        return "ingestion-job"

    def get_job_status(self, job_id: str) -> SimpleNamespace:
        return SimpleNamespace(file_details=[])


class _AdmissionFailingIngestor(_SuccessfulIngestor):
    def __init__(self, error_type: type[Exception]) -> None:
        super().__init__()
        self.error_type = error_type

    def submit_job(self, paths, *args, **kwargs) -> str:
        self.submitted_paths = list(paths)
        raise self.error_type()


class _BlockingIngestor(_SuccessfulIngestor):
    def __init__(self) -> None:
        super().__init__()
        self.submission_started = threading.Event()
        self.release_submission = threading.Event()

    def submit_job(self, paths, *args, **kwargs) -> str:
        self.submitted_paths = list(paths)
        self.submission_started.set()
        if not self.release_submission.wait(timeout=5):
            raise AssertionError("test did not release ingestion submission")
        return "ingestion-job"


class _JobStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__("private backend response")
        self.status_code = status_code


def test_loads_existing_shared_upload_environment_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FILE_UPLOAD_ACCEPTED_TYPES", "pdf, .DOCX, .pptx, .txt")
    monkeypatch.setenv("FILE_UPLOAD_MAX_SIZE_MB", "1.5")
    monkeypatch.setenv("FILE_UPLOAD_MAX_FILE_COUNT", "7")

    limits = get_upload_limits()

    assert limits.accepted_extensions == frozenset({".pdf", ".docx", ".pptx", ".txt"})
    assert limits.max_file_bytes == int(1.5 * 1024 * 1024)
    assert limits.max_total_bytes == limits.max_file_bytes
    assert limits.max_files == 7


@pytest.mark.parametrize(("status_code", "detail"), [(404, "not found"), (410, "expired")])
def test_unified_job_status_preserves_missing_and_expired_semantics(status_code: int, detail: str) -> None:
    endpoint = _job_status_endpoint(add_unified_document_routes)

    def get_job_status(_job_id: str) -> None:
        raise _JobStatusError(status_code)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(job_id="job-1", ingestor=SimpleNamespace(get_job_status=get_job_status)))

    assert exc_info.value.status_code == status_code
    assert detail in exc_info.value.detail
    assert "private backend response" not in exc_info.value.detail


def test_file_count_accepts_boundary_and_rejects_plus_one() -> None:
    limits = _limits(max_files=2)

    validate_upload_count(2, limits)
    with pytest.raises(UploadValidationError) as exc:
        validate_upload_count(3, limits)

    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_streams_valid_upload_to_private_temp_file() -> None:
    saved = await save_validated_upload(
        _upload("../unsafe/report.txt", b"grounded research", "text/plain"),
        limits=_limits(),
        remaining_total_bytes=2048,
    )
    try:
        assert saved.original_filename == "report.txt"
        assert saved.size_bytes == len(b"grounded research")
        assert Path(saved.path).read_bytes() == b"grounded research"
        assert os.stat(saved.path).st_mode & 0o777 == 0o600
    finally:
        os.unlink(saved.path)


@pytest.mark.asyncio
async def test_accepts_file_at_exact_size_boundary() -> None:
    saved = await save_validated_upload(
        _upload("boundary.txt", b"x" * 10, "text/plain"),
        limits=_limits(max_file_bytes=10),
        remaining_total_bytes=10,
    )
    try:
        assert saved.size_bytes == 10
    finally:
        os.unlink(saved.path)


@pytest.mark.asyncio
async def test_cancellation_waits_for_descriptor_write_before_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    write_started = threading.Event()
    release_write = threading.Event()
    captured_descriptor: list[int] = []
    created_paths: list[str] = []
    real_write = upload_security_module._write_all_to_descriptor

    def blocking_write(descriptor: int, content: bytes) -> None:
        captured_descriptor.append(descriptor)
        write_started.set()
        if not release_write.wait(timeout=5):
            raise AssertionError("test did not release descriptor write")
        real_write(descriptor, content)

    monkeypatch.setattr(upload_security_module, "_write_all_to_descriptor", blocking_write)
    save_task = asyncio.create_task(
        save_validated_upload(
            _upload("cancelled.txt", b"research", "text/plain"),
            limits=_limits(),
            remaining_total_bytes=2048,
            on_temp_path_created=created_paths.append,
        )
    )

    assert await asyncio.to_thread(write_started.wait, 5)
    save_task.cancel()
    await asyncio.sleep(0)
    assert not save_task.done()
    os.fstat(captured_descriptor[0])

    release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await save_task

    assert not Path(created_paths[0]).exists()
    with pytest.raises(OSError):
        os.fstat(captured_descriptor[0])


@pytest.mark.asyncio
async def test_rejects_file_larger_than_server_limit() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("large.txt", b"x" * 11, "text/plain"),
            limits=_limits(max_file_bytes=10),
            remaining_total_bytes=100,
        )

    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_rejects_file_that_exceeds_remaining_aggregate_limit() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("second.txt", b"x" * 6, "text/plain"),
            limits=_limits(max_file_bytes=10, max_total_bytes=10),
            remaining_total_bytes=5,
        )

    assert exc.value.status_code == 413
    assert exc.value.detail == "Total upload size limit exceeded"


@pytest.mark.asyncio
async def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("payload.exe", b"MZ", "application/octet-stream"),
            limits=_limits(),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_invalid_utf8_after_initial_validation_window() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.txt", (b"x" * 8192) + b"\xff", "text/plain"),
            limits=_limits(max_file_bytes=16 * 1024),
            remaining_total_bytes=16 * 1024,
        )

    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_accepts_docx_with_exact_required_members() -> None:
    saved = await save_validated_upload(
        _upload("report.docx", _office_document(), _DOCX_CONTENT_TYPE),
        limits=_limits(accepted_extensions=frozenset({".docx"})),
        remaining_total_bytes=2048,
    )
    try:
        assert saved.original_filename == "report.docx"
    finally:
        os.unlink(saved.path)


@pytest.mark.asyncio
async def test_accepts_docx_with_reordered_central_directory_entries() -> None:
    saved = await save_validated_upload(
        _upload("report.docx", _reverse_central_directory_entries(_office_document()), _DOCX_CONTENT_TYPE),
        limits=_limits(accepted_extensions=frozenset({".docx"})),
        remaining_total_bytes=2048,
    )
    try:
        assert saved.original_filename == "report.docx"
    finally:
        os.unlink(saved.path)


@pytest.mark.asyncio
async def test_accepts_streaming_docx_with_data_descriptors() -> None:
    saved = await save_validated_upload(
        _upload("report.docx", _streaming_office_document(), _DOCX_CONTENT_TYPE),
        limits=_limits(accepted_extensions=frozenset({".docx"})),
        remaining_total_bytes=2048,
    )
    try:
        assert saved.original_filename == "report.docx"
    finally:
        os.unlink(saved.path)


@pytest.mark.asyncio
async def test_accepts_pptx_with_exact_required_members() -> None:
    saved = await save_validated_upload(
        _upload("slides.pptx", _office_document(required_member="ppt/presentation.xml"), _PPTX_CONTENT_TYPE),
        limits=_limits(accepted_extensions=frozenset({".pptx"})),
        remaining_total_bytes=2048,
    )
    try:
        assert saved.original_filename == "slides.pptx"
    finally:
        os.unlink(saved.path)


@pytest.mark.asyncio
async def test_rejects_malformed_office_archive() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", b"not a ZIP archive", _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_docx_without_exact_required_document_member() -> None:
    content = _office_document(required_member="word/media/image.png")
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_office_archive_with_duplicate_entry_name() -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        content = _office_document(extra_entries={"word/document.xml": b"<duplicate/>"})

    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415
    assert "duplicate entries" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_office_archive_over_entry_count_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_security_module, "MAX_OFFICE_ARCHIVE_ENTRIES", 2)
    content = _office_document(extra_entries={"word/extra.xml": b"<extra/>"})

    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415
    assert "structure" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_encrypted_office_archive_entry() -> None:
    content = _mark_first_zip_entry_encrypted(_office_document())
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415
    assert "Encrypted" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_office_archive_with_zip_bomb_ratio() -> None:
    content = _office_document(extra_entries={"word/large.xml": b"A" * (256 * 1024)})
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(
                max_file_bytes=1024 * 1024,
                max_total_bytes=1024 * 1024,
                accepted_extensions=frozenset({".docx"}),
            ),
            remaining_total_bytes=1024 * 1024,
        )

    assert exc.value.status_code == 415
    assert "compression ratio" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_office_archive_when_central_directory_forges_small_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual decompressed bytes, not a forged central-directory size, enforce the ceiling."""
    large_entry = bytes(range(256)) * 16
    content = _forge_central_entry_size(
        _office_document(extra_entries={"word/large.xml": large_entry}),
        "word/large.xml",
        uncompressed_size=1,
    )
    monkeypatch.setattr(upload_security_module, "MAX_OFFICE_ENTRY_UNCOMPRESSED_BYTES", 1024)

    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(
                max_file_bytes=16 * 1024,
                max_total_bytes=16 * 1024,
                accepted_extensions=frozenset({".docx"}),
            ),
            remaining_total_bytes=16 * 1024,
        )

    assert exc.value.status_code == 415
    assert "expands beyond" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_office_archive_with_forged_crc() -> None:
    """CRC verification is computed from decompressed bytes, not trusted metadata."""
    content = _forge_entry_crc(_office_document(), "word/document.xml", crc=0)

    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415
    assert "metadata" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_office_archive_with_forged_data_descriptor_crc() -> None:
    content = _forge_first_data_descriptor_crc(_streaming_office_document(), crc=0)

    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.docx", content, _DOCX_CONTENT_TYPE),
            limits=_limits(accepted_extensions=frozenset({".docx"})),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415
    assert "metadata" in exc.value.detail


@pytest.mark.asyncio
async def test_rejects_declared_mime_mismatch() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.pdf", b"%PDF-1.7\n", "text/html"),
            limits=_limits(),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_content_that_does_not_match_extension() -> None:
    with pytest.raises(UploadValidationError) as exc:
        await save_validated_upload(
            _upload("report.pdf", b"not a PDF", "application/pdf"),
            limits=_limits(),
            remaining_total_bytes=2048,
        )

    assert exc.value.status_code == 415


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
def test_upload_route_documents_security_error_responses(
    register_routes: Callable[[APIRouter], None],
) -> None:
    route = _upload_route(register_routes)
    responses = route.responses

    assert route.description == UPLOAD_ENDPOINT_DESCRIPTION
    assert responses[413]["description"] == "Upload size, file-count, or ingestion-capacity limit exceeded"
    assert responses[415]["description"] == "Unsupported, malformed, or mismatched file content"
    assert responses[503]["description"] == "Document ingestion is temporarily at capacity"


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
@pytest.mark.parametrize(
    ("error_type", "status_code", "detail"),
    [
        (IngestionBatchTooLargeError, 413, "Upload batch exceeds the configured ingestion capacity"),
        (IngestionCapacityError, 503, "Document ingestion is temporarily at capacity; retry later"),
    ],
)
@pytest.mark.asyncio
async def test_upload_route_maps_admission_errors_and_cleans_request_files(
    register_routes: Callable[[APIRouter], None],
    error_type: type[Exception],
    status_code: int,
    detail: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILE_UPLOAD_ACCEPTED_TYPES", ".txt")
    endpoint = _upload_endpoint(register_routes)
    ingestor = _AdmissionFailingIngestor(error_type)

    with pytest.raises(HTTPException) as exc:
        await endpoint(
            collection_name="private",
            files=[_upload("report.txt", b"research", "text/plain")],
            ingestor=ingestor,
        )

    assert exc.value.status_code == status_code
    assert exc.value.detail == detail
    assert "Retry-After" not in (exc.value.headers or {})
    assert ingestor.submitted_paths
    assert all(not Path(path).exists() for path in ingestor.submitted_paths)


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
@pytest.mark.asyncio
async def test_mixed_disallowed_upload_batch_is_atomic_and_cleans_saved_files(
    register_routes: Callable[[APIRouter], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILE_UPLOAD_ACCEPTED_TYPES", ".txt")
    created_paths: list[str] = []
    real_save = upload_security_module.save_validated_upload

    async def capture_saved_path(*args, on_temp_path_created, **kwargs):
        def capture(path: str) -> None:
            created_paths.append(path)
            on_temp_path_created(path)

        return await real_save(*args, on_temp_path_created=capture, **kwargs)

    monkeypatch.setattr(upload_security_module, "save_validated_upload", capture_saved_path)
    endpoint = _upload_endpoint(register_routes)
    ingestor = _SuccessfulIngestor()

    with pytest.raises(HTTPException) as exc:
        await endpoint(
            collection_name="private",
            files=[
                _upload("report.txt", b"research", "text/plain"),
                _upload(
                    "slides.pptx",
                    _office_document(required_member="ppt/presentation.xml"),
                    _PPTX_CONTENT_TYPE,
                ),
            ],
            ingestor=ingestor,
        )

    assert exc.value.status_code == 415
    assert ingestor.submitted_paths == []
    assert created_paths
    assert all(not Path(path).exists() for path in created_paths)


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
@pytest.mark.asyncio
async def test_upload_route_does_not_expose_internal_ingestion_errors(
    register_routes: Callable[[APIRouter], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint = _upload_endpoint(register_routes)

    with pytest.raises(HTTPException) as exc:
        await endpoint(
            collection_name="private",
            files=[_upload("report.txt", b"research", "text/plain")],
            ingestor=_FailingIngestor(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to submit ingestion job"
    assert "RuntimeError" in caplog.text
    assert "secret database connection details" not in caplog.text


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
@pytest.mark.asyncio
async def test_upload_route_cancellation_removes_all_request_owned_temp_files(
    register_routes: Callable[[APIRouter], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation after one saved file removes every path still owned by the request."""
    endpoint = _upload_endpoint(register_routes)
    first_path = tmp_path / "first.txt"
    first_path.write_bytes(b"research")
    save_count = 0

    async def save_then_cancel(*_args, on_temp_path_created, **_kwargs):
        nonlocal save_count
        save_count += 1
        path = first_path if save_count == 1 else tmp_path / "second.txt"
        path.write_bytes(b"research")
        on_temp_path_created(str(path))
        if save_count == 1:
            return SimpleNamespace(path=str(path), original_filename="first.txt", size_bytes=8)
        raise asyncio.CancelledError

    monkeypatch.setattr(upload_security_module, "save_validated_upload", save_then_cancel)

    with pytest.raises(asyncio.CancelledError):
        await endpoint(
            collection_name="private",
            files=[
                _upload("first.txt", b"research", "text/plain"),
                _upload("second.txt", b"more", "text/plain"),
            ],
            ingestor=_SuccessfulIngestor(),
        )

    assert not first_path.exists()
    assert not (tmp_path / "second.txt").exists()


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
@pytest.mark.asyncio
async def test_upload_route_success_transfers_temp_file_ownership_to_ingestion(
    register_routes: Callable[[APIRouter], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A submitted job retains its input even if the request then finishes."""
    endpoint = _upload_endpoint(register_routes)
    submitted_path = tmp_path / "submitted.txt"
    submitted_path.write_bytes(b"research")

    async def save_upload(*_args, on_temp_path_created, **_kwargs):
        on_temp_path_created(str(submitted_path))
        return SimpleNamespace(path=str(submitted_path), original_filename="submitted.txt", size_bytes=8)

    monkeypatch.setattr(upload_security_module, "save_validated_upload", save_upload)
    ingestor = _SuccessfulIngestor()

    response = await endpoint(
        collection_name="private",
        files=[_upload("submitted.txt", b"research", "text/plain")],
        ingestor=ingestor,
    )

    assert response.job_id == "ingestion-job"
    assert ingestor.submitted_paths == [str(submitted_path)]
    assert submitted_path.exists()


@pytest.mark.parametrize("register_routes", [add_legacy_document_routes, add_unified_document_routes])
@pytest.mark.asyncio
async def test_upload_route_submission_does_not_block_event_loop(
    register_routes: Callable[[APIRouter], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint = _upload_endpoint(register_routes)
    submitted_path = tmp_path / "submitted.txt"
    submitted_path.write_bytes(b"research")

    async def save_upload(*_args, on_temp_path_created, **_kwargs):
        on_temp_path_created(str(submitted_path))
        return SimpleNamespace(path=str(submitted_path), original_filename="submitted.txt", size_bytes=8)

    monkeypatch.setattr(upload_security_module, "save_validated_upload", save_upload)
    ingestor = _BlockingIngestor()
    safety_fired = threading.Event()

    def release_submission_on_timeout() -> None:
        safety_fired.set()
        ingestor.release_submission.set()

    safety_timer = threading.Timer(10, release_submission_on_timeout)
    safety_timer.start()
    request_task = asyncio.create_task(
        endpoint(
            collection_name="private",
            files=[_upload("submitted.txt", b"research", "text/plain")],
            ingestor=ingestor,
        )
    )
    try:
        assert await asyncio.to_thread(ingestor.submission_started.wait, 5)
        await asyncio.sleep(0)
        assert not safety_fired.is_set(), "submit_job blocked the event loop until the deadlock fuse fired"
        assert not ingestor.release_submission.is_set(), "submit_job blocked the event loop"
        ingestor.release_submission.set()
        response = await asyncio.wait_for(request_task, 5)
    finally:
        safety_timer.cancel()
        safety_timer.join()
        ingestor.release_submission.set()
        if not request_task.done():
            await asyncio.gather(request_task, return_exceptions=True)
        submitted_path.unlink(missing_ok=True)

    assert not safety_fired.is_set()
    assert response.job_id == "ingestion-job"


@pytest.mark.asyncio
async def test_cancelled_submission_quiesces_worker_before_upload_cleanup(tmp_path: Path) -> None:
    submitted_path = tmp_path / "submitted.txt"
    submitted_path.write_bytes(b"research")
    batch = ValidatedUploadBatch(
        temp_paths=[str(submitted_path)],
        original_filenames=[submitted_path.name],
        total_bytes=submitted_path.stat().st_size,
    )
    submission_started = threading.Event()
    release_submission = threading.Event()

    def submit() -> str:
        submission_started.set()
        if not release_submission.wait(timeout=5):
            raise AssertionError("test did not release ingestion submission")
        return "ingestion-job"

    submit_task = asyncio.create_task(submit_validated_upload_batch(submit, batch))
    assert await asyncio.to_thread(submission_started.wait, 5)
    submit_task.cancel()
    await asyncio.sleep(0)

    assert not submit_task.done()
    assert submitted_path.exists()

    release_submission.set()
    with pytest.raises(asyncio.CancelledError):
        await submit_task

    assert batch._ownership_transferred
    assert submitted_path.exists()
