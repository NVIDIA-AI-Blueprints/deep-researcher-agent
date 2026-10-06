# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side validation and bounded streaming for document uploads."""

from __future__ import annotations

import asyncio
import binascii
import codecs
import logging
import math
import os
import struct
import tempfile
import zipfile
import zlib
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_FILES = 10
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_ACCEPTED_UPLOAD_TYPES = frozenset({".pdf", ".docx", ".txt", ".md"})
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
MAX_OFFICE_ARCHIVE_ENTRIES = 10_000
MAX_OFFICE_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_OFFICE_ENTRY_UNCOMPRESSED_BYTES = MAX_OFFICE_UNCOMPRESSED_BYTES
MAX_OFFICE_COMPRESSION_RATIO = 100
OFFICE_DECOMPRESSION_CHUNK_BYTES = 64 * 1024
UPLOAD_ENDPOINT_DESCRIPTION = (
    "Upload files for ingestion into a collection. Returns a job ID to poll ingestion status. "
    "Accepted formats and limits are deployment-configurable."
)
_OFFICE_REQUIRED_MEMBERS = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
}

_DECLARED_CONTENT_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    ".pptx": frozenset({"application/vnd.openxmlformats-officedocument.presentationml.presentation"}),
    ".txt": frozenset({"text/plain"}),
    ".md": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
    ".html": frozenset({"text/html"}),
    ".json": frozenset({"application/json", "text/json", "text/plain"}),
    ".csv": frozenset({"text/csv", "application/csv", "text/plain"}),
    ".yaml": frozenset({"application/yaml", "text/yaml", "text/plain"}),
    ".yml": frozenset({"application/yaml", "text/yaml", "text/plain"}),
    ".log": frozenset({"text/plain"}),
    ".png": frozenset({"image/png"}),
    ".jpg": frozenset({"image/jpeg"}),
    ".jpeg": frozenset({"image/jpeg"}),
}
_GENERIC_CONTENT_TYPES = frozenset({"", "application/octet-stream"})
_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".html", ".json", ".csv", ".yaml", ".yml", ".log"})
_ZIP_LOCAL_FILE_HEADER = struct.Struct("<4s5H3I2H")
_ZIP_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_ZIP_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_ZIP_DATA_DESCRIPTOR_SIGNATURE = 0x08074B50
_ZIP_DATA_DESCRIPTOR_FLAG = 0x08
_ZIP_ENCRYPTED_FLAGS = 0x41
_ZIP_UTF8_FLAG = 0x800
_ZIP64_SIZE_SENTINEL = 0xFFFFFFFF


class UploadValidationError(ValueError):
    """A safely reportable upload rejection."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class UploadLimits:
    """Configured upload resource limits."""

    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    accepted_extensions: frozenset[str]


@dataclass(frozen=True)
class SavedUpload:
    """A validated upload saved to a private temporary file."""

    path: str
    original_filename: str
    size_bytes: int


@dataclass
class ValidatedUploadBatch:
    """Request-owned validated files ready for ingestion-job submission."""

    temp_paths: list[str]
    original_filenames: list[str]
    total_bytes: int
    _ownership_transferred: bool = False

    def transfer_ownership(self) -> None:
        """Transfer temporary-file cleanup responsibility to the ingestion job."""
        self._ownership_transferred = True


async def submit_validated_upload_batch(
    submit_job: Callable[[], str],
    batch: ValidatedUploadBatch,
) -> str:
    """Run blocking ingestion submission without racing temporary-file cleanup.

    ``asyncio.to_thread`` cannot stop its worker when the request is cancelled.
    Wait for that worker to quiesce before the upload context can remove files;
    if submission succeeded, transfer cleanup ownership even though the client
    disconnected.
    """
    submit_task = asyncio.create_task(asyncio.to_thread(submit_job))
    try:
        job_id = await asyncio.shield(submit_task)
    except asyncio.CancelledError:
        while not submit_task.done():
            try:
                await asyncio.shield(submit_task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not submit_task.cancelled():
            error = submit_task.exception()
            if error is None:
                batch.transfer_ownership()
        raise
    batch.transfer_ownership()
    return job_id


@dataclass(frozen=True)
class _RawOfficeArchiveEntry:
    """ZIP entry facts measured from local headers and decompressed bytes."""

    filename: str
    header_offset: int
    flag_bits: int
    compress_type: int
    crc: int
    compressed_bytes: int
    uncompressed_bytes: int


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        return default
    if value < 1:
        logger.warning("%s must be positive; using default %d", name, default)
        return default
    return value


def _positive_megabytes_env(name: str, default_bytes: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default_bytes
    try:
        value = float(raw_value)
    except ValueError:
        value = 0
    if not math.isfinite(value) or value <= 0:
        logger.warning("%s must be a positive finite number; using default %d bytes", name, default_bytes)
        return default_bytes
    return int(value * 1024 * 1024)


def get_upload_limits() -> UploadLimits:
    """Load the shared frontend/backend upload limits from environment variables."""
    configured_types = os.environ.get("FILE_UPLOAD_ACCEPTED_TYPES")
    if configured_types is None:
        accepted_extensions = DEFAULT_ACCEPTED_UPLOAD_TYPES
    else:
        accepted_extensions = frozenset(
            extension if extension.startswith(".") else f".{extension}"
            for raw_extension in configured_types.split(",")
            if (extension := raw_extension.strip().lower())
        )
        if not accepted_extensions:
            logger.warning(
                "FILE_UPLOAD_ACCEPTED_TYPES is empty; using default accepted extensions",
            )
            accepted_extensions = DEFAULT_ACCEPTED_UPLOAD_TYPES

    max_bytes = _positive_megabytes_env("FILE_UPLOAD_MAX_SIZE_MB", DEFAULT_MAX_UPLOAD_BYTES)
    return UploadLimits(
        max_files=_positive_int_env("FILE_UPLOAD_MAX_FILE_COUNT", DEFAULT_MAX_UPLOAD_FILES),
        max_file_bytes=max_bytes,
        max_total_bytes=max_bytes,
        accepted_extensions=accepted_extensions,
    )


def validate_upload_count(file_count: int, limits: UploadLimits) -> None:
    """Reject requests that exceed the configured file-count limit."""
    if file_count > limits.max_files:
        raise UploadValidationError(413, f"At most {limits.max_files} files may be uploaded at once")


def _safe_filename(filename: str | None) -> str:
    normalized = (filename or "unknown").replace("\\", "/")
    return Path(normalized).name or "unknown"


def _validate_declared_type(extension: str, content_type: str | None) -> None:
    declared = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if declared in _GENERIC_CONTENT_TYPES:
        return
    allowed = _DECLARED_CONTENT_TYPES.get(extension)
    if allowed is None or declared not in allowed:
        raise UploadValidationError(415, f"Content type '{declared}' is not allowed for '{extension}' files")


def _read_exact(file_obj, size: int) -> bytes:
    """Read an exact number of archive bytes or reject a truncated payload."""
    content = file_obj.read(size)
    if len(content) != size:
        raise UploadValidationError(415, "Office document archive structure is invalid")
    return content


def _decode_zip_filename(raw_name: bytes, flag_bits: int) -> str:
    encoding = "utf-8" if flag_bits & _ZIP_UTF8_FLAG else "cp437"
    try:
        filename = raw_name.decode(encoding)
    except UnicodeDecodeError as exc:
        raise UploadValidationError(415, "Office document archive metadata is invalid") from exc
    if not filename or "\x00" in filename:
        raise UploadValidationError(415, "Office document archive metadata is invalid")
    return filename


def _account_decompressed_bytes(
    content: bytes,
    *,
    entry_bytes: int,
    prior_archive_bytes: int,
    crc: int,
) -> tuple[int, int]:
    """Update measured expansion and CRC while enforcing hard byte ceilings."""
    updated_entry_bytes = entry_bytes + len(content)
    if (
        updated_entry_bytes > MAX_OFFICE_ENTRY_UNCOMPRESSED_BYTES
        or prior_archive_bytes + updated_entry_bytes > MAX_OFFICE_UNCOMPRESSED_BYTES
    ):
        raise UploadValidationError(415, "Office document expands beyond the supported size limit")
    return updated_entry_bytes, binascii.crc32(content, crc) & 0xFFFFFFFF


def _read_deflated_entry(file_obj, *, prior_archive_bytes: int) -> tuple[int, int, int]:
    """Measure one raw DEFLATE stream without trusting ZIP size metadata."""
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    compressed_bytes = 0
    uncompressed_bytes = 0
    crc = 0

    while not decompressor.eof:
        chunk = file_obj.read(OFFICE_DECOMPRESSION_CHUNK_BYTES)
        if not chunk:
            raise UploadValidationError(415, "Office document archive contains truncated compressed data")
        pending = chunk
        while not decompressor.eof:
            remaining = min(
                MAX_OFFICE_ENTRY_UNCOMPRESSED_BYTES - uncompressed_bytes,
                MAX_OFFICE_UNCOMPRESSED_BYTES - prior_archive_bytes - uncompressed_bytes,
            )
            output_limit = min(max(remaining + 1, 1), OFFICE_DECOMPRESSION_CHUNK_BYTES)
            output = decompressor.decompress(pending, output_limit)
            unused = decompressor.unused_data
            unconsumed = decompressor.unconsumed_tail
            # At EOF CPython can expose the same post-stream bytes through both
            # unused_data and unconsumed_tail. They are alternatives, not
            # additive, for determining how much of this input was compressed
            # payload.
            remainder = unused if decompressor.eof else unconsumed
            consumed = len(pending) - len(remainder)
            if consumed < 0:
                raise UploadValidationError(415, "Office document compressed data is invalid")
            compressed_bytes += consumed
            uncompressed_bytes, crc = _account_decompressed_bytes(
                output,
                entry_bytes=uncompressed_bytes,
                prior_archive_bytes=prior_archive_bytes,
                crc=crc,
            )
            if decompressor.eof:
                if unused:
                    file_obj.seek(-len(unused), os.SEEK_CUR)
                break
            if unconsumed:
                pending = unconsumed
                continue
            if len(output) == output_limit:
                # max_length can leave already-decoded output buffered even
                # after every input byte was consumed. Drain it without reading
                # more archive bytes so the next ZIP record is never misfed to
                # the DEFLATE stream.
                pending = b""
                continue
            if pending and consumed == 0 and not output:
                raise UploadValidationError(415, "Office document compressed data is invalid")
            break

    return compressed_bytes, uncompressed_bytes, crc


def _read_stored_entry(
    file_obj,
    *,
    declared_size: int,
    prior_archive_bytes: int,
) -> tuple[int, int, int]:
    """Measure one stored ZIP entry using its bounded local-header extent."""
    if declared_size > MAX_OFFICE_ENTRY_UNCOMPRESSED_BYTES:
        raise UploadValidationError(415, "Office document expands beyond the supported size limit")
    remaining = declared_size
    uncompressed_bytes = 0
    crc = 0
    while remaining:
        content = _read_exact(file_obj, min(remaining, OFFICE_DECOMPRESSION_CHUNK_BYTES))
        remaining -= len(content)
        uncompressed_bytes, crc = _account_decompressed_bytes(
            content,
            entry_bytes=uncompressed_bytes,
            prior_archive_bytes=prior_archive_bytes,
            crc=crc,
        )
    return declared_size, uncompressed_bytes, crc


def _validate_measured_expansion(*, compressed_bytes: int, uncompressed_bytes: int) -> None:
    if uncompressed_bytes and (
        compressed_bytes == 0 or uncompressed_bytes > compressed_bytes * MAX_OFFICE_COMPRESSION_RATIO
    ):
        raise UploadValidationError(415, "Office document compression ratio exceeds the safety limit")


def _validate_data_descriptor(
    file_obj,
    *,
    crc: int,
    compressed_bytes: int,
    uncompressed_bytes: int,
) -> None:
    """Consume and verify a non-ZIP64 data descriptor."""
    first_value = struct.unpack("<I", _read_exact(file_obj, 4))[0]
    if first_value == _ZIP_DATA_DESCRIPTOR_SIGNATURE:
        following = _read_exact(file_obj, 12)
        declared_crc, declared_compressed, declared_uncompressed = struct.unpack("<III", following)
        if (declared_crc, declared_compressed, declared_uncompressed) != (
            crc,
            compressed_bytes,
            uncompressed_bytes,
        ):
            # A descriptor without the optional signature is ambiguous only when
            # the actual CRC itself equals the signature value.
            if crc != _ZIP_DATA_DESCRIPTOR_SIGNATURE:
                raise UploadValidationError(415, "Office document archive metadata is invalid")
            unsigned_compressed, unsigned_uncompressed = struct.unpack("<II", following[:8])
            if (unsigned_compressed, unsigned_uncompressed) != (compressed_bytes, uncompressed_bytes):
                raise UploadValidationError(415, "Office document archive metadata is invalid")
            file_obj.seek(-4, os.SEEK_CUR)
            return
    else:
        declared_crc = first_value
        declared_compressed, declared_uncompressed = struct.unpack("<II", _read_exact(file_obj, 8))

    if (declared_crc, declared_compressed, declared_uncompressed) != (
        crc,
        compressed_bytes,
        uncompressed_bytes,
    ):
        raise UploadValidationError(415, "Office document archive metadata is invalid")


def _scan_raw_office_entries(path: str) -> list[_RawOfficeArchiveEntry]:
    """Stream local ZIP records and measure every entry before reading the central directory."""
    entries: list[_RawOfficeArchiveEntry] = []
    seen_names: set[str] = set()
    total_compressed = 0
    total_uncompressed = 0

    with open(path, "rb") as file_obj:
        while True:
            header_offset = file_obj.tell()
            signature = _read_exact(file_obj, 4)
            if signature == _ZIP_CENTRAL_DIRECTORY_SIGNATURE:
                break
            if signature != _ZIP_LOCAL_FILE_SIGNATURE:
                raise UploadValidationError(415, "Office document archive structure is invalid")

            header = _ZIP_LOCAL_FILE_HEADER.unpack(signature + _read_exact(file_obj, _ZIP_LOCAL_FILE_HEADER.size - 4))
            (
                _signature,
                _version_needed,
                flag_bits,
                compress_type,
                _modified_time,
                _modified_date,
                declared_crc,
                declared_compressed,
                declared_uncompressed,
                filename_length,
                extra_length,
            ) = header
            if flag_bits & _ZIP_ENCRYPTED_FLAGS:
                raise UploadValidationError(415, "Encrypted Office documents are not supported")
            if _ZIP64_SIZE_SENTINEL in {declared_compressed, declared_uncompressed}:
                raise UploadValidationError(415, "ZIP64 Office documents are not supported")
            if compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise UploadValidationError(415, "Office document uses an unsupported compression method")

            filename = _decode_zip_filename(_read_exact(file_obj, filename_length), flag_bits)
            _read_exact(file_obj, extra_length)
            if filename in seen_names:
                raise UploadValidationError(415, "Office document archive contains duplicate entries")
            seen_names.add(filename)
            if len(seen_names) > MAX_OFFICE_ARCHIVE_ENTRIES:
                raise UploadValidationError(415, "Office document archive structure is invalid")

            uses_descriptor = bool(flag_bits & _ZIP_DATA_DESCRIPTOR_FLAG)
            if compress_type == zipfile.ZIP_STORED:
                if uses_descriptor:
                    raise UploadValidationError(415, "Streaming stored Office entries are not supported")
                compressed_bytes, uncompressed_bytes, crc = _read_stored_entry(
                    file_obj,
                    declared_size=declared_compressed,
                    prior_archive_bytes=total_uncompressed,
                )
            else:
                compressed_bytes, uncompressed_bytes, crc = _read_deflated_entry(
                    file_obj,
                    prior_archive_bytes=total_uncompressed,
                )

            _validate_measured_expansion(
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
            )
            if uses_descriptor:
                for declared, measured in (
                    (declared_crc, crc),
                    (declared_compressed, compressed_bytes),
                    (declared_uncompressed, uncompressed_bytes),
                ):
                    if declared not in {0, measured}:
                        raise UploadValidationError(415, "Office document archive metadata is invalid")
                _validate_data_descriptor(
                    file_obj,
                    crc=crc,
                    compressed_bytes=compressed_bytes,
                    uncompressed_bytes=uncompressed_bytes,
                )
            elif (declared_crc, declared_compressed, declared_uncompressed) != (
                crc,
                compressed_bytes,
                uncompressed_bytes,
            ):
                raise UploadValidationError(415, "Office document archive metadata is invalid")

            total_compressed += compressed_bytes
            total_uncompressed += uncompressed_bytes
            entries.append(
                _RawOfficeArchiveEntry(
                    filename=filename,
                    header_offset=header_offset,
                    flag_bits=flag_bits,
                    compress_type=compress_type,
                    crc=crc,
                    compressed_bytes=compressed_bytes,
                    uncompressed_bytes=uncompressed_bytes,
                )
            )

    if not entries:
        raise UploadValidationError(415, "Office document archive structure is invalid")
    _validate_measured_expansion(
        compressed_bytes=total_compressed,
        uncompressed_bytes=total_uncompressed,
    )
    return entries


def _validate_office_archive(path: str, *, extension: str) -> None:
    try:
        raw_entries = _scan_raw_office_entries(path)
        with zipfile.ZipFile(path) as archive:
            central_entries = archive.infolist()
            if len(central_entries) != len(raw_entries):
                raise UploadValidationError(415, "Office document archive metadata is invalid")
            central_by_header_offset = {entry.header_offset: entry for entry in central_entries}
            if len(central_by_header_offset) != len(central_entries):
                raise UploadValidationError(415, "Office document archive metadata is invalid")
            for raw_entry in raw_entries:
                central_entry = central_by_header_offset.get(raw_entry.header_offset)
                if (
                    central_entry is None
                    or central_entry.filename != raw_entry.filename
                    or central_entry.flag_bits != raw_entry.flag_bits
                    or central_entry.compress_type != raw_entry.compress_type
                    or central_entry.CRC != raw_entry.crc
                    or central_entry.compress_size != raw_entry.compressed_bytes
                    or central_entry.file_size != raw_entry.uncompressed_bytes
                ):
                    raise UploadValidationError(415, "Office document archive metadata is invalid")

            names = [entry.filename for entry in raw_entries]
            required_members = {"[Content_Types].xml", _OFFICE_REQUIRED_MEMBERS[extension]}
            if not required_members.issubset(names):
                raise UploadValidationError(415, "Office document content does not match its filename extension")
    except (zipfile.BadZipFile, zlib.error, struct.error) as exc:
        raise UploadValidationError(415, "Office document is not a valid archive") from exc


def _validate_file_content(path: str, extension: str) -> None:
    with open(path, "rb") as file_obj:
        header = file_obj.read(8192)
    if not header:
        raise UploadValidationError(415, "Empty documents are not supported")

    if extension == ".pdf":
        if b"%PDF-" not in header[:1024]:
            raise UploadValidationError(415, "PDF content does not match its filename extension")
        return
    if extension == ".docx":
        _validate_office_archive(path, extension=extension)
        return
    if extension == ".pptx":
        _validate_office_archive(path, extension=extension)
        return
    if extension == ".png":
        if not header.startswith(b"\x89PNG\r\n\x1a\n"):
            raise UploadValidationError(415, "PNG content does not match its filename extension")
        return
    if extension in {".jpg", ".jpeg"}:
        if not header.startswith(b"\xff\xd8\xff"):
            raise UploadValidationError(415, "JPEG content does not match its filename extension")
        return
    if extension in _TEXT_EXTENSIONS:
        try:
            decoder = codecs.getincrementaldecoder("utf-8-sig")()
            with open(path, "rb") as file_obj:
                while chunk := file_obj.read(UPLOAD_READ_CHUNK_BYTES):
                    if b"\x00" in chunk:
                        raise UploadValidationError(415, "Text document contains binary data")
                    decoder.decode(chunk, final=False)
                decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise UploadValidationError(415, "Text document is not valid UTF-8") from exc
        return

    raise UploadValidationError(415, f"No content validator is available for '{extension}' files")


def _write_all_to_descriptor(descriptor: int, content: bytes) -> None:
    """Write a complete chunk to the already-open private temporary file."""
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("Temporary upload write made no progress")
        remaining = remaining[written:]


async def _write_upload_chunk(descriptor: int, content: bytes) -> None:
    """Offload one descriptor write and let it quiesce before propagating cancellation."""
    write_task = asyncio.create_task(asyncio.to_thread(_write_all_to_descriptor, descriptor, content))
    try:
        await asyncio.shield(write_task)
    except asyncio.CancelledError:
        # asyncio.to_thread cannot stop the underlying OS write. Keep the
        # descriptor open until that write finishes so cleanup cannot close and
        # reuse the fd underneath the worker thread.
        while not write_task.done():
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not write_task.cancelled():
            write_task.exception()
        raise


async def save_validated_upload(
    upload: UploadFile,
    *,
    limits: UploadLimits,
    remaining_total_bytes: int,
    on_temp_path_created: Callable[[str], None] | None = None,
) -> SavedUpload:
    """Stream one upload to disk while enforcing type and byte limits."""
    original_filename = _safe_filename(upload.filename)
    extension = Path(original_filename).suffix.lower()
    if extension not in limits.accepted_extensions:
        raise UploadValidationError(415, f"File type '{extension or '(none)'}' is not allowed")
    _validate_declared_type(extension, upload.content_type)

    effective_limit = min(limits.max_file_bytes, remaining_total_bytes)
    if effective_limit < 1:
        raise UploadValidationError(413, "Total upload size limit exceeded")

    descriptor, path = tempfile.mkstemp(prefix="deep-researcher-upload-", suffix=extension)
    size_bytes = 0
    try:
        # Register request ownership before the first await. This closes the
        # cancellation race where the helper creates a file but the caller never
        # receives the SavedUpload return value.
        if on_temp_path_created is not None:
            on_temp_path_created(path)
        while chunk := await upload.read(UPLOAD_READ_CHUNK_BYTES):
            size_bytes += len(chunk)
            if size_bytes > effective_limit:
                if effective_limit == limits.max_file_bytes:
                    raise UploadValidationError(
                        413,
                        f"File exceeds the maximum upload size of {limits.max_file_bytes} bytes",
                    )
                raise UploadValidationError(413, "Total upload size limit exceeded")
            await _write_upload_chunk(descriptor, chunk)

        os.close(descriptor)
        descriptor = -1
        await asyncio.to_thread(_validate_file_content, path, extension)
        return SavedUpload(path=path, original_filename=original_filename, size_bytes=size_bytes)
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


@asynccontextmanager
async def validated_upload_batch(
    files: list[UploadFile],
    *,
    limits: UploadLimits | None = None,
) -> AsyncIterator[ValidatedUploadBatch]:
    """Validate and save one request's files, cleaning them until ownership transfers."""
    if not files:
        raise UploadValidationError(400, "No files provided")
    effective_limits = limits or get_upload_limits()
    validate_upload_count(len(files), effective_limits)
    batch = ValidatedUploadBatch(temp_paths=[], original_filenames=[], total_bytes=0)
    try:
        for upload in files:
            saved = await save_validated_upload(
                upload,
                limits=effective_limits,
                remaining_total_bytes=effective_limits.max_total_bytes - batch.total_bytes,
                on_temp_path_created=batch.temp_paths.append,
            )
            batch.total_bytes += saved.size_bytes
            batch.original_filenames.append(saved.original_filename)
            logger.debug("Saved validated upload to %s", saved.path)
        yield batch
    finally:
        if not batch._ownership_transferred:
            for path in batch.temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
