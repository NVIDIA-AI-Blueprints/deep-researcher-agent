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

"""ArtifactManager - the host-side harvest + validation envelope.

Harvests bytes via the backend's ``download_files`` (never a CLI), confines paths
to ``artifact_dir``, validates them (MIME-from-bytes, extension allowlist,
size/count quotas), hashes, dedups per-job by digest, persists via the
``ArtifactStore``, and emits SSE artifact events. The agent only ever sees
``artifact://<id>`` references; bytes never enter its context.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shlex
import threading
import uuid
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from typing import Any

from ..config import ArtifactCaptureConfig
from ..logging_utils import log_sandbox_failure
from .manifest import ManifestEntry
from .manifest import parse_manifest
from .models import Artifact
from .models import ArtifactKind
from .models import ArtifactProvenance
from .models import ArtifactStatus
from .store import ArtifactStore

if TYPE_CHECKING:
    from deepagents.backends.sandbox import BaseSandbox

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"

# Markdown image references the agent writes as ![caption](artifact://<filename or id>).
_ARTIFACT_REF_RE = re.compile(r"!\[([^\]]*)\]\(artifact://([^)]+)\)")

# Magic-number sniffing for the formats we inline-render or commonly produce.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)

_EXT_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".ipynb": "application/x-ipynb+json",
    ".pdf": "application/pdf",
}

_MIME_KIND: dict[str, ArtifactKind] = {
    "image/png": ArtifactKind.IMAGE,
    "image/jpeg": ArtifactKind.IMAGE,
    "image/webp": ArtifactKind.IMAGE,
    "image/gif": ArtifactKind.IMAGE,
    "image/svg+xml": ArtifactKind.IMAGE,
    "text/csv": ArtifactKind.TABLE,
    "application/json": ArtifactKind.DATASET,
    "text/markdown": ArtifactKind.TEXT,
    "application/x-ipynb+json": ArtifactKind.NOTEBOOK,
    "application/pdf": ArtifactKind.DOCUMENT,
}


# Raster images must be magic-confirmed; only these may render inline (the rest are
# download-only until/unless sanitized) to prevent stored-XSS via SVG/HTML/notebooks.
_RASTER_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_INLINE_SAFE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp"})


def _magic_mime(data: bytes) -> str | None:
    """Return the MIME implied by content magic bytes, or None if unrecognized."""
    for signature, mime in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sniff_mime(data: bytes, filename: str) -> str:
    """Resolve MIME from content magic bytes, falling back to the extension."""
    magic = _magic_mime(data)
    if magic is not None:
        return magic
    return _EXT_MIME.get(PurePosixPath(filename).suffix.lower(), "application/octet-stream")


def _resolve_mime(data: bytes, filename: str) -> str | None:
    """Resolve a trusted MIME or return None when content/extension indicate spoofing.

    Raster image extensions must be magic-confirmed; a mismatch between confident
    magic and the extension (e.g. a PDF named ``.png``) is rejected.
    """
    magic = _magic_mime(data)
    ext_mime = _EXT_MIME.get(PurePosixPath(filename).suffix.lower())
    if magic is not None:
        both_raster = ext_mime in _RASTER_IMAGE_MIMES and magic in _RASTER_IMAGE_MIMES
        if ext_mime is not None and ext_mime != magic and not both_raster:
            return None
        return magic
    if ext_mime in _RASTER_IMAGE_MIMES:
        # Claims to be a raster image but has no matching magic bytes -> spoof/corrupt.
        return None
    return ext_mime or "application/octet-stream"


def _sanitize(data: bytes, mime: str) -> bytes | None:
    """Validate active-content formats; return None to reject what we cannot make safe."""
    if mime == "image/svg+xml":
        # Regex stripping cannot fully neutralize SVG (javascript: URIs, <foreignObject>,
        # external references, CSS payloads), and the content endpoint serves bytes as the
        # stored MIME, so a partial clean still leaves a stored-XSS vector. Fail closed and
        # reject SVG until a vetted allowlist sanitizer (e.g. DOMPurify-equivalent) exists.
        return None
    return data


class ArtifactManager:
    """Harvests, validates, and persists artifacts produced inside the sandbox."""

    def __init__(
        self,
        *,
        job_id: str,
        backend: BaseSandbox,
        store: ArtifactStore,
        config: ArtifactCaptureConfig,
        artifact_dir: str,
        emit: Callable[[dict[str, Any]], None] | None = None,
        content_url_template: str = "/v1/jobs/async/job/{job_id}/artifacts/{artifact_id}/content",
    ) -> None:
        """Configure harvesting for one job against a backend, store, and artifact dir.

        Args:
            job_id: Owning job id used to scope and key artifacts.
            backend: Sandbox backend used to download/enumerate artifact files.
            store: Durable store for persisting metadata and bytes.
            config: Capture policy (quotas and allowed extensions).
            artifact_dir: Sandbox directory that confines harvestable paths.
            emit: Optional SSE emitter for artifact/warning events.
            content_url_template: Template for the artifact content endpoint URL.
        """
        self.job_id = job_id
        self.backend = backend
        self.store = store
        self.config = config
        self.artifact_dir = artifact_dir.rstrip("/")
        self._emit = emit
        self._content_url_template = content_url_template
        self._lock = threading.Lock()
        self._final_harvest_lock = threading.Lock()
        self._final_harvest_done = False
        self._final_harvest_result: list[Artifact] = []
        self._seen: set[tuple[str, str]] = set()
        self._total_bytes = 0
        self._count = 0

    def final_harvest(self) -> list[Artifact]:
        """Harvest at a terminal boundary, with a directory scan fallback."""
        if not self.config.enabled:
            return []
        with self._final_harvest_lock:
            if self._final_harvest_done:
                return list(self._final_harvest_result)
            captured = self._harvest(scan=True)
            self._final_harvest_result = list(captured)
            self._final_harvest_done = True
            return captured

    def harvest_after_execute(self) -> list[Artifact]:
        """Checkpoint manifest-declared artifacts after a sandbox execute call."""
        if not self.config.enabled:
            return []
        return self._harvest(scan=False)

    def resolve_report_references(self, markdown: str, artifacts: list[Artifact] | None = None) -> str:
        """Validate ``artifact://`` image references against this job's artifacts.

        The agent references artifacts by filename (``artifact://<filename>``) since
        it does not know the host-assigned id. Known references are rewritten to the
        durable artifact id and preserved (the logical scheme is kept for the UI/PDF/
        CLI to resolve at their edge); unknown or foreign references are dropped.

        Args:
            markdown: The report body to rewrite.
            artifacts: Pre-fetched artifacts for this job; loaded from the store when omitted.
        """
        if artifacts is None:
            try:
                artifacts = self.store.list(self.job_id)
            except Exception as exc:  # noqa: BLE001 - report resolution must not fail the job
                log_sandbox_failure(
                    logger,
                    operation="artifact_reference_load",
                    reason_code="artifact_store_read_failed",
                    exc=exc,
                    sandbox=self.job_id,
                )
                return markdown

        by_id = {a.artifact_id: a for a in artifacts}
        by_name = {a.filename: a for a in artifacts}

        def _replace(match: re.Match[str]) -> str:
            """Rewrite a known reference to its durable id; drop unknown ones."""
            caption = match.group(1)
            token = match.group(2).strip()
            artifact = by_id.get(token) or by_name.get(token) or by_name.get(PurePosixPath(token).name)
            if artifact is None:
                logger.warning("Dropping report reference to unknown artifact: %s", token)
                return ""
            return f"![{caption}](artifact://{artifact.artifact_id})"

        return _ARTIFACT_REF_RE.sub(_replace, markdown)

    def ensure_inline_artifacts_embedded(self, markdown: str, artifacts: list[Artifact] | None = None) -> str:
        """Append any harvested inline image not already embedded under a ``## Figures`` section.

        Safety net for when the model produces a chart but forgets to embed it: every
        magic-verified raster image flagged ``inline`` is guaranteed to surface in the report.
        Artifacts already referenced (by durable id) are left untouched and never duplicated.

        Args:
            markdown: The report body to augment.
            artifacts: Pre-fetched artifacts for this job; loaded from the store when omitted.
        """
        if artifacts is None:
            try:
                artifacts = self.store.list(self.job_id)
            except Exception as exc:  # noqa: BLE001 - embedding must not fail the job
                log_sandbox_failure(
                    logger,
                    operation="artifact_inline_load",
                    reason_code="artifact_store_read_failed",
                    exc=exc,
                    sandbox=self.job_id,
                )
                return markdown

        orphans = [
            a
            for a in artifacts
            if a.inline and a.kind == ArtifactKind.IMAGE and f"artifact://{a.artifact_id}" not in markdown
        ]
        if not orphans:
            return markdown

        lines = ["", "## Figures", ""]
        for artifact in orphans:
            caption = artifact.caption or artifact.title or artifact.filename
            lines.append(f"![{caption}](artifact://{artifact.artifact_id})")
            lines.append("")
        logger.info(
            "Auto-embedded %d inline figure(s) the report did not reference (job=%s)",
            len(orphans),
            self.job_id,
        )
        return markdown.rstrip() + "\n" + "\n".join(lines)

    def append_artifact_index(self, markdown: str, artifacts: list[Artifact] | None = None) -> str:
        """Append a ``## Generated Artifacts`` section crediting sandbox-produced outputs.

        Lists every harvested artifact (charts, CSVs, etc.) so figures and their backing data
        are credited alongside the report's external sources. Harvest is job-scoped (each job
        writes to its own ``artifact_dir/<job_id>``), so this lists only the current job's
        outputs - not leftovers from other jobs sharing a persistent sandbox.

        Args:
            markdown: The report body to augment.
            artifacts: Pre-fetched artifacts for this job; loaded from the store when omitted.
        """
        if artifacts is None:
            try:
                artifacts = self.store.list(self.job_id)
            except Exception as exc:  # noqa: BLE001 - indexing must not fail the job
                log_sandbox_failure(
                    logger,
                    operation="artifact_index_load",
                    reason_code="artifact_store_read_failed",
                    exc=exc,
                    sandbox=self.job_id,
                )
                return markdown

        if not artifacts:
            return markdown

        lines = ["", "## Generated Artifacts", ""]
        for artifact in artifacts:
            descriptor = artifact.caption or artifact.title or artifact.kind.value
            lines.append(f"- `{artifact.filename}` - {descriptor} (generated in the analysis sandbox)")
        return markdown.rstrip() + "\n" + "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _harvest(self, *, scan: bool) -> list[Artifact]:
        """Discover and capture artifacts under the lock; optionally scan the directory."""
        with self._lock:
            entries = self._discover(scan=scan)
            captured: list[Artifact] = []
            for entry in entries:
                artifact = self._capture(entry)
                if artifact is not None:
                    captured.append(artifact)
            if captured:
                # Structured lifecycle log (no secrets): provider tokens never appear here.
                logger.info(
                    "Artifact harvest: job=%s scan=%s captured=%d total_bytes=%d count=%d",
                    self.job_id,
                    scan,
                    len(captured),
                    self._total_bytes,
                    self._count,
                )
            return captured

    def _discover(self, *, scan: bool) -> list[ManifestEntry]:
        """Return manifest entries, unioned with a directory scan when ``scan`` is set."""
        entries: list[ManifestEntry] = list(self._read_manifest())
        if not scan:
            return entries
        # Final harvest: union the manifest with a directory scan so allowed outputs the
        # manifest omitted (e.g. a CSV written alongside a declared PNG) are still captured.
        # Dedup by path; manifest entries win since they carry title/caption/inline metadata.
        seen = {entry.path for entry in entries}
        for scanned in self._scan_dir():
            if scanned.path not in seen:
                entries.append(scanned)
                seen.add(scanned.path)
        return entries

    def _read_manifest(self) -> list[ManifestEntry]:
        """Download and parse ``manifest.json``; return [] if absent or invalid."""
        manifest_path = f"{self.artifact_dir}/{_MANIFEST_NAME}"
        try:
            responses = self.backend.download_files([manifest_path])
        except Exception:  # noqa: BLE001 - missing manifest is normal
            return []
        for resp in responses:
            _, content, error = _extract_download(resp)
            if error or content is None:
                continue
            manifest = parse_manifest(content.decode("utf-8", errors="replace"))
            if manifest is not None:
                return list(manifest.artifacts)
        return []

    def _scan_dir(self) -> list[ManifestEntry]:
        """Enumerate allowed files in the artifact dir (bounded, best-effort fallback)."""
        try:
            response = self.backend.execute(f"find {shlex.quote(self.artifact_dir)} -type f")
        except Exception as exc:  # noqa: BLE001 - scan is best-effort
            log_sandbox_failure(
                logger,
                operation="artifact_scan",
                reason_code="artifact_scan_failed",
                exc=exc,
                sandbox=self.job_id,
            )
            return []
        output = getattr(response, "output", "") or ""
        entries: list[ManifestEntry] = []
        # Bound the scan so a flood of files can't drive one download round-trip each;
        # generous relative to the count quota, which is the real ceiling on stored files.
        max_scan = max(self.config.max_file_count * 5, 100)
        for line in output.splitlines():
            if len(entries) >= max_scan:
                logger.warning("Artifact scan truncated at %d files for job %s", max_scan, self.job_id)
                break
            path = line.strip()
            if not path or path.endswith(f"/{_MANIFEST_NAME}"):
                continue
            ext = PurePosixPath(path).suffix.lower()
            if ext not in self.config.allow_extensions:
                continue
            entries.append(ManifestEntry(path=path))
        return entries

    def _capture(self, entry: ManifestEntry) -> Artifact | None:
        """Validate, download, sanitize, and persist one entry; return the stored artifact.

        Returns ``None`` when the entry is rejected (confinement, allowlist, quota, size,
        MIME spoofing, sanitization, or per-run dedup).
        """
        # 0. Path-traversal confinement.
        if not self._is_confined(entry.path):
            logger.warning("Rejecting artifact outside artifact_dir: %s", entry.path)
            return None

        # 1. Extension allowlist.
        ext = PurePosixPath(entry.path).suffix.lower()
        if ext not in self.config.allow_extensions:
            self._emit_warning(entry.path, f"extension {ext} not allowed")
            return None

        # 1b. Count quota, checked BEFORE download so a flood of files cannot drive one
        # transfer round-trip per file before the quota stops storing.
        if self._count >= self.config.max_file_count:
            self._emit_warning(entry.path, "artifact count quota exceeded; summarize remaining outputs in text")
            return None

        # 2. Download bytes.
        try:
            responses = self.backend.download_files([entry.path])
        except Exception:  # noqa: BLE001 - per-file failure must not fail the job
            self._emit_warning(entry.path, "download failed")
            return None
        if not responses:
            return None
        _, data, error = _extract_download(responses[0])
        if error or data is None:
            self._emit_warning(entry.path, error or "no content")
            return None

        # 3. Size cap.
        if len(data) > self.config.max_file_bytes:
            self._emit_warning(entry.path, f"exceeds max_file_bytes ({len(data)})")
            return None

        # 4. Quota (count + cumulative bytes).
        if self._count >= self.config.max_file_count or self._total_bytes + len(data) > self.config.max_total_bytes:
            self._emit_warning(entry.path, "artifact quota exceeded; summarize remaining outputs in text")
            return None

        # 5. MIME from bytes; reject content/extension mismatch (spoofing).
        filename = PurePosixPath(entry.path).name
        mime = _resolve_mime(data, filename)
        if mime is None:
            self._emit_warning(entry.path, "content does not match its declared type")
            return None

        # 6. Sanitize active content (e.g. SVG scripts) before persisting.
        sanitized = _sanitize(data, mime)
        if sanitized is None:
            self._emit_warning(entry.path, "failed sanitization")
            return None
        data = sanitized

        # 7. Hash + per-run dedup (after sanitization so the digest matches stored bytes).
        digest = hashlib.sha256(data).hexdigest()
        if (entry.path, digest) in self._seen:
            return None

        kind = entry.kind if entry.kind != ArtifactKind.OTHER else _MIME_KIND.get(mime, ArtifactKind.OTHER)
        # Render gate: only magic-verified raster images may be embedded inline; SVG,
        # notebooks, PDFs, etc. are download-only until/unless deeper sanitization exists.
        inline = bool(entry.inline) and mime in _INLINE_SAFE_MIMES

        artifact = Artifact(
            artifact_id=f"art_{uuid.uuid4().hex}",
            job_id=self.job_id,
            kind=kind,
            mime_type=mime,
            filename=filename,
            sandbox_path=entry.path,
            storage_uri="",  # assigned by the store
            sha256=digest,
            size_bytes=len(data),
            title=entry.title,
            caption=entry.caption,
            inline=inline,
            provenance=ArtifactProvenance(),
            status=ArtifactStatus.PENDING,
        )

        # 8. Store first (durable), then emit (outbox discipline).
        stored = self.store.put(artifact, data)
        self._seen.add((entry.path, digest))
        # A dedup hit returns a pre-existing artifact (different id); it was already
        # accounted for and emitted on first capture, so don't charge quota or emit again.
        if stored.artifact_id == artifact.artifact_id:
            self._total_bytes += len(data)
            self._count += 1
            self._emit_artifact(stored)
        return stored

    def _is_confined(self, path: str) -> bool:
        """Return whether the normalized path stays within ``artifact_dir``."""
        try:
            resolved = PurePosixPath(path)
            if not resolved.is_absolute():
                resolved = PurePosixPath(self.artifact_dir) / resolved
            normalized = _normalize_posix(resolved)
            base = _normalize_posix(PurePosixPath(self.artifact_dir))
            return normalized == base or normalized.startswith(base + "/")
        except Exception:  # noqa: BLE001
            return False

    def _emit_artifact(self, artifact: Artifact) -> None:
        """Emit an ``artifact.update`` SSE event with the artifact's content URL."""
        if self._emit is None:
            return
        content_url = self._content_url_template.format(job_id=self.job_id, artifact_id=artifact.artifact_id)
        self._emit(artifact.to_sse_payload(content_url))

    def _emit_warning(self, path: str, reason: str) -> None:
        """Log and emit an ``artifact.warning`` SSE event for a rejected file."""
        logger.warning("Artifact rejected (%s): %s", reason, path)
        if self._emit is None:
            return
        self._emit({"type": "artifact.warning", "data": {"path": path, "reason": reason}})


def _normalize_posix(path: PurePosixPath) -> str:
    """Collapse ``.`` / ``..`` segments without touching the filesystem."""
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if parts and parts[-1] not in ("", "/"):
                parts.pop()
        elif part not in (".", "", "/"):
            parts.append(part)
    prefix = "/" if path.is_absolute() else ""
    return prefix + "/".join(parts)


def _extract_download(resp: Any) -> tuple[str, bytes | None, str | None]:
    """Defensively read (path, bytes, error) from a FileDownloadResponse."""
    path = getattr(resp, "path", "") or ""
    error = getattr(resp, "error", None)
    content = getattr(resp, "content", None)
    if content is None:
        content = getattr(resp, "data", None)
    if isinstance(content, str):
        content = content.encode("utf-8")
    return path, content, error
