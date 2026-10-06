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

"""Tests for the artifact runtime: manifest parsing, MIME sniffing, the harvest
validation pipeline (traversal/extension/size/quota/dedup/scan), and SQL or S3-compatible stores."""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest

from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import Artifact
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import ArtifactKind
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import ArtifactManager
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import ArtifactStatus
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import S3ArtifactBlobStore
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import SqlArtifactStore
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import parse_manifest
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts.manager import _normalize_posix
from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts.manager import _sniff_mime
from deep_researcher_agent.agents.deep_researcher.sandbox.config import ArtifactCaptureConfig

_ARTIFACT_DIR = "/workspace/deep-researcher-artifacts"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_PNG_SHA256 = sha256(_PNG).hexdigest()


class _FakeBackend:
    """Minimal BaseSandbox stand-in mapping sandbox paths to bytes."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.execute_calls: list[str] = []
        self.download_calls: list[list[str]] = []

    def download_files(self, paths: list[str]) -> list[Any]:
        self.download_calls.append(list(paths))
        return [
            SimpleNamespace(path=p, content=self.files.get(p), error=None if p in self.files else "not found")
            for p in paths
        ]

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        self.execute_calls.append(command)
        return SimpleNamespace(output="\n".join(self.files), exit_code=0)


def _manifest_bytes(path: str, kind: str = "image") -> bytes:
    return json.dumps({"version": 1, "artifacts": [{"path": path, "kind": kind, "inline": True}]}).encode("utf-8")


def _make_manager(store: Any, files: dict[str, bytes], **capture: Any) -> tuple[ArtifactManager, list]:
    emitted: list = []
    config = ArtifactCaptureConfig(enabled=True, **capture)
    manager = ArtifactManager(
        job_id="job-1",
        backend=_FakeBackend(files),
        store=store,
        config=config,
        artifact_dir=_ARTIFACT_DIR,
        emit=emitted.append,
    )
    return manager, emitted


class TestManifest:
    def test_parse_valid(self) -> None:
        manifest = parse_manifest(_manifest_bytes(f"{_ARTIFACT_DIR}/c.png").decode())
        assert manifest is not None
        assert manifest.artifacts[0].path == f"{_ARTIFACT_DIR}/c.png"

    def test_parse_invalid_returns_none(self) -> None:
        assert parse_manifest("not json{") is None


class TestSniffMime:
    def test_png_by_magic(self) -> None:
        assert _sniff_mime(_PNG, "x.bin") == "image/png"

    def test_csv_by_extension(self) -> None:
        assert _sniff_mime(b"a,b\n1,2\n", "data.csv") == "text/csv"


class TestNormalizePosix:
    def test_absolute_path_has_single_leading_slash(self) -> None:
        from pathlib import PurePosixPath

        # The absolute-root sentinel must not be re-appended as a path segment.
        assert (
            _normalize_posix(PurePosixPath("/sandbox/deep-researcher-artifacts"))
            == "/sandbox/deep-researcher-artifacts"
        )

    def test_collapses_dot_and_parent_segments(self) -> None:
        from pathlib import PurePosixPath

        assert (
            _normalize_posix(PurePosixPath("/sandbox/./sub/../deep-researcher-artifacts"))
            == "/sandbox/deep-researcher-artifacts"
        )

    def test_relative_path_has_no_leading_slash(self) -> None:
        from pathlib import PurePosixPath

        assert _normalize_posix(PurePosixPath("sub/deep-researcher-artifacts")) == "sub/deep-researcher-artifacts"


class TestHarvest:
    def test_store_and_scan_failures_do_not_log_exception_text(self, caplog: pytest.LogCaptureFixture) -> None:
        store = SimpleNamespace(list=lambda _job_id: (_ for _ in ()).throw(RuntimeError("credential=do-not-log")))
        manager, _ = _make_manager(store, {})

        with caplog.at_level(logging.WARNING):
            assert manager.resolve_report_references("report") == "report"
            manager.backend.execute = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
                RuntimeError("token=do-not-log")
            )
            assert manager._scan_dir() == []

        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text
        assert "token=do-not-log" not in caplog.text

    def test_captures_manifest_artifact(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, emitted = _make_manager(store, files)

        captured = manager.final_harvest()

        assert len(captured) == 1
        assert captured[0].mime_type == "image/png"
        assert captured[0].kind == ArtifactKind.IMAGE
        assert store.list("job-1")[0].filename == "chart.png"
        assert emitted and emitted[0]["type"] == "artifact.update"
        assert emitted[0]["name"] == "chart.png"
        assert emitted[0]["data"]["type"] == "file"
        assert emitted[0]["data"]["artifact_id"] == captured[0].artifact_id
        assert emitted[0]["data"]["content_url"].endswith(f"/{captured[0].artifact_id}/content")
        assert "content" not in emitted[0]["data"]  # bytes and URL-as-text never enter the payload

    def test_rejects_path_traversal(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        evil = "/etc/passwd.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(evil), evil: _PNG}
        manager, _ = _make_manager(store, files)
        assert manager.final_harvest() == []

    def test_enforces_extension_allowlist(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        exe = f"{_ARTIFACT_DIR}/evil.exe"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(exe), exe: _PNG}
        manager, _ = _make_manager(store, files)
        assert manager.final_harvest() == []

    def test_enforces_size_cap(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, _ = _make_manager(store, files, max_file_bytes=8)
        assert manager.final_harvest() == []

    def test_enforces_quota(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        a = f"{_ARTIFACT_DIR}/a.png"
        b = f"{_ARTIFACT_DIR}/b.png"
        manifest = json.dumps(
            {"version": 1, "artifacts": [{"path": a, "kind": "image"}, {"path": b, "kind": "image"}]}
        ).encode("utf-8")
        files = {f"{_ARTIFACT_DIR}/manifest.json": manifest, a: _PNG, b: _PNG + b"x"}
        manager, _ = _make_manager(store, files, max_file_count=1)
        captured = manager.final_harvest()
        assert len(captured) == 1

    def test_dedups_identical_content(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, _ = _make_manager(store, files)
        manager.final_harvest()
        first_downloads = list(manager.backend.download_calls)
        manager.final_harvest()  # same bytes again
        assert len(store.list("job-1")) == 1
        assert manager.backend.download_calls == first_downloads

    def test_checkpoint_harvest_uses_manifest_without_directory_scan(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, _ = _make_manager(store, files)

        captured = manager.harvest_after_execute()

        assert [artifact.filename for artifact in captured] == ["chart.png"]
        assert manager.backend.execute_calls == []

    def test_final_harvest_after_checkpoint_does_not_reemit(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        csv_path = f"{_ARTIFACT_DIR}/chart.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path),
            png_path: _PNG,
            csv_path: b"state,pop\nCA,39431263\n",
        }
        manager, emitted = _make_manager(store, files)

        checkpointed = manager.harvest_after_execute()
        assert [artifact.filename for artifact in checkpointed] == ["chart.png"]

        finalized = manager.final_harvest()
        # chart.png was already captured at checkpoint; only the scan-discovered CSV is new.
        assert [artifact.filename for artifact in finalized] == ["chart.csv"]
        assert len(emitted) == 2

    def test_scan_fallback_without_manifest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {png_path: _PNG}  # no manifest
        manager, _ = _make_manager(store, files)
        captured = manager.final_harvest()  # job_end allows scan
        assert len(captured) == 1
        assert captured[0].filename == "chart.png"

    def test_final_harvest_unions_manifest_and_scan(self, tmp_path: Any) -> None:
        # A manifest that declares only the PNG must not hide a sibling CSV at job end.
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        csv_path = f"{_ARTIFACT_DIR}/chart.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path),
            png_path: _PNG,
            csv_path: b"state,pop\nCA,39431263\n",
        }
        manager, _ = _make_manager(store, files)

        captured = manager.final_harvest()

        names = sorted(a.filename for a in captured)
        assert names == ["chart.csv", "chart.png"]

    def test_rejects_mime_spoof(self, tmp_path: Any) -> None:
        # A file claiming .png but with non-image content must be rejected.
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: b"#!/bin/sh\nrm -rf /\n"}
        manager, _ = _make_manager(store, files)
        assert manager.final_harvest() == []

    def test_rejects_svg_fail_closed(self, tmp_path: Any) -> None:
        # SVG cannot be reliably sanitized (javascript: URIs, <foreignObject>, CSS payloads),
        # and the content endpoint serves bytes as the stored MIME, so SVG is rejected at
        # harvest rather than partially cleaned and stored (stored-XSS prevention).
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        svg_path = f"{_ARTIFACT_DIR}/diagram.svg"
        svg = b'<svg onload="steal()"><script>alert(1)</script><rect/></svg>'
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(svg_path), svg_path: svg}
        manager, _ = _make_manager(store, files)

        assert manager.final_harvest() == []


class TestStore:
    def _artifact(self) -> Artifact:
        return Artifact(
            artifact_id="art_" + "a" * 32,
            job_id="job-1",
            kind=ArtifactKind.IMAGE,
            mime_type="image/png",
            filename="chart.png",
            sandbox_path=f"{_ARTIFACT_DIR}/chart.png",
            storage_uri="",
            sha256=_PNG_SHA256,
            size_bytes=len(_PNG),
        )

    def test_put_get_list_open(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        stored = store.put(self._artifact(), _PNG)
        assert stored.status.value == "available"
        assert store.get("job-1", stored.artifact_id) is not None
        assert len(store.list("job-1")) == 1
        assert b"".join(store.open_bytes("job-1", stored.artifact_id)) == _PNG

    def test_dedup_by_digest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        first = store.put(self._artifact(), _PNG)
        # Same content/digest but a DIFFERENT artifact_id: dedup must key on (job_id, sha256),
        # not on artifact_id, so the second put returns the existing row instead of inserting
        # a duplicate. Reusing the same id here would let an id-based regression pass silently.
        duplicate = self._artifact().model_copy(update={"artifact_id": "art_" + "b" * 32})
        again = store.put(duplicate, _PNG)
        assert again.artifact_id == first.artifact_id
        assert len(store.list("job-1")) == 1

    def test_cleanup_removes_old_artifacts(self, tmp_path: Any) -> None:
        from sqlalchemy import text

        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        stored = store.put(self._artifact(), _PNG)
        with store._engine.connect() as conn:
            conn.execute(
                text("UPDATE artifacts SET created_at = datetime('now', '-2 seconds') WHERE artifact_id = :id"),
                {"id": stored.artifact_id},
            )
            conn.commit()

        assert store.cleanup_old_artifacts(1) == 1
        assert store.get(stored.job_id, stored.artifact_id) is None


class _FakeStreamingBody:
    def __init__(self, data: bytes) -> None:
        self._stream = BytesIO(data)
        self.closed = False

    def iter_chunks(self, chunk_size: int) -> Any:
        while chunk := self._stream.read(chunk_size):
            yield chunk

    def close(self) -> None:
        self.closed = True
        self._stream.close()


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.fail_put = False
        self.fail_delete = False
        self.head_bucket_calls: list[str] = []

    def put_object(self, **kwargs: Any) -> None:
        if self.fail_put:
            raise RuntimeError("upload failed")
        location = (kwargs["Bucket"], kwargs["Key"])
        self.objects[location] = kwargs["Body"]

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        return {"Body": _FakeStreamingBody(self.objects[(kwargs["Bucket"], kwargs["Key"])])}

    def delete_object(self, **kwargs: Any) -> None:
        if self.fail_delete:
            raise RuntimeError("delete failed")
        location = (kwargs["Bucket"], kwargs["Key"])
        self.objects.pop(location, None)

    def head_bucket(self, **kwargs: Any) -> None:
        self.head_bucket_calls.append(kwargs["Bucket"])


class TestS3Store:
    def _artifact(self) -> Artifact:
        return Artifact(
            artifact_id="art_" + "a" * 32,
            job_id="job-1",
            kind=ArtifactKind.IMAGE,
            mime_type="image/png",
            filename="chart.png",
            sandbox_path=f"{_ARTIFACT_DIR}/chart.png",
            storage_uri="",
            sha256=_PNG_SHA256,
            size_bytes=len(_PNG),
        )

    def _store(
        self,
        tmp_path: Any,
        client: _FakeS3Client,
    ) -> SqlArtifactStore:
        blob_store = S3ArtifactBlobStore(
            bucket="deep-researcher-artifacts",
            prefix="artifacts/v1",
            endpoint_url="http://minio:9000",
            client=client,
        )
        return SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db", blob_store=blob_store)

    def test_put_open_and_delete(self, tmp_path: Any) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)

        stored = store.put(self._artifact(), _PNG)

        assert stored.storage_uri == f"s3://deep-researcher-artifacts/artifacts/v1/job-1/{stored.artifact_id}"
        assert stored.status == ArtifactStatus.AVAILABLE
        assert b"".join(store.open_bytes("job-1", stored.artifact_id)) == _PNG
        assert store.delete_job("job-1") == 1
        assert client.objects == {}
        assert store.get("job-1", stored.artifact_id) is None

    def test_validate_checks_configured_bucket(self, tmp_path: Any) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)

        store.validate()

        assert client.head_bucket_calls == ["deep-researcher-artifacts"]

    def test_s3_bytes_leave_sql_content_null(self, tmp_path: Any) -> None:
        from sqlalchemy import text

        store = self._store(tmp_path, _FakeS3Client())
        stored = store.put(self._artifact(), _PNG)
        with store._engine.connect() as conn:
            content = conn.execute(
                text("SELECT content FROM artifacts WHERE artifact_id = :artifact_id"),
                {"artifact_id": stored.artifact_id},
            ).scalar_one()
        assert content is None

    def test_failed_upload_removes_metadata(self, tmp_path: Any, caplog: pytest.LogCaptureFixture) -> None:
        client = _FakeS3Client()
        client.fail_put = True
        store = self._store(tmp_path, client)
        artifact = self._artifact()

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="upload failed"):
                store.put(artifact, _PNG)

        assert store.get(artifact.job_id, artifact.artifact_id) is None
        assert b"".join(store.open_bytes(artifact.job_id, artifact.artifact_id)) == b""
        assert "RuntimeError" in caplog.text
        assert "upload failed" not in caplog.text

    def test_failed_delete_retains_metadata_for_retry(self, tmp_path: Any) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)
        stored = store.put(self._artifact(), _PNG)
        client.fail_delete = True

        assert store.delete_job(stored.job_id) == 0
        assert store.get(stored.job_id, stored.artifact_id) is not None

    def test_failed_metadata_delete_is_retried(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)
        stored = store.put(self._artifact(), _PNG)

        def fail_connect() -> None:
            raise RuntimeError("database unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(store._engine, "connect", fail_connect)
            assert store._delete_artifact(stored) == 0

        assert client.objects == {}
        assert store.get(stored.job_id, stored.artifact_id) is not None
        assert store._delete_artifact(stored) == 1
        assert store.get(stored.job_id, stored.artifact_id) is None


class TestArtifactStoreFactory:
    def test_sql_is_the_default_provider(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER", raising=False)

        store = build_artifact_store(f"sqlite:///{tmp_path}/jobs.db")

        assert isinstance(store, SqlArtifactStore)
        assert store._blob_store.scheme == "db"

    def test_s3_provider_uses_configured_blob_store(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import factory

        fake_blob_store = S3ArtifactBlobStore(bucket="deep-researcher-artifacts", client=_FakeS3Client())
        monkeypatch.setenv("DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER", "s3")
        monkeypatch.setenv("DEEP_RESEARCHER_ARTIFACT_S3_BUCKET", "deep-researcher-artifacts")
        monkeypatch.setattr(factory, "S3ArtifactBlobStore", lambda **_kwargs: fake_blob_store)
        factory._build_artifact_store.cache_clear()

        store = build_artifact_store(f"sqlite:///{tmp_path}/jobs.db")

        assert isinstance(store, SqlArtifactStore)
        assert store._blob_store is fake_blob_store
        factory._build_artifact_store.cache_clear()

    def test_rejects_unknown_provider(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from deep_researcher_agent.agents.deep_researcher.sandbox.artifacts import factory

        monkeypatch.setenv("DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER", "unknown")
        factory._build_artifact_store.cache_clear()

        with pytest.raises(ValueError, match="Unsupported DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER"):
            build_artifact_store(f"sqlite:///{tmp_path}/jobs.db")

        factory._build_artifact_store.cache_clear()


class TestEnsureInlineArtifactsEmbedded:
    """The safety net that surfaces produced inline figures the report forgot to embed."""

    def _store_with(self, tmp_path: Any, *artifacts: Artifact) -> SqlArtifactStore:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        for index, artifact in enumerate(artifacts):
            store.put(artifact, _PNG + bytes([index]))
        return store

    def _image(self, artifact_id: str, *, inline: bool = True) -> Artifact:
        return Artifact(
            artifact_id=artifact_id,
            job_id="job-1",
            kind=ArtifactKind.IMAGE,
            mime_type="image/png",
            filename=f"{artifact_id}.png",
            sandbox_path=f"{_ARTIFACT_DIR}/{artifact_id}.png",
            storage_uri="",
            sha256=artifact_id.ljust(64, "0"),
            size_bytes=len(_PNG),
            title="Chart Title",
            caption="A caption",
            inline=inline,
        )

    def test_appends_orphan_inline_image(self, tmp_path: Any) -> None:
        artifact_id = "art_" + "a" * 32
        store = self._store_with(tmp_path, self._image(artifact_id))
        manager, _ = _make_manager(store, {})

        result = manager.ensure_inline_artifacts_embedded("# Report\n\nNo figures here.\n")

        assert "## Figures" in result
        assert f"![A caption](artifact://{artifact_id})" in result

    def test_does_not_duplicate_referenced_image(self, tmp_path: Any) -> None:
        artifact_id = "art_" + "b" * 32
        store = self._store_with(tmp_path, self._image(artifact_id))
        manager, _ = _make_manager(store, {})
        markdown = f"# Report\n\n![Inline](artifact://{artifact_id})\n"

        result = manager.ensure_inline_artifacts_embedded(markdown)

        assert result == markdown
        assert "## Figures" not in result

    def test_append_artifact_index_lists_all_artifacts(self, tmp_path: Any) -> None:
        png = self._image("art_" + "e" * 32)
        csv = Artifact(
            artifact_id="art_" + "f" * 32,
            job_id="job-1",
            kind=ArtifactKind.TABLE,
            mime_type="text/csv",
            filename="chart.csv",
            sandbox_path=f"{_ARTIFACT_DIR}/chart.csv",
            storage_uri="",
            sha256="f" * 64,
            size_bytes=8,
            caption="Plotted values",
        )
        store = self._store_with(tmp_path, png, csv)
        manager, _ = _make_manager(store, {})

        result = manager.append_artifact_index("# Report\n\nBody.\n")

        assert "## Generated Artifacts" in result
        assert "`art_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png`" in result
        assert "`chart.csv` - Plotted values (generated in the analysis sandbox)" in result

    def test_append_artifact_index_noop_without_artifacts(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        manager, _ = _make_manager(store, {})
        assert manager.append_artifact_index("# Report\n") == "# Report\n"

    def test_skips_non_inline_and_non_image(self, tmp_path: Any) -> None:
        non_inline = self._image("art_" + "c" * 32, inline=False)
        csv = Artifact(
            artifact_id="art_" + "d" * 32,
            job_id="job-1",
            kind=ArtifactKind.TABLE,
            mime_type="text/csv",
            filename="data.csv",
            sandbox_path=f"{_ARTIFACT_DIR}/data.csv",
            storage_uri="",
            sha256="d" * 64,
            size_bytes=8,
            inline=True,
        )
        store = self._store_with(tmp_path, non_inline, csv)
        manager, _ = _make_manager(store, {})

        result = manager.ensure_inline_artifacts_embedded("# Report\n")

        assert result == "# Report\n"
