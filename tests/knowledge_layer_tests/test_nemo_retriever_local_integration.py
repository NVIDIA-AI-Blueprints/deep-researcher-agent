# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free integration coverage for embedded NeMo Retriever and LanceDB."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from knowledge_layer.nemo_retriever._local_client import _load_nrl_bindings
from knowledge_layer.nemo_retriever.local_adapter import NemoRetrieverLocalIngestor
from knowledge_layer.nemo_retriever.local_adapter import NemoRetrieverLocalRetriever
from PIL import Image
from PIL import ImageDraw

from deep_researcher_agent.knowledge import JobState
from deep_researcher_agent.knowledge.schema import ContentType

_PHYSICAL_TABLE_PATTERN = re.compile(r"\bnrl_[0-9a-f]{40}\b")
_UVICORN_PORT_PATTERN = re.compile(r"Uvicorn running on http://127\.0\.0\.1:(\d+)")


class _EmbeddingHandler(BaseHTTPRequestHandler):
    calls: list[dict[str, Any]] = []
    authorization_headers: list[str | None] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        self.calls.append(request)
        self.authorization_headers.append(self.headers.get("Authorization"))
        inputs = request.get("input", [])
        payload = {
            "object": "list",
            "model": request.get("model"),
            "data": [
                {"object": "embedding", "index": index, "embedding": [1.0, 0.0, 0.0]}
                for index, _value in enumerate(inputs)
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


@dataclass
class _RemoteFakeState:
    filenames: list[str] = field(default_factory=list)
    extract_params: list[Any] = field(default_factory=list)
    extract_call_kwargs: list[dict[str, Any]] = field(default_factory=list)
    embed_params: list[Any] = field(default_factory=list)
    query_calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)


class _DeterministicIngestor:
    """Replace only NRL's remote extraction/embedding graph for this test."""

    def __init__(self, state: _RemoteFakeState):
        self._state = state
        self._filename = ""
        self._buffer: BytesIO | None = None

    def buffers(self, value: tuple[str, BytesIO]) -> _DeterministicIngestor:
        self._filename, self._buffer = value
        self._state.filenames.append(self._filename)
        return self

    def extract(self, params: Any, **kwargs: Any) -> _DeterministicIngestor:
        self._state.extract_params.append(params)
        self._state.extract_call_kwargs.append(kwargs)
        return self

    def embed(self, params: Any) -> _DeterministicIngestor:
        self._state.embed_params.append(params)
        return self

    def ingest(self) -> pd.DataFrame:
        assert self._buffer is not None
        assert self._buffer.getvalue().startswith(b"%PDF")
        common_metadata = {
            "source_path": self._filename,
            "source_metadata": {
                "source_id": self._filename,
                "source_name": self._filename,
            },
        }
        return pd.DataFrame(
            [
                {
                    "text": "Quarterly revenue increased.",
                    "_content_type": "text",
                    "metadata": {
                        **common_metadata,
                        "content_metadata": {"page_number": 1},
                    },
                    "text_embeddings_1b_v2": {"embedding": [1.0, 0.0, 0.0]},
                },
                {
                    "text": "Blue bars show quarterly revenue growth.",
                    "_content_type": "chart_caption",
                    "metadata": {
                        **common_metadata,
                        "content_metadata": {
                            "page_number": 1,
                            "stored_image_uri": "file:///artifacts/revenue-chart.png",
                            "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
                        },
                    },
                    "text_embeddings_1b_v2": {"embedding": [0.95, 0.05, 0.0]},
                },
            ]
        )


def _make_bindings(state: _RemoteFakeState) -> Any:
    bindings = _load_nrl_bindings()

    def create_ingestor(*, run_mode: str) -> _DeterministicIngestor:
        assert run_mode == "inprocess"
        return _DeterministicIngestor(state)

    def infer_microservice(queries: list[str], **kwargs: Any) -> list[list[float]]:
        state.query_calls.append((queries, kwargs))
        return [[1.0, 0.0, 0.0] for _query in queries]

    return replace(
        bindings,
        create_ingestor=create_ingestor,
        infer_microservice=infer_microservice,
    )


def _write_multimodal_pdf(path: Path) -> None:
    image = Image.new("RGB", (480, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 25), "Quarterly revenue", fill="black")
    for index, height in enumerate((70, 120, 180)):
        left = 80 + index * 110
        draw.rectangle((left, 270 - height, left + 60, 270), fill=(30, 110, 210))
    image.save(path, "PDF", resolution=96)


def _write_text_pdf(path: Path) -> None:
    """Write a dependency-free PDF whose text is extractable by PDFium."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
    ]
    page_content = b"BT /F1 24 Tf 72 720 Td (Quarterly revenue increased substantially.) Tj ET\n"
    objects.append(b"<< /Length %d >>\nstream\n" % len(page_content) + page_content + b"endstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode())
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(document)


def _wait_for_job(ingestor: NemoRetrieverLocalIngestor, job_id: str) -> Any:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = ingestor.get_job_status(job_id)
        if status.is_terminal:
            return status
        time.sleep(0.02)
    raise AssertionError("embedded NeMo Retriever ingestion did not finish")


def _wait_for_server_port(process: subprocess.Popen[str], log_path: Path) -> int:
    deadline = time.monotonic() + 90
    log = ""
    while time.monotonic() < deadline:
        if log_path.exists():
            log = log_path.read_text(encoding="utf-8", errors="replace")
            match = _UVICORN_PORT_PATTERN.search(log)
            if match is not None:
                return int(match.group(1))
        if process.poll() is not None:
            break
        time.sleep(0.1)
    log = _PHYSICAL_TABLE_PATTERN.sub("[redacted-table]", log[-6000:])
    raise RuntimeError(f"nat serve did not publish its assigned port (exit={process.poll()}):\n{log}")


def _wait_until_ready(client: httpx.Client, process: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    log = _PHYSICAL_TABLE_PATTERN.sub("[redacted-table]", log[-6000:])
    raise RuntimeError(f"nat serve did not become ready (exit={process.poll()}):\n{log}")


def _wait_for_http_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        response = client.get(f"/v1/documents/{job_id}/status")
        response.raise_for_status()
        status = response.json()
        if status["status"] in {"completed", "failed"}:
            return status
        time.sleep(0.1)
    raise RuntimeError(f"ingestion job {job_id} did not finish")


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    """Stop NAT and wait for storage handles to close before test cleanup."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _assert_second_process_is_locked(data_dir: Path) -> None:
    script = """
import sys
from knowledge_layer.nemo_retriever._local_client import LocalRuntime
from knowledge_layer.nemo_retriever._local_client import LocalSettings
from knowledge_layer.nemo_retriever._local_client import NemoRetrieverLocalLockError

settings = LocalSettings.from_config({"data_dir": sys.argv[1], "scope": "local", "profile": "auto"})
try:
    runtime = LocalRuntime(settings)
except NemoRetrieverLocalLockError as error:
    raise SystemExit(0 if "already open by another process" in str(error) else 2)
runtime.close()
raise SystemExit(3)
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and source controlled test script
        [sys.executable, "-c", script, str(data_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


def _public_json(*values: Any) -> str:
    return json.dumps(
        [value.model_dump(mode="json") for value in values],
        sort_keys=True,
    )


def test_embedded_lancedb_lifecycle_survives_restart_without_ray_or_physical_id_leaks(tmp_path, monkeypatch):
    """Exercise Deep Researcher Agent lifecycle over actual pinned NRL collection APIs and operators."""
    nemo_params = pytest.importorskip(
        "nemo_retriever.common.params",
        reason="embedded LanceDB coverage runs in environments/nemo_retriever_local",
    )
    ray = pytest.importorskip("ray")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NRL_INFERENCE_API_KEY", raising=False)
    assert ray.is_initialized() is False

    state = _RemoteFakeState()
    bindings = _make_bindings(state)
    data_dir = tmp_path / "nemo-retriever"
    source = tmp_path / "multimodal-report.pdf"
    _write_multimodal_pdf(source)
    config = {
        "data_dir": str(data_dir),
        "scope": "local",
        "profile": "auto",
        "collection_ttl_hours": 24,
        "_bindings": bindings,
    }

    ingestor = NemoRetrieverLocalIngestor(config)
    retriever = NemoRetrieverLocalRetriever(config)
    try:
        runtime = ingestor._runtime
        assert type(runtime.vdb) is bindings.LanceDB
        assert type(runtime.ingest_operator) is bindings.IngestVdbOperator
        assert type(runtime.retrieve_operator) is bindings.RetrieveVdbOperator
        _assert_second_process_is_locked(data_dir)

        created = ingestor.create_collection(
            "reports",
            description="Local multimodal reports",
            metadata={"team": "research", "table_name": "must-not-leak"},
        )
        assert created.backend == "nemo_retriever_local"
        assert created.metadata["team"] == "research"
        assert "table_name" not in created.metadata

        first_job = ingestor.submit_job(
            [str(source)],
            "reports",
            {
                "original_filenames": ["multimodal-report.pdf"],
                "metadata": {"department": "finance"},
            },
        )
        first_status = _wait_for_job(ingestor, first_job)
        assert first_status.status == JobState.COMPLETED
        document_id = first_status.file_details[0].file_id
        assert first_status.file_details[0].chunks_created == 2
        assert not any((data_dir / ".staging").rglob("*"))

        files = ingestor.list_files("reports")
        assert len(files) == 1
        assert files[0].file_id == document_id
        assert files[0].file_name == "multimodal-report.pdf"
        assert files[0].chunk_count == 2

        first_result = asyncio.run(retriever.retrieve("revenue", "reports", top_k=5))
        assert first_result.success is True
        assert len(first_result.chunks) == 2
        assert all(chunk.score == 0 for chunk in first_result.chunks)
        assert all(chunk.distance is not None and math.isfinite(chunk.distance) for chunk in first_result.chunks)
        assert {chunk.content_type for chunk in first_result.chunks} == {ContentType.TEXT, ContentType.CHART}
        assert all(chunk.metadata["department"] == "finance" for chunk in first_result.chunks)

        public = _public_json(created, *files, *first_result.chunks)
        assert _PHYSICAL_TABLE_PATTERN.search(public) is None
        assert str(data_dir / "lancedb") not in public
        assert not {"physical_table", "lancedb_uri", "table_name"} & {
            key for chunk in first_result.chunks for key in chunk.metadata
        }
        assert bindings.default_embed_endpoint not in json.dumps(created.metadata)

        retry_job = ingestor.submit_job(
            [str(source)],
            "reports",
            {
                "original_filenames": ["multimodal-report.pdf"],
                "metadata": {"department": "finance"},
            },
        )
        retry_status = _wait_for_job(ingestor, retry_job)
        assert retry_status.status == JobState.COMPLETED
        assert retry_status.file_details[0].file_id == document_id
        assert retry_status.file_details[0].chunks_created == 2
        assert len(ingestor.list_files("reports")) == 1
        assert ingestor.get_collection("reports").chunk_count == 2

        expected_extract_defaults = nemo_params.ExtractParams().model_dump()
        assert len(state.extract_params) == 2
        assert all(params.model_dump() == expected_extract_defaults for params in state.extract_params)
        assert all(params.embed_modality == "text" for params in state.embed_params)
        assert all(params.input_type == "passage" for params in state.embed_params)
        assert state.filenames == ["multimodal-report.pdf", "multimodal-report.pdf"]
        assert state.query_calls[0][0] == ["revenue"]
        assert state.query_calls[0][1]["input_type"] == "query"
        assert state.query_calls[0][1]["nvidia_api_key"] is None
        assert ray.is_initialized() is False
    finally:
        retriever.close()
        ingestor.close()

    restarted_ingestor = NemoRetrieverLocalIngestor(config)
    restarted_retriever = NemoRetrieverLocalRetriever(config)
    try:
        reconciliation = restarted_ingestor._runtime.vdb.reconcile_collections()
        assert reconciliation == {"successes": 0, "failures": 0}
        collections = restarted_ingestor.list_collections()
        assert [collection.name for collection in collections] == ["reports"]
        assert collections[0].file_count == 1
        assert collections[0].chunk_count == 2

        restarted_result = asyncio.run(restarted_retriever.retrieve("revenue", "reports", top_k=5))
        assert restarted_result.success is True
        assert len(restarted_result.chunks) == 2
        assert all(chunk.score == 0 for chunk in restarted_result.chunks)
        assert all(chunk.distance is not None for chunk in restarted_result.chunks)
        assert _PHYSICAL_TABLE_PATTERN.search(_public_json(*collections, *restarted_result.chunks)) is None

        assert restarted_ingestor.delete_file(document_id, "reports") is True
        assert restarted_ingestor.list_files("reports") == []
        assert restarted_ingestor.delete_collection("reports") is True
        assert restarted_ingestor.list_collections() == []
        assert ray.is_initialized() is False
    finally:
        restarted_retriever.close()
        restarted_ingestor.close()


def test_real_fast_text_pipeline_uses_only_local_process_and_embedding_endpoint(tmp_path, monkeypatch):
    """Run NRL extraction and embedding without a Retriever service or generative LLM."""
    ray = pytest.importorskip("ray", reason="real NRL extraction runs in the isolated Python 3.12 environment")
    pytest.importorskip("nemo_retriever")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NRL_INFERENCE_API_KEY", raising=False)
    assert ray.is_initialized() is False

    _EmbeddingHandler.calls = []
    _EmbeddingHandler.authorization_headers = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    source = tmp_path / "real-fast-text.pdf"
    _write_text_pdf(source)
    config = {
        "data_dir": str(tmp_path / "real-fast-text-data"),
        "scope": "local",
        "profile": "fast-text",
        "embed_invoke_url": f"http://127.0.0.1:{server.server_port}/v1/embeddings",
        "embed_model_name": "nvidia/test-embedding",
        "collection_ttl_hours": 24,
    }
    ingestor = NemoRetrieverLocalIngestor(config)
    retriever = NemoRetrieverLocalRetriever(config)
    try:
        ingestor.create_collection("reports")
        job_id = ingestor.submit_job([str(source)], "reports")
        status = _wait_for_job(ingestor, job_id)
        assert status.status == JobState.COMPLETED
        assert status.file_details[0].chunks_created > 0

        result = asyncio.run(retriever.retrieve("revenue", "reports", top_k=3))
        assert result.success is True
        assert result.chunks
        assert "revenue" in result.chunks[0].content.lower()
        assert all(chunk.score == 0 for chunk in result.chunks)
        assert all(chunk.distance is not None and math.isfinite(chunk.distance) for chunk in result.chunks)
        assert {call["input_type"] for call in _EmbeddingHandler.calls} == {"passage", "query"}
        assert set(_EmbeddingHandler.authorization_headers) == {None}
        assert ray.is_initialized() is False
    finally:
        retriever.close()
        ingestor.close()
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_native_nat_serve_ingest_and_delete_without_retriever_service(tmp_path):
    """Exercise collection and document APIs through a native zero-deployment server."""
    pytest.importorskip("nemo_retriever", reason="native local E2E runs in the isolated Python 3.12 environment")
    project_root = Path(__file__).resolve().parents[2]
    executable_name = "nat.exe" if os.name == "nt" else "nat"
    nat_executable = Path(sys.executable).parent / executable_name
    assert nat_executable.is_file(), f"NAT executable is missing from the isolated environment: {nat_executable}"

    _EmbeddingHandler.calls = []
    _EmbeddingHandler.authorization_headers = []
    embedding_server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    embedding_thread = threading.Thread(target=embedding_server.serve_forever, daemon=True)
    embedding_thread.start()
    process: subprocess.Popen[str] | None = None
    try:
        source = tmp_path / "real-fast-text.pdf"
        log_path = tmp_path / "nat-serve.log"
        _write_text_pdf(source)
        env = dict(os.environ)
        for credential_name in ("NVIDIA_API_KEY", "NGC_API_KEY", "NRL_INFERENCE_API_KEY"):
            env.pop(credential_name, None)
        env.update(
            {
                "DEEP_RESEARCHER_AGENT_LLM_API_KEY": "local",  # pragma: allowlist secret
                "DEEP_RESEARCHER_AGENT_LLM_BASE_URL": f"http://127.0.0.1:{embedding_server.server_port}/v1",
                "DEEP_RESEARCHER_AGENT_LLM_MODEL": "openai/local-tool-model",
                "DEEP_RESEARCHER_CHECKPOINT_DB": str(tmp_path / "checkpoints.db"),
                "COLLECTION_NAME": "reports",
                "NAT_JOB_STORE_DB_URL": f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}",
                "NRL_EMBED_INVOKE_URL": f"http://127.0.0.1:{embedding_server.server_port}/v1/embeddings",
                "NRL_EMBED_MODEL_NAME": "nvidia/test-embedding",
                "NRL_LOCAL_DATA_DIR": str(tmp_path / "nemo-retriever"),
                "NRL_LOCAL_PROFILE": "fast-text",
                "NRL_SCOPE": "local",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(  # noqa: S603 - repository-owned executable and fixed arguments
                [
                    str(nat_executable),
                    "serve",
                    "--config_file",
                    str(project_root / "configs" / "config_web_nemo_retriever_local.yml"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                ],
                cwd=project_root,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                port = _wait_for_server_port(process, log_path)
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
                    _wait_until_ready(client, process, log_path)
                    created = client.post("/v1/collections", json={"name": "reports"})
                    created.raise_for_status()
                    uploaded = client.post(
                        "/v1/collections/reports/documents",
                        files={"files": (source.name, source.read_bytes(), "application/pdf")},
                    )
                    uploaded.raise_for_status()
                    upload = uploaded.json()
                    status = _wait_for_http_job(client, upload["job_id"])
                    assert status["status"] == "completed", status
                    assert status["file_details"][0]["chunks_created"] > 0

                    deleted = client.request(
                        "DELETE",
                        "/v1/collections/reports/documents",
                        json={"file_ids": upload["file_ids"]},
                    )
                    deleted.raise_for_status()
                    assert deleted.json()["total_deleted"] == 1
                    client.delete("/v1/collections/reports").raise_for_status()
            finally:
                # Windows cannot remove SQLite/LanceDB files while NAT has them open.
                _stop_process(process)
        assert {call["input_type"] for call in _EmbeddingHandler.calls} == {"passage"}
        assert set(_EmbeddingHandler.authorization_headers) == {None}
        assert "NemoRetrieverLocalLockError" not in log_path.read_text(encoding="utf-8")
    finally:
        _stop_process(process)
        embedding_server.shutdown()
        embedding_server.server_close()
        embedding_thread.join(timeout=10)
