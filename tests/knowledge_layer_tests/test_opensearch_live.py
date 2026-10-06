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

"""Live OpenSearch integration tests.

These tests are opt-in because they create and delete real OpenSearch indexes.
They use deterministic local embeddings so they only require OpenSearch access,
not NVIDIA_API_KEY or external embedding calls.
"""

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from knowledge_layer.opensearch.adapter import OpenSearchIngestor
from knowledge_layer.opensearch.adapter import OpenSearchRetriever

from deep_researcher_agent.knowledge.schema import FileStatus
from deep_researcher_agent.knowledge.schema import JobState


def _env_bool(name: str, default: bool = False) -> bool:
    """env bool."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _env_bool("DEEP_RESEARCHER_OPENSEARCH_LIVE_TESTS"),
        reason="Set DEEP_RESEARCHER_OPENSEARCH_LIVE_TESTS=1 to run live OpenSearch integration tests.",
    ),
]


def _live_config() -> dict[str, Any]:
    """live config."""
    pytest.importorskip("opensearchpy")

    auth_type = os.environ.get("OPENSEARCH_AUTH_TYPE", "none").lower()
    username = os.environ.get("OPENSEARCH_USERNAME")
    password = os.environ.get("OPENSEARCH_PASSWORD")

    if auth_type == "basic" and (not username or not password):
        pytest.fail("OPENSEARCH_AUTH_TYPE=basic requires OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD.")
    if auth_type == "sigv4":
        pytest.importorskip("boto3")

    prefix = os.environ.get("DEEP_RESEARCHER_OPENSEARCH_LIVE_INDEX_PREFIX", "deep-researcher-live")
    run_suffix = uuid.uuid4().hex[:8]

    return {
        "endpoint": os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
        "auth_type": auth_type,
        "username": username,
        "password": password,
        "verify_certs": _env_bool("OPENSEARCH_VERIFY_CERTS", True),
        "ca_certs": os.environ.get("OPENSEARCH_CA_CERTS"),
        "aws_region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "aws_service": os.environ.get("OPENSEARCH_AWS_SERVICE", "aoss"),
        "index_prefix": f"{prefix}-{run_suffix}",
        "embedding_dim": 4,
        "engine": os.environ.get("OPENSEARCH_ENGINE", "faiss"),
        "space_type": os.environ.get("OPENSEARCH_SPACE_TYPE", "cosinesimil"),
        "chunk_size": 32,
        "chunk_overlap": 0,
        "timeout": int(os.environ.get("OPENSEARCH_TIMEOUT", "120")),
        "bulk_batch_size": 10,
    }


def _test_embedding(text: str) -> list[float]:
    """test embedding."""
    text = text.lower()
    if "alpha" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "beta" in text:
        return [0.0, 1.0, 0.0, 0.0]
    return [0.5, 0.5, 0.0, 0.0]


def _patch_embeddings(adapter: OpenSearchIngestor | OpenSearchRetriever) -> None:
    """patch embeddings."""
    adapter._embed_texts = lambda texts: [_test_embedding(text) for text in texts]


def _wait_for_job(ingestor: OpenSearchIngestor, job_id: str, timeout_seconds: int = 60):
    """wait for job."""
    deadline = time.time() + timeout_seconds
    job = ingestor.get_job_status(job_id)
    while time.time() < deadline and not job.is_terminal:
        time.sleep(0.25)
        job = ingestor.get_job_status(job_id)
    return job


def _retrieve_with_retry(
    retriever: OpenSearchRetriever,
    query: str,
    collection_name: str,
    top_k: int = 3,
    timeout_seconds: int = 30,
):
    """retrieve with retry."""
    deadline = time.time() + timeout_seconds
    result = None
    while time.time() < deadline:
        result = asyncio.run(retriever.retrieve(query, collection_name, top_k=top_k))
        if result.success and result.chunks:
            return result
        time.sleep(0.5)
    return result


@pytest.fixture
def live_backend() -> tuple[OpenSearchIngestor, OpenSearchRetriever, Callable[[str], None]]:
    """Live backend."""
    config = _live_config()
    ingestor = OpenSearchIngestor(config)
    retriever = OpenSearchRetriever(config)
    _patch_embeddings(ingestor)
    _patch_embeddings(retriever)

    created_collections: list[str] = []

    def track_collection(collection_name: str) -> None:
        """Track collection."""
        created_collections.append(collection_name)

    yield ingestor, retriever, track_collection

    for collection_name in reversed(created_collections):
        ingestor.delete_collection(collection_name)


def test_live_opensearch_collection_lifecycle(live_backend):
    """Test that live opensearch collection lifecycle."""
    ingestor, _, track_collection = live_backend
    collection_name = f"live-lifecycle-{uuid.uuid4().hex[:8]}"
    track_collection(collection_name)

    assert asyncio.run(ingestor.health_check())

    created = ingestor.create_collection(collection_name, description="Live OpenSearch lifecycle test")

    assert created.name == collection_name
    assert created.backend == "opensearch"
    assert created.metadata["index_name"].startswith(ingestor.index_prefix)

    fetched = ingestor.get_collection(collection_name)
    assert fetched is not None
    assert fetched.name == collection_name

    listed_names = {collection.name for collection in ingestor.list_collections()}
    assert collection_name in listed_names

    assert ingestor.delete_collection(collection_name)
    assert ingestor.get_collection(collection_name) is None


def test_live_opensearch_ingest_retrieve_and_delete(tmp_path, live_backend):
    """Test that live opensearch ingest retrieve and delete."""
    ingestor, retriever, track_collection = live_backend
    collection_name = f"live-ingest-{uuid.uuid4().hex[:8]}"
    track_collection(collection_name)

    alpha_file = tmp_path / "alpha.txt"
    beta_file = tmp_path / "beta.txt"
    alpha_file.write_text("alpha alpha alpha revenue roadmap vector document", encoding="utf-8")
    beta_file.write_text("beta beta beta security operations vector document", encoding="utf-8")

    ingestor.create_collection(collection_name, description="Live OpenSearch ingestion test")
    job_id = ingestor.submit_job(
        [str(alpha_file), str(beta_file)],
        collection_name,
        config={
            "original_filenames": ["alpha.txt", "beta.txt"],
            "metadata": {"suite": "opensearch-live"},
        },
    )

    job = _wait_for_job(ingestor, job_id)

    assert job.status == JobState.COMPLETED, job.model_dump()
    assert {detail.status for detail in job.file_details} == {FileStatus.SUCCESS}

    files = ingestor.list_files(collection_name)
    names = {file.file_name for file in files}
    assert names == {"alpha.txt", "beta.txt"}
    assert all(file.metadata["suite"] == "opensearch-live" for file in files)

    result = _retrieve_with_retry(retriever, "alpha roadmap", collection_name)

    assert result is not None
    assert result.success, result.error_message
    assert result.chunks
    assert result.chunks[0].file_name == "alpha.txt"
    assert "alpha" in result.chunks[0].content.lower()
    assert result.chunks[0].metadata["suite"] == "opensearch-live"

    file_ids = {detail.file_name: detail.file_id for detail in job.file_details}
    assert ingestor.delete_file(file_ids["alpha.txt"], collection_name)

    remaining_names = {file.file_name for file in ingestor.list_files(collection_name)}
    assert remaining_names == {"beta.txt"}
