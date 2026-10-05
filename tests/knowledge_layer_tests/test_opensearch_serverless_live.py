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

"""Live Amazon OpenSearch Serverless integration tests.

These tests are opt-in because they create and delete real AOSS indexes. They
force SigV4 service `aoss` and use deterministic local embeddings, so they
validate the OpenSearch Serverless data plane without requiring NVIDIA_API_KEY.

Run with exported env vars or same-line shell assignments, for example:
    DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_LIVE_TESTS=1 OPENSEARCH_URL=... AWS_REGION=... uv run python -m pytest ...
"""

import asyncio
import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from botocore.exceptions import BotoCoreError
from botocore.exceptions import NoCredentialsError
from knowledge_layer.opensearch.adapter import OpenSearchIngestor
from knowledge_layer.opensearch.adapter import OpenSearchRetriever

from deep_researcher_agent.knowledge.schema import FileStatus
from deep_researcher_agent.knowledge.schema import JobState

pytestmark = [
    pytest.mark.aws,
    pytest.mark.integration,
    pytest.mark.opensearch_serverless,
    pytest.mark.skipif(
        os.environ.get("DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_LIVE_TESTS", "").lower()
        not in {"1", "true", "yes", "on"},
        reason=(
            "Set and export DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_LIVE_TESTS=1, "
            "or pass it as a same-line env assignment, "
            "to run live Amazon OpenSearch Serverless tests."
        ),
    ),
]


def _env_bool(name: str, default: bool = False) -> bool:
    """env bool."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """env int."""
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _region_from_aoss_endpoint(endpoint: str) -> str | None:
    """region from aoss endpoint."""
    match = re.search(r"\.([a-z]{2}-[a-z]+-\d)\.aoss\.amazonaws\.com/?$", endpoint)
    return match.group(1) if match else None


def _serverless_config() -> dict[str, Any]:
    """serverless config."""
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("opensearchpy")

    endpoint = os.environ.get("OPENSEARCH_URL") or os.environ.get("AOSS_ENDPOINT")
    if not endpoint:
        pytest.fail("Amazon OpenSearch Serverless live tests require OPENSEARCH_URL or AOSS_ENDPOINT.")

    if ".aoss.amazonaws.com" not in endpoint and not _env_bool(
        "DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_ALLOW_CUSTOM_ENDPOINT"
    ):
        pytest.fail(
            "Amazon OpenSearch Serverless live tests expect an .aoss.amazonaws.com endpoint. "
            "Set DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_ALLOW_CUSTOM_ENDPOINT=1 for a custom/private endpoint."
        )

    aws_region = (
        os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or _region_from_aoss_endpoint(endpoint)
    )
    if not aws_region:
        pytest.fail("Set AWS_REGION/AWS_DEFAULT_REGION, or use a standard regional .aoss.amazonaws.com endpoint.")

    session = boto3.Session(region_name=aws_region)
    try:
        credentials = session.get_credentials()
        if credentials is None:
            raise NoCredentialsError()
        credentials.get_frozen_credentials()
    except (BotoCoreError, RuntimeError, NoCredentialsError) as e:
        profile = os.environ.get("AWS_PROFILE")
        profile_hint = f" --profile {profile}" if profile else ""
        pytest.fail(
            "Amazon OpenSearch Serverless live tests require valid, unexpired AWS credentials. "
            f"Credential refresh failed: {e}. "
            f"Run `aws sso login{profile_hint}` for SSO credentials, or export fresh "
            "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN values."
        )

    prefix = os.environ.get("DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_INDEX_PREFIX", "deep-researcher-aoss-live")
    run_suffix = uuid.uuid4().hex[:8]

    return {
        "endpoint": endpoint,
        "auth_type": "sigv4",
        "aws_region": aws_region,
        "aws_service": "aoss",
        "verify_certs": _env_bool("OPENSEARCH_VERIFY_CERTS", True),
        "ca_certs": os.environ.get("OPENSEARCH_CA_CERTS"),
        "index_prefix": f"{prefix}-{run_suffix}",
        "embedding_dim": 4,
        "engine": os.environ.get("OPENSEARCH_SERVERLESS_ENGINE", "faiss"),
        "space_type": os.environ.get("OPENSEARCH_SERVERLESS_SPACE_TYPE", "l2"),
        "chunk_size": 32,
        "chunk_overlap": 0,
        "timeout": int(os.environ.get("OPENSEARCH_TIMEOUT", "120")),
        "bulk_batch_size": 10,
    }


def _test_embedding(text: str) -> list[float]:
    """test embedding."""
    text = text.lower()
    if "aurora" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "bedrock" in text:
        return [0.0, 1.0, 0.0, 0.0]
    return [0.5, 0.5, 0.0, 0.0]


def _patch_embeddings(adapter: OpenSearchIngestor | OpenSearchRetriever) -> None:
    """patch embeddings."""
    adapter._embed_texts = lambda texts: [_test_embedding(text) for text in texts]


def _wait_for_job(ingestor: OpenSearchIngestor, job_id: str, timeout_seconds: int = 90):
    """wait for job."""
    deadline = time.time() + timeout_seconds
    job = ingestor.get_job_status(job_id)
    while time.time() < deadline and not job.is_terminal:
        time.sleep(0.5)
        job = ingestor.get_job_status(job_id)
    return job


def _visible_doc_count_with_retry(
    retriever: OpenSearchRetriever,
    collection_name: str,
    expected_count: int,
    timeout_seconds: int | None = None,
) -> int:
    """visible doc count with retry."""
    timeout_seconds = timeout_seconds or _env_int("DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_VISIBILITY_TIMEOUT", 180)
    index_name = retriever._index_name_for_collection(collection_name)
    client = retriever._get_client()
    deadline = time.time() + timeout_seconds
    count = 0

    while time.time() < deadline:
        try:
            response = client.search(
                index=index_name,
                body={
                    "size": 0,
                    "query": {"match_all": {}},
                },
                request_timeout=retriever.timeout,
            )
            total = response.get("hits", {}).get("total", 0)
            count = int(total.get("value", 0) if isinstance(total, dict) else total)
            if count >= expected_count:
                return count
        except Exception:
            count = 0
        time.sleep(2.0)

    return count


def _retrieve_with_retry(
    retriever: OpenSearchRetriever,
    query: str,
    collection_name: str,
    top_k: int = 3,
    filters: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
):
    """retrieve with retry."""
    timeout_seconds = timeout_seconds or _env_int("DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_RETRIEVAL_TIMEOUT", 180)
    deadline = time.time() + timeout_seconds
    result = None
    while time.time() < deadline:
        result = asyncio.run(retriever.retrieve(query, collection_name, top_k=top_k, filters=filters))
        if result.success and result.chunks:
            return result
        time.sleep(1.0)
    return result


def _list_files_with_retry(
    ingestor: OpenSearchIngestor,
    collection_name: str,
    expected_names: set[str],
    timeout_seconds: int | None = None,
):
    """list files with retry."""
    timeout_seconds = timeout_seconds or _env_int("DEEP_RESEARCHER_OPENSEARCH_SERVERLESS_VISIBILITY_TIMEOUT", 180)
    deadline = time.time() + timeout_seconds
    files = []
    while time.time() < deadline:
        files = ingestor.list_files(collection_name)
        if {file.file_name for file in files} == expected_names:
            return files
        time.sleep(1.0)
    return files


def _file_statuses_with_retry(
    ingestor: OpenSearchIngestor,
    collection_name: str,
    file_ids: dict[str, str],
    timeout_seconds: int = 45,
):
    """file statuses with retry."""
    deadline = time.time() + timeout_seconds
    statuses = {}
    while time.time() < deadline:
        statuses = {
            file_name: ingestor.get_file_status(file_id, collection_name) for file_name, file_id in file_ids.items()
        }
        if all(status is not None and status.status == FileStatus.SUCCESS for status in statuses.values()):
            return statuses
        time.sleep(1.0)
    return statuses


@pytest.fixture
def serverless_backend() -> tuple[OpenSearchIngestor, OpenSearchRetriever, Callable[[str], None]]:
    """Serverless backend."""
    config = _serverless_config()
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


def test_aoss_sigv4_health_and_collection_lifecycle(serverless_backend):
    """Test that aoss sigv4 health and collection lifecycle."""
    ingestor, _, track_collection = serverless_backend
    collection_name = f"aoss-lifecycle-{uuid.uuid4().hex[:8]}"
    track_collection(collection_name)

    assert ingestor.auth_type == "sigv4"
    assert ingestor.aws_service == "aoss"
    assert asyncio.run(ingestor.health_check())

    created = ingestor.create_collection(collection_name, description="AOSS lifecycle test")

    assert created.name == collection_name
    assert created.backend == "opensearch"
    assert created.metadata["index_name"].startswith(ingestor.index_prefix)

    fetched = ingestor.get_collection(collection_name)
    assert fetched is not None
    assert fetched.name == collection_name

    index_meta = ingestor._get_index_meta(ingestor._index_name_for_collection(collection_name))
    assert index_meta["backend"] == "opensearch"
    assert index_meta["collection_name"] == collection_name

    assert ingestor.delete_collection(collection_name)
    assert ingestor.get_collection(collection_name) is None


def test_aoss_vector_ingest_retrieve_filter_and_delete(tmp_path, serverless_backend):
    """Test that aoss vector ingest retrieve filter and delete."""
    ingestor, retriever, track_collection = serverless_backend
    collection_name = f"aoss-ingest-{uuid.uuid4().hex[:8]}"
    track_collection(collection_name)

    aurora_file = tmp_path / "aurora.txt"
    bedrock_file = tmp_path / "bedrock.txt"
    aurora_file.write_text("aurora aurora vector search document for serverless retrieval", encoding="utf-8")
    bedrock_file.write_text("bedrock bedrock vector search document for serverless retrieval", encoding="utf-8")

    ingestor.create_collection(collection_name, description="AOSS vector ingestion test")
    job_id = ingestor.submit_job(
        [str(aurora_file), str(bedrock_file)],
        collection_name,
        config={
            "original_filenames": ["aurora.txt", "bedrock.txt"],
            "metadata": {"suite": "aoss-live", "provider": "aws"},
        },
    )

    job = _wait_for_job(ingestor, job_id)

    assert job.status == JobState.COMPLETED, job.model_dump()
    assert {detail.status for detail in job.file_details} == {FileStatus.SUCCESS}

    file_ids = {detail.file_name: detail.file_id for detail in job.file_details}
    statuses = _file_statuses_with_retry(ingestor, collection_name, file_ids)
    assert set(statuses) == {"aurora.txt", "bedrock.txt"}
    assert all(status is not None and status.status == FileStatus.SUCCESS for status in statuses.values())
    assert all(status.metadata["suite"] == "aoss-live" for status in statuses.values() if status is not None)

    visible_count = _visible_doc_count_with_retry(retriever, collection_name, expected_count=2)
    assert visible_count >= 2

    result = _retrieve_with_retry(retriever, "aurora semantic search", collection_name)

    assert result is not None
    assert result.success, result.error_message
    assert result.chunks
    assert result.chunks[0].file_name == "aurora.txt"
    assert result.chunks[0].metadata["provider"] == "aws"

    filtered = _retrieve_with_retry(
        retriever,
        "bedrock semantic search",
        collection_name,
        top_k=2,
        filters={"file_name": "aurora.txt"},
    )

    assert filtered is not None
    assert filtered.success
    assert filtered.chunks
    assert {chunk.file_name for chunk in filtered.chunks} == {"aurora.txt"}

    assert ingestor.delete_file(file_ids["aurora.txt"], collection_name)

    remaining_files = _list_files_with_retry(ingestor, collection_name, {"bedrock.txt"})
    remaining_names = {file.file_name for file in remaining_files}
    assert remaining_names == {"bedrock.txt"}
