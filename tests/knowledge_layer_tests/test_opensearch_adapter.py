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

"""Tests for the OpenSearch Knowledge Layer adapter."""

import asyncio
import logging
import threading
import time
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from knowledge_layer.opensearch import adapter as opensearch_adapter
from knowledge_layer.opensearch.adapter import OpenSearchIngestor
from knowledge_layer.opensearch.adapter import OpenSearchRetriever

from deep_researcher_agent.knowledge.schema import Chunk
from deep_researcher_agent.knowledge.schema import ContentType
from deep_researcher_agent.knowledge.schema import FileInfo
from deep_researcher_agent.knowledge.schema import FileStatus
from deep_researcher_agent.knowledge.schema import JobState


class FakeOpenSearchIndices:
    def __init__(self, client: "FakeOpenSearchClient"):
        """Initialise FakeOpenSearchIndices."""
        self._client = client

    def exists(self, index: str) -> bool:
        """Return True if the index exists in the fake store."""
        return index in self._client.indexes

    def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Create a fake index and return an acknowledged response."""
        with self._client.lock:
            self._client.indexes[index] = body
            self._client.docs.setdefault(index, {})
        return {"acknowledged": True}

    def delete(self, index: str) -> dict[str, Any]:
        """Delete the fake index and its documents."""
        with self._client.lock:
            self._client.indexes.pop(index, None)
            self._client.docs.pop(index, None)
        return {"acknowledged": True}

    def get(self, index: str) -> dict[str, Any]:
        """Return mapping metadata for one or all matching fake indexes."""
        with self._client.lock:
            if "*" in index:
                prefix = index.rstrip("*")
                return {
                    name: {"mappings": body.get("mappings", {})}
                    for name, body in self._client.indexes.items()
                    if name.startswith(prefix)
                }
            if index not in self._client.indexes:
                raise KeyError(index)
            return {index: {"mappings": self._client.indexes[index].get("mappings", {})}}

    def put_mapping(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Update the _meta section of an existing fake index mapping."""
        with self._client.lock:
            mappings = self._client.indexes[index].setdefault("mappings", {})
            if "_meta" in body:
                mappings["_meta"] = body["_meta"]
        return {"acknowledged": True}


class FakeOpenSearchClient:
    def __init__(self):
        """Initialise FakeOpenSearchClient."""
        self.lock = threading.RLock()
        self.indexes: dict[str, dict[str, Any]] = {}
        self.docs: dict[str, dict[str, dict[str, Any]]] = {}
        self.indices = FakeOpenSearchIndices(self)

    def count(self, index: str) -> dict[str, int]:
        """Return the document count for the given fake index."""
        with self.lock:
            return {"count": len(self.docs.get(index, {}))}

    def bulk(self, body: list[dict[str, Any]], refresh: bool = True, request_timeout: int | None = None):
        """Record a bulk indexing request in the fake store."""
        del refresh, request_timeout
        with self.lock:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                index = action["index"]["_index"]
                doc_id = action["index"]["_id"]
                self.docs.setdefault(index, {})[doc_id] = doc
        return {"errors": False}

    def search(self, index: str, body: dict[str, Any], request_timeout: int | None = None) -> dict[str, Any]:
        """Execute a fake search, supporting composite aggregations."""
        del request_timeout
        with self.lock:
            docs = self.docs.get(index, {})
            aggs = (body or {}).get("aggs") or {}
            if "by_file" in aggs:
                # List-files composite aggregation: group by (file_id, file_name), paginate via after_key.
                by_file = aggs["by_file"]
                composite = by_file.get("composite") or {}
                # Extract ordered source aliases and their index fields.
                sources = composite.get("sources", [])
                source_aliases = [list(s.keys())[0] for s in sources]
                source_fields = {alias: list(src.values())[0]["terms"]["field"] for src in sources for alias in src}
                page_size = int(composite.get("size", 10))
                after_key = composite.get("after") or {}
                source_filter = by_file["aggs"]["doc"]["top_hits"].get("_source")
                grouped: dict[tuple, list[dict[str, Any]]] = {}
                for doc_id, doc in docs.items():
                    key_tuple = tuple(doc.get(source_fields[a], "unknown") for a in source_aliases)
                    grouped.setdefault(key_tuple, []).append({"_id": doc_id, "_source": doc})
                # composite iterates keys in deterministic (sorted) order.
                ordered_keys = sorted(grouped)
                if after_key:
                    after_tuple = tuple(after_key.get(a, "") for a in source_aliases)
                    ordered_keys = [k for k in ordered_keys if k > after_tuple]
                page_keys = ordered_keys[:page_size]
                buckets = []
                for key_tuple in page_keys:
                    group_docs = grouped[key_tuple]
                    ct_counts: dict[str, int] = {}
                    for hit in group_docs:
                        ct = hit["_source"].get("content_type")
                        if ct:
                            ct_counts[ct] = ct_counts.get(ct, 0) + 1
                    top_source = self._filter_source(group_docs[0]["_source"], source_filter)
                    key_dict = {alias: key_tuple[i] for i, alias in enumerate(source_aliases)}
                    buckets.append(
                        {
                            "key": key_dict,
                            "doc_count": len(group_docs),
                            "doc": {"hits": {"hits": [{"_source": top_source}]}},
                            "content_types": {"buckets": [{"key": k, "doc_count": v} for k, v in ct_counts.items()]},
                        }
                    )
                agg_result: dict[str, Any] = {"buckets": buckets}
                # Emit after_key only when more pages remain — mirrors real OpenSearch.
                if len(page_keys) == page_size and len(ordered_keys) > page_size:
                    last = page_keys[-1]
                    agg_result["after_key"] = {alias: last[i] for i, alias in enumerate(source_aliases)}
                return {"hits": {"hits": []}, "aggregations": {"by_file": agg_result}}
            hits = []
            for doc_id, source_doc in docs.items():
                source = self._filter_source(source_doc, body.get("_source"))
                hits.append({"_id": doc_id, "_index": index, "_score": 0.87, "_source": source})
            return {"hits": {"hits": hits[: body.get("size", 10)]}}

    def delete_by_query(
        self,
        index: str,
        body: dict[str, Any],
        refresh: bool = True,
        conflicts: str = "proceed",
    ) -> dict[str, int]:
        """Delete documents matching the query from the fake store."""
        del refresh, conflicts
        bool_query = body["query"]["bool"]
        match_terms = bool_query.get("filter") or bool_query.get("should") or []
        deleted = 0
        with self.lock:
            for doc_id, doc in list(self.docs.get(index, {}).items()):
                if any(self._matches_term(doc, term.get("term", {})) for term in match_terms):
                    self.docs[index].pop(doc_id, None)
                    deleted += 1
        return {"deleted": deleted}

    def ping(self) -> bool:
        """Return True to simulate a reachable cluster."""
        return True

    def _filter_source(self, source: dict[str, Any], source_filter: Any) -> dict[str, Any]:
        """Apply an _source include/exclude filter to a document."""
        if isinstance(source_filter, list):
            return {key: source[key] for key in source_filter if key in source}
        if isinstance(source_filter, dict):
            excluded = set(source_filter.get("excludes", []))
            return {key: value for key, value in source.items() if key not in excluded}
        return dict(source)

    def _matches_term(self, doc: dict[str, Any], term: dict[str, Any]) -> bool:
        """Return True if any term in the query matches the document."""
        for key, value in term.items():
            if doc.get(key) == value:
                return True
        return False


class FakeAossTransport:
    def __init__(self):
        """Initialise FakeAossTransport."""
        self.requests: list[tuple[str, str]] = []

    def perform_request(self, method: str, path: str):
        """Record a transport-level HTTP request."""
        self.requests.append((method, path))
        return ""


class FakeAossClient:
    def __init__(self):
        """Initialise FakeAossClient."""
        self.bulk_body: list[dict[str, Any]] | None = None
        self.bulk_bodies: list[list[dict[str, Any]]] = []
        self.bulk_refresh: bool | str | None = None
        self.search_hits: list[dict[str, Any]] = []
        self.search_responses: list[list[dict[str, Any]]] = []
        self.search_calls = 0
        self.search_bodies: list[dict[str, Any]] = []
        self.transport = FakeAossTransport()

    def ping(self) -> bool:
        """Return True to simulate a reachable cluster."""
        return False

    def bulk(self, body: list[dict[str, Any]], refresh: bool = True, request_timeout: int | None = None):
        """Record a bulk indexing request in the fake store."""
        del request_timeout
        self.bulk_body = body
        self.bulk_bodies.append(body)
        self.bulk_refresh = refresh
        return {"errors": False}

    def search(self, index: str, body: dict[str, Any], request_timeout: int | None = None) -> dict[str, Any]:
        """Execute a fake search, supporting composite aggregations."""
        del index, request_timeout
        self.search_calls += 1
        self.search_bodies.append(body)
        if self.search_responses:
            return {"hits": {"hits": self.search_responses.pop(0)}}
        hits = self.search_hits
        self.search_hits = []
        return {"hits": {"hits": hits}}


class FakeFuture:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        """Initialise FakeFuture."""
        self._result = result
        self._error = error

    def result(self):
        """Return the stored future result or raise the stored error."""
        if self._error:
            raise self._error
        return self._result


class FakeDaskClient:
    def __init__(self, submit_raises: Exception | None = None):
        """Initialise FakeDaskClient."""
        self.submissions: list[dict[str, Any]] = []
        self.closed = False
        self._submit_raises = submit_raises

    def close(self) -> None:
        """Mark the fake Dask client as closed."""
        self.closed = True

    def submit(self, fn, config, payloads, collection_name, **kwargs):
        """Record a task submission and return a FakeFuture."""
        if self._submit_raises is not None:
            raise self._submit_raises
        self.submissions.append(
            {
                "fn": fn,
                "config": config,
                "payloads": payloads,
                "collection_name": collection_name,
                "kwargs": kwargs,
            }
        )
        return FakeFuture(
            {
                "status": "completed",
                "files": [
                    {
                        "file_id": payload["file_id"],
                        "file_name": payload["file_name"],
                        "status": "success",
                        "chunks_created": 1,
                    }
                    for payload in payloads
                ],
                "total_chunks": len(payloads),
                "index_name": "deep-researcher-dask-docs",
                "embedding_model": "nvidia/test-embed",
            }
        )


def test_opensearch_backend_registers_with_factory():
    """Test that opensearch backend registers with factory."""
    from deep_researcher_agent.knowledge import factory
    from deep_researcher_agent.knowledge.factory import get_ingestor
    from deep_researcher_agent.knowledge.factory import get_retriever

    factory._INGESTOR_INSTANCES.pop("opensearch", None)
    factory._RETRIEVER_INSTANCES.pop("opensearch", None)

    ingestor = get_ingestor("opensearch", {"endpoint": "localhost:9200"})
    retriever = get_retriever("opensearch", {"endpoint": "localhost:9200"})

    assert ingestor.backend_name == "opensearch"
    assert retriever.backend_name == "opensearch"
    assert ingestor.endpoint == "http://localhost:9200"
    assert retriever.endpoint == "http://localhost:9200"


def test_aoss_health_check_falls_back_to_cat_indices():
    """Test that aoss health check falls back to cat indices."""
    ingestor = OpenSearchIngestor({"auth_type": "sigv4", "aws_service": "aoss"})
    fake_client = FakeAossClient()
    ingestor._client = fake_client

    assert asyncio.run(ingestor.health_check())
    assert fake_client.transport.requests == [("GET", "/_cat/indices")]


def test_aoss_bulk_index_omits_document_ids_and_explicit_refresh():
    """Test that aoss bulk index omits document ids and explicit refresh."""
    ingestor = OpenSearchIngestor({"auth_type": "sigv4", "aws_service": "aoss", "embedding_dim": 4})
    fake_client = FakeAossClient()
    ingestor._client = fake_client

    ingestor._bulk_index_documents(
        "deep-researcher-aoss-test",
        [
            {
                "chunk_id": "chunk-1",
                "file_id": "file-1",
                "file_name": "doc.txt",
                "content": "hello",
                "embedding": [0.1, 0.2, 0.3, 0.4],
            }
        ],
    )

    assert fake_client.bulk_body is not None
    assert fake_client.bulk_body[0] == {"index": {"_index": "deep-researcher-aoss-test"}}
    assert fake_client.bulk_refresh is False


def test_bulk_index_rejects_mismatched_embedding_dimension_before_request():
    """Ingestion vectors must match the mapping before any bulk request is sent."""
    ingestor = OpenSearchIngestor({"auth_type": "sigv4", "aws_service": "aoss", "embedding_dim": 4})
    fake_client = FakeAossClient()
    ingestor._client = fake_client

    with pytest.raises(RuntimeError, match="ingestion embedding 0 has dimension 3"):
        ingestor._bulk_index_documents(
            "deep-researcher-aoss-test",
            [{"chunk_id": "chunk-1", "embedding": [0.1, 0.2, 0.3]}],
        )

    assert fake_client.bulk_body is None


def test_aoss_delete_searches_then_bulk_deletes_generated_ids():
    """Test that aoss delete searches then bulk deletes generated ids."""
    ingestor = OpenSearchIngestor({"auth_type": "sigv4", "aws_service": "aoss", "bulk_batch_size": 2})
    fake_client = FakeAossClient()
    fake_client.search_hits = [{"_id": "generated-1"}, {"_id": "generated-2"}]
    ingestor._client = fake_client

    deleted = ingestor._delete_file_documents_for_aoss(
        "deep-researcher-aoss-test",
        {
            "query": {
                "bool": {
                    "should": [{"term": {"file_id": "file-1"}}],
                    "minimum_should_match": 1,
                }
            }
        },
    )

    assert deleted == 2
    assert fake_client.bulk_body == [
        {"delete": {"_index": "deep-researcher-aoss-test", "_id": "generated-1"}},
        {"delete": {"_index": "deep-researcher-aoss-test", "_id": "generated-2"}},
    ]
    assert fake_client.bulk_refresh is False


def test_aoss_delete_enumerates_all_pages_before_deleting():
    """AOSS deletion must enumerate every matching page before deleting, not just the first."""
    ingestor = OpenSearchIngestor(
        {
            "auth_type": "sigv4",
            "aws_service": "aoss",
            "bulk_batch_size": 2,
            "aoss_delete_backoff_seconds": 0,
        }
    )
    fake_client = FakeAossClient()
    # Full first page (2 hits) signals more pages; short second page (1 hit) ends enumeration.
    fake_client.search_responses = [
        [{"_id": "generated-1"}, {"_id": "generated-2"}],
        [{"_id": "generated-3"}],
    ]
    ingestor._client = fake_client

    deleted = ingestor._delete_file_documents_for_aoss(
        "deep-researcher-aoss-test",
        {"query": {"bool": {"should": [{"term": {"file_id": "file-1"}}], "minimum_should_match": 1}}},
    )

    assert deleted == 3
    deleted_ids = [action["delete"]["_id"] for body in fake_client.bulk_bodies for action in body]
    assert deleted_ids == ["generated-1", "generated-2", "generated-3"]
    # Enumeration uses a stable sort and does not fetch document sources.
    assert fake_client.search_bodies[0]["sort"] == [{"chunk_id": "asc"}]
    assert fake_client.search_bodies[0]["_source"] is False


def test_aoss_delete_fails_rather_than_reporting_partial_success_when_enumeration_stalls():
    """A stale view that keeps returning full pages must fail, never report partial deletion."""
    ingestor = OpenSearchIngestor(
        {
            "auth_type": "sigv4",
            "aws_service": "aoss",
            "bulk_batch_size": 1,
            "aoss_delete_max_batches": 3,
            "aoss_delete_backoff_seconds": 0,
        }
    )
    fake_client = FakeAossClient()
    # Full pages that never shrink or empty within the batch cap -> enumeration cannot complete.
    fake_client.search_responses = [[{"_id": "generated-1"}] for _ in range(5)]
    ingestor._client = fake_client

    with pytest.raises(RuntimeError):
        ingestor._delete_file_documents_for_aoss(
            "deep-researcher-aoss-test",
            {"query": {"bool": {"should": [{"term": {"file_id": "file-1"}}], "minimum_should_match": 1}}},
        )

    # Nothing may be deleted when the matching set could not be fully enumerated.
    assert fake_client.bulk_body is None


def test_index_name_helpers_are_opensearch_safe():
    """Test that index name helpers are opensearch safe."""
    assert opensearch_adapter._sanitize_index_part("Tenant A / Session +1") == "tenant-a-session-1"
    assert opensearch_adapter._sanitize_index_part("+++Bad") == "bad"
    assert len(opensearch_adapter._trim_index_name("a" * 300)) <= 255


def test_index_name_mapping_is_collision_safe():
    """Logical names that sanitize to the same base must map to distinct physical indexes."""
    ingestor = OpenSearchIngestor({"start_ttl_cleanup": False})
    names = ["Tenant A", "tenant-a", "tenant/a", "TENANT+A"]

    indexes = [ingestor._index_name_for_collection(n) for n in names]

    assert len(set(indexes)) == len(names)
    # Mapping is stable: same logical name always yields the same physical index.
    assert ingestor._index_name_for_collection("Tenant A") == ingestor._index_name_for_collection("Tenant A")


def test_deleting_collision_collection_does_not_delete_other():
    """Deleting one collection must not delete a different collection whose name normalizes alike."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor({"start_ttl_cleanup": False})
    ingestor._client = fake_client

    ingestor.create_collection("Tenant A")
    ingestor.create_collection("tenant-a")
    a_index = ingestor._index_name_for_collection("Tenant A")
    b_index = ingestor._index_name_for_collection("tenant-a")
    assert a_index != b_index

    assert ingestor.delete_collection("Tenant A") is True
    assert a_index not in fake_client.indexes
    assert b_index in fake_client.indexes


def test_ensure_index_rejects_reuse_for_different_collection():
    """_ensure_index must refuse a physical index already owned by another logical collection."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor({"start_ttl_cleanup": False})
    ingestor._client = fake_client

    index_name = ingestor._index_name_for_collection("alpha")
    fake_client.indexes[index_name] = {
        "mappings": {
            "_meta": {
                "backend": "opensearch",
                "collection_name": "beta",
                "embedding_model": ingestor.embed_model_name,
                "embedding_dim": ingestor.embedding_dim,
            }
        }
    }

    with pytest.raises(RuntimeError):
        ingestor._ensure_index("alpha")


@pytest.mark.parametrize(
    ("embedding_model", "embedding_dim"),
    [(None, None), ("nvidia/old-embed", 2048), ("nvidia/test-embed", 1024)],
)
def test_ensure_index_rejects_unknown_or_mismatched_embedding_configuration(
    embedding_model: str | None,
    embedding_dim: int | None,
):
    """Existing vectors must never be mixed with a different embedding space."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor(
        {"embed_model": "nvidia/test-embed", "embedding_dim": 2048, "start_ttl_cleanup": False}
    )
    ingestor._client = fake_client
    index_name = ingestor._index_name_for_collection("docs")
    fake_client.indexes[index_name] = {
        "mappings": {
            "_meta": {
                "backend": "opensearch",
                "collection_name": "docs",
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
            }
        }
    }

    with pytest.raises(RuntimeError, match="Delete and re-ingest"):
        ingestor._ensure_index("docs")


def test_retrieval_rejects_mismatched_embedding_configuration_before_query_embedding():
    """Retrieval must fail closed before querying with vectors from another embedding space."""
    fake_client = FakeOpenSearchClient()
    retriever = OpenSearchRetriever({"embed_model": "nvidia/new-embed", "embedding_dim": 2048})
    retriever._client = fake_client
    index_name = retriever._index_name_for_collection("docs")
    fake_client.indexes[index_name] = {
        "mappings": {
            "_meta": {
                "backend": "opensearch",
                "collection_name": "docs",
                "embedding_model": "nvidia/old-embed",
                "embedding_dim": 2048,
            }
        }
    }

    def unexpected_embedding(_texts):
        raise AssertionError("query embedding must not run for an incompatible index")

    retriever._embed_texts = unexpected_embedding
    result = asyncio.run(retriever.retrieve("question", "docs"))

    assert not result.success
    assert "embedding model" in (result.error_message or "")


def test_retrieval_rejects_mismatched_query_dimension_before_search():
    """Query vectors must match the persisted mapping before an OpenSearch request."""
    fake_client = FakeOpenSearchClient()
    retriever = OpenSearchRetriever({"embed_model": "nvidia/test-embed", "embedding_dim": 4})
    retriever._client = fake_client
    index_name = retriever._index_name_for_collection("docs")
    fake_client.indexes[index_name] = retriever._index_mapping("docs")
    retriever._embed_texts = lambda _texts: [[0.1, 0.2, 0.3]]
    search_calls = 0

    def unexpected_search(*_args, **_kwargs):
        nonlocal search_calls
        search_calls += 1
        raise AssertionError("search must not run for a mismatched query vector")

    fake_client.search = unexpected_search

    result = asyncio.run(retriever.retrieve("question", "docs"))

    assert not result.success
    assert "query embedding 0 has dimension 3" in (result.error_message or "")
    assert search_calls == 0


def test_retrieval_rejects_mismatched_collection_owner_before_embedding_or_search():
    """Retrieval must fail before model or OpenSearch calls when an index has another owner."""
    fake_client = FakeOpenSearchClient()
    retriever = OpenSearchRetriever({"embed_model": "nvidia/test-embed", "embedding_dim": 4})
    retriever._client = fake_client
    index_name = retriever._index_name_for_collection("docs")
    fake_client.indexes[index_name] = retriever._index_mapping("another-collection")
    embedding_calls = 0
    search_calls = 0

    def unexpected_embedding(_texts):
        nonlocal embedding_calls
        embedding_calls += 1
        raise AssertionError("query embedding must not run for an index owned by another collection")

    def unexpected_search(*_args, **_kwargs):
        nonlocal search_calls
        search_calls += 1
        raise AssertionError("search must not run for an index owned by another collection")

    retriever._embed_texts = unexpected_embedding
    fake_client.search = unexpected_search

    result = asyncio.run(retriever.retrieve("question", "docs"))

    assert not result.success
    assert "belongs to collection 'another-collection'" in (result.error_message or "")
    assert embedding_calls == 0
    assert search_calls == 0


def test_session_collection_names_are_safe_dynamic_indexes():
    """Test that session collection names are safe dynamic indexes."""
    ingestor = OpenSearchIngestor({"index_prefix": "deep-researcher-prod", "start_ttl_cleanup": False})
    collection_name = "s_123E4567-E89B-12D3-A456-426614174000"

    index_name = ingestor._index_name_for_collection(collection_name)

    # Readable sanitized prefix is preserved; a stable disambiguator suffix keeps the mapping injective.
    assert index_name.startswith("deep-researcher-prod-s_123e4567-e89b-12d3-a456-426614174000-")
    assert index_name == ingestor._index_name_for_collection(collection_name)


def test_create_collection_rejects_reserved_metadata_keys():
    """Caller metadata must not be able to overwrite adapter-owned _meta fields."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor({"embedding_dim": 4, "start_ttl_cleanup": False})
    ingestor._client = fake_client

    ingestor.create_collection("docs")
    index_name = ingestor._index_name_for_collection("docs")
    original_meta = dict(fake_client.indexes[index_name]["mappings"]["_meta"])

    for reserved in ("backend", "collection_name", "embedding_model", "embedding_dim", "created_at", "updated_at"):
        with pytest.raises(ValueError):
            ingestor.create_collection("docs", metadata={reserved: "attacker"})

    # None of the rejected calls may have mutated the adapter-owned fields.
    assert fake_client.indexes[index_name]["mappings"]["_meta"] == original_meta


def test_create_collection_stores_non_reserved_metadata():
    """Non-reserved caller metadata is still persisted under _meta."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor({"embedding_dim": 4, "start_ttl_cleanup": False})
    ingestor._client = fake_client

    ingestor.create_collection("docs", metadata={"tenant": "acme", "team": "search"})
    index_name = ingestor._index_name_for_collection("docs")
    meta = fake_client.indexes[index_name]["mappings"]["_meta"]

    assert meta["tenant"] == "acme"
    assert meta["team"] == "search"
    assert meta["backend"] == "opensearch"


def test_index_mapping_keeps_metadata_strings_filterable():
    """Test that index mapping keeps metadata strings filterable."""
    ingestor = OpenSearchIngestor({"embedding_dim": 4, "start_ttl_cleanup": False})

    mapping = ingestor._index_mapping("docs")

    assert mapping["mappings"]["dynamic_templates"] == [
        {
            "metadata_strings": {
                "path_match": "metadata.*",
                "match_mapping_type": "string",
                "mapping": {"type": "keyword", "ignore_above": 1024},
            }
        }
    ]


def test_search_body_includes_knn_filter():
    """Test that search body includes knn filter."""
    retriever = OpenSearchRetriever({"embedding_dim": 4, "vector_field": "vec"})

    body = retriever._build_search_body([0.1, 0.2, 0.3, 0.4], 3, {"file_name": "report.pdf", "topic": "roadmap"})

    assert body["size"] == 3
    assert body["query"]["knn"]["vec"]["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert body["query"]["knn"]["vec"]["k"] == 3
    assert body["_source"]["excludes"] == ["vec"]
    assert body["query"]["knn"]["vec"]["filter"] == {
        "bool": {
            "filter": [
                {"term": {"file_name": "report.pdf"}},
                {"term": {"metadata.topic": "roadmap"}},
            ]
        }
    }


def test_normalize_maps_opensearch_hit_to_chunk():
    """Test that normalize maps opensearch hit to chunk."""
    retriever = OpenSearchRetriever({"text_field": "body"})
    chunk = retriever.normalize(
        {
            "_id": "doc-1",
            "_index": "deep-researcher-default",
            "_score": 0.91,
            "_source": {
                "body": "OpenSearch content",
                "file_name": "report.pdf",
                "page_number": 2,
                "content_type": "text",
                "metadata": {"section": "intro"},
            },
        }
    )

    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "doc-1"
    assert chunk.content == "OpenSearch content"
    assert chunk.score == 0.91
    assert chunk.content_type == ContentType.TEXT
    assert chunk.display_citation == "report.pdf, p.2"
    assert chunk.metadata["section"] == "intro"
    assert chunk.metadata["index"] == "deep-researcher-default"


def test_ingestion_and_retrieval_with_fake_client(tmp_path):
    """Test that ingestion and retrieval with fake client."""
    fake_client = FakeOpenSearchClient()
    test_file = tmp_path / "doc.txt"
    test_file.write_text(
        "OpenSearch stores document chunks as vectors for Deep Researcher Agent retrieval. "
        "The adapter creates one index per collection and preserves citations.",
        encoding="utf-8",
    )

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "localhost:9200",
            "embedding_dim": 4,
            "chunk_size": 6,
            "chunk_overlap": 1,
            "index_prefix": "deep-researcher-test",
        }
    )
    ingestor._client = fake_client
    ingestor._embed_texts = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    file_info = ingestor.upload_file(str(test_file), "collection_a", metadata={"tenant": "alpha"})
    deadline = time.time() + 5
    job = ingestor.get_job_status(file_info.metadata["job_id"])
    while time.time() < deadline and not job.is_terminal:
        time.sleep(0.05)
        job = ingestor.get_job_status(file_info.metadata["job_id"])

    assert job.status == JobState.COMPLETED

    status = ingestor.get_file_status(file_info.file_id, "collection_a")
    assert status is not None
    assert status.status == FileStatus.SUCCESS
    assert status.chunk_count > 0

    files = ingestor.list_files("collection_a")
    assert len(files) == 1
    assert files[0].file_name == "doc.txt"
    assert files[0].chunk_count == status.chunk_count
    assert files[0].metadata["tenant"] == "alpha"

    retriever = OpenSearchRetriever(
        {
            "endpoint": "localhost:9200",
            "embedding_dim": 4,
            "index_prefix": "deep-researcher-test",
        }
    )
    retriever._client = fake_client
    retriever._embed_texts = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    result = asyncio.run(retriever.retrieve("How are chunks stored?", "collection_a", top_k=2))

    assert result.success
    assert result.backend == "opensearch"
    assert result.chunks
    assert result.chunks[0].file_name == "doc.txt"
    assert result.chunks[0].display_citation == "doc.txt"
    assert result.chunks[0].metadata["tenant"] == "alpha"
    # source_path leaks internal filesystem paths (e.g. /tmp/tmpXXX.pdf in byte-upload
    # paths) into API responses and LLM context — must never appear in retrieved metadata.
    assert "source_path" not in result.chunks[0].metadata

    assert ingestor.delete_file(file_info.file_id, "collection_a")
    assert ingestor.list_files("collection_a") == []


def test_indexed_documents_omit_internal_source_path(tmp_path):
    """source_path leaks internal filesystem paths (such as /tmp/tmpXXX.pdf temp files
    used in Dask and byte-upload modes) into the OpenSearch index, and via normalize()
    into every Chunk returned to API consumers and LLM context windows. Indexed docs
    must never carry it in metadata."""
    fake_client = FakeOpenSearchClient()
    test_file = tmp_path / "leak-check.txt"
    test_file.write_text("Short payload to chunk.", encoding="utf-8")

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "localhost:9200",
            "embedding_dim": 4,
            "chunk_size": 8,
            "chunk_overlap": 1,
            "index_prefix": "deep-researcher-leak",
        }
    )
    ingestor._client = fake_client
    ingestor._embed_texts = lambda texts: [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    file_info = ingestor.upload_file(str(test_file), "collection_leak")
    deadline = time.time() + 5
    job = ingestor.get_job_status(file_info.metadata["job_id"])
    while time.time() < deadline and not job.is_terminal:
        time.sleep(0.05)
        job = ingestor.get_job_status(file_info.metadata["job_id"])
    assert job.status == JobState.COMPLETED

    # Inspect every indexed document directly — the fake stores them under docs[index].
    indexed_docs = [doc for index_docs in fake_client.docs.values() for doc in index_docs.values()]
    assert indexed_docs, "expected at least one chunk to be indexed"
    for doc in indexed_docs:
        metadata = doc.get("metadata") or {}
        assert "source_path" not in metadata, f"source_path leaked into indexed metadata: {metadata!r}"


def test_ttl_cleanup_deletes_only_expired_opensearch_session_indexes():
    """Test that ttl cleanup deletes only expired opensearch session indexes."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor({"index_prefix": "deep-researcher-ttl", "start_ttl_cleanup": False})
    ingestor._client = fake_client
    ingestor._ttl_hours = 24
    ingestor._cleanup_interval_seconds = 3600

    old_collection = "s_old_session"
    new_collection = "s_new_session"
    ingestor.create_collection(old_collection)
    ingestor.create_collection(new_collection)
    old_index = ingestor._index_name_for_collection(old_collection)
    new_index = ingestor._index_name_for_collection(new_collection)
    old_meta = fake_client.indexes[old_index]["mappings"]["_meta"]
    old_meta["updated_at"] = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    new_meta = fake_client.indexes[new_index]["mappings"]["_meta"]
    new_meta["updated_at"] = datetime.now(UTC).isoformat()
    fake_client.indexes["deep-researcher-ttl-unrelated"] = {
        "mappings": {
            "_meta": {
                "backend": "other",
                "collection_name": "unrelated",
                "updated_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
            }
        }
    }

    ingestor._cleanup_expired_collections()

    assert old_index not in fake_client.indexes
    assert new_index in fake_client.indexes
    assert "deep-researcher-ttl-unrelated" in fake_client.indexes


def test_dask_ingestion_submits_bytes_payload_and_updates_job(tmp_path):
    """Test that dask ingestion submits bytes payload and updates job."""
    fake_dask = FakeDaskClient()
    fake_client = FakeOpenSearchClient()
    test_file = tmp_path / "dask.txt"
    test_file.write_text("distributed opensearch ingestion", encoding="utf-8")
    ingestor = OpenSearchIngestor(
        {
            "ingestion_mode": "dask",
            "dask_scheduler_address": "tcp://scheduler:8786",
            "dask_file_transfer": "bytes",
            "embed_model": "nvidia/test-embed",
            "start_ttl_cleanup": False,
        }
    )
    ingestor._client = fake_client
    ingestor._create_dask_client = lambda: fake_dask

    job_id = ingestor.submit_job(
        [str(test_file)],
        "docs",
        config={"original_filenames": ["dask.txt"], "metadata": {"tenant": "aws"}},
    )

    deadline = time.time() + 5
    job = ingestor.get_job_status(job_id)
    while time.time() < deadline and not job.is_terminal:
        time.sleep(0.05)
        job = ingestor.get_job_status(job_id)

    assert job.status == JobState.COMPLETED
    assert job.metadata["ingestion_mode"] == "dask"
    assert fake_dask.submissions
    submission = fake_dask.submissions[0]
    assert submission["collection_name"] == "docs"
    assert submission["config"]["start_ttl_cleanup"] is False
    assert "summary_llm" not in submission["config"]
    assert submission["payloads"][0]["file_name"] == "dask.txt"
    assert submission["payloads"][0]["data"] == b"distributed opensearch ingestion"
    assert "path" not in submission["payloads"][0]
    status = ingestor.get_file_status(job.file_details[0].file_id, "docs")
    assert status is not None
    assert status.status == FileStatus.SUCCESS
    assert status.chunk_count == 1


def test_worker_config_excludes_credentials():
    """Credentials must never be serialized into the Dask worker config."""
    ingestor = OpenSearchIngestor({"auth_type": "none", "start_ttl_cleanup": False})

    worker_config = ingestor._worker_config(
        {
            "username": "admin",
            "password": "s3cret",  # pragma: allowlist secret
            "summary_llm": "x",
            "endpoint": "localhost:9200",
        }
    )

    assert "username" not in worker_config
    assert "password" not in worker_config
    assert "s3cret" not in worker_config.values()
    assert worker_config["endpoint"] == "localhost:9200"


def test_worker_config_warns_when_summary_requested_in_distributed_mode(caplog):
    """Distributed ingestion cannot produce summaries; a request that asked for them must warn, not silently drop."""
    ingestor = OpenSearchIngestor({"auth_type": "none", "start_ttl_cleanup": False})

    with caplog.at_level(logging.WARNING):
        worker_config = ingestor._worker_config({"generate_summary": True})

    assert worker_config["generate_summary"] is False
    assert any("does not generate document summaries" in r.message for r in caplog.records)


def test_worker_config_requires_env_credentials_for_basic_auth(monkeypatch):
    """Basic-auth distributed ingestion fails fast unless workers can resolve creds from their env."""
    monkeypatch.delenv("OPENSEARCH_USERNAME", raising=False)
    monkeypatch.delenv("OPENSEARCH_PASSWORD", raising=False)
    ingestor = OpenSearchIngestor(
        {
            "auth_type": "basic",
            "username": "admin",
            "password": "s3cret",  # pragma: allowlist secret
            "start_ttl_cleanup": False,
        }
    )

    with pytest.raises(RuntimeError, match="worker environment"):
        ingestor._worker_config({"username": "admin", "password": "s3cret"})  # pragma: allowlist secret


def test_delete_file_preserves_inflight_tracking_when_nothing_deleted():
    """If delete_file finds no documents to delete (file still UPLOADING or
    INGESTING, or already gone), in-memory tracking must be left intact so
    get_file_status keeps returning the live job state."""
    fake_client = FakeOpenSearchClient()
    ingestor = OpenSearchIngestor(
        {
            "endpoint": "localhost:9200",
            "embedding_dim": 4,
            "index_prefix": "deep-researcher-test",
        }
    )
    ingestor._client = fake_client

    # Create an empty index — exists() returns True but delete_by_query finds nothing.
    fake_client.indices.create(index="deep-researcher-test-c", body={})

    # Mark a file as INGESTING in-memory; not yet in the index (job still running).
    file_id = "inflight-1"
    ingestor._files[file_id] = FileInfo(
        file_id=file_id,
        file_name="in-flight.pdf",
        collection_name="c",
        status=FileStatus.INGESTING,
        file_size=100,
        uploaded_at=datetime.now(tz=UTC),
        metadata={"job_id": "job-1"},
    )

    result = ingestor.delete_file(file_id, "c")

    assert result is False, "delete_file must return False when no documents were deleted"
    assert file_id in ingestor._files, (
        "in-memory tracking must survive a no-op delete so an INGESTING job is not silently dropped"
    )
    status = ingestor.get_file_status(file_id, "c")
    assert status is not None, "get_file_status must still see the in-flight entry"
    assert status.status == FileStatus.INGESTING


def test_dask_client_closed_when_submit_raises(tmp_path):
    """If client.submit() raises (scheduler unreachable, serialisation error,
    key conflict), _start_dask_ingestion must close the Dask client before
    propagating so the scheduler TCP connection does not leak across retries."""
    test_file = tmp_path / "dask.txt"
    test_file.write_text("doc", encoding="utf-8")
    ingestor = OpenSearchIngestor(
        {
            "ingestion_mode": "dask",
            "dask_scheduler_address": "tcp://scheduler:8786",
            "dask_file_transfer": "bytes",
            "start_ttl_cleanup": False,
        }
    )
    fake_dask = FakeDaskClient(submit_raises=RuntimeError("serialisation error"))
    ingestor._create_dask_client = lambda: fake_dask

    job_id = ingestor.submit_job([str(test_file)], "docs")
    job = ingestor.get_job_status(job_id)

    assert job.status == JobState.FAILED
    assert "serialisation error" in job.error_message
    assert fake_dask.closed, "Dask client must be closed when submit() raises"


def test_dask_ingestion_submission_failure_marks_job_failed(tmp_path):
    """Test that dask ingestion submission failure marks job failed."""
    test_file = tmp_path / "dask.txt"
    test_file.write_text("distributed opensearch ingestion", encoding="utf-8")
    ingestor = OpenSearchIngestor(
        {
            "ingestion_mode": "dask",
            "dask_scheduler_address": "tcp://scheduler:8786",
            "start_ttl_cleanup": False,
        }
    )
    ingestor._create_dask_client = lambda: (_ for _ in ()).throw(RuntimeError("scheduler unavailable"))

    job_id = ingestor.submit_job([str(test_file)], "docs")
    job = ingestor.get_job_status(job_id)

    assert job.status == JobState.FAILED
    assert "scheduler unavailable" in job.error_message


def test_dask_worker_task_constructs_backend_in_worker(monkeypatch):
    """Test that dask worker task constructs backend in worker."""
    from knowledge_layer.opensearch.distributed import run_opensearch_ingestion_task

    captured: dict[str, Any] = {}

    class WorkerIngestor:
        def __init__(self, config: dict[str, Any]):
            """Initialise WorkerIngestor."""
            captured["config"] = config
            self.text_field = "content"
            self.vector_field = "embedding"
            self.embed_model_name = config["embed_model"]

        def _ensure_index(self, collection_name: str) -> str:
            """Stub that records the collection name passed to _ensure_index."""
            captured["collection_name"] = collection_name
            return "deep-researcher-docs"

        def _documents_for_file(
            self,
            file_path: str,
            file_id: str,
            file_name: str,
            file_metadata: dict[str, Any] | None = None,
        ):
            """Return fake documents for the given file path."""
            captured["worker_file_exists"] = Path(file_path).exists()
            return (
                [
                    {
                        "chunk_id": "chunk-1",
                        "file_id": file_id,
                        "file_name": file_name,
                        "content": Path(file_path).read_text(encoding="utf-8"),
                        "metadata": file_metadata or {},
                    }
                ],
                "summary",
            )

        def _embed_texts(self, texts: list[str]) -> list[list[float]]:
            """Return dummy embeddings (zero vectors) for the given texts."""
            captured["texts"] = texts
            return [[0.1, 0.2, 0.3, 0.4]]

        def _bulk_index_documents(self, index_name: str, documents: list[dict[str, Any]]) -> None:
            """Record a bulk-index call without writing to OpenSearch."""
            captured["index_name"] = index_name
            captured["documents"] = documents

        def _update_collection_timestamp(self, collection_name: str) -> None:
            """No-op stub for collection timestamp updates."""
            captured["updated_collection"] = collection_name

    monkeypatch.setattr(opensearch_adapter, "OpenSearchIngestor", WorkerIngestor)

    result = run_opensearch_ingestion_task(
        {
            "auth_type": "sigv4",
            "aws_service": "aoss",
            "aws_region": "us-east-1",
            "embed_model": "nvidia/test-embed",
            "summary_llm": "not-worker-serializable",
        },
        [
            {
                "file_id": "file-1",
                "file_name": "worker.txt",
                "data": b"worker opensearch ingestion",
                "suffix": ".txt",
                "metadata": {"suite": "dask"},
            }
        ],
        "docs",
    )

    assert result["status"] == "completed"
    assert captured["config"]["auth_type"] == "sigv4"
    assert captured["config"]["aws_service"] == "aoss"
    assert captured["config"]["start_ttl_cleanup"] is False
    assert captured["config"]["generate_summary"] is False
    assert "summary_llm" not in captured["config"]
    assert captured["worker_file_exists"] is True
    assert captured["documents"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert captured["documents"][0]["metadata"]["suite"] == "dask"


def test_setup_backend_passes_opensearch_yaml_config(monkeypatch):
    """Test that setup backend passes opensearch yaml config."""
    pytest.importorskip("nat")
    from knowledge_layer.register import KnowledgeRetrievalConfig
    from knowledge_layer.register import _setup_backend

    monkeypatch.setenv("OPENSEARCH_USERNAME", "env-user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "env-pass")

    config = KnowledgeRetrievalConfig(
        backend="opensearch",
        collection_name="docs",
        opensearch_url="search.example.com",
        opensearch_auth_type="sigv4",
        opensearch_aws_region="us-west-2",
        opensearch_aws_service="aoss",
        opensearch_index_prefix="tenant-a",
        opensearch_embedding_dim=4,
        opensearch_chunk_size=200,
        opensearch_chunk_overlap=20,
        embed_model="nvidia/test-embed",
        embed_base_url="https://integrate.example/v1",
    )

    backend, backend_config = _setup_backend(config)

    assert backend == "opensearch"
    assert backend_config["endpoint"] == "search.example.com"
    assert backend_config["auth_type"] == "sigv4"
    assert backend_config["aws_region"] == "us-west-2"
    assert backend_config["aws_service"] == "aoss"
    assert backend_config["index_prefix"] == "tenant-a"
    assert backend_config["embedding_dim"] == 4
    assert backend_config["chunk_size"] == 200
    assert backend_config["chunk_overlap"] == 20
    assert backend_config["embed_model"] == "nvidia/test-embed"
    assert backend_config["embed_base_url"] == "https://integrate.example/v1"


def test_opensearch_password_is_secret_and_not_leaked_in_repr():
    """The password field must redact in model representations and dumps."""
    pytest.importorskip("nat")
    from knowledge_layer.register import KnowledgeRetrievalConfig

    config = KnowledgeRetrievalConfig(
        backend="opensearch",
        collection_name="docs",
        opensearch_auth_type="basic",
        opensearch_username="admin",
        opensearch_password="s3cret",  # pragma: allowlist secret
    )

    assert "s3cret" not in repr(config)
    assert "s3cret" not in str(config.model_dump())


def test_setup_backend_resolves_secret_password_for_adapter(monkeypatch):
    """_setup_backend must hand the adapter the plaintext password so basic auth still works."""
    pytest.importorskip("nat")
    from knowledge_layer.register import KnowledgeRetrievalConfig
    from knowledge_layer.register import _setup_backend

    monkeypatch.delenv("OPENSEARCH_PASSWORD", raising=False)
    config = KnowledgeRetrievalConfig(
        backend="opensearch",
        collection_name="docs",
        opensearch_auth_type="basic",
        opensearch_username="admin",
        opensearch_password="s3cret",  # pragma: allowlist secret
    )

    _, backend_config = _setup_backend(config)

    assert backend_config["username"] == "admin"
    assert backend_config["password"] == "s3cret"  # pragma: allowlist secret


def test_setup_backend_uses_opensearch_environment_defaults(monkeypatch):
    """Test that setup backend uses opensearch environment defaults."""
    pytest.importorskip("nat")
    from knowledge_layer.register import KnowledgeRetrievalConfig
    from knowledge_layer.register import _setup_backend

    monkeypatch.setenv("OPENSEARCH_URL", "https://env.us-east-1.aoss.amazonaws.com")
    monkeypatch.setenv("OPENSEARCH_AUTH_TYPE", "sigv4")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("OPENSEARCH_AWS_SERVICE", "aoss")
    monkeypatch.setenv("OPENSEARCH_INDEX_PREFIX", "deep-researcher-env")
    monkeypatch.setenv("OPENSEARCH_INGESTION_MODE", "auto")
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://scheduler:8786")
    monkeypatch.setenv("OPENSEARCH_DASK_FILE_TRANSFER", "paths")

    config = KnowledgeRetrievalConfig(backend="opensearch")
    backend, backend_config = _setup_backend(config)

    assert backend == "opensearch"
    assert backend_config["endpoint"] == "https://env.us-east-1.aoss.amazonaws.com"
    assert backend_config["auth_type"] == "sigv4"
    assert backend_config["aws_region"] == "us-east-1"
    assert backend_config["aws_service"] == "aoss"
    assert backend_config["index_prefix"] == "deep-researcher-env"
    assert backend_config["ingestion_mode"] == "auto"
    assert backend_config["dask_scheduler_address"] == "tcp://scheduler:8786"
    assert backend_config["dask_file_transfer"] == "paths"


def test_ingestor_embed_raises_when_hosted_api_and_missing_key(monkeypatch):
    """Hosted NVIDIA API with no key should raise a clear error before HTTP."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    from knowledge_layer.opensearch.adapter import OpenSearchIngestor

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "http://localhost:9200",
            "auth_type": "none",
            "embed_base_url": "https://integrate.api.nvidia.com/v1",
            "start_ttl_cleanup": False,
        }
    )
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        ingestor._embed_texts(["hello world"])


def test_ingestor_embed_allows_local_nim_without_key(monkeypatch):
    """Self-hosted NIM with no key should pass through without complaint."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    from knowledge_layer.opensearch.adapter import OpenSearchIngestor

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "http://localhost:9200",
            "auth_type": "none",
            "embed_base_url": "http://nim-embed.ns-nim.svc.cluster.local:8000/v1",
            "start_ttl_cleanup": False,
        }
    )

    class _FakeOpenAI:
        def __init__(self, base_url, api_key):
            """Initialise _FakeOpenAI."""
            fake_emb = type("D", (), {"embedding": [0.0] * 4})()
            fake_resp = type("R", (), {"data": [fake_emb]})()
            self.embeddings = type("E", (), {"create": staticmethod(lambda **kw: fake_resp)})()

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    result = ingestor._embed_texts(["hello"])
    assert result == [[0.0, 0.0, 0.0, 0.0]]


def test_ensure_index_recovers_when_concurrent_create_races(monkeypatch):
    """Two concurrent jobs both see not-exists and both call create(); the
    losing call must not raise — re-check exists() and treat the index as
    ready if another worker already created it."""
    from knowledge_layer.opensearch.adapter import OpenSearchIngestor
    from opensearchpy.exceptions import RequestError

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "http://localhost:9200",
            "auth_type": "none",
            "start_ttl_cleanup": False,
        }
    )

    exists_calls: list[str] = []

    def fake_exists(index: str) -> bool:
        # First call (pre-create) returns False; subsequent re-check after the
        # race loss returns True (the winning worker has created it).
        """Stub index-exists check used by the ingestor fixture."""
        exists_calls.append(index)
        return len(exists_calls) >= 2

    def fake_create(index: str, body: dict) -> None:
        """Stub index-creation call used by the ingestor fixture."""
        raise RequestError(
            400,
            "resource_already_exists_exception",
            {
                "error": {
                    "type": "resource_already_exists_exception",
                    "reason": f"index [{index}/abc] already exists",
                }
            },
        )

    fake_client = type("C", (), {})()
    fake_client.indices = type("I", (), {"exists": staticmethod(fake_exists), "create": staticmethod(fake_create)})()
    monkeypatch.setattr(ingestor, "_get_client", lambda: fake_client)
    monkeypatch.setattr(
        ingestor,
        "_get_index_meta",
        lambda _index: {
            "collection_name": "smoke",
            "embedding_model": ingestor.embed_model_name,
            "embedding_dim": ingestor.embedding_dim,
        },
    )

    # Must not raise — race recovery should swallow the exists exception.
    result = ingestor._ensure_index("smoke")
    assert result.startswith("deep-researcher-smoke")
    assert len(exists_calls) == 2, "expected pre-create check + post-failure recovery check"


def test_ensure_index_rejects_different_owner_after_concurrent_create_race(monkeypatch):
    """A worker losing an index-creation race must not accept another collection's index."""
    from opensearchpy.exceptions import RequestError

    ingestor = OpenSearchIngestor({"start_ttl_cleanup": False})
    exists_calls = 0

    def fake_exists(index: str) -> bool:
        nonlocal exists_calls
        del index
        exists_calls += 1
        return exists_calls >= 2

    def fake_create(index: str, body: dict[str, Any]) -> None:
        del body
        raise RequestError(400, "resource_already_exists_exception", {"index": index})

    fake_client = type("C", (), {})()
    fake_client.indices = type("I", (), {"exists": staticmethod(fake_exists), "create": staticmethod(fake_create)})()
    monkeypatch.setattr(ingestor, "_get_client", lambda: fake_client)
    monkeypatch.setattr(
        ingestor,
        "_get_index_meta",
        lambda _index: {
            "collection_name": "different-collection",
            "embedding_model": ingestor.embed_model_name,
            "embedding_dim": ingestor.embedding_dim,
        },
    )

    with pytest.raises(RuntimeError, match="different-collection"):
        ingestor._ensure_index("smoke")


def test_list_files_aggregates_and_avoids_10k_hit_truncation(monkeypatch):
    """list_files must request 0 hits and aggregate by file_name so collections
    with more than the 10k index.max_result_window are not silently truncated.
    Chunk counts must come from bucket doc_count, not from counted hits."""
    from knowledge_layer.opensearch.adapter import OpenSearchIngestor

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "http://localhost:9200",
            "auth_type": "none",
            "start_ttl_cleanup": False,
        }
    )

    captured_body: dict[str, Any] = {}

    def fake_search(index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Return fake search results driven by the test fixture."""
        captured_body.update(body)
        return {
            "hits": {"hits": []},
            "aggregations": {
                "by_file": {
                    # No after_key — single page, iteration terminates.
                    "buckets": [
                        {
                            "key": {"file_id": "f1", "file_name": "huge.pdf"},
                            "doc_count": 50_000,
                            "doc": {
                                "hits": {
                                    "hits": [
                                        {
                                            "_source": {
                                                "file_id": "f1",
                                                "file_name": "huge.pdf",
                                                "file_size": 12_345_678,
                                                "created_at": "2026-05-01T00:00:00Z",
                                                "updated_at": "2026-05-01T00:01:00Z",
                                                "metadata": {"k": "v"},
                                            }
                                        }
                                    ]
                                }
                            },
                            "content_types": {"buckets": [{"key": "text", "doc_count": 50_000}]},
                        },
                        {
                            "key": {"file_id": "f2", "file_name": "small.md"},
                            "doc_count": 3,
                            "doc": {
                                "hits": {
                                    "hits": [
                                        {
                                            "_source": {
                                                "file_id": "f2",
                                                "file_name": "small.md",
                                                "file_size": 1234,
                                                "created_at": "2026-05-02T00:00:00Z",
                                                "updated_at": "2026-05-02T00:00:30Z",
                                                "metadata": {},
                                            }
                                        }
                                    ]
                                }
                            },
                            "content_types": {"buckets": []},
                        },
                    ]
                }
            },
        }

    fake_client = type("C", (), {})()
    fake_client.indices = type("I", (), {"exists": staticmethod(lambda index: True)})()
    fake_client.search = staticmethod(fake_search)
    monkeypatch.setattr(ingestor, "_get_client", lambda: fake_client)

    files = ingestor.list_files("smoke")

    assert captured_body.get("size") == 0, "search body must request 0 hits and aggregate instead"
    by_file_agg = (captured_body.get("aggs") or {}).get("by_file") or {}
    assert "composite" in by_file_agg, "must use composite aggregation to paginate exhaustively"

    by_name = {f.file_name: f for f in files}
    assert set(by_name) == {"huge.pdf", "small.md"}
    # 50 000 > 10 000 max_result_window — only an aggregation can carry this count.
    assert by_name["huge.pdf"].chunk_count == 50_000
    assert by_name["huge.pdf"].file_size == 12_345_678
    assert by_name["huge.pdf"].metadata["content_types"] == ["text"]
    assert by_name["small.md"].chunk_count == 3


def test_list_files_paginates_composite_until_after_key_exhausted(monkeypatch):
    """Composite aggregation pages through every distinct file_name. With more files
    than a single page can hold, list_files must follow after_key until exhaustion —
    otherwise large collections silently lose files past the first page."""
    from knowledge_layer.opensearch.adapter import OpenSearchIngestor

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "http://localhost:9200",
            "auth_type": "none",
            "start_ttl_cleanup": False,
        }
    )

    # 7 distinct files; emit in pages of 3. Must take ceil(7/3) = 3 requests.
    # Use distinct file_id and file_name to catch bugs where only one key drives pagination.
    all_files = [(f"id{i:02d}", f"f{i:02d}.txt") for i in range(7)]
    page_size = 3
    search_calls: list[dict[str, Any] | None] = []

    def fake_search(index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Return fake search results driven by the test fixture."""
        composite = body["aggs"]["by_file"]["composite"]
        after = composite.get("after")
        search_calls.append(after)
        after_key = (after["file_id"], after["file_name"]) if after else None
        remaining = [(fid, fname) for fid, fname in all_files if after_key is None or (fid, fname) > after_key]
        page = remaining[:page_size]
        buckets = [
            {
                "key": {"file_id": fid, "file_name": fname},
                "doc_count": 1,
                "doc": {"hits": {"hits": [{"_source": {"file_id": fid, "file_name": fname}}]}},
                "content_types": {"buckets": []},
            }
            for fid, fname in page
        ]
        agg: dict[str, Any] = {"buckets": buckets}
        if len(page) == page_size and len(remaining) > page_size:
            agg["after_key"] = {"file_id": page[-1][0], "file_name": page[-1][1]}
        return {"hits": {"hits": []}, "aggregations": {"by_file": agg}}

    fake_client = type("C", (), {})()
    fake_client.indices = type("I", (), {"exists": staticmethod(lambda index: True)})()
    fake_client.search = staticmethod(fake_search)
    monkeypatch.setattr(ingestor, "_get_client", lambda: fake_client)

    files = ingestor.list_files("paginated")

    # All 7 files surface, none dropped.
    assert sorted(f.file_name for f in files) == [fname for _, fname in all_files]
    # First call has no after; subsequent calls carry both composite keys in the cursor.
    assert search_calls[0] is None
    assert search_calls[1] == {"file_id": "id02", "file_name": "f02.txt"}
    assert search_calls[2] == {"file_id": "id05", "file_name": "f05.txt"}
    assert len(search_calls) == 3, f"expected 3 paginated requests, got {len(search_calls)}"


def test_ensure_index_reraises_when_create_fails_for_other_reasons(monkeypatch):
    """If create() fails for a non-race reason (index still missing after the
    failure), the original exception must propagate so the caller can fail
    the job rather than silently swallowing it."""
    from knowledge_layer.opensearch.adapter import OpenSearchIngestor
    from opensearchpy.exceptions import RequestError

    ingestor = OpenSearchIngestor(
        {
            "endpoint": "http://localhost:9200",
            "auth_type": "none",
            "start_ttl_cleanup": False,
        }
    )

    def fake_exists(index: str) -> bool:  # Always not-exists, even after failure
        """Stub index-exists check used by the ingestor fixture."""
        return False

    def fake_create(index: str, body: dict) -> None:
        """Stub index-creation call used by the ingestor fixture."""
        raise RequestError(
            400,
            "invalid_index_name_exception",
            {"error": {"type": "invalid_index_name_exception", "reason": "bad name"}},
        )

    fake_client = type("C", (), {})()
    fake_client.indices = type("I", (), {"exists": staticmethod(fake_exists), "create": staticmethod(fake_create)})()
    monkeypatch.setattr(ingestor, "_get_client", lambda: fake_client)

    with pytest.raises(RequestError):
        ingestor._ensure_index("smoke")
