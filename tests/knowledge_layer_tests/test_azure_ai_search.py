# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.core.exceptions import ServiceRequestError
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes.models import SearchIndex
from knowledge_layer.azure_ai_search import adapter as azure_adapter
from knowledge_layer.azure_ai_search.adapter import AzureAISearchIngestor
from knowledge_layer.azure_ai_search.adapter import AzureAISearchRetriever
from knowledge_layer.azure_ai_search.adapter import _build_index_schema
from knowledge_layer.azure_ai_search.adapter import _coerce_page_number
from knowledge_layer.azure_ai_search.adapter import _encode_marker
from knowledge_layer.azure_ai_search.adapter import _index_name_for_config
from knowledge_layer.azure_ai_search.adapter import _iter_index_batches
from knowledge_layer.azure_ai_search.adapter import _new_marker
from knowledge_layer.azure_ai_search.adapter import _resolve_filenames
from knowledge_layer.azure_ai_search.adapter import _validate_index_name
from knowledge_layer.azure_ai_search.adapter import _validate_index_schema
from knowledge_layer.register import KnowledgeRetrievalConfig
from knowledge_layer.register import _format_results
from knowledge_layer.register import _setup_backend
from pydantic import SecretStr

from deep_researcher_agent.knowledge import BaseIngestor
from deep_researcher_agent.knowledge import BaseRetriever
from deep_researcher_agent.knowledge import ContentType
from deep_researcher_agent.knowledge import RetrievalResult
from deep_researcher_agent.knowledge.factory import is_ingestor_registered
from deep_researcher_agent.knowledge.factory import is_retriever_registered
from deep_researcher_agent.knowledge.schema import FileStatus
from deep_researcher_agent.knowledge.schema import JobState


class FakeIndexingResult:
    def __init__(self, key: str, succeeded: bool = True, error_message: str | None = None):
        self.key = key
        self.succeeded = succeeded
        self.error_message = error_message


class FakeSearchResults(list):
    def __init__(self, documents: list[dict], facets: dict | None = None):
        super().__init__(documents)
        self._facets = facets or {}

    def get_facets(self):
        return self._facets


def _literal(filter_text: str | None, field: str, operator: str = "eq") -> str | None:
    if not filter_text:
        return None
    match = re.search(rf"{field} {operator} '((?:''|[^'])*)'", filter_text)
    return match.group(1).replace("''", "'") if match else None


class FakeSearchClient:
    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.upload_batches: list[list[dict]] = []
        self.delete_batches: list[list[dict]] = []
        self.search_filters: list[str | None] = []
        self.search_requests: list[dict] = []
        self.fail_upload_ids: set[str] = set()
        self.upload_response_failures_remaining = 0
        self.delete_failures_remaining: dict[str, int] = {}
        self.hidden_searches_remaining = 0
        self.stale_deleted_searches_remaining = 0
        self.deleted_documents: dict[str, dict] = {}
        self.upload_visibility_delays: dict[tuple[str | None, str | None], int] = {}
        self.pending_uploads: dict[str, tuple[dict, int]] = {}
        self.stale_uploaded_searches_remaining = 0
        self.stale_uploaded_documents: dict[str, dict] = {}

    def _publish(self, document: dict):
        key = document["id"]
        if previous := self.documents.get(key):
            self.stale_uploaded_documents[key] = dict(previous)
        self.documents[key] = document

    def upload_documents(self, documents: list[dict]):
        self.upload_batches.append(documents)
        results = []
        for document in documents:
            key = document["id"]
            succeeded = key not in self.fail_upload_ids
            if succeeded:
                pending = dict(document)
                delay = self.upload_visibility_delays.get((pending.get("record_type"), pending.get("status")), 0)
                if delay:
                    self.pending_uploads[key] = (pending, delay)
                else:
                    self.pending_uploads.pop(key, None)
                    self._publish(pending)
            results.append(FakeIndexingResult(key, succeeded, None if succeeded else "rejected"))
        if self.upload_response_failures_remaining:
            self.upload_response_failures_remaining -= 1
            raise ServiceRequestError("response lost after Azure accepted the upload")
        return results

    def delete_documents(self, documents: list[dict]):
        self.delete_batches.append(documents)
        results = []
        for document in documents:
            key = document["id"]
            remaining = self.delete_failures_remaining.get(key, 0)
            succeeded = remaining <= 0
            if remaining > 0:
                self.delete_failures_remaining[key] = remaining - 1
            if succeeded:
                self.pending_uploads.pop(key, None)
                deleted = self.documents.pop(key, None)
                if deleted:
                    self.deleted_documents[key] = deleted
            results.append(FakeIndexingResult(key, succeeded, None if succeeded else "retry"))
        return results

    def get_document(self, key: str):
        if key not in self.documents:
            raise ResourceNotFoundError("missing")
        return dict(self.documents[key])

    def search(self, search_text="*", filter=None, select=None, order_by=None, top=None, **kwargs):
        del order_by
        self.search_filters.append(filter)
        self.search_requests.append(
            {"search_text": search_text, "filter": filter, "select": select, "top": top, **kwargs}
        )
        equals = {
            field: _literal(filter, field)
            for field in ("record_type", "collection_id", "file_id", "file_name", "status")
        }
        excluded_record_type = _literal(filter, "record_type", "ne")
        after_id = _literal(filter, "id", "gt")
        search_in = re.search(r"search\.in\(file_id, '((?:''|[^'])*)', ','\)", filter or "")
        included_file_ids = set(search_in.group(1).replace("''", "'").split(",")) if search_in else None

        def matches(document):
            return (
                not any(value is not None and document.get(field) != value for field, value in equals.items())
                and (excluded_record_type is None or document.get("record_type") != excluded_record_type)
                and (after_id is None or document["id"] > after_id)
                and (included_file_ids is None or document.get("file_id") in included_file_ids)
            )

        for key, (document, remaining) in list(self.pending_uploads.items()):
            if not matches(document):
                continue
            if remaining > 1:
                self.pending_uploads[key] = (document, remaining - 1)
            else:
                self._publish(document)
                del self.pending_uploads[key]
        documents = list(self.documents.values())
        if self.stale_uploaded_searches_remaining > 0:
            stale_documents = [document for document in self.stale_uploaded_documents.values() if matches(document)]
            if stale_documents:
                stale_ids = {document["id"] for document in stale_documents}
                documents = [document for document in documents if document["id"] not in stale_ids] + stale_documents
                self.stale_uploaded_searches_remaining -= 1
        if self.stale_deleted_searches_remaining > 0 and self.deleted_documents:
            documents.extend(self.deleted_documents.values())
            self.stale_deleted_searches_remaining -= 1
        if self.hidden_searches_remaining > 0:
            documents = []
            self.hidden_searches_remaining -= 1
        documents = sorted(documents, key=lambda item: item["id"])
        for field, value in equals.items():
            if value is not None:
                documents = [document for document in documents if document.get(field) == value]
        if excluded_record_type is not None:
            documents = [document for document in documents if document.get("record_type") != excluded_record_type]
        if after_id is not None:
            documents = [document for document in documents if document["id"] > after_id]
        if included_file_ids is not None:
            documents = [document for document in documents if document.get("file_id") in included_file_ids]
        facet_documents = documents
        if top is not None:
            documents = documents[:top]
        facets = {}
        if "facets" in kwargs and any(str(value).startswith("record_type") for value in kwargs["facets"]):
            counts: dict[str, int] = {}
            for document in facet_documents:
                record_type = document.get("record_type")
                if record_type:
                    counts[record_type] = counts.get(record_type, 0) + 1
            facets["record_type"] = [{"value": value, "count": count} for value, count in counts.items()]
        if select:
            documents = [{key: document.get(key) for key in select} for document in documents]
        else:
            documents = [dict(document) for document in documents]
        return FakeSearchResults(documents, facets)

    def get_document_count(self):
        return len(self.documents)


class FakeIndexClient:
    def __init__(self):
        self.indexes: dict[str, SearchIndex] = {}
        self.race_on_create = False
        self.fail_create = False

    def get_index(self, name: str):
        if name not in self.indexes:
            raise ResourceNotFoundError("missing")
        return self.indexes[name]

    def create_index(self, index: SearchIndex):
        if self.race_on_create:
            self.indexes[index.name] = index
            raise RuntimeError("concurrent create")
        if self.fail_create:
            raise RuntimeError("create failed")
        self.indexes[index.name] = index
        return index

    def create_or_update_index(self, index: SearchIndex, **kwargs):
        del kwargs
        self.indexes[index.name] = index
        return index

    def list_indexes(self):
        return list(self.indexes.values())

    def list_index_names(self):
        return list(self.indexes)

    def delete_index(self, name: str):
        if name not in self.indexes:
            raise ResourceNotFoundError("missing")
        del self.indexes[name]


class FakeEmbedding:
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions

    def get_query_embedding(self, query):
        del query
        return [0.1] * self.dimensions

    def get_text_embedding_batch(self, texts):
        return [[0.1] * self.dimensions for _ in texts]


class FakeNode:
    def __init__(self, text: str, page: str = "1"):
        self.text = text
        self.metadata = {"page_label": page}

    def get_content(self):
        return self.text


class FakeSplitter:
    def __init__(self, nodes: list[FakeNode]):
        self.nodes = nodes

    def get_nodes_from_documents(self, documents):
        del documents
        return self.nodes


def _config(**overrides):
    config = {
        "endpoint": "https://example.search.windows.net",
        "api_key": SecretStr("test-key"),
        "embed_model": "test-embed",
        "embed_dim": 4,
        "embed_base_url": "https://integrate.api.nvidia.com/v1",
        "start_ttl_cleanup": False,
        "index_prefix": "deep-researcher-test",
    }
    config.update(overrides)
    return config


def _ingestor(*, index_client=None, search_client=None, **overrides):
    ingestor = AzureAISearchIngestor(_config(**overrides))
    ingestor._index_client = index_client or FakeIndexClient()
    search_client = search_client or FakeSearchClient()
    ingestor._search_client = search_client
    ingestor._index_validated = False
    return ingestor, search_client


def _write_file_manifest(
    ingestor,
    file_id: str,
    collection_name: str = "docs",
    file_name: str = "report.pdf",
    *,
    status: FileStatus = FileStatus.SUCCESS,
    chunk_count: int = 0,
    summary: str | None = None,
    ingested_at: datetime | None = None,
):
    info = azure_adapter.FileInfo(
        file_id=file_id,
        file_name=file_name,
        collection_name=collection_name,
        status=status,
        error_message="bad file" if status == FileStatus.FAILED else None,
        file_size=42,
        chunk_count=chunk_count,
        uploaded_at=datetime.now(UTC),
        ingested_at=ingested_at or datetime.now(UTC),
        metadata={"section": "finance"},
    )
    ingestor._write_file_manifest(info, summary=summary)
    return info


def _install_reader(monkeypatch):
    class FakeReader:
        def __init__(self, input_files):
            self.input_files = input_files

        def load_data(self):
            return [object()]

    monkeypatch.setattr("llama_index.core.SimpleDirectoryReader", FakeReader)


def _run_threads_synchronously(monkeypatch):
    class SynchronousThread:
        def __init__(self, *, target, args, **kwargs):
            del kwargs
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(azure_adapter.threading, "Thread", SynchronousThread)


def test_backend_registered_and_implements_sdk_contracts():
    assert is_ingestor_registered("azure_ai_search")
    assert is_retriever_registered("azure_ai_search")
    assert issubclass(AzureAISearchIngestor, BaseIngestor)
    assert issubclass(AzureAISearchRetriever, BaseRetriever)


def test_config_requires_endpoint_and_omits_removed_azure_options(monkeypatch):
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SEARCH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="azure_search_endpoint"):
        KnowledgeRetrievalConfig(backend="azure_ai_search")
    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
    )
    assert config.azure_search_api_key is None
    for field in ("chunk_size", "chunk_overlap", "use_hybrid", "use_semantic_ranker", "summary_max_chars"):
        assert field not in KnowledgeRetrievalConfig.model_fields


def test_config_uses_shared_environment_defaults(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://env.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "env-search-key")
    monkeypatch.setenv("DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX", "env-deep-researcher")
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_BASE_URL", "https://embed.example.com/v1")
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_MODEL", "env-embed")
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_DIM", "8")

    config = KnowledgeRetrievalConfig(backend="azure_ai_search")
    backend, backend_config = _setup_backend(config)
    adapter_config = azure_adapter._coerce_config(None)

    assert backend == "azure_ai_search"
    assert backend_config["endpoint"] == "https://env.search.windows.net/"
    assert backend_config["api_key"].get_secret_value() == "env-search-key"
    assert backend_config["index_prefix"] == "env-deep-researcher"
    assert backend_config["embed_base_url"] == "https://embed.example.com/v1"
    assert backend_config["embed_model"] == "env-embed"
    assert backend_config["embed_dim"] == 8
    assert adapter_config.endpoint == "https://env.search.windows.net"


def test_config_uses_defaults_for_empty_azure_environment_values(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX", "")
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_DIM", "")

    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
    )

    assert config.azure_search_index_prefix == "deep-researcher"
    assert config.embed_dim == 2048


def test_shared_embedding_defaults_match_adapter(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCHER_EMBED_BASE_URL", "")
    monkeypatch.delenv("DEEP_RESEARCHER_EMBED_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCHER_EMBED_DIM", raising=False)

    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
    )
    adapter_config = azure_adapter._coerce_config({"endpoint": "https://example.search.windows.net"})

    assert str(config.embed_base_url) == "https://integrate.api.nvidia.com/v1"
    assert adapter_config.embed_base_url == "https://integrate.api.nvidia.com/v1"
    assert config.embed_model == "nvidia/nemotron-3-embed-1b"
    assert config.embed_dim == 2048
    assert adapter_config.embed_model == config.embed_model
    assert adapter_config.embed_dim == config.embed_dim


def test_search_credential_uses_api_key_or_default_credential():
    with_key = azure_adapter._coerce_config(
        {"endpoint": "https://example.search.windows.net", "api_key": SecretStr("test-key")}
    )
    without_key = azure_adapter._coerce_config({"endpoint": "https://example.search.windows.net", "api_key": None})

    assert isinstance(azure_adapter._build_search_credential(with_key), AzureKeyCredential)
    assert isinstance(azure_adapter._build_search_credential(without_key), DefaultAzureCredential)


def test_setup_backend_preserves_secrets_and_prefix(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_RETRIEVER_BACKEND", "llamaindex")
    monkeypatch.setenv("KNOWLEDGE_INGESTOR_BACKEND", "llamaindex")
    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
        azure_search_api_key="test-search-key",  # pragma: allowlist secret
        azure_search_index_prefix="tenant-deep-researcher",
    )

    backend, backend_config = _setup_backend(config)

    assert backend == "azure_ai_search"
    assert isinstance(backend_config["api_key"], SecretStr)
    assert backend_config["index_prefix"] == "tenant-deep-researcher"


def test_search_api_key_survives_nat_json_serialization():
    api_key = "test-search-key"  # pragma: allowlist secret
    config = KnowledgeRetrievalConfig(
        backend="azure_ai_search",
        azure_search_endpoint="https://example.search.windows.net",
        azure_search_api_key=api_key,
    )

    serialized = config.model_dump(mode="json", by_alias=True, round_trip=True)

    assert serialized["azure_search_api_key"] == api_key


def test_index_name_is_stable_and_changes_with_embedding_configuration():
    index = _index_name_for_config("Deep Researcher Agent Prod", "test/embed", 2048)

    assert index == _index_name_for_config("Deep Researcher Agent Prod", "test/embed", 2048)
    assert index != _index_name_for_config("Deep Researcher Agent Prod", "other/embed", 2048)
    assert index != _index_name_for_config("Deep Researcher Agent Prod", "test/embed", 1024)
    assert len(index) <= 128
    assert re.fullmatch(r"[a-z0-9-]+", index)
    _validate_index_name(index)
    with pytest.raises(ValueError, match="Invalid Azure AI Search index name"):
        _validate_index_name("invalid_name")


def test_marker_and_schema_validation_reject_unowned_and_mismatched_indexes():
    cfg = SimpleNamespace(embed_model="test-embed", embed_dim=4, index_prefix="deep-researcher-test")
    marker = _new_marker(cfg)
    index = _build_index_schema("deep-researcher-docs-123456789abc", 4, _encode_marker(marker))

    assert _validate_index_schema(index, cfg)["schema_version"] == 1
    assert "metadata" not in marker
    assert "collection" not in marker
    index.description = "unmanaged"
    with pytest.raises(RuntimeError, match="not owned"):
        _validate_index_schema(index, cfg)
    index.description = _encode_marker(marker)
    embedding = next(field for field in index.fields if field.name == "embedding")
    embedding.vector_search_dimensions = 8
    with pytest.raises(RuntimeError, match="vector profile or dimensions"):
        _validate_index_schema(index, cfg)


def test_create_collection_handles_race_and_ignores_unmanaged_indexes():
    ingestor, _client = _ingestor()
    ingestor._index_client.race_on_create = True
    created = ingestor.create_collection("docs", description="Documents", metadata={"tenant": "alpha"})
    ingestor._index_client.indexes["unrelated"] = SearchIndex(name="unrelated", fields=[], description="other")

    assert created.name == "docs"
    assert created.metadata["tenant"] == "alpha"
    assert [collection.name for collection in ingestor.list_collections()] == ["docs"]


def test_multiple_collections_share_one_physical_index_and_keep_marker_bounded():
    ingestor, _client = _ingestor()
    large_value = "x" * 10_000

    ingestor.create_collection("docs", description=large_value, metadata={"large": large_value})
    ingestor.create_collection("private")

    assert len(ingestor._index_client.indexes) == 1
    index = ingestor._index_client.indexes[ingestor._physical_index_name()]
    assert len(index.description) < 4_000
    assert large_value not in index.description
    assert {item.name for item in ingestor.list_collections()} == {"docs", "private"}


def test_create_waits_for_collection_manifest_search_visibility(monkeypatch):
    ingestor, client = _ingestor()
    client.hidden_searches_remaining = 2
    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_DELAY_SECONDS", 0)

    ingestor.create_collection("docs")

    assert client.hidden_searches_remaining == 0
    assert [item.name for item in ingestor.list_collections()] == ["docs"]


def test_mismatched_and_foreign_indexes_are_ignored():
    ingestor, _client = _ingestor()
    mismatched_marker = _new_marker(ingestor.cfg)
    mismatched_marker["schema_version"] = azure_adapter._SCHEMA_VERSION + 1
    ingestor._index_client.indexes["deep-researcher-test-docs-mismatched"] = _build_index_schema(
        "deep-researcher-test-docs-mismatched",
        ingestor.cfg.embed_dim,
        _encode_marker(mismatched_marker),
    )
    ingestor._index_client.indexes["foreign"] = SearchIndex(name="foreign", fields=[], description="foreign")

    assert ingestor.list_collections() == []


def test_create_collection_propagates_real_create_failure():
    ingestor, _client = _ingestor()
    ingestor._index_client.fail_create = True

    with pytest.raises(RuntimeError, match="create failed"):
        ingestor.create_collection("docs")


@pytest.mark.parametrize(
    ("hit", "expected"),
    [
        ({"@search.reranker_score": 2.0, "@search.score": 0.9}, 0.9),
        ({"@search.score": 8.0}, 1.0),
        ({"@search.score": -1.0}, 0.0),
        ({"@search.score": 0.75}, 0.75),
    ],
)
def test_normalize_clamps_search_score(hit, expected):
    chunk = AzureAISearchRetriever.__new__(AzureAISearchRetriever).normalize(
        {"id": "chunk-1", "chunk": "text", "file_name": "report.pdf", **hit}
    )
    assert chunk.score == expected


def test_normalize_parses_metadata_and_populates_citation():
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    cited = retriever.normalize(
        {
            "id": "chunk-1",
            "chunk": "text",
            "file_id": "file-1",
            "file_name": "report.pdf",
            "page_number": "3",
            "metadata": '{"section":"intro"}',
        }
    )
    fallback = retriever.normalize({"page_number": 0})

    assert cited.display_citation == "report.pdf, p.3"
    assert cited.content_type is ContentType.TEXT
    assert cited.metadata == {"section": "intro", "file_id": "file-1"}
    assert fallback.content == ""
    assert fallback.file_name == "unknown"
    UUID(fallback.chunk_id)


def test_shared_formatter_retains_source_and_citation_lines():
    chunk = AzureAISearchRetriever.__new__(AzureAISearchRetriever).normalize(
        {"id": "chunk-1", "chunk": "text", "file_name": "report.pdf", "page_number": 3}
    )
    formatted = _format_results(RetrievalResult(query="query", backend="azure_ai_search", chunks=[chunk]), "query")

    assert "Source: report.pdf" in formatted
    assert "Citation: report.pdf, p.3" in formatted
    assert "Relevance Score: 0.00" in formatted
    assert "Vector Distance:" not in formatted


@pytest.mark.asyncio
async def test_retrieve_builds_scoped_hybrid_search_request():
    class FakeClient:
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [{"id": "chunk-1", "chunk": "answer", "file_name": "report.pdf"}]

    client = FakeClient()
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    retriever._embedding = FakeEmbedding(2)
    retriever._get_client = lambda collection_name: client

    result = await retriever.retrieve("hello", "session-1", top_k=5, filters={})

    assert result.success
    assert client.kwargs["search_text"] == "hello"
    assert len(client.kwargs["vector_queries"]) == 1
    assert "record_type eq 'chunk'" in client.kwargs["filter"]
    assert "collection_id eq 'session-1'" in client.kwargs["filter"]
    assert "query_type" not in client.kwargs
    assert client.kwargs["select"] == ["id", "chunk", "file_id", "file_name", "page_number", "metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", [{"$filter": "file_id ne ''"}, {"file_name": "report.pdf"}])
async def test_retrieve_rejects_filters(filters):
    retriever = AzureAISearchRetriever.__new__(AzureAISearchRetriever)
    retriever._get_client = lambda collection_name: pytest.fail(f"unexpected Azure request for {collection_name}")

    result = await retriever.retrieve("hello", "session-1", filters=filters)

    assert not result.success
    assert result.error_message == "Azure AI Search metadata filters are not supported"


def test_submit_job_uses_one_canonical_file_id(monkeypatch, tmp_path):
    class NoStartThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(azure_adapter.threading, "Thread", NoStartThread)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, _client = _ingestor()

    job_id = ingestor.submit_job(
        [str(path)],
        "docs",
        {"original_filenames": ["original.txt"], "metadata": {"large": "x" * 10_000}},
    )
    job = ingestor.get_job_status(job_id)
    file_id = job.file_details[0].file_id

    assert job.status == JobState.PENDING
    assert file_id in ingestor._files
    assert ingestor._files[file_id].file_name == "original.txt"
    assert ingestor._files[file_id].metadata["large"] == "x" * 10_000
    assert _client.documents == {}


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_submit_job_rolls_back_manifests_when_second_write_fails(monkeypatch, tmp_path, rollback_fails):
    _run_threads_synchronously(monkeypatch)
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in paths:
        path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    write_manifest = ingestor._write_file_manifest
    writes = []

    def fail_second_manifest(info, **kwargs):
        assert info.metadata["job_id"] in ingestor._jobs
        writes.append(info.file_id)
        if len(writes) == 2:
            raise RuntimeError("second manifest failed")
        return write_manifest(info, **kwargs)

    monkeypatch.setattr(ingestor, "_write_file_manifest", fail_second_manifest)
    if rollback_fails:

        def fail_rollback(*args, **kwargs):
            raise RuntimeError("rollback unavailable")

        monkeypatch.setattr(ingestor, "_delete_document_ids", fail_rollback)

    job_id = ingestor.submit_job([str(path) for path in paths], "docs")
    job = ingestor.get_job_status(job_id)

    assert job.status == JobState.FAILED
    assert job.processed_files == 2
    assert all(detail.status == FileStatus.FAILED for detail in job.file_details)
    assert all(ingestor._files[detail.file_id].status == FileStatus.FAILED for detail in job.file_details)
    assert "second manifest failed" in job.error_message
    if rollback_fails:
        assert "manifest rollback failed: rollback unavailable" in job.error_message
    else:
        assert not [document for document in client.documents.values() if document.get("record_type") == "file"]
        assert client.delete_batches
    assert all(path.exists() for path in paths)


def test_submit_job_records_thread_start_failure(monkeypatch, tmp_path):
    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(azure_adapter.threading, "Thread", FailingThread)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()

    job_id = ingestor.submit_job([str(path)], "docs", {"cleanup_files": True})
    job = ingestor.get_job_status(job_id)

    assert job.status == JobState.FAILED
    assert job.processed_files == 1
    assert job.file_details[0].status == FileStatus.FAILED
    assert "thread unavailable" in job.error_message
    assert client.documents == {}
    assert not path.exists()


def test_upload_file_preserves_caller_owned_source_by_default(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    _run_threads_synchronously(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, _client = _ingestor()
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("content")])

    ingestor.upload_file(str(path), "docs")

    assert path.exists()


def test_job_completion_waits_for_terminal_manifest_and_chunks(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    _run_threads_synchronously(monkeypatch)
    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_DELAY_SECONDS", 0)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("first"), FakeNode("second")])
    client.upload_visibility_delays[("file", FileStatus.SUCCESS.value)] = 2
    client.upload_visibility_delays[("chunk", None)] = 2

    job_id = ingestor.submit_job([str(path)], "docs")
    job = ingestor.get_job_status(job_id)
    file_id = job.file_details[0].file_id

    assert job.status == JobState.COMPLETED
    assert client.pending_uploads == {}
    assert ingestor.get_file_status(file_id, "docs").status == FileStatus.SUCCESS
    client.search_filters.clear()
    client.stale_uploaded_searches_remaining = 1
    assert ingestor.delete_file(file_id, "docs")
    assert client.stale_uploaded_searches_remaining == 0
    assert not any("record_type eq 'chunk'" in (filter_text or "") for filter_text in client.search_filters)
    assert not [document for document in client.documents.values() if document.get("record_type") == "chunk"]
    assert ingestor.delete_collection("docs")


def test_finalization_timeout_does_not_leave_retryable_success(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    _run_threads_synchronously(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("content")])

    wait_for_visibility = ingestor._wait_for_job_visibility
    attempts = 0

    def timeout_once(job_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("visibility timeout")
        wait_for_visibility(job_id)

    monkeypatch.setattr(ingestor, "_wait_for_job_visibility", timeout_once)
    job_ids = [ingestor.submit_job([str(path)], "docs") for _ in range(2)]
    jobs = [ingestor.get_job_status(job_id) for job_id in job_ids]
    successful_ids = [file.file_id for file in ingestor.list_files("docs") if file.status == FileStatus.SUCCESS]

    assert {
        "states": [(job.status, job.file_details[0].status) for job in jobs],
        "successful_ids": successful_ids,
    } == {
        "states": [(JobState.FAILED, FileStatus.FAILED), (JobState.COMPLETED, FileStatus.SUCCESS)],
        "successful_ids": [jobs[1].file_details[0].file_id],
    }
    failed_file_id = jobs[0].file_details[0].file_id
    assert not [document for document in client.documents.values() if document.get("file_id") == failed_file_id]


def test_multi_file_visibility_polling_uses_bounded_wait_rounds(monkeypatch):
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "unrelated", chunk_count=1)
    unrelated_chunk_id = "chunk-unrelated-00000000"
    client.pending_uploads[unrelated_chunk_id] = (
        {
            "id": unrelated_chunk_id,
            "record_type": "chunk",
            "collection_id": "docs",
            "file_id": "unrelated",
        },
        5,
    )
    file_details = []
    for index in range(3):
        file_id = f"file-{index}"
        _write_file_manifest(ingestor, file_id, chunk_count=1)
        client.documents[f"chunk-{file_id}-00000000"] = {
            "id": f"chunk-{file_id}-00000000",
            "record_type": "chunk",
            "collection_id": "docs",
            "file_id": file_id,
        }
        file_details.append(
            azure_adapter.FileProgress(
                file_id=file_id,
                file_name=f"file-{index}.txt",
                status=FileStatus.SUCCESS,
                chunks_created=1,
            )
        )
    ingestor._jobs["job"] = azure_adapter.IngestionJobStatus(
        job_id="job",
        status=JobState.PROCESSING,
        collection_name="docs",
        backend="azure_ai_search",
        submitted_at=datetime.now(UTC),
        total_files=len(file_details),
        file_details=file_details,
    )
    waits = []
    monkeypatch.setattr(azure_adapter.time, "sleep", waits.append)

    ingestor._wait_for_job_visibility("job")

    assert len(waits) <= 2 * (azure_adapter._CONSISTENCY_STABLE_READS - 1)
    assert client.pending_uploads[unrelated_chunk_id][1] == 5
    assert "search.in(file_id" in client.search_filters[-1]
    assert "unrelated" not in client.search_filters[-1]


def test_collection_delete_prefers_tracked_terminal_file_state():
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    stale = _write_file_manifest(ingestor, "file-1", status=FileStatus.INGESTING)
    ingestor._files[stale.file_id] = stale.model_copy(update={"status": FileStatus.SUCCESS, "chunk_count": 1})
    client.documents["chunk-file-1-00000000"] = {
        "id": "chunk-file-1-00000000",
        "record_type": "chunk",
        "collection_id": "docs",
        "file_id": "file-1",
    }

    assert ingestor.delete_collection("docs")
    assert not [document for document in client.documents.values() if document.get("collection_id") == "docs"]


def test_restarted_collection_delete_waits_for_hidden_children(monkeypatch):
    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_DELAY_SECONDS", 0)
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "file-1", chunk_count=1)
    client.documents["chunk-file-1-00000000"] = {
        "id": "chunk-file-1-00000000",
        "record_type": "chunk",
        "collection_id": "docs",
        "file_id": "file-1",
    }
    restarted, _client = _ingestor(index_client=ingestor._index_client, search_client=client)
    client.hidden_searches_remaining = 10

    assert restarted.delete_collection("docs")
    assert client.hidden_searches_remaining == 0
    remaining_ids = {
        document["id"] for document in client.documents.values() if document.get("collection_id") == "docs"
    }
    assert remaining_ids == set()


def test_collection_delete_fences_concurrent_submission(monkeypatch, tmp_path):
    monkeypatch.setattr(
        azure_adapter.threading,
        "Thread",
        lambda *args, **kwargs: SimpleNamespace(start=lambda: None),
    )
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    write_manifest = ingestor._write_collection_manifest
    submitted_jobs = []
    submission_errors = []

    def submit_after_fence(*args, **kwargs):
        manifest = write_manifest(*args, **kwargs)
        if kwargs.get("status") == azure_adapter._COLLECTION_DELETING:
            try:
                submitted_jobs.append(ingestor.submit_job([str(path)], "docs"))
            except ValueError as error:
                submission_errors.append(str(error))
        return manifest

    monkeypatch.setattr(ingestor, "_write_collection_manifest", submit_after_fence)

    assert ingestor.delete_collection("docs")
    assert submitted_jobs == []
    assert submission_errors == ["Collection 'docs' is being deleted"]
    assert not [document for document in client.documents.values() if document.get("collection_id") == "docs"]
    assert ingestor._jobs == {}
    assert ingestor._files == {}


def test_post_upload_failure_rolls_back_chunks_and_retains_failed_manifest(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    _run_threads_synchronously(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("content")])
    client.upload_visibility_delays[("chunk", None)] = 4

    def fail_timestamp(collection):
        del collection
        raise RuntimeError("fail")

    monkeypatch.setattr(ingestor, "_update_collection_timestamp", fail_timestamp)

    uploaded = ingestor.upload_file(str(path), "docs")
    status = ingestor.get_file_status(uploaded.file_id, "docs")

    assert status.status == FileStatus.FAILED
    assert client.pending_uploads == {}
    assert not [document for document in client.documents.values() if document.get("record_type") == "chunk"]


def test_success_manifest_failure_rolls_back_uploaded_chunks(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    _run_threads_synchronously(monkeypatch)
    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_DELAY_SECONDS", 0)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("content")])
    client.upload_visibility_delays[("chunk", None)] = azure_adapter._CONSISTENCY_ATTEMPTS + 1
    write_manifest = ingestor._write_file_manifest

    def fail_success_manifest(info, **kwargs):
        if info.status == FileStatus.SUCCESS:
            raise RuntimeError("success manifest failed")
        return write_manifest(info, **kwargs)

    monkeypatch.setattr(ingestor, "_write_file_manifest", fail_success_manifest)

    job = ingestor.get_job_status(ingestor.submit_job([str(path)], "docs"))
    file_id = job.file_details[0].file_id

    assert (job.status, job.file_details[0].status) == (JobState.FAILED, FileStatus.FAILED)
    assert "success manifest failed" in job.file_details[0].error_message
    assert not [
        document
        for document in client.documents.values()
        if document.get("record_type") == "chunk" and document.get("file_id") == file_id
    ]
    assert not [
        document
        for document, _remaining in client.pending_uploads.values()
        if document.get("record_type") == "chunk" and document.get("file_id") == file_id
    ]


def test_response_loss_rolls_back_all_attempted_chunk_ids(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("first"), FakeNode("second")])
    client.upload_visibility_delays[("chunk", None)] = 4
    client.upload_response_failures_remaining = 1

    with pytest.raises(ServiceRequestError, match="response lost"):
        ingestor._process_file(
            path=str(path),
            collection_name="docs",
            file_id="new",
            file_name="document.txt",
            file_size=7,
            uploaded_at=datetime.now(UTC),
            metadata={},
        )

    attempted_ids = {"chunk-new-00000000", "chunk-new-00000001"}
    assert attempted_ids & client.documents.keys() == set()
    assert attempted_ids & client.pending_uploads.keys() == set()


def test_response_loss_during_manifest_update_preserves_collection():
    ingestor, client = _ingestor()
    ingestor.create_collection("docs", description="before")
    manifest_id = ingestor._get_collection_manifest("docs")["id"]
    client.upload_response_failures_remaining = 1

    with pytest.raises(ServiceRequestError, match="response lost"):
        ingestor._write_collection_manifest("docs", description="after")

    assert client.documents[manifest_id]["description"] == "after"


def test_batches_respect_action_count_and_payload_size():
    documents = [{"id": str(index), "chunk": "x" * 10} for index in range(5)]
    count_batches = list(_iter_index_batches(documents, "upload", max_actions=2, max_bytes=10_000))
    byte_batches = list(_iter_index_batches(documents, "upload", max_actions=100, max_bytes=120))

    assert [len(batch) for batch in count_batches] == [2, 2, 1]
    assert len(byte_batches) > 1
    with pytest.raises(ValueError, match="16 MiB"):
        list(_iter_index_batches([{"id": "large", "chunk": "x" * 500}], "upload", max_bytes=100))


def test_partial_upload_rolls_back_successful_actions():
    ingestor, client = _ingestor()
    documents = [{"id": f"file-{index}", "chunk": "text"} for index in range(3)]
    client.fail_upload_ids.add("file-1")

    with pytest.raises(RuntimeError, match="rejected upload"):
        ingestor._upload_documents(client, documents)

    assert client.documents == {}
    assert client.delete_batches


def test_delete_retries_per_document_failures():
    ingestor, client = _ingestor()
    client.documents["chunk-1"] = {"id": "chunk-1"}
    client.delete_failures_remaining["chunk-1"] = 1

    ingestor._delete_document_ids(client, ["chunk-1"])

    assert "chunk-1" not in client.documents
    assert len(client.delete_batches) == 2


def test_persistent_delete_failure_is_reported():
    ingestor, client = _ingestor()
    client.documents["chunk-1"] = {"id": "chunk-1"}
    client.delete_failures_remaining["chunk-1"] = 10

    with pytest.raises(RuntimeError, match="rejected delete"):
        ingestor._delete_document_ids(client, ["chunk-1"])

    assert "chunk-1" in client.documents


def test_pagination_is_exhaustive_and_file_ids_are_odata_escaped(monkeypatch):
    monkeypatch.setattr(azure_adapter, "_PAGE_SIZE", 2)
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    for index in range(5):
        _write_file_manifest(ingestor, f"file-{index}", file_name=f"file-{index}.txt")

    assert len(ingestor.list_files("docs")) == 5
    crafted = "x' or file_id ne ''"
    assert ingestor._delete_file_documents(crafted, "docs") == 0
    assert any("file_id eq 'x'' or file_id ne '''''" in (item or "") for item in client.search_filters)


def test_same_name_upload_keeps_existing_file_chunks(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("new content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("new")])
    client.documents["old-00000000"] = {
        "id": "old-00000000",
        "record_type": "chunk",
        "collection_id": "docs",
        "chunk": "old",
        "embedding": [0.1] * 4,
        "file_id": "old",
        "file_name": "document.txt",
        "page_number": 1,
        "chunk_index": 0,
        "file_size": 100,
        "uploaded_at": datetime.now(UTC),
        "ingested_at": datetime.now(UTC),
        "metadata": "{}",
    }

    count, summary, _ingested_at = ingestor._process_file(
        path=str(path),
        collection_name="docs",
        file_id="new",
        file_name="document.txt",
        file_size=11,
        uploaded_at=datetime.now(UTC),
        metadata={"tenant": "alpha"},
    )

    assert count == 1
    assert summary is None
    assert {document["id"] for document in client.documents.values() if document.get("record_type") == "chunk"} == {
        "old-00000000",
        "chunk-new-00000000",
    }
    assert client.documents["chunk-new-00000000"]["metadata"] == '{"tenant":"alpha"}'


def test_failed_upload_rolls_back_new_chunks_and_preserves_existing_duplicate(monkeypatch, tmp_path):
    _install_reader(monkeypatch)
    path = tmp_path / "document.txt"
    path.write_text("new content", encoding="utf-8")
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor._embedding = FakeEmbedding()
    ingestor._splitter = FakeSplitter([FakeNode("first"), FakeNode("second")])
    client.documents["old-00000000"] = {
        "id": "old-00000000",
        "record_type": "chunk",
        "collection_id": "docs",
        "file_id": "old",
        "file_name": "document.txt",
    }
    client.fail_upload_ids.add("chunk-new-00000001")

    with pytest.raises(RuntimeError, match="rejected upload"):
        ingestor._process_file(
            path=str(path),
            collection_name="docs",
            file_id="new",
            file_name="document.txt",
            file_size=11,
            uploaded_at=datetime.now(UTC),
            metadata={},
        )

    chunk_ids = {document["id"] for document in client.documents.values() if document.get("record_type") == "chunk"}
    assert chunk_ids == {"old-00000000"}


def test_duplicate_file_manifests_are_independent():
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "old", file_name="document.txt")
    _write_file_manifest(ingestor, "new", file_name="document.txt")

    assert [(item.file_id, item.file_name) for item in ingestor.list_files("docs")] == [
        ("new", "document.txt"),
        ("old", "document.txt"),
    ]


def test_metadata_counts_and_status_round_trip():
    ingestor, client = _ingestor()
    ingestor.create_collection("docs", metadata={"tenant": "alpha"})
    _write_file_manifest(ingestor, "file-1", chunk_count=3)
    for index in range(3):
        client.documents[f"chunk-file-1-{index:08d}"] = {
            "id": f"chunk-file-1-{index:08d}",
            "record_type": "chunk",
            "collection_id": "docs",
            "file_id": "file-1",
            "file_name": "report.pdf",
        }

    files = ingestor.list_files("docs")
    collection = ingestor.get_collection("docs")

    assert files[0].file_id == "file-1"
    assert files[0].chunk_count == 3
    assert files[0].metadata == {"section": "finance"}
    assert ingestor.get_file_status("file-1", "docs") == files[0]
    assert collection.file_count == 1
    assert collection.chunk_count == 3
    assert collection.metadata["tenant"] == "alpha"
    facet_request = next(request for request in client.search_requests if request.get("facets"))
    assert facet_request["top"] == 0
    assert facet_request["facets"] == ["record_type,count:0"]


def test_file_status_and_deletion_do_not_cross_collection_boundaries():
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    ingestor.create_collection("private")
    _write_file_manifest(ingestor, "shared-id", "docs")
    _write_file_manifest(ingestor, "shared-id", "private")
    for collection_name in ("docs", "private"):
        key = f"chunk-{collection_name}"
        client.documents[key] = {
            "id": key,
            "record_type": "chunk",
            "collection_id": collection_name,
            "file_id": "shared-id",
        }

    assert ingestor.delete_file("shared-id", "docs")
    assert ingestor.get_file_status("shared-id", "docs") is None
    assert ingestor.get_file_status("shared-id", "private") is not None
    assert "chunk-docs" not in client.documents
    assert "chunk-private" in client.documents


def test_delete_waits_for_stale_file_manifest_to_disappear(monkeypatch):
    ingestor, client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "file-1")
    client.stale_deleted_searches_remaining = 2
    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_DELAY_SECONDS", 0)

    assert ingestor.delete_file("file-1", "docs")
    assert client.stale_deleted_searches_remaining == 0
    client.stale_deleted_searches_remaining = 1
    assert ingestor.list_files("docs") == []


def test_search_count_requires_stable_matches(monkeypatch):
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    observations = iter([[], [], [{"id": "stale"}], [], [], []])
    reads = 0

    def iter_documents(*args, **kwargs):
        del args, kwargs
        nonlocal reads
        reads += 1
        return iter(next(observations))

    monkeypatch.setattr(ingestor, "_iter_documents", iter_documents)
    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_DELAY_SECONDS", 0)

    ingestor._wait_for_search_count("file_id eq 'file-1'", expected_count=0, label="file", stable_reads=3)

    assert reads == 6


def test_delete_timeout_preserves_retryable_bookkeeping(monkeypatch):
    ingestor, client = _ingestor(generate_summary=True)
    ingestor.create_collection("docs")
    now = datetime.now(UTC)
    _write_file_manifest(
        ingestor,
        "older",
        summary="Older summary",
        ingested_at=now - timedelta(minutes=1),
    )
    newer = _write_file_manifest(ingestor, "newer", summary="Newer summary", ingested_at=now)
    ingestor._files["newer"] = newer
    client.stale_deleted_searches_remaining = 2
    registered = []
    timestamp_updates = []
    update_collection_timestamp = ingestor._update_collection_timestamp

    def track_collection_timestamp(collection_name):
        timestamp_updates.append(collection_name)
        update_collection_timestamp(collection_name)

    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_ATTEMPTS", 1)
    monkeypatch.setattr(
        azure_adapter,
        "register_summary",
        lambda collection, file_name, summary: registered.append((collection, file_name, summary)),
    )
    monkeypatch.setattr(ingestor, "_update_collection_timestamp", track_collection_timestamp)

    with pytest.raises(RuntimeError, match="file 'newer' manifest and chunks"):
        ingestor.delete_file("newer", "docs")

    assert "newer" in ingestor._files
    assert ("docs", "newer") not in ingestor._deleted_files
    assert registered == []
    assert timestamp_updates == []

    monkeypatch.setattr(azure_adapter, "_CONSISTENCY_ATTEMPTS", 3)
    client.stale_deleted_searches_remaining = 0
    client.deleted_documents.clear()

    assert ingestor.delete_file("newer", "docs")
    assert registered == [("docs", "report.pdf", "Older summary")]
    assert timestamp_updates == ["docs"]
    assert not ingestor.delete_file("newer", "docs")


def test_failed_uploads_remain_visible():
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "failed", file_name="failed.txt", status=FileStatus.FAILED)

    files = ingestor.list_files("docs")

    assert [(item.file_id, item.status) for item in files] == [("failed", FileStatus.FAILED)]


def test_ttl_deletes_only_expired_owned_collection_and_clears_summary(monkeypatch):
    ingestor, client = _ingestor(generate_summary=True)
    ingestor.create_collection("old")
    ingestor.create_collection("new")
    old_manifest = next(document for document in client.documents.values() if document.get("collection_id") == "old")
    old_manifest["updated_at"] = datetime.now(UTC) - timedelta(hours=25)
    ingestor._index_client.indexes["unrelated"] = SearchIndex(name="unrelated", fields=[], description="other")
    cleared = []
    monkeypatch.setattr(azure_adapter, "clear_collection_summaries", cleared.append)
    ingestor._ttl_hours = 24

    ingestor._cleanup_expired_collections()

    assert ingestor.get_collection("old") is None
    assert ingestor.get_collection("new") is not None
    assert ingestor._physical_index_name() in ingestor._index_client.indexes
    assert "unrelated" in ingestor._index_client.indexes
    assert cleared == ["old"]


def test_collection_summary_clears_only_after_confirmed_delete(monkeypatch):
    ingestor, _client = _ingestor(generate_summary=True)
    ingestor.create_collection("docs")
    cleared = []
    monkeypatch.setattr(azure_adapter, "clear_collection_summaries", cleared.append)

    assert ingestor.delete_collection("docs")
    assert cleared == ["docs"]
    assert not ingestor.delete_collection("docs")
    assert cleared == ["docs"]


def test_file_summary_clears_only_after_confirmed_delete(monkeypatch):
    ingestor, client = _ingestor(generate_summary=True)
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "file-1", summary="Summary")
    client.documents["chunk-file-1-00000000"] = {
        "id": "chunk-file-1-00000000",
        "record_type": "chunk",
        "collection_id": "docs",
        "file_id": "file-1",
        "file_name": "report.pdf",
    }
    client.delete_failures_remaining["chunk-file-1-00000000"] = 10
    cleared = []
    monkeypatch.setattr(azure_adapter, "unregister_summary", lambda collection, file_name: cleared.append(file_name))

    with pytest.raises(RuntimeError, match="rejected delete"):
        ingestor.delete_file("file-1", "docs")

    assert cleared == []


def test_summary_cleanup_is_skipped_when_summaries_are_disabled(monkeypatch):
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "file-1")
    cleared = []
    monkeypatch.setattr(
        azure_adapter,
        "unregister_summary",
        lambda collection, file_name: cleared.append((collection, file_name)),
    )
    monkeypatch.setattr(azure_adapter, "clear_collection_summaries", lambda collection: cleared.append((collection,)))

    assert ingestor.delete_file("file-1", "docs")
    assert ingestor.delete_collection("docs")
    assert cleared == []


def test_deleting_newest_duplicate_restores_previous_summary(monkeypatch):
    ingestor, _client = _ingestor(generate_summary=True)
    ingestor.create_collection("docs")
    now = datetime.now(UTC)
    _write_file_manifest(
        ingestor,
        "older",
        file_name="report.pdf",
        summary="Older summary",
        ingested_at=now - timedelta(minutes=1),
    )
    _write_file_manifest(ingestor, "newer", file_name="report.pdf", summary="Newer summary", ingested_at=now)
    registered = []
    monkeypatch.setattr(
        azure_adapter,
        "register_summary",
        lambda collection, file_name, summary: registered.append((collection, file_name, summary)),
    )

    assert ingestor.delete_file("newer", "docs")
    assert registered == [("docs", "report.pdf", "Older summary")]


def test_active_file_and_collection_deletion_are_rejected():
    ingestor, _client = _ingestor()
    ingestor.create_collection("docs")
    _write_file_manifest(ingestor, "active", status=FileStatus.INGESTING)

    with pytest.raises(ValueError, match="while it is ingesting"):
        ingestor.delete_file("active", "docs")
    with pytest.raises(ValueError, match="while files are ingesting"):
        ingestor.delete_collection("docs")
    assert ingestor.get_collection("docs") is not None


@pytest.mark.asyncio
async def test_health_checks_run_sync_sdk_off_event_loop(monkeypatch):
    ingestor, _client = _ingestor()
    calls = []

    async def fake_to_thread(function, *args):
        calls.append(function)
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    assert await ingestor.health_check()
    assert len(calls) == 1


def test_index_schema_uses_requested_dimensions_and_fields():
    schema = _build_index_schema("deep-researcher-session-123456789abc", 4096)
    fields = {field.name: field for field in schema.fields}

    assert fields["embedding"].vector_search_dimensions == 4096
    assert fields["file_id"].filterable
    assert fields["id"].sortable
    assert fields["record_type"].facetable
    assert fields["collection_id"].filterable
    assert schema.semantic_search is None


def test_splitter_uses_fixed_chunking_configuration():
    ingestor, _client = _ingestor()

    assert ingestor.splitter.chunk_size == 1024
    assert ingestor.splitter.chunk_overlap == 128


def test_filename_page_normalization_and_error_translation(tmp_path):
    paths = [str(tmp_path / "tmp-one"), str(tmp_path / "tmp-two")]

    assert _resolve_filenames(paths, ["one.pdf", "two.docx"]) == ["one.pdf", "two.docx"]
    assert _resolve_filenames(paths, {paths[0]: "mapped.txt"}) == ["mapped.txt", "tmp-two"]
    assert _coerce_page_number("4") == 4
    assert _coerce_page_number("cover") is None
    assert (
        AzureAISearchIngestor._translate_error(ServiceRequestError("connection refused"))
        == "AI Search service unavailable: connection refused"
    )
