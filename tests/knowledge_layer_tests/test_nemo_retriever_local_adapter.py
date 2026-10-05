# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from knowledge_layer.llamaindex.adapter import _extract_images_from_pdf
from knowledge_layer.nemo_retriever import _local_client
from knowledge_layer.nemo_retriever import adapter as service_adapter
from knowledge_layer.nemo_retriever._local_client import LocalRuntime
from knowledge_layer.nemo_retriever._local_client import LocalSettings
from knowledge_layer.nemo_retriever._local_client import NemoRetrieverLocalDependencyError
from knowledge_layer.nemo_retriever._local_client import NemoRetrieverLocalError
from knowledge_layer.nemo_retriever._local_client import NemoRetrieverLocalLockError
from knowledge_layer.nemo_retriever._local_client import NemoRetrieverLocalOwnershipError
from knowledge_layer.nemo_retriever.local_adapter import NemoRetrieverLocalIngestor
from knowledge_layer.nemo_retriever.local_adapter import NemoRetrieverLocalRetriever
from knowledge_layer.register import KnowledgeRetrievalConfig
from knowledge_layer.register import _setup_backend
from PIL import Image
from pydantic import SecretStr

from aiq_agent.knowledge import JobState
from aiq_agent.knowledge.factory import is_ingestor_registered
from aiq_agent.knowledge.factory import is_retriever_registered
from aiq_agent.knowledge.factory import release_ingestor
from aiq_agent.knowledge.schema import FileStatus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_DIR = PROJECT_ROOT / "environments" / "nemo_retriever_local"
NRL_REVISION = "c80f4a5189ee10b98cbdb93e2f853ceb7b699c3b"  # pragma: allowlist secret


class FakeNotFound(LookupError):
    pass


class FakeConflict(RuntimeError):
    pass


class FakeInvalidRequest(RuntimeError):
    pass


class FakeLockTimeout(RuntimeError):
    pass


class FakeModel:
    def __init__(self, **values: Any):
        self._values = values
        self.__dict__.update(values)

    def model_dump(self) -> dict[str, Any]:
        return dict(self._values)


class FakeLock:
    acquired_paths: set[str] = set()

    def __init__(self, path: str, *, thread_local: bool):
        assert thread_local is False
        self.path = path
        self.acquired = False

    def acquire(self, timeout: float) -> None:
        assert timeout == 0
        if self.path in self.acquired_paths:
            raise FakeLockTimeout
        self.acquired_paths.add(self.path)
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.acquired_paths.remove(self.path)
            self.acquired = False


class AlwaysLocked(FakeLock):
    def acquire(self, timeout: float) -> None:
        raise FakeLockTimeout


class FakeVDB:
    def __init__(self, **kwargs: Any):
        self.constructor_kwargs = kwargs
        self.collections: dict[str, dict[str, Any]] = {}
        self.documents: dict[tuple[str, str], dict[str, Any]] = {}
        self.reconcile_calls = 0
        self.hits: list[dict[str, Any]] = []

    def reconcile_collections(self) -> dict[str, int]:
        self.reconcile_calls += 1
        return {"successes": 0, "failures": 0}

    def create_collection(self, *, scope: str, request: FakeModel) -> FakeModel:
        if request.name in self.collections:
            raise FakeConflict
        now = "2026-08-13T12:00:00+00:00"
        item = {
            "name": request.name,
            "scope": scope,
            "status": "active",
            "description": request.description,
            "metadata": request.metadata,
            "created_at": now,
            "updated_at": now,
            "expires_at": request.expires_at,
        }
        self.collections[request.name] = item
        return FakeModel(**item)

    def get_collection(self, *, scope: str, collection_name: str) -> FakeModel:
        try:
            item = self.collections[collection_name]
        except KeyError as error:
            raise FakeNotFound from error
        assert item["scope"] == scope
        return FakeModel(**item)

    def list_collections(self, *, scope: str, limit: int, continuation_token: str | None) -> FakeModel:
        assert limit == 100
        assert continuation_token is None
        return FakeModel(
            items=[FakeModel(**item) for item in self.collections.values() if item["scope"] == scope],
            next_token=None,
        )

    def delete_collection(self, *, scope: str, collection_name: str, if_exists: bool) -> FakeModel:
        assert if_exists
        item = self.collections.pop(collection_name, None)
        self.documents = {key: value for key, value in self.documents.items() if key[0] != collection_name}
        return FakeModel(
            name=collection_name,
            scope=scope,
            existed=item is not None,
            deleted=item is not None,
            status="deleted",
            cleanup_pending=False,
        )

    def write_collection(self, records: list[Any], *, context: FakeModel) -> FakeModel:
        written = sum(len(batch) for batch in records) if records and isinstance(records[0], list) else len(records)
        now = "2026-08-13T12:02:00+00:00"
        if written:
            self.documents[(context.collection_name, context.document_id)] = {
                "document_id": context.document_id,
                "collection_name": context.collection_name,
                "scope": context.scope,
                "filename": context.filename,
                "content_sha256": context.content_sha256,
                "document_version": context.document_version,
                "status": "completed",
                "chunk_count": written,
                "job_id": context.job_id,
                "created_at": now,
                "updated_at": now,
                "error": None,
            }
        return FakeModel(written=written, total_rows=len(self.documents))

    def list_documents(
        self,
        *,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> FakeModel:
        assert limit == 100
        assert continuation_token is None
        items = [
            FakeModel(**item)
            for (name, _document_id), item in self.documents.items()
            if name == collection_name and item["scope"] == scope
        ]
        return FakeModel(items=items, next_token=None)

    def get_document(self, *, scope: str, collection_name: str, document_id: str) -> FakeModel:
        try:
            item = self.documents[(collection_name, document_id)]
        except KeyError as error:
            raise FakeNotFound from error
        assert item["scope"] == scope
        return FakeModel(**item)

    def delete_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> FakeModel:
        assert if_exists
        existed = self.documents.pop((collection_name, document_id), None) is not None
        return FakeModel(
            document_id=document_id,
            collection_name=collection_name,
            scope=scope,
            existed=existed,
            deleted=existed,
            status="deleted",
            cleanup_pending=False,
        )

    def health(self) -> dict[str, Any]:
        return {"collections": len(self.collections)}


class FakeIngestOperator:
    def __init__(self, *, vdb: FakeVDB):
        self.vdb = vdb

    def run(self, data: Any, **kwargs: Any) -> FakeModel:
        records = (
            data if data == [] or (isinstance(data, list) and data and isinstance(data[0], list)) else [[{"row": 1}]]
        )
        return self.vdb.write_collection(records, context=kwargs["collection_context"])


class FakeRetrieveOperator:
    def __init__(self, *, vdb: FakeVDB):
        self.vdb = vdb

    def run(self, vectors: Any, **kwargs: Any) -> tuple[list[list[dict[str, Any]]], list[str]]:
        assert vectors == [[0.1, 0.2]]
        assert kwargs["query_texts"]
        return ([self.vdb.hits], ["dense"])


class FakeGraphIngestor:
    def __init__(self, state: SimpleNamespace):
        self.state = state
        self.filename = ""

    def buffers(self, value: tuple[str, Any]) -> FakeGraphIngestor:
        self.filename = value[0]
        self.state.buffer_names.append(value[0])
        return self

    def extract(self, params: FakeModel, **kwargs: Any) -> FakeGraphIngestor:
        self.state.extract_params.append(params.model_dump())
        self.state.extract_call_kwargs.append(kwargs)
        return self

    def embed(self, params: FakeModel) -> FakeGraphIngestor:
        self.state.embed_params.append(params.model_dump())
        return self

    def ingest(self) -> list[dict[str, Any]]:
        self.state.ingest_started.set()
        self.state.ingest_gate.wait(timeout=5)
        if self.filename.startswith("bad"):
            raise RuntimeError(f"failed with secret {self.state.secret}")
        if self.filename.startswith("empty"):
            return []
        return [{"text": "chunk", "path": self.filename}]


def _bindings(*, locked: bool = False, gate_open: bool = True) -> tuple[SimpleNamespace, SimpleNamespace]:
    state = SimpleNamespace(
        run_modes=[],
        buffer_names=[],
        extract_params=[],
        extract_call_kwargs=[],
        embed_params=[],
        plan_requests=[],
        sidecar_calls=[],
        query_calls=[],
        ingest_started=threading.Event(),
        ingest_gate=threading.Event(),
        secret="top-secret",  # pragma: allowlist secret
    )
    if gate_open:
        state.ingest_gate.set()

    def create_ingestor(*, run_mode: str) -> FakeGraphIngestor:
        state.run_modes.append(run_mode)
        return FakeGraphIngestor(state)

    def infer_microservice(data: list[str], **kwargs: Any) -> list[list[float]]:
        state.query_calls.append((data, kwargs))
        return [[0.1, 0.2]]

    def resolve_ingest_plan(request: Any) -> Any:
        state.plan_requests.append(request)
        extract_values = {"profile_marker": request.source.profile}
        if request.extract.extract_api_key is not None:
            extract_values["api_key"] = request.extract.extract_api_key
        embed_values = {"model_name": request.embed.embed_model_name}
        return SimpleNamespace(
            create_kwargs={"run_mode": request.runtime.run_mode},
            extract_params=FakeModel(**extract_values),
            embed_params=FakeModel(**embed_values),
            split_config=None,
            extract_call_kwargs=lambda: {},
        )

    def sidecar(records: Any, **kwargs: Any) -> Any:
        state.sidecar_calls.append(kwargs)
        records[0][0].setdefault("metadata", {}).setdefault("content_metadata", {}).update(
            {key: kwargs["meta_df"][0][key] for key in kwargs["meta_fields"]}
        )
        return records

    bindings = SimpleNamespace(
        create_ingestor=create_ingestor,
        LanceDB=FakeVDB,
        IngestVdbOperator=FakeIngestOperator,
        RetrieveVdbOperator=FakeRetrieveOperator,
        CollectionWriteContext=FakeModel,
        CollectionCreateRequest=FakeModel,
        IngestOperation=SimpleNamespace(APPEND="append"),
        VDBInvalidRequest=FakeInvalidRequest,
        VDBResourceNotFound=FakeNotFound,
        VDBResourceConflict=FakeConflict,
        IngestPlanRequest=FakeModel,
        IngestSourceOptions=FakeModel,
        IngestRuntimeOptions=FakeModel,
        IngestExtractOptions=FakeModel,
        IngestEmbedOptions=FakeModel,
        resolve_ingest_plan=resolve_ingest_plan,
        resolve_embed_model=lambda model: model or "nvidia/upstream-default",
        resolve_remote_api_key=lambda explicit=None: explicit or state.secret,
        infer_microservice=infer_microservice,
        default_embed_endpoint="https://upstream.default/v1/embeddings",
        to_client_vdb_records=lambda _data: [
            [
                {
                    "document_type": "text",
                    "metadata": {
                        "content": "chunk",
                        "embedding": [0.1, 0.2],
                        "content_metadata": {},
                        "source_metadata": {"source_name": state.buffer_names[-1]},
                    },
                }
            ]
        ],
        apply_sidecar_metadata_to_client_batches=sidecar,
        pandas=SimpleNamespace(DataFrame=lambda rows: rows),
        FileLock=AlwaysLocked if locked else FakeLock,
        FileLockTimeout=FakeLockTimeout,
    )
    return bindings, state


def _config(tmp_path: Path, bindings: Any, **overrides: Any) -> dict[str, Any]:
    return {
        "data_dir": str(tmp_path / "nrl"),
        "scope": "local",
        "profile": "auto",
        "inference_api_key": SecretStr("top-secret"),  # pragma: allowlist secret
        "collection_ttl_hours": 24,
        "_bindings": bindings,
        **overrides,
    }


def _wait_terminal(ingestor: NemoRetrieverLocalIngestor, job_id: str) -> Any:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = ingestor.get_job_status(job_id)
        if status.is_terminal:
            return status
        time.sleep(0.01)
    raise AssertionError("local ingestion job did not finish")


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_registration_is_separate_and_missing_dependencies_are_lazy(tmp_path, monkeypatch):
    assert is_ingestor_registered("nemo_retriever_local")
    assert is_retriever_registered("nemo_retriever_local")
    assert is_ingestor_registered("nemo_retriever")
    assert is_retriever_registered("nemo_retriever")

    def missing() -> Any:
        raise NemoRetrieverLocalDependencyError("isolated Python 3.12 environment")

    monkeypatch.setattr(_local_client, "_load_nrl_bindings", missing)
    with pytest.raises(NemoRetrieverLocalDependencyError, match="Python 3.12"):
        NemoRetrieverLocalIngestor({"data_dir": str(tmp_path), "scope": "local", "profile": "auto"})


def test_missing_pinned_nrl_embedding_default_has_actionable_dependency_error(monkeypatch):
    operator_module = pytest.importorskip("nemo_retriever.operators.embed.cpu_operator")
    monkeypatch.delattr(operator_module._BatchEmbedCPUActor, "DEFAULT_EMBED_INVOKE_URL")

    with pytest.raises(
        NemoRetrieverLocalDependencyError,
        match=r"uv sync --project environments/nemo_retriever_local --frozen",
    ):
        _local_client._load_nrl_bindings()


def test_nrl_tls_environment_boolean_is_strict(monkeypatch):
    monkeypatch.setenv("NRL_VERIFY_SSL", "false")
    assert service_adapter._settings({"scope": "scope"}).verify_ssl is False

    monkeypatch.setenv("NRL_VERIFY_SSL", "enabled")
    with pytest.raises(ValueError, match="NRL_VERIFY_SSL must be a boolean"):
        service_adapter._settings({"scope": "scope"})


def test_local_backend_config_is_distinct_and_secret_safe(monkeypatch):
    monkeypatch.delenv("NRL_LOCAL_DATA_DIR", raising=False)
    monkeypatch.delenv("NRL_LOCAL_PROFILE", raising=False)
    secret = "local-inference-secret"  # pragma: allowlist secret
    config = KnowledgeRetrievalConfig(
        backend="nemo_retriever_local",
        backend_config={"scope": "local", "inference_api_key": secret},
    )

    backend, backend_config = _setup_backend(config)
    settings = LocalSettings.from_config(backend_config)

    assert backend == "nemo_retriever_local"
    assert settings.data_dir.name == "nemo_retriever"
    assert settings.profile == "auto"
    assert settings.scope == "local"
    assert isinstance(config.backend_config["inference_api_key"], SecretStr)
    assert isinstance(backend_config["inference_api_key"], SecretStr)
    assert isinstance(settings.inference_api_key, SecretStr)
    assert secret not in repr(config)
    assert secret not in repr(backend_config)
    assert is_retriever_registered("nemo_retriever_local")
    assert is_ingestor_registered("nemo_retriever_local")


def test_local_backend_requires_scope_and_known_upstream_profile(monkeypatch):
    monkeypatch.delenv("NRL_SCOPE", raising=False)
    monkeypatch.delenv("NRL_LOCAL_PROFILE", raising=False)
    assert LocalSettings.from_config({}).scope == "local"
    with pytest.raises(ValueError, match="explicit nrl_scope"):
        LocalSettings.from_config({"scope": " "})
    with pytest.raises(ValueError, match="auto.*fast-text"):
        LocalSettings.from_config({"scope": "local", "profile": "custom"})
    with pytest.raises(ValueError, match="Unsupported nemo_retriever_local backend_config option.*typo"):
        KnowledgeRetrievalConfig(backend="nemo_retriever_local", backend_config={"typo": True})


def test_release_ingestor_only_evicts_the_expected_cached_instance(monkeypatch):
    from aiq_agent.knowledge import factory

    expected = object()
    replacement = object()
    monkeypatch.setitem(factory._INGESTOR_INSTANCES, "nemo_retriever_local", replacement)

    assert release_ingestor("nemo_retriever_local", expected) is False
    assert factory._INGESTOR_INSTANCES["nemo_retriever_local"] is replacement
    assert release_ingestor("nemo_retriever_local", replacement) is True
    assert "nemo_retriever_local" not in factory._INGESTOR_INSTANCES


def test_local_environment_is_python_312_and_pins_unmodified_nrl() -> None:
    project = _read_toml(ENVIRONMENT_DIR / "pyproject.toml")

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert "nemo-retriever" in project["project"]["dependencies"]
    assert "filelock>=3.20.3,<4" in project["project"]["dependencies"]
    assert project["tool"]["uv"]["sources"]["nemo-retriever"] == {
        "git": "https://github.com/NVIDIA/NeMo-Retriever.git",
        "rev": NRL_REVISION,
        "subdirectory": "nemo_retriever",
    }

    lock = _read_toml(ENVIRONMENT_DIR / "uv.lock")
    packages = {package["name"]: package for package in lock["package"]}
    assert lock["requires-python"] == "==3.12.*"
    locked_nrl_source = packages["nemo-retriever"]["source"]["git"]
    assert f"rev={NRL_REVISION}" in locked_nrl_source
    assert locked_nrl_source.endswith(f"#{NRL_REVISION}")
    assert packages["pypdfium2"]["version"] == "4.30.0"


def test_normal_knowledge_layer_keeps_nrl_isolated_and_accepts_pdfium_430() -> None:
    project = _read_toml(PROJECT_ROOT / "sources" / "knowledge_layer" / "pyproject.toml")

    assert "pypdfium2>=4.30.0,!=4.30.1,<6" in project["project"]["optional-dependencies"]["llamaindex"]
    all_requirements = project["project"]["optional-dependencies"]["all"]
    assert not any("nemo-retriever" in requirement for requirement in all_requirements)


def test_local_profile_selects_embedded_backend_and_upstream_auto_defaults() -> None:
    config_path = PROJECT_ROOT / "configs" / "config_web_nemo_retriever_local.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    functions = config["functions"]
    knowledge = config["functions"]["knowledge_search"]
    local_llm = config["llms"]["agent_llm"]

    assert config["general"]["use_uvloop"] is False
    assert config["general"]["front_end"]["dask_workers"] == "threads"
    assert config["workflow"]["use_async_deep_research"] is False
    sources = {source["id"]: source for source in functions["data_sources"]["sources"]}
    assert sources["web_search"]["tools"] == ["web_search_tool", "advanced_web_search_tool"]
    assert sources["knowledge_layer"]["tools"] == ["knowledge_search"]
    assert functions["web_search_tool"] == {
        "_type": "tavily_web_search",
        "max_results": 5,
        "max_content_length": 1000,
    }
    assert functions["advanced_web_search_tool"] == {
        "_type": "tavily_web_search",
        "max_results": 2,
        "advanced_search": True,
    }
    assert functions["shallow_research_agent"]["exclude_tools"] == ["advanced_web_search_tool"]
    assert functions["deep_research_agent"]["exclude_tools"] == ["web_search_tool"]
    assert local_llm["_type"] == "openai"
    assert local_llm["model_name"] == "${AIQ_AGENT_LLM_MODEL:-openai/local-tool-model}"
    assert local_llm["base_url"] == "${AIQ_AGENT_LLM_BASE_URL:-http://127.0.0.1:1234/v1}"
    assert local_llm["api_key"] == "${AIQ_AGENT_LLM_API_KEY:-local}"
    assert local_llm["max_tokens"] == 32768
    assert "parallel_tool_calls" not in local_llm
    for function in functions.values():
        if not isinstance(function, dict):
            continue
        for role in (
            "llm",
            "orchestrator_llm",
            "source_router_llm",
            "researcher_llm",
            "planner_llm",
            "writer_llm",
        ):
            if role in function:
                assert function[role] == "agent_llm"
    assert knowledge["backend"] == "nemo_retriever_local"
    assert knowledge["backend_config"]["scope"] == "${NRL_SCOPE:-local}"
    assert knowledge["backend_config"]["data_dir"] == "${NRL_LOCAL_DATA_DIR:-.aiq-data/nemo_retriever}"
    assert knowledge["backend_config"]["profile"] == "${NRL_LOCAL_PROFILE:-auto}"
    assert knowledge["generate_summary"] is False


def test_nemo_retriever_profiles_only_keep_supported_verbose_setting() -> None:
    for filename in ("config_web_nemo_retriever.yml", "config_web_nemo_retriever_local.yml"):
        config = yaml.safe_load((PROJECT_ROOT / "configs" / filename).read_text(encoding="utf-8"))
        functions = config["functions"]

        assert "verbose" not in functions["intent_classifier"]
        assert "verbose" not in functions["clarifier_agent"]
        assert functions["shallow_research_agent"]["verbose"] is True
        assert "verbose" not in functions["deep_research_agent"]
        assert "verbose" not in config["workflow"]


def test_nemo_retriever_shallow_profiles_pin_citation_policy() -> None:
    for filename in ("config_web_nemo_retriever.yml", "config_web_nemo_retriever_local.yml"):
        config = yaml.safe_load((PROJECT_ROOT / "configs" / filename).read_text(encoding="utf-8"))
        shallow_agent = config["functions"]["shallow_research_agent"]

        assert shallow_agent["max_llm_turns"] == 10
        assert shallow_agent["max_tool_iterations"] == 5
        assert shallow_agent["enforce_citations"] is False


def test_pdf_image_extraction_supports_pdfium_4_and_5(tmp_path) -> None:
    pdf_path = tmp_path / "embedded-image.pdf"
    Image.new("RGB", (320, 240), color=(28, 92, 164)).save(pdf_path, format="PDF")

    images = _extract_images_from_pdf(str(pdf_path), min_width=1, min_height=1)

    assert len(images) == 1
    assert images[0]["page_number"] == 1
    assert images[0]["width"] == 320
    assert images[0]["height"] == 240
    assert images[0]["image_bytes"].startswith(b"\xff\xd8")


def test_pinned_nrl_auto_profile_contract_when_isolated_environment_is_installed():
    params_module = pytest.importorskip("nemo_retriever.common.params")
    plan_module = pytest.importorskip("nemo_retriever.ingest.plan")
    ray = pytest.importorskip("ray")

    assert plan_module.profile_extract_defaults("auto") == {}
    defaults = params_module.ExtractParams()
    assert defaults.extract_text is True
    assert defaults.extract_images is True
    assert defaults.extract_tables is True
    assert defaults.extract_charts is True
    assert defaults.extract_page_as_image is True
    assert defaults.use_page_elements is True
    assert defaults.use_table_structure is False
    assert "embed_modality" in params_module.EmbedParams.model_fields
    assert ray.is_initialized() is False


def test_pinned_nrl_plans_pptx_through_document_pdf_branch_for_both_profiles(tmp_path):
    plan_module = pytest.importorskip("nemo_retriever.ingest.plan")
    presentation = tmp_path / "slides.pptx"
    presentation.touch()

    for profile in ("auto", "fast-text"):
        plan = plan_module.resolve_ingest_plan(
            plan_module.IngestPlanRequest(
                source=plan_module.IngestSourceOptions(
                    documents=[str(presentation)],
                    profile=profile,
                    input_type="auto",
                ),
                runtime=plan_module.IngestRuntimeOptions(run_mode="inprocess"),
                extract=plan_module.IngestExtractOptions(),
                embed=plan_module.IngestEmbedOptions(),
            )
        )

        assert [(branch.spec.family, branch.spec.extraction_mode) for branch in plan.branches] == [("pdf", "pdf")]
        assert plan.branches[0].input_paths == (str(presentation),)


def test_settings_restrict_profile_and_redact_secret(tmp_path):
    settings = LocalSettings.from_config(
        {
            "data_dir": str(tmp_path),
            "scope": "local",
            "profile": "auto",
            "inference_api_key": SecretStr("do-not-render"),
        }
    )
    assert "do-not-render" not in repr(settings)
    assert isinstance(settings.inference_api_key, SecretStr)
    with pytest.raises(ValueError, match="auto.*fast-text"):
        LocalSettings.from_config({"data_dir": str(tmp_path), "scope": "local", "profile": "custom"})


def test_local_environment_and_explicit_backend_config_are_equivalent(tmp_path, monkeypatch):
    values = {
        "NRL_LOCAL_DATA_DIR": str(tmp_path / "nrl"),
        "NRL_SCOPE": "workspace",
        "NRL_LOCAL_PROFILE": "fast-text",
        "NRL_PAGE_ELEMENTS_INVOKE_URL": "https://page.example/v1/infer",
        "NRL_OCR_INVOKE_URL": "https://ocr.example/v1/infer",
        "NRL_TABLE_STRUCTURE_INVOKE_URL": "https://table.example/v1/infer",
        "NRL_EMBED_INVOKE_URL": "https://embed.example/v1/embeddings",
        "NRL_EMBED_MODEL_NAME": "embed-model",
        "NRL_EMBED_MODEL_PROVIDER_PREFIX": "provider",
        "NRL_INFERENCE_API_KEY": "environment-secret",  # pragma: allowlist secret
        "NRL_COLLECTION_TTL_HOURS": "48",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    from_environment = LocalSettings.from_config({})
    explicit = LocalSettings.from_config(
        {
            "data_dir": values["NRL_LOCAL_DATA_DIR"],
            "scope": values["NRL_SCOPE"],
            "profile": values["NRL_LOCAL_PROFILE"],
            "page_elements_invoke_url": values["NRL_PAGE_ELEMENTS_INVOKE_URL"],
            "ocr_invoke_url": values["NRL_OCR_INVOKE_URL"],
            "table_structure_invoke_url": values["NRL_TABLE_STRUCTURE_INVOKE_URL"],
            "embed_invoke_url": values["NRL_EMBED_INVOKE_URL"],
            "embed_model_name": values["NRL_EMBED_MODEL_NAME"],
            "embed_model_provider_prefix": values["NRL_EMBED_MODEL_PROVIDER_PREFIX"],
            "inference_api_key": SecretStr(values["NRL_INFERENCE_API_KEY"]),
            "collection_ttl_hours": values["NRL_COLLECTION_TTL_HOURS"],
        }
    )
    public_config = KnowledgeRetrievalConfig(backend="nemo_retriever_local")
    public_settings = LocalSettings.from_config(public_config.backend_config)

    assert from_environment.compatibility_key == explicit.compatibility_key
    assert public_settings.compatibility_key == from_environment.compatibility_key
    assert from_environment.data_dir == explicit.data_dir
    assert values["NRL_INFERENCE_API_KEY"] not in repr(public_config)
    assert "environment-secret" not in repr(from_environment)


def test_local_cleanup_boolean_is_strict(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="cleanup_files must be a boolean"):
        ingestor.submit_job([str(source)], "reports", {"cleanup_files": "sometimes"})

    ingestor.close()


def test_runtime_matches_vectordb_construction_and_process_lock(tmp_path):
    bindings, _state = _bindings()
    abandoned = tmp_path / "nrl" / ".staging" / "interrupted" / "upload"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_text("process-local", encoding="utf-8")
    runtime = LocalRuntime(LocalSettings.from_config(_config(tmp_path, bindings)), bindings)
    assert runtime.vdb.constructor_kwargs == {
        "uri": str(tmp_path / "nrl" / "lancedb"),
        "table_name": "nemo_retriever",
        "vector_dim": None,
        "overwrite": False,
        "build_index": False,
        "_service_table_schema": True,
        "expiration_cleanup_enabled": True,
    }
    assert runtime.vdb.reconcile_calls == 1
    assert not abandoned.exists()
    runtime.close()

    locked_bindings, _state = _bindings(locked=True)
    with pytest.raises(NemoRetrieverLocalLockError, match="already open by another process"):
        LocalRuntime(LocalSettings.from_config(_config(tmp_path, locked_bindings)), locked_bindings)


def test_runtime_health_check_contains_backend_failures(tmp_path):
    bindings, _state = _bindings()
    runtime = LocalRuntime(LocalSettings.from_config(_config(tmp_path, bindings)), bindings)
    try:
        runtime.vdb.health = lambda: {"catalog": {"healthy": True}}
        assert runtime.health_check() is True

        runtime.vdb.health = lambda: {"catalog": {"healthy": False}}
        assert runtime.health_check() is False

        runtime.vdb.health = lambda: []
        assert runtime.health_check() is False

        def fail_health() -> Any:
            raise RuntimeError("backend unavailable")

        runtime.vdb.health = fail_health
        assert runtime.health_check() is False

        runtime.close()
        assert runtime.health_check() is False
    finally:
        runtime.close()


def test_last_release_waits_for_lock_release_without_blocking_other_directories(tmp_path, monkeypatch):
    bindings, _state = _bindings()
    config = _config(tmp_path, bindings)
    first = NemoRetrieverLocalIngestor(config)
    first_runtime = first._runtime
    original_close = first_runtime.close
    close_started = threading.Event()
    allow_close = threading.Event()

    def delayed_close() -> None:
        close_started.set()
        assert allow_close.wait(timeout=5)
        original_close()

    monkeypatch.setattr(first_runtime, "close", delayed_close)
    close_thread = threading.Thread(target=first.close)
    close_thread.start()
    assert close_started.wait(timeout=2)

    closing = _local_client._RUNTIMES_CLOSING[first._runtime_handle._key]
    wait_started = threading.Event()
    original_wait = closing.wait

    def observed_wait(timeout: float | None = None) -> bool:
        wait_started.set()
        return original_wait(timeout)

    monkeypatch.setattr(closing, "wait", observed_wait)

    replacement: list[Any] = []
    replacement_ready = threading.Event()

    def reacquire() -> None:
        try:
            replacement.append(NemoRetrieverLocalIngestor(config))
        except Exception as error:  # noqa: BLE001 - asserted below
            replacement.append(error)
        finally:
            replacement_ready.set()

    acquire_thread = threading.Thread(target=reacquire)
    acquire_thread.start()
    assert wait_started.wait(timeout=2)

    other_bindings, _other_state = _bindings()
    other = NemoRetrieverLocalIngestor(_config(tmp_path, other_bindings, data_dir=str(tmp_path / "other-nrl")))
    assert replacement_ready.is_set() is False
    other.close()

    allow_close.set()
    close_thread.join(timeout=5)
    acquire_thread.join(timeout=5)
    assert not close_thread.is_alive()
    assert not acquire_thread.is_alive()
    assert len(replacement) == 1
    assert isinstance(replacement[0], NemoRetrieverLocalIngestor)
    assert replacement[0]._runtime is not first_runtime

    replacement_runtime = replacement[0]._runtime
    replacement_close = replacement_runtime.close
    close_calls = 0
    close_calls_lock = threading.Lock()

    def counted_close() -> None:
        nonlocal close_calls
        with close_calls_lock:
            close_calls += 1
        replacement_close()

    monkeypatch.setattr(replacement_runtime, "close", counted_close)
    duplicate_closes = [threading.Thread(target=replacement[0].close) for _position in range(2)]
    for thread in duplicate_closes:
        thread.start()
    for thread in duplicate_closes:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in duplicate_closes)
    assert close_calls == 1


def test_runtime_startup_failure_stops_worker_cleans_staging_and_releases_lock(tmp_path, monkeypatch):
    bindings, _state = _bindings()
    data_dir = tmp_path / "nrl"
    original_start = threading.Thread.start
    started_threads: list[threading.Thread] = []

    def fail_reconciler_start(thread: threading.Thread) -> None:
        if thread.name == "nemo-retriever-local-reconciler":
            raise RuntimeError("reconciler start failed")
        original_start(thread)
        started_threads.append(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_reconciler_start)
    with pytest.raises(RuntimeError, match="reconciler start failed"):
        LocalRuntime(LocalSettings.from_config(_config(tmp_path, bindings)), bindings)

    assert len(started_threads) == 1
    assert started_threads[0].name == "nemo-retriever-local-worker"
    assert not started_threads[0].is_alive()
    assert not (data_dir / ".staging").exists()

    monkeypatch.setattr(threading.Thread, "start", original_start)
    replacement = LocalRuntime(LocalSettings.from_config(_config(tmp_path, bindings)), bindings)
    replacement.close()


def test_collection_listing_and_get_hide_inactive_and_expired_rows(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("active")
    ingestor.create_collection("deleting")
    ingestor.create_collection("expired")
    ingestor._runtime.vdb.collections["deleting"]["status"] = "deleting"
    ingestor._runtime.vdb.collections["expired"]["expires_at"] = "2020-01-01T00:00:00+00:00"

    assert [item.name for item in ingestor.list_collections()] == ["active"]
    assert ingestor.get_collection("active") is not None
    assert ingestor.get_collection("deleting") is None
    assert ingestor.get_collection("expired") is None
    assert ingestor._runtime._assert_owned_collection("deleting")["status"] == "deleting"
    ingestor.close()


def test_collection_listing_and_get_tolerate_reconciliation_transition(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")

    def transitioned(**_kwargs: Any) -> FakeModel:
        raise FakeInvalidRequest("Collection 'reports' is deleting")

    ingestor._runtime.vdb.list_documents = transitioned

    assert ingestor.list_collections() == []
    assert ingestor.get_collection("reports") is None
    ingestor.close()


def test_local_collection_and_document_pagination_produce_exact_counts(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("first")
    ingestor.create_collection("second")
    now = "2026-08-13T12:02:00+00:00"
    for position, chunks in enumerate((2, 3)):
        document_id = f"doc-{position}"
        ingestor._runtime.vdb.documents[("first", document_id)] = {
            "document_id": document_id,
            "collection_name": "first",
            "scope": "local",
            "filename": f"file-{position}.pdf",
            "content_sha256": f"sha-{position}",
            "document_version": f"v-{position}",
            "status": "completed",
            "chunk_count": chunks,
            "job_id": "job",
            "created_at": now,
            "updated_at": now,
            "error": None,
        }

    def collection_page(*, scope: str, limit: int, continuation_token: str | None):
        assert scope == "local"
        assert limit == 100
        names = ["first"] if continuation_token is None else ["second"]
        return FakeModel(
            items=[FakeModel(**ingestor._runtime.vdb.collections[name]) for name in names],
            next_token="collections-page-2" if continuation_token is None else None,
        )

    def document_page(*, scope: str, collection_name: str, limit: int, continuation_token: str | None):
        assert scope == "local"
        assert limit == 100
        documents = [
            item for (name, _document_id), item in ingestor._runtime.vdb.documents.items() if name == collection_name
        ]
        if collection_name == "first":
            page = documents[:1] if continuation_token is None else documents[1:]
            next_token = "documents-page-2" if continuation_token is None else None
        else:
            page = documents
            next_token = None
        return FakeModel(items=[FakeModel(**item) for item in page], next_token=next_token)

    ingestor._runtime.vdb.list_collections = collection_page
    ingestor._runtime.vdb.list_documents = document_page

    collections = ingestor.list_collections()
    assert [item.name for item in collections] == ["first", "second"]
    assert collections[0].file_count == 2
    assert collections[0].chunk_count == 5
    assert [item.file_id for item in ingestor.list_files("first")] == ["doc-0", "doc-1"]
    ingestor.close()


def test_submit_is_non_blocking_stable_and_uses_exact_upstream_profile(tmp_path):
    bindings, state = _bindings(gate_open=False)
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports", metadata={"team": "research", "table_name": "hidden"})
    source = tmp_path / "temporary-upload"
    source.write_text("same bytes", encoding="utf-8")

    job_id = ingestor.submit_job(
        [str(source)],
        "reports",
        {
            "original_filenames": ["report.pdf"],
            "cleanup_files": True,
            "metadata": {"category": "finance"},
        },
    )
    pending = ingestor.get_job_status(job_id)
    assert pending.file_details[0].file_id
    assert pending.file_details[0].file_name == "report.pdf"
    assert not source.exists()
    assert state.ingest_started.wait(timeout=2)
    assert not pending.is_terminal

    state.ingest_gate.set()
    completed = _wait_terminal(ingestor, job_id)
    assert completed.status == JobState.COMPLETED
    assert completed.file_details[0].status == FileStatus.SUCCESS
    assert state.run_modes == ["inprocess"]
    assert state.buffer_names == ["report.pdf"]
    assert state.extract_params == [  # pragma: allowlist secret
        {"profile_marker": "auto", "api_key": "top-secret"}  # pragma: allowlist secret
    ]
    assert state.embed_params[0]["model_name"] == "nvidia/upstream-default"
    assert "embed_modality" not in state.embed_params[0]
    assert state.sidecar_calls[0]["meta_fields"] == ["category"]
    assert state.sidecar_calls[0]["meta_source_field"] == "__aiq_source"
    assert not list((tmp_path / "nrl" / ".staging").rglob("*"))

    retry_source = tmp_path / "retry.pdf"
    retry_source.write_text("same bytes", encoding="utf-8")
    retry_id = ingestor.submit_job([str(retry_source)], "reports", {"original_filenames": ["report.pdf"]})
    retried = _wait_terminal(ingestor, retry_id)
    assert retried.file_details[0].file_id == completed.file_details[0].file_id
    assert len(ingestor.list_files("reports")) == 1
    collection = ingestor.get_collection("reports")
    assert collection is not None
    assert collection.file_count == 1
    assert collection.chunk_count == 1
    assert "table_name" not in collection.metadata
    ingestor.close()


def test_document_identity_survives_retry_at_a_different_batch_position(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    other = tmp_path / "other.pdf"
    report = tmp_path / "report.pdf"
    retry = tmp_path / "retry-copy.pdf"
    other.write_text("other bytes", encoding="utf-8")
    report.write_text("stable bytes", encoding="utf-8")
    retry.write_text("stable bytes", encoding="utf-8")

    first_job = ingestor.submit_job(
        [str(other), str(report)],
        "reports",
        {"original_filenames": ["other.pdf", "report.pdf"]},
    )
    first_status = _wait_terminal(ingestor, first_job)
    original_id = next(item.file_id for item in first_status.file_details if item.file_name == "report.pdf")

    retry_job = ingestor.submit_job(
        [str(retry)],
        "reports",
        {"original_filenames": ["report.pdf"]},
    )
    retry_status = _wait_terminal(ingestor, retry_job)

    assert retry_status.file_details[0].file_id == original_id
    assert len(ingestor.list_files("reports")) == 2
    ingestor.close()


def test_duplicate_logical_documents_in_one_batch_are_rejected_and_cleaned(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    first = tmp_path / "first.pdf"
    duplicate = tmp_path / "duplicate.pdf"
    first.write_text("same bytes", encoding="utf-8")
    duplicate.write_text("same bytes", encoding="utf-8")

    with pytest.raises(ValueError, match="same logical document more than once"):
        ingestor.submit_job(
            [str(first), str(duplicate)],
            "reports",
            {"original_filenames": ["report.pdf", "report.pdf"]},
        )

    assert not list((tmp_path / "nrl" / ".staging").rglob("*"))
    ingestor.close()


def test_delete_rejects_queued_document_then_succeeds_after_ingestion(tmp_path):
    bindings, state = _bindings(gate_open=False)
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")

    job_id = ingestor.submit_job([str(source)], "reports")
    document_id = ingestor.get_job_status(job_id).file_details[0].file_id
    assert state.ingest_started.wait(timeout=2)

    with pytest.raises(NemoRetrieverLocalError, match="still being ingested"):
        ingestor.delete_file(document_id, "reports")

    state.ingest_gate.set()
    assert _wait_terminal(ingestor, job_id).status == JobState.COMPLETED
    assert ingestor.delete_file(document_id, "reports") is True
    assert ingestor.list_files("reports") == []
    ingestor.close()


def test_delete_collection_rejects_active_ingestion_then_succeeds(tmp_path):
    bindings, state = _bindings(gate_open=False)
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")

    job_id = ingestor.submit_job([str(source)], "reports")
    assert state.ingest_started.wait(timeout=2)

    with pytest.raises(NemoRetrieverLocalError, match="still has documents being ingested"):
        ingestor.delete_collection("reports")

    state.ingest_gate.set()
    assert _wait_terminal(ingestor, job_id).status == JobState.COMPLETED
    assert ingestor.delete_collection("reports") is True
    assert ingestor.get_collection("reports") is None
    ingestor.close()


def test_submit_revalidates_collection_after_staging(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")
    admission_lock = threading.Lock()
    admission_lock.acquire()
    ingestor._runtime._submit_lock = admission_lock
    submitted: list[Any] = []

    def submit() -> None:
        try:
            submitted.append(ingestor.submit_job([str(source)], "reports"))
        except Exception as error:  # noqa: BLE001 - assert the public adapter error below
            submitted.append(error)

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    deadline = time.monotonic() + 2
    staging_root = tmp_path / "nrl" / ".staging"
    while time.monotonic() < deadline and not list(staging_root.rglob("*.pdf")):
        time.sleep(0.01)
    assert list(staging_root.rglob("*.pdf")), "submission did not finish private staging"

    ingestor._runtime.vdb.delete_collection(scope="local", collection_name="reports", if_exists=True)
    admission_lock.release()
    submit_thread.join(timeout=5)

    assert not submit_thread.is_alive()
    assert len(submitted) == 1
    assert isinstance(submitted[0], NemoRetrieverLocalError)
    assert not list(staging_root.rglob("*"))
    ingestor.close()


def test_zero_chunk_extraction_fails_without_persisting_a_document(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "empty.pdf"
    source.write_text("no extractable content", encoding="utf-8")

    status = _wait_terminal(ingestor, ingestor.submit_job([str(source)], "reports"))

    assert status.status == JobState.FAILED
    assert status.file_details[0].status == FileStatus.FAILED
    assert "extracted no chunks" in (status.file_details[0].error_message or "")
    assert ingestor.list_files("reports") == []
    ingestor.close()


def test_sidecar_join_key_cannot_be_overwritten_by_user_metadata(tmp_path):
    bindings, state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")

    job_id = ingestor.submit_job(
        [str(source)],
        "reports",
        {
            "metadata": {
                "source": "user-value",
                "__aiq_source": "also-user-value",
                "category": "finance",
            }
        },
    )
    assert _wait_terminal(ingestor, job_id).status == JobState.COMPLETED
    call = state.sidecar_calls[0]
    assert call["meta_source_field"] == "___aiq_source"
    assert call["meta_df"][0]["___aiq_source"] == "report.pdf"
    assert call["meta_df"][0]["source"] == "user-value"
    assert call["meta_df"][0]["__aiq_source"] == "also-user-value"
    assert call["meta_fields"] == ["source", "__aiq_source", "category"]
    ingestor.close()


def test_public_errors_redact_credentials_endpoints_and_physical_storage(tmp_path):
    bindings, _state = _bindings()
    endpoint = "https://endpoint.example.test/v1/embeddings"
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings, embed_invoke_url=endpoint))
    physical_table = f"nrl_{'a' * 40}"
    message = ingestor._runtime.public_error(
        RuntimeError(  # pragma: allowlist secret
            "secret=top-secret "  # pragma: allowlist secret
            f"endpoint={endpoint} table={physical_table} database={tmp_path / 'nrl' / 'lancedb'}"
        )
    )

    assert "top-secret" not in message
    assert endpoint not in message
    assert physical_table not in message
    assert str(tmp_path / "nrl") not in message
    assert "[redacted]" in message
    assert "[redacted-table]" in message
    ingestor.close()


@pytest.mark.parametrize(
    ("nrl_status", "expected"),
    [
        ("pending", FileStatus.UPLOADING),
        ("appending", FileStatus.INGESTING),
        ("replacing", FileStatus.INGESTING),
        ("deleting", FileStatus.INGESTING),
        ("completed", FileStatus.SUCCESS),
        ("failed", FileStatus.FAILED),
    ],
)
def test_document_status_mapping_and_persisted_error_redaction(tmp_path, nrl_status, expected):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    physical_table = f"nrl_{'b' * 40}"
    private_error = (
        "secret=top-secret "  # pragma: allowlist secret
        f"table={physical_table} at {tmp_path / 'nrl' / 'lancedb'}"
    )
    info = ingestor._runtime._file_info(
        {
            "document_id": "doc-1",
            "filename": "report.pdf",
            "collection_name": "reports",
            "status": nrl_status,
            "chunk_count": 1,
            "created_at": "2026-08-13T12:00:00+00:00",
            "updated_at": "2026-08-13T12:01:00+00:00",
            "error": private_error,
        }
    )

    assert info.status == expected
    assert (info.ingested_at is not None) is (expected == FileStatus.SUCCESS)
    assert "top-secret" not in (info.error_message or "")
    assert physical_table not in (info.error_message or "")
    assert str(tmp_path / "nrl") not in (info.error_message or "")
    ingestor.close()


def test_local_crud_and_batch_delete_errors_are_sanitized_at_adapter_boundary(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    physical_table = f"nrl_{'c' * 40}"
    private_error = RuntimeError(  # pragma: allowlist secret
        f"secret=top-secret table={physical_table} database={tmp_path / 'nrl' / 'lancedb'}"  # pragma: allowlist secret
    )

    def fail(**_kwargs: Any) -> Any:
        raise private_error

    ingestor._runtime.vdb.list_collections = fail
    with pytest.raises(NemoRetrieverLocalError) as caught:
        ingestor.list_collections()
    assert "top-secret" not in str(caught.value)
    assert physical_table not in str(caught.value)
    assert str(tmp_path / "nrl") not in str(caught.value)

    ingestor._runtime.vdb.delete_document = fail
    result = ingestor.delete_files(["doc-1"], "reports")
    assert result["successful"] == []
    assert len(result["failed"]) == 1
    serialized = str(result["failed"][0])
    assert "top-secret" not in serialized
    assert physical_table not in serialized
    assert str(tmp_path / "nrl") not in serialized
    ingestor.close()


def test_submit_racing_shutdown_is_rejected_and_staging_is_cleaned(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")

    class GateLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.waiting = threading.Event()
            self.lock.acquire()

        def __enter__(self):
            self.waiting.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    gate = GateLock()
    ingestor._runtime._submit_lock = gate
    submitted: list[Any] = []

    def submit() -> None:
        try:
            submitted.append(ingestor.submit_job([str(source)], "reports"))
        except Exception as error:  # noqa: BLE001 - assertion captures the exact public error below
            submitted.append(error)

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    assert gate.waiting.wait(timeout=2)
    with ingestor._runtime._state_lock:
        ingestor._runtime._closed = True
    gate.lock.release()
    submit_thread.join(timeout=5)

    assert not submit_thread.is_alive()
    assert len(submitted) == 1
    assert isinstance(submitted[0], NemoRetrieverLocalError)
    assert "runtime is closed" in str(submitted[0])
    assert not list((tmp_path / "nrl" / ".staging").rglob("*"))

    # The real close transition must also wait for the same admission lock.
    with ingestor._runtime._state_lock:
        ingestor._runtime._closed = False
    gate.waiting.clear()
    gate.lock.acquire()
    close_thread = threading.Thread(target=ingestor.close)
    close_thread.start()
    assert gate.waiting.wait(timeout=2)
    assert ingestor._runtime._closed is False
    gate.lock.release()
    close_thread.join(timeout=5)
    assert not close_thread.is_alive()


@pytest.mark.parametrize("corruption", ["missing_job", "missing_progress"])
def test_stale_work_is_cleaned_and_does_not_stop_the_next_job(tmp_path, corruption):
    bindings, state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    stale_source = tmp_path / "stale.pdf"
    valid_source = tmp_path / "valid.pdf"
    stale_source.write_text("stale", encoding="utf-8")
    valid_source.write_text("valid", encoding="utf-8")

    # Hold the state lock so the worker cannot inspect the first item until its
    # process-local bookkeeping has been deliberately corrupted.
    with ingestor._runtime._state_lock:
        stale_job_id = ingestor.submit_job([str(stale_source)], "reports")
        valid_job_id = ingestor.submit_job([str(valid_source)], "reports")
        stale_job = ingestor._runtime._jobs[stale_job_id]
        stale_document_id = stale_job.files[0].file_id
        if corruption == "missing_job":
            del ingestor._runtime._jobs[stale_job_id]
        else:
            stale_job.files.clear()

    valid_status = _wait_terminal(ingestor, valid_job_id)
    assert valid_status.status == JobState.COMPLETED
    assert state.buffer_names == ["valid.pdf"]
    with ingestor._runtime._state_lock:
        assert ("reports", stale_document_id) not in ingestor._runtime._active_writes
    assert not list((tmp_path / "nrl" / ".staging").rglob("*"))

    stale_status = ingestor.get_job_status(stale_job_id)
    assert stale_status.status == JobState.FAILED
    if corruption == "missing_progress":
        assert stale_status.is_terminal
        assert stale_status.processed_files == 0
    ingestor.close()


def test_partial_failure_secret_redaction_and_cleanup(tmp_path):
    bindings, state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    good.write_text("good", encoding="utf-8")
    bad.write_text("bad", encoding="utf-8")
    job_id = ingestor.submit_job([str(good), str(bad)], "reports")
    status = _wait_terminal(ingestor, job_id)
    assert status.status == JobState.COMPLETED
    assert status.error_message == "NeMo Retriever ingestion completed with one or more failed files"
    failed = next(item for item in status.file_details if item.status == FileStatus.FAILED)
    assert "top-secret" not in (failed.error_message or "")
    assert "[redacted]" in (failed.error_message or "")
    assert not list((tmp_path / "nrl" / ".staging").rglob("*"))
    assert state.run_modes == ["inprocess", "inprocess"]
    ingestor.close()


def test_query_preserves_distance_rejects_filters_and_uses_upstream_defaults(tmp_path):
    bindings, state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    retriever = NemoRetrieverLocalRetriever(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    ingestor._runtime.vdb.hits = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "text": "answer",
            "distance": -1.75,
            "filename": "report.pdf",
            "page_number": 2,
            "content_type": "text",
            "metadata": {"section": "summary", "physical_table": "private"},
        }
    ]
    result = asyncio.run(retriever.retrieve("question", "reports", top_k=3))
    assert result.success
    assert result.chunks[0].distance == -1.75
    assert result.chunks[0].score == 0
    assert "physical_table" not in result.chunks[0].metadata
    query_kwargs = state.query_calls[0][1]
    assert query_kwargs["embedding_endpoint"] == "https://upstream.default/v1/embeddings"
    assert query_kwargs["input_type"] == "query"
    assert query_kwargs["nvidia_api_key"] == "top-secret"  # pragma: allowlist secret

    filtered = asyncio.run(retriever.retrieve("question", "reports", filters={"team": "research"}))
    assert not filtered.success
    assert "filters are not supported" in (filtered.error_message or "")
    retriever.close()
    ingestor.close()


def test_collection_ownership_mismatch_is_rejected(tmp_path):
    bindings, _state = _bindings()
    ingestor = NemoRetrieverLocalIngestor(_config(tmp_path, bindings))
    ingestor.create_collection("reports")
    ingestor._runtime.vdb.collections["reports"]["metadata"]["aiq_nemo_retriever_local"]["profile"] = "fast-text"
    with pytest.raises(NemoRetrieverLocalOwnershipError, match="different adapter"):
        ingestor.get_collection("reports")
    ingestor.close()
