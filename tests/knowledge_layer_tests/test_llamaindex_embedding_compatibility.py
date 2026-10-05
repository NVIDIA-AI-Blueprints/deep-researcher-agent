# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-identity checks for persisted LlamaIndex/Chroma collections."""

from types import SimpleNamespace

import pytest
from knowledge_layer.llamaindex.adapter import _CHROMA_EMBEDDING_MODEL_KEY
from knowledge_layer.llamaindex.adapter import _get_nvidia_api_key
from knowledge_layer.llamaindex.adapter import _validate_chroma_embedding_model


def test_chroma_collection_accepts_matching_embedding_model() -> None:
    collection = SimpleNamespace(metadata={_CHROMA_EMBEDDING_MODEL_KEY: "nvidia/test-embed"})

    _validate_chroma_embedding_model(collection, "docs", "nvidia/test-embed")


def test_embedding_api_key_override_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_EMBED_API_KEY", "embedding-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "default-key")

    assert _get_nvidia_api_key() == "embedding-key"


def test_embedding_api_key_falls_back_to_nvidia_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIQ_EMBED_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "default-key")

    assert _get_nvidia_api_key() == "default-key"


@pytest.mark.parametrize("metadata", [{}, {_CHROMA_EMBEDDING_MODEL_KEY: "nvidia/old-embed"}])
def test_chroma_collection_rejects_unknown_or_mismatched_embedding_model(metadata: dict[str, str]) -> None:
    collection = SimpleNamespace(metadata=metadata)

    with pytest.raises(RuntimeError, match="Delete and re-ingest"):
        _validate_chroma_embedding_model(collection, "docs", "nvidia/new-embed")
