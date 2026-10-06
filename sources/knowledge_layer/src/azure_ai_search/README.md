<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Azure AI Search backend

This backend stores Deep Researcher Agent document chunks and vectors in Azure AI Search. It
uses the shared `knowledge_retrieval` NAT function, Knowledge API, session
collection routing, summary store, and citation formatter.

## Install

```bash
uv pip install -e "sources/knowledge_layer[azure_ai_search]"
```

## Configure

Set service and model credentials in the environment:

```bash
export AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
export NVIDIA_API_KEY=<embedding-api-key>
# Optional: setting this selects API-key auth instead of DefaultAzureCredential.
export AZURE_SEARCH_API_KEY=<search-admin-key>
```

```yaml
functions:
  knowledge_search:
    _type: knowledge_retrieval
    backend: azure_ai_search
    collection_name: ${COLLECTION_NAME:-deep_researcher_default}
    top_k: 5

    generate_summary: true
    summary_model: summary_llm
    summary_db: ${DEEP_RESEARCHER_SUMMARY_DB:-sqlite+aiosqlite:///./summaries.db}
```

Explicit YAML values still override the environment-backed defaults. Azure
Search uses `AZURE_SEARCH_ENDPOINT` and optional `AZURE_SEARCH_API_KEY`.
Embedding configuration shares `DEEP_RESEARCHER_EMBED_BASE_URL`, `DEEP_RESEARCHER_EMBED_MODEL`, and
`NVIDIA_API_KEY` with the LlamaIndex backend; Azure additionally accepts
`DEEP_RESEARCHER_EMBED_DIM` and `DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX`. The index prefix must be
unique to one Deep Researcher Agent deployment sharing a search service.

When `AZURE_SEARCH_API_KEY` is absent, the adapter uses
`DefaultAzureCredential`. Set `AZURE_CLIENT_ID` when a user-assigned identity
should be selected. Enable role-based access on the search service and grant
the identity `Search Service Contributor` for index management plus `Search
Index Data Contributor` for document ingestion and retrieval. Assign both roles
at the search-service scope because logical collections share one physical index.

The adapter parses PDF, DOCX, TXT, and Markdown uploads with LlamaIndex,
creates one namespaced Azure AI Search index per deployment prefix, schema
version, embedding model, and dimension, and always performs balanced hybrid
retrieval. Collection and file manifests isolate logical collections in that
index. Documents use fixed 1024-token chunks with 128-token overlap. Only the
index carrying the matching Deep Researcher Agent ownership/schema marker is visible or mutable.

Upload responses return canonical UUID file IDs used by job progress, list,
status, and delete operations. Same-name uploads coexist independently under
different file IDs. Upload and delete requests stay below Azure's 1,000-action
and 16 MiB limits, and every per-document result is checked.

Collections use the shared Knowledge Layer TTL settings:
`DEEP_RESEARCHER_COLLECTION_TTL_HOURS` defaults to 24 hours and
`DEEP_RESEARCHER_TTL_CLEANUP_INTERVAL_SECONDS` defaults to 3600 seconds. Successful file
and collection deletion also clears corresponding summary records.

`embed_dim` must match both the embedding model output and the selected index.
Changing from a 2048-dimensional model to `nvidia/nv-embed-v1` at 4096
dimensions selects a different physical index and requires re-ingestion. The
adapter validates ownership, fields, vector profile, and dimensions before use;
it does not alter an incompatible schema.

For direct API tests, use the same collection or conversation context used for
upload. A standalone chat request without that context falls back to the
configured `collection_name`.
