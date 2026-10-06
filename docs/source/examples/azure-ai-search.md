<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Azure AI Search knowledge layer

Use Azure AI Search as the document store while retaining Deep Researcher Agent's existing
upload API, per-conversation collection routing, document summaries, and
citations. This example assumes the Azure AI Search service and embedding
endpoint already exist; it does not deploy Azure infrastructure.

## Prerequisites

- Create or select an [Azure AI Search service](https://learn.microsoft.com/azure/search/search-create-service-portal).
- Choose a tier whose [document and storage limits](https://learn.microsoft.com/azure/search/search-limits-quotas-capacity)
  cover the expected data volume. Deep Researcher Agent stores logical collections in one physical index.
- Copy the service endpoint from the Azure portal. For key authentication, also copy an admin key. Otherwise,
  [enable role-based access](https://learn.microsoft.com/azure/search/keyless-connections) and assign the roles below.

Install the backend dependency:

```bash
uv pip install -e "sources/knowledge_layer[azure_ai_search]"
```

Create `deploy/.env` when needed. The Azure entries are commented in the
template so non-Azure runs do not change; uncomment the endpoint and the
settings required for your authentication mode:

```bash
cp -n deploy/.env.example deploy/.env
```

```text
NVIDIA_API_KEY=<embedding-api-key>
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
# API-key authentication:
# AZURE_SEARCH_API_KEY=<search-admin-key>
# User-assigned managed identity:
# AZURE_CLIENT_ID=<managed-identity-client-id>
# Optional deployment-unique prefix (default: deep-researcher):
# DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX=deep-researcher
```

## Grant managed identity access

When `AZURE_SEARCH_API_KEY` is absent, enable role-based access on the Azure AI
Search service and grant the workload identity both of these built-in roles:

| Role | Used for |
|------|----------|
| `Search Service Contributor` | Create and inspect the shared Deep Researcher Agent index. |
| `Search Index Data Contributor` | Upload, query, and delete index documents. |

Assign the roles at the search-service scope. The principal ID is the object ID
of the system-assigned or user-assigned managed identity running Deep Researcher Agent.

Start the backend and UI with the checked-in Azure configuration:

```bash
./scripts/start_e2e.sh --config_file configs/config_web_azure_ai_search.yml
```

`start_e2e.sh` sources `deploy/.env` before starting the backend, so
uncommented values in that file replace same-named shell exports. Keep the
Azure settings there, or leave them commented before relying on exported
values.

`AZURE_SEARCH_API_KEY` selects API-key authentication when present; otherwise
the adapter uses `DefaultAzureCredential`. Set `AZURE_CLIENT_ID` to select a
user-assigned identity. Embeddings share `DEEP_RESEARCHER_EMBED_BASE_URL`,
`DEEP_RESEARCHER_EMBED_MODEL`, and `NVIDIA_API_KEY` with the LlamaIndex backend.
Azure-specific optional settings are `DEEP_RESEARCHER_EMBED_DIM` and
`DEEP_RESEARCHER_AZURE_SEARCH_INDEX_PREFIX`. The prefix must uniquely identify one Deep Researcher Agent
deployment when a search service is shared.

Changing the embedding model or dimension selects a different physical index
and requires re-ingestion. Frontend WebSocket queries use the conversation ID
as the collection; direct API tests must supply equivalent context or query the
configured fallback collection.

The backend stores collection, file, and chunk records in one physical index
selected by `azure_search_index_prefix`, schema version, embedding model, and
dimension. Every operation applies internal collection filters. Retrieval is
always hybrid, and ingestion uses fixed 1024-token chunks with 128-token
overlap. File IDs returned by upload are authoritative for status and
delete operations; same-name uploads coexist independently.
