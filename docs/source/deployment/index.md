<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment

The Deep Researcher Agent blueprint supports multiple deployment methods. Choose the one that best fits your environment and operational requirements.

| Method | Best For | Prerequisites |
|--------|----------|---------------|
| [Docker Compose](./docker-compose.md) | Local development, team demos, single-node deployments | Docker Engine, Docker Compose v2 |
| [Kubernetes (Helm)](./kubernetes.md) | Multi-node clusters, production | Kubernetes cluster, Helm v3.x |
| Manual (no containers) | Development and debugging | Python 3.11--3.13, system dependencies (refer to [Installation](../get-started/installation.md)) |

## Published Release Artifacts

Deep Researcher Agent v2.2.0 is published on NVIDIA NGC. Use the exact versioned references below for release deployments.

| Artifact | Type | Versioned reference |
|----------|------|---------------------|
| [`aiq-agent`](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/containers/aiq-agent/2.2.0) | Container image | `nvcr.io/nvidia/blueprint/aiq-agent:2.2.0` |
| [`aiq-frontend`](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/containers/aiq-frontend/2.2.0) | Container image | `nvcr.io/nvidia/blueprint/aiq-frontend:2.2.0` |
| [`aiq2-web`](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/helm-charts/aiq2-web/2.2.0) | Helm chart | `nvidia/blueprint/aiq2-web:2.2.0` |

## Architecture Overview

All containerized deployments run the same three services:

- **Backend** (`deep-researcher-agent`) -- [FastAPI](https://fastapi.tiangolo.com/) server with an embedded [Dask](https://www.dask.org/) scheduler and worker for background job processing.
- **Frontend** (`deep-researcher-blueprint-ui`) -- [Next.js](https://nextjs.org/) web UI that communicates with the backend API.
- **Database** (`postgres`) -- [PostgreSQL](https://www.postgresql.org/) instance for async job storage, [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) checkpoints, and document summaries.

## Deployment Guides

- **[Docker Compose](./docker-compose.md)** -- Full Docker Compose reference covering environment setup, the standard LlamaIndex stack, Foundational RAG (FRAG) integration, database configuration, and troubleshooting.

- **[Kubernetes (Helm)](./kubernetes.md)** -- Helm chart deployment for Kubernetes clusters, including NGC image pull secrets, configuration switching, FRAG integration, and troubleshooting.

- **[Amazon OpenSearch Serverless](./aws-opensearch-serverless.md)** -- EKS and OpenSearch Serverless deployment notes for the built-in OpenSearch knowledge backend.

- **[Docker Build System](./docker-build.md)** -- Multi-stage Dockerfile architecture, build targets (dev vs. release), base images, and startup scripts (`entrypoint.py` and `start_web.py`).

- **[Authentication](./authentication.md)** -- Enable OAuth/OIDC sign-in, configure backend JWT validation, and use Deep Researcher Agent user tokens in tools and MCP pass-through integrations.

- **[Async Job Content Encryption](./content-encryption.md)** -- Configure encryption at rest for async final reports and selected artifact event content, including Vault Transit and static-key modes.

- **[Observability](./observability.md)** -- NeMo Relay logging, ATOF traces, Phoenix OTEL export, redaction, and cost data.

- **[Production Considerations](./production.md)** -- Guidance on managed databases, horizontal scaling, security hardening, monitoring, and resource requirements.

- **[OpenShell](./openshell.md)** -- Optional policy-bound execution for generated deep-research code, including supported platforms, authenticated gateway ownership, policy/config pairing, deterministic live acceptance, and safe cleanup.

## Quick Start

For the fastest path to a running stack:

```bash
# 1. Configure environment
cp deploy/.env.example deploy/.env
# Edit deploy/.env with your API keys

# 2. Start services
cd deploy/compose
docker compose --env-file ../.env -f docker-compose.yaml up -d --build
```

Open [http://localhost:3000](http://localhost:3000) to access the web UI.
