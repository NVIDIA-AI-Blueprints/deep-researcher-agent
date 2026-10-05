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

# =============================================================================
# Multi-stage build for AI-Q Blueprint Backend
# =============================================================================
#
# Build targets:
#   docker build --target dev -t aiq:dev .     # Development (includes CLI)
#   docker build --target release -t aiq:prod . # Production (web only, validates env)
#
# =============================================================================

# =============================================================================
# Stage 1: Builder (shared between dev and release)
# =============================================================================
FROM nvcr.io/nvidia/base/ubuntu:noble-20260217 AS builder

WORKDIR /app

# Install system dependencies and bootstrap Python for uv.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN python3 -m pip install --break-system-packages --no-cache-dir uv \
    && uv python install 3.13.14 \
    && UV_PYTHON_BIN="$(uv python find 3.13.14)" \
    && ln -sf "$UV_PYTHON_BIN" /usr/local/bin/python \
    && ln -sf "$UV_PYTHON_BIN" /usr/local/bin/python3

RUN uv venv /app/.venv --python /usr/local/bin/python

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_PYTHON=/app/.venv/bin/python

# Copy workspace configuration and lockfile
COPY pyproject.toml uv.lock ./

# Copy all source packages
COPY src/ ./src/
COPY sources/ ./sources/
COPY frontends/aiq_api/ ./frontends/aiq_api/
COPY frontends/cli/ ./frontends/cli/
COPY frontends/debug/ ./frontends/debug/
COPY configs/ ./configs/
# Only copy runtime scripts from deploy/ — never copy the full directory
# to avoid leaking .env, Helm charts, compose files, or other dev artifacts.
COPY deploy/entrypoint.py deploy/start_web.py ./deploy/

# Install dependencies using uv sync. We do NOT pass --no-dev: the workspace
# source packages (tavily-web-search, exa-web-search, ...) and their runtime
# deps (e.g. langchain-tavily) are declared in the [dependency-groups] dev group,
# so --no-dev would omit them and the editable source installs below would
# import-fail at runtime. Syncing with the dev group installs those deps from the
# frozen lock (still pinned, no re-resolve). TODO(deploy): move the source
# packages to a dedicated non-dev group to keep test-only tooling out of the image.
RUN uv sync --frozen --extra pii --extra s3

# Install workspace packages (without CLI for base). The root install does not
# need workspace source overrides because dependencies were synced above; avoid
# parsing opt-in workspace groups whose source trees are intentionally absent.
RUN uv pip install --no-sources --no-deps -e . \
    && uv pip install --no-deps -e ./sources/google_scholar_paper_search \
    && uv pip install --no-deps -e ./sources/tavily_web_search \
    && uv pip install --no-deps -e ./sources/exa_web_search \
    && uv pip install --no-deps -e ./sources/nimble_web_search \
    && uv pip install --no-deps -e "./sources/knowledge_layer[all]" \
    && uv pip install --no-deps -e ./frontends/aiq_api \
    && uv pip install "psycopg[binary]>=3.0.0"

RUN /app/.venv/bin/python -c \
    "import asyncio; import aiq_api; import knowledge_layer; import presidio_analyzer; import presidio_anonymizer; import spacy; from nemoguardrails.library.sensitive_data_detection.actions import mask_sensitive_data; from nemoguardrails.rails.llm.config import RailsConfig; config = RailsConfig.from_content(yaml_content='rails:\n  config:\n    sensitive_data_detection:\n      output:\n        entities:\n          - EMAIL_ADDRESS\n'); assert spacy.util.is_package('en_core_web_lg'); assert asyncio.run(mask_sensitive_data(source='output', text='Contact customer@example.com', config=config)) == 'Contact <EMAIL_ADDRESS>'; print('✓ Base packages and PII runtime installed')"

# Keep immutable application assets root-owned; only runtime data is writable by the service user.
RUN chmod +x /app/deploy/start_web.py \
    && mkdir -p /app/data \
    && chown 1000:1000 /app/data

# =============================================================================
# Optional OpenShell-enabled builder
# =============================================================================
#
# OpenShell remains an explicitly selected deployment dependency rather than a
# dependency of every AI-Q image. The published adapter currently declares an
# older DeepAgents range than AI-Q uses, so install it without dependency
# resolution after installing the certified SDK. The adapter surface is covered
# by AI-Q's OpenShell tests against the locked DeepAgents runtime.
FROM builder AS openshell-builder

RUN OPENSHELL_VERSION="$(/app/.venv/bin/python -c \
        "import pathlib,tomllib; c=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['tool']['aiq']['openshell']; print(c['release-tag'].removeprefix('v'))")" \
    && OPENSHELL_ADAPTER_VERSION="$(/app/.venv/bin/python -c \
        "import pathlib,tomllib; c=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['tool']['aiq']['openshell']; print(c['adapter-version'])")" \
    && uv pip install "openshell==${OPENSHELL_VERSION}" \
    && uv pip install --no-deps "langchain-nvidia-openshell==${OPENSHELL_ADAPTER_VERSION}" \
    && /app/.venv/bin/python -c \
        "import langchain_nvidia_openshell, openshell; print(f'✓ OpenShell SDK {openshell.__version__} and adapter {langchain_nvidia_openshell.__version__} installed')"

# =============================================================================
# Stage 2: Development (includes CLI)
# =============================================================================
FROM builder AS dev-builder

# Install CLI and debug UI for development
RUN uv pip install --no-deps -e ./frontends/cli \
    && uv pip install --no-deps -e ./frontends/debug

RUN /app/.venv/bin/python -c "import aiq_api; import aiq_research_cli; import aiq_debug; import knowledge_layer; print('✓ All packages installed')"

# =============================================================================
# Stage 3: Development runtime
# =============================================================================
FROM nvcr.io/nvidia/distroless/python:3.13-v4.0.8 AS dev

WORKDIR /app

COPY --from=dev-builder /opt/uv-python /opt/uv-python
COPY --from=dev-builder /app /app
COPY LICENSE LICENSE-THIRD-PARTY /licenses/

USER 1000:1000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["python", "/app/deploy/entrypoint.py"]

# =============================================================================
# Stage 4: Release (production - no CLI)
# =============================================================================
FROM nvcr.io/nvidia/distroless/python:3.13-v4.0.8 AS release

WORKDIR /app

# Copy from base builder (no CLI)
COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /app /app
COPY LICENSE LICENSE-THIRD-PARTY /licenses/

USER 1000:1000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV APP_ENV=production

EXPOSE 8000

ENTRYPOINT ["python", "/app/deploy/entrypoint.py"]
