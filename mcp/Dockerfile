# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build from the repository root:
#   docker build --target release -f mcp/Dockerfile -t deep-researcher-mcp-server:local .

ARG PYTHON_IMAGE=python:3.13.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ARG UV_VERSION=0.11.26

WORKDIR /app

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON_DOWNLOADS=never

# MCP is an independent uv project with its own lock. Preserve the repository
# layout for its reviewed local path dependencies, but do not copy or resolve
# the root workspace lock.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY sources/knowledge_layer/ ./sources/knowledge_layer/
COPY sources/tavily_web_search/ ./sources/tavily_web_search/
COPY mcp/LICENSE mcp/pyproject.toml mcp/README.md mcp/uv.lock ./mcp/
COPY mcp/src/ ./mcp/src/
COPY mcp/scripts/check_runtime_dependencies.py ./mcp/scripts/check_runtime_dependencies.py
COPY configs/config_mcp.yml ./configs/config_mcp.yml

# Install exactly the production dependency closure from the frozen public
# MCP lock. Local path packages are installed as wheels, not editables, so
# the release image needs only the virtual environment and runtime config.
RUN --mount=type=cache,id=deep-researcher-mcp-uv,target=/root/.cache/uv,sharing=locked \
    uv sync \
        --project /app/mcp \
        --frozen \
        --no-dev \
        --no-default-groups \
        --no-editable \
        --python /usr/local/bin/python

# The import canary loads deep_researcher_mcp.server, which resolves its workflow config at
# import time; point it at the same config the release image ships.
RUN DEEP_RESEARCHER_MCP_CONFIG=/app/configs/config_mcp.yml \
    /opt/venv/bin/python mcp/scripts/check_runtime_dependencies.py --verify-imports

FROM ${PYTHON_IMAGE} AS release

LABEL org.opencontainers.image.title="NVIDIA Deep Researcher Agent MCP Server" \
      org.opencontainers.image.description="Public FastMCP server for the NVIDIA Deep Researcher Agent Blueprint" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 deep-researcher \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/deep-researcher --shell /usr/sbin/nologin deep-researcher \
    && install -d --owner=10001 --group=10001 /app/relay

WORKDIR /app

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --chown=10001:10001 configs/config_mcp.yml /app/configs/config_mcp.yml
COPY --chown=10001:10001 LICENSE LICENSE-THIRD-PARTY /licenses/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV HOME=/home/deep-researcher
ENV XDG_CACHE_HOME=/home/deep-researcher/.cache
ENV DEEP_RESEARCHER_MCP_CONFIG=/app/configs/config_mcp.yml
ENV DEEP_RESEARCHER_MCP_HOST=0.0.0.0
ENV DEEP_RESEARCHER_MCP_PORT=9001
ENV DEEP_RESEARCHER_MCP_PATH=/mcp
ENV DEEP_RESEARCHER_MCP_WORKERS=1
ENV DEEP_RESEARCHER_MCP_LOG_LEVEL=INFO

USER 10001:10001

EXPOSE 9001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import http.client, os; connection = http.client.HTTPConnection('127.0.0.1', int(os.environ.get('DEEP_RESEARCHER_MCP_PORT', '9001')), timeout=3); connection.request('GET', '/health'); response = connection.getresponse(); response.read(); connection.close(); raise SystemExit(0 if response.status == 200 else 1)"]

ENTRYPOINT ["python", "-m", "deep_researcher_mcp.server"]
