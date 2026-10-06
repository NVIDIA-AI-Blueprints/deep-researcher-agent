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
# Deep Researcher Agent Blueprint UI - Production Dockerfile
# =============================================================================
#
# Multi-stage build for the Next.js frontend application.
# Uses NVIDIA authorized Ubuntu base image.
#
# -----------------------------------------------------------------------------
# BUILD
# -----------------------------------------------------------------------------
#
#   docker build -f deploy/Dockerfile -t deep-researcher-blueprint-ui:latest .
#
# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
#
# With OAuth authentication:
#
#   docker run -p 3000:3000 \
#     -e BACKEND_URL=http://backend:8000 \
#     -e NEXTAUTH_SECRET=$(openssl rand -base64 32) \
#     -e NEXTAUTH_URL=http://localhost:3000 \
#     -e OAUTH_CLIENT_ID=your-client-id \
#     -e OAUTH_CLIENT_SECRET=your-client-secret \
#     -e OAUTH_ISSUER=https://your-oidc-provider.com \
#     deep-researcher-blueprint-ui:latest
#
# Without authentication (development/testing):
#
#   docker run -p 3000:3000 \
#     -e BACKEND_URL=http://backend:8000 \
#     -e REQUIRE_AUTH=false \
#     deep-researcher-blueprint-ui:latest
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (all runtime - no rebuild needed)
# -----------------------------------------------------------------------------
#
# Backend:
#   BACKEND_URL                     - Backend API URL (default: http://localhost:8000)
#
# Authentication:
#   REQUIRE_AUTH                    - Set to 'true' to require OAuth login (default: false)
#   NEXTAUTH_SECRET                 - Session encryption secret (required if auth enabled)
#   NEXTAUTH_URL                    - Public URL where app is hosted (required if auth enabled)
#                                     Also determines cookie security:
#                                     - http://...  → non-secure cookies
#                                     - https://... → secure cookies
#   SECURE_COOKIES                  - Explicit cookie security override (optional)
#
# OAuth (required if REQUIRE_AUTH=true):
#   OAUTH_CLIENT_ID                 - OAuth client ID from your OIDC provider
#   OAUTH_CLIENT_SECRET             - OAuth client secret
#   OAUTH_ISSUER                    - OIDC issuer URL (enables auto-discovery)
#
# =============================================================================

# =============================================================================
# Stage 1: Base with Node.js
# =============================================================================
FROM nvcr.io/nvidia/base/ubuntu:jammy-20260217 AS base

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS deps

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci --ignore-scripts

# =============================================================================
# Stage 3: Builder
# =============================================================================
FROM base AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Defense-in-depth: remove any env files that may have leaked past .dockerignore
RUN rm -f .env .env.local .env.*.local .env.development .env.production 2>/dev/null || true

ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# Remove devDependencies — keeps dev-only packages out of the production image
RUN npm ci --omit=dev --ignore-scripts

# =============================================================================
# Stage 4: Production Runner
# =============================================================================
FROM base AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

RUN groupadd --system --gid 1001 nodejs \
    && useradd --system --uid 1001 --gid nodejs nextjs \
    && rm -rf /usr/lib/node_modules/npm /usr/bin/npm /usr/bin/npx

COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json
COPY --from=builder --chown=nextjs:nodejs /app/next.config.ts ./next.config.ts
COPY --from=builder --chown=nextjs:nodejs /app/server.js ./server.js
COPY --from=builder --chown=nextjs:nodejs /app/websocket-cookie.js ./websocket-cookie.js

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
