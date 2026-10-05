-- SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- =============================================================================
-- Deep Researcher Agent MCP - Database Initialization (idempotent - safe to re-run)
-- =============================================================================
--
-- MCP-owned submit/poll ledger inside DEEP_RESEARCHER_CHECKPOINT_DB. Keep the executable
-- SQL in sync with deep_researcher_mcp.job_store. Application startup also applies this
-- schema under a PostgreSQL advisory lock for safe multi-replica startup.
--
-- The legacy migration component name aiq_maas_mcp is intentionally preserved
-- so upgrades reuse the existing migration history for these physical tables.
-- =============================================================================

-- Run this script while connected to the database selected by DEEP_RESEARCHER_CHECKPOINT_DB.

CREATE TABLE IF NOT EXISTS mcp_schema_migrations (
    component TEXT NOT NULL,
    version INTEGER NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (component, version)
);

CREATE TABLE IF NOT EXISTS mcp_jobs (
    job_id UUID PRIMARY KEY,
    principal TEXT NOT NULL,
    query TEXT NOT NULL,
    depth TEXT NOT NULL CHECK (depth IN ('shallow', 'deep', 'meta')),
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'complete', 'failed')),
    result TEXT,
    error TEXT,
    poll_count INTEGER NOT NULL DEFAULT 0,
    runner_id TEXT,
    heartbeat_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

ALTER TABLE mcp_jobs ADD COLUMN IF NOT EXISTS poll_count INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_mcp_jobs_principal_job_id ON mcp_jobs(principal, job_id);
CREATE INDEX IF NOT EXISTS idx_mcp_jobs_expires_at ON mcp_jobs(expires_at);
CREATE INDEX IF NOT EXISTS idx_mcp_jobs_state_updated_at ON mcp_jobs(state, updated_at);
CREATE INDEX IF NOT EXISTS idx_mcp_jobs_runner_state ON mcp_jobs(runner_id, state);

INSERT INTO mcp_schema_migrations (component, version)
VALUES ('aiq_maas_mcp', 1)
ON CONFLICT (component, version) DO NOTHING;

INSERT INTO mcp_schema_migrations (component, version)
VALUES ('aiq_maas_mcp', 2)
ON CONFLICT (component, version) DO NOTHING;
