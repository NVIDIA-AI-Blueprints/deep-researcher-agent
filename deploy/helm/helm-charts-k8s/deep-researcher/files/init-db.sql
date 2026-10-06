-- =============================================================================
-- Deep Researcher Agent Blueprint - Database Initialization (idempotent — safe to re-run)
-- KEEP IN SYNC with deploy/compose/init-db.sql (the source of truth)
-- =============================================================================
--
-- Run by the backend init container on every pod start. All statements are
-- idempotent (IF NOT EXISTS) so re-runs are safe.
--
-- Databases:
--   - deep_researcher_jobs         (job metadata, events, document summaries)
--   - deep_researcher_checkpoints  (LangGraph conversation state)
--
-- Tables in deep_researcher_jobs:
--   - job_info                — NAT JobStore metadata (status, timestamps, expiry)
--   - job_access              — DeepResearcher-owned job ownership/access control metadata
--   - deep_research_admission — Atomic capacity and submission-rate reservations
--   - job_events              — SSE streaming events and job event persistence
--   - summaries               — Document summaries (collection + filename keyed)
--
-- Tables in deep_researcher_checkpoints:
--   - checkpoints           — LangGraph conversation checkpoints
--   - checkpoint_blobs      — LangGraph binary state data
--   - checkpoint_writes     — LangGraph pending writes
--   - checkpoint_migrations — LangGraph schema version tracking
--
-- =============================================================================

-- Create checkpoints database if it doesn't exist
SELECT 'CREATE DATABASE deep_researcher_checkpoints' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'deep_researcher_checkpoints')\gexec

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE deep_researcher_jobs TO deep_researcher;
GRANT ALL PRIVILEGES ON DATABASE deep_researcher_checkpoints TO deep_researcher;

-- =============================================================================
-- Create tables in deep_researcher_jobs database
-- =============================================================================
\connect deep_researcher_jobs

-- Job metadata table (NAT JobStore)
CREATE TABLE IF NOT EXISTS job_info (
    job_id VARCHAR PRIMARY KEY,
    status VARCHAR NOT NULL,
    config_file VARCHAR,
    error VARCHAR,
    output_path VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    expiry_seconds INTEGER,
    output VARCHAR,
    is_expired BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_job_info_status ON job_info(status);
CREATE INDEX IF NOT EXISTS idx_job_info_created_at ON job_info(created_at);

CREATE TABLE IF NOT EXISTS job_access (
    job_id VARCHAR PRIMARY KEY,
    owner_auth_type VARCHAR NOT NULL,
    owner_subject VARCHAR NOT NULL,
    owner_email VARCHAR,
    conversation_id VARCHAR,
    agent_type VARCHAR,
    submission_token VARCHAR,
    submission_expires_at DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE job_access ADD COLUMN IF NOT EXISTS conversation_id VARCHAR;
ALTER TABLE job_access ADD COLUMN IF NOT EXISTS agent_type VARCHAR;
ALTER TABLE job_access ADD COLUMN IF NOT EXISTS submission_token VARCHAR;
ALTER TABLE job_access ADD COLUMN IF NOT EXISTS submission_expires_at DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_job_access_owner ON job_access(owner_auth_type, owner_subject);
-- Supports the report-follow-up default: "latest completed report job in this conversation".
CREATE INDEX IF NOT EXISTS idx_job_access_conversation ON job_access(conversation_id);

-- Deep-research admission reservations. Successful rows remain for the rolling
-- submission-rate window; terminal/orphaned rows are cleaned on later admits.
CREATE TABLE IF NOT EXISTS deep_research_admission (
    job_id VARCHAR PRIMARY KEY,
    reservation_token VARCHAR NOT NULL,
    owner_auth_type VARCHAR NOT NULL,
    owner_subject VARCHAR NOT NULL,
    admitted_at DOUBLE PRECISION NOT NULL,
    reservation_expires_at DOUBLE PRECISION NOT NULL
);

ALTER TABLE deep_research_admission ADD COLUMN IF NOT EXISTS reservation_token VARCHAR;
DELETE FROM deep_research_admission WHERE reservation_token IS NULL;
ALTER TABLE deep_research_admission ALTER COLUMN reservation_token SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_deep_research_admission_owner
ON deep_research_admission(owner_auth_type, owner_subject, admitted_at);

-- Job events table (SSE streaming, event persistence)
CREATE TABLE IF NOT EXISTS job_events (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    event_data TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id_id ON job_events(job_id, id);

-- Document summaries table
CREATE TABLE IF NOT EXISTS summaries (
    collection VARCHAR(256) NOT NULL,
    filename VARCHAR(512) NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (collection, filename)
);

CREATE INDEX IF NOT EXISTS idx_summaries_collection ON summaries(collection);

-- =============================================================================
-- Create LangGraph checkpoint tables in deep_researcher_checkpoints database
-- These must exist before backends connect. Previously left to the app,
-- but if postgres restarts without a backend restart, the tables are lost
-- and running backends crash with "relation checkpoints does not exist".
-- =============================================================================
\connect deep_researcher_checkpoints

CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
