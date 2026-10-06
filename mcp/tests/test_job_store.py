# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres job-store construction tests."""

import pytest

from deep_researcher_mcp.job_store import JobStore


def test_job_store_builds_quoted_table_names() -> None:
    store = JobStore("postgresql://localhost/deep-researcher", schema="mcp_runtime")

    assert store._jobs_table == '"mcp_runtime".mcp_jobs'
    assert store._migrations_table == '"mcp_runtime".mcp_schema_migrations'


def test_job_store_rejects_invalid_schema_identifier() -> None:
    with pytest.raises(ValueError, match="Invalid Postgres identifier"):
        JobStore("postgresql://localhost/deep-researcher", schema="public;drop table mcp_jobs")
