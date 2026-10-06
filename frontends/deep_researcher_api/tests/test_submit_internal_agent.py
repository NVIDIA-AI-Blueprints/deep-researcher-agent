# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""submit_agent_job must enforce the internal (public=False) agent gate itself.

The HTTP /submit route filters internal agents, but the submission helper is the
real trust boundary: any caller that forwards a caller-influenced agent_type must
not be able to launch an internal-only agent without explicitly opting in.
"""

from __future__ import annotations

import pytest

from deep_researcher_agent.auth import Principal


@pytest.mark.asyncio
async def test_submit_internal_agent_rejected_without_allow_internal(monkeypatch):
    import deep_researcher_api.jobs.submit as submit_mod

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setenv("REQUIRE_AUTH", "false")

    # report_rewriter is registered with public=False.
    with pytest.raises(submit_mod.InternalAgentError):
        await submit_mod.submit_agent_job(
            agent_type="report_rewriter",
            input_text="rewrite",
            owner="user@example.com",
            principal=Principal(type="jwt", sub="user-1", email="user@example.com"),
        )


@pytest.mark.asyncio
async def test_submit_internal_agent_allowed_with_allow_internal(monkeypatch):
    import deep_researcher_api.jobs.submit as submit_mod
    import nat.front_ends.fastapi.async_jobs.job_store as job_store_mod

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setenv("REQUIRE_AUTH", "false")

    submitted = {}

    class _FakeJobStore:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_job_id(self, job_id):
            return job_id or "child-1"

        async def submit_job(self, *args, **kwargs):
            submitted["called"] = True

    monkeypatch.setattr(job_store_mod, "JobStore", _FakeJobStore)
    monkeypatch.setattr(submit_mod, "create_job_access", lambda *a, **k: None)

    job_id = await submit_mod.submit_agent_job(
        agent_type="report_rewriter",
        input_text="rewrite",
        owner="user@example.com",
        principal=Principal(type="jwt", sub="user-1", email="user@example.com"),
        allow_internal=True,
    )
    assert job_id == "child-1"
    assert submitted.get("called") is True
