# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""submit_agent_job forwards the owner's canonical user_id to the worker."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from deep_researcher_agent.auth import Principal
from deep_researcher_api.jobs import submit as submit_mod
from deep_researcher_api.mcp_auth.provider import principal_user_id


class _FakeJobStore:
    last_job_args = None

    def __init__(self, **_kwargs):
        pass

    def ensure_job_id(self, job_id):
        return job_id or "job-1"

    async def submit_job(self, *, job_id, expiry_seconds, job_fn, job_args):
        _FakeJobStore.last_job_args = job_args


@pytest.fixture
def patched(monkeypatch):
    import nat.front_ends.fastapi.async_jobs.job_store as js_mod

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://localhost:8786")
    monkeypatch.setattr(js_mod, "JobStore", _FakeJobStore)
    monkeypatch.setattr(
        submit_mod,
        "get_agent_config",
        lambda _t: SimpleNamespace(class_path="pkg.mod.Agent", config_name="deep_research_agent", public=True),
    )
    monkeypatch.setattr(submit_mod, "create_job_access", MagicMock())
    monkeypatch.setattr(submit_mod, "renew_job_access_reservation", MagicMock(return_value=True))
    monkeypatch.setattr(submit_mod, "reserve_deep_research_job", AsyncMock(return_value="admission-token"))
    monkeypatch.setattr(submit_mod, "renew_deep_research_job_reservation", AsyncMock(return_value=True))
    _FakeJobStore.last_job_args = None


def test_submit_forwards_owner_user_id_before_admission_token(patched):
    principal = Principal(type="jwt", sub="user-1", email="u@example.com")
    asyncio.run(
        submit_mod.submit_agent_job(
            agent_type="deep_researcher",
            input_text="query",
            owner="u@example.com",
            principal=principal,
            auth_token="token-1",
        )
    )
    job_args = _FakeJobStore.last_job_args
    assert job_args is not None
    # Trailing worker args are:
    # data_sources, auth_token, encryption policy, initial_files, output_metadata,
    # owner_user_id, admission fencing token.
    assert job_args[-2] == principal_user_id(principal) == "jwt:user-1"
    assert job_args[-6] == "token-1"
    assert job_args[-5].mode == "off"
    assert isinstance(job_args[-1], str)


def test_context_user_id_binding_mechanism():
    """The worker binds owner_user_id via ContextState.user_id; Context.user_id reads it.

    Guards the contract runner.run_agent_job relies on (and that per_user_mcp_client
    reads via Context.get().user_id) against NAT-side changes.
    """
    from nat.builder.context import Context
    from nat.builder.context import ContextState

    token = ContextState.get().user_id.set("jwt:user-9")
    try:
        assert Context.get().user_id == "jwt:user-9"
    finally:
        ContextState.get().user_id.reset(token)
