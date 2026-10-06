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

"""Tests for the submit-path sandbox concurrency guard (Option A)."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from deep_researcher_api.routes import jobs as jobs_module

_PRINCIPAL = SimpleNamespace(type="user", sub="subject-1")

# Pin the caps so the assertions are independent of any DEEP_RESEARCHER_MAX_SANDBOXES_* set in CI/dev.
_PINNED_CAPS = {"DEEP_RESEARCHER_MAX_SANDBOXES_PER_PRINCIPAL": "5", "DEEP_RESEARCHER_MAX_SANDBOXES_GLOBAL": "50"}


def _run(coro):
    return asyncio.run(coro)


class TestAgentUsesSandbox:
    def test_true_when_enabled(self) -> None:
        builder = SimpleNamespace(get_function_config=lambda _n: SimpleNamespace(sandbox=SimpleNamespace(enabled=True)))
        assert jobs_module._agent_uses_sandbox(builder, "cfg") is True

    def test_false_when_disabled(self) -> None:
        cfg = SimpleNamespace(sandbox=SimpleNamespace(enabled=False))
        builder = SimpleNamespace(get_function_config=lambda _n: cfg)
        assert jobs_module._agent_uses_sandbox(builder, "cfg") is False

    def test_false_when_no_sandbox(self) -> None:
        builder = SimpleNamespace(get_function_config=lambda _n: SimpleNamespace(sandbox=None))
        assert jobs_module._agent_uses_sandbox(builder, "cfg") is False

    def test_false_when_config_lookup_errors(self) -> None:
        def _boom(_name: str):
            raise RuntimeError("no config")

        builder = SimpleNamespace(get_function_config=_boom)
        assert jobs_module._agent_uses_sandbox(builder, "cfg") is False


class TestEnforceSandboxConcurrency:
    def test_rejects_when_owner_over_limit(self) -> None:
        with (
            patch.dict(os.environ, _PINNED_CAPS, clear=False),
            patch("deep_researcher_api.jobs.access.count_active_jobs_for_owner", return_value=5),
            patch("deep_researcher_api.jobs.access.count_active_jobs_global", return_value=0),
            pytest.raises(HTTPException) as exc,
        ):
            _run(jobs_module._enforce_sandbox_concurrency("sqlite:///x", _PRINCIPAL))
        assert exc.value.status_code == 429

    def test_rejects_when_global_over_limit(self) -> None:
        with (
            patch.dict(os.environ, _PINNED_CAPS, clear=False),
            patch("deep_researcher_api.jobs.access.count_active_jobs_for_owner", return_value=0),
            patch("deep_researcher_api.jobs.access.count_active_jobs_global", return_value=50),
            pytest.raises(HTTPException) as exc,
        ):
            _run(jobs_module._enforce_sandbox_concurrency("sqlite:///x", _PRINCIPAL))
        assert exc.value.status_code == 503

    def test_allows_under_limit(self) -> None:
        with (
            patch.dict(os.environ, _PINNED_CAPS, clear=False),
            patch("deep_researcher_api.jobs.access.count_active_jobs_for_owner", return_value=1),
            patch("deep_researcher_api.jobs.access.count_active_jobs_global", return_value=1),
        ):
            _run(jobs_module._enforce_sandbox_concurrency("sqlite:///x", _PRINCIPAL))

    def test_fails_open_when_counts_unknown(self) -> None:
        with (
            patch.dict(os.environ, _PINNED_CAPS, clear=False),
            patch("deep_researcher_api.jobs.access.count_active_jobs_for_owner", return_value=None),
            patch("deep_researcher_api.jobs.access.count_active_jobs_global", return_value=None),
        ):
            _run(jobs_module._enforce_sandbox_concurrency("sqlite:///x", _PRINCIPAL))
