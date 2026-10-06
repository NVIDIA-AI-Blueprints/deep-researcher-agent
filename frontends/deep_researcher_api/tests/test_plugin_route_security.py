# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from deep_researcher_api import plugin
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi import fastapi_front_end_plugin_worker as nat_worker_module
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin


class _WorkflowInput(BaseModel):
    query: str


class _SessionManager:
    @staticmethod
    def get_workflow_input_schema():
        return _WorkflowInput

    @staticmethod
    def get_workflow_single_output_schema():
        return str

    @staticmethod
    def get_workflow_streaming_output_schema():
        return str


def _make_api_plugin(*, scheduler_address: str | None = None) -> plugin.DeepResearcherAPIPlugin:
    front_end_config = plugin.DeepResearcherAPIConfig(scheduler_address=scheduler_address)
    full_config = Config(general=GeneralConfig(front_end=front_end_config))
    return plugin.DeepResearcherAPIPlugin(full_config=full_config, config=front_end_config)


@pytest.mark.asyncio
async def test_deep_researcher_plugin_binds_local_dask_dashboard_to_loopback(monkeypatch):
    import dask.distributed as dask_distributed

    local_cluster_calls = []

    def local_cluster_factory(*args, **kwargs):
        local_cluster_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(dask_distributed, "LocalCluster", local_cluster_factory)

    async def run_nat_front_end(_self):
        # Match NAT 1.8's local import and constructor call.
        from dask.distributed import LocalCluster

        LocalCluster(n_workers=1)

    monkeypatch.setattr(FastApiFrontEndPlugin, "run", run_nat_front_end)

    await _make_api_plugin().run()

    assert local_cluster_calls == [
        (
            (),
            {
                "n_workers": 1,
                "dashboard_address": "127.0.0.1:8787",
            },
        ),
    ]
    assert dask_distributed.LocalCluster is local_cluster_factory


@pytest.mark.asyncio
async def test_deep_researcher_plugin_leaves_external_dask_scheduler_path_untouched(monkeypatch):
    import dask.distributed as dask_distributed

    local_cluster_factory = MagicMock()
    monkeypatch.setattr(dask_distributed, "LocalCluster", local_cluster_factory)

    async def run_nat_front_end(_self):
        assert dask_distributed.LocalCluster is local_cluster_factory

    monkeypatch.setattr(FastApiFrontEndPlugin, "run", run_nat_front_end)

    await _make_api_plugin(scheduler_address="tcp://scheduler.example:8786").run()

    local_cluster_factory.assert_not_called()
    assert dask_distributed.LocalCluster is local_cluster_factory


@pytest.mark.asyncio
async def test_deep_researcher_plugin_uses_external_dask_scheduler_from_environment(monkeypatch):
    import dask.distributed as dask_distributed

    scheduler_address = "tls://scheduler.example:8786"
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", scheduler_address)
    local_cluster_factory = MagicMock()
    monkeypatch.setattr(dask_distributed, "LocalCluster", local_cluster_factory)

    async def run_nat_front_end(self):
        assert self.front_end_config.scheduler_address == scheduler_address
        assert dask_distributed.LocalCluster is local_cluster_factory

    monkeypatch.setattr(FastApiFrontEndPlugin, "run", run_nat_front_end)

    await _make_api_plugin().run()

    local_cluster_factory.assert_not_called()
    assert dask_distributed.LocalCluster is local_cluster_factory


@pytest.mark.asyncio
async def test_deep_researcher_plugin_restores_local_cluster_factory_when_startup_fails(monkeypatch):
    import dask.distributed as dask_distributed

    def local_cluster_factory(*_args, **_kwargs):
        raise RuntimeError("cluster startup failed")

    monkeypatch.setattr(dask_distributed, "LocalCluster", local_cluster_factory)

    async def run_nat_front_end(_self):
        from dask.distributed import LocalCluster

        LocalCluster()

    monkeypatch.setattr(FastApiFrontEndPlugin, "run", run_nat_front_end)

    with pytest.raises(RuntimeError, match="cluster startup failed"):
        await _make_api_plugin().run()

    assert dask_distributed.LocalCluster is local_cluster_factory


@pytest.mark.asyncio
async def test_deep_researcher_worker_does_not_register_nat_direct_async_generation_routes(monkeypatch):
    """Dask jobs must enter through Deep Researcher Agent's guarded async job API."""
    monkeypatch.setenv("NAT_CONFIG_FILE", "config.yml")
    monkeypatch.delenv("NAT_DASK_SCHEDULER_ADDRESS", raising=False)
    worker = plugin.DeepResearcherAPIWorker(Config(general=GeneralConfig(front_end=plugin.DeepResearcherAPIConfig())))
    worker._dask_available = True
    worker._job_store = object()

    session_manager = _SessionManager()
    monkeypatch.setattr(worker, "_create_session_manager", AsyncMock(return_value=session_manager))
    monkeypatch.setattr(nat_worker_module, "add_authorization_route", AsyncMock())
    monkeypatch.setattr(nat_worker_module, "add_execution_routes", AsyncMock())
    monkeypatch.setattr(nat_worker_module, "add_monitor_route", AsyncMock())
    monkeypatch.setattr(nat_worker_module, "add_health_route", AsyncMock())
    monkeypatch.setattr(nat_worker_module, "add_static_files_route", AsyncMock())
    monkeypatch.setattr(nat_worker_module, "add_chat_routes", AsyncMock())
    monkeypatch.setattr(nat_worker_module, "add_websocket_routes", AsyncMock())
    monkeypatch.setattr(worker, "initialize_evaluators", AsyncMock())
    monkeypatch.setattr(worker, "_install_signal_handlers", MagicMock())

    async def add_guarded_job_routes(app, _builder, configured_worker):
        assert configured_worker._dask_available is True

        @app.post("/v1/jobs/async/submit")
        async def submit_guarded_job():
            return {"job_id": "guarded"}

    monkeypatch.setattr(plugin, "register_job_routes", add_guarded_job_routes)

    app = FastAPI()
    await worker.add_routes(app, MagicMock())
    paths = {route.path for route in app.routes}

    assert "/v1/workflow" in paths
    assert "/generate" in paths
    assert "/v1/jobs/async/submit" in paths

    assert "/v1/workflow/async" not in paths
    assert "/v1/workflow/async/job/{job_id}" not in paths
    assert "/generate/async" not in paths
    assert "/generate/async/job/{job_id}" not in paths
    assert worker._dask_available is True
