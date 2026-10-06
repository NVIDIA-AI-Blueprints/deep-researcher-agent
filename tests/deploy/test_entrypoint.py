# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment entrypoint invariants for embedded and shared Dask modes."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import call

import pytest
from distributed import Client
from distributed import LocalCluster


def _load_entrypoint():
    path = Path(__file__).parents[2] / "deploy" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("deep_researcher_deploy_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_dask_scheduler_is_loopback_only() -> None:
    entrypoint = _load_entrypoint()

    command = entrypoint._scheduler_args(8786)

    assert entrypoint._DASK_LOOPBACK_HOST == "127.0.0.1"
    assert command[command.index("--host") + 1] == entrypoint._DASK_LOOPBACK_HOST
    assert command[command.index("--dashboard-address") + 1] == f"{entrypoint._DASK_LOOPBACK_HOST}:8787"
    assert command[command.index("--port") + 1] == "8786"


def test_embedded_dask_worker_is_loopback_only_and_preserves_limits() -> None:
    entrypoint = _load_entrypoint()

    command = entrypoint._worker_args(
        8786,
        "2",
        "3",
        memory_limit="800MB",
        lifetime="3600s",
        lifetime_restart=True,
    )

    assert command[1] == f"tcp://{entrypoint._DASK_LOOPBACK_HOST}:8786"
    assert command[command.index("--host") + 1] == entrypoint._DASK_LOOPBACK_HOST
    assert command[command.index("--dashboard-address") + 1] == f"{entrypoint._DASK_LOOPBACK_HOST}:0"
    assert command[command.index("--nworkers") + 1] == "2"
    assert command[command.index("--nthreads") + 1] == "3"
    assert command[command.index("--memory-limit") + 1] == "800MB"
    assert command[command.index("--lifetime") + 1] == "3600s"
    assert "--lifetime-restart" in command
    assert "--no-dashboard" in command


def test_embedded_dask_worker_can_disable_lifetime_restart() -> None:
    entrypoint = _load_entrypoint()

    command = entrypoint._worker_args(
        8786,
        "1",
        "4",
        lifetime="3600s",
        lifetime_restart=False,
    )

    assert "--lifetime" in command
    assert "--lifetime-restart" not in command
    assert "--memory-limit" not in command


def test_worker_and_web_processes_receive_scheduler_address(monkeypatch) -> None:
    entrypoint = _load_entrypoint()
    processes = [MagicMock(), MagicMock(), MagicMock()]
    for process in processes:
        process.poll.return_value = 0
        process.wait.return_value = 0
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.setattr(entrypoint.subprocess, "Popen", popen)
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", MagicMock())
    monkeypatch.setattr(entrypoint, "_install_signal_handlers", MagicMock())
    monkeypatch.setattr(entrypoint.time, "sleep", MagicMock())

    assert entrypoint.main() == 0

    expected_address = f"tcp://{entrypoint._DASK_LOOPBACK_HOST}:8786"
    worker_env = popen.call_args_list[1].kwargs["env"]
    web_env = popen.call_args_list[2].kwargs["env"]
    assert worker_env["NAT_DASK_SCHEDULER_ADDRESS"] == expected_address
    assert web_env["NAT_DASK_SCHEDULER_ADDRESS"] == expected_address


def test_external_scheduler_starts_only_web_server(monkeypatch) -> None:
    entrypoint = _load_entrypoint()
    scheduler_address = "tls://deep-researcher-dask-scheduler:8786"
    wait_for_scheduler = MagicMock()
    run_web_server = MagicMock(return_value=17)
    popen = MagicMock(side_effect=AssertionError("external mode must not launch local Dask processes"))
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", scheduler_address)
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", wait_for_scheduler)
    monkeypatch.setattr(entrypoint, "_run_web_server", run_web_server)
    monkeypatch.setattr(entrypoint.subprocess, "Popen", popen)

    assert entrypoint.main() == 17
    wait_for_scheduler.assert_called_once_with(scheduler_address)
    run_web_server.assert_called_once_with()
    popen.assert_not_called()


def test_external_scheduler_rejects_plaintext_address(monkeypatch) -> None:
    entrypoint = _load_entrypoint()
    wait_for_scheduler = MagicMock()
    run_web_server = MagicMock()
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://deep-researcher-dask-scheduler:8786")
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", wait_for_scheduler)
    monkeypatch.setattr(entrypoint, "_run_web_server", run_web_server)

    with pytest.raises(SystemExit, match="must use tls://"):
        entrypoint.main()

    wait_for_scheduler.assert_not_called()
    run_web_server.assert_not_called()


@pytest.mark.parametrize(
    "scheduler_address",
    (
        "tls://",
        "tls://scheduler",
        "tls://:8786",
        "tls://scheduler:",
        "tls://scheduler:not-a-port",
        "tls://scheduler:0",
    ),
)
def test_external_scheduler_rejects_incomplete_tls_address(monkeypatch, scheduler_address: str) -> None:
    entrypoint = _load_entrypoint()
    wait_for_scheduler = MagicMock()
    run_web_server = MagicMock()
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", scheduler_address)
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", wait_for_scheduler)
    monkeypatch.setattr(entrypoint, "_run_web_server", run_web_server)

    with pytest.raises(SystemExit, match="must include a valid host and port"):
        entrypoint.main()

    wait_for_scheduler.assert_not_called()
    run_web_server.assert_not_called()


@pytest.mark.parametrize(
    "scheduler_address",
    (
        "tls://user@scheduler:8786",
        "tls://user:password@scheduler:8786",  # pragma: allowlist secret
        "tls://scheduler:8786/path",
        "tls://scheduler:8786?option=value",
        "tls://scheduler:8786#fragment",
    ),
)
def test_external_scheduler_rejects_additional_url_components(monkeypatch, scheduler_address: str) -> None:
    entrypoint = _load_entrypoint()
    wait_for_scheduler = MagicMock()
    run_web_server = MagicMock()
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", scheduler_address)
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", wait_for_scheduler)
    monkeypatch.setattr(entrypoint, "_run_web_server", run_web_server)

    with pytest.raises(SystemExit, match="must not include userinfo, path, query, or fragment"):
        entrypoint.main()

    wait_for_scheduler.assert_not_called()
    run_web_server.assert_not_called()


def test_worker_startup_failure_cleans_up_scheduler(monkeypatch) -> None:
    entrypoint = _load_entrypoint()
    scheduler_proc = MagicMock()
    popen = MagicMock(side_effect=[scheduler_proc, OSError("worker startup failed")])
    terminate_process = MagicMock()
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.delenv("NAT_DASK_SCHEDULER_ADDRESS", raising=False)
    monkeypatch.setattr(entrypoint.subprocess, "Popen", popen)
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", MagicMock())
    monkeypatch.setattr(entrypoint, "_terminate_process", terminate_process)

    with pytest.raises(OSError, match="worker startup failed"):
        entrypoint.main()

    assert popen.call_count == 2
    terminate_process.assert_called_once_with(scheduler_proc)


def test_web_startup_failure_cleans_up_local_dask(monkeypatch) -> None:
    entrypoint = _load_entrypoint()
    scheduler_proc = MagicMock()
    worker_proc = MagicMock()
    popen = MagicMock(side_effect=[scheduler_proc, worker_proc, OSError("web startup failed")])
    terminate_process = MagicMock()
    monkeypatch.setattr(entrypoint.sys, "argv", ["entrypoint.py"])
    monkeypatch.delenv("NAT_DASK_SCHEDULER_ADDRESS", raising=False)
    monkeypatch.setattr(entrypoint.subprocess, "Popen", popen)
    monkeypatch.setattr(entrypoint, "_wait_for_scheduler", MagicMock())
    monkeypatch.setattr(entrypoint, "_terminate_process", terminate_process)
    monkeypatch.setattr(entrypoint.time, "sleep", MagicMock())

    with pytest.raises(OSError, match="web startup failed"):
        entrypoint.main()

    assert popen.call_count == 3
    assert terminate_process.call_args_list == [
        call(None),
        call(worker_proc),
        call(scheduler_proc),
    ]


def test_embedded_dask_worker_advertises_only_loopback() -> None:
    entrypoint = _load_entrypoint()
    worker_proc: subprocess.Popen[str] | None = None

    with (
        LocalCluster(
            n_workers=0,
            host=entrypoint._DASK_LOOPBACK_HOST,
            protocol="tcp",
            dashboard_address=None,
        ) as cluster,
        Client(cluster) as client,
    ):
        scheduler_port = cluster.scheduler_address.rsplit(":", maxsplit=1)[1]
        worker_proc = subprocess.Popen(
            entrypoint._worker_args(int(scheduler_port), "1", "1"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            client.wait_for_workers(1, timeout="20s")
            workers = client.scheduler_info()["workers"]
            assert len(workers) == 1
            worker_address, worker_info = next(iter(workers.items()))
            assert worker_address.startswith(f"tcp://{entrypoint._DASK_LOOPBACK_HOST}:")
            assert worker_info["host"] == entrypoint._DASK_LOOPBACK_HOST
            assert worker_info["nanny"].startswith(f"tcp://{entrypoint._DASK_LOOPBACK_HOST}:")
        finally:
            entrypoint._terminate_process(worker_proc)
