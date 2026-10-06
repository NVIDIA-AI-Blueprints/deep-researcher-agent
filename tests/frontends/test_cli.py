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

import logging
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from deep_researcher_research_cli import cli


@pytest.mark.parametrize(
    ("arguments", "expected_level"),
    [([], logging.WARNING), (["--verbose"], logging.INFO)],
)
def test_main_sets_log_level_from_verbose_flag(monkeypatch, arguments: list[str], expected_level: int) -> None:
    observed: dict[str, int] = {}

    class LoggingConfigured(Exception):
        pass

    def configure_logging(**kwargs) -> None:
        observed["level"] = kwargs["level"]
        raise LoggingConfigured

    monkeypatch.setattr(sys, "argv", ["deep-researcher-research", *arguments])
    monkeypatch.setattr(cli.logging, "basicConfig", configure_logging)

    with pytest.raises(LoggingConfigured):
        cli.main()

    assert observed == {"level": expected_level}


@pytest.mark.parametrize("flush_fails", [False, True])
@pytest.mark.asyncio
async def test_interactive_loop_flushes_relay_before_display_and_next_prompt(monkeypatch, flush_fails: bool) -> None:
    events: list[str] = []
    responses = iter(["research this", "q"])

    async def prompt_async(*args, **kwargs):  # noqa: ARG001
        events.append("prompt")
        return next(responses)

    async def flush_async() -> None:
        events.append("flush")
        if flush_fails:
            raise RuntimeError("private relay failure")

    class Runner:
        async def result(self, *, to_type):  # noqa: ARG002
            events.append("result")
            return "answer"

    class Session:
        @asynccontextmanager
        async def run(self, user_input):  # noqa: ARG002
            yield Runner()
            events.append("run-exit")

    class SessionManager:
        @asynccontextmanager
        async def session(self, *, user_input_callback):  # noqa: ARG002
            yield Session()

    monkeypatch.setattr(cli.prompt_session, "prompt_async", prompt_async)
    monkeypatch.setattr(cli.nemo_relay.subscribers, "flush_async", flush_async)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "parse_and_display_response", lambda *args, **kwargs: events.append("display"))
    monkeypatch.setattr(
        cli.ContextState,
        "get",
        lambda: SimpleNamespace(conversation_id=SimpleNamespace(set=lambda value: None)),
    )

    await cli.interactive_loop(SessionManager(), verbose=True)

    assert events == ["prompt", "result", "run-exit", "flush", "display", "prompt"]
