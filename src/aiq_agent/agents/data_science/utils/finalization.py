# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reserve a no-tool model turn before the graph recursion boundary."""

from __future__ import annotations

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import SystemMessage

from .analysis_runtime import get_analysis_run

_FINALIZATION_INSTRUCTION = """
FINALIZATION TURN: Tool use is now disabled so the request completes before the
runtime step limit. Synthesize the best supported final answer from evidence
already present in the conversation. Do not ask a question, request another
tool, or describe work you would do later. Preserve requested answer-choice
labels exactly when present, and include the required Sources section. State
material missing evidence as a bounded caveat rather than abandoning the answer.
""".strip()


class FinalizationReserveMiddleware(AgentMiddleware):
    """Force a no-tool synthesis call after a bounded number of model turns."""

    def __init__(self, max_model_calls: int) -> None:
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        self.max_model_calls = max_model_calls

    async def awrap_model_call(self, request, handler):
        """Count model calls and remove tools once the finalization reserve begins."""

        run_state = get_analysis_run()
        if run_state is None:
            return await handler(request)
        run_state.model_calls += 1
        if not run_state.force_finalization and run_state.model_calls < self.max_model_calls:
            return await handler(request)
        run_state.force_finalization = True
        prior = request.system_message.text if request.system_message is not None else ""
        instruction = run_state.finalization_instruction or _FINALIZATION_INSTRUCTION
        system_message = SystemMessage(content=f"{prior}\n\n{instruction}".strip())
        return await handler(
            request.override(
                system_message=system_message,
                tools=[],
                tool_choice=None,
            )
        )


__all__ = ["FinalizationReserveMiddleware"]
