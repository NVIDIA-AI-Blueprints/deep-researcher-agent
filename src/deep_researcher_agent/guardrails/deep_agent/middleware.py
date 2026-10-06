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

"""Deep-agent Guardrails middleware."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage

from deep_researcher_agent.agents.deep_researcher.models import DeepResearchAgentState
from deep_researcher_agent.guardrails.deep_agent.config import DeepAgentGuardrailsConfig
from deep_researcher_agent.guardrails.interface.middleware import _GUARDRAILS_FAILURE_REFUSAL
from deep_researcher_agent.guardrails.interface.middleware import GuardrailsMixin
from nat.builder.builder import Builder
from nat.middleware.middleware import InvocationContext


class _DeepAgentGuardrails(GuardrailsMixin):
    """Provide Guardrails enforcement for deep-agent boundaries.

    This middleware evaluates configured policies around deep-agent invocation:
    selected input and output fields are checked, and blocked responses are
    returned in the deep-agent state schema when possible.
    """

    def __init__(self, config: DeepAgentGuardrailsConfig, builder: Builder):
        """Initialize deep-agent Guardrails with its registered config."""
        super().__init__(config=config, builder=builder)

    def _build_blocked_agent_state(
        self,
        context: InvocationContext,
        block_message: str,
        original_output: object | None = None,
    ) -> DeepResearchAgentState | str:
        """Return a deep-agent response that preserves the agent state schema.

        When a deep-agent state is available, the refusal is appended as the
        next assistant message for input blocks so downstream callers still
        receive ``DeepResearchAgentState``. If no state can be resolved, return
        the raw refusal text.
        """
        state = self._get_deep_agent_state_from_invocation_context(context, original_output)
        if state is None:
            return block_message
        return state.model_copy(update={"messages": [*state.messages, AIMessage(content=block_message)]})

    def _get_deep_agent_state_from_invocation_context(
        self,
        context: InvocationContext,
        original_output: object | None = None,
    ) -> DeepResearchAgentState | None:
        """Return the deep-agent state available from invocation context."""
        modified_args = getattr(context, "modified_args", ())
        original_args = getattr(context, "original_args", ())
        for value in (
            original_output,
            context.output,
            modified_args[0] if modified_args else None,
            original_args[0] if original_args else None,
        ):
            if isinstance(value, DeepResearchAgentState):
                return value
        return None

    def _extract_latest_message_text(
        self,
        value: Any,
        path_parts: list[str],
    ) -> tuple[str, Callable[[str], None]] | None:
        """Return the latest message text selected by path with a setter for rewrites."""
        if isinstance(value, list):
            for index in range(len(value) - 1, -1, -1):
                item = value[index]
                if not path_parts and isinstance(item, str):
                    return item, self._set_modified_rail_value_in_list(value, index)

                selection = self._extract_latest_message_text(item, path_parts)
                if selection is not None:
                    return selection

        if not path_parts:
            return None

        segment, *remaining_path = path_parts

        if value.__class__.__name__ == segment:
            return self._extract_latest_message_text(value, remaining_path)

        attr: Any = getattr(value, segment, None)
        if attr is None:
            return None

        if remaining_path:
            return self._extract_latest_message_text(attr, remaining_path)

        if isinstance(attr, str):
            return attr, self._set_modified_rail_value(value, segment)

        if isinstance(attr, list):
            return self._extract_latest_message_text(attr, [])

        return None

    # -------------------------------------------------------------------------
    # GuardrailsMixin override hooks
    # -------------------------------------------------------------------------

    def _on_pre_invoke_blocked(self, context: InvocationContext, block_message: str) -> DeepResearchAgentState | str:
        """Return a deep-agent state when input rails block."""
        return self._build_blocked_agent_state(context, block_message)

    def _on_post_invoke_blocked(
        self,
        context: InvocationContext,
        block_message: str,
        original_output: object,
    ) -> DeepResearchAgentState | str:
        """Replace blocked deep-agent output content with the refusal."""
        return super()._on_post_invoke_blocked(context, block_message, original_output)

    def _build_emergency_output_refusal(
        self,
        context: InvocationContext,
        original_output: object,
    ) -> DeepResearchAgentState:
        """Return a minimal refusal state without traversing protected output."""
        return DeepResearchAgentState(messages=[AIMessage(content=_GUARDRAILS_FAILURE_REFUSAL)])

    def _iter_targets_at_path(self, value: Any, path: str) -> Iterator[tuple[str, Callable[[str], None]]]:
        """Yield the latest deep-agent message for the inherited field-selection hook."""
        selected_message = self._extract_latest_message_text(value, path.split("."))
        if selected_message is not None:
            yield selected_message
