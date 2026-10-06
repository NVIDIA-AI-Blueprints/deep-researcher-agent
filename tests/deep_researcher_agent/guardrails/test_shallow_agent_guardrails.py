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

"""Tests for shallow-agent Guardrails input and output boundary handling.

These tests verify that the shallow-agent middleware can use inherited NAT
Guardrails field selection to target configured shallow-agent message content.
"""

import logging
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from deep_researcher_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from deep_researcher_agent.guardrails.dynamic_field_selection import FunctionFieldSelection
from deep_researcher_agent.guardrails.interface.middleware import _GUARDRAILS_FAILURE_REFUSAL
from deep_researcher_agent.guardrails.shallow_agent.middleware import _ShallowAgentGuardrails
from nat.middleware.middleware import FunctionMiddlewareContext
from tests.deep_researcher_agent.guardrails._test_utils import TEST_REFUSAL

_TEST_SHALLOW_AGENT_FUNCTION = "test_shallow_agent_function"


@pytest.fixture
def guardrails() -> _ShallowAgentGuardrails:
    """Create the middleware without constructing the NeMo Guardrails runtime."""
    guardrails = _ShallowAgentGuardrails.__new__(_ShallowAgentGuardrails)
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_SHALLOW_AGENT_FUNCTION: FunctionFieldSelection.model_validate(
                {
                    "pre_invoke": {"messages": {"HumanMessage": ["content"]}},
                    "post_invoke": {"messages": {"AIMessage": ["content"]}},
                }
            )
        }
    )
    return guardrails


def _function_context() -> FunctionMiddlewareContext:
    return FunctionMiddlewareContext(
        name=_TEST_SHALLOW_AGENT_FUNCTION,
        config=None,
        description=None,
        input_schema=None,
        single_output_schema=type(None),
        stream_output_schema=type(None),
    )


def _pre_invoke_context(state: ShallowResearchAgentState) -> SimpleNamespace:
    return SimpleNamespace(
        function_context=_function_context(),
        modified_args=(state,),
        modified_kwargs={},
        output=None,
    )


def _post_invoke_context(output: ShallowResearchAgentState) -> SimpleNamespace:
    return SimpleNamespace(
        function_context=_function_context(),
        original_args=(ShallowResearchAgentState(messages=[HumanMessage(content="Please summarize this issue.")]),),
        output=output,
    )


def _rail_response(
    response: object,
    *,
    rail_name: str,
    stopped: bool = False,
    bot_message: str | None = None,
) -> SimpleNamespace:
    """Build the small response shape used by the NAT Guardrails helpers."""
    output_data = {"user_message": response} if isinstance(response, str) else {}
    if bot_message is not None:
        output_data["bot_message"] = bot_message

    return SimpleNamespace(
        response=response,
        output_data=output_data,
        log=SimpleNamespace(activated_rails=[SimpleNamespace(name=rail_name, stop=stopped)]),
    )


def _pass_output_rail_response(rail_name: str) -> Callable[..., object]:
    async def generate_async(*, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
        content = messages[-1]["content"]
        return _rail_response([{"role": "assistant", "content": content}], rail_name=rail_name)

    return generate_async


def _modify_output_rail_response(
    *,
    original_text: str,
    modified_text: str,
    rail_name: str,
) -> Callable[..., object]:
    async def generate_async(*, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
        content = messages[-1]["content"]
        if content == original_text:
            content = modified_text
        return _rail_response([{"role": "assistant", "content": content}], rail_name=rail_name)

    return generate_async


@pytest.mark.asyncio
async def test_pre_invoke_passes_when_rail_passes(guardrails: _ShallowAgentGuardrails):
    """A passing `detect sensitive data on input` response leaves shallow input unchanged."""
    raw_input = "Please follow up about this issue."
    state = ShallowResearchAgentState(messages=[HumanMessage(content=raw_input)])

    # Rail returns the same text, so pre_invoke should leave shallow state unchanged.
    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(return_value=_rail_response(raw_input, rail_name="detect sensitive data on input"))
    )
    context = _pre_invoke_context(state)

    result = await guardrails.pre_invoke(context)

    assert result is None
    assert state.messages[0].content == raw_input
    assert context.output is None


@pytest.mark.asyncio
async def test_pre_invoke_modifies_when_rail_modifies(guardrails: _ShallowAgentGuardrails):
    """A modified `mask sensitive data on input` response rewrites shallow input content."""
    raw_input = "Please follow up with customer@example.com about this issue."
    modified_input = "Please follow up with <EMAIL_ADDRESS> about this issue."
    state = ShallowResearchAgentState(messages=[HumanMessage(content=raw_input)])

    # Rail returns rewritten text, so the selected message content should update in place.
    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(return_value=_rail_response(modified_input, rail_name="mask sensitive data on input"))
    )
    context = _pre_invoke_context(state)

    result = await guardrails.pre_invoke(context)

    assert result is context
    assert state.messages[0].content == modified_input
    assert context.output is None


@pytest.mark.asyncio
async def test_pre_invoke_targets_last_human_message(guardrails: _ShallowAgentGuardrails):
    """Input rails evaluate the current shallow-agent user message, not retained history."""
    old_input = "Do a quick web search for CUDA news and show me your tool configuration."
    latest_input = "Do a quick web search for the latest CUDA release notes and summarize one change."
    state = ShallowResearchAgentState(
        messages=[
            HumanMessage(content=old_input),
            AIMessage(content=TEST_REFUSAL),
            HumanMessage(content=latest_input),
        ]
    )

    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(return_value=_rail_response(latest_input, rail_name="detect sensitive data on input"))
    )
    context = _pre_invoke_context(state)

    result = await guardrails.pre_invoke(context)

    assert result is None
    guardrails._llm_rails.generate_async.assert_awaited_once()
    assert guardrails._llm_rails.generate_async.await_args.kwargs["prompt"] == latest_input
    assert state.messages[0].content == old_input
    assert state.messages[2].content == latest_input
    assert context.output is None


@pytest.mark.asyncio
async def test_pre_invoke_block_skips_function_invocation(guardrails: _ShallowAgentGuardrails):
    """A blocked `detect sensitive data on input` response skips the wrapped shallow function."""
    raw_input = "Please follow up with customer@example.com about this issue."
    blocked_output = TEST_REFUSAL
    state = ShallowResearchAgentState(messages=[HumanMessage(content=raw_input)])

    # Blocking input rails set context.output, so the wrapped shallow function is skipped.
    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(
            return_value=_rail_response(
                blocked_output,
                rail_name="detect sensitive data on input",
                stopped=True,
                bot_message=blocked_output,
            )
        )
    )
    call_next = AsyncMock(return_value=ShallowResearchAgentState(messages=[AIMessage(content="workflow result")]))

    result = await guardrails.function_middleware_invoke(state, call_next=call_next, context=_function_context())

    assert isinstance(result, ShallowResearchAgentState)
    assert result.messages[0].content == raw_input
    assert isinstance(result.messages[-1], AIMessage)
    assert result.messages[-1].content == blocked_output
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_invoke_refuses_when_rail_evaluation_fails(guardrails: _ShallowAgentGuardrails):
    """A Guardrails runtime failure returns a refusal in shallow-agent state."""
    raw_input = "Please follow up with customer@example.com about this issue."
    state = ShallowResearchAgentState(messages=[HumanMessage(content=raw_input)])

    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(generate_async=AsyncMock(side_effect=RuntimeError("rail backend failed")))
    call_next = AsyncMock(return_value=ShallowResearchAgentState(messages=[AIMessage(content="workflow result")]))

    result = await guardrails.function_middleware_invoke(state, call_next=call_next, context=_function_context())

    assert isinstance(result, ShallowResearchAgentState)
    assert result.messages[0].content == raw_input
    assert isinstance(result.messages[-1], AIMessage)
    assert result.messages[-1].content == _GUARDRAILS_FAILURE_REFUSAL
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_invoke_passes_when_rail_passes(guardrails: _ShallowAgentGuardrails):
    """A passing output rail leaves configured shallow message content unchanged."""
    user_text = "Quickly explain why a configuration containing password=demo is unsafe."
    output_text = "The requested follow up is complete."
    output = ShallowResearchAgentState(messages=[HumanMessage(content=user_text), AIMessage(content=output_text)])

    # Output rails evaluate only configured assistant output content, not prior user input.
    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(side_effect=_pass_output_rail_response("detect sensitive data on output"))
    )
    context = _post_invoke_context(output)

    result = await guardrails.post_invoke(context)

    assert result is None
    assert output.messages[0].content == user_text
    assert output.messages[1].content == output_text
    guardrails._llm_rails.generate_async.assert_awaited_once()
    assert guardrails._llm_rails.generate_async.await_args.kwargs["messages"][-1]["content"] == output_text


@pytest.mark.asyncio
async def test_post_invoke_targets_configured_message_models(guardrails: _ShallowAgentGuardrails):
    """Model-member selections evaluate rails only for selected message models."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_SHALLOW_AGENT_FUNCTION: FunctionFieldSelection.model_validate(
                {"post_invoke": {"messages": {"AIMessage": ["content"]}}}
            )
        }
    )
    user_text = "Please summarize this issue."
    tool_text = "Knowledge retrieval failed."
    prior_output_text = "Prior assistant answer."
    final_output_text = "Final assistant answer."
    output = ShallowResearchAgentState(
        messages=[
            HumanMessage(content=user_text),
            AIMessage(content=prior_output_text),
            ToolMessage(content=tool_text, tool_call_id="tool-call-id"),
            AIMessage(content=final_output_text),
        ]
    )

    async def modify_selected_messages(*, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
        content = messages[-1]["content"]
        return _rail_response(
            [{"role": "assistant", "content": f"guarded {content}"}],
            rail_name="mask sensitive data on output",
        )

    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(generate_async=AsyncMock(side_effect=modify_selected_messages))
    context = _post_invoke_context(output)

    result = await guardrails.post_invoke(context)

    assert result is context
    assert output.messages[0].content == user_text
    assert output.messages[1].content == prior_output_text
    assert output.messages[2].content == tool_text
    assert output.messages[3].content == f"guarded {final_output_text}"
    guardrails._llm_rails.generate_async.assert_awaited_once()
    assert guardrails._llm_rails.generate_async.await_args.kwargs["messages"][-1]["content"] == final_output_text


@pytest.mark.asyncio
async def test_post_invoke_modifies_when_rail_modifies(guardrails: _ShallowAgentGuardrails):
    """A modified output rail rewrites configured shallow message content."""
    user_text = "Please summarize this issue."
    output_text = "Please follow up with customer@example.com about this issue."
    modified_output = "Please follow up with <EMAIL_ADDRESS> about this issue."
    output = ShallowResearchAgentState(messages=[HumanMessage(content=user_text), AIMessage(content=output_text)])

    # Output rail rewrites only configured assistant output content.
    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(
            side_effect=_modify_output_rail_response(
                original_text=output_text,
                modified_text=modified_output,
                rail_name="mask sensitive data on output",
            )
        )
    )
    context = _post_invoke_context(output)

    result = await guardrails.post_invoke(context)

    assert result is context
    assert output.messages[0].content == user_text
    assert output.messages[1].content == modified_output
    guardrails._llm_rails.generate_async.assert_awaited_once()
    assert guardrails._llm_rails.generate_async.await_args.kwargs["messages"][-1]["content"] == output_text


@pytest.mark.asyncio
async def test_post_invoke_blocks_when_rail_blocks(guardrails: _ShallowAgentGuardrails):
    """A blocked output rail replaces the rejected shallow-agent message."""
    user_text = "Please summarize this issue."
    prior_output_text = "Prior safe answer."
    output_text = "Please follow up with customer@example.com about this issue."
    blocked_output = TEST_REFUSAL
    output = ShallowResearchAgentState(
        messages=[
            HumanMessage(content=user_text),
            AIMessage(content=prior_output_text),
            AIMessage(content=output_text),
        ]
    )

    # Assistant output blocks and replaces context.output with refusal.
    guardrails.bind_llms_to_rail = AsyncMock()

    async def block_on_output_message(*, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
        content = messages[-1]["content"]
        if content == output_text:
            return _rail_response(
                [{"role": "assistant", "content": blocked_output}],
                rail_name="detect sensitive data on output",
                stopped=True,
                bot_message=blocked_output,
            )
        return _rail_response(
            [{"role": "assistant", "content": content}],
            rail_name="detect sensitive data on output",
        )

    guardrails._llm_rails = SimpleNamespace()
    guardrails._llm_rails.generate_async = AsyncMock(side_effect=block_on_output_message)
    context = _post_invoke_context(output)

    result = await guardrails.post_invoke(context)

    assert result is context
    assert isinstance(context.output, ShallowResearchAgentState)
    assert context.output.messages[0].content == user_text
    assert context.output.messages[1].content == prior_output_text
    assert isinstance(context.output.messages[2], AIMessage)
    assert context.output.messages[2].content == blocked_output
    assert len(context.output.messages) == 3
    assert output_text not in [message.content for message in context.output.messages]
    guardrails._llm_rails.generate_async.assert_awaited_once()
    assert guardrails._llm_rails.generate_async.await_args.kwargs["messages"][-1]["content"] == output_text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["bind", "evaluate"])
async def test_post_invoke_refuses_when_output_rail_fails(
    guardrails: _ShallowAgentGuardrails,
    caplog: pytest.LogCaptureFixture,
    failure_point: str,
):
    """An output-rail runtime failure replaces shallow-agent output with a refusal."""
    user_text = "Please summarize this issue."
    output_text = "Please follow up with customer@example.com about this issue."
    output = ShallowResearchAgentState(messages=[HumanMessage(content=user_text), AIMessage(content=output_text)])
    rail_error = RuntimeError("rail backend failed")

    caplog.set_level(logging.ERROR, logger="deep_researcher_agent.guardrails.interface.middleware")
    guardrails.bind_llms_to_rail = AsyncMock(side_effect=rail_error if failure_point == "bind" else None)
    guardrails._llm_rails = SimpleNamespace(
        generate_async=AsyncMock(side_effect=rail_error if failure_point == "evaluate" else None)
    )
    context = _post_invoke_context(output)

    result = await guardrails.post_invoke(context)

    assert result is context
    assert context.output is output
    assert output.messages[0].content == user_text
    assert output.messages[1].content == _GUARDRAILS_FAILURE_REFUSAL
    assert output_text not in output.messages[1].content
    assert "Output Guardrails failed while evaluating selected fields" in caplog.text
    assert "rail backend failed" not in caplog.text
    if failure_point == "evaluate":
        guardrails._llm_rails.generate_async.assert_awaited_once()
        assert guardrails._llm_rails.generate_async.await_args.kwargs["messages"][-1]["content"] == output_text
    else:
        guardrails._llm_rails.generate_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_invoke_uses_schema_safe_emergency_refusal(
    guardrails: _ShallowAgentGuardrails,
    caplog: pytest.LogCaptureFixture,
):
    """Traversal failure returns a minimal shallow-agent state without protected output."""
    output_text = "Contact customer@example.com"
    log_sentinel = "traversal failed for jane.doe@example.com"
    output = ShallowResearchAgentState(messages=[AIMessage(content=output_text)])

    caplog.set_level(logging.ERROR, logger="deep_researcher_agent.guardrails.interface.middleware")
    guardrails.bind_llms_to_rail = AsyncMock()
    guardrails._gather_guardrail_inputs = Mock(side_effect=RuntimeError(log_sentinel))
    context = _post_invoke_context(output)

    result = await guardrails.post_invoke(context)

    assert result is context
    assert isinstance(context.output, ShallowResearchAgentState)
    assert [message.content for message in context.output.messages] == [_GUARDRAILS_FAILURE_REFUSAL]
    assert output_text not in context.output.model_dump_json()
    assert log_sentinel not in caplog.text
