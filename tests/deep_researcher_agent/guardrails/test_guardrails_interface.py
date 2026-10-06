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

"""Tests for shared Guardrails middleware dynamic field selection."""

import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import pytest
from pydantic import BaseModel

from deep_researcher_agent.guardrails.dynamic_field_selection import FunctionFieldSelection
from deep_researcher_agent.guardrails.interface.middleware import _NEMO_GUARDRAILS_RUNTIME_LOGGER
from deep_researcher_agent.guardrails.interface.middleware import GuardrailsMixin
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware import GuardrailsMiddleware

_TEST_FUNCTION = "test_guarded_function"


class _StringMessage(BaseModel):
    content: str


class _StringListMessage(BaseModel):
    content: str | list[str | dict[str, object]]


class _StringTupleMessage(BaseModel):
    content: tuple[str, ...]


class _MessageWithNonStringContent(BaseModel):
    content: int


class _HumanMessage(BaseModel):
    content: str


class _AIMessage(BaseModel):
    content: str


class _ToolMessage(BaseModel):
    content: str


class _StateWithAnnotatedUnion(BaseModel):
    messages: list[Annotated[_StringMessage | _StringListMessage, "message choices"]]


class _StateWithTupleMessages(BaseModel):
    messages: tuple[_StringMessage, ...]


class _StateWithSetMessages(BaseModel):
    messages: set[_StringMessage]


class _StateWithFrozenSetMessages(BaseModel):
    messages: frozenset[_StringMessage]


class _StateWithSequenceMessages(BaseModel):
    messages: Sequence[_StringMessage]


class _StateWithTupleMessageContent(BaseModel):
    messages: list[_StringTupleMessage]


class _StateWithInvalidAnnotatedUnion(BaseModel):
    messages: list[Annotated[_StringMessage | _MessageWithNonStringContent, "message choices"]]


class _StateWithMessageUnion(BaseModel):
    messages: list[Annotated[_HumanMessage | _AIMessage | _ToolMessage, "message choices"]]


@pytest.fixture
def guardrails() -> GuardrailsMixin:
    """Create the middleware without constructing the NeMo Guardrails runtime."""
    guardrails = GuardrailsMixin.__new__(GuardrailsMixin)
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={_TEST_FUNCTION: FunctionFieldSelection.model_validate({"messages": ["content"]})}
    )
    return guardrails


def test_initialization_suppresses_nemo_runtime_context_info_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Guardrails initialization prevents dependency INFO logs from exposing context."""

    def skip_base_initialization(
        _middleware: GuardrailsMiddleware,
        config: object,
        builder: object,
    ) -> None:
        del config, builder

    caplog.set_level(logging.INFO, logger=_NEMO_GUARDRAILS_RUNTIME_LOGGER)
    monkeypatch.setattr(GuardrailsMiddleware, "__init__", skip_base_initialization)

    GuardrailsMixin(config=SimpleNamespace(), builder=SimpleNamespace())

    runtime_logger = logging.getLogger(_NEMO_GUARDRAILS_RUNTIME_LOGGER)
    runtime_logger.info("event context includes api_key=fake-secret-value")
    runtime_logger.warning("runtime warning without request context")

    assert runtime_logger.level == logging.WARNING
    assert "fake-secret-value" not in caplog.text
    assert "runtime warning without request context" in caplog.text

    runtime_logger.setLevel(logging.ERROR)
    GuardrailsMixin(config=SimpleNamespace(), builder=SimpleNamespace())
    assert runtime_logger.level == logging.ERROR


def _discovered_function(input_schema: type[BaseModel]) -> SimpleNamespace:
    return SimpleNamespace(
        name=_TEST_FUNCTION,
        instance=SimpleNamespace(input_schema=input_schema, single_output_schema=type(None)),
    )


def test_validates_path_through_annotated_union(guardrails: GuardrailsMixin):
    """Config validation accepts paths shared by every annotated union member."""
    guardrails._validate_guarded_field_paths(_discovered_function(_StateWithAnnotatedUnion))


def test_rejects_invalid_path_through_annotated_union(
    guardrails: GuardrailsMixin,
):
    """Config validation rejects annotated unions with non-string target fields."""
    with pytest.raises(ValueError, match="messages.content"):
        guardrails._validate_guarded_field_paths(_discovered_function(_StateWithInvalidAnnotatedUnion))


@pytest.mark.parametrize(
    "input_schema",
    [
        _StateWithTupleMessages,
        _StateWithSetMessages,
        _StateWithFrozenSetMessages,
        _StateWithSequenceMessages,
        _StateWithTupleMessageContent,
    ],
)
def test_rejects_container_annotations_not_traversed_at_runtime(
    guardrails: GuardrailsMixin,
    input_schema: type[BaseModel],
):
    """Config validation rejects containers that runtime traversal cannot rewrite."""
    with pytest.raises(ValueError, match="messages.content"):
        guardrails._validate_guarded_field_paths(_discovered_function(input_schema))


def test_targets_configured_union_members_and_rewrites_in_place(guardrails: GuardrailsMixin):
    """Model-member field selections target only the selected union models."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_FUNCTION: FunctionFieldSelection.model_validate(
                {"messages": {"_AIMessage": ["content"], "_ToolMessage": ["content"]}}
            )
        }
    )
    state = _StateWithMessageUnion(
        messages=[
            _HumanMessage(content="user question"),
            _AIMessage(content="prior answer"),
            _ToolMessage(content="tool output"),
            _AIMessage(content="final answer"),
        ]
    )

    targets = list(
        guardrails._gather_guardrail_inputs(state, guardrails._resolve_guarded_targets(_TEST_FUNCTION), None)
    )

    assert [text for text, _setter in targets] == ["prior answer", "final answer", "tool output"]

    targets[0][1]("rewritten prior answer")
    targets[1][1]("rewritten final answer")
    targets[2][1]("rewritten tool output")

    assert state.messages[0].content == "user question"
    assert state.messages[1].content == "rewritten prior answer"
    assert state.messages[2].content == "rewritten tool output"
    assert state.messages[3].content == "rewritten final answer"


def test_shared_selection_applies_to_pre_and_post(guardrails: GuardrailsMixin):
    """Field selections without phase keys apply to both pre and post rails."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_FUNCTION: FunctionFieldSelection.model_validate({"messages": {"_HumanMessage": ["content"]}})
        }
    )

    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "pre_invoke") == [
        "messages._HumanMessage.content"
    ]
    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "post_invoke") == [
        "messages._HumanMessage.content"
    ]


def test_resolves_phase_specific_targets(guardrails: GuardrailsMixin):
    """Phase-specific field selections keep input and output targets separate."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_FUNCTION: FunctionFieldSelection.model_validate(
                {
                    "pre_invoke": {"messages": {"_HumanMessage": ["content"]}},
                    "post_invoke": {"messages": {"_AIMessage": ["content"]}},
                }
            )
        }
    )

    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "pre_invoke") == [
        "messages._HumanMessage.content"
    ]
    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "post_invoke") == [
        "messages._AIMessage.content"
    ]
    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, None) == [
        "messages._HumanMessage.content",
        "messages._AIMessage.content",
    ]

    state = _StateWithMessageUnion(
        messages=[
            _HumanMessage(content="user question"),
            _AIMessage(content="assistant answer"),
        ]
    )

    pre_targets = list(
        guardrails._gather_guardrail_inputs(
            state,
            guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "pre_invoke"),
            None,
        )
    )
    post_targets = list(
        guardrails._gather_guardrail_inputs(
            state,
            guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "post_invoke"),
            None,
        )
    )

    assert [text for text, _setter in pre_targets] == ["user question"]
    assert [text for text, _setter in post_targets] == ["assistant answer"]


def test_validates_phase_specific_targets(guardrails: GuardrailsMixin):
    """Config validation checks phase-specific field selections at startup."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_FUNCTION: FunctionFieldSelection.model_validate(
                {
                    "pre_invoke": {"messages": {"_HumanMessage": ["content"]}},
                    "post_invoke": {"messages": {"_AIMessage": ["content"]}},
                }
            )
        }
    )

    guardrails._validate_guarded_field_paths(_discovered_function(_StateWithMessageUnion))


def test_missing_phase_specific_selection_returns_no_targets(guardrails: GuardrailsMixin):
    """A phase without configured fields does not evaluate rails."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_FUNCTION: FunctionFieldSelection.model_validate(
                {"pre_invoke": {"messages": {"_HumanMessage": ["content"]}}}
            )
        }
    )

    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "pre_invoke") == [
        "messages._HumanMessage.content"
    ]
    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, "post_invoke") == []
    assert guardrails._resolve_guarded_targets_for_phase(_TEST_FUNCTION, None) == ["messages._HumanMessage.content"]


def test_rejects_invalid_union_member_selection(guardrails: GuardrailsMixin):
    """Config validation rejects model-member selections that are not in the union."""
    guardrails._guardrails_config = SimpleNamespace(
        workflow_functions={
            _TEST_FUNCTION: FunctionFieldSelection.model_validate(
                {"pre_invoke": {"messages": {"_MissingMessage": ["content"]}}}
            )
        }
    )

    with pytest.raises(ValueError, match="messages._MissingMessage.content"):
        guardrails._validate_guarded_field_paths(_discovered_function(_StateWithMessageUnion))
