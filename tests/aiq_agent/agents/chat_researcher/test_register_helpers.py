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

"""Tests for chat_researcher register.py helper functions."""

import logging
import uuid
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.chat_researcher.models import RESEARCH_WORKFLOW_FAILURE_ERROR
from aiq_agent.agents.chat_researcher.models import WorkflowFailure
from aiq_agent.agents.chat_researcher.models import WorkflowSuccess
from aiq_agent.agents.chat_researcher.utils import _extract_query_and_sources
from aiq_agent.agents.chat_researcher.utils import _extract_text_from_message
from aiq_agent.common.logging_utils import log_identifier_ref


def test_conversation_log_uses_opaque_reference(caplog) -> None:
    from aiq_agent.agents.chat_researcher.register import _log_conversation_reference

    conversation_id = str(uuid.uuid4())
    caplog.set_level(logging.INFO, logger="aiq_agent.agents.chat_researcher.register")

    _log_conversation_reference("Thread reference for checkpointing: %s", conversation_id)

    assert conversation_id not in caplog.text
    assert log_identifier_ref(conversation_id) in caplog.text


class TestWorkflowResponse:
    def test_success_contains_explicit_result(self):
        from aiq_agent.agents.chat_researcher.register import _render_workflow_response

        response = _render_workflow_response(
            {"messages": [AIMessage(content="research answer")]},
            model="workflow",
        )

        assert response.workflow_outcome == WorkflowSuccess(result="research answer")
        assert response.choices[0].message.content == "research answer"

    def test_failure_preserves_fallback_text_and_public_error(self):
        from aiq_agent.agents.chat_researcher.register import _render_workflow_response

        failure = WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        response = _render_workflow_response(
            {"messages": [AIMessage(content="Please try again.")], "workflow_outcome": failure},
            model="workflow",
        )

        assert response.workflow_outcome == failure
        assert response.choices[0].message.content == "Please try again."


class TestReportFollowUpHelpers:
    """Tests for report follow-up helper prompt shaping."""

    def test_build_report_ask_prompt_anchors_answer_to_parent_report(self):
        from aiq_agent.agents.chat_researcher.register import _build_report_ask_prompt

        prompt = _build_report_ask_prompt(
            question="What is the main risk?",
            report_markdown="# Report\n\nRisk is rollout complexity.",
            source_summary_markdown="- [1] https://example.com",
        )

        assert "What is the main risk?" in prompt
        assert "# Report" in prompt
        assert "Risk is rollout complexity." in prompt
        assert "- [1] https://example.com" in prompt
        assert "Answer using only the parent report" in prompt

    @pytest.mark.asyncio
    async def test_answer_from_report_context_uses_single_bounded_llm_call(self):
        """Report Q&A answers with one direct LLM call over the bounded prompt.

        It must NOT route through the shallow/deep research agents or any tools —
        report ask is bounded to parent report context and never triggers live research.
        """
        from aiq_agent.agents.chat_researcher.register import _answer_from_report_context

        captured = {}

        class FakeLLM:
            async def ainvoke(self, messages):
                captured["messages"] = messages
                return AIMessage(content="Bounded answer from report.")

        answer = await _answer_from_report_context(
            FakeLLM(),
            question="What is the main risk?",
            report_markdown="# Report\n\nRisk is rollout complexity.",
            source_summary_markdown="- [1] https://example.com",
        )

        assert answer == "Bounded answer from report."
        sent = captured["messages"][0].content
        assert "What is the main risk?" in sent
        assert "Risk is rollout complexity." in sent
        assert "Answer using only the parent report" in sent

    @pytest.mark.asyncio
    async def test_answer_from_report_context_falls_back_on_empty_response(self):
        """An empty/whitespace LLM completion yields a bounded fallback, never a blank answer."""
        from aiq_agent.agents.chat_researcher.register import _answer_from_report_context

        class EmptyLLM:
            async def ainvoke(self, messages):
                return AIMessage(content="   ")

        answer = await _answer_from_report_context(
            EmptyLLM(),
            question="What is the risk?",
            report_markdown="# Report",
            source_summary_markdown="",
        )

        assert answer.strip()
        assert "does not contain enough information" in answer.lower()

    @pytest.mark.asyncio
    async def test_answer_from_report_context_reraises_timeout(self):
        from aiq_agent.agents.chat_researcher.register import _answer_from_report_context

        class TimedOutLLM:
            async def ainvoke(self, messages):
                raise TimeoutError

        with pytest.raises(TimeoutError):
            await _answer_from_report_context(
                TimedOutLLM(),
                question="What is the risk?",
                report_markdown="# Report",
                source_summary_markdown="",
            )


class TestExtractTextFromMessageString:
    """Tests for _extract_text_from_message with string inputs."""

    def test_extract_from_string(self):
        """Test extracting text from a plain string."""
        result = _extract_text_from_message("Hello world")
        assert result == "Hello world"

    def test_extract_from_none(self):
        """Test that None returns None."""
        result = _extract_text_from_message(None)
        assert result is None

    def test_extract_from_empty_string(self):
        """Test extracting from empty string."""
        result = _extract_text_from_message("")
        assert result == ""


class TestExtractTextFromMessageObject:
    """Tests for _extract_text_from_message with message objects."""

    def test_extract_from_human_message(self):
        """Test extracting from HumanMessage."""
        message = HumanMessage(content="User query")
        result = _extract_text_from_message(message)
        assert result == "User query"

    def test_extract_from_ai_message(self):
        """Test extracting from AIMessage."""
        message = AIMessage(content="AI response")
        result = _extract_text_from_message(message)
        assert result == "AI response"

    def test_extract_from_object_with_content_attribute(self):
        """Test extracting from object with content attribute."""
        obj = MagicMock()
        obj.content = "Content from attribute"
        result = _extract_text_from_message(obj)
        assert result == "Content from attribute"


class TestExtractTextFromMessageMultipart:
    """Tests for _extract_text_from_message with multipart content."""

    def test_extract_from_multipart_content_list(self):
        """Test extracting from message with list content."""
        obj = MagicMock()
        part1 = MagicMock()
        part1.type = "text"
        part1.text = "First part"
        part2 = MagicMock()
        part2.type = "text"
        part2.text = "Second part"
        obj.content = [part1, part2]

        result = _extract_text_from_message(obj)
        assert "First part" in result
        assert "Second part" in result

    def test_extract_from_dict_multipart(self):
        """Test extracting from dict with multipart content."""
        message = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "url": "http://example.com"},
                {"type": "text", "text": "World"},
            ]
        }
        result = _extract_text_from_message(message)
        assert "Hello" in result
        assert "World" in result

    def test_extract_skips_non_text_parts(self):
        """Test that non-text parts are skipped."""
        obj = MagicMock()
        text_part = MagicMock()
        text_part.type = "text"
        text_part.text = "Text content"
        image_part = MagicMock()
        image_part.type = "image"
        obj.content = [text_part, image_part]

        result = _extract_text_from_message(obj)
        assert result == "Text content"


class TestExtractTextFromMessageDict:
    """Tests for _extract_text_from_message with dict inputs."""

    def test_extract_from_dict_with_content(self):
        """Test extracting from dict with content key."""
        message = {"content": "Dict content"}
        result = _extract_text_from_message(message)
        assert result == "Dict content"

    def test_extract_from_dict_with_text_key(self):
        """Test extracting from dict with text key."""
        message = {"text": "Text value"}
        result = _extract_text_from_message(message)
        assert result == "Text value"

    def test_extract_from_dict_content_takes_precedence(self):
        """Test that content key is preferred over text key in dict lists."""
        message = {"content": "Content value", "text": "Text value"}
        result = _extract_text_from_message(message)
        assert result == "Content value"


class TestExtractQueryAndSourcesDict:
    """Tests for _extract_query_and_sources with dict payloads."""

    def test_extract_from_simple_dict(self):
        """Test extracting from simple dict structure."""
        payload = {
            "content": {
                "messages": [{"role": "user", "content": "What is AI?"}],
                "data_sources": ["web_search"],
            }
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "What is AI?"
        assert sources == ["web_search"]

    def test_extract_from_dict_with_message_key(self):
        """Test extracting from dict with message key."""
        payload = {"message": "Direct message"}
        query, sources = _extract_query_and_sources(payload)
        assert query == "Direct message"
        assert sources is None

    def test_extract_from_dict_with_text_key(self):
        """Test extracting from dict with text key."""
        payload = {"text": "Text message"}
        query, sources = _extract_query_and_sources(payload)
        assert query == "Text message"
        assert sources is None

    def test_extract_prefers_last_user_message(self):
        """Test that last user message is preferred."""
        payload = {
            "content": {
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                ]
            }
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Second question"
        assert sources is None

    def test_extract_data_sources_from_payload_level(self):
        """Test extracting data_sources from payload level."""
        payload = {
            "data_sources": ["confluence", "sharepoint"],
            "content": {
                "messages": [{"role": "user", "content": "Query"}],
            },
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Query"
        assert sources == ["confluence", "sharepoint"]

    def test_extract_data_sources_from_content_level(self):
        """Test extracting data_sources from content level."""
        payload = {
            "content": {
                "messages": [{"role": "user", "content": "Query"}],
                "data_sources": ["google_drive"],
            },
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Query"
        assert sources == ["google_drive"]


class TestExtractQueryAndSourcesObject:
    """Tests for _extract_query_and_sources with object payloads."""

    def test_extract_from_object_with_messages(self):
        """Test extracting from object with messages attribute."""
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Object query"

        payload = MagicMock()
        payload.messages = [user_msg]
        payload.data_sources = None

        query, sources = _extract_query_and_sources(payload)
        assert query == "Object query"
        assert sources is None

    def test_extract_from_object_with_data_sources(self):
        """Test extracting from object with data_sources attribute."""
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Query with sources"

        payload = MagicMock()
        payload.messages = [user_msg]
        payload.data_sources = ["web_search", "confluence"]

        query, sources = _extract_query_and_sources(payload)
        assert query == "Query with sources"
        assert sources == ["web_search", "confluence"]


class TestExtractQueryAndSourcesString:
    """Tests for _extract_query_and_sources with string payloads."""

    def test_extract_from_plain_string(self):
        """Test extracting from plain string (no inline JSON)."""
        query, sources = _extract_query_and_sources("Plain query string")
        assert query == "Plain query string"
        assert sources is None

    def test_extract_from_json_string(self):
        """Test extracting from JSON string."""
        payload = '{"query": "JSON query", "data_sources": ["web_search"]}'
        query, sources = _extract_query_and_sources(payload)
        assert query == "JSON query"
        assert sources == ["web_search"]


class TestExtractQueryAndSourcesInlineJson:
    """Tests for inline JSON extraction in messages."""

    def test_extract_inline_json_in_message_content(self):
        """Test extracting inline JSON from message content."""
        payload = {
            "content": {
                "messages": [
                    {
                        "role": "user",
                        "content": ('{"query": "Inline JSON", "data_sources": ["sharepoint"]}'),
                    }
                ]
            }
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Inline JSON"
        assert sources == ["sharepoint"]

    def test_inline_sources_used_when_payload_sources_missing(self):
        """Test inline sources are used when payload has no sources."""
        payload = {
            "content": {
                "messages": [
                    {
                        "role": "user",
                        "content": ('{"query": "Test", "data_sources": ["jira"]}'),
                    }
                ]
            }
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Test"
        assert sources == ["jira"]

    def test_payload_sources_take_precedence_over_inline(self):
        """Test that payload-level sources take precedence."""
        payload = {
            "data_sources": ["confluence"],
            "content": {
                "messages": [
                    {
                        "role": "user",
                        "content": ('{"query": "Test", "data_sources": ["jira"]}'),
                    }
                ]
            },
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Test"
        assert sources == ["confluence"]


class TestExtractQueryAndSourcesEdgeCases:
    """Edge case tests for _extract_query_and_sources."""

    def test_extract_empty_messages_list(self):
        """Test with empty messages list."""
        payload = {"content": {"messages": []}}
        query, sources = _extract_query_and_sources(payload)
        assert query == ""
        assert sources is None

    def test_extract_no_user_messages(self):
        """Test with no user messages in list."""
        payload = {
            "content": {
                "messages": [
                    {"role": "assistant", "content": "AI response"},
                    {"role": "system", "content": "System prompt"},
                ]
            }
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "System prompt"
        assert sources is None

    def test_extract_with_langchain_messages(self):
        """Test with actual LangChain message objects in dict payload."""
        payload = {
            "content": {
                "messages": [
                    {"role": "user", "content": "LangChain query"},
                    {"role": "assistant", "content": "LangChain response"},
                ]
            }
        }

        query, sources = _extract_query_and_sources(payload)
        assert query == "LangChain query"
        assert sources is None

    def test_extract_with_explicit_data_sources_list(self):
        """Test with explicitly specified data sources list."""
        payload = {
            "data_sources": ["web_search", "confluence"],
            "content": {
                "messages": [{"role": "user", "content": "Query"}],
            },
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Query"
        assert sources == ["web_search", "confluence"]


class TestResolveEffectiveReportJobId:
    """Precedence + conversation-scoped fallback for the active report job."""

    @pytest.mark.asyncio
    async def test_client_supplied_id_wins_without_lookup(self, monkeypatch):
        from aiq_agent.agents.chat_researcher.register import _resolve_effective_report_job_id
        from aiq_api.jobs import access

        lookup = MagicMock(return_value="from-db")
        monkeypatch.setattr(access, "get_latest_report_job_for_conversation", lookup)

        result = await _resolve_effective_report_job_id(
            "client-id", conversation_id="conv-A", principal=None, is_input_mode=False
        )
        assert result == "client-id"
        lookup.assert_not_called()  # precedence short-circuits the DB query

    @pytest.mark.asyncio
    async def test_falls_back_to_conversation_last_report(self, monkeypatch):
        from aiq_agent.agents.chat_researcher.register import _resolve_effective_report_job_id
        from aiq_api.jobs import access

        monkeypatch.setattr(access, "get_latest_report_job_for_conversation", MagicMock(return_value="last-report"))

        result = await _resolve_effective_report_job_id(
            None, conversation_id="conv-A", principal=None, is_input_mode=False
        )
        assert result == "last-report"

    @pytest.mark.asyncio
    async def test_no_fallback_for_input_mode_or_missing_conversation(self, monkeypatch):
        from aiq_agent.agents.chat_researcher.register import _resolve_effective_report_job_id
        from aiq_api.jobs import access

        lookup = MagicMock(return_value="should-not-be-used")
        monkeypatch.setattr(access, "get_latest_report_job_for_conversation", lookup)

        # --input mode: deliberately discards continuity
        assert await _resolve_effective_report_job_id(None, "conv-A", None, is_input_mode=True) is None
        # no client conversation id (generated thread): no fallback
        assert await _resolve_effective_report_job_id(None, None, None, is_input_mode=False) is None
        lookup.assert_not_called()

    @pytest.mark.asyncio
    async def test_lookup_error_degrades_to_none(self, monkeypatch):
        from aiq_agent.agents.chat_researcher.register import _resolve_effective_report_job_id
        from aiq_api.jobs import access

        monkeypatch.setattr(
            access, "get_latest_report_job_for_conversation", MagicMock(side_effect=RuntimeError("db down"))
        )
        result = await _resolve_effective_report_job_id(None, "conv-A", None, is_input_mode=False)
        assert result is None


class TestResolveSubmissionQuery:
    """Query derivation for async research submissions.

    Regression cover for the hybrid route, which never passes through
    ``clarifier_node`` and so never has ``original_query`` set.
    """

    @staticmethod
    def _state(messages, original_query=None):
        state = MagicMock()
        state.messages = messages
        state.original_query = original_query
        return state

    def test_uses_the_latest_user_message_across_turns(self) -> None:
        from aiq_agent.agents.chat_researcher.register import _resolve_submission_query

        state = self._state(
            [
                HumanMessage(content="What tables exist?"),
                AIMessage(content="Several."),
                HumanMessage(content="Which state has the most orders?"),
            ]
        )

        assert _resolve_submission_query(state) == "Which state has the most orders?"

    def test_latest_message_wins_over_a_stale_original_query(self) -> None:
        """A previous turn's original_query must not leak into this turn's job."""
        from aiq_agent.agents.chat_researcher.register import _resolve_submission_query

        state = self._state(
            [
                HumanMessage(content="What tables exist?"),
                AIMessage(content="Several."),
                HumanMessage(content="Which state has the most orders?"),
            ],
            original_query="What tables exist?",
        )

        assert _resolve_submission_query(state) == "Which state has the most orders?"

    def test_falls_back_to_original_query_when_no_user_message(self) -> None:
        from aiq_agent.agents.chat_researcher.register import _resolve_submission_query

        state = self._state([AIMessage(content="Only assistant turns.")], original_query="Earlier question")

        assert _resolve_submission_query(state) == "Earlier question"

    def test_raises_without_messages(self) -> None:
        from aiq_agent.agents.chat_researcher.register import _resolve_submission_query

        with pytest.raises(RuntimeError, match="without messages"):
            _resolve_submission_query(self._state([]))
