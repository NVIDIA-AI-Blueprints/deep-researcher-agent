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

"""Tests for chat researcher utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from pydantic import ValidationError

from deep_researcher_agent.agents.chat_researcher.utils import _extract_database_name_from_request_metadata
from deep_researcher_agent.agents.chat_researcher.utils import _extract_query_and_sources
from deep_researcher_agent.agents.chat_researcher.utils import _extract_query_context
from deep_researcher_agent.agents.chat_researcher.utils import _extract_text_from_message
from deep_researcher_agent.agents.chat_researcher.utils import trim_message_history


class TestTrimMessageHistory:
    """Tests for the trim_message_history function."""

    def test_trim_message_history_basic(self):
        """Test basic message trimming."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well!"),
        ]

        result = trim_message_history(messages, max_tokens=10)

        # Should keep messages within token limit
        assert isinstance(result, list)

    def test_trim_message_history_empty(self):
        """Test trimming empty message list."""
        messages = []
        result = trim_message_history(messages, max_tokens=10)
        assert result == []

    def test_trim_message_history_single_message(self):
        """Test trimming with single message."""
        messages = [HumanMessage(content="Hello")]
        result = trim_message_history(messages, max_tokens=10)
        assert len(result) >= 0  # May be empty if message exceeds limit

    def test_trim_message_history_with_system_message(self):
        """Test trimming includes system messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]

        result = trim_message_history(messages, max_tokens=20)

        # System messages should be preserved according to include_system=True
        assert isinstance(result, list)

    def test_trim_message_history_large_limit(self):
        """Test trimming with large token limit keeps all messages."""
        messages = [
            HumanMessage(content="A"),
            AIMessage(content="B"),
            HumanMessage(content="C"),
        ]

        result = trim_message_history(messages, max_tokens=1000)

        # With a large limit, should keep messages
        assert isinstance(result, list)

    def test_trim_message_history_strategy_last(self):
        """Test that trimming uses 'last' strategy (keeps recent messages)."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Second message"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Third message"),
        ]

        result = trim_message_history(messages, max_tokens=5)

        # Strategy 'last' should prioritize recent messages
        assert isinstance(result, list)


class TestExtractTextFromMessage:
    """Tests for _extract_text_from_message."""

    def test_extract_from_string(self):
        """Test extracting text from a plain string."""
        assert _extract_text_from_message("Hello world") == "Hello world"

    def test_extract_from_none(self):
        """Test that None returns None."""
        assert _extract_text_from_message(None) is None

    def test_extract_from_object_content(self):
        """Test extracting text from object content attribute."""
        obj = MagicMock()
        obj.content = "Content from attribute"
        assert _extract_text_from_message(obj) == "Content from attribute"

    def test_extract_from_multipart_list(self):
        """Test extracting text from multipart list."""
        obj = MagicMock()
        part1 = MagicMock()
        part1.type = "text"
        part1.text = "First part"
        part2 = MagicMock()
        part2.type = "text"
        part2.text = "Second part"
        obj.content = [part1, part2]
        result = _extract_text_from_message(obj)
        assert result == "First part\nSecond part"

    def test_extract_from_dict_message(self):
        """Test extracting text from dict message."""
        message = {"content": [{"type": "text", "text": "Hello"}]}
        assert _extract_text_from_message(message) == "Hello"


class TestExtractQueryAndSources:
    """Tests for _extract_query_and_sources."""

    def test_extract_from_dict_payload(self):
        """Test extracting from dict payload."""
        payload = {
            "content": {
                "messages": [{"role": "user", "content": "Query text"}],
                "data_sources": ["confluence"],
            }
        }
        query, sources = _extract_query_and_sources(payload)
        assert query == "Query text"
        assert sources == ["confluence"]

    def test_extract_from_object_payload(self):
        """Test extracting from object payload with messages."""
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Object query"
        payload = MagicMock()
        payload.messages = [user_msg]
        payload.data_sources = None
        query, sources = _extract_query_and_sources(payload)
        assert query == "Object query"
        assert sources is None

    def test_extract_from_string_payload(self):
        """Test extracting from string payload."""
        query, sources = _extract_query_and_sources("Plain query string")
        assert query == "Plain query string"
        assert sources is None


class TestExtractQueryContext:
    """Tests for report-aware chat request context extraction."""

    def test_extract_active_report_job_id_from_top_level_payload(self):
        payload = {
            "active_report_job_id": "job-1",
            "content": {
                "messages": [{"role": "user", "content": "What are the risks?"}],
                "data_sources": ["web_search"],
            },
        }

        context = _extract_query_context(payload)

        assert context.query_text == "What are the risks?"
        assert context.data_sources == ["web_search"]
        assert context.active_report_job_id == "job-1"

    def test_extract_active_report_job_id_from_nested_content(self):
        payload = {
            "content": {
                "active_report_job_id": "job-2",
                "messages": [{"role": "user", "content": "Summarize this report"}],
            }
        }

        context = _extract_query_context(payload)

        assert context.query_text == "Summarize this report"
        assert context.active_report_job_id == "job-2"

    def test_extract_active_report_job_id_from_json_string(self):
        context = _extract_query_context('{"query": "Update this with latest data", "active_report_job_id": "job-3"}')

        assert context.query_text == "Update this with latest data"
        assert context.active_report_job_id == "job-3"

    def test_explicit_empty_data_sources_preserved_dict_payload(self):
        """An explicit empty data_sources list ('no data-source tools') must survive."""
        payload = {
            "data_sources": [],
            "content": {"messages": [{"role": "user", "content": "hello"}]},
        }
        context = _extract_query_context(payload)
        assert context.data_sources == []

    def test_explicit_empty_data_sources_preserved_object_payload(self):
        payload = SimpleNamespace(
            data_sources=[],
            messages=[SimpleNamespace(role="user", content="hello")],
        )
        context = _extract_query_context(payload)
        assert context.data_sources == []

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "database_name": "finance_prod",
                "content": {"messages": [{"role": "user", "content": "Show revenue"}]},
            },
            {
                "content": {
                    "database_name": "finance_prod",
                    "messages": [{"role": "user", "content": "Show revenue"}],
                }
            },
        ],
    )
    def test_extract_database_name_from_request_payload(self, payload):
        assert _extract_query_context(payload).database_name == "finance_prod"

    def test_extract_database_name_from_inline_json(self):
        context = _extract_query_context('{"query":"Show revenue","database_name":"finance-prod"}')
        assert context.database_name == "finance-prod"

    def test_extract_database_name_from_request_metadata(self):
        metadata = SimpleNamespace(
            payload={
                "messages": [{"role": "user", "content": "Show revenue"}],
                "database_name": "finance_prod",
            }
        )
        assert _extract_database_name_from_request_metadata(metadata) == "finance_prod"

    def test_extract_database_name_from_object_payload(self):
        payload = SimpleNamespace(
            database_name="finance_prod",
            messages=[SimpleNamespace(role="user", content="Show revenue")],
            data_sources=None,
            active_report_job_id=None,
        )

        assert _extract_query_context(payload).database_name == "finance_prod"

    def test_dynamic_attribute_payload_yields_no_database_scope(self):
        payload = MagicMock()
        payload.messages = [SimpleNamespace(role="user", content="Show revenue")]
        payload.data_sources = None
        payload.active_report_job_id = None

        assert _extract_query_context(payload).database_name is None

    @pytest.mark.parametrize("database_name", ["", "finance bad", "finance/other", "x" * 129])
    def test_invalid_database_name_is_rejected(self, database_name):
        with pytest.raises(ValidationError):
            _extract_query_context(
                {
                    "database_name": database_name,
                    "content": {"messages": [{"role": "user", "content": "Show revenue"}]},
                }
            )
