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

"""Tests for callback handlers and logging utilities."""

import logging
from unittest.mock import MagicMock

import pytest

from deep_researcher_agent.common.callbacks import ResearchLogger
from deep_researcher_agent.common.callbacks import VerboseTraceCallback
from deep_researcher_agent.common.logging_utils import log_content_metadata


class TestResearchLogger:
    """Tests for the ResearchLogger class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock(spec=logging.Logger)

    def test_research_logger_init_with_verbose_param(self, mock_logger):
        """Test ResearchLogger initialization with explicit verbose parameter."""
        logger = ResearchLogger(mock_logger, verbose=True)
        assert logger.verbose is True

        logger_non_verbose = ResearchLogger(mock_logger, verbose=False)
        assert logger_non_verbose.verbose is False

    def test_research_logger_defaults_to_non_verbose(self, mock_logger, monkeypatch):
        """Research logging does not consult a process-global verbosity switch."""
        monkeypatch.setenv("DEEP_RESEARCHER_VERBOSE", "true")
        assert ResearchLogger(mock_logger).verbose is False

    def test_section_logs_info(self, mock_logger):
        """Test section method logs at info level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.section("TEST", "Test message")
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "TEST" in call_args
        assert "Test message" in call_args

    def test_success_logs_info(self, mock_logger):
        """Test success method logs at info level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.success("DONE", "Operation completed")
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "DONE" in call_args

    def test_info_logs_info(self, mock_logger):
        """Test info method logs at info level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.info("INFO", "Information message")
        mock_logger.info.assert_called_once()

    def test_detail_logs_only_when_verbose(self, mock_logger):
        """Test detail method only logs when verbose is True."""
        logger_verbose = ResearchLogger(mock_logger, verbose=True)
        logger_verbose.detail("Detail message")
        assert mock_logger.info.called

        mock_logger.reset_mock()
        logger_non_verbose = ResearchLogger(mock_logger, verbose=False)
        logger_non_verbose.detail("Detail message")
        assert not mock_logger.info.called

    def test_item_logs_info(self, mock_logger):
        """Test item method logs at info level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.item("ITEM", "Item content")
        mock_logger.info.assert_called_once()

    def test_result_logs_info(self, mock_logger):
        """Test result method logs at info level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.result("RESULT", "Result content")
        mock_logger.info.assert_called_once()

    def test_warning_logs_warning(self, mock_logger):
        """Test warning method logs at warning level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.warning("WARN", "Warning message")
        mock_logger.warning.assert_called_once()

    def test_error_logs_error(self, mock_logger):
        """Test error method logs at error level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.error("ERR", "Error message")
        mock_logger.error.assert_called_once()

    def test_skip_logs_info(self, mock_logger):
        """Test skip method logs at info level."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.skip("SKIP", "Skipped operation")
        mock_logger.info.assert_called_once()

    def test_query_logs_query_info(self, mock_logger):
        """Test query method logs query information."""
        logger = ResearchLogger(mock_logger, verbose=True)
        logger.query("q1", "What is AI?")
        assert mock_logger.info.call_count == 2  # query ID and query text

    def test_query_non_verbose(self, mock_logger):
        """Test query method in non-verbose mode."""
        logger = ResearchLogger(mock_logger, verbose=False)
        logger.query("q1", "What is AI?")
        assert mock_logger.info.call_count == 1  # Only query ID

    def test_tool_call_logs_info(self, mock_logger):
        """Test tool_call method logs tool information."""
        logger = ResearchLogger(mock_logger, verbose=True)
        logger.tool_call("web_search", "search query")
        assert mock_logger.info.call_count == 2

    def test_tool_result_logs_info(self, mock_logger):
        """Test tool_result method logs result information."""
        logger = ResearchLogger(mock_logger, verbose=True)
        logger.tool_result("web_search", "Some result", chars=100)
        mock_logger.info.assert_called()

    def test_relevancy_logs_info(self, mock_logger):
        """Test relevancy method logs relevancy information."""
        logger = ResearchLogger(mock_logger, verbose=True)
        logger.relevancy(5, 10, "Good matches found")
        assert mock_logger.info.call_count == 2

    def test_relevant_item_logs_only_when_verbose(self, mock_logger):
        """Test relevant_item method only logs when verbose."""
        logger = ResearchLogger(mock_logger, verbose=True)
        logger.relevant_item("Article Title", "http://example.com")
        assert mock_logger.info.call_count == 2

        mock_logger.reset_mock()
        logger_non_verbose = ResearchLogger(mock_logger, verbose=False)
        logger_non_verbose.relevant_item("Article Title", "http://example.com")
        assert not mock_logger.info.called

    def test_banner_logs_header(self, mock_logger):
        """Test banner method logs agent header."""
        logger = ResearchLogger(mock_logger, verbose=True)
        logger.banner("DeepResearcher", "What is quantum computing?", depth="deep")
        assert mock_logger.info.call_count >= 4

    def test_content_specific_methods_never_log_raw_values(self, caplog):
        """Verbose research logs retain correlation metadata, never customer content."""
        secret = "nvapi-vdr-fake-secret-do-not-log"  # pragma: allowlist secret
        test_logger = logging.getLogger("tests.safe_research_logger")
        caplog.set_level(logging.DEBUG, logger=test_logger.name)
        research_logger = ResearchLogger(test_logger, verbose=True)

        research_logger.query(secret, secret)
        research_logger.tool_call("web_search", secret)
        research_logger.tool_result("web_search", secret, chars=len(secret))
        research_logger.relevancy(1, 1, secret)
        research_logger.relevant_item(secret, secret)
        research_logger.banner("DeepResearcher", secret, detail=secret)

        assert secret not in caplog.text
        assert log_content_metadata(secret) in caplog.text


class TestVerboseTraceCallback:
    """Tests for the VerboseTraceCallback class."""

    def test_callback_init_defaults(self):
        """Test VerboseTraceCallback initialization with defaults."""
        callback = VerboseTraceCallback()
        assert callback.log_reasoning is True
        assert callback.max_chars == 5000
        assert callback.current_input is None
        assert callback.active_chains == {}
        assert callback.depth == 0

    def test_callback_init_custom_values(self):
        """Test VerboseTraceCallback initialization with custom values."""
        callback = VerboseTraceCallback(log_reasoning=False, max_chars=1000)
        assert callback.log_reasoning is False
        assert callback.max_chars == 1000

    def test_get_indent(self):
        """Test _get_indent returns correct indentation."""
        callback = VerboseTraceCallback()
        assert callback._get_indent() == ""

        callback.depth = 1
        assert callback._get_indent() == "  "

        callback.depth = 3
        assert callback._get_indent() == "      "

    def test_on_chain_start_increments_depth(self):
        """Test on_chain_start increments depth for tracked chains."""
        callback = VerboseTraceCallback()
        run_id = "test-run-id"

        callback.on_chain_start(
            serialized={"name": "TestAgent"},
            inputs={},
            run_id=run_id,
        )

        assert callback.depth == 1
        assert run_id in callback.active_chains
        assert callback.active_chains[run_id] == "TestAgent"

    def test_on_chain_start_extracts_input(self):
        """Test on_chain_start extracts current input from messages."""
        callback = VerboseTraceCallback()

        mock_message = MagicMock()
        mock_message.content = "User query content"

        callback.on_chain_start(
            serialized={"name": "Agent"},
            inputs={"messages": [mock_message]},
            run_id="run-123",
        )

        assert callback.current_input == log_content_metadata("User query content")

    def test_on_chain_end_decrements_depth(self):
        """Test on_chain_end decrements depth for tracked chains."""
        callback = VerboseTraceCallback()
        run_id = "test-run-id"

        # Start a chain
        callback.on_chain_start(
            serialized={"name": "TestAgent"},
            inputs={},
            run_id=run_id,
        )
        assert callback.depth == 1

        # End the chain
        callback.on_chain_end(outputs={}, run_id=run_id)
        assert callback.depth == 0
        assert run_id not in callback.active_chains

    def test_on_chain_end_depth_floor(self):
        """Test on_chain_end doesn't go below 0 depth."""
        callback = VerboseTraceCallback()
        callback.depth = 0

        callback.on_chain_end(outputs={}, run_id="unknown-id")
        assert callback.depth == 0

    def test_on_tool_start(self):
        """Test on_tool_start handles tool information."""
        callback = VerboseTraceCallback()

        # Should not raise
        callback.on_tool_start(
            serialized={"name": "web_search"},
            input_str="search query here",
        )

    def test_on_tool_end(self):
        """Test on_tool_end handles output."""
        callback = VerboseTraceCallback()

        # Should not raise
        callback.on_tool_end(output="Tool result content")

    def test_on_tool_error(self):
        """Test on_tool_error handles exceptions."""
        callback = VerboseTraceCallback()

        # Should not raise
        callback.on_tool_error(error=Exception("Test error"))

    def test_on_agent_action(self):
        """Test on_agent_action handles agent actions."""
        callback = VerboseTraceCallback()

        mock_action = MagicMock()
        mock_action.tool = "web_search"
        mock_action.tool_input = {"query": "test"}

        # Should not raise
        callback.on_agent_action(action=mock_action)

    def test_on_agent_finish(self):
        """Test on_agent_finish handles agent completion."""
        callback = VerboseTraceCallback()

        mock_finish = MagicMock()
        mock_finish.return_values = {"output": "Final answer"}

        # Should not raise
        callback.on_agent_finish(finish=mock_finish)

    def test_on_llm_start_with_serialized(self):
        """Test on_llm_start handles serialized LLM info."""
        callback = VerboseTraceCallback()
        callback.current_input = "Test input"

        # Should not raise
        callback.on_llm_start(
            serialized={"name": "gpt-4", "id": ["langchain", "llms", "openai"]},
            prompts=["Hello, how are you?"],
        )

    def test_on_llm_start_without_serialized(self):
        """Test on_llm_start without serialized info."""
        callback = VerboseTraceCallback()

        # Should not raise with None serialized
        callback.on_llm_start(
            serialized=None,
            prompts=["Test prompt"],
            name="TestLLM",
        )

    def test_on_llm_start_with_serialized_id(self):
        """Test on_llm_start extracts name from serialized id list."""
        callback = VerboseTraceCallback()

        callback.on_llm_start(
            serialized={"id": ["langchain", "chat_models", "ChatOpenAI"]},
            prompts=["Test"],
        )

    def test_on_llm_end_with_generations(self):
        """Test on_llm_end handles LLM result with generations."""
        callback = VerboseTraceCallback()

        mock_generation = MagicMock()
        mock_generation.text = "Generated response text"
        mock_generation.message = None

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_llm_end_empty_generations(self):
        """Test on_llm_end handles empty generations."""
        callback = VerboseTraceCallback()

        mock_result = MagicMock()
        mock_result.generations = []

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_llm_end_with_message(self):
        """Test on_llm_end handles generation with message."""
        callback = VerboseTraceCallback()

        mock_message = MagicMock()
        mock_message.content = "Message content"
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {}

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_llm_end_with_reasoning_content(self):
        """Test on_llm_end handles reasoning content."""
        callback = VerboseTraceCallback(log_reasoning=True)

        mock_message = MagicMock()
        mock_message.content = "Response content"
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {"reasoning_content": "This is the reasoning..."}

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_llm_end_with_tool_calls(self):
        """Test on_llm_end handles tool calls in message."""
        callback = VerboseTraceCallback()

        mock_message = MagicMock()
        mock_message.content = "Response with tool calls"
        mock_message.tool_calls = [
            {"name": "web_search", "args": {"query": "test query"}},
            {"name": "doc_search", "args": {"doc_id": "123"}},
        ]
        mock_message.additional_kwargs = {}

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_llm_end_with_response_metadata(self):
        """Test on_llm_end handles response metadata with token usage."""
        callback = VerboseTraceCallback()

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {}
        mock_message.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model_name": "gpt-4",
        }

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_llm_end_truncates_long_content(self):
        """Test on_llm_end truncates content exceeding max_chars."""
        callback = VerboseTraceCallback(max_chars=100)

        mock_message = MagicMock()
        mock_message.content = "A" * 200  # Content longer than max_chars
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {}

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        # Should not raise
        callback.on_llm_end(response=mock_result)

    def test_on_chain_start_with_id_list(self):
        """Test on_chain_start extracts name from id list."""
        callback = VerboseTraceCallback()

        callback.on_chain_start(
            serialized={"id": ["langchain", "agents", "MyAgent"]},
            inputs={},
            run_id="test-run",
        )

        assert callback.active_chains["test-run"] == "MyAgent"

    def test_on_chain_start_fallback_name(self):
        """Test on_chain_start uses fallback name."""
        callback = VerboseTraceCallback()

        callback.on_chain_start(
            serialized=None,
            inputs={},
            run_id="test-run",
            name="FallbackAgent",
        )

        assert callback.active_chains["test-run"] == "FallbackAgent"

    def test_on_agent_action_with_messages(self):
        """Test on_agent_action handles messages in tool_input."""
        callback = VerboseTraceCallback()

        mock_message = MagicMock()
        mock_message.content = "User query for the tool"

        mock_action = MagicMock()
        mock_action.tool = "sub_agent"
        mock_action.tool_input = {"messages": [mock_message]}

        # Should not raise
        callback.on_agent_action(action=mock_action)

    def test_on_agent_finish_empty_output(self):
        """Test on_agent_finish handles empty output."""
        callback = VerboseTraceCallback()

        mock_finish = MagicMock()
        mock_finish.return_values = {}

        # Should not raise
        callback.on_agent_finish(finish=mock_finish)

    def test_on_tool_start_without_serialized(self):
        """Test on_tool_start without serialized info."""
        callback = VerboseTraceCallback()

        callback.on_tool_start(
            serialized=None,
            input_str="Tool input",
            name="fallback_tool",
        )

    def test_on_tool_end_long_output(self):
        """Test on_tool_end truncates long output."""
        callback = VerboseTraceCallback()

        long_output = "A" * 2000  # Long output

        # Should not raise
        callback.on_tool_end(output=long_output)

    def test_log_message_details_no_reasoning(self):
        """Test _log_message_details with log_reasoning disabled."""
        callback = VerboseTraceCallback(log_reasoning=False)

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {"reasoning_content": "Reasoning text"}

        # Should not log reasoning when disabled
        callback._log_message_details(mock_message)

    def test_log_message_details_empty_content(self):
        """Test _log_message_details with empty content."""
        callback = VerboseTraceCallback()

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {}

        # Should not raise
        callback._log_message_details(mock_message)

    def test_verbose_callback_never_logs_prompt_model_or_tool_content(self, caplog):
        """Verbose traces expose only safe metadata across every content-bearing hook."""
        secret = "nvapi-vdr-fake-secret-do-not-log"  # pragma: allowlist secret
        caplog.set_level(logging.DEBUG, logger="deep_researcher_agent.common.callbacks")
        callback = VerboseTraceCallback(log_reasoning=True)

        input_message = MagicMock()
        input_message.content = secret
        callback.on_chain_start(
            serialized={"name": "TestAgent"},
            inputs={"messages": [input_message]},
            run_id="run-secret-test",
        )
        callback.on_llm_start(serialized={"name": "test-model"}, prompts=[secret])

        response_message = MagicMock()
        response_message.content = secret
        response_message.additional_kwargs = {"reasoning_content": secret}
        response_message.tool_calls = [{"name": "web_search", "args": {"query": secret}}]
        response_message.response_metadata = {}
        generation = MagicMock()
        generation.message = response_message
        result = MagicMock()
        result.generations = [[generation]]
        callback.on_llm_end(result)

        callback.on_tool_start({"name": "web_search"}, secret)
        callback.on_tool_end(secret)
        callback.on_tool_error(RuntimeError(secret))
        action = MagicMock()
        action.tool = "researcher"
        action.tool_input = {"messages": [input_message]}
        callback.on_agent_action(action)
        finish = MagicMock()
        finish.return_values = {"output": secret}
        callback.on_agent_finish(finish)

        assert secret not in caplog.text
        assert log_content_metadata(secret) in caplog.text
