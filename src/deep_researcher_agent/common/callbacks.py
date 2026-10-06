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

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.common.logging_utils import log_identifier_ref

logger = logging.getLogger(__name__)


SUPPRESS_OUTPUT_ARTIFACT_TAG = "deep-researcher:suppress-output-artifact"


YELLOW = "\033[33m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[39m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET_ALL = "\033[0m"


class ResearchLogger:
    """Colored logging utilities for research agents."""

    def __init__(self, logger_instance: logging.Logger, verbose: bool = False):
        self.logger = logger_instance
        self.verbose = verbose

    def section(self, label: str, message: str, *args):
        self.logger.info(f"{BOLD}[{label}]{RESET_ALL} {message}", *args)

    def success(self, label: str, message: str, *args):
        self.logger.info(f"{GREEN}[{label}]{RESET} {message}", *args)

    def info(self, label: str, message: str, *args):
        self.logger.info(f"{CYAN}[{label}]{RESET} {message}", *args)

    def detail(self, message: str, *args):
        if self.verbose:
            self.logger.info(f"{DIM}  {message}{RESET_ALL}", *args)

    def item(self, label: str, message: str, *args):
        self.logger.info(f"{MAGENTA}[{label}]{RESET} {message}", *args)

    def result(self, label: str, message: str, *args):
        self.logger.info(f"{GREEN}[{label}]{RESET} {message}", *args)

    def warning(self, label: str, message: str, *args):
        self.logger.warning(f"{YELLOW}[{label}]{RESET} {message}", *args)

    def error(self, label: str, message: str, *args):
        self.logger.error(f"{RED}[{label}]{RESET} {message}", *args)

    def skip(self, label: str, message: str, *args):
        self.logger.info(f"{DIM}[{label}]{RESET_ALL} {message}", *args)

    def query(self, query_id: str, query_text: str):
        self.logger.info(f"{MAGENTA}[Query]{RESET} query_ref={log_identifier_ref(query_id)}")
        if self.verbose:
            self.logger.info(f"{YELLOW}  Query: {log_content_metadata(query_text)}{RESET}")

    def tool_call(self, tool_name: str, input_preview: str | None = None):
        self.logger.info(f"{GREEN}[Tool Call]{RESET} {tool_name}")
        if self.verbose and input_preview:
            self.logger.info(f"{DIM}  Input: {log_content_metadata(input_preview)}{RESET_ALL}")

    def tool_result(self, tool_name: str, result_preview: str | None = None, chars: int | None = None):
        msg = f"{tool_name}: {chars} chars" if chars is not None else tool_name
        self.logger.info(f"{GREEN}[Tool Result]{RESET} {msg}")
        if self.verbose and result_preview:
            self.logger.debug(f"{DIM}  Result: {log_content_metadata(result_preview)}{RESET_ALL}")

    def relevancy(self, relevant: int, total: int, reasoning: str | None = None):
        self.logger.info(f"{CYAN}[Relevancy]{RESET} {relevant}/{total} relevant")
        if self.verbose and reasoning:
            self.logger.info(f"{DIM}  Reasoning: {log_content_metadata(reasoning)}{RESET_ALL}")

    def relevant_item(self, title: str, url: str | None = None):
        if self.verbose:
            self.logger.info(f"{GREEN}    [Relevant]{RESET} title_{log_content_metadata(title)}")
            if url:
                self.logger.info(f"{DIM}      URL: {log_content_metadata(url)}{RESET_ALL}")

    def banner(self, agent_name: str, query: str, **kwargs):
        self.logger.info("=" * 80)
        self.logger.info(f"{BOLD}[{agent_name}]{RESET_ALL} Starting research")
        self.logger.info(f"{YELLOW}Query:{RESET} {log_content_metadata(query)}")
        for key, value in kwargs.items():
            self.logger.info(f"{DIM}{key}: {log_content_metadata(value)}{RESET_ALL}")
        self.logger.info("=" * 80)


class VerboseTraceCallback(BaseCallbackHandler):
    """Callback handler that logs detailed agent traces with live output."""

    def __init__(self, log_reasoning: bool = True, max_chars: int = 5000):
        self.log_reasoning = log_reasoning
        self.max_chars = max_chars
        self.current_input: str | None = None
        self.active_chains: dict[Any, str] = {}
        self.depth = 0

    def _get_indent(self) -> str:
        return "  " * self.depth

    def on_chain_start(self, serialized: dict | None, inputs: dict, **kwargs) -> None:
        run_id = kwargs.get("run_id")

        if serialized:
            chain_name = serialized.get("name") or (
                serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "chain"
            )
        else:
            chain_name = kwargs.get("name", "chain")

        if run_id:
            self.active_chains[run_id] = chain_name
            self.depth += 1

        if inputs and "messages" in inputs:
            messages = inputs["messages"]
            if messages and hasattr(messages[-1], "content"):
                self.current_input = log_content_metadata(messages[-1].content)

        if "subagent" in chain_name.lower() or "agent" in chain_name.lower():
            logger.info("%s%s[Chain Start] %s%s", self._get_indent(), CYAN, chain_name, RESET)

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        run_id = kwargs.get("run_id")
        if run_id and run_id in self.active_chains:
            chain_name = self.active_chains.pop(run_id)
            if "subagent" in chain_name.lower() or "agent" in chain_name.lower():
                logger.info("%s%s[Chain End] %s%s", self._get_indent(), CYAN, chain_name, RESET)
            self.depth = max(0, self.depth - 1)

    def on_llm_start(self, serialized: dict | None, prompts: list[str], **kwargs) -> None:
        if serialized:
            model_name = serialized.get("name") or (
                serialized.get("id", ["LLM"])[-1] if serialized.get("id") else "LLM"
            )
        else:
            model_name = kwargs.get("name", "LLM")

        print(flush=True)
        logger.info("-" * 30)
        logger.info("%s[AGENT]%s %s", BOLD, RESET_ALL, model_name)
        if self.current_input:
            logger.info("%sAgent input: %s%s", YELLOW, self.current_input, RESET)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        if not response.generations:
            return

        generation = response.generations[0][0]
        message = generation.message if hasattr(generation, "message") else None

        if message:
            self._log_message_details(message)
        else:
            text = generation.text if hasattr(generation, "text") else str(generation)
            logger.info("%sAgent response: %s%s", CYAN, log_content_metadata(text), RESET)

        logger.info("-" * 30)
        print(flush=True)

    def _log_message_details(self, message: Any) -> None:
        reasoning_content = None
        if hasattr(message, "additional_kwargs"):
            reasoning_content = message.additional_kwargs.get("reasoning_content")

        if reasoning_content and self.log_reasoning:
            logger.info("%s[Reasoning] %s%s", MAGENTA, log_content_metadata(reasoning_content), RESET_ALL)

        content = getattr(message, "content", "")
        if content:
            logger.info("%s[Agent Response] %s%s", CYAN, log_content_metadata(content), RESET)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            logger.info("%s[Tool Calls] %d tool(s) requested%s", GREEN, len(tool_calls), RESET)
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                logger.info("%s  → %s%s", GREEN, tool_name, RESET)
                logger.info("%s    Args: %s%s", DIM, log_content_metadata(tool_args), RESET_ALL)

        if hasattr(message, "response_metadata"):
            metadata = message.response_metadata
            token_usage = metadata.get("token_usage", {})
            model_name = metadata.get("model_name", "unknown")
            if token_usage:
                logger.info(
                    "%s[Tokens] prompt=%s, completion=%s, model=%s%s",
                    DIM,
                    token_usage.get("prompt_tokens", "N/A"),
                    token_usage.get("completion_tokens", "N/A"),
                    model_name,
                    RESET_ALL,
                )

    def on_tool_start(self, serialized: dict | None, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown") if serialized else kwargs.get("name", "unknown")
        logger.info("%s[Tool Start] %s%s", GREEN, tool_name, RESET)
        logger.info("%s  Input: %s%s", DIM, log_content_metadata(input_str), RESET_ALL)

    def on_tool_end(self, output: str, **kwargs) -> None:
        logger.info("%s[Tool Result] %s%s", GREEN, log_content_metadata(output), RESET)

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        logger.error(
            "%s[Tool Error] error_type=%s detail_%s%s",
            RED,
            type(error).__name__,
            log_content_metadata(error),
            RESET,
        )

    def on_agent_action(self, action: Any, **kwargs) -> None:
        tool_name = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", {})
        logger.info("%s[Agent Action] Delegating to: %s%s", MAGENTA, tool_name, RESET)
        if isinstance(tool_input, dict) and "messages" in tool_input:
            msg_content = tool_input["messages"][-1].content if tool_input["messages"] else ""
            logger.info("%s  Task: %s%s", DIM, log_content_metadata(msg_content), RESET_ALL)

    def on_agent_finish(self, finish: Any, **kwargs) -> None:
        output = getattr(finish, "return_values", {})
        if output:
            logger.info("%s[Agent Finish] %s%s", GREEN, log_content_metadata(output), RESET)
