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

"""Tests for the ChatResearcherAgent."""

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.chat_researcher.agent import ChatResearcherAgent
from aiq_agent.agents.chat_researcher.models import RESEARCH_WORKFLOW_FAILURE_ERROR
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.chat_researcher.models import ChatResearcherState
from aiq_agent.agents.chat_researcher.models import DepthDecision
from aiq_agent.agents.chat_researcher.models import IntentResult
from aiq_agent.agents.chat_researcher.models import WorkflowFailure
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import EmptySourceRegistryReason
from aiq_agent.common.logging_utils import log_content_metadata
from aiq_agent.common.logging_utils import log_identifier_ref


class TestChatResearcherAgent:
    """Tests for the ChatResearcherAgent class."""

    @pytest.fixture
    def mock_intent_classifier(self):
        """Create a mock combined orchestration (intent + depth + meta) function."""

        async def classifier(state):
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(
                    decision="shallow",
                    raw_reasoning="Simple query",
                ),
            }

        return classifier

    @pytest.fixture
    def mock_shallow_research(self):
        """Create a mock shallow research function."""

        async def shallow(state_input):
            messages = state_input.messages if hasattr(state_input, "messages") else state_input
            result = MagicMock()
            result.messages = list(messages) + [
                AIMessage(content="Here's a quick answer with sources."),
            ]
            return result

        return shallow

    @pytest.fixture
    def mock_deep_research(self):
        """Create a mock deep research function."""

        async def deep(state):
            result = MagicMock()
            result.messages = list(state.messages) + [
                AIMessage(content="Here's a comprehensive report."),
            ]
            return result

        return deep

    @pytest.fixture
    def mock_clarifier(self):
        """Create a mock clarifier function."""

        async def clarifier(state_input):
            messages = state_input.messages if hasattr(state_input, "messages") else state_input
            result = MagicMock()
            result.messages = list(messages)
            result.clarifier_log = "User clarified: technical focus"
            return result

        return clarifier

    def test_init_with_defaults(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test ChatResearcherAgent initialization with defaults."""
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        assert agent.enable_escalation is True
        assert agent.callbacks == []
        assert agent.graph is not None

    def test_init_with_escalation_disabled(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test ChatResearcherAgent initialization with escalation disabled."""
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_escalation=False,
        )

        assert agent.enable_escalation is False

    def test_init_with_callbacks(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test ChatResearcherAgent initialization with callbacks."""
        callbacks = [MagicMock()]
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            callbacks=callbacks,
        )

        assert agent.callbacks == callbacks

    def test_graph_property(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test that graph property returns the compiled graph."""
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        assert agent.graph is not None

    @pytest.mark.asyncio
    async def test_run_meta_intent_flow(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test run() handles meta intent correctly (orchestration returns meta + messages)."""

        async def meta_intent_classifier(state):
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [
                    AIMessage(content="Hello! I'm an AI assistant."),
                ],
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=meta_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Hello!")])
        result = await agent.run(state, thread_id="test-thread")

        assert result is not None
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_intent_routing_failure_is_bounded(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        async def failing_router(_state):
            raise RuntimeError("private provider detail")

        agent = ChatResearcherAgent(
            intent_classifier_fn=failing_router,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        with caplog.at_level(logging.WARNING):
            result = await agent.run(
                ChatResearcherState(messages=[HumanMessage(content="Research this")]),
                thread_id="test-routing-failure",
            )

        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "private provider detail" not in result["messages"][-1].content
        assert "Intent routing failed (error_type=RuntimeError)" in caplog.text
        assert "private provider detail" not in caplog.text

    @pytest.mark.asyncio
    async def test_typed_intent_failure_terminates_without_downstream_nodes(self):
        calls: list[str] = []

        async def failed_classifier(_state):
            return {
                "user_intent": IntentResult(intent="meta", target="meta", raw=None),
                "messages": [AIMessage(content="Please try again.")],
                "workflow_outcome": WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR),
            }

        async def unexpected_downstream(*_args, **_kwargs):
            calls.append("downstream")
            raise AssertionError("terminal intent failure must not invoke downstream nodes")

        agent = ChatResearcherAgent(
            intent_classifier_fn=failed_classifier,
            shallow_research_fn=unexpected_downstream,
            deep_research_fn=unexpected_downstream,
            clarifier_fn=unexpected_downstream,
            report_ask_fn=unexpected_downstream,
            report_edit_fn=unexpected_downstream,
            hybrid_research_fn=unexpected_downstream,
        )

        result = await agent.run(
            ChatResearcherState(messages=[HumanMessage(content="Research this")]),
            thread_id="test-typed-intent-failure",
        )

        assert calls == []
        assert result["messages"][-1].content == "Please try again."
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

    @pytest.mark.asyncio
    async def test_run_shallow_research_flow(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test run() handles shallow research flow (orchestration returns research + shallow)."""
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_escalation=False,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="What is CUDA?")])
        result = await agent.run(state, thread_id="test-thread")

        assert result is not None
        assert result.get("workflow_outcome") is None

    @pytest.mark.asyncio
    async def test_run_logs_query_metadata_without_customer_content(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        secret_query = "research CUDA using nvapi-vdr-fake-secret-do-not-log"  # pragma: allowlist secret
        caplog.set_level(logging.INFO, logger="aiq_agent.agents.chat_researcher.agent")
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_escalation=False,
        )

        await agent.run(
            ChatResearcherState(messages=[HumanMessage(content=secret_query)]),
            thread_id="test-safe-logging",
        )

        assert secret_query not in caplog.text
        assert "nvapi-vdr-fake-secret-do-not-log" not in caplog.text
        assert log_content_metadata(secret_query) in caplog.text

    @pytest.mark.asyncio
    async def test_shallow_research_failure_sets_terminal_outcome(
        self,
        mock_intent_classifier,
        mock_deep_research,
        mock_clarifier,
    ):
        async def shallow_boom(_state):
            raise RuntimeError("provider detail must stay private")

        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=shallow_boom,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_escalation=False,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="What is CUDA?")])
        result = await agent.run(state, thread_id="test-shallow-fail")

        assert "try again" in result["messages"][-1].content.lower()
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "provider detail" not in result["workflow_outcome"].error

    @pytest.mark.parametrize("depth", ["shallow", "deep"])
    @pytest.mark.parametrize("generated_answer", [None, "Sanitized generated answer."])
    @pytest.mark.asyncio
    async def test_empty_source_failure_returns_generated_answer_and_typed_remediation(
        self,
        mock_clarifier,
        depth,
        generated_answer,
    ):
        async def orchestration(_state):
            return {
                "user_intent": IntentResult(intent="research", target="new_research", raw=None),
                "depth_decision": DepthDecision(decision=depth, raw_reasoning="test"),
            }

        async def empty_sources(_state):
            raise EmptySourceRegistryError(
                reason=EmptySourceRegistryReason.NO_SOURCES_SELECTED,
                generated_answer=generated_answer,
            )

        agent = ChatResearcherAgent(
            intent_classifier_fn=orchestration,
            shallow_research_fn=empty_sources,
            deep_research_fn=empty_sources,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            enable_escalation=False,
        )

        result = await agent.run(
            ChatResearcherState(messages=[HumanMessage(content="Research this")]),
            thread_id=f"test-empty-sources-{depth}-{generated_answer is not None}",
        )

        content = result["messages"][-1].content
        if generated_answer:
            assert content.startswith(f"{generated_answer}\n\n")
        assert "No data sources are selected" in content
        assert "Please try again" not in content
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

    @pytest.mark.parametrize("depth", ["shallow", "deep"])
    @pytest.mark.asyncio
    async def test_unavailable_source_tools_preserve_actionable_remediation(self, mock_clarifier, depth):
        async def orchestration(_state):
            return {
                "user_intent": IntentResult(intent="research", target="new_research", raw=None),
                "depth_decision": DepthDecision(decision=depth, raw_reasoning="test"),
            }

        async def unavailable_sources(_state):
            raise EmptySourceRegistryError(reason=EmptySourceRegistryReason.SOURCE_TOOLS_UNAVAILABLE)

        agent = ChatResearcherAgent(
            intent_classifier_fn=orchestration,
            shallow_research_fn=unavailable_sources,
            deep_research_fn=unavailable_sources,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            enable_escalation=False,
        )

        result = await agent.run(
            ChatResearcherState(messages=[HumanMessage(content="Research this")]),
            thread_id=f"test-unavailable-sources-{depth}",
        )

        assert "data source tools are currently unavailable" in result["messages"][-1].content
        assert "returned no results" not in result["messages"][-1].content
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

    @pytest.mark.asyncio
    async def test_run_deep_research_flow(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test run() handles deep research flow (orchestration returns research + deep)."""

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(
                    decision="deep",
                    raw_reasoning="Complex",
                ),
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Compare CUDA vs OpenCL")],
        )
        result = await agent.run(state, thread_id="test-thread")

        assert result is not None

    @pytest.mark.asyncio
    async def test_hybrid_target_routes_to_placeholder(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        calls = []

        catalog = CatalogRoutingResponse(
            request_id="catalog-request-1",
            coverage=1,
            candidates=[
                {
                    "label": "ColumnAttribute",
                    "attribute": "recognized_revenue",
                    "term": "Revenue",
                    "id": "attr:revenue",
                }
            ],
        )

        async def hybrid_orchestration(_state):
            return {
                "user_intent": IntentResult(intent="research", target="hybrid_research"),
                "catalog_context": catalog,
                "catalog_request_id": catalog.request_id,
            }

        async def hybrid_placeholder(state):
            calls.append((state.catalog_request_id, state.catalog_context))
            return {}

        async def fail_if_called(_state):
            raise AssertionError("classic research should not run")

        agent = ChatResearcherAgent(
            intent_classifier_fn=hybrid_orchestration,
            shallow_research_fn=fail_if_called,
            deep_research_fn=fail_if_called,
            clarifier_fn=mock_clarifier,
            hybrid_research_fn=hybrid_placeholder,
        )

        await agent.run(
            ChatResearcherState(messages=[HumanMessage(content="Analyze revenue")]),
            thread_id="test-hybrid-placeholder",
        )

        assert calls == [("catalog-request-1", catalog)]

    @pytest.mark.asyncio
    async def test_hybrid_research_failure_is_bounded(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        async def hybrid_orchestration(_state):
            return {"user_intent": IntentResult(intent="research", target="hybrid_research")}

        async def failing_hybrid_research(_state):
            raise RuntimeError("private provider detail")

        agent = ChatResearcherAgent(
            intent_classifier_fn=hybrid_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            hybrid_research_fn=failing_hybrid_research,
        )

        with caplog.at_level(logging.ERROR):
            result = await agent.run(
                ChatResearcherState(messages=[HumanMessage(content="Analyze revenue")]),
                thread_id="test-hybrid-failure",
            )

        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "try again" in result["messages"][-1].content.lower()
        assert "private provider detail" not in result["messages"][-1].content
        assert "Hybrid research failed (error_type=RuntimeError)" in caplog.text
        assert "private provider detail" not in caplog.text

    @pytest.mark.asyncio
    async def test_unconfigured_hybrid_research_is_bounded(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        async def hybrid_orchestration(_state):
            return {"user_intent": IntentResult(intent="research", target="hybrid_research")}

        agent = ChatResearcherAgent(
            intent_classifier_fn=hybrid_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        with caplog.at_level(logging.ERROR):
            result = await agent.run(
                ChatResearcherState(messages=[HumanMessage(content="Analyze revenue")]),
                thread_id="test-unconfigured-hybrid-research",
            )

        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "try again" in result["messages"][-1].content.lower()
        assert "not configured" not in result["messages"][-1].content.lower()
        assert "Hybrid research failed (error_type=RuntimeError)" in caplog.text
        assert "not configured" not in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_run_report_ask_routes_to_inline_report_answer(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Report ask turns use the report ask hook instead of shallow/deep research."""
        captured_state = {}

        async def report_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research",
                    target="report",
                    report_action="ask",
                    raw=None,
                )
            }

        async def report_ask(state):
            captured_state["active_report_job_id"] = state.active_report_job_id
            captured_state["query"] = state.messages[-1].content
            return "The report's main risk is integration complexity."

        async def fail_if_called(_state):
            raise AssertionError("research path should not run for report ask")

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=fail_if_called,
            deep_research_fn=fail_if_called,
            clarifier_fn=mock_clarifier,
            report_ask_fn=report_ask,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="What is the biggest risk in this report?")],
            active_report_job_id="job-1",
        )
        result = await agent.run(state, thread_id="test-thread")

        assert result["messages"][-1].content == "The report's main risk is integration complexity."
        assert captured_state == {
            "active_report_job_id": "job-1",
            "query": "What is the biggest risk in this report?",
        }

    @pytest.mark.asyncio
    async def test_run_report_edit_routes_to_child_job_submitter(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Report edit turns submit a child job instead of running shallow/deep research inline."""
        captured_state = {}

        async def report_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research",
                    target="report",
                    report_action="edit",
                    raw=None,
                )
            }

        async def submit_report_edit(state):
            captured_state["active_report_job_id"] = state.active_report_job_id
            captured_state["query"] = state.messages[-1].content
            return "child-job-1"

        async def fail_if_called(_state):
            raise AssertionError("research path should not run for report edit")

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=fail_if_called,
            deep_research_fn=fail_if_called,
            clarifier_fn=mock_clarifier,
            report_edit_job_submitter=submit_report_edit,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Remove the appendix")],
            active_report_job_id="job-1",
        )
        result = await agent.run(state, thread_id="test-thread")

        import json

        assert json.loads(result["messages"][-1].content) == {
            "type": "job_escalation",
            "kind": "report_edit",
            "job_id": "child-job-1",
        }
        assert captured_state == {
            "active_report_job_id": "job-1",
            "query": "Remove the appendix",
        }

    @pytest.mark.asyncio
    async def test_run_report_edit_runs_inline_when_no_submitter(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """With no async submitter (CLI), report edit runs inline over the in-session report."""

        async def report_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research",
                    target="report",
                    report_action="edit",
                    raw=None,
                )
            }

        revised_report = "# FIFA World Cup 2026\n\n_Messi and Ronaldo walk into a bar..._\n\nBody."

        async def inline_edit(state):
            return revised_report

        async def fail_if_called(_state):
            raise AssertionError("research path should not run for report edit")

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=fail_if_called,
            deep_research_fn=fail_if_called,
            clarifier_fn=mock_clarifier,
            report_edit_fn=inline_edit,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="add a messi-ronaldo joke under the title")],
            last_report_markdown="# FIFA World Cup 2026\n\nBody.",
        )
        result = await agent.run(state, thread_id="test-inline-edit")

        assert result["messages"][-1].content == revised_report
        assert result["last_report_markdown"] == revised_report

    @pytest.mark.asyncio
    async def test_run_deep_research_submitter_emits_structured_escalation(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """A submitted deep-research job yields a structured escalation payload, not a prose sentence."""
        import json

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="Complex"),
            }

        async def submit_deep(state):
            return "deep-job-1"

        async def fail_if_called(_state):
            raise AssertionError("inline deep research should not run when a submitter is set")

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=fail_if_called,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            deep_research_job_submitter=submit_deep,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Compare CUDA vs OpenCL")])
        result = await agent.run(state, thread_id="test-thread-deep-escalation")

        assert json.loads(result["messages"][-1].content) == {
            "type": "job_escalation",
            "kind": "deep_research",
            "job_id": "deep-job-1",
        }

    @pytest.mark.asyncio
    async def test_run_deep_research_submitter_surfaces_safe_admission_failure(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        from aiq_api.jobs.admission import JobPrincipalCapacityExceededError

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="Complex"),
            }

        async def reject_submit(_state):
            raise JobPrincipalCapacityExceededError

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            deep_research_job_submitter=reject_submit,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Compare CUDA vs OpenCL")])
        result = await agent.run(state, thread_id="test-thread-admission-failure")

        assert result["messages"][-1].content == JobPrincipalCapacityExceededError.public_message
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

    @pytest.mark.asyncio
    async def test_deep_research_seeds_parent_report_for_inline_delta(
        self,
        mock_shallow_research,
        mock_clarifier,
    ):
        """An inline delta (use_parent_report_context + in-session report) seeds the report into the deep agent FS."""
        captured = {}

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research", target="new_research", use_parent_report_context=True, raw=None
                ),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="delta"),
            }

        async def deep(state):
            captured["files"] = dict(state.files)
            captured["query"] = state.messages[-1].content
            result = MagicMock()
            result.messages = list(state.messages) + [AIMessage(content="# Updated Report\n\nWith delta.")]
            return result

        async def seed_files(state):
            return {
                "/shared/original_report.md": state.last_report_markdown,
                "/shared/source_summary.md": "- src",
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=deep,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            report_seed_files_fn=seed_files,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="expand the economic impact section with new 2025 data")],
            last_report_markdown="# FIFA World Cup 2026\n\nBody.",
        )
        result = await agent.run(state, thread_id="test-delta-seed")

        assert captured["files"].get("/shared/original_report.md") == "# FIFA World Cup 2026\n\nBody."
        assert captured["query"] == "expand the economic impact section with new 2025 data"
        assert result["last_report_markdown"] == "# Updated Report\n\nWith delta."

    @pytest.mark.asyncio
    async def test_delta_research_bypasses_clarifier(
        self,
        mock_shallow_research,
        mock_deep_research,
    ):
        captured = {}

        async def deep_orchestration(_state):
            return {
                "user_intent": IntentResult(
                    intent="research", target="new_research", use_parent_report_context=True, raw=None
                ),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="delta"),
            }

        async def clarifier(_state):
            raise AssertionError("report delta should not invoke the clarifier")

        async def submit_deep(state):
            captured["query"] = state.original_query
            captured["uses_parent_report"] = state.user_intent.use_parent_report_context
            return "delta-job"

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=clarifier,
            enable_clarifier=True,
            deep_research_job_submitter=submit_deep,
        )
        query = "update the report with Japan statistics"
        state = ChatResearcherState(
            messages=[
                HumanMessage(content="write the original report"),
                AIMessage(content="the original report"),
                HumanMessage(content=query),
            ],
            active_report_job_id="parent-job",
        )

        result = await agent.run(state, thread_id="test-delta-bypass-clarifier")

        assert captured == {"query": query, "uses_parent_report": True}
        assert "delta-job" in result["messages"][-1].content

    @pytest.mark.asyncio
    async def test_inline_deep_research_sends_clean_query_not_report_history(
        self,
        mock_shallow_research,
        mock_clarifier,
    ):
        """Inline deep research sends only the query (like the async job), not the prior report-laden
        history, so a large prior report cannot bloat/break the writer."""
        captured = {}

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(intent="research", target="new_research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="x"),
            }

        async def deep(state):
            captured["messages"] = list(state.messages)
            result = MagicMock()
            result.messages = list(state.messages) + [AIMessage(content="# Report")]
            return result

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=deep,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
        )

        huge_prior_report = "# Old Report\n\n" + ("x" * 5000)
        state = ChatResearcherState(
            messages=[
                HumanMessage(content="old question"),
                AIMessage(content=huge_prior_report),
                HumanMessage(content="now compare 2026 and 2022 world cups"),
            ],
        )
        await agent.run(state, thread_id="test-clean-msgs")

        assert len(captured["messages"]) == 1
        assert "Old Report" not in captured["messages"][0].content
        assert "compare 2026 and 2022" in captured["messages"][0].content

    @pytest.mark.asyncio
    async def test_inline_deep_research_degrades_gracefully_on_failure(
        self,
        mock_shallow_research,
        mock_clarifier,
    ):
        """A deep-research failure in the synchronous CLI degrades to a chat message, not a hard crash."""

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(intent="research", target="new_research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="x"),
            }

        async def deep_boom(state):
            raise ValueError("writer-agent did not produce a final Markdown answer")

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=deep_boom,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="compare 2026 and 2022 world cups")])
        result = await agent.run(state, thread_id="test-deep-fail")

        content = result["messages"][-1].content
        assert content
        assert "try again" in content.lower()
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "writer-agent" not in result["workflow_outcome"].error

    @pytest.mark.asyncio
    async def test_report_ask_timeout_sets_terminal_failure(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        """A report-ask timeout returns user-safe text and a terminal failure."""

        async def report_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research",
                    target="report",
                    report_action="ask",
                    raw=None,
                )
            }

        async def report_ask_raises(_state):
            raise TimeoutError

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            report_ask_fn=report_ask_raises,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Summarize this report")],
            active_report_job_id="job-1",
        )
        with caplog.at_level(logging.WARNING):
            result = await agent.run(state, thread_id="test-thread")

        content = result["messages"][-1].content
        assert isinstance(content, str) and content.strip()
        assert content == "The report service took too long to respond. Please try again."

        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

        # The report job UUID is a bearer capability: only its redacted ref may be logged.
        assert "job-1" not in caplog.text
        assert log_identifier_ref("job-1") in caplog.text

    @pytest.mark.asyncio
    async def test_report_ask_failure_redacts_job_uuid_in_logs(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        """A non-timeout report-ask failure degrades gracefully and never logs the raw job UUID."""

        async def report_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research",
                    target="report",
                    report_action="ask",
                    raw=None,
                )
            }

        async def report_ask_raises(_state):
            raise ValueError("upstream boom")

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            report_ask_fn=report_ask_raises,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Summarize this report")],
            active_report_job_id="job-1",
        )
        with caplog.at_level(logging.WARNING):
            result = await agent.run(state, thread_id="test-thread")

        content = result["messages"][-1].content
        assert isinstance(content, str) and content.strip()
        assert "couldn't" in content.lower() or "could not" in content.lower()
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

        # The report job UUID is a bearer capability: only its redacted ref may be logged.
        assert "job-1" not in caplog.text
        assert log_identifier_ref("job-1") in caplog.text

    @pytest.mark.asyncio
    async def test_report_edit_degrades_gracefully_when_hook_raises(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
        caplog,
    ):
        """A failing report-edit hook returns a user-facing message, not an unhandled error."""

        async def report_orchestration(state):
            return {
                "user_intent": IntentResult(
                    intent="research",
                    target="report",
                    report_action="edit",
                    raw=None,
                )
            }

        async def report_edit_raises(_state):
            raise RuntimeError("boom")

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            report_edit_job_submitter=report_edit_raises,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Rewrite this report")],
            active_report_job_id="job-1",
        )
        with caplog.at_level(logging.WARNING):
            result = await agent.run(state, thread_id="test-thread")

        content = result["messages"][-1].content
        assert isinstance(content, str) and content.strip()
        assert "couldn't" in content.lower() or "could not" in content.lower()
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

        # The report job UUID is a bearer capability: only its redacted ref may be logged.
        assert "job-1" not in caplog.text
        assert log_identifier_ref("job-1") in caplog.text

    @pytest.mark.asyncio
    async def test_run_with_empty_messages(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test run() handles empty messages."""
        agent = ChatResearcherAgent(
            intent_classifier_fn=mock_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(messages=[])
        result = await agent.run(state, thread_id="test-thread")

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_without_thread_id(
        self,
        mock_intent_classifier,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test run() works without thread_id."""

        async def meta_intent_classifier(state):
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content="Hi there!")],
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=meta_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Hi")])
        result = await agent.run(state)

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_propagates_data_sources(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test that run() propagates data_sources to the graph."""
        captured_state = {}

        async def capturing_intent_classifier(state):
            captured_state["data_sources"] = state.data_sources
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content="Hello!")],
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=capturing_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Hello!")],
            data_sources=["gdrive", "confluence"],
        )
        await agent.run(state, thread_id="test-thread")

        assert captured_state["data_sources"] == ["gdrive", "confluence"]

    @pytest.mark.asyncio
    async def test_run_propagates_none_data_sources(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test that run() propagates None data_sources correctly."""
        captured_state = {}

        async def capturing_intent_classifier(state):
            captured_state["data_sources"] = state.data_sources
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content="Hello!")],
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=capturing_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Hello!")],
            data_sources=None,
        )
        await agent.run(state, thread_id="test-thread")

        assert captured_state["data_sources"] is None

    @pytest.mark.asyncio
    async def test_run_propagates_empty_data_sources(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Test that run() propagates empty data_sources list."""
        captured_state = {}

        async def capturing_intent_classifier(state):
            captured_state["data_sources"] = state.data_sources
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content="Hello!")],
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=capturing_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="Hello!")],
            data_sources=[],
        )
        await agent.run(state, thread_id="test-thread")

        assert captured_state["data_sources"] == []

    @pytest.mark.asyncio
    async def test_run_propagates_and_clears_database_scope(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Each turn passes its optional database scope without retaining an earlier value."""
        captured_scopes = []

        async def capturing_intent_classifier(state):
            captured_scopes.append(state.database_name)
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content="Hello!")],
            }

        agent = ChatResearcherAgent(
            intent_classifier_fn=capturing_intent_classifier,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
        )
        thread_id = "test-database-scope"

        await agent.run(
            ChatResearcherState(
                messages=[HumanMessage(content="Show internal revenue")],
                database_name="finance_prod",
            ),
            thread_id=thread_id,
        )
        await agent.run(
            {"messages": [HumanMessage(content="Who holds this public office?")]},
            thread_id=thread_id,
        )

        assert captured_scopes == ["finance_prod", None]

    @pytest.mark.asyncio
    async def test_in_session_report_routes_ask_without_job_id(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """A report produced inline (no job id) still routes follow-up 'ask' to the report hook."""

        async def report_orchestration(state):
            return {"user_intent": IntentResult(intent="research", target="report", report_action="ask", raw=None)}

        async def report_ask(state):
            return "From the in-session report: the main risk is integration complexity."

        async def fail_if_called(_state):
            raise AssertionError("research path should not run for an in-session report ask")

        agent = ChatResearcherAgent(
            intent_classifier_fn=report_orchestration,
            shallow_research_fn=fail_if_called,
            deep_research_fn=fail_if_called,
            clarifier_fn=mock_clarifier,
            report_ask_fn=report_ask,
        )

        state = ChatResearcherState(
            messages=[HumanMessage(content="What is the biggest risk in this report?")],
            active_report_job_id=None,
            last_report_markdown="# Report\n\nThe main risk is integration complexity.",
        )
        result = await agent.run(state, thread_id="test-thread-insession")

        assert result["messages"][-1].content.startswith("From the in-session report")

    @pytest.mark.asyncio
    async def test_deep_research_captures_last_report_markdown(
        self,
        mock_shallow_research,
        mock_clarifier,
    ):
        """A synchronous deep-research turn captures its report into last_report_markdown."""

        async def deep_orchestration(state):
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="complex"),
            }

        async def deep(state):
            result = MagicMock()
            result.messages = list(state.messages) + [AIMessage(content="# Deep Report\n\nFindings.")]
            return result

        agent = ChatResearcherAgent(
            intent_classifier_fn=deep_orchestration,
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=deep,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Compare CUDA vs OpenCL")])
        result = await agent.run(state, thread_id="test-thread-capture")

        assert result["last_report_markdown"] == "# Deep Report\n\nFindings."


class TestHybridResearchAsyncSubmission:
    """The hybrid (data-science) route must submit a durable job, not run in-request.

    A DS run takes minutes. Running it inline ties the answer to the open websocket,
    so a refresh loses the work -- the same failure mode the async job path exists to
    prevent for deep research.
    """

    @pytest.fixture
    def mock_shallow_research(self):
        async def shallow(state_input):
            messages = state_input.messages if hasattr(state_input, "messages") else state_input
            result = MagicMock()
            result.messages = list(messages) + [AIMessage(content="Here's a quick answer with sources.")]
            return result

        return shallow

    @pytest.fixture
    def mock_deep_research(self):
        async def deep(state):
            result = MagicMock()
            result.messages = list(state.messages) + [AIMessage(content="Here's a comprehensive report.")]
            return result

        return deep

    @pytest.fixture
    def mock_clarifier(self):
        async def clarifier(state_input):
            messages = state_input.messages if hasattr(state_input, "messages") else state_input
            result = MagicMock()
            result.messages = list(messages)
            result.clarifier_log = "User clarified: technical focus"
            return result

        return clarifier

    @staticmethod
    def _hybrid_intent():
        async def _route(state):
            return {
                "user_intent": IntentResult(intent="research", raw=None, target="hybrid_research"),
                "depth_decision": DepthDecision(decision="shallow", raw_reasoning="Structured data"),
            }

        return _route

    @pytest.mark.asyncio
    async def test_submitter_emits_data_science_escalation(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        import json

        submitted = {}

        async def submit_hybrid(state):
            submitted["state"] = state
            return "ds-job-1"

        async def fail_if_called(_state):
            raise AssertionError("inline hybrid research must not run when a submitter is set")

        agent = ChatResearcherAgent(
            intent_classifier_fn=self._hybrid_intent(),
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            hybrid_research_fn=fail_if_called,
            hybrid_research_job_submitter=submit_hybrid,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Which state has the most orders?")])
        result = await agent.run(state, thread_id="test-thread-hybrid-escalation")

        assert json.loads(result["messages"][-1].content) == {
            "type": "job_escalation",
            "kind": "data_science",
            "job_id": "ds-job-1",
        }
        # The submitted state must carry the turn's own question, not the thread's first
        # message, so the worker analyzes what the user actually asked.
        assert submitted["state"].messages[-1].content == "Which state has the most orders?"

    @pytest.mark.asyncio
    async def test_runs_inline_when_no_submitter_is_configured(
        self,
        mock_shallow_research,
        mock_deep_research,
        mock_clarifier,
    ):
        """Without a scheduler the CLI has no job store, so hybrid must still answer inline."""
        called = {}

        async def inline_hybrid(_state):
            called["yes"] = True
            return {"messages": [AIMessage(content="inline hybrid answer")]}

        agent = ChatResearcherAgent(
            intent_classifier_fn=self._hybrid_intent(),
            shallow_research_fn=mock_shallow_research,
            deep_research_fn=mock_deep_research,
            clarifier_fn=mock_clarifier,
            enable_clarifier=False,
            hybrid_research_fn=inline_hybrid,
        )

        state = ChatResearcherState(messages=[HumanMessage(content="Which state has the most orders?")])
        result = await agent.run(state, thread_id="test-thread-hybrid-inline")

        assert called.get("yes") is True
        assert result["messages"][-1].content == "inline hybrid answer"
