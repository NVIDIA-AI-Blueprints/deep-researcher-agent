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

"""Tests for the IntentClassifier node (combined intent + depth + meta orchestration)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.agents.chat_researcher.models import RESEARCH_WORKFLOW_FAILURE_ERROR
from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState
from deep_researcher_agent.agents.chat_researcher.models import WorkflowFailure
from deep_researcher_agent.agents.chat_researcher.nodes.intent_classifier import IntentClassifier
from deep_researcher_agent.agents.chat_researcher.preclassification import preclassified_depth


class TestIntentClassifier:
    """Tests for the IntentClassifier class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm

    def test_init_with_defaults(self, mock_llm):
        """Test IntentClassifier initialization with defaults."""
        classifier = IntentClassifier(llm=mock_llm)
        assert classifier.llm == mock_llm
        assert classifier.tools_info == []
        assert classifier.prompt is not None
        assert classifier.callbacks == []

    def test_init_with_tools_info(self, mock_llm):
        """Test IntentClassifier initialization with tools_info."""
        tools = [{"name": "web_search", "description": "Search the web"}]
        classifier = IntentClassifier(llm=mock_llm, tools_info=tools)
        assert classifier.tools_info == tools

    def test_init_with_custom_prompt(self, mock_llm):
        """Test IntentClassifier initialization with custom prompt."""
        custom_prompt = "Custom intent prompt {{ query }}"
        classifier = IntentClassifier(llm=mock_llm, prompt=custom_prompt)
        assert classifier.prompt == custom_prompt

    def test_init_with_callbacks(self, mock_llm):
        """Test IntentClassifier initialization with callbacks."""
        callbacks = [MagicMock()]
        classifier = IntentClassifier(llm=mock_llm, callbacks=callbacks)
        assert classifier.callbacks == callbacks

    @pytest.mark.asyncio
    async def test_run_classifies_meta_intent(self, mock_llm):
        """Test run() returns dict with meta intent and messages when LLM returns meta JSON."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"meta","meta_response":"Hi there!","research_depth":null,"depth_reasoning":null}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Hello?")])

        result = await classifier.run(state)

        assert isinstance(result, dict)
        assert result["user_intent"].intent == "meta"
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Hi there!"
        assert "workflow_outcome" not in result
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_classifies_research_intent(self, mock_llm):
        """Test run() returns dict with research intent and depth_decision when LLM returns research JSON."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","meta_response":null,"research_depth":"shallow","depth_reasoning":"Simple query"}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="What is CUDA?")])

        result = await classifier.run(state)

        assert isinstance(result, dict)
        assert result["user_intent"].intent == "research"
        assert result["depth_decision"].decision == "shallow"
        assert result["depth_decision"].raw_reasoning == "Simple query"

    @pytest.mark.asyncio
    async def test_run_parses_report_ask_route_with_active_report(self, mock_llm):
        """Test report ask routing is preserved when an active report is present."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_ask","meta_response":null,"research_depth":null,'
            '"route_reasoning":"Question refers to the active report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="What are the risks in this report?")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert result["user_intent"].intent == "research"
        assert result["user_intent"].target == "report"
        assert result["user_intent"].report_action == "ask"
        assert "depth_decision" not in result

    @pytest.mark.asyncio
    async def test_run_downgrades_report_route_without_active_report(self, mock_llm):
        """Test report routing is ignored when no active report id exists."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_cosmetic_edit","meta_response":null,"research_depth":"shallow",'
            '"route_reasoning":"Asked to edit a report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Make this shorter")])

        result = await classifier.run(state)

        assert result["user_intent"].target == "new_research"
        assert result["user_intent"].report_action is None
        assert result["depth_decision"].decision == "shallow"

    @pytest.mark.asyncio
    async def test_run_parses_parent_context_deep_research_route(self, mock_llm):
        """Test latest/update requests can route to deep research with parent report context."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_delta_research","meta_response":null,"research_depth":"deep",'
            '"route_reasoning":"Requires fresh evidence against active report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="Update this with latest data")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert result["user_intent"].target == "new_research"
        assert result["user_intent"].use_parent_report_context is True
        assert result["depth_decision"].decision == "deep"
        assert "workflow_outcome" not in result

    @pytest.mark.asyncio
    async def test_prompt_exposes_only_semantic_route_not_derived_workflow_fields(self, mock_llm):
        """The LLM should judge route; Python derives target/report_action/context fields."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_delta_research","meta_response":null,"research_depth":"deep",'
            '"route_reasoning":"Requires fresh evidence against active report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="Can we write a report on this from a player performance POV?")],
            active_report_job_id="job-1",
        )

        await classifier.run(state)

        rendered_prompt = mock_llm.ainvoke.call_args.args[0][0].content
        route_schema = (
            '"route": "report_ask" | "report_cosmetic_edit" | "report_delta_research" | "standalone_research" | "meta"'
        )
        delta_example = '"rewrite this report from a player-performance POV" -> route = "report_delta_research"'
        assert route_schema in rendered_prompt
        assert delta_example in rendered_prompt
        assert '"make this shorter" -> route = "report_cosmetic_edit"' in rendered_prompt
        assert 'Explicit report-generation requests ("write/create/generate a report")' in rendered_prompt
        assert "Compatibility fields:" not in rendered_prompt
        assert '"target":' not in rendered_prompt
        assert '"report_action":' not in rendered_prompt
        assert '"use_parent_report_context":' not in rendered_prompt

    @pytest.mark.asyncio
    async def test_run_treats_meta_route_as_meta_even_when_intent_contradicts_it(self, mock_llm):
        """route=meta must not produce target=meta plus a research depth decision."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"meta","meta_response":"I can help with research reports.",'
            '"research_depth":"shallow","route_reasoning":"Asked about system capabilities."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="What can you do?")])

        result = await classifier.run(state)

        assert result["user_intent"].intent == "meta"
        assert result["user_intent"].target == "meta"
        assert result["messages"][0].content == "I can help with research reports."
        assert "depth_decision" not in result

    @pytest.mark.asyncio
    async def test_run_maps_delta_research_route_to_parent_context_deep_research(self, mock_llm):
        """Evidence-bearing POV rewrites should run delta research against the parent report."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_delta_research",'
            '"meta_response":null,"research_depth":null,"route_reasoning":"New analytical POV."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="Rewrite this report from a player performance POV")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert result["user_intent"].target == "new_research"
        assert result["user_intent"].report_action is None
        assert result["user_intent"].use_parent_report_context is True
        assert result["depth_decision"].decision == "deep"

    @pytest.mark.asyncio
    async def test_run_maps_cosmetic_edit_route_to_report_edit(self, mock_llm):
        """Cosmetic edits should use the bounded report rewriter."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_cosmetic_edit",'
            '"meta_response":null,"research_depth":null,"route_reasoning":"Style-only edit."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="Make this shorter")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert result["user_intent"].target == "report"
        assert result["user_intent"].report_action == "edit"
        assert "depth_decision" not in result

    @pytest.mark.asyncio
    async def test_run_legacy_report_route_without_action_falls_back_to_research(self, mock_llm):
        """Legacy target=report payloads without report_action are not keyword-classified in code."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","target":"report","report_action":null,'
            '"meta_response":null,"research_depth":null,"use_parent_report_context":false,'
            '"depth_reasoning":"Refers to the active report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="Remove the methodology section")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert result["user_intent"].target == "new_research"
        assert result["user_intent"].report_action is None
        assert result["depth_decision"].decision == "shallow"

    @pytest.mark.asyncio
    async def test_run_maps_report_ask_route_to_report_ask(self, mock_llm):
        """The LLM-owned report_ask route maps to bounded report QA."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_ask",'
            '"meta_response":null,"research_depth":null,"route_reasoning":"Question about active report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="Summarize this report")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert result["user_intent"].target == "report"
        assert result["user_intent"].report_action == "ask"

    @pytest.mark.asyncio
    async def test_run_defaults_to_research_on_ambiguous(self, mock_llm):
        """Test run() defaults to research when LLM returns intent that is not meta or research."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"unknown_intent","meta_response":null,"research_depth":"shallow","depth_reasoning":""}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Something")])

        result = await classifier.run(state)

        # Invalid/ambiguous intent is normalized to research so workflow continues
        assert result["user_intent"].intent == "research"
        assert result["depth_decision"].decision == "shallow"

    @pytest.mark.asyncio
    async def test_run_empty_messages_returns_dict_no_llm_call(self, mock_llm):
        """Test run() with empty messages returns dict with research + depth_decision, no LLM call."""
        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[])

        result = await classifier.run(state)

        assert isinstance(result, dict)
        assert result["user_intent"].intent == "research"
        assert result["depth_decision"].decision == "deep"
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_error",
        [
            pytest.param(RuntimeError("[400] private provider detail"), id="provider-4xx"),
            pytest.param(RuntimeError("[500] private provider detail"), id="provider-5xx"),
            pytest.param(ConnectionError("private provider detail"), id="connection"),
            pytest.param(TimeoutError("private provider detail"), id="direct-timeout"),
            pytest.param(
                RuntimeError("upstream returned 504 Gateway Timeout: private provider detail"),
                id="wrapped-timeout",
            ),
            pytest.param(RuntimeError("[404] private provider detail"), id="provider-unavailable"),
        ],
    )
    async def test_run_marks_provider_failures_failed(self, mock_llm, provider_error):
        """Provider failures end the graph with safe text and an explicit failed outcome."""
        mock_llm.ainvoke = AsyncMock(side_effect=provider_error)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Test query")])

        result = await classifier.run(state)

        assert isinstance(result, dict)
        assert result["user_intent"].intent == "meta"
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "private provider detail" not in result["messages"][0].content
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)

        from deep_researcher_agent.agents.chat_researcher.register import _render_workflow_response

        response = _render_workflow_response(result, model="workflow")
        assert response.workflow_outcome == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "private provider detail" not in response.model_dump_json()

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self, mock_llm):
        """Test run() passes callbacks via config to LLM ainvoke(rendered_prompt, config=...)."""
        mock_response = MagicMock()
        mock_response.content = '{"intent":"meta","meta_response":"Hi","research_depth":null,"depth_reasoning":null}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_callback = MagicMock()
        classifier = IntentClassifier(llm=mock_llm, callbacks=[mock_callback])
        state = ChatResearcherState(messages=[HumanMessage(content="Hi there")])

        await classifier.run(state)

        call_args = mock_llm.ainvoke.call_args
        # ainvoke(rendered_prompt, config=config)
        assert call_args[0][0]  # first positional arg is the prompt string
        config = call_args[1].get("config", {})
        callbacks = config.get("callbacks")
        assert callbacks == [mock_callback]

    @pytest.mark.asyncio
    async def test_run_does_not_pass_prior_report_content_to_classifier_llm(self, mock_llm):
        """The router should classify the latest query, not continue the prior report conversation."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","route":"report_delta_research","meta_response":null,"research_depth":"deep",'
            '"route_reasoning":"Needs more evidence for the active report."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[
                HumanMessage(content="Write a report on reinforcement learning for AI agents"),
                AIMessage(content="Previous report body with benchmark sections and citations"),
                HumanMessage(content="can you rewrite the report to add more information on benchmarks"),
            ],
            active_report_job_id="job-1",
        )

        await classifier.run(state)

        classifier_messages = mock_llm.ainvoke.call_args.args[0]
        assert classifier_messages == [classifier_messages[0]]
        assert isinstance(classifier_messages[0], SystemMessage)
        assert "can you rewrite the report to add more information on benchmarks" in classifier_messages[0].content
        assert "Previous report body with benchmark sections and citations" not in classifier_messages[0].content

    @pytest.mark.asyncio
    async def test_run_meta_in_longer_response(self, mock_llm):
        """Test run() parses meta from JSON in response."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"meta","meta_response":"The intent is meta because it\'s a greeting.",'
            '"research_depth":null,"depth_reasoning":null}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Hello!")])

        result = await classifier.run(state)

        assert result["user_intent"].intent == "meta"

    @pytest.mark.asyncio
    async def test_run_research_in_longer_response(self, mock_llm):
        """Test run() parses research from JSON in response."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"intent":"research","meta_response":null,'
            '"research_depth":"deep","depth_reasoning":"This requires research."}'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="What is CUDA?")])

        result = await classifier.run(state)

        assert result["user_intent"].intent == "research"
        assert result["depth_decision"].decision == "deep"

    @pytest.mark.asyncio
    async def test_run_invalid_json_after_repair_marks_workflow_failed(self, mock_llm):
        """Two malformed classifier responses stop instead of launching research."""
        mock_response = MagicMock()
        mock_response.content = "not valid json at all with private provider detail"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Test")])

        result = await classifier.run(state)

        assert mock_llm.ainvoke.call_count == 2
        assert result["user_intent"].intent == "meta"
        assert "depth_decision" not in result
        assert result["workflow_outcome"] == WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR)
        assert "private provider detail" not in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_run_repairs_invalid_json_classifier_output(self, mock_llm):
        """Invalid classifier prose gets one JSON-only repair attempt before fallback."""
        bad_response = MagicMock()
        bad_response.content = "I should rewrite the report instead of returning JSON."
        repaired_response = MagicMock()
        repaired_response.content = (
            '{"intent":"research","route":"report_delta_research","meta_response":null,"research_depth":"deep",'
            '"route_reasoning":"Rewrite asks for more benchmark evidence."}'
        )
        mock_llm.ainvoke = AsyncMock(side_effect=[bad_response, repaired_response])

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(
            messages=[HumanMessage(content="can you rewrite the report to add more information on benchmarks")],
            active_report_job_id="job-1",
        )

        result = await classifier.run(state)

        assert mock_llm.ainvoke.call_count == 2
        repair_messages = mock_llm.ainvoke.call_args_list[1].args[0]
        assert len(repair_messages) == 1
        assert isinstance(repair_messages[0], SystemMessage)
        assert "Return only one valid JSON object" in repair_messages[0].content
        assert "I should rewrite the report instead of returning JSON." in repair_messages[0].content
        assert result["user_intent"].target == "new_research"
        assert result["user_intent"].use_parent_report_context is True
        assert result["depth_decision"].decision == "deep"
        assert "workflow_outcome" not in result

    def test_load_default_prompt_fallback(self, mock_llm):
        """Test _load_default_prompt returns fallback when not found."""
        with patch(
            "deep_researcher_agent.agents.chat_researcher.nodes.intent_classifier.load_prompt",
            side_effect=FileNotFoundError(),
        ):
            classifier = IntentClassifier(llm=mock_llm)
            prompt_lower = classifier.prompt.lower()
            assert "meta" in prompt_lower or "research" in prompt_lower

    @pytest.mark.asyncio
    @pytest.mark.parametrize("preset", ["shallow", "deep"])
    async def test_run_reuses_preclassified_depth_without_llm(self, preset):
        """A caller-supplied depth short-circuits: research intent, preset depth, no LLM call."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=AssertionError("intent LLM must not be invoked when preclassified"))

        classifier = IntentClassifier(llm=llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Who founded NVIDIA?")])

        with preclassified_depth(preset):
            result = await classifier.run(state)

        llm.ainvoke.assert_not_called()
        assert result["user_intent"].intent == "research"
        assert result["user_intent"].target == "new_research"
        assert result["depth_decision"].decision == preset
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_run_ignores_invalid_preclassified_depth(self, mock_llm):
        """An out-of-range preclassified depth coerces to None, so the LLM classifies normally."""
        mock_response = MagicMock()
        mock_response.content = '{"intent":"research","research_depth":"shallow"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(llm=mock_llm)
        state = ChatResearcherState(messages=[HumanMessage(content="Who founded NVIDIA?")])

        with preclassified_depth("sideways"):
            result = await classifier.run(state)

        mock_llm.ainvoke.assert_called_once()
        assert result["user_intent"].intent == "research"
        assert result["depth_decision"].decision == "shallow"
