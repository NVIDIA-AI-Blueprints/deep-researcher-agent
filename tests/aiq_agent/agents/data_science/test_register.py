# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Data Science Agent NAT registration."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.chat_researcher.models import ChatResearcherState
from aiq_agent.agents.data_science import register as data_science_register
from aiq_agent.agents.data_science.models import DataScienceAgentState
from aiq_agent.agents.data_science.utils.structured_data_guardrails import StructuredDataCallGuardMiddleware
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.data_source_registry import populate_from_config
from aiq_agent.common.data_source_registry import reset_registry


@tool
def _dummy_search(query: str) -> str:
    """Return a configured test result."""
    return query


@tool("protected__read")
def _protected_search(query: str) -> str:
    """Return one protected test result."""
    return query


def test_config_inherits_registry_tools_and_rejects_unknown_fields():
    config = data_science_register.DataScienceAgentConfig(llm="model")

    assert config.tools == []
    assert config.exclude_tools == []
    assert config.recursion_limit == 64
    assert config.interaction_mode == "interactive"
    assert config.response_mode == "standard"
    assert config.ontology_provider is None
    assert config.structured_catalog_call_limit is None
    assert config.structured_text_to_sql_call_limit is None
    assert config.structured_cache_repeated_calls is True
    assert config.python_call_limit is None
    assert config.finalization_model_call_limit is None
    with pytest.raises(ValueError, match="models"):
        data_science_register.DataScienceAgentConfig(llm="model", models={"planner": "model"})
    with pytest.raises(ValueError, match="interaction_mode"):
        data_science_register.DataScienceAgentConfig(llm="model", interaction_mode="batch")
    with pytest.raises(ValueError, match="response_mode"):
        data_science_register.DataScienceAgentConfig(llm="model", response_mode="brief")


def test_config_validates_provider_neutral_tool_roles() -> None:
    config = data_science_register.DataScienceAgentConfig(
        llm="model",
        ontology_provider={
            "provider": "gsf",
            "catalog_tools": ["gsf__catalog_search"],
            "analytical_tools": ["gsf__text_to_sql"],
            "predictive_tools": ["gsf__text_to_pql"],
        },
    )

    assert config.ontology_provider is not None
    assert config.ontology_provider.catalog_tool_names == frozenset({"gsf__catalog_search"})
    assert config.ontology_provider.analytical_tool_names == frozenset({"gsf__text_to_sql"})
    assert config.ontology_provider.predictive_tool_names == frozenset({"gsf__text_to_pql"})

    with pytest.raises(ValueError, match="tool roles overlap"):
        data_science_register.DataScienceAgentConfig(
            llm="model",
            ontology_provider={
                "provider": "gsf",
                "catalog_tools": ["gsf__catalog_search"],
                "analytical_tools": ["gsf__catalog_search"],
            },
        )


def test_ontology_provider_tools_must_be_mapped_to_one_registry_source() -> None:
    provider = data_science_register.DataScienceAgentConfig(
        llm="model",
        ontology_provider={
            "provider": "gsf",
            "catalog_tools": ["gsf__catalog_search"],
            "analytical_tools": ["gsf__text_to_sql"],
            "predictive_tools": ["gsf__text_to_pql"],
        },
    ).ontology_provider
    assert provider is not None

    try:
        reset_registry()
        populate_from_config(
            [
                {"id": "catalog", "name": "Catalog", "tools": ["gsf__catalog_search"]},
                {"id": "execution", "name": "Execution", "tools": ["gsf__text_to_sql", "gsf__text_to_pql"]},
            ]
        )
        with pytest.raises(ValueError, match="must map to the same data source"):
            data_science_register._validate_ontology_provider_source_mapping(provider)

        reset_registry()
        populate_from_config(
            [{"id": "structured_data", "name": "GSF", "tools": ["gsf__catalog_search", "gsf__text_to_sql"]}]
        )
        with pytest.raises(ValueError, match=r"unmapped tools: gsf__text_to_pql"):
            data_science_register._validate_ontology_provider_source_mapping(provider)

        reset_registry()
        populate_from_config(
            [{"id": "structured_data", "name": "GSF", "tools": ["gsf"]}],
            group_names={"gsf"},
        )
        data_science_register._validate_ontology_provider_source_mapping(provider)
    finally:
        reset_registry()


@pytest.mark.asyncio
async def test_registration_inherits_registry_refs_and_runs_selected_tools():
    reset_registry()
    populate_from_config(
        [{"id": "structured_data", "name": "GSF", "tools": ["gsf"]}],
        group_names={"gsf"},
    )
    builder = MagicMock()
    catalog_tool = _dummy_search.model_copy(update={"name": "gsf__catalog_search"})
    execution_tool = _dummy_search.model_copy(update={"name": "gsf__text_to_sql"})
    builder.get_tools = AsyncMock(return_value=[catalog_tool, execution_tool])
    builder.get_llm = AsyncMock(return_value=MagicMock())
    config = data_science_register.DataScienceAgentConfig(llm="model")

    registration = data_science_register.data_science_agent.__wrapped__(config, builder)
    with patch.object(data_science_register, "get_all_tool_refs", return_value=["gsf"]):
        function_info = await anext(registration)
    try:
        sentinel = DataScienceAgentState(
            messages=[HumanMessage(content="answer"), AIMessage(content="grounded")],
        )
        with patch.object(data_science_register.DataScienceAgent, "run", AsyncMock(return_value=sentinel)):
            result = await function_info.single_fn(DataScienceAgentState(messages=[HumanMessage(content="query")]))
    finally:
        await registration.aclose()
        reset_registry()

    builder.get_tools.assert_awaited_once_with(
        tool_names=["gsf"],
        wrapper_type=data_science_register.LLMFrameworkEnum.LANGCHAIN,
    )
    assert result is sentinel


@pytest.mark.asyncio
async def test_registration_passes_headless_mode_to_agent():
    reset_registry()
    populate_from_config(
        [{"id": "structured_data", "name": "GSF", "tools": ["gsf"]}],
        group_names={"gsf"},
    )
    builder = MagicMock()
    catalog_tool = _dummy_search.model_copy(update={"name": "gsf__catalog_search"})
    analytical_tool = _dummy_search.model_copy(update={"name": "gsf__text_to_sql"})
    predictive_tool = _dummy_search.model_copy(update={"name": "gsf__text_to_pql"})
    builder.get_tools = AsyncMock(return_value=[catalog_tool, analytical_tool, predictive_tool])
    builder.get_llm = AsyncMock(return_value=MagicMock())
    config = data_science_register.DataScienceAgentConfig(
        llm="model",
        interaction_mode="headless",
        response_mode="fdabench_choice",
        ontology_provider={
            "provider": "gsf",
            "catalog_tools": ["gsf__catalog_search"],
            "analytical_tools": ["gsf__text_to_sql"],
            "predictive_tools": ["gsf__text_to_pql"],
        },
        structured_catalog_call_limit=2,
        structured_text_to_sql_call_limit=6,
        python_call_limit=7,
        finalization_model_call_limit=28,
        verbose=True,
    )

    registration = data_science_register.data_science_agent.__wrapped__(config, builder)
    with patch.object(data_science_register, "get_all_tool_refs", return_value=["gsf"]):
        function_info = await anext(registration)
    try:
        sentinel = DataScienceAgentState(messages=[HumanMessage(content="grounded")])
        with patch.object(data_science_register, "DataScienceAgent") as agent_cls:
            agent_cls.return_value.run = AsyncMock(return_value=sentinel)
            result = await function_info.single_fn(DataScienceAgentState(messages=[HumanMessage(content="query")]))

        assert result is sentinel
        assert agent_cls.call_args.kwargs["interaction_mode"] == "headless"
        assert agent_cls.call_args.kwargs["response_mode"] == "fdabench_choice"
        guard = agent_cls.call_args.kwargs["structured_guard"]
        assert isinstance(guard, StructuredDataCallGuardMiddleware)
        assert guard.provider == "gsf"
        assert guard.catalog_tools == frozenset({"gsf__catalog_search"})
        assert guard.text_to_sql_tools == frozenset({"gsf__text_to_sql"})
        assert guard.budget.catalog_calls == 2
        assert guard.budget.text_to_sql_calls == 6
        assert agent_cls.call_args.kwargs["python_call_limit"] == 7
        assert agent_cls.call_args.kwargs["finalization_model_call_limit"] == 28
        callbacks = agent_cls.call_args.kwargs["callbacks"]
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], data_science_register.VerboseTraceCallback)
    finally:
        await registration.aclose()
        reset_registry()


@pytest.mark.asyncio
async def test_registration_resolves_selected_protected_tools_per_request():
    pytest.importorskip("aiq_api")
    reset_registry()
    populate_from_config(
        [
            {"id": "web", "name": "Web", "tools": ["_dummy_search"]},
            {
                "id": "protected",
                "name": "Protected",
                "requires_auth": True,
                "per_user_auth": {
                    "required": True,
                    "type": "mcp_oauth2",
                    "provider": "test",
                    "mcp_server_id": "test",
                },
                "tools": ["protected"],
            },
        ],
        group_names={"protected"},
    )
    builder = MagicMock()
    builder.get_tools = AsyncMock(return_value=[_dummy_search])
    builder.get_llm = AsyncMock(return_value=MagicMock())
    config = data_science_register.DataScienceAgentConfig(llm="model", tools=["_dummy_search"])
    registration = data_science_register.data_science_agent.__wrapped__(config, builder)
    function_info = await anext(registration)
    sentinel = DataScienceAgentState(messages=[HumanMessage(content="protected result")])
    try:
        with (
            patch("aiq_api.jobs.access.require_verified_principal", return_value=MagicMock()) as principal,
            patch("aiq_api.mcp_auth.provider.principal_user_id", return_value="user-1") as user_id,
            patch(
                "aiq_api.mcp_auth.runtime_tools.open_per_user_mcp_tools",
                AsyncMock(return_value=[_protected_search]),
            ) as open_tools,
            patch.object(data_science_register, "DataScienceAgent") as agent_cls,
        ):
            agent_cls.return_value.run = AsyncMock(return_value=sentinel)
            result = await function_info.single_fn(
                DataScienceAgentState(
                    messages=[HumanMessage(content="Read my protected source")],
                    data_sources=["protected"],
                )
            )

        assert result is sentinel
        principal.assert_called_once_with()
        user_id.assert_called_once()
        open_tools.assert_awaited_once()
        assert agent_cls.call_args.kwargs["tools"] == [_protected_search]
    finally:
        await registration.aclose()
        reset_registry()


@pytest.mark.asyncio
async def test_registration_reraises_rejected_principal():
    pytest.importorskip("aiq_api")
    reset_registry()
    populate_from_config(
        [
            {
                "id": "protected",
                "name": "Protected",
                "requires_auth": True,
                "per_user_auth": {
                    "required": True,
                    "type": "mcp_oauth2",
                    "provider": "test",
                    "mcp_server_id": "test",
                },
                "tools": ["protected"],
            },
        ],
        group_names={"protected"},
    )
    builder = MagicMock()
    builder.get_tools = AsyncMock(return_value=[_dummy_search])
    builder.get_llm = AsyncMock(return_value=MagicMock())
    config = data_science_register.DataScienceAgentConfig(llm="model")
    registration = data_science_register.data_science_agent.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        with (
            patch(
                "aiq_api.jobs.access.require_verified_principal",
                side_effect=HTTPException(status_code=403, detail="forbidden"),
            ),
            patch.object(data_science_register, "DataScienceAgent") as agent_cls,
            pytest.raises(HTTPException) as raised,
        ):
            await function_info.single_fn(
                DataScienceAgentState(
                    messages=[HumanMessage(content="Read my protected source")],
                    data_sources=["protected"],
                )
            )

        assert raised.value.status_code == 403
        agent_cls.assert_not_called()
    finally:
        await registration.aclose()
        reset_registry()


@pytest.mark.asyncio
async def test_registration_surfaces_reconnect_for_unavailable_protected_source():
    pytest.importorskip("aiq_api")
    from aiq_api.mcp_auth.runtime_tools import PerUserMcpSourceUnavailableError

    reset_registry()
    populate_from_config(
        [
            {"id": "web", "name": "Web", "tools": ["_dummy_search"]},
            {
                "id": "protected",
                "name": "Protected",
                "requires_auth": True,
                "per_user_auth": {
                    "required": True,
                    "type": "mcp_oauth2",
                    "provider": "test",
                    "mcp_server_id": "test",
                },
                "tools": ["protected"],
            },
        ],
        group_names={"protected"},
    )
    builder = MagicMock()
    builder.get_tools = AsyncMock(return_value=[_dummy_search])
    builder.get_llm = AsyncMock(return_value=MagicMock())
    config = data_science_register.DataScienceAgentConfig(llm="model", tools=["_dummy_search"])
    registration = data_science_register.data_science_agent.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        with (
            patch("aiq_api.jobs.access.require_verified_principal", return_value=MagicMock()),
            patch("aiq_api.mcp_auth.provider.principal_user_id", return_value="user-1"),
            patch(
                "aiq_api.mcp_auth.runtime_tools.open_per_user_mcp_tools",
                AsyncMock(side_effect=PerUserMcpSourceUnavailableError(["protected"])),
            ),
            patch.object(data_science_register, "DataScienceAgent") as agent_cls,
        ):
            result = await function_info.single_fn(
                DataScienceAgentState(
                    messages=[HumanMessage(content="Read my protected source")],
                    data_sources=["protected"],
                )
            )

        assert "protected" in str(result.messages[-1].content)
        assert "Reconnect them in the data sources panel" in str(result.messages[-1].content)
        agent_cls.assert_not_called()
    finally:
        await registration.aclose()
        reset_registry()


@pytest.mark.asyncio
async def test_direct_workflow_returns_typed_no_source_response():
    error = EmptySourceRegistryError(generated_answer="The backend returned no rows.")
    agent_fn = MagicMock()
    agent_fn.ainvoke = AsyncMock(side_effect=error)
    builder = MagicMock()
    builder.get_function = AsyncMock(return_value=agent_fn)
    config = data_science_register.DataScienceWorkflowConfig()

    registration = data_science_register.data_science_workflow.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        response = await function_info.single_fn("Rank users")
    finally:
        await registration.aclose()

    builder.get_function.assert_awaited_once_with("data_science_agent")
    assert response.choices[0].message.content == error.public_response


@pytest.mark.asyncio
async def test_direct_workflow_maps_request_context():
    agent_fn = MagicMock()
    agent_fn.ainvoke = AsyncMock(return_value=DataScienceAgentState(messages=[AIMessage(content="done")]))
    builder = MagicMock()
    builder.get_function = AsyncMock(return_value=agent_fn)
    config = data_science_register.DataScienceWorkflowConfig()

    registration = data_science_register.data_science_workflow.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        response = await function_info.single_fn(
            {
                "text": "Rank users",
                "data_sources": ["structured_data"],
                "database_name": "benchmark_db",
            }
        )
    finally:
        await registration.aclose()

    invoked_state = agent_fn.ainvoke.await_args.args[0]
    assert invoked_state.messages == [HumanMessage(content="Rank users")]
    assert invoked_state.data_sources == ["structured_data"]
    assert invoked_state.database_name == "benchmark_db"
    assert response.choices[0].message.content == "done"


@pytest.mark.asyncio
async def test_hybrid_adapter_maps_router_context_and_returns_only_final_response():
    catalog = CatalogRoutingResponse(
        request_id="catalog-1",
        coverage=0.75,
        candidates=[
            {
                "label": "ColumnAttribute",
                "attribute": "recognized_revenue",
                "term": "Revenue",
                "id": "attr:revenue",
            }
        ],
        uncovered_entities=["public market comparison"],
    )
    input_message = HumanMessage(content="Compare enterprise revenue with the public market")
    tool_message = ToolMessage(content="rows", tool_call_id="gsf-call-1", name="gsf__text_to_sql")
    final_message = AIMessage(content="Enterprise revenue increased relative to the public market.")
    agent_fn = MagicMock()
    agent_fn.ainvoke = AsyncMock(
        return_value=DataScienceAgentState(messages=[input_message, tool_message, final_message])
    )
    builder = MagicMock()
    builder.get_function = AsyncMock(return_value=agent_fn)
    config = data_science_register.DataScienceHybridAdapterConfig(agent="data_science_agent")

    registration = data_science_register.data_science_hybrid_adapter.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        result = await function_info.single_fn(
            ChatResearcherState(
                messages=[input_message],
                data_sources=["structured_data", "web_search"],
                user_info={"tenant": "acme"},
                database_name="benchmark_db",
                catalog_context=catalog,
                catalog_request_id="catalog-1",
            )
        )
    finally:
        await registration.aclose()

    builder.get_function.assert_awaited_once_with(config.agent)
    invoked_state = agent_fn.ainvoke.await_args.args[0]
    assert invoked_state.messages == [input_message]
    assert invoked_state.data_sources == ["structured_data", "web_search"]
    assert invoked_state.user_info == {"tenant": "acme"}
    assert invoked_state.database_name == "benchmark_db"
    assert invoked_state.catalog_request_id == "catalog-1"
    assert invoked_state.catalog_context == catalog
    assert result == {"messages": [final_message]}


@pytest.mark.asyncio
async def test_hybrid_adapter_rejects_missing_new_final_response():
    input_message = HumanMessage(content="Analyze revenue")
    agent_fn = MagicMock()
    agent_fn.ainvoke = AsyncMock(return_value=DataScienceAgentState(messages=[input_message]))
    builder = MagicMock()
    builder.get_function = AsyncMock(return_value=agent_fn)
    config = data_science_register.DataScienceHybridAdapterConfig(agent="data_science_agent")

    registration = data_science_register.data_science_hybrid_adapter.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        with pytest.raises(RuntimeError, match="no final response"):
            await function_info.single_fn(ChatResearcherState(messages=[input_message]))
    finally:
        await registration.aclose()
