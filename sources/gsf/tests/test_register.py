# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GSF NAT function-group registration."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from gsf.client import GSFClient
from gsf.models import CatalogSearchRequest
from gsf.models import CatalogSearchResponse
from gsf.models import TextToPQLRequest
from gsf.models import TextToPQLResponse
from gsf.models import TextToSQLRequest
from gsf.models import TextToSQLResponse
from gsf.register import GSFFunctionGroupConfig
from gsf.register import GSFPasswordAuthConfig
from gsf.register import _request_trace_headers
from gsf.register import gsf_function_group

_TEST_PASSWORD = "${TEST_GSF_PASSWORD}"


class FakeClientContext:
    """Expose a mocked client through an asynchronous context manager."""

    def __init__(self, client: MagicMock) -> None:
        """Store the mocked GSF client."""

        self.client = client

    async def __aenter__(self) -> MagicMock:
        """Return the mocked GSF client."""

        return self.client

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Complete context cleanup without suppressing failures."""

        pass


def test_password_auth_is_optional_and_carries_only_environment_name() -> None:
    """Keep password values out of serializable function-group configuration."""

    default_config = GSFFunctionGroupConfig(base_url="https://gsf.example")
    password_config = GSFFunctionGroupConfig(
        base_url="https://gsf.example",
        auth={
            "mode": "password",
            "email": "developer@example.com",
            "password": "CUSTOM_GSF_PASSWORD",  # pragma: allowlist secret
        },
    )

    assert default_config.auth is None
    assert isinstance(password_config.auth, GSFPasswordAuthConfig)
    assert password_config.auth.password == "CUSTOM_GSF_PASSWORD"  # pragma: allowlist secret
    assert "password=SecretStr" not in repr(password_config)


def test_password_auth_reads_environment_after_workflow_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the exact secret after a JSON round trip like NAT's worker boundary."""

    password = "literal ${MUST_NOT_EXPAND} # fragment"  # pragma: allowlist secret
    monkeypatch.setenv("GSF_PASSWORD", password)
    monkeypatch.setenv("MUST_NOT_EXPAND", "changed")

    config = GSFFunctionGroupConfig(
        base_url="https://gsf.example",
        auth={
            "mode": "password",
            "email": "developer@example.com",
        },
    )
    serialized = config.model_dump_json()
    reconstructed = GSFFunctionGroupConfig.model_validate_json(serialized)
    client = GSFClient.from_config(reconstructed)

    assert reconstructed.auth is not None
    assert reconstructed.auth.password == "GSF_PASSWORD"  # pragma: allowlist secret
    assert client._password_auth_password is not None
    assert client._password_auth_password.get_secret_value() == password
    assert password not in serialized
    assert "**********" not in serialized


@pytest.mark.parametrize("password_value", [None, ""])
@pytest.mark.asyncio
async def test_missing_password_registers_unavailable_group_without_client_construction(
    monkeypatch: pytest.MonkeyPatch,
    password_value: str | None,
) -> None:
    """Keep every GSF tool discoverable and safely unavailable when its configured password is absent."""

    environment_name = "CUSTOM_GSF_PASSWORD"  # pragma: allowlist secret
    if password_value is None:
        monkeypatch.delenv(environment_name, raising=False)
    else:
        monkeypatch.setenv(environment_name, password_value)
    config = GSFFunctionGroupConfig(
        base_url="https://gsf.example",
        auth={
            "mode": "password",
            "email": "developer@example.com",
            "password": environment_name,
        },
    )

    with (
        patch("gsf.register.GSFClient.from_config") as from_config,
        patch("gsf.client.httpx.AsyncClient") as http_client,
        patch("gsf.client.GSFClient._sign_in_with_password", new_callable=AsyncMock) as sign_in,
    ):
        async with gsf_function_group(config, MagicMock()) as group:
            tools = await group.get_accessible_functions()
            expected_tools = {
                "gsf__catalog_search": (CatalogSearchRequest, {"question": "Find revenue metrics"}),
                "gsf__text_to_sql": (TextToSQLRequest, {"question": "Show revenue"}),
                "gsf__text_to_pql": (TextToPQLRequest, {"question": "Forecast revenue"}),
            }

            assert set(tools) == set(expected_tools)
            for tool_name, (input_schema, request) in expected_tools.items():
                tool = tools[tool_name]
                result = json.loads(await tool.ainvoke(request))

                assert tool.input_schema is input_schema
                assert tool.description is not None
                assert f"(unavailable - missing {environment_name})" in tool.description
                assert "unavailable" in tool.description.lower()
                assert result == {
                    "status": "error",
                    "code": "authentication_required",
                    "retryable": False,
                    "message": "GSF password authentication is unavailable because its configured password is not set.",
                }
                assert environment_name not in json.dumps(result)

    from_config.assert_not_called()
    http_client.assert_not_called()
    sign_in.assert_not_awaited()


def test_completion_retry_and_wall_timeout_are_bounded() -> None:
    """Validate the narrow completion reliability controls."""

    config = GSFFunctionGroupConfig(
        base_url="https://gsf.example",
        completion_wall_timeout_seconds=600,
        max_completion_retries=1,
    )

    assert config.completion_wall_timeout_seconds == 600
    assert config.max_completion_retries == 1

    with pytest.raises(ValueError):
        GSFFunctionGroupConfig(base_url="https://gsf.example", max_completion_retries=3)


def test_request_trace_headers_forwards_only_allowlisted_nonempty_values() -> None:
    """Forward only nonempty request headers on the tracing allowlist."""

    context = MagicMock()
    context.metadata.headers = {
        "Traceparent": "00-trace",
        "x-request-id": "request-1",
        "baggage": "",
        "Authorization": "Bearer secret",
        "Cookie": "session=secret",
    }

    with patch("gsf.register.Context.get", return_value=context):
        headers = _request_trace_headers()

    assert headers == {
        "Traceparent": "00-trace",
        "x-request-id": "request-1",
    }


@pytest.mark.asyncio
async def test_group_exposes_only_requested_tools(text_to_sql_response: dict) -> None:
    """Expose only tools selected by the function-group include list."""

    client = MagicMock()
    client.text_to_sql = AsyncMock(return_value=TextToSQLResponse.model_validate(text_to_sql_response))
    config = GSFFunctionGroupConfig(base_url="https://gsf.example", include=["text_to_sql"])

    with patch("gsf.register.GSFClient.from_config", return_value=FakeClientContext(client)):
        async with gsf_function_group(config, MagicMock()) as group:
            tools = await group.get_accessible_functions()

    assert set(tools) == {"gsf__text_to_sql"}


@pytest.mark.asyncio
async def test_group_exposes_text_to_pql_when_requested(text_to_pql_response: dict) -> None:
    """Expose the prediction tool only when selected by the function-group include list."""

    client = MagicMock()
    client.text_to_pql = AsyncMock(return_value=TextToPQLResponse.model_validate(text_to_pql_response))
    config = GSFFunctionGroupConfig(base_url="https://gsf.example", include=["text_to_pql"])

    with patch("gsf.register.GSFClient.from_config", return_value=FakeClientContext(client)):
        async with gsf_function_group(config, MagicMock()) as group:
            tools = await group.get_accessible_functions()

    assert set(tools) == {"gsf__text_to_pql"}


@pytest.mark.asyncio
async def test_catalog_search_resolves_token_per_invocation(catalog_search_response: dict) -> None:
    """Resolve a fresh bearer token for a catalog invocation."""

    client = MagicMock()
    client.catalog_search = AsyncMock(return_value=CatalogSearchResponse.model_validate(catalog_search_response))
    config = GSFFunctionGroupConfig(base_url="https://gsf.example", include=["catalog_search"])

    with (
        patch("gsf.register.GSFClient.from_config", return_value=FakeClientContext(client)),
        patch("gsf.register._get_auth_token", return_value="token-one"),
        patch("gsf.register._request_trace_headers", return_value={}),
    ):
        async with gsf_function_group(config, MagicMock()) as group:
            tool = (await group.get_accessible_functions())["gsf__catalog_search"]
            result = json.loads(await tool.ainvoke({"question": "Find revenue metrics"}))

    assert result["request_id"] == "gsf-catalog-request-1"
    assert client.catalog_search.await_args.kwargs["token"] == "token-one"


@pytest.mark.asyncio
async def test_text_to_sql_resolves_token_per_invocation(text_to_sql_response: dict) -> None:
    """Resolve bearer tokens independently for consecutive SQL calls."""

    client = MagicMock()
    client.text_to_sql = AsyncMock(return_value=TextToSQLResponse.model_validate(text_to_sql_response))
    config = GSFFunctionGroupConfig(base_url="https://gsf.example", include=["text_to_sql"])

    with (
        patch("gsf.register.GSFClient.from_config", return_value=FakeClientContext(client)),
        patch("gsf.register._get_auth_token", side_effect=["token-one", "token-two"]),
        patch("gsf.register._request_trace_headers", return_value={}),
    ):
        async with gsf_function_group(config, MagicMock()) as group:
            tool = (await group.get_accessible_functions())["gsf__text_to_sql"]
            first = json.loads(await tool.ainvoke({"question": "First question"}))
            second = json.loads(await tool.ainvoke({"question": "Second question"}))

    assert first["request_id"] == "gsf-request-1"
    assert second["request_id"] == "gsf-request-1"
    assert first["thoughts"] == "- Constructing SQL: Used quarterly_results."
    assert second["thoughts"] == "- Constructing SQL: Used quarterly_results."
    assert "response" not in first
    assert "response" not in second
    assert client.text_to_sql.await_args_list[0].kwargs["token"] == "token-one"
    assert client.text_to_sql.await_args_list[1].kwargs["token"] == "token-two"
    assert "token" not in client.__dict__


@pytest.mark.asyncio
async def test_explicit_password_auth_does_not_resolve_user_token(
    text_to_sql_response: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip user-token resolution when password mode is explicit."""

    monkeypatch.setenv("GSF_PASSWORD", "configured-password")  # pragma: allowlist secret
    client = MagicMock()
    client.text_to_sql = AsyncMock(return_value=TextToSQLResponse.model_validate(text_to_sql_response))
    config = GSFFunctionGroupConfig(
        base_url="https://gsf.example",
        include=["text_to_sql"],
        auth={
            "mode": "password",
            "email": "developer@example.com",
            "password": "GSF_PASSWORD",  # pragma: allowlist secret
        },
    )

    with (
        patch("gsf.register.GSFClient.from_config", return_value=FakeClientContext(client)),
        patch("gsf.register._get_auth_token") as get_auth_token,
        patch("gsf.register._request_trace_headers", return_value={}),
    ):
        async with gsf_function_group(config, MagicMock()) as group:
            tool = (await group.get_accessible_functions())["gsf__text_to_sql"]
            result = json.loads(await tool.ainvoke({"question": "Show data"}))

    assert result["request_id"] == "gsf-request-1"
    get_auth_token.assert_not_called()
    assert client.text_to_sql.await_args.kwargs["token"] is None


@pytest.mark.asyncio
async def test_missing_authentication_fails_closed() -> None:
    """Return an authentication error without invoking the GSF client."""

    client = MagicMock()
    client.text_to_sql = AsyncMock()
    config = GSFFunctionGroupConfig(base_url="https://gsf.example", include=["text_to_sql"])

    with (
        patch("gsf.register.GSFClient.from_config", return_value=FakeClientContext(client)),
        patch("gsf.register._get_auth_token", return_value=None),
    ):
        async with gsf_function_group(config, MagicMock()) as group:
            tool = (await group.get_accessible_functions())["gsf__text_to_sql"]
            result = json.loads(await tool.ainvoke({"question": "Show data"}))

    assert result["code"] == "authentication_required"
    client.text_to_sql.assert_not_awaited()
