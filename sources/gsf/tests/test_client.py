# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed GSF HTTP client."""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest
from gsf.client import GSFClient
from gsf.errors import GSFError
from gsf.errors import GSFErrorCode
from gsf.models import CatalogSearchRequest
from gsf.models import TextToPQLRequest
from gsf.models import TextToSQLRequest
from pydantic import SecretStr

_TEST_PASSWORD = "${TEST_GSF_PASSWORD}"


def _sse_response(answer: dict) -> httpx.Response:
    """Build a representative GSF SSE result response."""

    events = [
        'data: {"type":"step","node":"construct_sql_from_candidates"}',
        "",
        f"data: {json.dumps({'type': 'result', 'answer': answer})}",
        "",
        "data: [DONE]",
        "",
    ]
    return httpx.Response(
        200,
        content="\n".join(events).encode(),
        headers={"content-type": "text/event-stream", "x-request-id": "header-request"},
    )


def _sse_error_response(message: str) -> httpx.Response:
    """Build a representative terminal GSF SSE error response."""

    event = json.dumps({"type": "error", "error": message})
    return httpx.Response(
        200,
        content=f"data: {event}\n\ndata: [DONE]\n\n".encode(),
        headers={"content-type": "text/event-stream", "x-request-id": "header-request"},
    )


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("developer@example.com", None),
        (None, SecretStr(_TEST_PASSWORD)),
    ],
)
def test_password_auth_requires_both_email_and_password(email: str | None, password: SecretStr | None) -> None:
    """Reject incomplete password-session credentials."""

    with pytest.raises(ValueError):
        GSFClient(
            base_url="https://gsf.example",
            password_auth_email=email,
            password_auth_password=password,
        )


@pytest.mark.parametrize("timeout", [0, -1])
def test_completion_wall_timeout_must_be_positive(timeout: float) -> None:
    with pytest.raises(ValueError, match="completion wall timeout must be positive"):
        GSFClient(
            base_url="https://gsf.example",
            completion_wall_timeout_seconds=timeout,
        )


def test_completion_retry_count_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="completion retries cannot be negative"):
        GSFClient(
            base_url="https://gsf.example",
            max_completion_retries=-1,
        )


@pytest.mark.asyncio
async def test_catalog_search_uses_entity_coverage_path_maps_scope_and_bounds_candidates(
    catalog_search_api_response: dict,
) -> None:
    """Map catalog scope and enforce the candidate limit."""

    seen_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json=catalog_search_api_response,
            headers={"x-request-id": "header-request"},
        )

    client = GSFClient(base_url="https://gsf.example/", transport=httpx.MockTransport(handler))
    async with client:
        result = await client.catalog_search(
            CatalogSearchRequest(
                question="Find revenue metrics",
                database_name="benchmark_db",
                max_results=1,
                max_distance=0.5,
            ),
            token="user-token",
            trace_headers={"traceparent": "00-trace", "authorization": "do-not-forward"},
        )

    assert seen_request is not None
    assert seen_request.url == "https://gsf.example/api/question-entity-coverage"
    assert seen_request.headers["authorization"] == "Bearer user-token"
    assert seen_request.headers["accept"] == "application/json"
    assert seen_request.headers["traceparent"] == "00-trace"
    assert json.loads(seen_request.content) == {
        "question": "Find revenue metrics",
        "max_distance": 0.5,
        "target_db": "benchmark_db",
    }
    assert result.request_id == "header-request"
    assert result.coverage == 0.5
    assert [candidate.id for candidate in result.candidates] == ["attr:revenue"]
    assert result.uncovered_entities is None
    assert result.truncated is True


@pytest.mark.asyncio
async def test_catalog_search_maps_missing_endpoint_to_capability_unavailable() -> None:
    """Map a missing catalog endpoint to an unavailable capability."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.catalog_search(CatalogSearchRequest(question="Find revenue"), token="user-token")

    assert raised.value.code is GSFErrorCode.CAPABILITY_UNAVAILABLE


@pytest.mark.asyncio
async def test_text_to_sql_maps_database_to_target_db_and_bounds_rows(chat_sql_answer: dict) -> None:
    """Map SQL scope and enforce the configured row limit."""

    seen_request: httpx.Request | None = None
    chat_sql_answer["rows"] = None
    chat_sql_answer["columns"] = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return _sse_response(chat_sql_answer)

    client = GSFClient(
        base_url="https://gsf.example/",
        default_max_rows=1,
        transport=httpx.MockTransport(handler),
    )
    async with client:
        result = await client.text_to_sql(
            TextToSQLRequest(question="Show revenue", database_name="benchmark_db", max_rows=20),
            token="user-token",
            trace_headers={"traceparent": "00-trace", "authorization": "do-not-forward"},
        )

    assert seen_request is not None
    assert seen_request.url == "https://gsf.example/api/chat/completions"
    assert seen_request.headers["authorization"] == "Bearer user-token"
    assert seen_request.headers["accept"] == "text/event-stream"
    assert seen_request.headers["traceparent"] == "00-trace"
    assert json.loads(seen_request.content) == {
        "question": "Show revenue",
        "prediction": False,
        "target_db": "benchmark_db",
    }
    assert [column.name for column in result.columns] == ["revenue"]
    assert result.rows == [{"revenue": 100}]
    assert result.truncated is True
    assert result.thoughts == "- Constructing SQL: Used quarterly_results."
    assert "response" not in result.model_dump()


@pytest.mark.asyncio
async def test_text_to_sql_omits_optional_target_db(chat_sql_answer: dict) -> None:
    """Omit target_db when no database scope is requested."""

    seen_payload: dict | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        return _sse_response(chat_sql_answer)

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        result = await client.text_to_sql(
            TextToSQLRequest(question="Show revenue"),
            token="user-token",
        )

    assert seen_payload == {"question": "Show revenue", "prediction": False}
    assert result.sql == "SELECT revenue FROM quarterly_results"


@pytest.mark.asyncio
async def test_text_to_sql_preserves_unicode_line_separator_in_sse(chat_sql_answer: dict) -> None:
    """Preserve Unicode separators inside SSE JSON string values."""

    thoughts = "First decision.\u2028Second decision."
    answer = {**chat_sql_answer, "thoughts": thoughts}

    async def handler(_request: httpx.Request) -> httpx.Response:
        result_event = json.dumps({"type": "result", "answer": answer}, ensure_ascii=False)
        return httpx.Response(
            200,
            content=f"data: {result_event}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        result = await client.text_to_sql(TextToSQLRequest(question="Show revenue"), token="user-token")

    assert result.thoughts == thoughts


@pytest.mark.asyncio
async def test_text_to_sql_skips_malformed_intermediate_sse_event(chat_sql_answer: dict) -> None:
    """Ignore a malformed progress event before a valid terminal result."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        result_event = json.dumps({"type": "result", "answer": chat_sql_answer})
        return httpx.Response(
            200,
            content=f"data: {{malformed-step\n\ndata: {result_event}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        result = await client.text_to_sql(TextToSQLRequest(question="Show revenue"), token="user-token")

    assert result.sql == "SELECT revenue FROM quarterly_results"


@pytest.mark.asyncio
async def test_text_to_sql_retries_terminal_recursion_failure(chat_sql_answer: dict) -> None:
    """Replay a completed GSF request after its stochastic recursion-limit failure."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _sse_error_response("Agent failed: Recursion limit of 45 reached. GRAPH_RECURSION_LIMIT")
        return _sse_response(chat_sql_answer)

    client = GSFClient(
        base_url="https://gsf.example",
        max_completion_retries=1,
        transport=httpx.MockTransport(handler),
    )
    with patch("gsf.client.asyncio.sleep", new=AsyncMock()):
        async with client:
            result = await client.text_to_sql(TextToSQLRequest(question="Show revenue"), token="user-token")

    assert attempts == 2
    assert result.sql == "SELECT revenue FROM quarterly_results"


@pytest.mark.asyncio
async def test_text_to_sql_does_not_retry_unknown_terminal_failure() -> None:
    """Do not replay semantic or otherwise unclassified GSF failures."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _sse_error_response("The requested field is not available")

    client = GSFClient(
        base_url="https://gsf.example",
        max_completion_retries=1,
        transport=httpx.MockTransport(handler),
    )
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show revenue"), token="user-token")

    assert attempts == 1
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_text_to_sql_enforces_completion_wall_timeout() -> None:
    """Bound total completion time even when HTTP streaming remains active."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return _sse_error_response("unreachable")

    client = GSFClient(
        base_url="https://gsf.example",
        completion_wall_timeout_seconds=0.001,
        transport=httpx.MockTransport(handler),
    )
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show revenue"), token="user-token")

    assert raised.value.code is GSFErrorCode.TIMEOUT
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_text_to_sql_retries_completion_wall_timeout(chat_sql_answer: dict) -> None:
    """Apply the completion retry budget after an initial wall timeout."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(0.05)
        return _sse_response(chat_sql_answer)

    client = GSFClient(
        base_url="https://gsf.example",
        completion_wall_timeout_seconds=0.01,
        max_completion_retries=1,
        transport=httpx.MockTransport(handler),
    )
    with patch.object(GSFClient, "_retry_delay", return_value=0):
        async with client:
            result = await client.text_to_sql(TextToSQLRequest(question="Show revenue"), token="user-token")

    assert attempts == 2
    assert result.sql == "SELECT revenue FROM quarterly_results"


@pytest.mark.asyncio
async def test_text_to_pql_uses_prediction_routing_without_database_scope(chat_pql_answer: dict) -> None:
    """Route predictions without selecting a database in the AI-Q tool call."""

    seen_payload: dict | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        return _sse_response(chat_pql_answer)

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        result = await client.text_to_pql(
            TextToPQLRequest(question="Predict churn risk"),
            token="user-token",
        )

    assert seen_payload == {
        "question": "Predict churn risk",
        "prediction": True,
    }
    assert result.pql == "PREDICT churn FOR customers NEXT 30 DAYS"
    assert result.response == "A churn prediction query was generated."
    assert result.thoughts == "- Constructing PQL: Predicted customer churn over 30 days."
    assert [column.name for column in result.columns] == ["customer_id", "score"]
    assert result.rows == [{"customer_id": "customer-1", "score": 0.9}]
    assert result.truncated is False


@pytest.mark.asyncio
async def test_text_to_pql_supports_optional_benchmark_database_scope(chat_pql_answer: dict) -> None:
    """Forward an explicitly configured benchmark database without making it the default path."""

    seen_payload: dict | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        return _sse_response(chat_pql_answer)

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        await client.text_to_pql(
            TextToPQLRequest(question="Predict churn risk", database_name="benchmark_db"),
            token="user-token",
        )

    assert seen_payload == {
        "question": "Predict churn risk",
        "prediction": True,
        "target_db": "benchmark_db",
    }


@pytest.mark.asyncio
async def test_text_to_pql_preserves_failure_diagnostics_without_pql() -> None:
    """Return GSF's prediction failure explanation even when it produced no PQL."""

    answer = {
        "response": "Prediction could not be completed because no valid entity was found.",
        "thoughts": "- Preparing candidates: No primary-key entity matched the question.",
        "sql_code": None,
        "sql_columns": [],
        "sql_response_from_db": None,
    }

    async def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(answer)

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        result = await client.text_to_pql(TextToPQLRequest(question="Predict demand"), token="user-token")

    assert result.pql is None
    assert result.response == "Prediction could not be completed because no valid entity was found."
    assert result.rows == []


@pytest.mark.asyncio
async def test_password_auth_logs_in_uses_cookie_and_signs_out(chat_sql_answer: dict) -> None:
    """Use one Better Auth cookie from sign-in through sign-out."""

    seen_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/api/auth/sign-in/email":
            assert json.loads(request.content) == {
                "email": "developer@example.com",
                "password": _TEST_PASSWORD,
            }
            assert request.headers["origin"] == "https://gsf.example"
            assert request.headers["referer"] == "https://gsf.example/"
            return httpx.Response(
                200,
                json={"user": {"email": "developer@example.com"}},
                headers={"set-cookie": "better-auth.session_token=session-value; Path=/; HttpOnly; SameSite=Lax"},
            )
        if request.url.path == "/api/auth/sign-out":
            assert "better-auth.session_token=session-value" in request.headers["cookie"]
            assert request.headers["origin"] == "https://gsf.example"
            assert request.headers["referer"] == "https://gsf.example/"
            return httpx.Response(200, json={"success": True})

        assert request.url.path == "/api/chat/completions"
        assert "better-auth.session_token=session-value" in request.headers["cookie"]
        assert "authorization" not in request.headers
        return _sse_response(chat_sql_answer)

    client = GSFClient(
        base_url="https://gsf.example",
        password_auth_email="developer@example.com",  # pragma: allowlist secret
        password_auth_password=SecretStr(_TEST_PASSWORD),
        transport=httpx.MockTransport(handler),
    )
    async with client:
        result = await client.text_to_sql(TextToSQLRequest(question="Show data"), token=None)

    assert seen_paths == [
        "/api/auth/sign-in/email",
        "/api/chat/completions",
        "/api/auth/sign-out",
    ]
    assert result.sql == "SELECT revenue FROM quarterly_results"


@pytest.mark.asyncio
async def test_password_session_is_reused_across_tool_calls(chat_sql_answer: dict) -> None:
    """Reuse one password session across multiple GSF calls."""

    seen_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/api/auth/sign-in/email":
            return httpx.Response(
                200,
                json={"user": {"email": "developer@example.com"}},
                headers={"set-cookie": "better-auth.session_token=session-value; Path=/; HttpOnly; SameSite=Lax"},
            )
        if request.url.path == "/api/auth/sign-out":
            return httpx.Response(200, json={"success": True})

        assert "better-auth.session_token=session-value" in request.headers["cookie"]
        return _sse_response(chat_sql_answer)

    client = GSFClient(
        base_url="https://gsf.example",
        password_auth_email="developer@example.com",  # pragma: allowlist secret
        password_auth_password=SecretStr(_TEST_PASSWORD),
        transport=httpx.MockTransport(handler),
    )
    async with client:
        await client.text_to_sql(TextToSQLRequest(question="First question"), token=None)
        await client.text_to_sql(TextToSQLRequest(question="Second question"), token=None)

    assert seen_paths == [
        "/api/auth/sign-in/email",
        "/api/chat/completions",
        "/api/chat/completions",
        "/api/auth/sign-out",
    ]


@pytest.mark.asyncio
async def test_password_sign_in_retries_rate_limit_before_opening_session() -> None:
    """Retry a transient sign-in rate limit using the shared retry policy."""

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/api/auth/sign-in/email":
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json={"user": {"email": "developer@example.com"}})
        assert request.url.path == "/api/auth/sign-out"
        return httpx.Response(200, json={"success": True})

    client = GSFClient(
        base_url="https://gsf.example",
        max_retries=1,
        password_auth_email="developer@example.com",  # pragma: allowlist secret
        password_auth_password=SecretStr(_TEST_PASSWORD),
        transport=httpx.MockTransport(handler),
    )
    async with client:
        pass

    assert attempts == 2


@pytest.mark.asyncio
async def test_password_sign_in_does_not_retry_invalid_credentials() -> None:
    """Fail immediately when GSF rejects the configured credentials."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401)

    client = GSFClient(
        base_url="https://gsf.example",
        max_retries=2,
        password_auth_email="developer@example.com",  # pragma: allowlist secret
        password_auth_password=SecretStr(_TEST_PASSWORD),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GSFError) as raised:
        async with client:
            pass

    assert raised.value.code is GSFErrorCode.AUTHENTICATION_REQUIRED
    assert str(raised.value) == "GSF password sign-in was rejected."
    assert attempts == 1


@pytest.mark.asyncio
async def test_client_without_password_auth_requires_bearer_before_http() -> None:
    """Fail before HTTP when neither password nor bearer auth exists."""

    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token=None)

    assert raised.value.code is GSFErrorCode.AUTHENTICATION_REQUIRED
    assert calls == 0


@pytest.mark.asyncio
async def test_client_normalizes_forbidden_without_leaking_body() -> None:
    """Map forbidden responses without exposing upstream response text."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="secret database details")

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert raised.value.code is GSFErrorCode.FORBIDDEN
    assert "secret" not in raised.value.message


@pytest.mark.asyncio
async def test_client_retries_rate_limit_then_succeeds(chat_sql_answer: dict) -> None:
    """Retry a rate-limited request with deterministic jitter."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429)
        return _sse_response(chat_sql_answer)

    client = GSFClient(base_url="https://gsf.example", max_retries=1, transport=httpx.MockTransport(handler))
    with (
        patch("gsf.client.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch("gsf.client.random.random", return_value=0.5),
    ):
        async with client:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert attempts == 2
    sleep.assert_awaited_once_with(0.75)


@pytest.mark.asyncio
async def test_client_honors_retry_after_for_rate_limit(chat_sql_answer: dict) -> None:
    """Prefer GSF's bounded Retry-After delay over local jitter."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "12"})
        return _sse_response(chat_sql_answer)

    client = GSFClient(base_url="https://gsf.example", max_retries=1, transport=httpx.MockTransport(handler))
    with patch("gsf.client.asyncio.sleep", new_callable=AsyncMock) as sleep:
        async with client:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert attempts == 2
    sleep.assert_awaited_once_with(12.0)


@pytest.mark.asyncio
async def test_client_raises_rate_limited_after_retries_are_exhausted() -> None:
    """Raise RATE_LIMITED after exhausting all configured attempts."""

    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429)

    client = GSFClient(base_url="https://gsf.example", max_retries=2, transport=httpx.MockTransport(handler))
    with (
        patch("gsf.client.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch("gsf.client.random.random", return_value=0.5),
    ):
        async with client:
            with pytest.raises(GSFError) as raised:
                await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert attempts == 3
    assert sleep.await_count == 2
    assert raised.value.code is GSFErrorCode.RATE_LIMITED
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_client_retries_connect_timeout_then_raises_timeout() -> None:
    """Retry connect timeouts before returning the normalized timeout."""

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("GSF connection timed out", request=request)

    client = GSFClient(base_url="https://gsf.example", max_retries=2, transport=httpx.MockTransport(handler))
    with (
        patch("gsf.client.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch("gsf.client.random.random", return_value=0.5),
    ):
        async with client:
            with pytest.raises(GSFError) as raised:
                await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert attempts == 3
    assert sleep.await_count == 2
    assert raised.value.code is GSFErrorCode.TIMEOUT
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_client_does_not_retry_read_timeout() -> None:
    """Avoid replaying a request after a read timeout."""

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("GSF response timed out", request=request)

    client = GSFClient(base_url="https://gsf.example", max_retries=2, transport=httpx.MockTransport(handler))
    with patch("gsf.client.asyncio.sleep", new_callable=AsyncMock) as sleep:
        async with client:
            with pytest.raises(GSFError) as raised:
                await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert raised.value.code is GSFErrorCode.TIMEOUT
    assert attempts == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_does_not_retry_transport_error() -> None:
    """Avoid replaying a request after a generic transport failure."""

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("GSF connection failed", request=request)

    client = GSFClient(base_url="https://gsf.example", max_retries=2, transport=httpx.MockTransport(handler))
    with patch("gsf.client.asyncio.sleep", new_callable=AsyncMock) as sleep:
        async with client:
            with pytest.raises(GSFError) as raised:
                await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert attempts == 1
    assert raised.value.code is GSFErrorCode.UPSTREAM_ERROR
    assert raised.value.retryable is True
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_rejects_oversized_response(chat_sql_answer: dict) -> None:
    """Reject a response that exceeds the configured byte ceiling."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(chat_sql_answer)

    client = GSFClient(
        base_url="https://gsf.example",
        max_response_bytes=10,
        transport=httpx.MockTransport(handler),
    )
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert raised.value.code is GSFErrorCode.RESPONSE_TOO_LARGE


@pytest.mark.asyncio
async def test_client_rejects_malformed_response() -> None:
    """Reject a malformed non-SSE response."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert raised.value.code is GSFErrorCode.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_client_maps_sse_error_without_leaking_message() -> None:
    """Map SSE errors without exposing upstream error text."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"type":"error","message":"secret database details"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = GSFClient(base_url="https://gsf.example", transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(GSFError) as raised:
            await client.text_to_sql(TextToSQLRequest(question="Show data"), token="user-token")

    assert raised.value.code is GSFErrorCode.UPSTREAM_ERROR
    assert "secret" not in raised.value.message
