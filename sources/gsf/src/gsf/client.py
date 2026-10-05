# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed asynchronous client for GSF HTTP capabilities."""

import asyncio
import json
import logging
import os
import random
import re
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import SecretStr
from pydantic import ValidationError

from .errors import GSFError
from .errors import GSFErrorCode
from .models import CatalogSearchRequest
from .models import CatalogSearchResponse
from .models import ResultColumn
from .models import TextToPQLRequest
from .models import TextToPQLResponse
from .models import TextToSQLRequest
from .models import TextToSQLResponse

FORWARDED_HEADER_NAMES = frozenset({"baggage", "traceparent", "tracestate", "x-correlation-id", "x-request-id"})
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_PASSWORD_SIGN_IN_PATH = "api/auth/sign-in/email"  # pragma: allowlist secret
_PASSWORD_SIGN_OUT_PATH = "api/auth/sign-out"  # pragma: allowlist secret
_HTTP_MAX_CONNECTIONS = 100
_HTTP_MAX_KEEPALIVE_CONNECTIONS = 20
_MAX_RETRY_DELAY_SECONDS = 30.0
_SSE_LINE_SPLIT = re.compile(r"\r\n|\r|\n")
_RETRYABLE_COMPLETION_ERROR_MARKERS = (
    "graph_recursion_limit",
    "recursion limit of",
)

logger = logging.getLogger(__name__)


class GSFClient:
    """NAT-independent client sharing one bounded HTTP connection pool."""

    def __init__(
        self,
        *,
        base_url: str,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        completion_wall_timeout_seconds: float | None = None,
        max_completion_retries: int = 0,
        max_response_bytes: int = 5_000_000,
        default_max_rows: int = 1_000,
        password_auth_email: str | None = None,
        password_auth_password: SecretStr | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize bounded HTTP behavior and optional password authentication."""

        if (password_auth_email is None) != (password_auth_password is None):
            raise ValueError("GSF password authentication requires both email and password")
        if completion_wall_timeout_seconds is not None and completion_wall_timeout_seconds <= 0:
            raise ValueError("GSF completion wall timeout must be positive when configured")
        if max_completion_retries < 0:
            raise ValueError("GSF completion retries cannot be negative")
        self._base_url = base_url.rstrip("/")
        self._api_base_url = f"{self._base_url}/api"
        self._max_retries = max_retries
        self._completion_wall_timeout_seconds = completion_wall_timeout_seconds
        self._max_completion_retries = max_completion_retries
        self._max_response_bytes = max_response_bytes
        self._default_max_rows = default_max_rows
        self._password_auth_email = password_auth_email
        self._password_auth_password = password_auth_password
        self._transport = transport
        self._timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_config(cls, config: Any) -> "GSFClient":
        """Construct a client from the GSF function-group config without importing NAT."""

        password_auth = getattr(config, "auth", None)
        password: SecretStr | None = None
        if password_auth is not None:
            password_value = os.environ.get(password_auth.password)
            if not password_value:
                raise ValueError(f"GSF password authentication requires {password_auth.password}")
            password = SecretStr(password_value)
        return cls(
            base_url=str(config.base_url),
            connect_timeout_seconds=config.connect_timeout_seconds,
            read_timeout_seconds=config.read_timeout_seconds,
            max_retries=config.max_retries,
            completion_wall_timeout_seconds=config.completion_wall_timeout_seconds,
            max_completion_retries=config.max_completion_retries,
            max_response_bytes=config.max_response_bytes,
            default_max_rows=config.default_max_rows,
            password_auth_email=password_auth.email if password_auth is not None else None,
            password_auth_password=password,
        )

    async def __aenter__(self) -> "GSFClient":
        """Open the HTTP client and establish a configured password session."""

        limits = httpx.Limits(
            max_connections=_HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=_HTTP_MAX_KEEPALIVE_CONNECTIONS,
        )
        self._client = httpx.AsyncClient(timeout=self._timeout, limits=limits, transport=self._transport)
        if self._password_auth_email is not None:
            try:
                await self._sign_in_with_password(self._client)
            except BaseException:
                await self._client.aclose()
                self._client = None
                raise
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Close the password session and shared HTTP client."""

        client = self._client
        if client is not None:
            try:
                if self._password_auth_email is not None:
                    await self._sign_out_password_session(client)
            finally:
                await client.aclose()
            self._client = None

    async def text_to_sql(
        self,
        request: TextToSQLRequest,
        *,
        token: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> TextToSQLResponse:
        """Run the SQL branch of GSF chat completions and normalize its answer."""

        max_rows = min(request.max_rows, self._default_max_rows)
        payload: dict[str, Any] = {
            "question": request.question,
            "prediction": False,
        }
        if request.database_name is not None:
            payload["target_db"] = request.database_name

        answer, request_id = await self._chat_completions(
            payload,
            token=token,
            trace_headers=trace_headers,
        )
        return self._normalize_text_to_sql(answer, request_id=request_id, max_rows=max_rows)

    async def catalog_search(
        self,
        request: CatalogSearchRequest,
        *,
        token: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> CatalogSearchResponse:
        """Find GSF semantic candidates and measure entity coverage."""

        payload: dict[str, Any] = {
            "question": request.question,
            "max_distance": request.max_distance,
        }
        if request.database_name is not None:
            payload["target_db"] = request.database_name

        body, request_id, _content_type = await self._post(
            "question-entity-coverage",
            payload,
            token=token,
            trace_headers=trace_headers,
            capability="GSF entity coverage",
        )
        data = self._parse_json_data(body, request_id=request_id)
        return self._normalize_catalog_search(data, request_id=request_id, max_results=request.max_results)

    async def text_to_pql(
        self,
        request: TextToPQLRequest,
        *,
        token: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> TextToPQLResponse:
        """Run the prediction branch of GSF chat completions and normalize its answer."""

        max_rows = min(request.max_rows, self._default_max_rows)
        payload: dict[str, Any] = {
            "question": request.question,
            "prediction": True,
        }
        if request.database_name is not None:
            payload["target_db"] = request.database_name

        answer, request_id = await self._chat_completions(
            payload,
            token=token,
            trace_headers=trace_headers,
        )
        return self._normalize_text_to_pql(answer, request_id=request_id, max_rows=max_rows)

    async def _chat_completions(
        self,
        payload: dict[str, Any],
        *,
        token: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Call GSF chat completions and return its normalized final event."""

        attempts = self._max_completion_retries + 1
        for attempt in range(attempts):
            try:
                async with asyncio.timeout(self._completion_wall_timeout_seconds):
                    body, request_id, content_type = await self._post(
                        "chat/completions",
                        payload,
                        token=token,
                        trace_headers=trace_headers,
                        capability="GSF chat completions",
                        accept="text/event-stream",
                    )
                return self._parse_chat_answer(body, content_type=content_type, request_id=request_id), request_id
            except TimeoutError as exc:
                error = GSFError(
                    GSFErrorCode.TIMEOUT,
                    "GSF chat completions exceeded its wall-clock limit.",
                    retryable=True,
                )
                if attempt + 1 >= attempts:
                    raise error from exc
                logger.warning(
                    "Retrying a timed-out GSF completion (%s/%s)",
                    attempt + 1,
                    self._max_completion_retries,
                )
                await asyncio.sleep(self._retry_delay(attempt, None))
            except GSFError as error:
                if not error.retryable or attempt + 1 >= attempts:
                    raise
                logger.warning(
                    "Retrying a terminal GSF completion failure (%s/%s)",
                    attempt + 1,
                    self._max_completion_retries,
                )
                await asyncio.sleep(self._retry_delay(attempt, None))

        raise AssertionError("unreachable")

    async def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        token: str | None,
        trace_headers: Mapping[str, str] | None,
        capability: str,
        accept: str = "application/json",
    ) -> tuple[bytes, str | None, str]:
        """Send one authenticated bounded POST with safe retry behavior."""

        client = self._require_client()
        if self._password_auth_email is None and not token:
            raise GSFError(
                GSFErrorCode.AUTHENTICATION_REQUIRED,
                "GSF authentication is required.",
            )
        headers = self._build_headers(token, trace_headers, accept=accept)
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            response: httpx.Response | None = None
            try:
                request = client.build_request(
                    "POST",
                    f"{self._api_base_url}/{endpoint}",
                    json=payload,
                    headers=headers,
                )
                response = await client.send(request, stream=True)
                request_id = response.headers.get("x-request-id")
                if response.status_code >= 400:
                    error = self._http_error(response.status_code, capability, request_id)
                    if error.retryable and attempt + 1 < attempts:
                        delay = self._retry_delay(attempt, response.headers.get("retry-after"))
                        await response.aclose()
                        response = None
                        await asyncio.sleep(delay)
                        continue
                    raise error

                body = await self._read_bounded(response, request_id=request_id)
                return body, request_id, response.headers.get("content-type", "")
            except GSFError:
                raise
            except httpx.ConnectTimeout as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(self._retry_delay(attempt, None))
                    continue
                raise GSFError(
                    GSFErrorCode.TIMEOUT,
                    f"{capability} timed out.",
                    retryable=True,
                ) from exc
            except httpx.TimeoutException as exc:
                raise GSFError(
                    GSFErrorCode.TIMEOUT,
                    f"{capability} timed out.",
                    retryable=True,
                ) from exc
            except httpx.TransportError as exc:
                raise GSFError(
                    GSFErrorCode.UPSTREAM_ERROR,
                    f"{capability} could not reach GSF.",
                    retryable=True,
                ) from exc
            finally:
                if response is not None:
                    await response.aclose()

        raise AssertionError("unreachable")

    async def _sign_in_with_password(self, client: httpx.AsyncClient) -> None:
        """Establish a Better Auth password session on the provided client."""

        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                response = await client.post(
                    f"{self._base_url}/{_PASSWORD_SIGN_IN_PATH}",
                    json={
                        "email": self._password_auth_email,
                        "password": self._password_auth_password.get_secret_value(),
                    },
                    headers=self._auth_origin_headers(),
                )
            except httpx.ConnectTimeout as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(self._retry_delay(attempt, None))
                    continue
                raise GSFError(
                    GSFErrorCode.TIMEOUT,
                    "GSF password sign-in timed out.",
                    retryable=True,
                ) from exc
            except httpx.TimeoutException as exc:
                raise GSFError(
                    GSFErrorCode.TIMEOUT,
                    "GSF password sign-in timed out.",
                    retryable=True,
                ) from exc
            except httpx.TransportError as exc:
                raise GSFError(
                    GSFErrorCode.UPSTREAM_ERROR,
                    "GSF password sign-in could not reach GSF.",
                    retryable=True,
                ) from exc

            if response.status_code < 400:
                return

            request_id = response.headers.get("x-request-id")
            error = self._http_error(response.status_code, "GSF password sign-in", request_id)
            if error.retryable and attempt + 1 < attempts:
                await asyncio.sleep(self._retry_delay(attempt, response.headers.get("retry-after")))
                continue
            if error.code in {GSFErrorCode.AUTHENTICATION_REQUIRED, GSFErrorCode.FORBIDDEN}:
                raise GSFError(
                    error.code,
                    "GSF password sign-in was rejected.",
                    request_id=request_id,
                )
            raise error

        raise AssertionError("unreachable")

    async def _sign_out_password_session(self, client: httpx.AsyncClient) -> None:
        """Best-effort sign out of the active Better Auth password session."""

        try:
            response = await client.post(
                f"{self._base_url}/{_PASSWORD_SIGN_OUT_PATH}",
                json={},
                headers=self._auth_origin_headers(),
            )
            if response.status_code >= 400:
                logger.warning("GSF password session cleanup returned HTTP %s", response.status_code)
        except httpx.HTTPError:
            logger.warning("GSF password session cleanup did not complete")

    def _require_client(self) -> httpx.AsyncClient:
        """Return the active client or reject use outside its context."""

        if self._client is None:
            raise RuntimeError("GSFClient must be used as an async context manager")
        return self._client

    def _auth_origin_headers(self) -> dict[str, str]:
        """Build consistent origin headers for Better Auth mutations."""

        return {"Origin": self._base_url, "Referer": f"{self._base_url}/"}

    @staticmethod
    def _build_headers(
        token: str | None,
        trace_headers: Mapping[str, str] | None,
        *,
        accept: str,
    ) -> dict[str, str]:
        """Build headers while forwarding only approved tracing metadata."""

        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        for name, value in (trace_headers or {}).items():
            if name.lower() in FORWARDED_HEADER_NAMES and value:
                headers[name] = value
        return headers

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float:
        """Return a capped server-directed or jittered retry delay."""

        if retry_after:
            try:
                return max(0.0, min(float(retry_after), _MAX_RETRY_DELAY_SECONDS))
            except ValueError:
                pass
        return min(2**attempt, _MAX_RETRY_DELAY_SECONDS) * (0.5 + random.random() / 2)

    async def _read_bounded(self, response: httpx.Response, *, request_id: str | None) -> bytes:
        """Read a response without exceeding the configured byte ceiling."""

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_response_bytes:
                    raise self._response_too_large(request_id)
            except ValueError:
                pass

        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self._max_response_bytes:
                raise self._response_too_large(request_id)
        return bytes(body)

    @classmethod
    def _parse_chat_answer(cls, body: bytes, *, content_type: str, request_id: str | None) -> dict[str, Any]:
        """Extract the final answer object from JSON or an SSE response."""

        try:
            text = body.decode("utf-8")
            if "text/event-stream" not in content_type and not text.lstrip().startswith(("data:", ":")):
                payload = json.loads(text)
                return cls._answer_from_event(payload, request_id=request_id)

            data_lines: list[str] = []
            for line in [*_SSE_LINE_SPLIT.split(text), ""]:
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line or not data_lines:
                    continue
                event_data = "\n".join(data_lines)
                data_lines.clear()
                if event_data == "[DONE]":
                    continue
                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError:
                    # Step/progress events are observational. A malformed one
                    # must not hide a later, valid terminal result event.
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "error":
                    upstream_message = cls._completion_error_message(event)
                    raise GSFError(
                        GSFErrorCode.UPSTREAM_ERROR,
                        "GSF chat completions failed.",
                        retryable=cls._is_retryable_completion_error(upstream_message),
                        request_id=request_id,
                    )
                if event.get("type") == "result":
                    return cls._answer_from_event(event, request_id=request_id)
        except GSFError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned an invalid response.",
                request_id=request_id,
            ) from exc

        raise GSFError(
            GSFErrorCode.INVALID_RESPONSE,
            "GSF response did not contain a final result.",
            request_id=request_id,
        )

    @staticmethod
    def _completion_error_message(event: Mapping[str, Any]) -> str:
        """Extract an upstream error only for classification; never expose it to callers."""

        for field in ("error", "message", "detail"):
            value = event.get(field)
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                nested = value.get("message")
                if isinstance(nested, str):
                    return nested
        return ""

    @staticmethod
    def _is_retryable_completion_error(message: str) -> bool:
        """Recognize GSF terminal failures observed to be stochastic and safe to replay."""

        normalized = message.casefold()
        return any(marker in normalized for marker in _RETRYABLE_COMPLETION_ERROR_MARKERS)

    @staticmethod
    def _parse_json_data(body: bytes, *, request_id: str | None) -> dict[str, Any]:
        """Parse a JSON response and unwrap its optional data envelope."""

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned an invalid response.",
                request_id=request_id,
            ) from exc

        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        if not isinstance(payload, dict):
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned an invalid response.",
                request_id=request_id,
            )
        return payload

    @staticmethod
    def _answer_from_event(payload: Any, *, request_id: str | None) -> dict[str, Any]:
        """Validate and extract a final chat answer from an event payload."""

        if isinstance(payload, dict) and isinstance(payload.get("answer"), dict):
            return payload["answer"]
        if isinstance(payload, dict) and payload.get("type") is None:
            return payload
        raise GSFError(
            GSFErrorCode.INVALID_RESPONSE,
            "GSF returned an invalid final answer.",
            request_id=request_id,
        )

    @classmethod
    def _normalize_text_to_sql(
        cls,
        answer: Mapping[str, Any],
        *,
        request_id: str | None,
        max_rows: int,
    ) -> TextToSQLResponse:
        """Normalize current and legacy GSF SQL fields into AI-Q output."""

        sql = answer.get("sql") or answer.get("sql_code")
        if not isinstance(sql, str) or not sql.strip():
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF did not return validated SQL.",
                request_id=request_id,
            )

        rows = cls._normalize_rows(answer.get("rows") or answer.get("sql_response_from_db"), request_id)
        upstream_truncated = bool(answer.get("truncated", False))
        truncated = upstream_truncated or len(rows) > max_rows
        rows = rows[:max_rows]
        columns = cls._normalize_columns(answer.get("columns") or answer.get("sql_columns"))
        if not columns and rows:
            columns = [ResultColumn(name=str(name)) for name in rows[0]]

        try:
            return TextToSQLResponse(
                request_id=answer.get("request_id") or request_id,
                thoughts=answer.get("thoughts"),
                sql=sql,
                columns=columns,
                rows=rows,
                truncated=truncated,
                custom_analyses_used=answer.get("custom_analyses_used"),
                objects_used=answer.get("objects_used"),
                joins_used=answer.get("joins_used"),
                semantic_context=answer.get("semantic_context"),
                validation_attempts=answer.get("validation_attempts"),
                assumptions=answer.get("assumptions"),
                warnings=answer.get("warnings"),
                timings=answer.get("timings"),
            )
        except ValidationError as exc:
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned invalid text-to-SQL data.",
                request_id=request_id,
            ) from exc

    @staticmethod
    def _normalize_catalog_search(
        data: Mapping[str, Any],
        *,
        request_id: str | None,
        max_results: int,
    ) -> CatalogSearchResponse:
        """Validate catalog candidates and enforce the result ceiling."""

        candidates = data.get("candidates")
        if not isinstance(candidates, list):
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned invalid catalog-search data.",
                request_id=request_id,
            )

        truncated = len(candidates) > max_results
        try:
            return CatalogSearchResponse(
                request_id=data.get("request_id") or request_id,
                coverage=data.get("coverage"),
                candidates=candidates[:max_results],
                uncovered_entities=data.get("uncovered_entities"),
                truncated=truncated,
            )
        except ValidationError as exc:
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned invalid catalog-search data.",
                request_id=request_id,
            ) from exc

    @classmethod
    def _normalize_text_to_pql(
        cls,
        answer: Mapping[str, Any],
        *,
        request_id: str | None,
        max_rows: int,
    ) -> TextToPQLResponse:
        """Normalize GSF's shared chat fields into bounded prediction output."""

        # GSF's prediction branch currently returns the PQL in ``sql_code`` so
        # its frontend can render SQL and prediction results with one shape.
        pql = answer.get("pql") or answer.get("pql_code") or answer.get("sql_code")
        pql = pql if isinstance(pql, str) and pql.strip() else None
        response = answer.get("response")
        if pql is None and (not isinstance(response, str) or not response.strip()):
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned neither PQL nor prediction diagnostics.",
                request_id=request_id,
            )

        rows = cls._normalize_rows(answer.get("rows") or answer.get("sql_response_from_db"), request_id)
        upstream_truncated = bool(answer.get("truncated", False))
        truncated = upstream_truncated or len(rows) > max_rows
        rows = rows[:max_rows]
        columns = cls._normalize_columns(answer.get("columns") or answer.get("sql_columns"))
        if not columns and rows:
            columns = [ResultColumn(name=str(name)) for name in rows[0]]

        try:
            return TextToPQLResponse(
                request_id=answer.get("request_id") or request_id,
                response=response,
                thoughts=answer.get("thoughts"),
                pql=pql,
                columns=columns,
                rows=rows,
                truncated=truncated,
                custom_analyses_used=answer.get("custom_analyses_used"),
                objects_used=answer.get("objects_used"),
                semantic_context=answer.get("semantic_context"),
                assumptions=answer.get("assumptions"),
                warnings=answer.get("warnings"),
                timings=answer.get("timings"),
            )
        except ValidationError as exc:
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned invalid text-to-PQL data.",
                request_id=request_id,
            ) from exc

    @staticmethod
    def _normalize_columns(value: Any) -> list[ResultColumn]:
        """Normalize structured or name-only column metadata."""

        if not isinstance(value, list):
            return []
        columns: list[ResultColumn] = []
        for column in value:
            if isinstance(column, dict):
                name = column.get("name") or column.get("column_name") or column.get("id")
                if name is not None:
                    columns.append(
                        ResultColumn(
                            name=str(name),
                            data_type=column.get("data_type") or column.get("type"),
                        )
                    )
            elif column is not None:
                columns.append(ResultColumn(name=str(column)))
        return columns

    @staticmethod
    def _normalize_rows(value: Any, request_id: str | None) -> list[dict[str, Any]]:
        """Normalize GSF row payload variants into dictionaries."""

        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise GSFError(
                    GSFErrorCode.INVALID_RESPONSE,
                    "GSF returned invalid query rows.",
                    request_id=request_id,
                ) from exc
        if isinstance(value, dict):
            return [value]
        if not isinstance(value, list):
            raise GSFError(
                GSFErrorCode.INVALID_RESPONSE,
                "GSF returned invalid query rows.",
                request_id=request_id,
            )

        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append(item)
                continue
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError as exc:
                    raise GSFError(
                        GSFErrorCode.INVALID_RESPONSE,
                        "GSF returned invalid query rows.",
                        request_id=request_id,
                    ) from exc
                if isinstance(parsed, dict):
                    rows.append(parsed)
                elif isinstance(parsed, list) and all(isinstance(row, dict) for row in parsed):
                    rows.extend(parsed)
                else:
                    raise GSFError(
                        GSFErrorCode.INVALID_RESPONSE,
                        "GSF returned invalid query rows.",
                        request_id=request_id,
                    )
            else:
                raise GSFError(
                    GSFErrorCode.INVALID_RESPONSE,
                    "GSF returned invalid query rows.",
                    request_id=request_id,
                )
        return rows

    def _response_too_large(self, request_id: str | None) -> GSFError:
        """Build an error for an oversized GSF response."""

        return GSFError(
            GSFErrorCode.RESPONSE_TOO_LARGE,
            "GSF response exceeded the configured size limit.",
            request_id=request_id,
        )

    @staticmethod
    def _http_error(status_code: int, capability: str, request_id: str | None) -> GSFError:
        """Map an HTTP failure to the stable GSF error contract."""

        if status_code == 401:
            return GSFError(
                GSFErrorCode.AUTHENTICATION_REQUIRED,
                "GSF authentication is required.",
                request_id=request_id,
            )
        if status_code == 403:
            return GSFError(GSFErrorCode.FORBIDDEN, "GSF access is forbidden.", request_id=request_id)
        if status_code == 404:
            return GSFError(
                GSFErrorCode.CAPABILITY_UNAVAILABLE,
                f"{capability} is unavailable.",
                request_id=request_id,
            )
        if status_code in {400, 422}:
            return GSFError(
                GSFErrorCode.INVALID_REQUEST,
                "GSF rejected the request.",
                request_id=request_id,
            )
        if status_code == 429:
            return GSFError(
                GSFErrorCode.RATE_LIMITED,
                "GSF rate limit was reached.",
                retryable=True,
                request_id=request_id,
            )
        if status_code in _RETRYABLE_STATUS_CODES:
            return GSFError(
                GSFErrorCode.UPSTREAM_ERROR,
                "GSF is temporarily unavailable.",
                retryable=True,
                request_id=request_id,
            )
        return GSFError(
            GSFErrorCode.UPSTREAM_ERROR,
            "GSF request failed.",
            request_id=request_id,
        )
