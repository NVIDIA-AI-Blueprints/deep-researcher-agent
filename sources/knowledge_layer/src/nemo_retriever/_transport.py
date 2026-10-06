# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled HTTP transport for the public NeMo Retriever REST API."""

from __future__ import annotations

import asyncio
import math
import mimetypes
import re
import ssl
import time
from datetime import UTC
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MAX_RETRY_DELAY_S = 30.0
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[^\s,;]+")


class NemoRetrieverError(RuntimeError):
    """Base error raised by the Deep Researcher Agent NeMo Retriever adapter."""


class NemoRetrieverTransportError(NemoRetrieverError):
    """The NeMo Retriever deployment could not be reached reliably."""


class NemoRetrieverHTTPError(NemoRetrieverError):
    """The NeMo Retriever deployment rejected a request."""

    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class NemoRetrieverCompatibilityError(NemoRetrieverHTTPError):
    """The deployed service does not provide the expected job-scoped API."""


def _tls_verify(verify_ssl: bool, ca_bundle: str | None) -> bool | ssl.SSLContext:
    if not verify_ssl:
        if ca_bundle:
            raise ValueError("nrl_ca_bundle cannot be used when nrl_verify_ssl is false")
        return False
    if not ca_bundle:
        return True
    path = Path(ca_bundle)
    if not path.is_file():
        raise ValueError(f"nrl_ca_bundle does not exist or is not a file: {path}")
    return ssl.create_default_context(cafile=str(path))


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


class _NRLTransport:
    """Own an event-loop-independent connection pool for an NRL deployment.

    Async callers delegate to the pooled, thread-safe ``httpx.Client`` so the
    adapter remains safe when Deep Researcher Agent invokes it from more than one event loop.
    """

    def __init__(
        self,
        *,
        base_url: str,
        scope: str,
        api_token: str | None,
        connect_timeout_s: float,
        request_timeout_s: float,
        max_retries: int,
        verify_ssl: bool,
        ca_bundle: str | None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._max_retries = max_retries
        self._owns_client = client is None
        headers = {"Accept": "application/json", "X-NRL-Scope": scope}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        timeout = httpx.Timeout(request_timeout_s, connect=connect_timeout_s)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        verify = _tls_verify(verify_ssl, ca_bundle)
        self._headers = httpx.Headers(headers)
        self._client = client or httpx.Client(timeout=timeout, limits=limits, verify=verify)

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _redact(self, value: str) -> str:
        redacted = value
        if self._api_token:
            redacted = redacted.replace(self._api_token, "[REDACTED]")
        return _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * (2**attempt), 30.0)

    def _response_error(
        self,
        response: httpx.Response,
        operation: str,
        *,
        compatibility_route: bool,
    ) -> NemoRetrieverHTTPError:
        status = response.status_code
        if compatibility_route and status in {404, 410}:
            return NemoRetrieverCompatibilityError(
                "NeMo Retriever rejected a job-scoped API route with "
                f"HTTP {status}. "
                f"Confirm Deep Researcher Agent and the NRL service use compatible collection-management API versions.",
                status_code=status,
            )
        reason = {
            400: "request validation failed",
            401: "authentication failed",
            403: "scope authorization failed",
            404: "resource was not found",
            409: "request conflicted with existing state",
            410: "API route is no longer available",
            422: "request validation failed",
            429: "service rate limit was exceeded",
        }.get(status, "service request failed")
        return NemoRetrieverHTTPError(
            self._redact(f"NeMo Retriever {operation} failed with HTTP {status}: {reason}"),
            status_code=status,
        )

    def _should_retry(self, response: httpx.Response, attempt: int, *, retryable: bool) -> bool:
        return retryable and response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries

    def _delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
        return min(retry_after, _MAX_RETRY_DELAY_S) if retry_after is not None else self._backoff(attempt)

    def _request_headers(self, supplied: Any = None) -> httpx.Headers:
        headers = httpx.Headers(self._headers)
        if supplied is not None:
            headers.update(supplied)
        return headers

    def request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        retryable: bool = False,
        compatibility_route: bool = False,
        **kwargs: Any,
    ) -> Any:
        url = self._url(path)
        request_kwargs = dict(kwargs)
        request_kwargs["headers"] = self._request_headers(request_kwargs.get("headers"))
        can_retry = method.upper() in _SAFE_METHODS or retryable
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(method, url, **request_kwargs)
            except httpx.TransportError as error:
                if not can_retry or attempt >= self._max_retries:
                    raise NemoRetrieverTransportError(
                        self._redact(
                            f"NeMo Retriever {operation} transport failure after retries: {type(error).__name__}"
                        )
                    ) from error
                time.sleep(self._backoff(attempt))
                continue
            if self._should_retry(response, attempt, retryable=can_retry):
                time.sleep(self._delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise self._response_error(response, operation, compatibility_route=compatibility_route)
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError as error:
                raise NemoRetrieverError(f"NeMo Retriever {operation} returned malformed JSON") from error
        raise AssertionError("unreachable")

    async def arequest_json(self, method: str, path: str, *, operation: str, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.request_json, method, path, operation=operation, **kwargs)

    def upload_document(
        self,
        *,
        job_id: str,
        file_path: Path,
        filename: str,
        manifest_entry_id: str,
        metadata: str,
        retryable: bool = False,
    ) -> Any:
        url = self._url(f"/v1/ingest/job/{job_id}/document")
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        for attempt in range(self._max_retries + 1):
            try:
                with file_path.open("rb") as stream:
                    response = self._client.post(
                        url,
                        headers=self._request_headers(),
                        files={"file": (filename, stream, content_type)},
                        data={"metadata": metadata, "manifest_entry_id": manifest_entry_id},
                    )
            except (httpx.TransportError, OSError) as error:
                if not retryable or attempt >= self._max_retries:
                    raise NemoRetrieverTransportError(
                        self._redact(
                            f"NeMo Retriever upload for {filename!r} failed after retries: {type(error).__name__}"
                        )
                    ) from error
                time.sleep(self._backoff(attempt))
                continue
            if self._should_retry(response, attempt, retryable=retryable):
                time.sleep(self._delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise self._response_error(response, "job document upload", compatibility_route=True)
            try:
                return response.json()
            except ValueError as error:
                raise NemoRetrieverError(f"NeMo Retriever upload for {filename!r} returned malformed JSON") from error
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
