# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public Deep Researcher Agent MCP server with a supported, unauthenticated client."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from datetime import timedelta
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx
from mcp.client.streamable_http import streamable_http_client

from mcp import ClientSession

EXPECTED_SERVER_NAME = "deep_researcher_deep_research"
EXPECTED_TOOLS = {"get_final_report", "poll_query", "submit_query"}
EXPECTED_HEALTH_STATUS = "ready"
UNKNOWN_JOB_ID = "00000000-0000-4000-8000-000000000000"
FORBIDDEN_REQUEST_HEADERS = {"authorization"}


def _health_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("--url must be an absolute HTTP or HTTPS URL")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("--url must not include credentials, query parameters, or fragments")
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


async def _wait_for_health(client: httpx.AsyncClient, url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") == EXPECTED_HEALTH_STATUS:
                    return
                last_error = f"unexpected health payload: {payload!r}"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
        await asyncio.sleep(1)
    raise RuntimeError(f"MCP health check timed out after {timeout_seconds:g}s: {last_error}")


async def _smoke(endpoint: str, timeout_seconds: float) -> dict[str, object]:
    health_url = _health_url(endpoint)
    observed_headers: list[set[str]] = []

    async def record_request(request: httpx.Request) -> None:
        observed_headers.append(set(request.headers))

    timeout = httpx.Timeout(30, connect=5)
    async with httpx.AsyncClient(timeout=timeout, event_hooks={"request": [record_request]}) as client:
        await _wait_for_health(client, health_url, timeout_seconds)

        async with streamable_http_client(
            endpoint,
            http_client=client,
            terminate_on_close=False,
        ) as (read_stream, write_stream, get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=30),
            ) as session:
                initialized = await session.initialize()
                if initialized.serverInfo.name != EXPECTED_SERVER_NAME:
                    raise AssertionError(
                        f"unexpected MCP server name: {initialized.serverInfo.name!r}; "
                        f"expected {EXPECTED_SERVER_NAME!r}"
                    )
                if get_session_id() is not None:
                    raise AssertionError("stateless MCP server unexpectedly issued a session ID")

                listed = await session.list_tools()
                tool_names = {tool.name for tool in listed.tools}
                if len(listed.tools) != len(EXPECTED_TOOLS) or tool_names != EXPECTED_TOOLS:
                    raise AssertionError(
                        f"unexpected MCP tools: {[tool.name for tool in listed.tools]!r}; "
                        f"expected exactly {sorted(EXPECTED_TOOLS)!r}"
                    )

                result = await session.call_tool("poll_query", {"job_id": UNKNOWN_JOB_ID})
                if result.isError:
                    raise AssertionError(f"poll_query returned an MCP tool error: {result.content!r}")
                expected_result = {"state": "not_found", "error": "job_not_found"}
                if result.structuredContent != expected_result:
                    raise AssertionError(
                        f"unexpected poll_query result: {result.structuredContent!r}; expected {expected_result!r}"
                    )

    forbidden = sorted(header for headers in observed_headers for header in headers & FORBIDDEN_REQUEST_HEADERS)
    if forbidden:
        raise AssertionError(f"smoke client sent forbidden authentication headers: {forbidden!r}")
    if not observed_headers:
        raise AssertionError("smoke client did not make any HTTP requests")

    return {
        "endpoint": endpoint,
        "server": EXPECTED_SERVER_NAME,
        "tools": sorted(EXPECTED_TOOLS),
        "authentication_headers_sent": False,
        "unknown_job_result": {"state": "not_found", "error": "job_not_found"},
        "status": "ok",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:9001/mcp",
        help="Streamable HTTP MCP endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=120,
        help="seconds to wait for /health before failing (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.health_timeout <= 0:
        raise SystemExit("--health-timeout must be greater than zero")
    result = asyncio.run(_smoke(args.url, args.health_timeout))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
