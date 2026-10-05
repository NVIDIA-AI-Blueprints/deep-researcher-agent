# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register GSF capabilities as one NAT function group."""

import logging
import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import HttpUrl

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

from .client import FORWARDED_HEADER_NAMES
from .client import GSFClient
from .errors import GSFError
from .errors import GSFErrorCode
from .errors import GSFToolError
from .models import CatalogSearchRequest
from .models import TextToPQLRequest
from .models import TextToSQLRequest

logger = logging.getLogger(__name__)


class GSFPasswordAuthConfig(BaseModel):
    """Explicit GSF password-session configuration for development and evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["password"]
    email: str = Field(min_length=1)
    password: str = Field(default="GSF_PASSWORD", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class GSFFunctionGroupConfig(FunctionGroupBaseConfig, name="gsf"):
    """Shared configuration for AI-Q's GSF tools."""

    base_url: HttpUrl
    auth: GSFPasswordAuthConfig | None = None
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    read_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    completion_wall_timeout_seconds: float | None = Field(default=None, gt=0)
    max_completion_retries: int = Field(default=0, ge=0, le=2)
    max_response_bytes: int = Field(default=5_000_000, ge=1)
    default_max_rows: int = Field(default=1_000, ge=1)


def _tool_error(error: GSFError) -> str:
    """Serialize an internal GSF exception for safe tool output."""

    return GSFToolError.from_exception(error).model_dump_json(exclude_none=True)


def _get_auth_token() -> str | None:
    """Lazily resolve the current AI-Q token for bearer authentication."""

    try:
        from aiq_agent.auth.utils import get_auth_token
    except ImportError as exc:
        raise GSFError(
            GSFErrorCode.AUTHENTICATION_REQUIRED,
            "AI-Q authentication support is unavailable.",
        ) from exc
    return get_auth_token()


def _resolve_request_token(config: GSFFunctionGroupConfig) -> str | None:
    """Resolve a bearer token unless an explicit password session is configured."""

    if config.auth is not None:
        return None
    token = _get_auth_token()
    if not token:
        raise GSFError(
            GSFErrorCode.AUTHENTICATION_REQUIRED,
            "GSF authentication is required.",
        )
    return token


def _request_trace_headers() -> Mapping[str, str]:
    """Read and filter tracing headers from the current NAT context."""

    try:
        metadata = Context.get().metadata
        incoming = metadata.headers if metadata else None
    except Exception as exc:
        logger.debug(
            "Unable to read request trace headers from NAT context (error_type=%s)",
            type(exc).__name__,
        )
        return {}
    if not incoming:
        return {}
    return {name: value for name, value in incoming.items() if name.lower() in FORWARDED_HEADER_NAMES and value}


def _missing_password_environment(config: GSFFunctionGroupConfig) -> str | None:
    """Return the configured password environment name when it has no value."""

    if config.auth is None or os.environ.get(config.auth.password):
        return None
    return config.auth.password


def _unavailable_password_group(config: GSFFunctionGroupConfig, environment_name: str) -> FunctionGroup:
    """Build typed unavailable stubs without constructing a GSF client."""

    error = GSFError(
        GSFErrorCode.AUTHENTICATION_REQUIRED,
        "GSF password authentication is unavailable because its configured password is not set.",
    )

    async def catalog_search(request: CatalogSearchRequest) -> str:
        """Return a typed authentication error without contacting GSF."""

        del request
        return _tool_error(error)

    async def text_to_sql(request: TextToSQLRequest) -> str:
        """Return a typed authentication error without contacting GSF."""

        del request
        return _tool_error(error)

    async def text_to_pql(request: TextToPQLRequest) -> str:
        """Return a typed authentication error without contacting GSF."""

        del request
        return _tool_error(error)

    group = FunctionGroup(config=config)
    group.add_function(
        "catalog_search",
        catalog_search,
        input_schema=CatalogSearchRequest,
        description=f"GSF catalog search (unavailable - missing {environment_name}).",
    )
    group.add_function(
        "text_to_sql",
        text_to_sql,
        input_schema=TextToSQLRequest,
        description=f"GSF text-to-SQL (unavailable - missing {environment_name}).",
    )
    group.add_function(
        "text_to_pql",
        text_to_pql,
        input_schema=TextToPQLRequest,
        description=f"GSF text-to-PQL (unavailable - missing {environment_name}).",
    )
    return group


@register_function_group(config_type=GSFFunctionGroupConfig)
async def gsf_function_group(config: GSFFunctionGroupConfig, _builder: Builder):
    """Build namespaced GSF tools around one shared HTTP client."""

    missing_password_environment = _missing_password_environment(config)
    if missing_password_environment is not None:
        logger.warning(
            "GSF password environment variable %s is not set. The GSF tools will be registered as unavailable.",
            missing_password_environment,
        )
        yield _unavailable_password_group(config, missing_password_environment)
        return

    async with GSFClient.from_config(config) as client:

        async def catalog_search(request: CatalogSearchRequest) -> str:
            """Find GSF semantic candidates relevant to an enterprise-data question.

            Use the returned entity coverage and ranked candidates to ground routing and later analytical calls. This
            returns catalog context, not a final answer to the user's question.
            """

            try:
                result = await client.catalog_search(
                    request,
                    token=_resolve_request_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except GSFError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error(
                    "Unexpected GSF catalog-search failure (error_type=%s)",
                    type(exc).__name__,
                )
                return _tool_error(
                    GSFError(
                        GSFErrorCode.UPSTREAM_ERROR,
                        "GSF catalog search failed.",
                    )
                )

        async def text_to_sql(request: TextToSQLRequest) -> str:
            """Generate validated SQL and return bounded rows from authorized enterprise data.

            Use for an analytical question after the relevant structured-data scope is known. The result contains SQL
            and rows, plus semantic context, warnings, and provenance when GSF provides them. AI-Q remains responsible
            for analysis and synthesis.
            """

            try:
                result = await client.text_to_sql(
                    request,
                    token=_resolve_request_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except GSFError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error(
                    "Unexpected GSF text-to-SQL failure (error_type=%s)",
                    type(exc).__name__,
                )
                return _tool_error(
                    GSFError(
                        GSFErrorCode.UPSTREAM_ERROR,
                        "GSF text-to-SQL failed.",
                    )
                )

        async def text_to_pql(request: TextToPQLRequest) -> str:
            """Run a prediction question through GSF's Kumo/PQL path.

            Use for questions that ask what is likely to happen, which entities are most likely to experience an
            outcome, or for a forecast. The result preserves GSF's generated PQL, prediction response, bounded rows,
            and available diagnostics so callers can distinguish successful execution from a reported failure.
            """

            try:
                result = await client.text_to_pql(
                    request,
                    token=_resolve_request_token(config),
                    trace_headers=_request_trace_headers(),
                )
                return result.model_dump_json(exclude_none=True)
            except GSFError as error:
                return _tool_error(error)
            except Exception as exc:
                logger.error(
                    "Unexpected GSF text-to-PQL failure (error_type=%s)",
                    type(exc).__name__,
                )
                return _tool_error(
                    GSFError(
                        GSFErrorCode.UPSTREAM_ERROR,
                        "GSF text-to-PQL failed.",
                    )
                )

        group = FunctionGroup(config=config)
        group.add_function(
            "catalog_search",
            catalog_search,
            input_schema=CatalogSearchRequest,
            description=catalog_search.__doc__,
        )
        group.add_function(
            "text_to_sql",
            text_to_sql,
            input_schema=TextToSQLRequest,
            description=text_to_sql.__doc__,
        )
        group.add_function(
            "text_to_pql",
            text_to_pql,
            input_schema=TextToPQLRequest,
            description=text_to_pql.__doc__,
        )
        yield group
