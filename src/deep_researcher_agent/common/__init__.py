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

"""Common utilities for the Deep Researcher Agent blueprint."""

from __future__ import annotations

import datetime
import logging

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChoice
from nat.data_models.api_server import ChoiceMessage
from nat.data_models.api_server import Usage
from nat.data_models.api_server import UserMessageContentRoleType

from .callbacks import VerboseTraceCallback
from .citation_verification import SourceRegistry
from .citation_verification import get_or_create_session_registry
from .citation_verification import get_session_registry
from .citation_verification import register_source_parser
from .citation_verification import reset_session_registry
from .citation_verification import sanitize_report
from .citation_verification import set_session_registry
from .citation_verification import verify_citations
from .data_source_registry import get_all_tool_refs
from .data_source_registry import get_source_id_for_tool
from .data_sources import DEFAULT_DATA_SOURCES
from .data_sources import all_mapped_tools_filtered_out
from .data_sources import extract_messages_and_sources
from .data_sources import filter_tools_by_sources
from .data_sources import format_data_source_tools
from .data_sources import parse_data_sources
from .json_utils import extract_json
from .llm_provider import LLMProvider
from .llm_provider import LLMRole
from .message_utils import get_latest_user_query
from .prompt_utils import load_prompt
from .prompt_utils import render_prompt_template
from .tool_validation import format_tool_unavailability_error
from .tool_validation import format_user_facing_tool_error
from .tool_validation import validate_research_source_configuration
from .tool_validation import validate_tool_availability

logger = logging.getLogger(__name__)

# Shared checkpointer caches
_checkpointers: dict[str, BaseCheckpointSaver] = {}
_postgres_pools: dict[str, AsyncConnectionPool] = {}

__all__ = [
    "DEFAULT_DATA_SOURCES",
    "LLMProvider",
    "LLMRole",
    "SourceRegistry",
    "VerboseTraceCallback",
    "all_mapped_tools_filtered_out",
    "extract_json",
    "extract_messages_and_sources",
    "filter_tools_by_sources",
    "format_data_source_tools",
    "format_tool_unavailability_error",
    "format_user_facing_tool_error",
    "get_all_tool_refs",
    "get_checkpointer",
    "get_or_create_session_registry",
    "get_source_id_for_tool",
    "get_session_registry",
    "get_latest_user_query",
    "is_postgres_dsn",
    "load_prompt",
    "parse_data_sources",
    "register_source_parser",
    "render_prompt_template",
    "reset_session_registry",
    "sanitize_report",
    "set_session_registry",
    "validate_research_source_configuration",
    "validate_tool_availability",
    "verify_citations",
]


def _create_chat_response(
    content: str,
    response_id: str = "conversational_response",
    model: str | None = None,
) -> ChatResponse:
    """Create a standardized ChatResponse object."""
    return ChatResponse(
        id=response_id,
        model=model or "unknown-model",
        choices=[
            ChatResponseChoice(
                index=0,
                message=ChoiceMessage(content=content, role=UserMessageContentRoleType.ASSISTANT),
                finish_reason="stop",
            )
        ],
        created=datetime.datetime.now(datetime.UTC),
        usage=Usage(),
    )


def is_postgres_dsn(value: str) -> bool:
    """Return True when the checkpoint DSN is a Postgres URL."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        return parsed.scheme in ("postgresql", "postgres")
    except ValueError:
        return value.startswith(("postgresql://", "postgres://"))


async def get_checkpointer(checkpoint_db: str) -> BaseCheckpointSaver:
    """Return a shared checkpointer for the given database/DSN.

    This function caches checkpointers by database path/DSN to avoid
    creating multiple connections to the same database. It supports
    both SQLite (file path) and Postgres (DSN) backends.

    Args:
        checkpoint_db: SQLite database path or Postgres DSN.

    Returns:
        A configured and initialized checkpointer instance.
    """
    checkpointer = _checkpointers.get(checkpoint_db)
    if checkpointer is not None:
        return checkpointer

    if is_postgres_dsn(checkpoint_db):
        pool = _postgres_pools.get(checkpoint_db)
        if pool is None:
            pool = AsyncConnectionPool(
                conninfo=checkpoint_db,
                min_size=1,
                max_size=3,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            _postgres_pools[checkpoint_db] = pool
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("Postgres checkpointer initialized via async pool.")
    else:
        conn = await aiosqlite.connect(checkpoint_db)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        logger.info("SQLite checkpointer initialized: %s", checkpoint_db)

    _checkpointers[checkpoint_db] = checkpointer
    return checkpointer
