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

"""Tests for common module __init__.py utilities."""

import os
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from deep_researcher_agent.common import DEFAULT_DATA_SOURCES
from deep_researcher_agent.common import _create_chat_response
from deep_researcher_agent.common import extract_messages_and_sources
from deep_researcher_agent.common import filter_tools_by_sources
from deep_researcher_agent.common import format_data_source_tools
from deep_researcher_agent.common import get_checkpointer
from deep_researcher_agent.common import is_postgres_dsn
from deep_researcher_agent.common import parse_data_sources


class TestCreateChatResponse:
    """Tests for the _create_chat_response function."""

    def test_create_chat_response_basic(self):
        """Test basic chat response creation."""
        response = _create_chat_response("Hello, world!")

        assert response.id == "conversational_response"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello, world!"
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].index == 0

    def test_create_chat_response_custom_id(self):
        """Test chat response creation with custom ID."""
        response = _create_chat_response("Test content", response_id="custom_id")

        assert response.id == "custom_id"
        assert response.choices[0].message.content == "Test content"

    def test_create_chat_response_custom_model(self):
        """Test chat response creation with custom model."""
        response = _create_chat_response("Test content", model="deep_research_workflow")

        assert response.model == "deep_research_workflow"

    def test_create_chat_response_default_model(self):
        """Test chat response creation falls back to unknown-model."""
        response = _create_chat_response("Test content")

        assert response.model == "unknown-model"

    def test_create_chat_response_empty_content(self):
        """Test chat response with empty content."""
        response = _create_chat_response("")

        assert response.choices[0].message.content == ""

    def test_create_chat_response_long_content(self):
        """Test chat response with long content."""
        long_content = "A" * 10000
        response = _create_chat_response(long_content)

        assert response.choices[0].message.content == long_content
        assert len(response.choices[0].message.content) == 10000

    def test_create_chat_response_has_created_timestamp(self):
        """Test that response has a created timestamp."""
        response = _create_chat_response("Test")

        assert response.created is not None

    def test_create_chat_response_has_usage(self):
        """Test that response has usage field."""
        response = _create_chat_response("Test")

        assert response.usage is not None

    def test_create_chat_response_special_characters(self):
        """Test chat response with special characters."""
        special_content = "Hello\n\tWorld! 🌍 <script>alert('xss')</script>"
        response = _create_chat_response(special_content)

        assert response.choices[0].message.content == special_content

    def test_create_chat_response_unicode(self):
        """Test chat response with Unicode content."""
        unicode_content = "こんにちは世界 Привет мир مرحبا بالعالم"
        response = _create_chat_response(unicode_content)

        assert response.choices[0].message.content == unicode_content


class TestDataSourcesExports:
    """Tests to verify data sources utilities are properly exported."""

    def test_default_data_sources_exported(self):
        """Test DEFAULT_DATA_SOURCES is exported."""
        assert DEFAULT_DATA_SOURCES is not None
        assert isinstance(DEFAULT_DATA_SOURCES, list)
        assert "web_search" in DEFAULT_DATA_SOURCES

    def test_parse_data_sources_exported(self):
        """Test parse_data_sources is exported and functional."""
        result = parse_data_sources(["web_search"])
        assert result == ["web_search"]

    def test_filter_tools_by_sources_exported(self):
        """Test filter_tools_by_sources is exported and functional."""
        result = filter_tools_by_sources([], None)
        assert result == []

    def test_extract_messages_and_sources_exported(self):
        """Test extract_messages_and_sources is exported and functional."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Test")]
        result_messages, result_sources = extract_messages_and_sources(messages)
        assert result_messages == messages
        assert result_sources is None

    def test_format_data_source_tools_exported(self):
        """Test format_data_source_tools is exported and functional."""
        result = format_data_source_tools(["web_search"])
        assert len(result) == 1
        # Falls back to title-cased ID when registry is empty
        assert result[0]["name"] == "Web Search"

    def test_get_all_tool_refs_exported(self):
        """Test get_all_tool_refs is exported."""
        from deep_researcher_agent.common import get_all_tool_refs

        assert callable(get_all_tool_refs)

    def test_get_source_id_for_tool_exported(self):
        """Test get_source_id_for_tool is exported."""
        from deep_researcher_agent.common import get_source_id_for_tool

        assert callable(get_source_id_for_tool)


class TestIsPostgresDsn:
    """Tests for the is_postgres_dsn function."""

    def test_postgresql_scheme(self):
        """Test that postgresql:// scheme is recognized."""
        assert is_postgres_dsn("postgresql://user:pass@localhost:5432/db") is True

    def test_postgres_scheme(self):
        """Test that postgres:// scheme is recognized."""
        assert is_postgres_dsn("postgres://user:pass@localhost:5432/db") is True

    def test_sqlite_path_not_postgres(self):
        """Test that SQLite paths are not recognized as Postgres."""
        assert is_postgres_dsn("./checkpoints.db") is False
        assert is_postgres_dsn("/tmp/checkpoints.db") is False

    def test_empty_string(self):
        """Test that empty string is not Postgres."""
        assert is_postgres_dsn("") is False

    def test_http_url_not_postgres(self):
        """Test that HTTP URLs are not recognized as Postgres."""
        assert is_postgres_dsn("http://localhost:5432") is False
        assert is_postgres_dsn("https://example.com") is False

    def test_postgres_with_options(self):
        """Test Postgres DSN with connection options."""
        dsn = "postgresql://user:pass@localhost:5432/db?sslmode=require"
        assert is_postgres_dsn(dsn) is True

    def test_postgres_without_port(self):
        """Test Postgres DSN without explicit port."""
        assert is_postgres_dsn("postgresql://user:pass@localhost/db") is True


class TestGetCheckpointer:
    """Tests for the get_checkpointer function."""

    @pytest.fixture(autouse=True)
    def clear_checkpointer_cache(self):
        """Clear the checkpointer cache before each test."""
        import deep_researcher_agent.common as common_module

        common_module._checkpointers.clear()
        common_module._postgres_pools.clear()
        yield
        common_module._checkpointers.clear()
        common_module._postgres_pools.clear()

    @pytest.mark.asyncio
    async def test_sqlite_checkpointer_creation(self):
        """Test that SQLite checkpointer is created for file paths."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            checkpointer = await get_checkpointer(db_path)

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            assert isinstance(checkpointer, AsyncSqliteSaver)
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_checkpointer_caching(self):
        """Test that checkpointers are cached and reused."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            checkpointer1 = await get_checkpointer(db_path)
            checkpointer2 = await get_checkpointer(db_path)

            assert checkpointer1 is checkpointer2
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_different_paths_different_checkpointers(self):
        """Test that different paths create different checkpointers."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f1:
            db_path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f2:
            db_path2 = f2.name

        try:
            checkpointer1 = await get_checkpointer(db_path1)
            checkpointer2 = await get_checkpointer(db_path2)

            assert checkpointer1 is not checkpointer2
        finally:
            os.unlink(db_path1)
            os.unlink(db_path2)

    @pytest.mark.asyncio
    async def test_postgres_checkpointer_creation(self):
        """Test that Postgres checkpointer is created for Postgres DSNs."""
        postgres_dsn = "postgresql://user:pass@localhost:5432/testdb"

        mock_pool = MagicMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()

        with (
            patch("deep_researcher_agent.common.AsyncConnectionPool", return_value=mock_pool),
            patch("deep_researcher_agent.common.AsyncPostgresSaver", return_value=mock_checkpointer),
        ):
            checkpointer = await get_checkpointer(postgres_dsn)

            assert checkpointer is mock_checkpointer
            mock_checkpointer.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_postgres_pool_caching(self):
        """Test that Postgres pools are cached and reused."""
        postgres_dsn = "postgresql://user:pass@localhost:5432/testdb"

        mock_pool = MagicMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()

        with (
            patch("deep_researcher_agent.common.AsyncConnectionPool", return_value=mock_pool) as mock_pool_class,
            patch("deep_researcher_agent.common.AsyncPostgresSaver", return_value=mock_checkpointer),
        ):
            await get_checkpointer(postgres_dsn)

            import deep_researcher_agent.common as common_module

            common_module._checkpointers.clear()

            await get_checkpointer(postgres_dsn)

            assert mock_pool_class.call_count == 1
