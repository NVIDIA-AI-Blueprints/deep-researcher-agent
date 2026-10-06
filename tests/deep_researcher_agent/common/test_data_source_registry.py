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

"""Tests for data_source_registry module."""

import pytest

from deep_researcher_agent.common.data_source_registry import DataSourceEntry
from deep_researcher_agent.common.data_source_registry import DataSourceMeta
from deep_researcher_agent.common.data_source_registry import get_all_sources
from deep_researcher_agent.common.data_source_registry import get_all_tool_refs
from deep_researcher_agent.common.data_source_registry import get_source
from deep_researcher_agent.common.data_source_registry import get_source_id_for_tool
from deep_researcher_agent.common.data_source_registry import populate_from_config
from deep_researcher_agent.common.data_source_registry import register_tool_sources
from deep_researcher_agent.common.data_source_registry import reset_registry
from deep_researcher_agent.common.data_source_registry import set_group_source_map
from deep_researcher_agent.common.data_source_registry import set_tool_source_map


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


class TestPopulateFromConfig:
    """Tests for populate_from_config (YAML-driven registration)."""

    def test_populates_registry_and_tool_map(self):
        """Test that populate_from_config loads sources and tool mappings."""
        populate_from_config(
            [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web.",
                    "tools": ["web_search_tool", "advanced_web_search_tool"],
                },
                {
                    "id": "knowledge_layer",
                    "name": "Knowledge Base",
                    "description": "Search documents.",
                    "tools": ["knowledge_search"],
                },
            ]
        )

        assert len(get_all_sources()) == 2
        meta = get_source("web_search")
        assert meta is not None
        assert meta.name == "Web Search"

        assert get_source_id_for_tool("web_search_tool") == "web_search"
        assert get_source_id_for_tool("advanced_web_search_tool") == "web_search"
        assert get_source_id_for_tool("knowledge_search") == "knowledge_layer"

    def test_default_enabled_false(self):
        """Test that default_enabled can be set to False."""
        populate_from_config(
            [
                {"id": "disabled_src", "name": "D", "description": "D", "default_enabled": False},
            ]
        )

        meta = get_source("disabled_src")
        assert meta is not None
        assert meta.default_enabled is False

    def test_requires_auth_true(self):
        """Test that requires_auth is stored on the metadata."""
        populate_from_config(
            [
                {"id": "eci", "name": "ECI", "description": "Enterprise.", "requires_auth": True},
            ]
        )

        meta = get_source("eci")
        assert meta is not None
        assert meta.requires_auth is True

    def test_requires_auth_defaults_to_false(self):
        """Test that requires_auth defaults to False when omitted."""
        populate_from_config(
            [
                {"id": "web", "name": "Web", "description": "Web search."},
            ]
        )

        meta = get_source("web")
        assert meta.requires_auth is False

    def test_missing_name_uses_title_cased_id(self):
        """Test that missing name falls back to title-cased ID."""
        populate_from_config([{"id": "my_custom_search", "description": "Custom search."}])

        meta = get_source("my_custom_search")
        assert meta.name == "My Custom Search"

    def test_source_without_tools(self):
        """Test that a source with no tools is registered but has no tool mappings."""
        populate_from_config([{"id": "empty_src", "name": "Empty", "description": "No tools."}])

        assert get_source("empty_src") is not None
        assert get_source_id_for_tool("anything") is None

    def test_group_refs_populate_prefix_map(self):
        """Test that tools listed as group_names enable prefix matching."""
        populate_from_config(
            [{"id": "my_group", "name": "My Group", "description": "A group source.", "tools": ["my_group"]}],
            group_names={"my_group"},
        )

        assert get_source_id_for_tool("my_group__tool_a") == "my_group"
        assert get_source_id_for_tool("my_group__tool_b") == "my_group"
        assert get_source_id_for_tool("web_search_tool") is None

    def test_child_tool_matches_data_source_ref_when_group_detection_misses_it(self):
        """MCP-style child tools still match refs declared under data_sources."""
        populate_from_config(
            [{"id": "mcp_time", "name": "MCP Time", "description": "Get current time.", "tools": ["mcp_time"]}]
        )

        assert get_source_id_for_tool("mcp_time__get_current_time") == "mcp_time"


class TestGetAllSources:
    def test_empty_registry(self):
        assert get_all_sources() == []

    def test_returns_all_registered(self):
        populate_from_config(
            [
                {"id": "a", "name": "A", "description": "A"},
                {"id": "b", "name": "B", "description": "B"},
            ]
        )

        sources = get_all_sources()
        ids = {s.id for s in sources}
        assert ids == {"a", "b"}


class TestGetSource:
    def test_returns_none_for_unknown(self):
        assert get_source("nonexistent") is None

    def test_returns_metadata_for_known(self):
        populate_from_config([{"id": "known", "name": "Known", "description": "Known source"}])

        meta = get_source("known")
        assert isinstance(meta, DataSourceMeta)
        assert meta.name == "Known"


class TestRegisterToolSources:
    def test_adds_new_mappings(self):
        register_tool_sources({"gdrive_search": "gdrive"})
        assert get_source_id_for_tool("gdrive_search") == "gdrive"

    def test_reregistering_same_pair_is_idempotent(self):
        register_tool_sources({"gdrive_search": "gdrive"})
        register_tool_sources({"gdrive_search": "gdrive"})
        assert get_source_id_for_tool("gdrive_search") == "gdrive"

    def test_collision_keeps_existing_owner_and_skips(self, caplog):
        register_tool_sources({"shared_tool": "source_a"})
        with caplog.at_level("WARNING"):
            register_tool_sources({"shared_tool": "source_b", "new_tool": "source_b"})

        # Existing owner preserved; conflicting remap skipped; unrelated key still added.
        assert get_source_id_for_tool("shared_tool") == "source_a"
        assert get_source_id_for_tool("new_tool") == "source_b"
        assert "shared_tool" in caplog.text


class TestToolSourceMap:
    def test_get_source_id_for_tool_returns_none_when_empty(self):
        assert get_source_id_for_tool("any_tool") is None

    def test_set_and_get_tool_source_map(self):
        set_tool_source_map({"web_search_tool": "web_search", "knowledge_search": "knowledge_layer"})

        assert get_source_id_for_tool("web_search_tool") == "web_search"
        assert get_source_id_for_tool("knowledge_search") == "knowledge_layer"
        assert get_source_id_for_tool("calculator") is None

    def test_reset_clears_tool_map(self):
        set_tool_source_map({"tool": "source"})
        reset_registry()
        assert get_source_id_for_tool("tool") is None

    def test_group_matches_nat_separator(self):
        """Function group tools use NAT's __ separator."""
        set_group_source_map({"my_group": "my_group"})

        assert get_source_id_for_tool("my_group__tool_a") == "my_group"
        assert get_source_id_for_tool("my_group__tool_b") == "my_group"

    def test_group_matches_legacy_separator(self):
        """Function group tools also match NAT's legacy . separator."""
        set_group_source_map({"my_group": "my_group"})

        assert get_source_id_for_tool("my_group.tool_a") == "my_group"

    def test_group_rejects_no_separator(self):
        """Bare prefix without NAT separator does not match."""
        set_group_source_map({"my_group": "my_group"})

        assert get_source_id_for_tool("my_group") is None
        assert get_source_id_for_tool("my_groupfoo") is None
        assert get_source_id_for_tool("my_group_tool") is None  # single underscore is not a NAT separator

    def test_exact_match_takes_priority_over_group(self):
        """Exact tool match should win over group prefix match."""
        set_tool_source_map({"my_group__special": "special_source"})
        set_group_source_map({"my_group": "my_group"})

        assert get_source_id_for_tool("my_group__special") == "special_source"
        assert get_source_id_for_tool("my_group__other") == "my_group"

    def test_overlapping_group_prefixes_match_longest(self):
        """When group prefixes overlap, the longest matching prefix wins."""
        set_group_source_map({"abc": "source_short", "abc_special": "source_long"})

        assert get_source_id_for_tool("abc_special__tool") == "source_long"
        assert get_source_id_for_tool("abc__tool") == "source_short"

    def test_unknown_tool_returns_none_with_groups(self):
        """Tools not matching any exact or prefix mapping return None."""
        set_tool_source_map({"web_search_tool": "web_search"})
        set_group_source_map({"my_group": "my_group"})

        assert get_source_id_for_tool("calculator") is None


class TestGetAllToolRefs:
    """Tests for get_all_tool_refs (auto-inherit helper)."""

    def test_empty_registry(self):
        assert get_all_tool_refs() == []

    def test_returns_individual_tools(self):
        populate_from_config(
            [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web.",
                    "tools": ["web_search_tool", "advanced_web_search_tool"],
                },
            ]
        )

        refs = get_all_tool_refs()
        assert "web_search_tool" in refs
        assert "advanced_web_search_tool" in refs

    def test_returns_group_names(self):
        populate_from_config(
            [
                {
                    "id": "eci",
                    "name": "Enterprise Search",
                    "description": "Search enterprise content.",
                    "tools": ["eci"],
                },
            ],
            group_names={"eci"},
        )

        refs = get_all_tool_refs()
        assert "eci" in refs

    def test_returns_both_individual_and_group(self):
        populate_from_config(
            [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Search the web.",
                    "tools": ["web_search_tool"],
                },
                {
                    "id": "eci",
                    "name": "Enterprise Search",
                    "description": "Search enterprise content.",
                    "tools": ["eci"],
                },
            ],
            group_names={"eci"},
        )

        refs = get_all_tool_refs()
        assert set(refs) == {"web_search_tool", "eci"}

    def test_returns_sorted(self):
        populate_from_config(
            [
                {
                    "id": "src",
                    "name": "Src",
                    "description": "Src.",
                    "tools": ["z_tool", "a_tool", "m_tool"],
                },
            ]
        )

        refs = get_all_tool_refs()
        assert refs == sorted(refs)


class TestDataSourceEntry:
    """Tests for the DataSourceEntry pydantic config model."""

    def test_requires_auth_defaults_false(self):
        entry = DataSourceEntry(id="web", name="Web", description="Web search.")
        assert entry.requires_auth is False

    def test_requires_auth_set_true(self):
        entry = DataSourceEntry(id="eci", name="ECI", requires_auth=True)
        assert entry.requires_auth is True
