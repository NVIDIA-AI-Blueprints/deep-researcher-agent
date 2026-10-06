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

"""Config-driven data source registry.

Data source display metadata and tool→source mappings are loaded from
a ``data_source_registry`` function in the YAML config.  Each source
entry declares its display name, description, and which NAT functions
or function groups belong to it.

Individual functions are matched by exact name.  Function groups are
matched by prefix using NAT's group separators (``__`` and legacy ``.``).
The register function auto-detects which refs are groups by checking
the builder at startup.

Example YAML::

    functions:
      data_sources:
        _type: data_source_registry
        sources:
          - id: web_search
            name: "Web Search"
            description: "Search the web for real-time information."
            tools:
              - web_search_tool
              - advanced_web_search_tool
          - id: eci
            name: "Enterprise Search"
            description: "Search Confluence, Google Drive, and more."
            requires_auth: true
            tools:
              - eci
"""

import logging
from dataclasses import dataclass
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

_GROUP_SEPARATORS = (FunctionGroup.SEPARATOR, FunctionGroup.LEGACY_SEPARATOR)

logger = logging.getLogger(__name__)


class PerUserAuthConfig(BaseModel):
    """Static per-user MCP auth declaration for a protected data source.

    This is the configuration-time metadata (declared in YAML). The dynamic,
    per-request state (status / connect_url / auth_url / expires_at) is computed
    by the API layer from this config plus the user's token storage state — see
    ``deep_researcher_api.mcp_auth``.
    """

    required: bool = Field(default=False, description="Whether per-user MCP auth is required for this source")
    type: Literal["mcp_oauth2"] = Field(default="mcp_oauth2", description="Auth mechanism (only mcp_oauth2 today)")
    provider: str | None = Field(default=None, description="Provider identifier, e.g. 'google'")
    mcp_server_id: str | None = Field(
        default=None,
        description="MCP server / auth-provider key used to look up the user's token in NAT token storage",
    )
    auth_provider: str | None = Field(
        default=None,
        description=(
            "Name of the NAT `authentication` (mcp_oauth2) provider for this source, resolved via the builder "
            "to derive OAuth settings + shared token storage (connect flow) and to build the per-user MCP "
            "client in the worker at job time. Defaults to mcp_server_id when unset. NOTE: no config-declared "
            "per_user_mcp_client function group is used — "
            "see deep_researcher_api.mcp_auth.runtime_tools for why (it breaks "
            "the interactive WebSocket path)."
        ),
    )
    tool_overrides: dict[str, dict[str, str]] | None = Field(
        default=None,
        description=(
            "Optional per-tool overrides for this source's MCP tools, keyed by the server's tool name, each "
            "mapping to {alias, description}. Use to give terse/blank MCP tools clear names and descriptions so "
            "the agent reliably selects them (e.g. gdrive_get_file -> a descriptive 'read a Drive file' tool)."
        ),
    )


@dataclass(frozen=True)
class DataSourceMeta:
    """Metadata for a registered data source."""

    id: str
    name: str
    description: str
    default_enabled: bool = True
    requires_auth: bool = False
    per_user_auth: PerUserAuthConfig | None = None


# ── Global state ──
_registry: dict[str, DataSourceMeta] = {}
_tool_source_map: dict[str, str] = {}  # exact tool name → source ID
_group_source_map: dict[str, str] = {}  # group prefix → source ID
# Combined (group refs + exact refs) pre-sorted longest-first for prefix
# matching. Exact refs participate in prefix matching too so that MCP-style
# child tools (``foo__bar``) resolve to their parent source even when the
# ref was declared without being marked as a function group — matches the
# behavior the NAT auto-detection would give in production but is robust
# to tests and configs that forget to pass ``group_names``.
_sorted_prefix_refs: list[tuple[str, str]] = []


# ── Config models ──
class DataSourceEntry(BaseModel):
    """A single data source with its display metadata and tool references."""

    id: str = Field(..., description="Source ID used in API and filtering (e.g. 'web_search')")
    name: str = Field(..., description="Display name shown in the UI")
    description: str = Field(default="", description="Human-readable description")
    default_enabled: bool = Field(default=True, description="Whether enabled by default")
    requires_auth: bool = Field(default=False, description="Whether the source requires user authentication")
    per_user_auth: PerUserAuthConfig | None = Field(
        default=None,
        description="Per-user MCP OAuth declaration for a protected source (see PerUserAuthConfig)",
    )
    tools: list[FunctionRef] = Field(
        default_factory=list,
        description="NAT functions or function groups that belong to this data source",
    )


class DataSourceRegistryConfig(FunctionBaseConfig, name="data_source_registry"):
    """Config-driven data source registry with NAT function references."""

    sources: list[DataSourceEntry] = Field(
        default_factory=list,
        description="List of data source definitions",
    )


def _populate(sources: list[dict], group_names: set[str] | None = None) -> None:
    """Shared logic for populating the registry and maps.

    Used by both the NAT register function and the test helper.

    Args:
        sources: List of dicts with keys: id, name, description, tools.
        group_names: Ref names to treat as function groups (prefix matching).
    """
    group_names = group_names or set()
    tool_map: dict[str, str] = {}
    group_map: dict[str, str] = {}

    for entry in sources:
        source_id = entry["id"]
        per_user_auth = entry.get("per_user_auth")
        if per_user_auth is not None and not isinstance(per_user_auth, PerUserAuthConfig):
            per_user_auth = PerUserAuthConfig.model_validate(per_user_auth)
        _registry[source_id] = DataSourceMeta(
            id=source_id,
            name=entry.get("name", source_id.replace("_", " ").title()),
            description=entry.get("description", ""),
            default_enabled=entry.get("default_enabled", True),
            requires_auth=entry.get("requires_auth", False),
            per_user_auth=per_user_auth,
        )
        for ref_name in entry.get("tools", []):
            if ref_name in group_names:
                group_map[ref_name] = source_id
            else:
                tool_map[ref_name] = source_id

    set_tool_source_map(tool_map)
    set_group_source_map(group_map)


@register_function(config_type=DataSourceRegistryConfig)
async def data_source_registry_fn(config: DataSourceRegistryConfig, builder: Any):
    """Populate the global registry and tool/group maps from YAML config.

    Auto-detects whether each tool ref is a function group (prefix match)
    or individual function (exact match) by checking the builder.
    """
    # builder._function_groups is a dict of registered function groups
    function_groups = set(getattr(builder, "_function_groups", {}))

    entries = [
        {
            "id": entry.id,
            "name": entry.name,
            "description": entry.description,
            "default_enabled": entry.default_enabled,
            "requires_auth": entry.requires_auth,
            "per_user_auth": entry.per_user_auth,
            "tools": [str(ref) for ref in entry.tools],
        }
        for entry in config.sources
    ]
    _populate(entries, group_names=function_groups)

    logger.info(
        "Loaded %d data source(s): %d tool(s), %d group(s)",
        len(config.sources),
        len(_tool_source_map),
        len(_group_source_map),
    )

    # NAT requires yielding a FunctionInfo; this is a config-only function
    async def _noop(query: str) -> str:
        """Data source registry (config-only, not a tool)."""
        return "This is a config-only function."

    yield FunctionInfo.from_fn(_noop, description="Data source registry (config-only)")


# ── Lookup helpers ──
def get_all_tool_refs() -> list[str]:
    """Return all tool refs (individual names + group names) from the registry.

    Used by agents that inherit tools automatically when their ``tools``
    config list is empty.
    """
    refs: set[str] = set()
    refs.update(_tool_source_map.keys())
    refs.update(_group_source_map.keys())
    return sorted(refs)


def get_all_sources() -> list[DataSourceMeta]:
    return list(_registry.values())


def get_source(source_id: str) -> DataSourceMeta | None:
    return _registry.get(source_id)


def get_source_id_for_tool(tool_name: str) -> str | None:
    """Look up source ID for a tool.

    Tries exact match first (individual functions), then prefix match
    against any registered ref (group or exact) using NAT's function
    group separators (``__`` and legacy ``.``). Prefix candidates are
    matched longest-first for deterministic results with overlapping
    refs.
    """
    source_id = _tool_source_map.get(tool_name)
    if source_id is not None:
        return source_id

    for ref_name, source_id in _sorted_prefix_refs:
        if any(tool_name.startswith(ref_name + sep) for sep in _GROUP_SEPARATORS):
            return source_id

    return None


def _rebuild_prefix_index() -> None:
    """Rebuild the combined prefix-match index from the tool/group maps.

    Group refs win over exact refs with the same name (they were explicitly
    declared as groups); otherwise longer refs win over shorter ones.
    """
    global _sorted_prefix_refs
    combined: dict[str, str] = dict(_tool_source_map)
    combined.update(_group_source_map)  # group refs take precedence on tie
    _sorted_prefix_refs = sorted(combined.items(), key=lambda item: len(item[0]), reverse=True)


def set_tool_source_map(mapping: dict[str, str]) -> None:
    """Set the exact tool→source mapping."""
    global _tool_source_map
    _tool_source_map = mapping
    _rebuild_prefix_index()


def set_group_source_map(mapping: dict[str, str]) -> None:
    """Set the group prefix→source mapping."""
    global _group_source_map
    _group_source_map = mapping
    _rebuild_prefix_index()


def register_tool_sources(mapping: dict[str, str]) -> None:
    """Merge additional exact tool→source mappings into the registry.

    Used for tools that only become known at runtime — notably a per-user MCP
    source's tools (resolved per request), which aren't in the static config.
    Registering them lets ``get_source_id_for_tool`` resolve their source so
    citation/source capture treats their results as sources (otherwise shallow
    research raises EmptySourceRegistryError). Keys are deterministic, so merging
    globally is idempotent.

    Re-registering the same tool→source pair is a no-op. A key that already maps
    to a *different* source is a collision (e.g. two sources exposing a tool with
    the same name); we keep the existing owner and skip the conflicting entry
    rather than silently reassigning citation ownership.
    """
    if not mapping:
        return
    global _tool_source_map
    merged = dict(_tool_source_map)
    for tool, source in mapping.items():
        existing = merged.get(tool)
        if existing is not None and existing != source:
            logger.warning(
                "Tool '%s' already maps to source '%s'; skipping conflicting registration to '%s'.",
                tool,
                existing,
                source,
            )
            continue
        merged[tool] = source
    _tool_source_map = merged
    _rebuild_prefix_index()


# ── Test helpers ──
def populate_from_config(sources: list[dict], group_names: set[str] | None = None) -> None:
    """Populate registry and maps directly (for testing without NAT startup).

    Args:
        sources: List of dicts with keys: id, name, description, tools (list of str).
        group_names: Set of tool ref names that should be treated as function groups
                     (prefix matching). If None, all refs are treated as exact matches.
    """
    _populate(sources, group_names=group_names)


def reset_registry() -> None:
    """Reset for testing."""
    _registry.clear()
    global _tool_source_map, _group_source_map, _sorted_prefix_refs
    _tool_source_map = {}
    _group_source_map = {}
    _sorted_prefix_refs = []
