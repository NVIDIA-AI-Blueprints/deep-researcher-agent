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

"""Source routing catalog helpers for deep research."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel
from pydantic import Field

from deep_researcher_agent.common import filter_tools_by_sources
from deep_researcher_agent.common import get_source_id_for_tool
from deep_researcher_agent.common.data_source_registry import get_source


class DomainCatalogEntry(BaseModel):
    """Config-loaded domain-to-source preference entry shown to the router subagent."""

    domain_id: str
    domain_name: str
    description: str = ""
    preferred_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    fallback_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    is_default: bool = False


class DomainCatalogConfig(BaseModel):
    """Root schema for a source-router domain catalog file."""

    domains: tuple[DomainCatalogEntry, ...] = Field(default_factory=tuple)
    default_domain_id: str | None = None


DEFAULT_GENERAL_RESEARCH_DOMAIN_ID = "general_research"


def _default_general_research_config(source_ids: Sequence[str]) -> DomainCatalogConfig:
    """Return a default route over the runtime-configured source set."""
    preferred_source_ids = tuple(sorted(source_ids))
    fallback_source_ids = ("web_search",) if "web_search" in preferred_source_ids else preferred_source_ids[:1]
    return DomainCatalogConfig(
        default_domain_id=DEFAULT_GENERAL_RESEARCH_DOMAIN_ID,
        domains=(
            DomainCatalogEntry(
                domain_id=DEFAULT_GENERAL_RESEARCH_DOMAIN_ID,
                domain_name="General Research",
                description=(
                    "Default broad research route used when no domain catalog is configured. "
                    "Routes to all configured runtime sources."
                ),
                preferred_source_ids=preferred_source_ids,
                fallback_source_ids=fallback_source_ids,
                is_default=True,
            ),
        ),
    )


class DomainCatalogRegistry:
    """File-backed domain catalog registry."""

    def __init__(self, config: DomainCatalogConfig | None = None) -> None:
        self.config = config or DomainCatalogConfig()
        self._domains = {entry.domain_id: entry for entry in self.config.domains}

    @classmethod
    def from_path(cls, path: str | Path | None) -> DomainCatalogRegistry:
        """Load a domain catalog registry from a YAML or JSON file."""
        if path is None:
            return cls()

        catalog_path = Path(path).expanduser()
        if not catalog_path.exists():
            raise FileNotFoundError(f"Domain catalog file not found: {catalog_path}")

        raw_text = catalog_path.read_text(encoding="utf-8")
        if catalog_path.suffix.lower() == ".json":
            raw_data = json.loads(raw_text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - PyYAML is present in normal NAT environments
                raise ImportError("Domain catalog YAML files require PyYAML. Use JSON or install PyYAML.") from exc
            raw_data = yaml.safe_load(raw_text) or {}

        return cls(DomainCatalogConfig.model_validate(raw_data))

    @property
    def default_domain_id(self) -> str | None:
        """Return configured default domain ID when available."""
        if self.config.default_domain_id:
            return self.config.default_domain_id
        for entry in self.config.domains:
            if entry.is_default:
                return entry.domain_id
        return self.config.domains[0].domain_id if self.config.domains else None

    def domain_payloads(self, available_source_ids: set[str]) -> list[dict[str, Any]]:
        """Build catalog entries with source availability annotations."""
        domains: list[dict[str, Any]] = []
        for entry in self.config.domains:
            configured_ids = set(entry.preferred_source_ids) | set(entry.fallback_source_ids)
            domain = entry.model_dump(mode="json")
            domain["preferred_source_ids"] = _filter_available(entry.preferred_source_ids, available_source_ids)
            domain["fallback_source_ids"] = _filter_available(entry.fallback_source_ids, available_source_ids)
            domain["unavailable_source_ids"] = sorted(configured_ids - available_source_ids)
            domains.append(domain)
        return domains


def runtime_source_tools(
    tools: Sequence[BaseTool],
    *,
    allowed_source_ids: Sequence[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Return configured source IDs mapped to exact runtime tool names.

    Only tools that resolve through the data source registry are included.
    Unmapped helper tools are deliberately kept out of source routing.
    """
    runtime_tools = list(tools)
    if allowed_source_ids is not None:
        runtime_tools = filter_tools_by_sources(runtime_tools, list(allowed_source_ids))

    source_tools: dict[str, list[dict[str, str]]] = {}
    for runtime_tool in runtime_tools:
        tool_name = getattr(runtime_tool, "name", "")
        if not tool_name:
            continue
        source_id = get_source_id_for_tool(tool_name)
        if source_id is None:
            continue
        source_tools.setdefault(source_id, []).append(
            {
                "name": tool_name,
                "description": getattr(runtime_tool, "description", "") or "",
            }
        )

    return {
        source_id: sorted(tool_entries, key=lambda entry: entry["name"])
        for source_id, tool_entries in sorted(source_tools.items())
    }


def _source_metadata(source_id: str) -> dict[str, Any]:
    """Return source metadata from the global registry when available."""
    source = get_source(source_id)
    if source is None:
        return {
            "source_id": source_id,
            "source_name": source_id.replace("_", " ").title(),
            "description": "",
            "default_enabled": True,
            "requires_auth": False,
        }
    return {
        "source_id": source.id,
        "source_name": source.name,
        "description": source.description,
        "default_enabled": source.default_enabled,
        "requires_auth": source.requires_auth,
    }


def _available_source_payload(source_id: str, tools_for_source: list[dict[str, str]]) -> dict[str, Any]:
    """Build one available source entry for the router catalog."""
    return {**_source_metadata(source_id), "tools": tools_for_source}


def _filter_available(source_ids: Sequence[str], available_source_ids: set[str]) -> list[str]:
    """Keep only source IDs that exist in the active runtime tool set."""
    return [source_id for source_id in source_ids if source_id in available_source_ids]


def _default_fallback_source_ids(available_source_ids: set[str]) -> list[str]:
    """Return a conservative fallback source order for requests with no clear domain."""
    if "web_search" in available_source_ids:
        return ["web_search"]
    return sorted(available_source_ids)[:1]


def source_catalog_payload(
    tools: Sequence[BaseTool],
    *,
    allowed_source_ids: Sequence[str] | None = None,
    domain_catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the complete source/domain catalog shown to source-router-agent."""
    source_tools = runtime_source_tools(tools, allowed_source_ids=allowed_source_ids)
    if domain_catalog_path is None:
        domain_registry = DomainCatalogRegistry(_default_general_research_config(list(source_tools)))
    else:
        domain_registry = DomainCatalogRegistry.from_path(domain_catalog_path)
    available_source_ids = set(source_tools)

    unmapped_tools = []
    for runtime_tool in tools:
        tool_name = getattr(runtime_tool, "name", "")
        if tool_name and get_source_id_for_tool(tool_name) is None:
            unmapped_tools.append(tool_name)

    return {
        "available_sources": [
            _available_source_payload(source_id, tools_for_source)
            for source_id, tools_for_source in source_tools.items()
        ],
        "domains": domain_registry.domain_payloads(available_source_ids),
        "default_domain_id": domain_registry.default_domain_id,
        "default_fallback_source_ids": _default_fallback_source_ids(available_source_ids),
        "unmapped_runtime_tools": sorted(unmapped_tools),
        "policy": [
            "Recommend only source IDs present in available_sources.",
            "Use exact tool names from available_sources.tools.",
            "If the best domain has unavailable preferred sources, use its available fallbacks.",
            "If no domain fits well, choose default_domain_id when present and use default_fallback_source_ids.",
            "If no domains are configured, set domain_id to unconfigured and use default_fallback_source_ids.",
            "The router is advisory. The planner still decides final ResearchQuery objects.",
        ],
    }


def build_lookup_source_catalog_tool(
    tools: Sequence[BaseTool],
    *,
    allowed_source_ids: Sequence[str] | None = None,
    domain_catalog_path: str | Path | None = None,
) -> BaseTool:
    """Build the router-only source catalog lookup tool."""

    @tool
    def lookup_source_catalog() -> str:
        """Return configured source domains, available source IDs, and exact source tool names."""
        return json.dumps(
            source_catalog_payload(
                tools,
                allowed_source_ids=allowed_source_ids,
                domain_catalog_path=domain_catalog_path,
            ),
            indent=2,
            ensure_ascii=False,
        )

    return lookup_source_catalog
