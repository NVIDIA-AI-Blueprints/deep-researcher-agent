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

"""Tools backed by the deep researcher source registry."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from ..custom_middleware import SourceRegistryMiddleware


def build_get_verified_sources_tool(registry_middleware: SourceRegistryMiddleware) -> BaseTool:
    """Build the get_verified_sources tool for the active source registry."""

    @tool
    def get_verified_sources(mode: str = "compact") -> str:
        """Returns verified source URLs captured from search tool calls.

        Call this tool during synthesis BEFORE writing the final answer. It
        returns the compact writer-facing source list by default: sources that
        researcher workers carried forward into ResearchNotes. Pass
        mode="full" only if a needed ResearchNotes source is missing from the
        compact list. Use ONLY returned sources in your final answer. Any
        other URL will be automatically removed.

        Args:
            mode: "compact" for ResearchNotes-backed sources, or "full" for
                every captured source in the registry.
        Returns:
            A numbered list of verified sources with titles and URLs.
        """
        source_mode = "full" if mode == "full" else "compact"
        source_list = registry_middleware.get_source_list_text(mode=source_mode)
        if source_list:
            return source_list
        return "No sources captured yet. Run research queries first."

    return get_verified_sources
