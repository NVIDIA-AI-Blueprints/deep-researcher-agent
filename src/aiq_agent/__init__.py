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

"""AI-Q Blueprint core package.

This module uses lazy imports to avoid loading heavy dependencies (langgraph, etc.)
when only lightweight submodules like `aiq_agent.knowledge` are needed.
"""

__all__ = [
    "chat_deepresearcher_agent",
    "data_science_agent",
    "shallow_research_agent",
    "deep_research_agent",
]

from typing import Any

# Cache for lazy-loaded modules to avoid repeated imports
_lazy_imports: dict[str, Any] = {}


def __getattr__(name: str):
    """Lazy import agents to avoid loading langgraph/ray dependencies unnecessarily.

    This allows `from aiq_agent.knowledge import ...` to work without pulling in
    the full agent stack and its heavy dependencies (langgraph, etc.).
    """
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "chat_deepresearcher_agent":
        from .agents import chat_deepresearcher_agent

        _lazy_imports[name] = chat_deepresearcher_agent
        return chat_deepresearcher_agent

    if name == "shallow_research_agent":
        from .agents import shallow_research_agent

        _lazy_imports[name] = shallow_research_agent
        return shallow_research_agent

    if name == "data_science_agent":
        from .agents import data_science_agent

        _lazy_imports[name] = data_science_agent
        return data_science_agent

    if name == "deep_research_agent":
        from .agents import deep_research_agent

        _lazy_imports[name] = deep_research_agent
        return deep_research_agent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
