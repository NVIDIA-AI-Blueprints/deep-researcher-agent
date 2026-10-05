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

"""
Agent registry for async job system.

Maps agent type identifiers to their class paths and config names.
This allows the job runner to dynamically load and instantiate agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a registered agent."""

    class_path: str
    """Full module path to the agent class, e.g., 'aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent'"""

    config_name: str
    """NAT config function name, e.g., 'deep_research_agent'"""

    description: str = ""
    """Human-readable description of the agent."""

    public: bool = True
    """Whether this agent should be exposed through public agent listing and submission surfaces."""


# Global agent registry
AGENT_REGISTRY: dict[str, AgentConfig] = {}


def register_agent(
    agent_type: str,
    class_path: str,
    config_name: str,
    description: str = "",
    public: bool = True,
) -> None:
    """
    Register an agent type for use with the async job system.

    Args:
        agent_type: Short identifier for the agent (e.g., 'deep_researcher').
        class_path: Full module path to the agent class.
        config_name: NAT config function name for this agent.
        description: Human-readable description.
        public: Whether this agent is visible to public agent listing and direct submission.

    Example:
        register_agent(
            agent_type="deep_researcher",
            class_path="aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent",
            config_name="deep_research_agent",
            description="Performs comprehensive multi-loop research",
        )
    """
    AGENT_REGISTRY[agent_type] = AgentConfig(
        class_path=class_path,
        config_name=config_name,
        description=description,
        public=public,
    )
    logger.debug("Registered agent: %s -> %s", agent_type, class_path)


def get_agent_config(agent_type: str) -> AgentConfig:
    """
    Get configuration for a registered agent.

    Args:
        agent_type: The agent type identifier.

    Returns:
        AgentConfig for the requested agent.

    Raises:
        KeyError: If agent_type is not registered.
    """
    if agent_type not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys()) or "(none)"
        raise KeyError(f"Unknown agent type: '{agent_type}'. Available: {available}")
    return AGENT_REGISTRY[agent_type]


# Register default agents
register_agent(
    agent_type="deep_researcher",
    class_path="aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent",
    config_name="deep_research_agent",
    description="Performs comprehensive multi-loop deep research",
)

register_agent(
    agent_type="shallow_researcher",
    class_path="aiq_agent.agents.shallow_researcher.agent.ShallowResearcherAgent",
    config_name="shallow_research_agent",
    description="Performs quick single-turn research",
)

register_agent(
    agent_type="report_rewriter",
    class_path="aiq_agent.agents.report_rewriter.agent.ReportRewriterAgent",
    config_name="deep_research_agent",
    description="Internal child agent for report edit follow-up",
    public=False,
)

register_agent(
    agent_type="data_science",
    class_path="aiq_agent.agents.data_science.agent.DataScienceAgent",
    config_name="data_science_agent",
    description="Analyzes enterprise structured data, documents, and web evidence in one autonomous loop",
)
