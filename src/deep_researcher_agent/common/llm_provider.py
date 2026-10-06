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

"""LLM Provider for role-based LLM access and A/B testing."""

from enum import StrEnum

from langchain_core.language_models import BaseChatModel


class LLMRole(StrEnum):
    """
    Semantic roles for LLMs in the research workflow.

    Allows mapping different LLM configurations to different roles
    for A/B testing and cost optimization.
    """

    ROUTER = "router"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    EVIDENCE_JUDGE = "evidence_judge"
    GRADER = "grader"
    SUMMARIZER = "summarizer"
    ORCHESTRATOR = "orchestrator"
    REFLECTION = "reflection"
    CLARIFIER = "clarifier"
    META_CHATTER = "meta_chatter"
    REPORT_WRITER = "report_writer"


class LLMProvider:
    """
    Role-based LLM provider for A/B testing different models per role.

    Allows configuring different LLMs for different semantic roles in
    the research workflow. Falls back to a default LLM if no specific
    LLM is configured for a role.

    Example:
        >>> provider = LLMProvider()
        >>> provider.set_default(nim_llm)
        >>> provider.configure(LLMRole.REPORT_WRITER, qwen_llm)
        >>>
        >>> # Uses qwen_llm
        >>> writer_llm = provider.get(LLMRole.REPORT_WRITER)
        >>>
        >>> # Falls back to nim_llm
        >>> router_llm = provider.get(LLMRole.ROUTER)
    """

    def __init__(self) -> None:
        self._llms: dict[LLMRole, BaseChatModel] = {}
        self._default: BaseChatModel | None = None

    def set_default(self, llm: BaseChatModel) -> None:
        """
        Set the default LLM for roles that don't have a specific configuration.

        Args:
            llm: The LangChain chat model to use as default.
        """
        self._default = llm

    def configure(self, role: LLMRole, llm: BaseChatModel) -> None:
        """
        Configure a specific LLM for a role.

        Args:
            role: The semantic role to configure.
            llm: The LangChain chat model to use for this role.
        """
        self._llms[role] = llm

    def get(self, role: LLMRole) -> BaseChatModel:
        """
        Get the LLM configured for a role.

        Args:
            role: The semantic role to get the LLM for.

        Returns:
            The LangChain chat model for this role.

        Raises:
            ValueError: If no LLM is configured for the role and no default is set.
        """
        if role in self._llms:
            return self._llms[role]
        if self._default is not None:
            return self._default
        raise ValueError(
            f"No LLM configured for role '{role}' and no default LLM set. Call set_default() or configure() first."
        )

    def has_role(self, role: LLMRole) -> bool:
        """Check if a specific LLM is configured for a role."""
        return role in self._llms

    def configured_roles(self) -> list[LLMRole]:
        """Get list of roles that have specific LLM configurations."""
        return list(self._llms.keys())
