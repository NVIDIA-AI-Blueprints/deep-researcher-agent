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

"""Tests for LLM Provider."""

from unittest.mock import MagicMock

import pytest

from deep_researcher_agent.common.llm_provider import LLMProvider
from deep_researcher_agent.common.llm_provider import LLMRole


class TestLLMRole:
    """Tests for the LLMRole enum."""

    def test_llm_role_values(self):
        """Test that LLMRole has expected values."""
        assert LLMRole.ROUTER == "router"
        assert LLMRole.PLANNER == "planner"
        assert LLMRole.RESEARCHER == "researcher"
        assert LLMRole.EVIDENCE_JUDGE == "evidence_judge"
        assert LLMRole.GRADER == "grader"
        assert LLMRole.SUMMARIZER == "summarizer"
        assert LLMRole.ORCHESTRATOR == "orchestrator"
        assert LLMRole.REFLECTION == "reflection"
        assert LLMRole.CLARIFIER == "clarifier"
        assert LLMRole.META_CHATTER == "meta_chatter"
        assert LLMRole.REPORT_WRITER == "report_writer"

    def test_llm_role_is_string_enum(self):
        """Test that LLMRole members are string-like."""
        assert isinstance(LLMRole.ROUTER, str)
        assert str(LLMRole.ROUTER) == "router"


class TestLLMProvider:
    """Tests for the LLMProvider class."""

    def test_init_empty_provider(self):
        """Test that a new provider has no configured LLMs."""
        provider = LLMProvider()
        assert provider.configured_roles() == []

    def test_set_default(self):
        """Test setting a default LLM."""
        provider = LLMProvider()
        mock_llm = MagicMock()
        provider.set_default(mock_llm)

        # Default should be used for any role
        assert provider.get(LLMRole.ROUTER) is mock_llm
        assert provider.get(LLMRole.PLANNER) is mock_llm

    def test_configure_specific_role(self):
        """Test configuring a specific LLM for a role."""
        provider = LLMProvider()
        default_llm = MagicMock(name="default")
        router_llm = MagicMock(name="router")

        provider.set_default(default_llm)
        provider.configure(LLMRole.ROUTER, router_llm)

        # Router should use specific LLM
        assert provider.get(LLMRole.ROUTER) is router_llm
        # Other roles should use default
        assert provider.get(LLMRole.PLANNER) is default_llm

    def test_get_without_default_raises_error(self):
        """Test that getting an unconfigured role without default raises ValueError."""
        provider = LLMProvider()

        with pytest.raises(ValueError) as exc_info:
            provider.get(LLMRole.ROUTER)

        assert "No LLM configured for role 'router'" in str(exc_info.value)
        assert "Call set_default() or configure() first" in str(exc_info.value)

    def test_get_with_only_role_configured(self):
        """Test getting a role that is configured without a default."""
        provider = LLMProvider()
        router_llm = MagicMock(name="router")
        provider.configure(LLMRole.ROUTER, router_llm)

        # Configured role should work
        assert provider.get(LLMRole.ROUTER) is router_llm

        # Unconfigured role should raise error
        with pytest.raises(ValueError):
            provider.get(LLMRole.PLANNER)

    def test_has_role_true(self):
        """Test has_role returns True for configured roles."""
        provider = LLMProvider()
        mock_llm = MagicMock()
        provider.configure(LLMRole.RESEARCHER, mock_llm)

        assert provider.has_role(LLMRole.RESEARCHER) is True

    def test_has_role_false(self):
        """Test has_role returns False for unconfigured roles."""
        provider = LLMProvider()
        assert provider.has_role(LLMRole.RESEARCHER) is False

    def test_has_role_false_even_with_default(self):
        """Test has_role returns False even when default is set."""
        provider = LLMProvider()
        provider.set_default(MagicMock())

        # has_role checks for specific configuration, not default fallback
        assert provider.has_role(LLMRole.RESEARCHER) is False

    def test_configured_roles_returns_only_explicit(self):
        """Test configured_roles returns only explicitly configured roles."""
        provider = LLMProvider()
        provider.set_default(MagicMock())
        provider.configure(LLMRole.ROUTER, MagicMock())
        provider.configure(LLMRole.PLANNER, MagicMock())

        roles = provider.configured_roles()
        assert LLMRole.ROUTER in roles
        assert LLMRole.PLANNER in roles
        assert len(roles) == 2

    def test_configure_overwrites_previous(self):
        """Test that configuring a role twice overwrites the previous LLM."""
        provider = LLMProvider()
        llm1 = MagicMock(name="llm1")
        llm2 = MagicMock(name="llm2")

        provider.configure(LLMRole.ROUTER, llm1)
        assert provider.get(LLMRole.ROUTER) is llm1

        provider.configure(LLMRole.ROUTER, llm2)
        assert provider.get(LLMRole.ROUTER) is llm2

    def test_set_default_overwrites_previous(self):
        """Test that setting default twice overwrites the previous default."""
        provider = LLMProvider()
        default1 = MagicMock(name="default1")
        default2 = MagicMock(name="default2")

        provider.set_default(default1)
        assert provider.get(LLMRole.ROUTER) is default1

        provider.set_default(default2)
        assert provider.get(LLMRole.ROUTER) is default2

    def test_multiple_roles_configured(self):
        """Test configuring multiple roles independently."""
        provider = LLMProvider()
        default_llm = MagicMock(name="default")
        router_llm = MagicMock(name="router")
        planner_llm = MagicMock(name="planner")
        researcher_llm = MagicMock(name="researcher")

        provider.set_default(default_llm)
        provider.configure(LLMRole.ROUTER, router_llm)
        provider.configure(LLMRole.PLANNER, planner_llm)
        provider.configure(LLMRole.RESEARCHER, researcher_llm)

        assert provider.get(LLMRole.ROUTER) is router_llm
        assert provider.get(LLMRole.PLANNER) is planner_llm
        assert provider.get(LLMRole.RESEARCHER) is researcher_llm
        assert provider.get(LLMRole.GRADER) is default_llm  # Falls back to default
        assert provider.get(LLMRole.SUMMARIZER) is default_llm  # Falls back to default
