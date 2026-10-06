# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Registration for deep-agent Guardrails middleware."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from deep_researcher_agent.guardrails.deep_agent.config import DeepAgentGuardrailsConfig
from deep_researcher_agent.guardrails.deep_agent.middleware import _DeepAgentGuardrails
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware


@register_middleware(config_type=DeepAgentGuardrailsConfig)
async def deep_agent_guardrails_middleware(
    config: DeepAgentGuardrailsConfig,
    builder: Builder,
) -> AsyncGenerator[_DeepAgentGuardrails, None]:
    """Build deep-agent Guardrails middleware from configuration."""
    middleware = _DeepAgentGuardrails(config=config, builder=builder)
    await middleware.bind_llms_to_rail()
    yield middleware
