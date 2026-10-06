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

"""Registration for Guardrails middleware."""

from __future__ import annotations

from deep_researcher_agent.guardrails.deep_agent.register import deep_agent_guardrails_middleware
from deep_researcher_agent.guardrails.shallow_agent.register import shallow_agent_guardrails_middleware
from deep_researcher_agent.guardrails.workflow.register import workflow_guardrails_middleware

__all__ = [
    "deep_agent_guardrails_middleware",
    "shallow_agent_guardrails_middleware",
    "workflow_guardrails_middleware",
]
