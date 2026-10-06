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

"""Deep Researcher Agent's NeMo Relay integration boundary.

Relay is initialized for every Deep Researcher Agent process.  Agent-specific integrations live
behind this package so observability wiring does not leak into research logic.
"""

from .config import RelayConfig
from .logging import register_logging_subscriber
from .runtime import agent_scope
from .runtime import ainvoke_tool_with_relay
from .runtime import ainvoke_with_relay
from .runtime import deepagents_kwargs
from .runtime import merge_langchain_middleware
from .runtime import run_agent
from .runtime import run_workflow
from .runtime import workflow_scope

__all__ = [
    "agent_scope",
    "ainvoke_tool_with_relay",
    "ainvoke_with_relay",
    "deepagents_kwargs",
    "merge_langchain_middleware",
    "register_logging_subscriber",
    "RelayConfig",
    "run_agent",
    "run_workflow",
    "workflow_scope",
]
