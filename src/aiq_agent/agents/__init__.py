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

"""Agents for the AI-Q Blueprint."""

from .chat_researcher import chat_deepresearcher_agent
from .data_science import data_science_agent
from .deep_researcher import deep_research_agent
from .shallow_researcher import shallow_research_agent

__all__ = [
    "chat_deepresearcher_agent",
    "data_science_agent",
    "shallow_research_agent",
    "deep_research_agent",
]
