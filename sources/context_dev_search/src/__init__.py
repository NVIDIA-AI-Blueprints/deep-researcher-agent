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

"""Context.dev web search and research paper search tools for NAT.

The NAT functions are defined in ``register`` (the plugin entry point) and
loaded on first access, so ``context_dev_search.client`` imports without NAT.
"""

__all__ = [
    "context_dev_paper_search",
    "context_dev_web_search",
]


def __getattr__(name: str):
    if name in __all__:
        from . import register

        return getattr(register, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
