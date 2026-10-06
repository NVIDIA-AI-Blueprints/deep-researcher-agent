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

"""Built-in sandbox providers.

Importing this package registers the built-in providers with the registry. Each
provider self-registers at import; adding a new provider is one new module here.
"""

from __future__ import annotations

from .modal import ModalSandboxProvider
from .openshell import OpenShellSandboxProvider

__all__ = [
    "ModalSandboxProvider",
    "OpenShellSandboxProvider",
]
