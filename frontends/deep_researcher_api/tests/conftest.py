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

"""Register a lightweight ``deep_researcher_api`` package before submodule imports.

Skips ``deep_researcher_api/__init__.py`` (plugin pulls NAT/Dask) so tests can load
``deep_researcher_api.auth`` and peers from ``src/`` without the full runtime stack.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_PKG_DIR = _SRC_ROOT / "deep_researcher_api"

if _PKG_DIR.is_dir():
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

    if "deep_researcher_api" not in sys.modules:
        _pkg = types.ModuleType("deep_researcher_api")
        _pkg.__path__ = [str(_PKG_DIR)]
        sys.modules["deep_researcher_api"] = _pkg
