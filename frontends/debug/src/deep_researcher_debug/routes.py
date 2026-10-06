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

"""Debug routes for development and testing.

Serves the debug console at /debug.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"


async def register_debug_routes(app: FastAPI) -> None:
    """Register debug routes for development."""
    console_path = STATIC_DIR / "deep_research_console.html"

    if not console_path.exists():
        return

    @app.get("/debug", include_in_schema=False)
    async def debug_console() -> FileResponse:
        """Serve the debug console."""
        return FileResponse(console_path, media_type="text/html")
