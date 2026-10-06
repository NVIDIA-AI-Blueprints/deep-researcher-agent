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

"""Intent classification result model."""

from typing import Any
from typing import Literal

from pydantic import BaseModel

RouteTarget = Literal["meta", "report", "new_research", "hybrid_research"]
ReportAction = Literal["ask", "edit"]


class IntentResult(BaseModel):
    """
    Result of intent classification.

    Attributes:
        intent: Classified intent - either 'meta' (greetings, chit-chat, capabilities)
                or 'research' (queries requiring data lookup and sources).
        target: Python-derived workflow target for the selected semantic route.
        report_action: Python-derived report interaction action, when target is ``report``.
        use_parent_report_context: Whether a deep-research route should seed the active report.
        raw: Optional raw classification response from the LLM.
    """

    intent: Literal["meta", "research"]
    target: RouteTarget = "new_research"
    report_action: ReportAction | None = None
    use_parent_report_context: bool = False
    raw: dict[str, Any] | None = None
