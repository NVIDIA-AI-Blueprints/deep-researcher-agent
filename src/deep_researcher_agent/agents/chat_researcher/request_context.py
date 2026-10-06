# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated request context shared by the chat researcher workflow."""

from typing import Annotated

from pydantic import BaseModel
from pydantic import StringConstraints

DatabaseName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
    ),
]


class ChatRequestContext(BaseModel):
    """Normalized chat request context extracted from NAT/OpenAI-style payloads."""

    query_text: str
    data_sources: list[str] | None = None
    active_report_job_id: str | None = None
    database_name: DatabaseName | None = None
