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

"""Artifact manifest parsing.

Generated code may write a ``manifest.json`` into the sandbox artifact directory to
declare its outputs explicitly (reliable). Directory scanning for changed allowed
files is the safety-net fallback for agents that forget to write a manifest.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from .models import ArtifactKind

logger = logging.getLogger(__name__)


class ManifestEntry(BaseModel):
    """A single declared artifact in a manifest."""

    path: str = Field(..., description="Absolute path inside the sandbox")
    kind: ArtifactKind = Field(default=ArtifactKind.OTHER)
    title: str | None = Field(default=None)
    caption: str | None = Field(default=None)
    inline: bool = Field(default=False)
    source_files: tuple[str, ...] = Field(default=(), description="Inputs used to produce this artifact")


class Manifest(BaseModel):
    """Top-level manifest schema written by generated code."""

    version: int = Field(default=1)
    artifacts: tuple[ManifestEntry, ...] = Field(default=())


def parse_manifest(raw: str) -> Manifest | None:
    """Parse a manifest JSON string into a :class:`Manifest`.

    Args:
        raw: The manifest file contents.

    Returns:
        A parsed ``Manifest``, or ``None`` if the content is invalid (a warning is
        logged; callers fall back to directory scanning).
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Artifact manifest is not valid JSON; falling back to scan")
        return None
    try:
        return Manifest.model_validate(data)
    except ValidationError:
        logger.warning("Artifact manifest failed schema validation; falling back to scan")
        return None
