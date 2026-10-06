# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opaque log correlation references."""

from deep_researcher_agent.common.logging_utils import log_content_metadata
from deep_researcher_agent.common.logging_utils import log_identifier_ref


def test_log_identifier_ref_is_stable_without_exposing_identifier() -> None:
    identifier = "8d312ad2-d097-42b8-93f1-6df4c084d6d4"

    first = log_identifier_ref(identifier)
    second = log_identifier_ref(identifier)

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 12
    assert identifier not in first
    assert identifier[:8] not in first
    assert log_identifier_ref("different") != first


def test_log_content_metadata_preserves_shape_without_exposing_content() -> None:
    content = "my fake secret is nvapi-vdr-do-not-log"

    metadata = log_content_metadata(content)

    assert metadata == f"chars={len(content)} ref={log_identifier_ref(content)}"
    assert content not in metadata
    assert "nvapi-vdr-do-not-log" not in metadata
