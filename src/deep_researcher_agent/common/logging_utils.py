# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for logging stable references without exposing sensitive content."""

import hashlib


def log_identifier_ref(identifier: str) -> str:
    """Return a stable correlation reference that does not reveal ``identifier``."""
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def log_content_metadata(content: object) -> str:
    """Return safe, correlatable metadata for content that must not be logged.

    Prompts, model responses, tool payloads, and exception details can contain
    credentials or private customer data. Logging their length and a stable
    digest preserves enough signal to correlate retries without writing the
    content itself to production logs.
    """
    text = content if isinstance(content, str) else str(content)
    return f"chars={len(text)} ref={log_identifier_ref(text)}"
