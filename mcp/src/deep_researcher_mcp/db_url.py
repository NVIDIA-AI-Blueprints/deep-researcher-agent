# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database URL helpers for the Deep Researcher Agent MCP runtime."""

from __future__ import annotations

import re
from urllib.parse import unquote
from urllib.parse import urlsplit


def normalize_postgres_url(value: str, *, label: str) -> str:
    """Return a Postgres URL normalized for drivers that expect bare schemes."""
    normalized = value.strip()
    normalized = re.sub(r"^postgresql\+[^:]+://", "postgresql://", normalized)
    normalized = re.sub(r"^postgres\+[^:]+://", "postgres://", normalized)
    if normalized.startswith("sqlite"):
        raise ValueError(f"{label} must be a Postgres DSN; SQLite is not supported alongside the MCP server")
    if not normalized.startswith(("postgresql://", "postgres://")):
        raise ValueError(f"{label} must be a Postgres DSN that starts with postgresql:// or postgres://")
    return normalized


def require_test_database_url(value: str, *, label: str) -> str:
    """Return a normalized Postgres test URL or fail before destructive test setup."""
    normalized = normalize_postgres_url(value, label=label)
    database_path = urlsplit(normalized).path
    if not re.fullmatch(r"/[^/]+", database_path):
        raise ValueError(f"{label} must target a test database whose name ends with _test or _tests")

    database_name = unquote(database_path[1:])
    if "/" in database_name or not database_name.casefold().endswith(("_test", "_tests")):
        raise ValueError(f"{label} must target a test database whose name ends with _test or _tests")
    return normalized
