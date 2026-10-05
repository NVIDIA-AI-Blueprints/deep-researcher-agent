# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database URL normalization tests."""

import pytest

from deep_researcher_mcp.db_url import normalize_postgres_url
from deep_researcher_mcp.db_url import require_test_database_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("postgresql://db.example/deep-researcher", "postgresql://db.example/deep-researcher"),
        ("postgres://db.example/deep-researcher", "postgres://db.example/deep-researcher"),
        ("postgresql+asyncpg://db.example/deep-researcher", "postgresql://db.example/deep-researcher"),
        ("postgresql+psycopg://db.example/deep-researcher", "postgresql://db.example/deep-researcher"),
        ("  postgres+asyncpg://db.example/deep-researcher  ", "postgres://db.example/deep-researcher"),
    ],
)
def test_normalize_postgres_url(value: str, expected: str) -> None:
    assert normalize_postgres_url(value, label="test URL") == expected


@pytest.mark.parametrize(
    "value", ["sqlite:///tmp/checkpoints.db", "https://db.example/deep-researcher", "db.example/deep-researcher", ""]
)
def test_normalize_postgres_url_rejects_non_postgres_values(value: str) -> None:
    with pytest.raises(ValueError, match="must be a Postgres DSN"):
        normalize_postgres_url(value, label="test URL")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("postgresql://db.example/deep_researcher_mcp_test", "postgresql://db.example/deep_researcher_mcp_test"),
        (
            "postgresql+asyncpg://db.example/DEEP_RESEARCHER_MCP_TEST",
            "postgresql://db.example/DEEP_RESEARCHER_MCP_TEST",
        ),
        ("postgresql://db.example/deep_researcher_mcp_tests", "postgresql://db.example/deep_researcher_mcp_tests"),
        (
            "postgresql+asyncpg://db.example/DEEP_RESEARCHER_MCP_TESTS",
            "postgresql://db.example/DEEP_RESEARCHER_MCP_TESTS",
        ),
    ],
)
def test_require_test_database_url_accepts_test_database_names(value: str, expected: str) -> None:
    assert require_test_database_url(value, label="test URL") == expected


@pytest.mark.parametrize(
    "value",
    [
        "postgresql://db.example/deep_researcher_mcp",
        "postgresql://db.example/postgres",
        "postgresql://db.example/deep_researcher_mcp_testing",
        "postgresql://db.example/deep-researcher%2Fmcp_test",
        "postgresql://db.example/deep_researcher_mcp_test2",
        "postgresql://db.example/",
        "postgresql://db.example/deep_researcher_mcp_test/extra",
    ],
)
def test_require_test_database_url_refuses_non_test_database_names(value: str) -> None:
    with pytest.raises(ValueError, match="must target a test database whose name ends with _test or _tests"):
        require_test_database_url(value, label="test URL")
