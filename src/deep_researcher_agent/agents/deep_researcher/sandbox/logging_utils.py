# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret-safe logging helpers for sandbox and artifact failure boundaries."""

from __future__ import annotations

import logging


def log_sandbox_failure(
    log: logging.Logger,
    *,
    operation: str,
    reason_code: str,
    exc: BaseException,
    provider: str | None = None,
    sandbox: str | None = None,
) -> None:
    """Log only allowlisted failure metadata, never exception text or traceback."""
    log.warning(
        "Sandbox failure: operation=%s reason=%s provider=%s sandbox=%s exception=%s",
        operation,
        reason_code,
        provider,
        sandbox,
        type(exc).__name__,
    )
