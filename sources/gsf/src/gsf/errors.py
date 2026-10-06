# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable, non-secret errors returned by the GSF integration."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict


class GSFErrorCode(StrEnum):
    """Error codes exposed by GSF-backed Deep Researcher Agent tools."""

    AUTHENTICATION_REQUIRED = "authentication_required"
    FORBIDDEN = "forbidden"
    INVALID_REQUEST = "invalid_request"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID_RESPONSE = "invalid_response"
    RESPONSE_TOO_LARGE = "response_too_large"
    UPSTREAM_ERROR = "upstream_error"


class GSFError(Exception):
    """Internal exception containing only caller-safe GSF failure details."""

    def __init__(
        self,
        code: GSFErrorCode,
        message: str,
        *,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        """Initialize a normalized failure without retaining response data."""

        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.request_id = request_id


class GSFToolError(BaseModel):
    """Serialized error envelope returned to an Deep Researcher Agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    code: GSFErrorCode
    retryable: bool
    request_id: str | None = None
    message: str

    @classmethod
    def from_exception(cls, error: GSFError) -> "GSFToolError":
        """Convert an internal exception to the serialized tool error shape."""

        return cls(
            code=error.code,
            retryable=error.retryable,
            request_id=error.request_id,
            message=error.message,
        )
