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

"""Auth components for the Deep Researcher Agent API frontend."""

from .base import TokenValidator
from .errors import AuthError
from .errors import TokenExpiredError
from .errors import TokenInvalidError
from .jwt_validator import JWTValidator
from .middleware import AuthMiddleware
from .middleware import get_current_trace_tags
from .middleware import get_current_user

__all__ = [
    "AuthError",
    "AuthMiddleware",
    "JWTValidator",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenValidator",
    "get_current_trace_tags",
    "get_current_user",
]
