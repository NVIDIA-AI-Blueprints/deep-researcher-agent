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

"""Per-task auth token context for async jobs.

Uses a ContextVar so concurrent jobs in the same Dask worker process
each see their own token without cross-task leakage.

The token fetcher is registered once at import time. It returns the
token from the current task's context, or None if no token is set.
"""

from contextvars import ContextVar

from deep_researcher_agent.auth import register_token_fetcher

job_auth_token: ContextVar[str | None] = ContextVar("job_auth_token", default=None)


def _job_token_fetcher() -> str | None:
    """Return the auth token for the current async task, if any."""
    return job_auth_token.get()


# Register once — the ContextVar ensures per-task isolation
register_token_fetcher(_job_token_fetcher, priority=5)
