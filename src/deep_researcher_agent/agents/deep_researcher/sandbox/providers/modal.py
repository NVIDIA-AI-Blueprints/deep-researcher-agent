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

"""Modal sandbox provider (cloud example)."""

from __future__ import annotations

import logging
import re
import shlex
from typing import TYPE_CHECKING

from deepagents.backends.sandbox import BaseSandbox

from ..base import SandboxProvider
from ..capabilities import SandboxCapabilities
from ..registry import register_sandbox_provider

if TYPE_CHECKING:
    from ..config import SandboxConfig

logger = logging.getLogger(__name__)

_IMPORT_HINT = (
    "The Modal sandbox backend requires the `langchain-modal` and `modal` packages. "
    "Install the updated Deep Researcher Agent dependencies and run `modal setup` before enabling a Modal sandbox."
)


def _validate_modal_sandbox_name(job_id: str) -> str:
    """Validate that ``job_id`` is a legal Modal object name.

    Args:
        job_id: Candidate sandbox name.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is too long, has illegal characters, or matches a
            reserved Modal app-id shape.
    """
    if len(job_id) > 64 or re.match(r"^[a-zA-Z0-9-_.]+$", job_id) is None or re.match(r"^ap-[a-zA-Z0-9]{22}$", job_id):
        raise ValueError(
            "Deep research job_id must be a valid Modal sandbox name: 64 characters or fewer, using only "
            "alphanumeric characters, dashes, periods, and underscores."
        )
    return job_id


def _is_modal_not_found_error(exc: Exception) -> bool:
    """Return whether ``exc`` is Modal's typed NotFoundError (stale container)."""
    try:
        import modal

        return isinstance(exc, modal.exception.NotFoundError)
    except ImportError:
        return exc.__class__.__name__ == "NotFoundError" and exc.__class__.__module__.startswith("modal")


class ModalSandboxProvider(SandboxProvider):
    """Job-scoped Modal backend.

    Modal enforces network blocking via the ``block_network`` create flag and
    supports deterministic termination, so it declares those capabilities.
    """

    provider_name = "modal"

    def __init__(self, config: SandboxConfig, job_id: str) -> None:
        """Initialize the provider, requiring the Modal SDK and adapter to import."""
        super().__init__(config, job_id)
        try:
            import langchain_modal  # noqa: F401
            import modal  # noqa: F401
        except ImportError as exc:
            raise ImportError(_IMPORT_HINT) from exc

    @classmethod
    def _scoped_name(cls, job_id: str) -> str:
        """Return the validated, job-scoped Modal sandbox name."""
        return _validate_modal_sandbox_name(job_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the guarantees the Modal backend can enforce."""
        return SandboxCapabilities(
            supports_network_policy=True,
            supports_resource_limits=True,
            supports_artifact_download=True,
            supports_cleanup=True,
        )

    def is_recoverable_error(self, exc: Exception) -> bool:
        """Return whether the error is a missing-sandbox condition worth one retry."""
        return _is_modal_not_found_error(exc)

    def _create_session(self) -> BaseSandbox:
        """Create a fresh, job-scoped Modal sandbox.

        Create-first semantics: unlike the legacy backend, this does NOT attach to
        an existing sandbox by name as its primary path (which risked binding a new
        job to a prior job's workspace). It creates fresh; only an
        ``AlreadyExistsError`` (this job's own sandbox from earlier in the run, since
        the name is the unique job id) falls back to attach.
        """
        try:
            import modal
            from langchain_modal import ModalSandbox
        except ImportError as exc:
            raise ImportError(_IMPORT_HINT) from exc

        cfg = self.config
        modal_cfg = cfg.providers.modal
        app = modal.App.lookup(name=modal_cfg.app_name, create_if_missing=True)

        image = modal.Image.from_registry(modal_cfg.image)
        if modal_cfg.python_packages:
            image = image.pip_install(*modal_cfg.python_packages)
        if cfg.workdir:
            image = image.run_commands(f"mkdir -p {shlex.quote(cfg.workdir)}")

        # Opt-in resource caps (None => Modal default). The capability gate has already
        # refused limits on providers that cannot enforce them, so passing them here is safe.
        resource_kwargs: dict[str, object] = {}
        if cfg.resources.cpu is not None:
            resource_kwargs["cpu"] = cfg.resources.cpu
        if cfg.resources.memory_mb is not None:
            resource_kwargs["memory"] = cfg.resources.memory_mb

        try:
            sandbox = modal.Sandbox.create(
                app=app,
                image=image,
                workdir=cfg.workdir,
                name=self.sandbox_name,
                timeout=cfg.timeout,
                idle_timeout=cfg.idle_timeout,
                block_network=cfg.block_network,
                **resource_kwargs,
            )
            logger.info(
                "Modal sandbox CREATED: name=%s image=%s workdir=%s timeout=%ds",
                self.sandbox_name,
                modal_cfg.image,
                cfg.workdir,
                cfg.timeout,
            )
        except modal.exception.AlreadyExistsError:
            sandbox = modal.Sandbox.from_name(modal_cfg.app_name, self.sandbox_name)
            logger.info("Modal sandbox attached to this job's existing instance: name=%s", self.sandbox_name)
        return ModalSandbox(sandbox=sandbox)


register_sandbox_provider("modal", ModalSandboxProvider)
