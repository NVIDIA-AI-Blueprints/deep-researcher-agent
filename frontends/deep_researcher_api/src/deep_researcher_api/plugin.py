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

"""
NAT plugin registration for unified Deep Researcher Agent API.

Combines Knowledge API (collections/documents) and Async Job API (agent jobs/SSE streaming).

Knowledge Layer Configuration:
    The Knowledge API uses the same ingestor instance as the knowledge_retrieval tool.
    Configure the backend via the knowledge_retrieval function in your workflow YAML:

    functions:
      knowledge_search:
        _type: knowledge_retrieval
        backend: foundational_rag
        rag_url: http://localhost:8081/v1
        ingest_url: http://localhost:8082/v1

    The API plugin will automatically use the configured backend. If no tool is
    configured, it falls back to environment variables (KNOWLEDGE_INGESTOR_BACKEND)
    or the default backend (llamaindex).
"""

import logging
import os
import signal
from collections.abc import Callable

from fastapi import APIRouter
from fastapi import FastAPI
from pydantic import Field

from deep_researcher_api.auth.middleware import AuthMiddleware
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_front_end
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase
from nat.utils.type_utils import override

from .jobs.connection_manager import get_connection_manager
from .jobs.event_store import EventStore
from .routes.collections import add_collection_routes
from .routes.documents import add_document_routes
from .routes.jobs import register_job_routes
from .websocket_reconnect import configure_websocket_auth
from .websocket_reconnect import install_reconnectable_handler

logger = logging.getLogger(__name__)

install_reconnectable_handler()

_DASK_LOOPBACK_DASHBOARD_ADDRESS = "127.0.0.1:8787"

_validators: list = []


def register_validator(validator) -> None:
    """Register a token validator with the API server.

    Call this before the server starts.  Validators are tried in order;
    the first successful result wins.  At least one validator must be
    registered when ``REQUIRE_AUTH=true``.

    Example::

        from deep_researcher_api.plugin import register_validator
        from mypackage.auth import MyCustomValidator
        register_validator(MyCustomValidator(...))
    """
    _validators.append(validator)


def _load_validators_from_entry_points() -> list:
    """Discover validators registered via the ``deep_researcher_api.validators`` entry-point group.

    Any installed package can contribute validators by declaring an entry point
    that returns a list of validator instances::

        # pyproject.toml (in the internal / deployment package)
        [project.entry-points."deep_researcher_api.validators"]
        my_provider = "mypackage.auth:get_validators"

        # mypackage/auth.py
        def get_validators() -> list:
            return [MyValidator(...)]

    This is the recommended way to add validators from a private package that
    has the public deep-researcher-api repo as a git submodule.
    """
    from importlib.metadata import entry_points

    validators = []
    for ep in entry_points(group="deep_researcher_api.validators"):
        try:
            factory = ep.load()
            result = factory()
            validators.extend(result if isinstance(result, list) else [result])
            logger.info(
                "Loaded validators from entry point '%s': %s",
                ep.name,
                [type(v).__name__ for v in (result if isinstance(result, list) else [result])],
            )
        except Exception as e:
            logger.warning("Failed to load validators from entry point '%s': %s", ep.name, e)
    return validators


class DeepResearcherAPIConfig(FastApiFrontEndConfig, name="deep_researcher_api"):
    """
    Configuration for unified Deep Researcher Agent API endpoints.

    Knowledge API:
        Automatically enabled when a knowledge_retrieval function is configured.
        Backend settings are inherited from that function's config.

    Async Job API:
        Configure db_url and expiry_seconds for job persistence.
    """

    db_url: str = Field(
        default="sqlite+aiosqlite:///./jobs.db",
        description="Database URL for job store and event store",
    )
    expiry_seconds: int = Field(
        default=86400,
        ge=600,
        le=604800,
        description="Job expiry time in seconds (default: 24 hours)",
    )


# Track if shutdown signal has been received (for force exit on second Ctrl+C)
_shutdown_signal_received = False


def _create_shutdown_signal_handler(
    original_handler: Callable | signal.Handlers | None,
    sig: signal.Signals,
) -> Callable:
    """
    Create a signal handler that signals SSE shutdown before calling the original handler.

    This ensures SSE connections are notified of shutdown before uvicorn cancels tasks.
    On second signal, force exits immediately.
    """

    def handler(signum, frame):
        global _shutdown_signal_received

        if _shutdown_signal_received:
            logger.warning("Second %s received, forcing exit...", sig.name)
            os._exit(1)

        _shutdown_signal_received = True
        logger.info("Signal %s received, signaling SSE shutdown... (press again to force quit)", sig.name)
        connection_manager = get_connection_manager()

        connection_manager.signal_shutdown()

        if original_handler and callable(original_handler):
            original_handler(signum, frame)
        elif original_handler == signal.SIG_DFL:
            signal.signal(sig, signal.SIG_DFL)
            signal.raise_signal(sig)

    return handler


class DeepResearcherAPIWorker(FastApiFrontEndPluginWorker):
    """
    Worker that adds unified Deep Researcher Agent API routes to the FastAPI app.

    Combines:
    - Knowledge API routes (collections, documents) - uses factory singleton
    - Async Job API routes (agent jobs, SSE streaming)
    """

    _original_sigint_handler: Callable | signal.Handlers | None = None
    _original_sigterm_handler: Callable | signal.Handlers | None = None

    @override
    def build_app(self) -> FastAPI:
        app = super().build_app()

        app.title = "Deep Researcher Agent API"
        app.description = "Async research jobs, knowledge management, and agent orchestration."
        app.version = "1.0.0"

        knowledge_router = APIRouter()
        add_collection_routes(knowledge_router)
        add_document_routes(knowledge_router)
        app.include_router(knowledge_router)
        logger.info("Knowledge API routes registered")

        require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
        validators = _validators + _load_validators_from_entry_points()
        if require_auth and not validators:
            raise RuntimeError(
                "REQUIRE_AUTH=true but no validators have been registered. "
                "Either call deep_researcher_api.plugin.register_validator() before starting the server, "
                "or declare an 'deep_researcher_api.validators' entry point in your package."
            )
        app.add_middleware(AuthMiddleware, validators=validators, require_auth=require_auth)
        configure_websocket_auth(validators=validators, require_auth=require_auth)
        logger.info(
            "AuthMiddleware registered (require_auth=%s, validators=%s)",
            require_auth,
            [type(v).__name__ for v in validators],
        )

        return app

    @override
    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):
        # NAT registers direct async-generation routes when Dask is available.
        # Those routes submit straight to NAT's JobStore and therefore bypass
        # Deep Researcher Agent's admission and job-ownership controls.  Hide Dask availability
        # only while NAT registers its routes, then restore it before adding the
        # guarded Deep Researcher Agent async job API below.
        dask_available = self._dask_available
        self._dask_available = False
        try:
            await super().add_routes(app, builder)
        finally:
            self._dask_available = dask_available

        # =====================================================================
        # Async Job API routes
        # =====================================================================
        await register_job_routes(app, builder, self)
        logger.info("Async Job API routes registered")

        self._install_signal_handlers()

        @app.on_event("shutdown")
        async def shutdown_sse_connections():
            """Gracefully close all active SSE connections and background tasks on shutdown."""
            logger.info("Shutting down SSE connections...")
            connection_manager = get_connection_manager()
            await connection_manager.shutdown(timeout=5.0)

            from .routes.jobs import stop_periodic_cleanup

            await stop_periodic_cleanup()

            await EventStore.dispose_all_engines_async()
            logger.info("SSE shutdown complete")

            self._restore_signal_handlers()

        enable_debug = os.environ.get("DEEP_RESEARCHER_ENABLE_DEBUG", "true").lower() not in {"0", "false", "no", "off"}
        if enable_debug:
            try:
                from deep_researcher_debug import register_debug_routes

                await register_debug_routes(app)
                logger.info("Debug console registered at /debug")
            except ImportError:
                pass
        else:
            logger.info("Debug console disabled by DEEP_RESEARCHER_ENABLE_DEBUG")

    def _install_signal_handlers(self):
        """Install signal handlers to notify SSE connections on shutdown."""
        try:
            self._original_sigint_handler = signal.getsignal(signal.SIGINT)
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)

            signal.signal(
                signal.SIGINT,
                _create_shutdown_signal_handler(self._original_sigint_handler, signal.SIGINT),
            )
            signal.signal(
                signal.SIGTERM,
                _create_shutdown_signal_handler(self._original_sigterm_handler, signal.SIGTERM),
            )
            logger.debug("Installed SSE shutdown signal handlers")
        except Exception as e:
            logger.warning("Failed to install signal handlers: %s", e)

    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        try:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            logger.debug("Restored original signal handlers")
        except Exception as e:
            logger.warning("Failed to restore signal handlers: %s", e)


class DeepResearcherAPIPlugin(FastApiFrontEndPlugin):
    """Plugin that adds unified Deep Researcher Agent API endpoints to the FastAPI server."""

    def __init__(self, full_config: Config, config: DeepResearcherAPIConfig):
        super().__init__(full_config=full_config)
        self.config = config

    @override
    async def run(self) -> None:
        scheduler_address = self.front_end_config.scheduler_address or os.environ.get("NAT_DASK_SCHEDULER_ADDRESS")
        if scheduler_address is not None:
            self.front_end_config.scheduler_address = scheduler_address
            await super().run()
            return

        # NAT 1.8 creates its LocalCluster directly and does not expose a
        # dashboard-address hook.  Its pinned Dask default is ``:8787``, which
        # binds the dashboard on every interface even though cluster RPC is
        # loopback-only.  Inject the safer constructor default for NAT's one
        # local-cluster creation without copying NAT's run implementation.
        #
        # Restore the module symbol inside the constructor wrapper, before
        # LocalCluster starts, so the compatibility shim cannot affect clusters
        # created later in the process.
        try:
            import dask.distributed as dask_distributed
        except ImportError:
            # Preserve NAT's optional-Dask behavior.
            await super().run()
            return

        local_cluster_factory = dask_distributed.LocalCluster

        def local_cluster_with_loopback_dashboard(*args, **kwargs):
            if dask_distributed.LocalCluster is local_cluster_with_loopback_dashboard:
                dask_distributed.LocalCluster = local_cluster_factory
            kwargs.setdefault("dashboard_address", _DASK_LOOPBACK_DASHBOARD_ADDRESS)
            return local_cluster_factory(*args, **kwargs)

        dask_distributed.LocalCluster = local_cluster_with_loopback_dashboard
        try:
            await super().run()
        finally:
            if dask_distributed.LocalCluster is local_cluster_with_loopback_dashboard:
                dask_distributed.LocalCluster = local_cluster_factory

    @override
    def get_worker_class(self) -> type[FastApiFrontEndPluginWorkerBase]:
        return DeepResearcherAPIWorker


@register_front_end(config_type=DeepResearcherAPIConfig)
async def register_deep_researcher_api(config: DeepResearcherAPIConfig, full_config: Config):
    """Register unified Deep Researcher Agent API with NAT framework."""
    yield DeepResearcherAPIPlugin(full_config=full_config, config=config)
