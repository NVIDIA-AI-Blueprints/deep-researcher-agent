# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT workflow loader + invoker used by the FastMCP tool handlers.

Owns the long-lived `load_workflow(...)` context manager so the compiled
LangGraph is built once at server startup and reused across tool calls. Exposes
two operations:

- :meth:`classify` invokes the ``intent_classifier`` NAT function directly to
  decide whether a query is shallow or deep (and surface any meta response).
  Used synchronously in ``submit_query`` to return a depth hint to the caller.
- :meth:`run_query` runs the full ``chat_deepresearcher_agent`` workflow to
  produce the final research answer. Called from the background task that the
  JobManager launches.
"""

import inspect
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherResponse
from deep_researcher_agent.agents.chat_researcher.models import WorkflowOutcome
from deep_researcher_agent.agents.chat_researcher.preclassification import preclassified_depth
from deep_researcher_agent.common.logging_utils import log_identifier_ref
from nat.builder.context import Context
from nat.runtime.loader import load_workflow
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Lifespan-scoped wrapper around a NAT workflow loaded from YAML."""

    def __init__(self, config_file: str | Path):
        self._config_file = Path(config_file)
        self._exit_stack: AsyncExitStack | None = None
        self._session_manager: SessionManager | None = None
        self._owned_checkpointer_keys: set[str] = set()

    async def start(self) -> None:
        if self._session_manager is not None:
            return
        logger.info("Loading NAT workflow from %s", self._config_file)
        checkpointers_before = _current_checkpointer_keys()
        self._exit_stack = AsyncExitStack()
        self._session_manager = await self._exit_stack.enter_async_context(load_workflow(str(self._config_file)))
        self._owned_checkpointer_keys = _current_checkpointer_keys() - checkpointers_before
        logger.info("NAT workflow ready")

    async def stop(self) -> None:
        if self._exit_stack is None:
            return
        logger.info("Shutting down NAT workflow")
        try:
            await self._exit_stack.aclose()
        finally:
            await self._close_owned_checkpointers()
            self._exit_stack = None
            self._session_manager = None

    async def _close_owned_checkpointers(self) -> None:
        # Resolve the caches before the empty-keys guard: if deep_researcher_agent renamed them,
        # start() recorded no owned keys (the snapshot read the old name), so an
        # early return here would hide the very failure we want to surface.
        caches = _resolve_checkpointer_caches()
        if caches is None:
            # Already warned. Drop ownership so a second stop() does not warn again.
            self._owned_checkpointer_keys.clear()
            return
        if not self._owned_checkpointer_keys:
            return

        checkpointers, postgres_pools = caches
        for key in list(self._owned_checkpointer_keys):
            checkpointer = checkpointers.pop(key, None)
            conn = getattr(checkpointer, "conn", None)
            if conn is not None:
                close = getattr(conn, "close", None)
                if close is not None:
                    await _maybe_await(close())
                    logger.debug("Closed SQLite checkpointer connection: %s", key)

            pool = postgres_pools.pop(key, None)
            if pool is not None:
                await _maybe_await(pool.close())
                logger.debug("Closed Postgres checkpointer pool: %s", key)

        self._owned_checkpointer_keys.clear()

    async def classify(self, query: str) -> dict[str, Any]:
        """Run the intent classifier node as a standalone call.

        Returns a dict containing ``user_intent`` and ``depth_decision`` keys,
        and possibly ``messages`` when the intent is meta.
        """
        if self._session_manager is None:
            raise RuntimeError("WorkflowRunner.start() must be called before classify()")

        from langchain_core.messages import HumanMessage

        from deep_researcher_agent.agents.chat_researcher.models import ChatResearcherState

        intent_fn = await self._session_manager.shared_builder.get_function("intent_classifier")
        state = ChatResearcherState(messages=[HumanMessage(content=query)])
        return await intent_fn.ainvoke(state)

    async def run_query(
        self,
        query: str,
        *,
        conversation_id: str,
        depth: str | None = None,
    ) -> WorkflowOutcome:
        """Run the full workflow and return its explicit terminal outcome.

        ``conversation_id`` is also LangGraph's checkpoint ``thread_id``; MCP
        async jobs pass their ``job_id`` so NAT resume/checkpoint state lines up
        with the MCP job ledger. Swallowed agent exceptions arrive as a typed
        failure instead of fallback text that could be mistaken for success.

        ``depth`` reuses a classification the caller already made (the JobManager
        classifies once in ``submit()`` to persist depth and pick a poll cadence).
        When set to ``shallow``/``deep`` the ``intent_classifier`` node skips its
        redundant LLM call and routes by this decision, so the persisted depth and
        the executed route stay identical. ``None`` leaves the graph to classify.
        """
        if self._session_manager is None:
            raise RuntimeError("WorkflowRunner.start() must be called before run_query()")

        logger.info("Running NAT workflow: conversation_ref=%s", log_identifier_ref(conversation_id))

        with Context.scope(conversation_id=conversation_id):
            with preclassified_depth(depth):
                async with self._session_manager.session(conversation_id=conversation_id) as session:
                    async with session.run(query) as runner:
                        response = await runner.result(to_type=ChatResearcherResponse)

        return response.workflow_outcome


def _resolve_checkpointer_caches() -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return deep_researcher_agent.common's ``(_checkpointers, _postgres_pools)`` caches, or ``None``.

    MCP shutdown reaches into these module-private dicts to close the checkpointer
    handles this runner created; there is no public cleanup API to use instead. If
    deep_researcher_agent renames, removes, or restructures either cache, cleanup would silently
    become a no-op and leak connections/pools — so warn and return ``None`` rather than
    letting a ``getattr(..., {})`` default hide the breakage.
    """
    try:
        from deep_researcher_agent import common as deep_researcher_common
    except Exception as exc:  # pragma: no cover - defensive shutdown path
        logger.warning(
            "deep_researcher_agent.common is unavailable; "
            "MCP checkpointer cleanup skipped and owned handles may leak: %s",
            exc,
        )
        return None

    checkpointers = getattr(deep_researcher_common, "_checkpointers", None)
    postgres_pools = getattr(deep_researcher_common, "_postgres_pools", None)
    missing = [
        name
        for name, cache in (("_checkpointers", checkpointers), ("_postgres_pools", postgres_pools))
        if not isinstance(cache, dict)
    ]
    if missing:
        logger.warning(
            "deep_researcher_agent.common is missing expected checkpointer cache(s) %s; MCP checkpointer "
            "cleanup is a no-op and owned handles may leak (deep_researcher_agent internals may have changed).",
            missing,
        )
        return None
    return checkpointers, postgres_pools


def _current_checkpointer_keys() -> set[str]:
    try:
        from deep_researcher_agent import common as deep_researcher_common
    except Exception:
        return set()
    return set(getattr(deep_researcher_common, "_checkpointers", {}).keys())


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value
