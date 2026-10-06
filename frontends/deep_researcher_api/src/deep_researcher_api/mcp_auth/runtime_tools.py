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

"""Per-job runtime resolution of per-user MCP source tools (built in code).

Deep Researcher Agent agents inherit data-source tools at *build* time, but a per-user MCP source's
tools are per-user/dynamic — the MCP client connects and enumerates tools using
the *user's* token. Two consequences shaped this design:

  * The tools can't be inherited statically by agents (no user at build time).
  * A ``per_user_mcp_client`` declared in the *config* is built by NAT's per-user
    *interactive* (WebSocket) session builder, which fails for a user with no
    token — breaking interactive chat. So we do NOT declare it in config.

Instead the headless async-job worker builds the per-user MCP client **in code**,
per job, after it has set ``Context.user_id`` to the job owner: it reads the MCP
endpoint from the source's ``mcp_oauth2`` auth provider, connects with the owner's
stored token (no interactive flow), enumerates the tools, and wraps them for the
agent. The client stays open via the caller's ``AsyncExitStack`` for the run.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack

from deep_researcher_agent.common.data_source_registry import get_all_sources
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class PerUserMcpSourceUnavailableError(RuntimeError):
    """An explicitly-selected protected MCP source could not be resolved at run time.

    Raised (rather than silently continuing) when the caller selected specific data
    sources and one of them is a configured per-user MCP source whose tools cannot
    be built — typically because the owner's token is missing or expired. Surfacing
    this lets the client prompt a reconnect instead of returning a web-only answer
    that misrepresents which sources were actually used.
    """

    def __init__(self, source_ids: list[str]) -> None:
        self.source_ids = source_ids
        names = ", ".join(source_ids)
        super().__init__(
            f"The following selected data source(s) are not connected (or the connection expired): {names}. "
            "Reconnect them in the data sources panel and try again."
        )


def _resolve_type_registry(builder):
    """Resolve NAT's type registry through dependency-tracking child builders."""
    current = builder
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        registry = getattr(current, "_registry", None)
        if registry is not None:
            return registry
        current = getattr(current, "_workflow_builder", None)
    raise TypeError(f"Could not resolve a NAT type registry from builder {type(builder).__name__}")


async def _token_usable(builder, cfg, source_id: str) -> bool:
    """Return whether the job owner has a usable (present, non-expired) token.

    If a token exists but is expired or has no credentials, it is deleted so the
    next ``get_status`` reports the source as disconnected (the UI then shows
    Reconnect) instead of falsely "connected". Best-effort: any error resolving
    the store is treated as "not usable" so we skip rather than crash the job.
    """
    from nat.builder.context import Context

    from .factory import _resolve_token_storage

    user_id = Context.get().user_id
    if not user_id:
        return False
    try:
        storage = await _resolve_token_storage(builder, cfg, source_id)
        auth = await storage.retrieve(user_id)
    except Exception as exc:
        logger.warning("Source '%s': could not read token state: %s", source_id, exc)
        return False

    if auth is None or not auth.credentials:
        return False  # nothing stored -> status is already not_connected
    if auth.is_expired():
        # Invalidate so the card stops showing "connected" and prompts Reconnect.
        try:
            await storage.delete(user_id)
        except Exception as exc:
            logger.debug("Source '%s': failed to delete expired token: %s", source_id, exc)
        logger.warning("Source '%s': stored token is expired; invalidated. User must reconnect.", source_id)
        return False
    return True


async def open_per_user_mcp_tools(
    *,
    builder,
    data_sources: list[str] | None,
    exit_stack: AsyncExitStack,
    wrapper_type: LLMFrameworkEnum | str = LLMFrameworkEnum.LANGCHAIN,
) -> list:
    """Build per-user MCP clients for selected protected sources; return their tools.

    Args:
        builder: The per-job ``WorkflowBuilder`` (resolves the auth provider + the
            framework tool wrapper).
        data_sources: Selected source ids, or ``None`` meaning "all" (so every
            connected protected source's tools are made available).
        exit_stack: An ``AsyncExitStack`` whose lifetime spans the agent run; the
            MCP client contexts are entered here and torn down when it closes.
        wrapper_type: Agent framework to wrap tools for.

    Returns:
        Framework-wrapped tools (possibly empty). Best-effort for the "all" case
        (``data_sources is None``): a failure resolving one source is logged and
        skipped so it never breaks the job.

    Raises:
        PerUserMcpSourceUnavailableError: when ``data_sources`` is an explicit list
            and one of the selected, configured per-user MCP sources cannot be
            resolved (missing/expired token, unreachable server). The caller asked
            for those sources specifically, so failing is preferable to silently
            answering without them.

    Precondition: ``Context.user_id`` must already be set to the job owner.
    """
    from nat.plugins.mcp.client.client_config import MCPServerConfig
    from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig
    from nat.plugins.mcp.client.client_config import PerUserMCPClientConfig
    from nat.plugins.mcp.client.client_impl import per_user_mcp_client_function_group

    selected = None if data_sources is None else {s.lower() for s in data_sources}
    tools: list = []
    # Explicitly-selected per-user sources we couldn't resolve -> fail closed below.
    unavailable: list[str] = []

    for source in get_all_sources():
        pua = source.per_user_auth
        if pua is None or not pua.required or not pua.auth_provider:
            continue
        explicitly_selected = selected is not None and source.id.lower() in selected
        if selected is not None and not explicitly_selected:
            continue

        try:
            # The mcp_oauth2 provider's server_url is the MCP endpoint; reuse it so
            # the connect flow and the job-time client target the same server, and
            # the client authenticates via the same provider (stored token lookup).
            provider = await builder.get_auth_provider(pua.auth_provider)
            server_url = str(getattr(getattr(provider, "config", None), "server_url", "") or "")
            if not server_url:
                logger.warning(
                    "Source '%s': auth provider '%s' has no server_url; cannot build MCP client.",
                    source.id,
                    pua.auth_provider,
                )
                if explicitly_selected:
                    unavailable.append(source.id)
                continue

            # Reconcile UI status with reality: the data-source card reports
            # "connected" from an offline token read, but the token can be expired
            # while the card still says connected. The use-site is authoritative —
            # if the owner's stored token is missing/expired here, invalidate it so
            # the next get_status returns not_connected/expired (UI -> Reconnect)
            # and skip, rather than failing the live MCP call and silently dropping
            # the tool while the UI keeps claiming connected.
            if not await _token_usable(builder, getattr(provider, "config", None), source.id):
                if explicitly_selected:
                    unavailable.append(source.id)
                continue

            # Give terse/blank MCP tools clear names + descriptions so the agent
            # reliably selects them over web search (declared on the source).
            tool_overrides = {
                name: MCPToolOverrideConfig(alias=ov.get("alias"), description=ov.get("description"))
                for name, ov in (pua.tool_overrides or {}).items()
            }
            client_cfg = PerUserMCPClientConfig(
                server=MCPServerConfig(transport="streamable-http", url=server_url, auth_provider=pua.auth_provider),
                tool_overrides=tool_overrides,
            )
            group = await exit_stack.enter_async_context(per_user_mcp_client_function_group(client_cfg, builder))
            fns = await group.get_accessible_functions()
            # Resolve the tool wrapper via the builder's type registry rather than
            # `builder._registry` directly: in server mode `builder` is a ChildBuilder,
            # which has no `_registry` (it delegates to its parent), so the attribute
            # access raised AttributeError and this whole block was silently swallowed
            # — dropping the selected source's tools and falling back to web search.
            wrapper = _resolve_type_registry(builder).get_tool_wrapper(llm_framework=wrapper_type)
            wrapped = [wrapper.build_fn(name, fn, builder) for name, fn in fns.items()]
            tools.extend(wrapped)

            # Map these runtime-resolved tools to their data source so the agents'
            # citation/source capture treats their results as sources. Without this,
            # get_source_id_for_tool returns None for them and shallow research raises
            # EmptySourceRegistryError ("no sources captured") even on a successful read.
            from deep_researcher_agent.common.data_source_registry import register_tool_sources

            register_tool_sources({getattr(t, "name", ""): source.id for t in wrapped if getattr(t, "name", "")})
            logger.info("Resolved %d per-user MCP tool(s) for source '%s'.", len(wrapped), source.id)
        except Exception:
            logger.exception(
                "Failed to resolve per-user MCP tools for source '%s'; continuing without them.",
                source.id,
            )
            if explicitly_selected:
                unavailable.append(source.id)

    # Fail closed for sources the caller singled out but we couldn't resolve (e.g.
    # a token that expired between submit-time preflight and job execution). For the
    # "all" case (data_sources is None) we stay best-effort — the user didn't ask
    # for these specifically, so a missing one shouldn't sink the whole run.
    if unavailable:
        raise PerUserMcpSourceUnavailableError(unavailable)

    return tools
