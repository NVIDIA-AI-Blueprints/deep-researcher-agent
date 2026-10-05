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

"""Deep research agent using deepagents library for multi-phase workflow."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool
from nemo_relay.integrations.deepagents import NemoRelayDeepAgentsCallbackHandler

from aiq_agent.common import LLMProvider
from aiq_agent.common import load_prompt
from aiq_agent.common import validate_research_source_configuration
from aiq_agent.common.citation_verification import CitationIntegrityError
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import EmptySourceRegistryReason
from aiq_agent.common.citation_verification import sanitize_report
from aiq_agent.common.citation_verification import source_entries_from_parent_context
from aiq_agent.common.citation_verification import verify_citations
from aiq_agent.common.logging_utils import log_content_metadata
from aiq_agent.relay import run_agent

from .custom_middleware import FinalReportCommitTracker
from .custom_middleware import SourceRegistryMiddleware
from .deepagents_runtime import DeepAgentsRuntime
from .deepagents_runtime import DeepResearchSandboxConfig
from .deepagents_runtime import DeepResearchSkillsConfig
from .factory import build_deep_research_graph
from .factory import build_deep_research_middleware_set
from .factory import build_deep_research_tool_set
from .models import DeepResearchAgentState
from .resource_limits import DeepResearchExecutionTimeout
from .resource_limits import DeepResearchResourceLimits
from .resource_limits import StateBudgetLedger
from .tools.source_tool_batching import DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS
from .tools.source_tool_batching import DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE
from .tools.source_tool_batching import activate_source_tool_budget
from .tools.source_tool_batching import reset_source_tool_budget

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESEARCH_CONCURRENCY = 6
DEFAULT_MAX_RESEARCHER_MODEL_CALLS = 100
PARENT_REPORT_CONTEXT_PATH = "/shared/parent_report_context.json"

# Path to this agent's directory (for loading prompts)
AGENT_DIR = Path(__file__).parent


class WorkflowOutputError(RuntimeError):
    """Raised when the deep-research workflow does not produce a final report."""


class DeepResearcherAgent:
    """
    Deep research agent using deepagents library for multi-phase workflow.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        callbacks: list[Any] | None = None,
        domain_catalog_path: str | None = None,
        enable_source_router: bool = True,
        enable_citation_verification: bool = True,
        skills: DeepResearchSkillsConfig | None = None,
        sandbox: DeepResearchSandboxConfig | None = None,
        job_id: str | None = None,
        artifact_db_url: str | None = None,
        artifact_emit: Callable[[dict[str, Any]], None] | None = None,
        max_research_concurrency: int = DEFAULT_MAX_RESEARCH_CONCURRENCY,
        max_researcher_model_calls: int = DEFAULT_MAX_RESEARCHER_MODEL_CALLS,
        max_concurrent_source_tool_calls: int = DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
        max_source_tool_batch_size: int = DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
        resource_limits: DeepResearchResourceLimits | None = None,
    ) -> None:
        """
        Initialize the deep researcher agent.

        Args:
            llm_provider: LLMProvider for role-based LLM access.
            tools: Optional sequence of LangChain tools for research.
            callbacks: Optional list of callbacks.
            domain_catalog_path: Optional YAML/JSON domain catalog path for source-router-agent.
            enable_source_router: Enable the advisory source-router-agent before planning.
            enable_citation_verification: Verify generated citations against the captured source registry.
            skills: Optional DeepAgents skills config.
            sandbox: Optional DeepAgents sandbox config.
            job_id: Optional async job identifier used to scope sandbox backends.
            max_research_concurrency: Maximum ResearchQuery items accepted and run concurrently per
                run_research_batch call.
            max_researcher_model_calls: Maximum normal model turns per researcher worker before finalization.
            max_concurrent_source_tool_calls: Shared source-tool concurrency limit across researcher workers.
            max_source_tool_batch_size: Maximum concrete inputs per batch-capable source tool call.
            resource_limits: Hard per-job request, state, source-call, and wall-clock limits.
        """
        self.llm_provider = llm_provider
        self.tools = list(tools) if tools else []
        self.callbacks = callbacks or []
        self.max_research_concurrency = max_research_concurrency
        self.max_researcher_model_calls = max_researcher_model_calls
        self.max_concurrent_source_tool_calls = max_concurrent_source_tool_calls
        self.max_source_tool_batch_size = max_source_tool_batch_size
        self.resource_limits = resource_limits or DeepResearchResourceLimits()
        self.domain_catalog_path = domain_catalog_path
        self.enable_source_router = enable_source_router
        self.enable_citation_verification = enable_citation_verification
        self.job_id = str(job_id) if job_id is not None else str(uuid4())

        self.deepagents_runtime = DeepAgentsRuntime(
            skills=skills,
            sandbox=sandbox,
            job_id=self.job_id,
            artifact_db_url=artifact_db_url,
            artifact_emit=artifact_emit,
        )

        try:
            self._prompts = self._load_prompts()
            source_tool_names = {tool.name for tool in self.tools}
            self.source_registry_middleware = SourceRegistryMiddleware(source_tool_names=source_tool_names)
            self.tool_set = build_deep_research_tool_set(
                self.tools,
                source_registry_middleware=self.source_registry_middleware,
                max_concurrent_source_tool_calls=self.max_concurrent_source_tool_calls,
                max_source_tool_batch_size=self.max_source_tool_batch_size,
            )
            self.middleware_set = build_deep_research_middleware_set(
                tool_set=self.tool_set,
                source_registry_middleware=self.source_registry_middleware,
                enable_source_router=self.enable_source_router,
                artifact_manager=self.deepagents_runtime.artifact_manager,
            )

            self.source_tool_names = self.tool_set.source_tool_names
            self.tools_info = self.tool_set.tools_info
            self.non_search_tools = self.tool_set.helper_tools
            self.all_tools = self.tool_set.all_tools
            self.research_source_tools = self.tool_set.research_source_tools
            self.researcher_tools = self.tool_set.researcher_tools
            self.writer_tools = self.tool_set.writer_tools
            self.researcher_middleware = self.middleware_set.researcher
            self.writer_middleware = self.middleware_set.writer
            self.orchestrator_middleware = self.middleware_set.orchestrator
            self.middleware = self.researcher_middleware
        except Exception:
            try:
                cleanup_succeeded = self.deepagents_runtime.finalize(interrupted=False)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the original construction failure
                logger.warning(
                    "Deep research runtime cleanup failed during agent construction (%s)",
                    type(cleanup_error).__name__,
                )
            else:
                if not cleanup_succeeded:
                    logger.warning("Deep research runtime cleanup reported failure during agent construction")
            raise

    def finalize(self, *, interrupted: bool) -> bool:
        """Release this request's sandbox runtime exactly once."""
        return self.deepagents_runtime.finalize(interrupted=interrupted)

    def _load_prompts(self) -> dict[str, str]:
        """Load all prompts for subagents."""
        prompts = {}
        prompt_names = ["planner", "researcher", "orchestrator", "writer", "source_router"]

        for name in prompt_names:
            prompts[name] = load_prompt(AGENT_DIR / "prompts", name)

        return prompts

    def _build_orchestrator_agent(
        self,
        state: DeepResearchAgentState,
        *,
        final_report_tracker: FinalReportCommitTracker,
        state_budget: StateBudgetLedger | None = None,
    ) -> Any:
        """Build the orchestrator graph for the current state."""
        return build_deep_research_graph(
            llm_provider=self.llm_provider,
            state=state,
            prompts=self._prompts,
            tools=self.tools,
            runtime=self.deepagents_runtime,
            tool_set=self.tool_set,
            middleware_set=self.middleware_set,
            source_registry_middleware=self.source_registry_middleware,
            callbacks=self.callbacks,
            domain_catalog_path=self.domain_catalog_path,
            enable_source_router=self.enable_source_router,
            max_research_concurrency=self.max_research_concurrency,
            max_researcher_model_calls=self.max_researcher_model_calls,
            resource_limits=self.resource_limits,
            final_report_tracker=final_report_tracker,
            state_budget=state_budget,
        )

    def _extract_final_markdown(
        self,
        result: dict | Any,
        files: dict[str, Any] | None = None,
        *,
        final_report_tracker: FinalReportCommitTracker,
    ) -> str | None:
        """Extract only Markdown committed by the writer during this run."""
        # Resolve result files first, then fall back to the passed-in files (state.files) and
        # finally an empty dict. Without the explicit grouping, `or files or {}` bound only to
        # the else branch, so a dict result lacking a usable "files" key silently discarded
        # the fallback even when output files existed.
        result_files = result.get("files", None) if isinstance(result, dict) else getattr(result, "files", None)
        files = result_files or files or {}
        committed = final_report_tracker.committed_text(files)
        return committed.strip() if committed is not None else None

    @staticmethod
    def _read_seed_file_text(files: dict[str, Any], path: str) -> str | None:
        entry = files.get(path)
        if isinstance(entry, dict):
            entry = entry.get("content")
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8")
        return entry if isinstance(entry, str) and entry.strip() else None

    def _seed_parent_sources(self, files: dict[str, Any]) -> None:
        """Register parent report sources so preserved citations verify in delta reports."""
        context_text = self._read_seed_file_text(files, PARENT_REPORT_CONTEXT_PATH)
        if not context_text:
            return
        parent_sources = source_entries_from_parent_context(context_text)
        seeded = self.source_registry_middleware.register_compact_sources(parent_sources)
        if seeded:
            logger.info("Seeded %d parent report source(s) into citation registry", seeded)

    @staticmethod
    def _replace_last_message_content(result: dict | Any, content: str) -> None:
        """Overwrite the final message content in-place with post-processed Markdown."""
        messages = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", None)
        if not messages:
            return
        last_msg = messages[-1]
        if hasattr(last_msg, "model_copy"):
            messages[-1] = last_msg.model_copy(update={"content": content})
        else:
            messages[-1] = type(last_msg)(content=content)

    async def run(self, state: DeepResearchAgentState) -> DeepResearchAgentState:
        """
        Execute deep research with multi-phase workflow.
        """
        # Both the NAT registrar and async job runner call this interface, so source
        # selection and availability must be enforced here rather than by either adapter.
        validate_research_source_configuration(state.data_sources, "deep research", self.tools)

        messages = state.messages
        query = ""
        if messages:
            query_content = messages[-1].content
            query = query_content if isinstance(query_content, str) else str(query_content)
        clarifier_result = state.clarifier_result or ""
        request_context_chars = len(query) + len(clarifier_result)
        if request_context_chars > self.resource_limits.max_input_chars:
            raise ValueError(
                "Deep-research query and clarification context exceeds the "
                f"{self.resource_limits.max_input_chars}-character limit"
            )

        prepared_files = self.deepagents_runtime.prepare_state_files(dict(state.files))
        if prepared_files != state.files:
            state = state.model_copy(update={"files": prepared_files})
        state_budget = StateBudgetLedger(
            limits=self.resource_limits,
            files=dict(state.files),
            sandbox_enabled=self.deepagents_runtime.execution_enabled,
        )
        self._seed_parent_sources(state.files)
        final_report_tracker = FinalReportCommitTracker()
        agent = self._build_orchestrator_agent(
            state,
            final_report_tracker=final_report_tracker,
            state_budget=state_budget,
        )

        if messages:
            logger.info("=" * 80)
            logger.info("Deep Research Subagent: Starting workflow")
            logger.info("Query: %s", log_content_metadata(query))
            logger.info("=" * 80)

        budget_token = activate_source_tool_budget(
            self.resource_limits.max_source_tool_calls,
            max_consecutive_failures=self.resource_limits.max_consecutive_source_tool_failures,
        )
        try:
            execution_timeout = asyncio.timeout(self.resource_limits.max_execution_seconds)
            try:
                async with execution_timeout:

                    async def _invoke_orchestrator() -> Any:
                        callbacks = [*self.callbacks, NemoRelayDeepAgentsCallbackHandler()]
                        return await agent.ainvoke(state, config={"callbacks": callbacks})

                    result = await run_agent(
                        "deep_research_agent",
                        _invoke_orchestrator,
                        input_value={
                            "message_count": len(messages),
                            "query_character_count": len(query),
                            "file_count": len(state.files),
                        },
                    )
            except TimeoutError as exc:
                # An inner provider/tool may raise TimeoutError for its own operation.
                # Only the asyncio.timeout context's deadline is a job-level resource
                # timeout that requires forcible sandbox teardown.
                if execution_timeout.expired():
                    raise DeepResearchExecutionTimeout("Deep-research execution time limit exceeded") from exc
                raise
            if execution_timeout.expired():
                # A coroutine can suppress its cancellation. The job still crossed
                # the hard wall-clock boundary and must not be treated as successful.
                raise DeepResearchExecutionTimeout("Deep-research execution time limit exceeded")

            final_message = self._extract_final_markdown(
                result,
                state.files,
                final_report_tracker=final_report_tracker,
            )
            # Capturing no sources is a research outcome failure even when citation
            # rewriting is disabled; enable_citation_verification only controls the latter.
            if not self.source_registry_middleware.has_sources():
                from aiq_agent.common.citation_verification import classify_empty_source_registry_reason
                from aiq_agent.common.tool_validation import validate_tool_availability

                _, available_count, unavailable = validate_tool_availability(
                    self.tools,
                    research_type="deep research",
                    enable_logging=False,
                )
                generated_answer = None
                if final_message is not None:
                    generated_answer = sanitize_report(final_message).sanitized_report
                raise EmptySourceRegistryError(
                    "deep research",
                    unavailable_tools=unavailable,
                    available_count=available_count,
                    reason=classify_empty_source_registry_reason(state.data_sources, available_count, unavailable),
                    generated_answer=generated_answer,
                )
            if final_message is None:
                raise WorkflowOutputError("writer-agent did not produce a final Markdown answer")

            # Post-process: verify citations against source registry
            citation_registry = None
            if self.enable_citation_verification and self.source_registry_middleware.has_sources():
                registry = self.source_registry_middleware.active_registry()
                verification = verify_citations(
                    final_message,
                    registry,
                    reference_sources=self.source_registry_middleware.get_source_entries(mode="compact"),
                )
                if verification.removed_citations:
                    logger.info(
                        "Citation verification removed %d invalid citation(s)",
                        len(verification.removed_citations),
                    )
                final_message = verification.verified_report
                citation_registry = registry
            # Post-process: sanitize report (strip body URLs, shortened URLs, unsafe URLs)
            sanitization = sanitize_report(final_message)
            final_message = sanitization.sanitized_report

            # Post-process: harvest sandbox artifacts and resolve artifact:// references so
            # generated charts/files render in the report. Inert (manager is None) unless a
            # sandbox + artifact_capture + db_url are configured. Blocking I/O off the loop.
            manager = self.deepagents_runtime.artifact_manager
            if manager is not None:
                try:
                    await asyncio.to_thread(manager.final_harvest)
                    produced = await asyncio.to_thread(manager.store.list, manager.job_id)
                    final_message = await asyncio.to_thread(manager.resolve_report_references, final_message, produced)
                    final_message = await asyncio.to_thread(
                        manager.ensure_inline_artifacts_embedded, final_message, produced
                    )
                    final_message = await asyncio.to_thread(manager.append_artifact_index, final_message, produced)
                except Exception:
                    # Best-effort: never discard an already verified/sanitized report because
                    # artifact harvest or embedding failed. final_message stays as-is.
                    logger.warning(
                        "Artifact post-processing failed; returning report without embedded artifacts",
                        exc_info=True,
                    )

            final_cited_urls: list[str] | None = None
            if citation_registry is not None:
                final_verification = verify_citations(final_message, citation_registry)
                if not final_verification.valid_citations:
                    raise CitationIntegrityError()
                final_message = final_verification.verified_report
                final_cited_urls = list(
                    dict.fromkeys(
                        citation["url"] for citation in final_verification.valid_citations if citation.get("url")
                    )
                )

            # Re-emit the verified/sanitized report so the frontend overwrites
            # the raw version that on_llm_end auto-emitted during ainvoke().
            for cb in self.callbacks:
                if hasattr(cb, "emit_final_report"):
                    cb.emit_final_report(final_message, cited_urls=final_cited_urls)
                    break

            self._replace_last_message_content(result, final_message)

            logger.info("=" * 80)
            logger.info("Deep Research Subagent: Workflow complete")
            logger.info("Final answer length: %d characters", len(final_message))
            logger.info("=" * 80)
            return DeepResearchAgentState.model_validate(result)

        except EmptySourceRegistryError as ex:
            if ex.reason is not EmptySourceRegistryReason.NO_SOURCE_RESULTS:
                logger.error(
                    "Deep Research Subagent failed (error_type=%s detail_%s)",
                    type(ex).__name__,
                    log_content_metadata(ex),
                )
            raise
        except Exception as ex:
            logger.error(
                "Deep Research Subagent failed (error_type=%s detail_%s)",
                type(ex).__name__,
                log_content_metadata(ex),
            )
            raise
        finally:
            reset_source_tool_budget(budget_token)
