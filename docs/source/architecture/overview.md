<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Overview

The NVIDIA AI-Q Blueprint is a multi-agent research system built on the
[NVIDIA NeMo Agent Toolkit](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html).
It uses a two-tier research architecture that keeps simple queries fast while
reserving multi-phase deep research for complex topics.

## System Flow

The following diagram shows the full request lifecycle from user query to
final response. Every query enters through the Intent Classifier, which
decides whether to respond directly (meta), perform a quick tool-augmented
lookup (shallow), or initiate a comprehensive multi-agent investigation (deep).

```{image} /_static/DeepResearcher-arch-light.png
:alt: AI-Q Architecture
:align: center
```

### Reading the Diagram

- The deep-research path is dynamic rather than a fixed linear chain. The
  optional source router informs planning, and the orchestrator dispatches
  independent research queries to reusable researcher workers concurrently
  before delegating final synthesis to the writer.
- **Shared Research State** represents the DeepAgents state and virtual
  `/shared/` filesystem used for the plan, research notes, progress, and final
  output. It is distinct from external data sources and the durable generated
  artifact store.
- **Skill Definitions** are host-side, read-only instructions assigned to
  specific roles. Research skills are exposed to researcher workers and
  synthesis skills to the writer by the shipped skills profile.
- **Job-Scoped Sandbox** is optional. When configured, agent-generated code
  runs through one provider sandbox per deep-research job; Modal and OpenShell
  are alternative providers. Inference, source-tool calls, credentials, and
  shared research state remain in the AI-Q process.

## Core Components

| Component | Role | Location |
| --------- | ---- | -------- |
| [Intent Classifier](agents/intent-classifier.md) | Single LLM call: classifies intent (meta/research) and depth (shallow/deep) | `agents/chat_researcher/nodes/intent_classifier.py` |
| [Clarifier Agent](agents/clarifier.md) | Optionally gathers missing context and the requested output type before deep research | `agents/clarifier/agent.py` |
| [Shallow Researcher](agents/shallow-researcher.md) | Fast, bounded tool-augmented research | `agents/shallow_researcher/agent.py` |
| [Deep Researcher](agents/deep-researcher.md) | Orchestrates optional source routing, structured planning, concurrent research workers, and writer-first synthesis | `agents/deep_researcher/agent.py` |
| Chat Researcher Orchestrator | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) state machine coordinating all agents | `agents/chat_researcher/agent.py` |
| [Data Source Registry](../customization/tools-and-sources.md) | Applies per-request source selection and exposes the active source-tool set to research agents | `common/data_source_registry.py` |
| [Knowledge Layer](../customization/knowledge-layer.md) | Pluggable document ingestion and retrieval through LlamaIndex, Foundational RAG, Azure AI Search, or OpenSearch | `sources/knowledge_layer/` |
| Shared Research State | Host-side DeepAgents state and `/shared/` virtual filesystem for plans, research notes, progress, and final output | `agents/deep_researcher/deepagents_runtime.py` |
| [Skills and Execution Runtime](agents/sandbox.md) | Exposes role-assigned, read-only skill definitions and optionally routes generated code to a job-scoped sandbox | `agents/deep_researcher/skills/`, `agents/deep_researcher/sandbox/` |

## Orchestrator State Machine

The `ChatResearcherAgent` builds a LangGraph `StateGraph` over `ChatResearcherState` with
six nodes and conditional edges:

```mermaid
graph LR
    IC[intent_classifier] -->|meta| END_NODE[END]
    IC -->|"new research or no report; depth=deep"| CL[clarifier]
    IC -->|"new research or no report; depth=shallow"| SR[shallow_research]
    SR -->|"escalate"| CL
    SR -->|"done"| END_NODE
    IC -->|"target=report; report available; action=ask"| RA[report_ask]
    IC -->|"target=report; report available; action=edit"| RE[report_edit]
    CL --> DR[deep_research]
    DR --> END_NODE
    RA --> END_NODE
    RE --> END_NODE

    style IC fill:#fff3e0
    style SR fill:#f3e5f5
    style CL fill:#fce4ec
    style DR fill:#e8eaf6
    style RA fill:#e0f2f1
    style RE fill:#e0f2f1
```

**Routing logic:**

1. **`route_after_orchestration`** -- After the intent classifier runs, the
   graph inspects `state.user_intent.intent`. If `meta`, the response is
   already in `state.messages` and the graph routes to `END`. If `research`,
   it checks `state.depth_decision.decision` to choose `shallow_research` or
   `clarifier`.

2. **Report follow-up** -- When `user_intent.target` is `report` and an
   *existing* report is available, the graph routes to `report_edit` if
   `report_action` is `edit`, and to `report_ask` otherwise. With no report
   available yet, a `target=report` request falls through to the normal shallow
   or deep research path.

3. **`should_escalate`** -- After shallow research completes, the graph
   evaluates whether the response warrants escalation to deep research. It
   checks for empty responses and escalation keywords ("unable to find",
   "need more research", "i don't have enough information") in the last
   800 characters of the AI response. When escalation triggers, the graph
   routes to the `clarifier` node (not directly to `deep_research`), so it
   can gather any missing context or output-shape preference before research.
   Research planning then occurs inside the deep-research workflow.
   Escalation is gated by the `enable_escalation` config flag.

## ChatResearcherState

The central state model carries data through the entire workflow:

| Field | Type | Purpose |
| ----- | ---- | ------- |
| `messages` | `list[AnyMessage]` | Conversation history (LangGraph message reducer) |
| `user_info` | `dict` or `None` | Authenticated user information for personalization |
| `data_sources` | `list[str]` or `None` | Hard per-request filter for registry-mapped source tools. Unmapped configured or utility tools remain active. |
| `user_intent` | `IntentResult` or `None` | Classification result: `intent` is `meta` or `research`; `target` is `meta`, `new_research`, or `report`; report targets carry `report_action` (`ask` or `edit`). |
| `depth_decision` | `DepthDecision` or `None` | Routing decision: `shallow` or `deep` |
| `final_report` | `str` or `None` | Final report output from deep research |
| `shallow_result` | `ShallowResult` or `None` | Result from shallow research path |
| `clarifier_result` | `str` or `None` | Clarification log containing missing context or output-shape preferences |
| `original_query` | `str` or `None` | Preserved user query for deep research |
| `available_documents` | `list[AvailableDocument]` or `None` | User-uploaded documents with summaries |

## Design Decisions

- **Two-tier routing**: Keeps common queries fast (single tool-calling loop)
  while reserving multi-stage deep research for complex cases. The intent
  classifier makes the routing decision in a single LLM call to minimize
  latency.

- **LangGraph state machine**: Provides explicit, testable routing with
  conversation checkpointing using `InMemorySaver` or persistent backends
  (SQLite, PostgreSQL).

- **Separated coordination and execution**: The deep-research orchestrator
  invokes optional source-router, planner, and writer task subagents. Research
  queries run through `run_research_batch` as concurrent invocations of a
  reusable researcher worker, not as `task()` subagents. The writer is the
  normative final-synthesis stage. If `/shared/output.md` is missing, the
  runtime can defensively accept a substantive orchestrator-authored Markdown
  report, but rejects short workflow chatter.

- **Toolkit-independent agents**: All agents receive dependencies through constructor
  injection for testability. NeMo Agent Toolkit registration is a thin layer in `register.py`
  files.

- **Data source filtering**: `data_sources` filters tools mapped in the data
  source registry before the active research agents are constructed. Unmapped
  configured tools remain active and are absent from the router catalog.
  Routing is advisory, and `ResearchQuery` preferred and fallback tool names
  are prompt guidance, while workers remain bound to the full request-filtered set.

- **Evaluation-driven defaults**: Routing and research budgets are tuned
  through benchmarks (FreshQA, Deep Research Bench) and can evolve as
  evaluation scores improve.

- **Citation verification and auditability**: Research paths capture sources
  for deterministic citation checks and URL sanitization. Deep-research
  citation verification is enabled by default and configurable with
  `enable_citation_verification`; it is not an unconditional workflow stage.
  Refer to
  [Deep Researcher -- Citation Verification](agents/deep-researcher.md#phase-5-citation-verification-post-processing)
  for details.

## References

- [Data Flow](data-flow.md) -- request lifecycle, SSE events, async jobs
- [NeMo Agent Toolkit documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html)
- [Development and Contributing Guide](../contributing/index.md)
