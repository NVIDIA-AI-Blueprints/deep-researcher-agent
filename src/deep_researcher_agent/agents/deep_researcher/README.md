# Deep Researcher Package

This package implements the DeepAgents-backed deep-research path. The
[Deep Researcher architecture guide](../../../../docs/source/architecture/agents/deep-researcher.md)
is the canonical description of its runtime flow, state, middleware, and
configuration. Keep detailed behavior in the architecture guide so the public and package-local
documentation do not diverge.

## Runtime Contract

- The orchestrator coordinates stage order. It does not call source tools
  directly and normally delegates final synthesis to `writer-agent`.
- When enabled, `source-router-agent` writes advisory routing guidance before
  planning. Routing cannot restore a source excluded by the request boundary.
- `planner-agent` returns a structured `ResearchPlan` containing independent
  `ResearchQuery` objects. Their `preferred_tools` and `fallback_tools` fields
  are prompt guidance; every worker retains the full request-filtered tool set.
- `run_research_batch` runs independent researcher workers concurrently up to
  `max_research_concurrency`. Source-tool calls and concrete batched inputs have
  separate configured bounds.
- Each worker returns structured `ResearchNotes`. The writer reads the plan,
  notes, and captured sources, then writes the normative final answer to
  `/shared/output.md`.
- `_salvage_inline_report()` is a defensive compatibility path when writer
  delegation is missed. An orchestrator-authored report is not the normative
  synthesis path.
- Clarification happens before deep research only when request scope or output
  shape is materially ambiguous. The clarifier does not create or approve the
  internal research plan.

## Package Map

| Path | Responsibility |
| ---- | -------------- |
| `agent.py` | Per-run agent lifecycle, final report extraction, citation verification, and defensive report salvage |
| `register.py` | NeMo Agent Toolkit configuration and component registration |
| `factory.py` | Role-specific graph, middleware, tool, and permission assembly |
| `deepagents_runtime.py` | Optional skills and sandbox runtime wiring |
| `models/subagent_contracts.py` | Structured routing, planning, and research-note contracts |
| `tools/source_routing.py` | Advisory source catalog lookup and route persistence |
| `tools/research.py` | Concurrent `ResearchQuery` worker dispatch and note persistence |
| `custom_middleware.py` | Plan persistence, source capture, retries, and role guards |
| `prompts/` | Orchestrator, router, planner, researcher, writer, and source-list templates |
| `sandbox/` | Sandbox providers and durable artifact collection |

## Related Documentation

- [Configuration Reference](../../../../docs/source/customization/configuration-reference.md)
- [Prompt Customization](../../../../docs/source/customization/prompts.md)
- [Deep Research Sandbox Notes](../../../../docs/source/architecture/agents/sandbox.md)
- [Deep Research Bench](../../../../frontends/benchmarks/deepresearch_bench/README.md)

Run the package tests from the repository root:

```bash
uv run pytest tests/deep_researcher_agent/agents/deep_researcher
```
