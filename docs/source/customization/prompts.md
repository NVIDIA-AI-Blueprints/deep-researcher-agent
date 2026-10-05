<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompts

Each agent in the AI-Q blueprint uses [Jinja2](https://jinja.palletsprojects.com/) templates to define its system prompt. These templates control the agent's persona, instructions, output format, and behavior. By editing these templates you can customize how agents reason, what they prioritize, and how they format responses -- all without modifying Python code.

## Prompt Template Inventory

| Template | Consumer | Purpose |
|----------|----------|---------|
| `src/aiq_agent/agents/chat_researcher/prompts/intent_classification.j2` | Intent Classifier | Classifies queries as meta or research, determines depth (shallow/deep), generates meta responses |
| `src/aiq_agent/agents/chat_researcher/prompts/context_aware_intent_router.j2` | Context-aware Intent Router | Classifies interactions, selects classic depth, and performs bounded catalog discovery for GSF-enabled workflows |
| `src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2` | Shallow Researcher | Defines the research persona, tool usage strategy, source hierarchy, and citation rules |
| `src/aiq_agent/agents/data_science/prompts/agent.j2` | Data Science Agent | Guides adaptive semantic discovery, structured queries, document/web evidence gathering, and analytical synthesis |
| `src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2` | Deep Research Orchestrator | Coordinates ordered routing, planning, batched research, and writer delegation; it does not call source tools directly |
| `src/aiq_agent/agents/deep_researcher/prompts/source_router.j2` | Source Router | Selects an advisory route from the request-allowed source catalog before planning |
| `src/aiq_agent/agents/deep_researcher/prompts/planner.j2` | Deep Research Planner | Grounds and returns a structured `ResearchPlan` with independent `ResearchQuery` objects |
| `src/aiq_agent/agents/deep_researcher/prompts/researcher.j2` | Researcher Worker | Executes one `ResearchQuery` and returns structured `ResearchNotes` using preferred and fallback tools as prompt guidance |
| `src/aiq_agent/agents/deep_researcher/prompts/writer.j2` | Report Writer | Synthesizes the plan, research notes, and captured sources into `/shared/output.md` |
| `src/aiq_agent/agents/deep_researcher/prompts/source_registry.j2` | Source Registry Middleware | Renders the captured source list exposed to the writer |
| `src/aiq_agent/agents/clarifier/prompts/research_clarification.j2` | Clarifier | Determines whether a request needs clarification and asks focused follow-up questions; it does not create or approve a plan |

## Template Directory Structure

Each agent stores its prompts in a `prompts/` subdirectory co-located with the agent code:

```
src/aiq_agent/agents/
    shallow_researcher/
        prompts/
            researcher.j2              # Single system prompt
    data_science/
        prompts/
            agent.j2                   # Adaptive analytical controller prompt
    deep_researcher/
        prompts/
            orchestrator.j2            # Orchestrator prompt
            source_router.j2            # Advisory source-routing prompt
            planner.j2                 # Research planner prompt
            researcher.j2              # Researcher-worker prompt
            writer.j2                  # Final synthesis prompt
            source_registry.j2         # Captured-source list fragment
    clarifier/
        prompts/
            research_clarification.j2  # Clarification prompt
    chat_researcher/
        prompts/
            intent_classification.j2   # Routing prompt
            context_aware_intent_router.j2  # Catalog-aware routing prompt
```

The naming convention follows each runtime role. `source_registry.j2` is a middleware-rendered fragment rather than an agent system prompt.

## How Templates Are Loaded

At runtime, templates flow through two utility functions in `src/aiq_agent/common/prompt_utils.py`:

```
prompts/researcher.j2  (Jinja2 source)
        │
        ▼
  load_prompt()         (reads file from disk)
        │
        ▼
  render_prompt_template()  (renders with variables)
        │
        ▼
  SystemMessage(content=...)  (sent to LLM)
```

### `load_prompt(path, name)`

Loads a raw template file from the agent's `prompts/` directory. Automatically appends `.j2` if the file is not found by exact name.

```python
from aiq_agent.common import load_prompt

# Load the template file as a string
template = load_prompt(Path(__file__).parent / "prompts", "researcher")
```

### `render_prompt_template(template, **kwargs)`

Renders a Jinja2 template string with the provided variables. Uses `jinja2.StrictUndefined` so that missing variables raise errors rather than producing silent empty strings.

```python
from aiq_agent.common import render_prompt_template

rendered = render_prompt_template(
    template,
    current_datetime="2026-02-16 10:30:00",
    tools=tools,
    user_info={"name": "Alice", "email": "alice@example.com"},
)
```

## Template Variables

Each template receives different variables depending on the agent context.

### Intent Classification

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | User context with `name` and `email` keys |
| `tools` | `list[dict]` | Available tools (each has `name` and `description` keys) |
| `query` | `str` | The user's query text |
| `active_report_available` | `bool` | Whether the conversation has a report that can be edited or extended |

The GSF-enabled context-aware router also receives catalog availability and
configured search bounds. For a mixed enterprise-and-public request, it sends a
contiguous enterprise-data span copied verbatim from the user request as the
catalog tool's `question`. It does not create a public-research subquery or
plan, and the original user request remains the downstream research input.
Python rejects a catalog question that is not a verbatim span of that request.

### Shallow Researcher

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | User context with `name` and `email` keys |
| `tools` | `list[dict]` | Available tools (each has `name` and `description` keys) |
| `available_documents` | `list[dict]` or `None` | Uploaded documents with `file_name` and `summary` keys |

### Data Science Agent

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | Authenticated user context |
| `tools` | `list[dict]` | Exact configured tools, with `name` and `description` |
| `interaction_mode` | `str` | `interactive` permits a concise clarification; `headless` requires a terminal answer without user input |
| `response_mode` | `str` | `standard` for normal reports or `fdabench_choice` for labeled-choice output when options are present |
| `structured_catalog_call_limit` | `int` or `None` | Optional request-local catalog-call ceiling rendered into the working rules |
| `structured_text_to_sql_call_limit` | `int` or `None` | Optional request-local text-to-SQL-call ceiling rendered into the working rules |
| `python_call_limit` | `int` or `None` | Optional call ceiling for stateless scientific Python execution |

### Deep Research Orchestrator

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `clarifier_result` | `str` or `None` | Clarification context from the clarifier agent |
| `available_documents` | `list[dict]` or `None` | Uploaded documents with `file_name` and `summary` keys |
| `user_info` | `dict` or `None` | Authenticated user context |
| `tools` | `list[dict]` | Orchestrator-callable helper tools and `run_research_batch`; source tools are intentionally excluded |
| `enable_source_router` | `bool` | Whether to run the advisory source-router stage |
| `max_research_concurrency` | `int` | Maximum `ResearchQuery` objects accepted in one research batch |
| `parent_report_context_available` | `bool` | Whether mounted parent-report files are available, enabling delta planning and revision instructions |
| `execution_enabled` | `bool` | Whether a sandbox exposes execution tools |
| `sandbox_workdir`, `sandbox_artifact_dir` | `str` | Per-job sandbox paths used when execution is enabled |

### Deep Research Source Router

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | Authenticated user context |
| `clarifier_result` | `str` or `None` | Clarification context used to choose an advisory route |
| `available_documents` | `list[dict]` or `None` | Uploaded documents available as routing context |

### Deep Research Planner

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | Authenticated user context |
| `tools` | `list[dict]` | Request-filtered source tools available for plan grounding |
| `available_documents` | `list[dict]` or `None` | Uploaded documents with `file_name` and `summary` keys |
| `enable_source_router` | `bool` | Whether `/shared/source_routing.json` may provide advisory guidance |
| `max_research_concurrency` | `int` | Preferred upper bound for independent queries in one batch |

### Deep Research Researcher

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | Authenticated user context |
| `tools` | `list[dict]` | Full request-filtered source-tool set available to each worker; plan preferences do not narrow this set |
| `available_documents` | `list[dict]` or `None` | Uploaded documents with `file_name` and `summary` keys |
| `sandbox_workdir`, `sandbox_artifact_dir` | `str` | Per-job sandbox paths available to skills when configured |

### Deep Research Writer

| Variable | Type | Description |
|----------|------|-------------|
| `current_datetime` | `str` | Current date and time string |
| `user_info` | `dict` or `None` | Authenticated user context |
| `parent_report_context_available` | `bool` | Whether mounted parent-report files are available for standalone revision synthesis with delta evidence |
| `sandbox_workdir`, `sandbox_artifact_dir` | `str` | Per-job sandbox paths for generated report artifacts |

The writer reads the persisted plan and research notes from `/shared/` and retrieves captured sources through `get_verified_sources`; those inputs are runtime files and tools rather than Jinja variables. The separate `source_registry.j2` fragment receives a `sources` list from `SourceRegistryMiddleware`.

### Research Clarification

| Variable | Type | Description |
|----------|------|-------------|
| `clarifier_result` | `str` or `None` | Previous clarification context (for multi-turn clarification) |
| `available_documents` | `list[dict]` or `None` | Uploaded documents with `file_name` and `summary` keys |
| `connected_sources` | `list[dict]` | Connected data sources that the later research phase can access |
| `tools` | `list[dict]` | Available tools (each has `name` and `description` keys) |
| `tool_names` | `list[str]` | List of tool name strings extracted from `tools` |

## Modifying Prompts

### Editing Existing Templates

The most common customization is editing the `.j2` files directly. Since templates are loaded from disk at startup, changes take effect on the next application restart.

**Example: Making the shallow researcher more concise**

Open `src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2` and modify the citation rules section:

```jinja
{#- 5. CITATION & FORMATTING -#}
## Citation Rules (STRICT)
Every claim must end with [1]. You MUST include a `**References:**` section.
- **Format**: `- [N] Title/Filename - URL/Citation`
- **Brevity**: Keep answers under 500 words unless the query explicitly asks for detail.
```

### Key Sections to Customize

Each template has well-defined sections you can target:

- **Intent Classifier** (`intent_classification.j2`) — Classification rules, depth determination, meta response style, output JSON schema
- **Shallow Researcher** (`researcher.j2`) — Source hierarchy, research rules, citation format, response formatting
- **Deep Research Orchestrator** (`orchestrator.j2`) — Ordered stage handoffs, bounded batch dispatch, retry rules, and writer delegation
- **Source Router** (`source_router.j2`) — Advisory source-domain selection and planner guidance
- **Deep Research Planner** (`planner.j2`) — Answer-shape analysis, plan grounding, and structured query generation
- **Researcher Worker** (`researcher.j2`) — Single-query research protocol, source preferences, tool-call batching, and structured notes
- **Report Writer** (`writer.j2`) — Final synthesis, source use, citation format, and `/shared/output.md` contract
- **Source Registry** (`source_registry.j2`) — Formatting for the captured-source list injected by middleware
- **Clarifier** (`research_clarification.j2`) — What counts as "sufficiently specified", question style, multi-turn policy

### Creating a New Template

To create a new prompt template for a custom or modified agent:

**Step 1: Create the file**

```bash
touch src/aiq_agent/agents/my_agent/prompts/system.j2
```

**Step 2: Write the template**

A well-structured prompt template has clearly defined sections:

```jinja
{#- 1. CONTEXT -#}
Current date and time: {{ current_datetime }}

You are a specialized research agent.

{#- 2. TOOLS -#}
## Available Tools
{% if tools %}
{% for tool in tools %}- **{{ tool.name }}**: {{ tool.description }}
{% endfor %}
{% else %}
**No tools available.**
{% endif %}

{#- 3. INSTRUCTIONS -#}
## Instructions
- Research the user's question thoroughly using the available tools.
- Always cite your sources with numbered references.
- If no results are found, state this clearly.

{#- 4. OUTPUT FORMAT -#}
## Response Format
Provide your answer with inline citations [1], [2], etc.
End with a **References:** section listing all sources.
```

**Step 3: Load it in your agent**

```python
class MyAgent:
    def __init__(self, ...):
        self.system_prompt = load_prompt(AGENT_DIR / "prompts", "system")

    async def run(self, query: str) -> str:
        rendered = render_prompt_template(
            self.system_prompt,
            tools=self._build_tools_info(),
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        # Use rendered as SystemMessage content
```

### Multi-Prompt Agents

Some agents use multiple prompts for different roles. The deep researcher loads five agent templates:

```python
prompt_names = ["planner", "researcher", "orchestrator", "writer", "source_router"]
prompts = {name: load_prompt(AGENT_DIR / "prompts", name) for name in prompt_names}
```

`source_registry.j2` is loaded separately by source-registry middleware. Each prompt is rendered with role-specific context, enabling independent model configuration and bounded tool access.

## Jinja2 Patterns

The templates use several Jinja2 patterns worth understanding before editing:

### Tool Detection

Templates detect available tool categories using namespace variables. The `namespace()` pattern is necessary because Jinja2's scoping rules prevent setting variables inside `for` loops that persist outside the loop.

```jinja
{% set ns = namespace(has_web=false, has_paper=false) %}
{% for tool in tools %}
  {% set t_name = tool.name.lower() %}
  {% if 'web' in t_name or 'tavily' in t_name %}
    {% set ns.has_web = true %}
  {% endif %}
{% endfor %}
```

### Conditional Rendering

Templates adapt based on which tools are available:

```jinja
{% if ns.has_web %}
4. **Web Search**: Use for general facts, news, or when other sources are unavailable.
{% endif %}
```

For advanced Jinja2 patterns (source hierarchy, default values, whitespace control), refer to the existing templates in `src/aiq_agent/agents/*/prompts/`.

## Testing Templates

Set the `DEBUG_PROMPTS` environment variable to log prompt-rendering metadata:

```bash
DEBUG_PROMPTS=1 .venv/bin/nat run --config_file configs/my_config.yml --input "test query"
```

This logs the rendered prompt size and a stable correlation reference. It does not log the rendered prompt or
user-provided content. Validate exact variable substitution and conditional rendering with unit tests that use
synthetic, non-sensitive inputs.

## Best Practices

1. **Test changes incrementally.** Modify one section at a time and verify the output before changing more. Run the application with `--input "your test query"` for quick iteration.

2. **Preserve the output format.** Many agents parse the LLM response as JSON (intent classifier, planner, clarifier). If you modify the output schema section, update the corresponding Python parser.

3. **Keep tool detection logic intact.** The `{% set ns = namespace(...) %}` blocks enable templates to adapt to different tool configurations. Removing them can cause incorrect instructions when tools are added or removed.

4. **Use Jinja2 comments for documentation.** Add `{#- ... -#}` comments to explain non-obvious prompt engineering decisions for future maintainers.

5. **Watch token budgets.** Longer system prompts consume tokens from the model's context window. This matters most for the deep research orchestrator, which needs context space for sub-agent reports.

6. **Match the existing style.** The templates use a consistent Markdown structure with headers, bold text, and numbered lists. Following this pattern helps the LLM parse instructions reliably.

7. **Test with multiple models.** Different LLMs interpret prompt instructions differently. If you switch models (refer to [Swapping Models](./swapping-models.md)), verify that the prompts still produce the expected behavior.

## Related

- [Configuration Reference](./configuration-reference.md) -- Full YAML config schema
- [Swapping Models](./swapping-models.md) -- Change which LLMs agents use
