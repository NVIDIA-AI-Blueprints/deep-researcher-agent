<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Guardrails

Deep Researcher Agent can use NeMo Guardrails through NeMo Agent Toolkit middleware to evaluate selected workflow and agent-boundary inputs and outputs. Guardrails can pass content through unchanged, block content with a configured refusal, or modify selected fields before execution continues.

```{important}
`configs/config_web_default_guardrails.yml` is a reference integration, not a universal end-to-end security policy.
Guardrail policies and acceptable false-positive/false-negative tradeoffs are deployment-specific. Operators must adapt
the rails to their requirements and test every exposed route and execution boundary before relying on them as a
security control.
```

Deep Researcher Agent provides middleware types for workflow, shallow-research, and deep-research boundaries. In
`configs/config_web_default_guardrails.yml`, the workflow middleware is explicitly attached under `workflow.middleware`.
The shallow middleware is dynamically attached by targeting `shallow_research_agent` under `workflow_functions`.
Because the async deep-research runner calls its agent directly, it reconstructs the middleware selection from the
`deep_research_agent` target and applies it around that call.

## Guarded Boundaries

| Boundary | Middleware | Applies To |
| --- | --- | --- |
| Workflow | `workflow_guardrails` | Workflow input and final assistant response; active when attached under `workflow.middleware`. |
| Shallow researcher | `shallow_agent_guardrails` | Shallow input or output message content; dynamically attached when `workflow_functions` targets `shallow_research_agent`. |
| Deep researcher | `deep_agent_guardrails` | Deep input or output message content; the async runner selects it when `workflow_functions` targets `deep_research_agent`. |

These middleware types use NAT/NeMo Guardrails for policy evaluation at Deep Researcher Agent workflow and agent boundaries.

## Guardrail Decisions

At each configured boundary, guardrails can make one of three decisions:

| Decision | Behavior |
| --- | --- |
| Pass | Continue with the original input or output. |
| Modify | Replace the selected input or output field with the modified content returned by the rail. |
| Block | Return the configured refusal response instead of continuing with the blocked content. |

Input- and output-rail evaluation exceptions are caught, logged, and converted to the middleware refusal response.
Output failures preserve the intercepted response schema and do not return the original unfiltered output.

Buffered output streams are evaluated as one logical assistant response before any chunk is emitted. This includes
streams that mix raw strings and structured response chunks. Modified output is redistributed across the buffered
chunks, terminal workflow outcomes are synchronized with every rewritten structured chunk, and blocked or failed
streams emit only a safe refusal.

## PII Runtime Dependencies

The built-in `sensitive_data_detection` action requires Presidio, spaCy, and a compatible spaCy language model. These
large dependencies are available through the `pii` project extra rather than the base Python package. Install Deep Researcher Agent with
`--extra pii` when running PII rails. The release Docker image includes this extra and verifies during the build that the
Presidio analyzer and anonymizer import, `en_core_web_lg` is installed, and email analysis succeeds.

## Configuration Shape

The guardrails configuration is placed in the top-level `middleware` section. Defining an entry makes that middleware
available. For named functions, `workflow_functions` dynamically attaches the middleware and selects the fields it
evaluates. For the workflow boundary, list the middleware under `workflow.middleware` and use its `workflow_functions`
block to select fields. A middleware without a matching `workflow_functions` field selection resolves to zero guarded
fields, so that boundary is not enforced. The `guardrails` block uses NAT/NeMo Guardrails configuration. Refer to
`configs/config_web_default_guardrails.yml` for the complete attachment and field-selection example.

```yaml
middleware:
  workflow_guardrails:
    _type: workflow_guardrails
    workflow_functions:
      "<workflow>":
        choices:
          - message.content
    guardrails:
      # NeMo Guardrails configuration.

  shallow_agent_guardrails:
    _type: shallow_agent_guardrails
    workflow_functions:
      shallow_research_agent:
        pre_invoke:
          messages:
            HumanMessage:
              - content
        post_invoke:
          messages:
            AIMessage:
              - content
    guardrails:
      # NeMo Guardrails configuration.

workflow:
  _type: chat_deepresearcher_agent
  middleware:
    - workflow_guardrails
```

The `shallow_agent_guardrails` target above activates shallow enforcement without a separate function-level middleware
list. For async deep research, the worker does not invoke the registered NAT function directly, so the Deep Researcher Agent runner
reconstructs the function middleware chain and selects middleware whose `workflow_functions` includes
`deep_research_agent`.

Do not also list the same middleware under `functions.<agent>.middleware`. Configuring both attachment mechanisms for
the same agent can evaluate the middleware twice on normal function calls, and the async deep-research runner rejects
duplicate middleware names.

## Field Selection

The `workflow_functions` entry names the function that dynamic middleware wraps and defines which string fields
guardrails evaluate and can modify. The async deep-research runner also uses that target to select middleware around its
direct worker call.

For nested response objects, selected fields can be dotted paths:

```yaml
workflow_functions:
  "<workflow>":
    choices:
      - message.content
```

For workflow-level guardrails, this selects `message.content` from each item in the final response `choices`.

Agent-boundary guardrails can also select message fields by message type. This lets the same agent state carry multiple message types while guardrails evaluate only the configured string fields.

```yaml
workflow_functions:
  shallow_research_agent:
    pre_invoke:
      messages:
        HumanMessage:
          - content
    post_invoke:
      messages:
        AIMessage:
          - content
```

In this example:

| Entry | Meaning |
| --- | --- |
| `pre_invoke` | Selects fields evaluated by input rails before the agent runs. |
| `post_invoke` | Selects fields evaluated by output rails after the agent returns. |
| `messages` | Selects the agent state's message list. |
| `HumanMessage` | Applies the listed field paths to user messages in that list. |
| `AIMessage` | Applies the listed field paths to assistant messages in that list. |
| `content` | Evaluates the message text. |

The shallow and deep researcher middleware types support this shape so input rails can evaluate user message content
and output rails can evaluate assistant message content. Only attached or runner-selected middleware is enforced.

## Supported Scope

Guardrails middleware is available at these Deep Researcher Agent boundaries:

- Workflow input
- Workflow output
- Shallow researcher input and output messages
- Deep researcher input and output messages

The reference profile actively guards all three boundaries described above.

This scope applies only to the middleware and fields explicitly attached in the active configuration. It does not
replace authentication, authorization, network controls, secret handling, or deployment-specific policy validation.

For the complete YAML schema and general configuration conventions, refer to [Configuration Reference](./configuration-reference.md).
