<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Deep Researcher Agent Blueprint

An NVIDIA blueprint for AI-powered deep research, built on the NeMo Agent Toolkit. Deep Researcher Agent is a deployable research
backend that teams can self-host and connect to deployment-owned models, data, authentication, policy controls,
storage, and observability. It combines intelligent query routing, multi-agent research pipelines, and pluggable
knowledge retrieval to deliver citation-backed answers. It is focused on governed research workflows rather than
general-purpose coding-agent execution.

## Where to Start

**First time here?** Follow the [Developer Guide](./get-started/developer-guide.md) — it walks you through the docs in the recommended order.

| I want to... | Go to |
|--------------|-------|
| Review recent changes | [Changelog](./resources/changelog.md) |
| Read the docs in order | [Developer Guide](./get-started/developer-guide.md) |
| Run the blueprint quickly | [Quick Start](./get-started/quick-start.md) |
| Understand how it works | [Architecture Overview](./architecture/overview.md) |
| Customize models, prompts, or tools | [Customization](./customization/index.md) |
| Add new tools or integrations | [Extending](./extending/index.md) |
| Deploy to production | [Deployment](./deployment/index.md) |
| Evaluate quality | [Evaluation](./evaluation/index.md) |

```{toctree}
:hidden:
:caption: Get Started

Overview <./get-started/index.md>
Installation <./get-started/installation.md>
Quick Start <./get-started/quick-start.md>
Developer Guide <./get-started/developer-guide.md>
```

```{toctree}
:hidden:
:caption: Architecture

Overview <./architecture/index.md>
Architecture <./architecture/overview.md>
Agents <./architecture/agents/index.md>
Data Flow <./architecture/data-flow.md>
Deep Research Sandbox <./architecture/agents/sandbox.md>
```

```{toctree}
:hidden:
:caption: Customization

Overview <./customization/index.md>
Configuration Reference <./customization/configuration-reference.md>
Swapping Models <./customization/swapping-models.md>
Tools and Sources <./customization/tools-and-sources.md>
You.com API Suite <./customization/you-com.md>
MCP Tools <./customization/mcp-tools.md>
Guardrails <./customization/guardrails.md>
Knowledge Layer <./customization/knowledge-layer.md>
Prompts <./customization/prompts.md>
Human-in-the-Loop <./customization/hitl.md>
```

```{toctree}
:hidden:
:caption: Extending

Plugin System <./extending/index.md>
Adding a Tool <./extending/adding-a-tool.md>
Adding a Data Source <./extending/adding-a-data-source.md>
```

```{toctree}
:hidden:
:caption: Integration

Overview <./integration/index.md>
Agent Skills <./integration/agent-skills.md>
REST API <./integration/rest-api.md>
MCP Server <./integration/mcp-server.md>
```

```{toctree}
:hidden:
:caption: Evaluation

Overview <./evaluation/index.md>
Benchmarks <./evaluation/benchmarks/index.md>
```

```{toctree}
:hidden:
:caption: Profiling

Profiling & Cost Analysis <./profiling/index.md>
```

```{toctree}
:hidden:
:caption: Deployment

Overview <./deployment/index.md>
Docker Compose <./deployment/docker-compose.md>
Docker Build System <./deployment/docker-build.md>
Authentication <./deployment/authentication.md>
Async Job Content Encryption <./deployment/content-encryption.md>
Observability <./deployment/observability.md>
Production <./deployment/production.md>
OpenShell <./deployment/openshell.md>
Kubernetes <./deployment/kubernetes.md>
Amazon OpenSearch Serverless <./deployment/aws-opensearch-serverless.md>
```

```{toctree}
:hidden:
:caption: Contributing

Overview <./contributing/index.md>
./contributing/development-setup.md
./contributing/code-organization.md
./contributing/code-style.md
./contributing/pr-workflow.md
./contributing/testing.md
```

```{toctree}
:hidden:
:caption: Reference

Knowledge Layer SDK <./reference/knowledge-layer-sdk.md>
```

```{toctree}
:hidden:
:caption: Resources

Changelog <./resources/changelog.md>
Troubleshooting <./resources/troubleshooting.md>
FAQ <./resources/faq.md>
```

```{toctree}
:hidden:
:caption: Examples

./examples/index.md
./examples/minimal-shallow-only.md
./examples/full-pipeline-llamaindex.md
./examples/full-pipeline-web.md
./examples/azure-ai-search.md
./examples/cli-with-local-nims.md
./examples/hybrid-frontier-model.md
./examples/skills-sandbox/index.md
```
