<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Developer Guide

New to Deep Researcher Agent? This page walks you through the
documentation in the order that will get you productive fastest.

## 1. Install

Set up Python, install dependencies with `uv`, and configure your environment
variables (primarily `NVIDIA_API_KEY`).

**Read:** [Installation](./installation.md)

## 2. Run the Agent

Launch the CLI and submit your first research query. This gives you a working
mental model of what the system does before you look at how it works.

**Read:** [Quick Start](./quick-start.md)

## 3. Understand the Architecture

Learn the two-path design — an intent classifier routes queries to either the
fast shallow researcher or the multi-phase deep researcher — and how data
flows through the system.

**Read:** [Architecture Overview](../architecture/overview.md) then
[Data Flow](../architecture/data-flow.md)

## 4. Explore Individual Agents

Each agent has its own page covering state models, configuration, prompt
templates, and internal flow diagrams.

- [Intent Classifier](../architecture/agents/intent-classifier.md) — Query routing
- [Shallow Researcher](../architecture/agents/shallow-researcher.md) — Fast, bounded tool-calling
- [Deep Researcher](../architecture/agents/deep-researcher.md) — Multi-phase subagent workflow
- [Clarifier](../architecture/agents/clarifier.md) — Human-in-the-loop before deep research

## 5. Customize and Extend

Once you understand the agents, learn how to tailor the system to your needs:

- [Swap LLMs](../customization/swapping-models.md) — Use different models for different roles
- [Enable or disable tools](../customization/tools-and-sources.md) — Configure which data sources agents can access
- [Edit prompts](../customization/prompts.md) — Modify agent behavior through Jinja2 templates
- [Add a new tool](../extending/adding-a-tool.md) — Integrate a new search API or data source
- [Configuration reference](../customization/configuration-reference.md) — Full YAML config guide

## 6. Deploy

Move from local development to Docker Compose.

**Read:** [Docker Compose](../deployment/docker-compose.md) then
[Production](../deployment/production.md)
