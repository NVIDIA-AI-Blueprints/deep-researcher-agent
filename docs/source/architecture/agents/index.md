<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agents

AI-Q uses a multi-agent architecture where an intent classifier routes queries to specialized research agents.

| Agent | Purpose | Speed | Depth |
|-------|---------|-------|-------|
| [Intent Classifier](./intent-classifier.md) | Route queries and determine research depth | Instant | — |
| [Clarifier](./clarifier.md) | Optionally gather missing context and the requested output type before deep research | Interactive | — |
| [Shallow Researcher](./shallow-researcher.md) | Fast, bounded research for simple questions | Fast (30-60s) | Surface |
| [Deep Researcher](./deep-researcher.md) | Advisory source routing, structured planning, concurrent evidence gathering, and writer synthesis | Several minutes; can exceed 10 | Deep |
| [Data Science Agent](./data-science-agent.md) | Adaptive analysis over GSF, the AI-Q knowledge layer, and web evidence | Variable | Analytical |

```{toctree}
:titlesonly:

intent-classifier.md
clarifier.md
shallow-researcher.md
deep-researcher.md
data-science-agent.md
sandbox.md
```
