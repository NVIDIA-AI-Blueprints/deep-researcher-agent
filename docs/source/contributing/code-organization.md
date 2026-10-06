<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Code Organization

High-level layout (refer to [Architecture Overview](../architecture/overview.md) for component roles):

```
src/deep_researcher_agent/
├── agents/              # Chat researcher, shallow/deep research, clarifier
│   ├── chat_researcher/   # Orchestrator, orchestration node (intent + meta + depth), nodes
│   ├── shallow_researcher/
│   ├── deep_researcher/   # See src/deep_researcher_agent/agents/deep_researcher/README.md
│   └── clarifier/
├── common/              # LLM provider, callbacks, prompt utils, data_sources
├── knowledge/           # Schema, factory, base retriever/ingestor, summary store
├── observability/       # OpenTelemetry header redaction exporter
├── auth/                # Auth utilities
└── fastapi_extensions/  # API route extensions
```

Configs live in `configs/`; refer to the [Customization guide](../customization/index.md) for configuration options. Frontends: `frontends/cli`, `frontends/deep_researcher_api`, `frontends/debug`, `frontends/ui`. Benchmarks: `frontends/benchmarks/`. Data sources and Knowledge Layer: `sources/`.
