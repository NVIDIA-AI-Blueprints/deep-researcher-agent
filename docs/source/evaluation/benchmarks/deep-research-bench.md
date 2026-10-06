<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Deep Research Bench Evaluation of NVIDIA Deep Researcher Agent Blueprint

[DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench/tree/main) is one of the most popular benchmarks for evaluating deep research agents. The benchmark was introduced in [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agent](https://arxiv.org/pdf/2506.11763). It contains 100 research  tasks (50 English, 50 Chinese) from 22 domains. It proposed 2 different evaluation metrics: RACE and FACT to assess the quality of the research reports.

- RACE: measures report generation quality across 4 dimensions
    - Comprehensiveness
    - Insight
    - Instruction Following
    - Readability
- FACT: evaluates retrieval and citation system using
    - Average Effective Citations: average # of valuable, verifiably supported information an agent retrieves and presents per task.
    - Citation Accuracy: measures the precision of an agent’s citations, reflecting its ability to ground statements with appropriate sources correctly.

## API Keys

```bash
export TAVILY_API_KEY=your_key              # For web search
export SERPER_API_KEY=your_key              # For Google Scholar paper search
export NVIDIA_API_KEY=your_key              # For agent execution (integrate.api.nvidia.com)
export OPENAI_API_KEY=your_key              # For frontier model in config (optional)
```

## Configuration Files

The following table lists the available configuration files:

| Config | Description |
|--------|-------------|
| `frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml` | Default: Nemotron for agent. Generates reports for submission to the official DRB evaluator. |

## Running Evaluation

### Step 1: Install the dataset

The dataset files are not included in the repository. We have included a script to retrieve them from the [Deep Research Bench Github Repository](https://github.com/Ayanami0730/deep_research_bench/tree/main) and format them for the NeMo Agent Toolkit evaluator.

To download the dataset files, run the following script:

```bash
python frontends/benchmarks/deepresearch_bench/scripts/download_drb_dataset.py
```

### Step 2: Generate reports using NAT evaluation harness

```bash
dotenv -f deploy/.env run nat eval --config_file frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml
```

### Step 3: Convert the output into a compatible format
```bash
python frontends/benchmarks/deepresearch_bench/scripts/export_drb_jsonl.py --input <path to your workflow_output.json> --output <path to the output file you want to create with .jsonl extension>
```

### Step 4: Run evaluation
Follow instructions in the [Deep Research Bench Github Repository](https://github.com/Ayanami0730/deep_research_bench/tree/main) to run evaluation and obtain scores.


## Optional: Relay and Phoenix Tracing

Deep Researcher Agent evaluation uses the same NeMo Relay observability path as interactive and
async workflows. ATOF is enabled by default. To visualize the evaluation in
Phoenix, enable the Relay OpenInference OTEL endpoint in the evaluated workflow
and start Phoenix before running `nat eval`.

Start server (separate terminal):

```bash
uvx --from arize-phoenix phoenix serve
```

```yaml
workflow:
  relay:
    observability:
      opentelemetry:
        enabled: true
        endpoints:
          - type: openinference
            endpoint: http://localhost:6006/v1/traces
            resource_attributes:
              openinference.project.name: deep-researcher-deepresearch-bench

eval:
  general:
    workflow_alias: "deep-researcher-deepresearch-v2-baseline"
```

See [Observability with NeMo Relay](../../deployment/observability.md) for ATOF
inspection, trace interpretation, project selection, and cost reporting.

### workflow_alias

The `workflow_alias` parameter provides a workflow-specific identifier for tracking evaluation runs:

| Parameter | Description |
|-----------|-------------|
| `workflow_alias` | Unique identifier for the workflow variant being evaluated. Used to group and compare runs across different configurations, models, or dataset subsets. |
