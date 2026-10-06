<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Deep Research Skills and Sandbox

This example shows how to run Deep Researcher Agent deep research with DeepAgents skills and a provider-backed sandbox. The reference
profile uses Modal; Deep Researcher Agent also includes an experimental, policy-bound OpenShell profile.

Skills let a research agent discover task-specific instructions only when they are relevant. Deep Researcher Agent mounts the assigned
skill definitions read-only from the host. A skill can teach the agent a repeatable workflow, such as extracting numeric
facts, normalizing a table, running calculations, and producing reusable text artifacts. When a skill invokes
`execute`, the generated code runs outside the Deep Researcher Agent process in one provider sandbox per deep-research job. Modal and
OpenShell implement this job-scoped contract.

For more background, refer to the LangChain DeepAgents docs:

- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [DeepAgents skills](https://docs.langchain.com/oss/python/deepagents/skills)

## What This Example Enables

The example config enables:

- built-in DeepAgents skills from `src/deep_researcher_agent/agents/deep_researcher/skills/`
- a fresh per-job Modal sandbox for Python execution
- Python packages useful for analysis, including `pandas`, `numpy`, `matplotlib`, and `pillow`
- virtual `/shared/` files for text artifacts that the orchestrator and subagents can read during the report workflow
- durable capture of supported charts and data files for async API jobs

The built-in collections currently expose these role-oriented skills:

| Collection | Default assignment | Skills |
| ---------- | ------------------ | ------ |
| `research` | `researcher-agent` | `data-table-analysis`, `forecast-analysis`, `lightweight-calculation` |
| `synthesis` | `writer-agent` | `long-form-report-writer`, `prediction-report-writer` |
| `visualization` | `writer-agent` | `chart-generation` |

The assignment is configurable. Skill definitions stay host-side and read-only;
only workflows that invoke `execute` require a sandbox.

**Models and report quality:** For clearer tables, stronger reasoning over numbers, and more reliable use of the data-table-analysis skill end-to-end, prefer **frontier-class models** for the orchestrator, planner, and researcher in your config ([Swapping models](../../customization/swapping-models.md)). Smaller or faster models may complete runs but often produce weaker structured outputs and more formatting mistakes in long reports.

## Prerequisites

Install and configure Deep Researcher Agent as usual, then make sure these credentials are available to the process running Deep Researcher Agent:

```bash
export NVIDIA_API_KEY="nvapi-..."              # pragma: allowlist secret
export TAVILY_API_KEY="tvly-..."               # pragma: allowlist secret
```

For sandbox execution, create a Modal account and configure Modal credentials. Modal uses a token ID and token secret:

```bash
export MODAL_TOKEN_ID="ak-..."                 # pragma: allowlist secret
export MODAL_TOKEN_SECRET="as-..."             # pragma: allowlist secret
```

You can also configure Modal locally with:

```bash
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

Refer to Modal's token configuration docs for details: [modal.config](https://modal.com/docs/reference/modal.config).

## Configuration

Use `configs/config_domain_routing_and_skills.yml`. The relevant section is:

```yaml
functions:
  deep_research_skills:
    _type: deep_research_skills
    agents:
      researcher-agent:
        - research
      writer-agent:
        - synthesis
        - visualization
    require_sandbox:
      - research

  deep_research_sandbox:
    _type: deep_research_sandbox
    provider: modal
    app_name: deep-researcher-deep-research
    image: python:3.13-slim
    packages:
      - matplotlib
      - numpy
      - pandas
      - pillow
    network: blocked
    artifact_capture:
      enabled: true
      max_file_bytes: 50000000
      allow_extensions: [.png, .jpg, .jpeg, .webp, .csv, .json, .md, .ipynb, .pdf]

  deep_research_agent:
    _type: deep_research_agent
    enable_citation_verification: false
    skills: deep_research_skills
    sandbox: deep_research_sandbox
```

Deep Researcher Agent validates the public skill collection names (`research`, `synthesis`, `visualization`) and resolves them to DeepAgents source paths internally. When skills are configured, Deep Researcher Agent mounts the configured built-in skill collections into the DeepAgents virtual filesystem. When the sandbox ref is present, DeepAgents `execute` calls run in the configured provider. Modal creates a fresh sandbox named for the job.

In the reference async API flow, artifact capture uses the job database configured by
`general.front_end.db_url` (`NAT_JOB_STORE_DB_URL`) for metadata. Artifact bytes use SQL BLOB storage in the job database
by default. For production, use S3-compatible object storage by setting `DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER=s3`,
`DEEP_RESEARCHER_ARTIFACT_S3_BUCKET`, and the standard AWS credentials; set `DEEP_RESEARCHER_ARTIFACT_S3_ENDPOINT_URL` for MinIO or another
compatible service. See [Production Artifact Storage](../../deployment/production.md#artifact-storage) for all options.

To evaluate OpenShell instead, use `configs/config_openshell.yml` after running
`scripts/openshell/setup_openshell.sh`. That profile creates one policy-bound
sandbox per job, verifies the effective policy and revision before use, and
deletes the sandbox at terminal cleanup. Attaching to an existing shared
sandbox is available only through explicit debug settings and is not
job-isolated.

## Run Synchronously with `nat run` (Non-Persistent)

```{warning}
`nat run` is a synchronous, single-run command. It does not create an async job record or connect the workflow to the
job-scoped artifact store. The final report is returned normally, and `/shared/` files can contribute to that report
during the run, but run state, `/shared/` content, and sandbox-generated files cannot be retrieved through the job or
artifact APIs after the command finishes.
```

```bash
dotenv -f deploy/.env run .venv/bin/nat run \
  --config_file configs/config_domain_routing_and_skills.yml \
  --input "Compare the top 10 publicly traded semiconductor companies by 2024 revenue. Build a markdown table with revenue, YoY growth, market cap, and gross margin. Then rank them and compute summary statistics. Use the data analysis tool for all calculations."
```

Use this mode to try the workflow when you only need its returned report. Do not use it when you need durable job state
or separately retrievable charts, CSVs, notebooks, or other generated files.

## Run with `nat serve` for Persistent Jobs and Artifacts

To retain job information and retrieve supported generated files after a run, start the async API with `nat serve`.
The configured `NAT_JOB_STORE_DB_URL` supplies the required job-scoped store. The reference config defaults to a local
SQLite database; production deployments should configure PostgreSQL and appropriate artifact blob storage.
For trusted local development, set `REQUIRE_AUTH=false` in `deploy/.env`; the commands below omit credentials on that
basis. When `REQUIRE_AUTH=true`, these job routes require authentication, so configure authentication and add the same
`Authorization: Bearer $DEEP_RESEARCHER_TOKEN` header to every `curl` command below.

```bash
dotenv -f deploy/.env run .venv/bin/nat serve \
  --config_file configs/config_domain_routing_and_skills.yml \
  --host 0.0.0.0 \
  --port 8000
```

In another terminal, submit a deep research request with a known job ID. Custom job IDs must be unique, so change this
value before repeating the example against the same job store:

```bash
curl -X POST http://localhost:8000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "skills-sandbox-example",
    "agent_type": "deep_researcher",
    "input": "Compare the top 10 publicly traded semiconductor companies by 2024 revenue. Build a markdown table and a CSV with revenue, YoY growth, market cap, and gross margin. Then rank them and compute summary statistics. Use the data analysis tool for all calculations."
  }'
```

Check the job until its status is either `success` or `failure`. If it reaches `failure`, inspect the response's `error`
field for the actionable failure message:

```bash
curl http://localhost:8000/v1/jobs/async/job/skills-sandbox-example
```

List its captured artifacts, then use an `artifact_id` from the response to download one:

```bash
curl http://localhost:8000/v1/jobs/async/job/skills-sandbox-example/artifacts
curl -OJ http://localhost:8000/v1/jobs/async/job/skills-sandbox-example/artifacts/{artifact_id}/content
```

Artifact capture is best-effort and limited to the configured file types and size. Stored artifacts are retrievable
rather than permanent: server-wide retention cleanup can remove an artifact independently of the job's expiry. For the
complete API contract, authentication guidance, and retention behavior, see
[Durable Sandbox Artifacts](../../integration/rest-api.md#durable-sandbox-artifacts).

## Example Queries

Use queries that require researched numeric facts plus computed tabular analysis.

**Example prompt:**

```text
Compare the top 10 publicly traded semiconductor companies by 2024 revenue. Build a markdown table with revenue, YoY growth, market cap, and gross margin. Then rank them and compute summary statistics. Use the data analysis tool for all calculations.
```

Additional prompts that exercise the same pattern:

```text
Compare AI infrastructure capex for Microsoft, Google, Meta, and Amazon over the last 8 quarters. Include QoQ and YoY growth.
```

```text
Compare R&D spend across the top 10 semiconductor companies and compute R&D as a percent of revenue.
```

Expected behavior:

1. The planner identifies that a skill should be used for structured quantitative analysis.
2. Researchers gather source-grounded input figures.
3. A matching researcher or writer reads the relevant `SKILL.md`.
4. The agent calls `execute` to run Python/pandas in the configured sandbox provider.
5. The agent writes markdown, CSV, or JSON text artifacts to `/shared/...` with `write_file`.
6. The final report cites the original sources for input figures and labels computed columns as calculations.

## Skill Files

Built-in deep research skills live under:

```text
src/deep_researcher_agent/agents/deep_researcher/skills/
```

Each skill should be a directory with a `SKILL.md` file:

```text
src/deep_researcher_agent/agents/deep_researcher/skills/
`-- research/
    `-- my-skill/
        `-- SKILL.md
```

The required hierarchy is `skills/<collection>/<skill>/SKILL.md`. The
collection directory is the public name assigned to an agent in
`deep_research_skills.agents`.

At minimum, `SKILL.md` needs frontmatter with a stable `name` and a clear `description`:

```markdown
---
name: my-skill
description: >
  Use this skill when the research task requires a specific repeatable workflow.
  Include trigger phrases and expected outputs so the agent can decide when to
  read this skill.
---

# My Skill

## When to Use

Use this skill for ...

## Execution Flow

1. Gather the required inputs.
2. Use the appropriate tools.
3. Write reusable outputs to `/shared/...` when another agent or the final report needs them.
```

Skill descriptions matter because DeepAgents uses the frontmatter description to decide whether the skill applies before reading the full file. Keep descriptions specific, list representative trigger phrases, and explicitly name required tools such as `execute`, `read_file`, or `write_file` when the workflow depends on them.

## Adding More Skills

To add a built-in Deep Researcher Agent deep research skill:

1. Create `src/deep_researcher_agent/agents/deep_researcher/skills/<collection>/<skill>/`.
2. Add a `SKILL.md` file with frontmatter and workflow instructions.
3. Put optional helper scripts, references, or templates inside the same skill directory.
4. Reference any helper files from `SKILL.md` so the agent knows when to read or run them.
5. Keep workflow instructions generic enough to handle variations of the task, but concrete enough to force required tool calls.
6. Run with `configs/config_domain_routing_and_skills.yml` and test a query that should trigger the new skill.

A skill added to a collection that is already assigned to the target agent needs
no config change. For a new collection, add the collection name to the target
agent under `deep_research_skills.agents`. Deep Researcher Agent collects the assigned skill
directories at runtime and exposes them to DeepAgents through an internal
`/skills/` source.

## Notes and Limitations

- The reference config uses a fresh Modal sandbox for code execution. The experimental OpenShell config also creates one
  physical sandbox per job and requires policy attestation plus terminal deletion. Shared attachment is debug-only.
- Text artifacts that need to survive for the report should be written through DeepAgents filesystem tools to `/shared/...`.
- `/shared/` is a virtual DeepAgents filesystem path. Use `ls`, `read_file`, `write_file`, and `edit_file` for `/shared/`; do not inspect `/shared/` with shell commands through `execute`.
- The sandbox is configured with `network: blocked`, so research should happen through Deep Researcher Agent search tools, not from sandbox code.
- The reference profile enables durable sandbox artifact capture for async API jobs. Successful `execute` calls
  checkpoint manifest-declared files, and success/failure terminal paths perform one final best-effort scan. A busy
  cancellation skips that scan and preserves earlier checkpoints. Adding a sandbox alone does not guarantee that every
  generated file is persisted or embedded in the report.
