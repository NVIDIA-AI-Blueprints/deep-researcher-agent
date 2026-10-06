<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Skills for Coding Harnesses

Deep Researcher Agent includes portable Agent Skills for coding harnesses that support skill-style instructions and helper scripts.

## Two kinds of Deep Researcher Agent skill

Deep Researcher Agent ships two distinct skill sets, separated by audience. This page documents
the **API-consumer** skills. The maintainer skills are documented in their own
[README](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/blob/develop/.agents/skills/README.md).

| | API-consumer skills | Maintainer skills |
| :-- | :-- | :-- |
| **Audience** | Users calling a running Deep Researcher Agent server | Developers changing the Deep Researcher Agent repo |
| **Location** | top-level `skills/` | `.agents/skills/` |
| **Examples** | `deep-researcher-deploy`, `deep-researcher-research` | `deep-researcher-add-data-source`, `deep-researcher-add-tool`, `deep-researcher-release-qa`, `deep-researcher-prepare-pr`, `deep-researcher-customize-prompts-models`, `deep-researcher-maintain-ci` |
| **Assumes** | A reachable Deep Researcher Agent backend | A repo checkout and dev toolchain |

The API-consumer skills are:

- `deep-researcher-deploy` helps an assistant clone or locate Deep Researcher Agent, choose an existing workflow config, deploy locally or in self-hosted environments, verify basic system health, optionally run deep research completion validation, troubleshoot, rebuild, and stop services.
- `deep-researcher-research` lets an assistant call a running local or self-hosted Deep Researcher Agent Blueprint server for routed `/chat` requests and async deep research job lifecycle operations.

The canonical packaged consumer skills live at:

```text
skills/deep-researcher-deploy/
skills/deep-researcher-research/
```

Each installed skill directory must contain `SKILL.md` at its root. The deploy skill keeps detailed guidance under `references/` so agents only load the path-specific material they need.

For harnesses that expect repository-local Agent Skills under `.agents/skills`,
this repository surfaces the consumer skills there with per-skill symlinks
(`.agents/skills/` itself is the maintainer skill home, not a symlink to
`skills/`):

```text
.agents/skills/deep-researcher-deploy -> ../../skills/deep-researcher-deploy
.agents/skills/deep-researcher-research -> ../../skills/deep-researcher-research
```

## Recommended Flow

Use the skills together rather than blending their responsibilities:

1. Use `deep-researcher-research` for research-shaped requests such as "deep research", "Deep Researcher Agent research", "research", or "use Deep Researcher Agent to answer". It checks `DEEP_RESEARCHER_SERVER_URL` first, then the default local backend.
2. If no backend is reachable, let `deep-researcher-research` ask whether the user already has an Deep Researcher Agent backend URL or wants `deep-researcher-deploy` to start and validate a local Skill backend.
3. Use `deep-researcher-deploy` directly for install/deploy/setup requests such as "install Deep Researcher Agent", "deploy Deep Researcher Agent", or "install deep research".
4. If the user asks which workflow config to use, let `deep-researcher-deploy` read `references/configs.md` and choose an existing repository config before deployment.
5. Use `deep-researcher-deploy` validation checks to confirm the backend and async-agent API are reachable. Confirm the UI only when that deployment mode intentionally starts it.
6. Hand the verified `DEEP_RESEARCHER_SERVER_URL` to `deep-researcher-research`.
7. Use `deep-researcher-research` for routed chat, async research, polling, report retrieval, streaming, and cancellation.
8. After deployment validation, ask whether the user wants to run optional deep research completion validation now or skip validation and try Deep Researcher Agent themselves.
9. Use `deep-researcher-deploy` deep research completion validation only when the user confirms, asks for release signoff, or wants proof that deep research can complete after deployment.

For local non-container use, the deploy skill should prefer the backend-only Agent Skill entry point:

```bash
./scripts/start_as_skill.sh --config_file configs/config_web_default_llamaindex.yml --port 8000
```

This starts the Deep Researcher Agent API backend required by `deep-researcher-research` without starting the browser UI.

## Report Follow-Up and Portable Outputs

The `deep-researcher-research` helper exposes the completed-report and durable-artifact operations as
public commands:

```bash
python3 $SKILL_DIR/scripts/deep_researcher.py report_edit <JOB_ID> "<EDIT_INSTRUCTIONS>"
python3 $SKILL_DIR/scripts/deep_researcher.py report <JOB_ID> --out-dir ./my-report
python3 $SKILL_DIR/scripts/deep_researcher.py artifacts <JOB_ID> --download-dir ./deep-researcher-artifacts
```

`report_edit` submits a child job for a cosmetic rewrite and polls it to completion; the
parent report remains unchanged. `report --out-dir` writes `report.md` plus an `artifacts/`
directory, downloads the job's durable artifacts, and rewrites embedded `artifact://`
image references to local files. `artifacts --download-dir` downloads the artifacts into
the requested directory and prints their local paths; omit `--download-dir` to list the
artifact metadata without downloading bytes.

## Example Invocations

After the skills are installed, users can ask their coding harness for Deep Researcher Agent actions in natural language. Research-shaped prompts route to `deep-researcher-research`; install, deploy, run, stop, UI, CLI, Docker, Helm, and troubleshooting prompts route to `deep-researcher-deploy`:

| User Prompt | Expected Route |
|---|---|
| "deep research on the Blackwell launch" | `deep-researcher-research` checks `DEEP_RESEARCHER_SERVER_URL` or the default local Skill backend, then uses routed `/chat` and async polling as needed. |
| "Deep Researcher Agent research this topic" | `deep-researcher-research` treats the request as research intent, not install intent. |
| "deploy Deep Researcher Agent" | `deep-researcher-deploy` asks which deployment mode the user wants, then validates the selected path and returns `DEEP_RESEARCHER_SERVER_URL`. |
| "install deep research" | `deep-researcher-deploy` asks which Deep Researcher Agent deployment mode the user wants before starting services. |
| "clone Deep Researcher Agent and run it" | `deep-researcher-deploy` locates or clones `NVIDIA-AI-Blueprints/deep-researcher-agent`, checks required environment values, then starts the selected default deployment. |
| "start the Deep Researcher Agent UI" | `deep-researcher-deploy` starts a deployment mode that includes the browser UI, such as local E2E or full Docker Compose. |
| "run Deep Researcher Agent with Docker Compose" | `deep-researcher-deploy` follows the Docker Compose path. For Agent Skill backend use, it should start `deep-researcher-agent` and dependencies without the frontend unless the user asks for UI. |
| "deploy Deep Researcher Agent with Helm" | `deep-researcher-deploy` follows the Kubernetes/Helm path and requires the user to provide or confirm cluster, namespace, registry, secret, ingress, and storage choices. |
| "which Deep Researcher Agent config should I use?" | `deep-researcher-deploy` reads `references/configs.md`, explains the existing configs, and selects a documented config path before deployment. |
| "check why Deep Researcher Agent is unhealthy" | `deep-researcher-deploy` runs health checks, inspects logs/status, and uses the troubleshooting reference for the active deployment mode. |
| "stop Deep Researcher Agent" | `deep-researcher-deploy` follows the shutdown path and asks before destructive cleanup such as deleting Docker volumes. |

## Prerequisites

- Python 3.11, 3.12, or 3.13.
- For `deep-researcher-deploy`: access to this repository or permission to clone `https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent`, plus the selected runtime such as Docker Compose, Node/npm for local web mode, or kubectl/Helm for Kubernetes mode.
- For `deep-researcher-research`: a local or self-hosted Deep Researcher Agent Blueprint server, usually at `http://localhost:8000`. Set `DEEP_RESEARCHER_SERVER_URL` only when using a different local or self-hosted server URL.

## Install From the NVIDIA Skills Catalog

The Deep Researcher Agent repository is the source location for these skills. If you only want to use Deep Researcher Agent as Agent Skills and do not need the full Deep Researcher Agent source checkout, install the Deep Researcher Agent skill set from the [NVIDIA Agent Skills catalog](https://github.com/NVIDIA/skills).

Install the Deep Researcher Agent skills together so deployment and research handoffs are available in the same harness session.

Use the repo-local instructions below when developing Deep Researcher Agent itself, validating changes before publication, or using a harness that does not support the catalog install path.

## Claude Code

Claude Code supports repo-local skills under `.claude/skills/`. This repository
keeps those paths as compatibility symlinks for both skill sets. The consumer
skills point into `skills/`; the maintainer skills point into `.agents/skills/`:

```text
.claude/skills/deep-researcher-deploy -> ../../skills/deep-researcher-deploy
.claude/skills/deep-researcher-research -> ../../skills/deep-researcher-research
.claude/skills/deep-researcher-add-data-source -> ../../.agents/skills/deep-researcher-add-data-source
.claude/skills/deep-researcher-add-tool -> ../../.agents/skills/deep-researcher-add-tool
.claude/skills/deep-researcher-release-qa -> ../../.agents/skills/deep-researcher-release-qa
.claude/skills/deep-researcher-prepare-pr -> ../../.agents/skills/deep-researcher-prepare-pr
.claude/skills/deep-researcher-customize-prompts-models -> ../../.agents/skills/deep-researcher-customize-prompts-models
.claude/skills/deep-researcher-maintain-ci -> ../../.agents/skills/deep-researcher-maintain-ci
```

To recreate the consumer-skill repo-local install manually:

```bash
mkdir -p .claude/skills
ln -s ../../skills/deep-researcher-deploy .claude/skills/deep-researcher-deploy
ln -s ../../skills/deep-researcher-research .claude/skills/deep-researcher-research
```

The maintainer-skill symlinks are managed alongside the maintainer skill set;
refer to the [maintainer skills README](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/blob/develop/.agents/skills/README.md) for how
those are added.

For a user-level install:

```bash
mkdir -p ~/.claude/skills
cp -R skills/deep-researcher-deploy ~/.claude/skills/deep-researcher-deploy
cp -R skills/deep-researcher-research ~/.claude/skills/deep-researcher-research
```

## Codex

For Codex or another Agent Skills-compatible tool, install the skill into the runtime's configured skills directory.

Generic install shape:

```text
<codex-skills-dir>/deep-researcher-deploy/SKILL.md
<codex-skills-dir>/deep-researcher-deploy/references/
<codex-skills-dir>/deep-researcher-research/SKILL.md
<codex-skills-dir>/deep-researcher-research/scripts/deep_researcher.py
```

Example:

```bash
mkdir -p <codex-skills-dir>
cp -R skills/deep-researcher-deploy <codex-skills-dir>/deep-researcher-deploy
cp -R skills/deep-researcher-research <codex-skills-dir>/deep-researcher-research
```

Replace `<codex-skills-dir>` with the skills directory configured for your Codex environment.

## OpenCode

OpenCode loads user skills from `~/.config/opencode/skills/`.

Install with:

```bash
mkdir -p ~/.config/opencode/skills
cp -R skills/deep-researcher-deploy ~/.config/opencode/skills/deep-researcher-deploy
cp -R skills/deep-researcher-research ~/.config/opencode/skills/deep-researcher-research
```

Restart OpenCode or start a new session after installation.

## Verify Installation

From the parent directory containing the installed skills, run:

```bash
test -f deep-researcher-deploy/SKILL.md
test -d deep-researcher-deploy/references
python3 deep-researcher-research/scripts/deep_researcher.py --help
```

The help check does not require a running backend and must exit successfully. Expected output starts with:

```text
Usage: deep_researcher.py <command> [args]
```

With an Deep Researcher Agent backend running, verify the public consumer boundary before invoking research:

```bash
python3 deep-researcher-research/scripts/deep_researcher.py health
python3 deep-researcher-research/scripts/deep_researcher.py agents
```
