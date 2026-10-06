---
name: deep-researcher-deploy
description: |
  Use when asked to install, deploy, run, validate, troubleshoot, or stop NVIDIA Deep Researcher Agent Blueprint infrastructure.
license: Apache-2.0
compatibility: |
  Designed for Claude Code, OpenCode, Codex, and Agent Skills-compatible tools. Requires Git, network
  access to GitHub, and one selected runtime path: Docker Compose v2 for the default local deployment,
  Python 3.11+ and uv for local process or CLI mode, Node.js 20+ and npm for local web UI mode, or
  kubectl 1.28+ and Helm 3.12+ for Kubernetes and Helm mode.
metadata:
  version: "2.2.0"
  author: "NVIDIA Deep Researcher Agent Blueprint Team <deep-researcher-blueprint@nvidia.com>"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags:
    - nvidia
    - deep-researcher
    - blueprint
    - deploy
    - operations
    - agent-skills
allowed-tools: Read Bash
---

# Deep Researcher Agent Deploy Skill

## When to Use This Skill

Use this skill to get a local or self-hosted NVIDIA Deep Researcher Agent Blueprint server running and verified for use by
`deep-researcher-research`.

This skill owns setup, deployment, operational checks, troubleshooting, and shutdown. It does not run deep
research itself. After deployment is healthy, hand off the verified server URL to `deep-researcher-research`.
The workflow stays explicit so deployment validation and handoff are repeatable across supported agent clients.

## Prerequisites

Users need:

- Access to clone or update `https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent`.
- Git available in the shell.
- One deployment runtime:
  - Docker Engine with Docker Compose v2 for the default durable local deployment.
  - Python 3.11+ and `uv` for local process or CLI mode.
  - Node.js 20+ and `npm` for local browser UI development mode.
  - `kubectl` 1.28+, Helm 3.12+, and access to a Kubernetes cluster for Helm mode.
- Network access to GitHub, NVIDIA-hosted model endpoints, and any selected search provider.
- Credentials stored outside chat. Hosted-model usage requires `NVIDIA_API_KEY`; web research requires at least
  one supported search provider key such as `TAVILY_API_KEY`, `SERPER_API_KEY`, or `EXA_API_KEY`.
- System capacity for the selected runtime. Docker Compose mode starts the Deep Researcher Agent backend and PostgreSQL by default;
  browser UI mode also uses frontend port `3000`. Self-hosted model or RAG deployments may require GPU resources.

Before writing secrets, verify `deploy/.env` is ignored:

```bash
git check-ignore deploy/.env
```

Expected output: `deploy/.env` or a matching ignore rule. If it is not ignored, stop and fix the ignore rule before
placing credentials in the file.

## Workflow

1. Locate or clone the Deep Researcher Agent repository.
2. Confirm the expected repository files exist.
3. Select the deployment mode.
4. Prepare `deploy/.env` without overwriting user secrets.
5. Check runtime prerequisites for the selected path.
6. Start the selected deployment.
7. Run basic validation.
8. Report the verified `DEEP_RESEARCHER_SERVER_URL` for `deep-researcher-research`.
9. Ask whether to run optional deep research completion validation.

### Step 1 - Locate or clone Deep Researcher Agent

If no Deep Researcher Agent checkout exists, read `references/locate-or-clone.md` before cloning. In an existing checkout, confirm the
required files:

```bash
pwd
test -f pyproject.toml
test -f deploy/.env.example
test -d configs
```

Expected output: `pwd` prints the Deep Researcher Agent repository path; the `test` commands exit with status 0 and no output.

### Step 2 - Select the deployment mode

If the user asks to install, deploy, set up, or run Deep Researcher Agent without naming a mode, ask:

```text
How do you want to run Deep Researcher Agent?

1. Skill backend - backend-only service for deep-researcher-research w/o browser UI.
2. CLI - interactive terminal Deep Researcher Agent.
3. UI - browser Deep Researcher Agent app with backend and frontend.
4. Custom - choose an existing Deep Researcher Agent config or review advanced customization docs before deployment.
```

Wait for the user's answer before starting services.

Do not ask this question when the user already specified a mode, such as Docker Compose, Helm, UI, CLI, or Agent Skill
backend. Do not ask the full mode question when `deep-researcher-research` routed here because a deep research request needs a
backend. In that case, prefer Agent Skill backend and ask only for permission to start it if needed.

### Step 3 - Prepare environment and secrets

Read `references/env-and-secrets.md` before changing `deploy/.env`.

```bash
if [ ! -f deploy/.env ]; then
  cp deploy/.env.example deploy/.env
  echo "created deploy/.env from deploy/.env.example"
fi
```

Expected output when the file is missing: `created deploy/.env from deploy/.env.example`. Expected output when the file
already exists: no output, and the existing file is preserved.

Never print secret values. If credentials are missing, ask the user to update `deploy/.env`; do not ask them to paste
secret values into chat.

### Step 4 - Route to the selected deployment path

Match the user request, then read the referenced file before acting:

| User Intent | Reference |
|---|---|
| No Deep Researcher Agent checkout exists, install Deep Researcher Agent, clone Deep Researcher Agent, locate repo | `references/locate-or-clone.md` |
| Configure environment, check API keys, inspect `.env` | `references/env-and-secrets.md` |
| Choose an Deep Researcher Agent workflow config, understand config files, set `BACKEND_CONFIG` or `CONFIG_FILE` | `references/configs.md` |
| Backend-only local server for `deep-researcher-research`, Deep Researcher Agent as an Agent Skill | `references/skill-backend.md` |
| Terminal assistant, CLI-only run, no web UI | `references/terminal-cli.md` |
| Quick local development run, start UI/backend without containers | `references/local-web.md` |
| Default durable local deployment, Docker Compose, containers, PostgreSQL | `references/docker-compose.md` |
| Kubernetes, Helm, cluster deployment | `references/kubernetes-helm.md` |
| Foundational RAG / FRAG integration | `references/frag.md` |
| Basic health checks, shallow smoke checks, handoff to `deep-researcher-research` | `references/validation.md` |
| Optional deep research completion validation | `references/end-to-end-validation.md` |
| Logs, unhealthy services, port conflicts, config failures | `references/troubleshooting.md` |
| Stop services, restart, rebuild, safe cleanup | `references/shutdown.md` |

### Step 5 - Validate and hand off

After startup, read `references/validation.md` and run the appropriate checks for the selected mode. For the default
local backend, verify health:

```bash
curl -sf http://localhost:8000/health
```

Expected output: a successful JSON health response or an empty successful response depending on the server build. If the
command fails, read `references/troubleshooting.md` and diagnose before claiming the backend is ready.

`deep-researcher-research` needs a reachable Deep Researcher Agent server URL. If the backend is on the default port, no extra configuration is
needed:

```bash
DEEP_RESEARCHER_SERVER_URL=http://localhost:8000
```

If the backend runs elsewhere, tell the user to set:

```bash
export DEEP_RESEARCHER_SERVER_URL="http://localhost:<PORT>"
```

Do not continue into deep research or deep research completion validation unless the user asks for it or confirms the
post-deploy validation prompt. This skill's success criterion is a deployed and basically validated server, not report
generation quality.

## Version Compatibility

**IMPORTANT:** This skill is designed for NVIDIA Deep Researcher Agent Blueprint version 2.2.0.

Semantic Versioning Compatibility Rules:

```text
Skill version: X.Y.Z
Blueprint version: A.B.C

Compatible IF:
1. A == X (Major versions MUST match)
2. B >= Y (Minor version must be equal or greater)
3. C can be anything (Patch version does not affect compatibility)
```

Examples:

- Skill version 2.2.0 is compatible with Blueprint version 2.2.0.
- Skill version 2.2.0 is compatible with Blueprint version 2.3.0.
- Skill version 2.2.0 is compatible with Blueprint version 2.2.5.
- Skill version 2.2.0 is not compatible with Blueprint version 3.0.0.
- Skill version 2.2.0 is not compatible with Blueprint version 2.1.0.

If your Blueprint version is not compatible:

1. Check for an updated skill version matching your Blueprint version.
2. Use a Blueprint version compatible with this skill.
3. Proceed with caution only when the user accepts the compatibility risk; deployment commands or config names may have
   changed.

## Security Best Practices

- Never print secret values. Check only whether required environment variables are set.
- Store credentials in `deploy/.env` or environment variables, not in chat transcripts, shell history, committed files,
  or example commands.
- Do not overwrite `deploy/.env` when it already exists.
- Ask before destructive cleanup such as deleting Docker volumes with `down -v`.
- Do not claim FRAG is ready unless both `RAG_SERVER_URL` and `RAG_INGEST_URL` are configured and reachable.
- Run verification commands yourself when possible.

## Limitations

- This skill prepares and validates Deep Researcher Agent infrastructure; it does not judge deep research report quality.
- It cannot provide or inspect secret values. Users must configure credentials outside chat.
- Helm, FRAG, custom config, and self-hosted model paths depend on infrastructure the user controls.
- Destructive cleanup, such as deleting Docker volumes, requires explicit user approval.

## Examples

### Example 1: Deploy a backend-only Skill server with Docker Compose

```bash
test -f deploy/.env || cp deploy/.env.example deploy/.env
git check-ignore deploy/.env
cd deploy/compose
BUILD_TARGET=release docker compose --env-file ../.env -f docker-compose.yaml config --quiet
BUILD_TARGET=release docker compose --env-file ../.env -f docker-compose.yaml up -d --build deep-researcher-agent
curl -sf http://localhost:8000/health
```

Expected output:

```text
deploy/.env
<docker compose starts deep-researcher-agent and dependencies>
<health endpoint returns a successful response>
```

If Docker, ports, credentials, or health checks fail, read `references/troubleshooting.md` before retrying.

### Example 2: Hand off a non-default backend URL to deep-researcher-research

```bash
export DEEP_RESEARCHER_SERVER_URL="http://localhost:8100"
curl -sf "$DEEP_RESEARCHER_SERVER_URL/health"
```

Expected output: a successful health response. Then tell the user to keep `DEEP_RESEARCHER_SERVER_URL` set before invoking
`deep-researcher-research`.

## References

| Topic | Documentation |
|---|---|
| Locate or clone Deep Researcher Agent | `references/locate-or-clone.md` |
| Environment and secrets | `references/env-and-secrets.md` |
| Workflow configs | `references/configs.md` |
| Agent Skill backend | `references/skill-backend.md` |
| CLI deployment | `references/terminal-cli.md` |
| Local web deployment | `references/local-web.md` |
| Docker Compose deployment | `references/docker-compose.md` |
| Kubernetes and Helm deployment | `references/kubernetes-helm.md` |
| FRAG integration | `references/frag.md` |
| Basic validation | `references/validation.md` |
| End-to-end validation | `references/end-to-end-validation.md` |
| Troubleshooting | `references/troubleshooting.md` |
| Shutdown and cleanup | `references/shutdown.md` |

## Common Issues

### Issue: Backend port is already in use

**Symptoms:**

- Docker Compose fails to bind port `8000`.
- `curl -sf http://localhost:8000/health` reaches an unexpected service or fails.

**Causes:**

- Another Deep Researcher Agent backend or local development server is already running.
- `PORT` in `deploy/.env` conflicts with an existing process.

**Solutions:**

1. Identify the process:
   ```bash
   lsof -nP -iTCP:8000 -sTCP:LISTEN
   ```
2. Either stop the conflicting process with the user's approval or set a different port in `deploy/.env`, such as
   `PORT=8100`.
3. Restart the selected deployment path and verify:
   ```bash
   curl -sf http://localhost:8100/health
   ```

### Issue: Required credentials are missing

**Symptoms:**

- Infrastructure starts, but model-backed chat or research requests fail.
- Logs mention unauthorized, forbidden, invalid key, or missing provider configuration.

**Causes:**

- `NVIDIA_API_KEY` is missing or empty.
- No supported search provider key is configured for web research.

**Solutions:**

1. Check presence without printing values by following `references/env-and-secrets.md`.
2. Ask the user to update `deploy/.env`; do not ask them to paste secrets into chat.
3. Rerun `references/validation.md` after the user updates credentials.

### Issue: Backend is healthy but not compatible with deep-researcher-research

**Symptoms:**

- `/health` succeeds, but `/chat` or `/v1/jobs/async/agents` fails.
- `deep-researcher-research` reports that async agents are unavailable.

**Causes:**

- The selected config is CLI-only or does not expose the web/API backend expected by the skill.
- `BACKEND_CONFIG` or `CONFIG_FILE` points at the wrong Deep Researcher Agent config.

**Solutions:**

1. Read `references/configs.md` and confirm the selected config is API-enabled.
2. For the default Skill backend, use `configs/config_web_default_llamaindex.yml`.
3. Restart the backend and rerun `references/validation.md`.

### Issue: Docker cleanup would remove useful state

**Symptoms:**

- Troubleshooting suggests `docker compose down -v`.
- The user may have local PostgreSQL job or checkpoint data they want to keep.

**Causes:**

- `down -v` removes Docker volumes.
- Rebuilds and restarts are often enough for config or image changes.

**Solutions:**

1. Prefer a normal restart from `references/shutdown.md`.
2. Ask for explicit approval before running volume deletion.
3. After cleanup, rerun deployment and validation from the selected route.
