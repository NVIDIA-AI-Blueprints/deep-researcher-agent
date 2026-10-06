# Development Scripts

This directory contains helper scripts for developing and running the Deep Researcher Agent blueprint.

## Available Scripts

### `setup.sh` - Initial Setup

Initializes the development environment, including Python dependencies and UI dependencies.

```bash
./scripts/setup.sh
```

### `dev.sh` - Development Helper

Main development command hub for common tasks.

```bash
./scripts/dev.sh <command>
```

**Commands:**

| Command | Description |
|---------|-------------|
| `test` | Run tests with pytest |
| `format` | Format code with isort and yapf |
| `lint` | Check code formatting (no changes) |
| `pre-commit` | Format code and run lint checks |
| `pylint` | Run pylint static analysis |
| `run` | Run the agent |
| `clean` | Remove build artifacts |
| `help` | Show help message |

### `start_cli.sh` - CLI Mode

Starts the agent in CLI mode with browser-based authentication.

```bash
./scripts/start_cli.sh
./scripts/start_cli.sh --verbose
```

**Options:**

| Option | Description |
|--------|-------------|
| `--verbose` or `-v` | Enable verbose logging |
| `--config_file <path>` | Use a custom configuration file |

### `openshell/` - OpenShell Sandbox Utilities

`openshell/setup_openshell.sh` installs the certified SDK/adapter, generates a policy, and builds
the reusable image. `openshell/install_gateway.sh` is the explicit Apple Silicon macOS
entry point for installing the official packaged gateway. `openshell/start_openshell_gateway.sh`
validates an authenticated registered
gateway and performs a disposable version/policy/selector/execution/cleanup probe. Deep Researcher Agent
then owns one attested physical sandbox per job. OpenShell `0.0.88` is the supported
version because it adds the Linux hard-Landlock file-path fix while retaining the required
policy-revision and request-label capabilities.
The quick start below is the Linux production pairing; activate the repository virtual
environment first. macOS requires the explicit local-demo pairing in the canonical guide.

```bash
source .venv/bin/activate
./scripts/openshell/setup_openshell.sh --openshell-version 0.0.88 --policy offline
./scripts/openshell/start_openshell_gateway.sh --gateway-name openshell
./scripts/start_e2e.sh --config_file configs/config_openshell.yml
```

For a macOS local demo:

```bash
/opt/homebrew/bin/bash ./scripts/openshell/setup_openshell.sh --local-demo --policy offline
./scripts/openshell/install_gateway.sh --dry-run
./scripts/openshell/install_gateway.sh
DEEP_RESEARCHER_OPENSHELL_REQUIRE_HARD_LANDLOCK=false \
  ./scripts/start_e2e.sh --start-openshell-gateway --config_file configs/config_openshell.yml
```

For supported platforms, production versus local-demo policy pairing, remote gateways,
live pytest acceptance, and troubleshooting, use the canonical
[OpenShell deployment guide](../docs/source/deployment/openshell.md). This README is only
the script-discovery surface.

### `start_server_in_debug_mode.sh` - Server Mode

Starts the NAT FastAPI server for deep research with async job support.

```bash
./scripts/start_server_in_debug_mode.sh
./scripts/start_server_in_debug_mode.sh --port 8080
./scripts/start_server_in_debug_mode.sh --config_file configs/config_web_frag.yml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--port <port>` | Server port (default: 8000) |
| `--config_file <path>` | Use a custom configuration file |

**Available Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `http://localhost:8000/docs` | API Documentation (Swagger UI) |
| `http://localhost:8000/debug` | Debug Console for testing async jobs |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/v1/jobs/async/agents` | List available agent types |
| `http://localhost:8000/v1/jobs/async/submit` | Submit async job (POST) |
| `http://localhost:8000/v1/jobs/async/job/{id}/stream` | SSE stream for job progress |

### `start_as_skill.sh` - Agent Skill Backend

Starts the Deep Researcher Agent API backend for use by Agent Skills such as `deep-researcher-research`. This does not start the Next.js UI and disables the optional debug console.

```bash
./scripts/start_as_skill.sh
./scripts/start_as_skill.sh --port 8100
./scripts/start_as_skill.sh --config_file configs/config_web_default_llamaindex.yml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--host <host>` | Server host (default: 0.0.0.0) |
| `--port <port>` | Server port (default: 8000) |
| `--config_file <path>` | Use an API-enabled configuration file |

**Available Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `http://localhost:8000/docs` | API Documentation (Swagger UI) |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/v1/jobs/async/agents` | List available agent types |
| `http://localhost:8000/v1/jobs/async/submit` | Submit async job (POST) |
| `http://localhost:8000/v1/jobs/async/job/{id}/stream` | SSE stream for job progress |

### `start_e2e.sh` - End-to-End Mode

Starts both backend and frontend for full WebSocket support and HITL workflows.
Complete the standard setup and activate the virtual environment first:

```bash
./scripts/setup.sh
source .venv/bin/activate
```

```bash
./scripts/start_e2e.sh
./scripts/start_e2e.sh --config_file configs/config_openshell.yml
./scripts/start_e2e.sh --start-openshell-gateway --config_file configs/config_openshell.yml --port 8080
```

**Options:**

| Option | Description |
|--------|-------------|
| `--config_file <path>` | Use a custom configuration file |
| `--port <port>` | Backend port and frontend backend URL (default: 8000) |
| `--start-openshell-gateway` | Start/reuse the packaged authenticated gateway and run its strict capability probe before E2E |

**Services:**

| Service | URL |
|---------|-----|
| Backend | `http://localhost:8000` |
| Frontend | `http://localhost:3000` |

**Available Configs:**

| Config File | Description |
|-------------|-------------|
| `configs/config_cli_default.yml` | CLI mode with web search (default) |
| `configs/config_web_frag.yml` | Server/E2E mode with Foundational RAG |
| `configs/config_web_default_llamaindex.yml` | Server/E2E mode with LlamaIndex |
| `configs/config_skills.yml` | Deep research with DeepAgents skills + Modal sandbox |
| `configs/config_openshell.yml` | Experimental per-job OpenShell sandbox + artifact capture (run `scripts/openshell/setup_openshell.sh` first) |

## Development Workflow

When developing new features:

1. **Update code**: Make your changes to the codebase
2. **Test your changes**:
   ```bash
   ./scripts/dev.sh test
   ```
3. **Format and lint**:
   ```bash
   ./scripts/dev.sh pre-commit
   ```
4. **Run the agent**:
   ```bash
   ./scripts/start_cli.sh
   # OR
   ./scripts/start_as_skill.sh
   # OR
   ./scripts/start_e2e.sh
   ```
