#!/bin/bash
# Setup script for Deep Researcher Agent blueprint development environment

set -euo pipefail

UV_MIN_VERSION="0.11.25"

version_at_least() {
    local current="$1"
    local minimum="$2"
    local current_major current_minor current_patch
    local minimum_major minimum_minor minimum_patch

    if [[ ! "${current}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    current_major="${BASH_REMATCH[1]}"
    current_minor="${BASH_REMATCH[2]}"
    current_patch="${BASH_REMATCH[3]}"

    if [[ ! "${minimum}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    minimum_major="${BASH_REMATCH[1]}"
    minimum_minor="${BASH_REMATCH[2]}"
    minimum_patch="${BASH_REMATCH[3]}"

    if (( current_major != minimum_major )); then
        (( current_major > minimum_major ))
        return
    fi
    if (( current_minor != minimum_minor )); then
        (( current_minor > minimum_minor ))
        return
    fi
    (( current_patch >= minimum_patch ))
}

echo "=== Deep Researcher Agent Blueprint Development Setup ==="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed"
else
    echo "uv is already installed"
fi

# Resolve uv binary for non-interactive shells (e.g., Jupyter).
# The installer may place uv in ~/.local/bin, which isn't always on PATH yet.
UV_BIN="$(command -v uv || true)"
if [ -z "${UV_BIN}" ] && [ -x "${HOME}/.local/bin/uv" ]; then
    export PATH="${HOME}/.local/bin:${PATH}"
    UV_BIN="${HOME}/.local/bin/uv"
fi

if [ -z "${UV_BIN}" ]; then
    echo "Error: uv was not found after installation."
    echo "Add uv to PATH (typically ${HOME}/.local/bin) and re-run setup."
    exit 1
fi

UV_CURRENT_VERSION="$("${UV_BIN}" --version)"
UV_CURRENT_VERSION="${UV_CURRENT_VERSION#uv }"
UV_CURRENT_VERSION="${UV_CURRENT_VERSION%% *}"
if ! version_at_least "${UV_CURRENT_VERSION}" "${UV_MIN_VERSION}"; then
    echo "Error: uv ${UV_MIN_VERSION} or newer is required; found ${UV_CURRENT_VERSION} at ${UV_BIN}."
    echo "The MCP project uses scoped dependency overrides introduced in uv ${UV_MIN_VERSION}."
    echo "Upgrade uv and re-run setup (CI and the MCP image use uv 0.11.26)."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
"${UV_BIN}" venv --python 3.13 --seed .venv
echo "Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install core framework with dev dependencies (uses uv.lock to pin versions)
echo ""
echo "Installing core framework with dev dependencies..."
"${UV_BIN}" sync --group dev
echo "Core framework installed"

# Install the MCP server in its isolated project environment. The MCP lockfile
# intentionally carries a different cryptography policy from the root project.
echo ""
echo "Installing MCP server with dev dependencies..."
"${UV_BIN}" sync --project mcp --extra dev
echo "MCP server installed in mcp/.venv"

# Install frontends (--no-deps: dependencies already resolved by uv sync)
echo ""
echo "Installing frontends..."
"${UV_BIN}" pip install -e ./frontends/cli
"${UV_BIN}" pip install -e ./frontends/debug
"${UV_BIN}" pip install -e ./frontends/deep_researcher_api
echo "Frontends installed (CLI, Debug, Deep Researcher Agent API)"

# Install benchmarks
echo ""
echo "Installing benchmarks..."
"${UV_BIN}" pip install -e ./frontends/benchmarks/freshqa
"${UV_BIN}" pip install -e ./frontends/benchmarks/deepsearch_qa
echo "Benchmarks installed"

# Install data sources
echo ""
echo "Installing data sources..."
"${UV_BIN}" pip install -e ./sources/gsf
"${UV_BIN}" pip install -e ./sources/tavily_web_search
"${UV_BIN}" pip install -e ./sources/exa_web_search
"${UV_BIN}" pip install -e ./sources/nimble_web_search
"${UV_BIN}" pip install -e ./sources/firecrawl_web_search
"${UV_BIN}" pip install -e ./sources/you_com
"${UV_BIN}" pip install -e ./sources/google_scholar_paper_search
"${UV_BIN}" pip install -e "./sources/knowledge_layer[llamaindex,foundational_rag]"
echo "Data Sources installed"

# Setup pre-commit
echo ""
echo "Setting up pre-commit hooks..."
pre-commit install
echo "Pre-commit hooks installed"

# Setup environment file
echo ""
if [ ! -f deploy/.env ]; then
    echo "Creating .env file from template..."
    cp deploy/.env.example deploy/.env
    echo "Please edit deploy/.env and add your NVIDIA_API_KEY"
else
    echo ".env file already exists"
fi

# Setup UI dependencies (optional)
echo ""
if [ -d "frontends/ui" ]; then
    echo "Setting up UI dependencies..."
    cd frontends/ui

    if command -v npm &> /dev/null; then
        npm ci
        echo "UI dependencies installed"
    else
        echo "npm not found. Skipping UI setup."
        echo "   Install Node.js 22+ to enable UI features"
    fi

    cd ../..
else
    echo "UI directory not found at frontends/ui"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Add your NVIDIA_API_KEY to deploy/.env"
echo "3. Run the agent:"
echo "   - CLI mode:        ./scripts/start_cli.sh"
echo "   - Skill backend:   ./scripts/start_as_skill.sh"
echo "   - Server mode:     ./scripts/start_server_in_debug_mode.sh"
echo "   - End-to-End (UI): ./scripts/start_e2e.sh"
echo "4. Run the MCP server from its isolated environment:"
echo "   uv run --project mcp deep-researcher-mcp-server"
echo ""
