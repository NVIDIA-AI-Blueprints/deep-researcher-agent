#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Set up NVIDIA OpenShell for AI-Q per-job policy-backed sandboxes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

OPENSHELL_RELEASE_TAG="$(awk -F'"' '
    $0 == "[tool.aiq.openshell]" { in_contract = 1; next }
    in_contract && /^\[/ { in_contract = 0 }
    in_contract && /^release-tag = / { print $2; exit }
' "$REPO_ROOT/pyproject.toml")"
OPENSHELL_ADAPTER_VERSION="$(awk -F'"' '
    $0 == "[tool.aiq.openshell]" { in_contract = 1; next }
    in_contract && /^\[/ { in_contract = 0 }
    in_contract && /^adapter-version = / { print $2; exit }
' "$REPO_ROOT/pyproject.toml")"
[[ "$OPENSHELL_RELEASE_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || { echo "ERROR: invalid OpenShell release contract" >&2; exit 1; }
[[ "$OPENSHELL_ADAPTER_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || { echo "ERROR: invalid OpenShell adapter contract" >&2; exit 1; }
DEFAULT_OPENSHELL_VERSION="${OPENSHELL_RELEASE_TAG#v}"
OPENSHELL_VERSION="${AIQ_OPENSHELL_VERSION:-}"
# Official OpenShell deepagents adapter: the `langchain-nvidia-openshell` partner
# package, now published on PyPI. Override with LANGCHAIN_NVIDIA_REPO to use a git
# spec or a local checkout (e.g. to test an unreleased adapter build).
DEFAULT_LANGCHAIN_NVIDIA_INSTALL_SPEC="langchain-nvidia-openshell==$OPENSHELL_ADAPTER_VERSION"
LANGCHAIN_NVIDIA_REPO="${LANGCHAIN_NVIDIA_REPO:-$DEFAULT_LANGCHAIN_NVIDIA_INSTALL_SPEC}"
IMAGE_NAME="${AIQ_OPENSHELL_IMAGE:-aiq-openshell-demo:latest}"
# Sandbox log verbosity baked into the image (RUST_LOG). Default `warn` is OpenShell's
# stock sandbox level; set to `debug` to surface in-container process/relay detail.
SANDBOX_LOG_LEVEL="${AIQ_OPENSHELL_SANDBOX_LOG_LEVEL:-warn}"
POLICY_PRESET="${AIQ_OPENSHELL_POLICY:-}"
POLICY_ALLOWLIST="${AIQ_OPENSHELL_POLICY_ALLOWLIST:-${AIQ_OPENSHELL_POLICY_SERVICES:-}}"
POLICY_FILE="${AIQ_OPENSHELL_POLICY_FILE:-$REPO_ROOT/configs/openshell/generated/aiq-openshell-policy.yaml}"
LANDLOCK_COMPATIBILITY="${AIQ_OPENSHELL_LANDLOCK_COMPATIBILITY:-hard_requirement}"
LANDLOCK_COMPATIBILITY_CLI=false
LOCAL_DEMO=false
DOCKER_BIN="${DOCKER_BIN:-}"
OPENSHELL_BIN="${OPENSHELL_BIN:-}"
BUILD_IMAGE=true
LIST_OPENSHELL_VERSIONS=false

SUPPORTED_SERVICES="github,pypi,nvidia,tavily,serper,huggingface,arxiv,semantic-scholar,npm"
SUPPORTED_POLICIES="offline,research,python-packages,ai-dev,custom"

usage() {
    cat <<EOF
Usage: scripts/openshell/setup_openshell.sh [options]

Sets up OpenShell for AI-Q:
  1. Detects macOS/Linux.
  2. Loads the AI-Q-certified exact OpenShell release contract.
  3. Installs the certified OpenShell Python package version.
  4. Installs the langchain-nvidia-openshell deepagents adapter.
  5. Generates an initial OpenShell policy.
  6. Resolves Docker and builds the reusable AI-Q sandbox image.

This script never starts, stops, registers, or probes a gateway. Run
scripts/openshell/start_openshell_gateway.sh after provisioning; per-job sandbox creation
remains owned by the AI-Q runtime.

Canonical operator guide: docs/source/deployment/openshell.md

Options:
  --openshell-version VERSION   Certified exact OpenShell version only (0.0.88).
  --policy CHOICE               Sandbox network policy.
                                Choices: $SUPPORTED_POLICIES
                                Default: asks in an interactive shell, offline otherwise.
  --allow LIST                  Comma-separated services for --policy custom.
                                Services: $SUPPORTED_SERVICES
  --policy-file PATH            Output policy file.
                                Default: configs/openshell/generated/aiq-openshell-policy.yaml
  --landlock-compatibility MODE hard_requirement (default) or best_effort (local demo only).
  --local-demo                 Shortcut for best_effort policy generation. Runtime commands
                               must set AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false.
  --image-name NAME             Docker image tag (default: aiq-openshell-demo:latest).
  --sandbox-log-level LEVEL     In-container OpenShell log verbosity baked into the
                                image via RUST_LOG (default: warn). Use "debug" to
                                surface process/relay detail in the sandbox logs.
  --langchain-nvidia SPEC       uv install spec or local langchain-nvidia checkout
                                for the langchain-nvidia-openshell adapter.
  --docker-bin PATH             Docker CLI path when docker is not on PATH.
  --skip-build                  Do not build the sandbox image.
  --list-policies               Print supported policy choices.
  --list-services               Print supported services for --allow.
  --list-openshell-versions     Print the certified OpenShell version.
  -h, --help                    Show this help.

Examples:
  scripts/openshell/setup_openshell.sh
  scripts/openshell/setup_openshell.sh --policy python-packages
  scripts/openshell/setup_openshell.sh --policy custom --allow github,pypi,nvidia,tavily
  scripts/openshell/setup_openshell.sh --openshell-version 0.0.88 --policy offline
  scripts/openshell/setup_openshell.sh --local-demo --policy offline
EOF
}

log() {
    echo ""
    echo "==> $*"
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openshell-version)
            OPENSHELL_VERSION="$2"
            shift 2
            ;;
        --policy)
            POLICY_PRESET="$2"
            shift 2
            ;;
        --allow)
            POLICY_ALLOWLIST="$2"
            shift 2
            ;;
        --policy-services)
            POLICY_ALLOWLIST="$2"
            shift 2
            ;;
        --policy-file)
            POLICY_FILE="$2"
            shift 2
            ;;
        --landlock-compatibility)
            LANDLOCK_COMPATIBILITY="$2"
            LANDLOCK_COMPATIBILITY_CLI=true
            shift 2
            ;;
        --local-demo)
            LOCAL_DEMO=true
            shift
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --sandbox-log-level)
            SANDBOX_LOG_LEVEL="$2"
            shift 2
            ;;
        --langchain-nvidia)
            LANGCHAIN_NVIDIA_REPO="$2"
            shift 2
            ;;
        --docker-bin)
            DOCKER_BIN="$2"
            shift 2
            ;;
        --skip-build)
            BUILD_IMAGE=false
            shift
            ;;
        --create-shared-debug-sandbox|--sandbox-name|--gateway-name|--gateway-port|--gateway-bin|--no-restart-gateway|--skip-sandbox)
            fail "Gateway and debug-sandbox lifecycle moved to scripts/openshell/start_openshell_gateway.sh"
            ;;
        --list-policies)
            echo "$SUPPORTED_POLICIES" | tr ',' '\n'
            exit 0
            ;;
        --list-services)
            echo "$SUPPORTED_SERVICES" | tr ',' '\n'
            exit 0
            ;;
        --list-openshell-versions)
            LIST_OPENSHELL_VERSIONS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            fail "Unknown option: $1"
            ;;
    esac
done

if [[ "$LOCAL_DEMO" == "true" ]]; then
    if [[ "$LANDLOCK_COMPATIBILITY_CLI" == "true" && "$LANDLOCK_COMPATIBILITY" != "best_effort" ]]; then
        fail "--local-demo conflicts with --landlock-compatibility $LANDLOCK_COMPATIBILITY"
    fi
    LANDLOCK_COMPATIBILITY="best_effort"
fi

detect_os() {
    case "$(uname -s)" in
        Darwin)
            OS_NAME=macos
            ;;
        Linux)
            OS_NAME=linux
            ;;
        *)
            fail "Unsupported operating system: $(uname -s). This script supports macOS and Linux."
            ;;
    esac
    echo "Detected OS: $OS_NAME"
}

require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        cat <<'EOF'
uv was not found on PATH.

Install uv, then rerun:

  curl -LsSf https://astral.sh/uv/install.sh | sh

If uv is already installed, open a new shell or add it to PATH before rerunning.
EOF
        exit 1
    fi
}

resolve_openshell_version() {
    if [[ -z "$OPENSHELL_VERSION" ]]; then
        OPENSHELL_VERSION="$DEFAULT_OPENSHELL_VERSION"
    fi
    if [[ "$OPENSHELL_VERSION" == "latest" || "$OPENSHELL_VERSION" != "$DEFAULT_OPENSHELL_VERSION" ]]; then
        fail "OpenShell '$OPENSHELL_VERSION' is not certified for this AI-Q release. Use exact version $DEFAULT_OPENSHELL_VERSION; certify upgrades in a separate development change."
    fi
    echo "OpenShell version selected: $OPENSHELL_VERSION"
}

ensure_aiq_env() {
    cd "$REPO_ROOT"
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating AI-Q virtual environment"
        ./scripts/setup.sh
    else
        log "Using existing AI-Q virtual environment"
    fi
    # Never let a caller's activated checkout redirect `uv pip` into another
    # repository. Provisioning must mutate only this repository's environment.
    export VIRTUAL_ENV="$VENV_DIR"
    case ":$PATH:" in
        *":$VENV_DIR/bin:"*) ;;
        *) export PATH="$VENV_DIR/bin:$PATH" ;;
    esac
}

install_openshell_python() {
    log "Installing OpenShell Python package exactly: openshell==$OPENSHELL_VERSION"
    uv pip install "openshell==$OPENSHELL_VERSION"

    local adapter_install_spec="$LANGCHAIN_NVIDIA_REPO"
    local editable_args=()
    if [[ -d "$LANGCHAIN_NVIDIA_REPO" ]]; then
        if [[ -f "$LANGCHAIN_NVIDIA_REPO/libs/openshell/pyproject.toml" ]]; then
            adapter_install_spec="$LANGCHAIN_NVIDIA_REPO/libs/openshell"
        elif [[ -f "$LANGCHAIN_NVIDIA_REPO/pyproject.toml" ]]; then
            adapter_install_spec="$LANGCHAIN_NVIDIA_REPO"
        else
            fail "langchain-nvidia-openshell package not found in local checkout: $LANGCHAIN_NVIDIA_REPO"
        fi
        editable_args=(-e)
    fi

    log "Installing langchain-nvidia-openshell adapter: $adapter_install_spec"
    # NOTE: expand editable_args only when non-empty; macOS bash 3.2 errors on
    # "${arr[@]}" for an empty array under `set -u`.
    local adapter_install_args=()
    if [[ ${#editable_args[@]} -gt 0 ]]; then
        adapter_install_args=("${editable_args[@]}")
    fi
    local adapter_install_failed=false
    if [[ ${#adapter_install_args[@]} -eq 0 ]]; then
        uv pip install "$adapter_install_spec" || adapter_install_failed=true
    else
        uv pip install "${adapter_install_args[@]}" "$adapter_install_spec" || adapter_install_failed=true
    fi
    if [[ "$adapter_install_failed" == "true" ]]; then
        cat <<EOF
ERROR: Could not install the langchain-nvidia-openshell adapter.

The adapter is published on PyPI as langchain-nvidia-openshell. The default install
resolves from PyPI; if your environment cannot reach it, point LANGCHAIN_NVIDIA_REPO at
an alternate uv install spec or a local checkout, then rerun. Examples:
  LANGCHAIN_NVIDIA_REPO=langchain-nvidia-openshell==$OPENSHELL_ADAPTER_VERSION scripts/openshell/setup_openshell.sh
  scripts/openshell/setup_openshell.sh --langchain-nvidia /path/to/langchain-nvidia
EOF
        exit 1
    fi

    # Adapter 0.1.0 still declares deepagents<0.6 and can otherwise downgrade AI-Q's
    # locked DeepAgents 0.6.x runtime. Restore the complete AI-Q lock while retaining
    # optional packages that are intentionally absent from the base project metadata.
    # This keeps repeated setup deterministic without pretending the upstream adapter
    # metadata is compatible; `pip check` remains a documented upstream limitation.
    log "Restoring locked AI-Q dependencies while retaining optional OpenShell packages"
    uv sync --dev --inexact

    # Adapter dependency resolution must not silently change the operator-selected
    # OpenShell SDK/CLI version. Reapply the exact pin after every dependent package.
    log "Reasserting exact OpenShell version after adapter install: openshell==$OPENSHELL_VERSION"
    uv pip install "openshell==$OPENSHELL_VERSION"

    local installed
    installed="$("$VENV_DIR/bin/python" - <<'PY'
import openshell
print(getattr(openshell, "__version__", "unknown"))
PY
)"
    # CLI, SDK, and gateway compatibility is validated again by the strict readiness
    # checker. Provisioning still guarantees that the selected local SDK is exact.
    if ! "$VENV_DIR/bin/python" - "$OPENSHELL_VERSION" "$installed" <<'PY'
import sys
sys.exit(0 if sys.argv[2] == sys.argv[1] else 1)
PY
    then
        fail "Installed openshell $installed does not match the requested version $OPENSHELL_VERSION"
    fi
    "$VENV_DIR/bin/python" - <<'PY'
import langchain_nvidia_openshell  # noqa: F401
PY
    echo "OpenShell Python package verified: $installed"
    echo "langchain-nvidia-openshell adapter verified"
}

resolve_openshell_cli() {
    local candidates=(
        "$OPENSHELL_BIN"
        "$VENV_DIR/bin/openshell"
        "$(command -v openshell || true)"
        "$HOME/.local/bin/openshell"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            OPENSHELL_BIN="$candidate"
            echo "OpenShell CLI: $OPENSHELL_BIN ($("$OPENSHELL_BIN" --version 2>/dev/null || true))"
            return
        fi
    done
    fail "OpenShell CLI was not found after installing openshell==$OPENSHELL_VERSION"
}

resolve_docker() {
    log "Resolving Docker CLI"
    # Docker Desktop stores both the CLI and credential helper here. Prepend the
    # directory before resolving `docker` so public-image pulls do not follow a
    # stale /usr/local symlink and then fail to locate docker-credential-desktop.
    local docker_desktop_bin="/Applications/Docker.app/Contents/Resources/bin"
    if [[ "$OS_NAME" == "macos" && -x "$docker_desktop_bin/docker" ]]; then
        export PATH="$docker_desktop_bin:$PATH"
    fi
    local candidates=(
        "$DOCKER_BIN"
        "$(command -v docker || true)"
        "/opt/homebrew/bin/docker"
        "/usr/local/bin/docker"
        "/usr/bin/docker"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            DOCKER_BIN="$candidate"
            echo "Docker CLI: $DOCKER_BIN"
            return
        fi
    done

    if [[ "$OS_NAME" == "macos" ]]; then
        candidate="$(find /opt/homebrew/Cellar/docker /usr/local/Cellar/docker -path '*/bin/docker' -type f 2>/dev/null | sort | tail -1 || true)"
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            DOCKER_BIN="$candidate"
            echo "Docker CLI: $DOCKER_BIN"
            return
        fi
        cat <<'EOF'
Docker CLI was not found.

Install or link Docker CLI, then rerun:

  brew install docker

If Docker is installed but not on PATH, rerun with:

  scripts/openshell/setup_openshell.sh --docker-bin /path/to/docker
EOF
    else
        cat <<'EOF'
Docker CLI was not found.

Install Docker for your Linux distribution, then rerun. For example:

  sudo apt-get update
  sudo apt-get install -y docker.io

If Docker is installed but not on PATH, rerun with:

  scripts/openshell/setup_openshell.sh --docker-bin /path/to/docker
EOF
    fi
    exit 1
}

configure_docker_host() {
    if [[ -n "${DOCKER_HOST:-}" ]]; then
        echo "DOCKER_HOST already set: $DOCKER_HOST"
        return
    fi
    local colima_openshell="$HOME/.colima/openshell/docker.sock"
    local colima_default="$HOME/.colima/default/docker.sock"
    if [[ -S "$colima_openshell" ]]; then
        export DOCKER_HOST="unix://$colima_openshell"
        echo "DOCKER_HOST: $DOCKER_HOST"
    elif [[ -S "$colima_default" ]]; then
        export DOCKER_HOST="unix://$colima_default"
        echo "DOCKER_HOST: $DOCKER_HOST"
    fi
}

verify_docker_runtime() {
    log "Verifying Docker runtime"
    if "$DOCKER_BIN" info >/dev/null 2>&1; then
        echo "Docker daemon is reachable"
        return
    fi

    if [[ -n "${DOCKER_HOST:-}" ]]; then
        cat <<EOF
Docker CLI was found, but the Docker daemon is not reachable.

DOCKER_HOST is set to:

  $DOCKER_HOST

Start that Docker daemon or update DOCKER_HOST, then rerun:

  scripts/openshell/setup_openshell.sh
EOF
        exit 1
    fi

    if [[ "$OS_NAME" == "macos" ]]; then
        local docker_desktop_app=""
        if [[ -d "/Applications/Docker.app" ]]; then
            docker_desktop_app="/Applications/Docker.app"
        elif [[ -d "$HOME/Applications/Docker.app" ]]; then
            docker_desktop_app="$HOME/Applications/Docker.app"
        fi

        if [[ -n "$docker_desktop_app" ]]; then
            cat <<EOF
Docker CLI was found, but the Docker daemon is not reachable.

No Colima socket was selected, so this setup expects Docker Desktop on macOS.
Docker Desktop appears to be installed at:

  $docker_desktop_app

Start Docker Desktop, wait until it reports that Docker is running, then rerun:

  scripts/openshell/setup_openshell.sh

If you use a remote Docker daemon, set DOCKER_HOST before rerunning.
EOF
        else
            cat <<'EOF'
Docker CLI was found, but the Docker daemon is not reachable.

No Colima socket was selected and Docker Desktop was not found in /Applications
or ~/Applications. Install and start Docker Desktop for macOS, then rerun:

  scripts/openshell/setup_openshell.sh

If you use Colima, start it first. For example:

  colima start

If you use a remote Docker daemon, set DOCKER_HOST before rerunning.
EOF
        fi
    else
        cat <<'EOF'
Docker CLI was found, but the Docker daemon is not reachable.

On Linux, start Docker and make sure your user can access it, then rerun. For example:

  sudo systemctl start docker
  docker info

If Docker is running but permission is denied, add your user to the docker group,
log out and back in, then rerun:

  sudo usermod -aG docker "$USER"

If you use a remote Docker daemon, set DOCKER_HOST before rerunning.
EOF
    fi
    exit 1
}

print_policy_menu() {
    cat <<'EOF'

Choose an OpenShell sandbox network policy:

  1. Offline (recommended)
     No sandbox network access. AI-Q tools gather data; sandbox code computes on inputs.

  2. Research APIs
     Allow GitHub, NVIDIA API, Tavily, and Serper.

  3. Python packages
     Allow GitHub and PyPI for package metadata/download checks.

  4. AI development
     Allow GitHub, PyPI, NVIDIA API, Tavily, Serper, Hugging Face, arXiv,
     Semantic Scholar, and npm.

  5. Custom
     Type a comma-separated allowlist.

EOF
}

choose_policy_interactively() {
    if [[ -n "$POLICY_PRESET" || -n "$POLICY_ALLOWLIST" ]]; then
        return
    fi
    if [[ ! -t 0 ]]; then
        POLICY_PRESET=offline
        return
    fi

    print_policy_menu
    local choice
    while true; do
        read -r -p "Policy choice [1]: " choice
        choice="${choice:-1}"
        case "$choice" in
            1|offline)
                POLICY_PRESET=offline
                return
                ;;
            2|research)
                POLICY_PRESET=research
                return
                ;;
            3|python|python-packages)
                POLICY_PRESET=python-packages
                return
                ;;
            4|ai|ai-dev)
                POLICY_PRESET=ai-dev
                return
                ;;
            5|custom)
                POLICY_PRESET=custom
                read -r -p "Allow services ($SUPPORTED_SERVICES): " POLICY_ALLOWLIST
                return
                ;;
            *)
                echo "Choose 1, 2, 3, 4, or 5."
                ;;
        esac
    done
}

resolve_policy() {
    choose_policy_interactively
    POLICY_PRESET="$(echo "${POLICY_PRESET:-}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    POLICY_ALLOWLIST="$(echo "${POLICY_ALLOWLIST:-}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"

    if [[ -z "$POLICY_PRESET" && -n "$POLICY_ALLOWLIST" ]]; then
        POLICY_PRESET=custom
    fi
    if [[ -z "$POLICY_PRESET" ]]; then
        POLICY_PRESET=offline
    fi
    if [[ "$POLICY_PRESET" == "none" ]]; then
        POLICY_PRESET=offline
    fi

    case "$POLICY_PRESET" in
        offline)
            POLICY_ALLOWLIST=offline
            ;;
        research)
            POLICY_ALLOWLIST=github,nvidia,tavily,serper
            ;;
        python-packages)
            POLICY_ALLOWLIST=github,pypi
            ;;
        ai-dev)
            POLICY_ALLOWLIST=github,pypi,nvidia,tavily,serper,huggingface,arxiv,semantic-scholar,npm
            ;;
        custom)
            if [[ -z "$POLICY_ALLOWLIST" ]]; then
                fail "--policy custom requires --allow, for example: --policy custom --allow github,pypi"
            fi
            ;;
        *)
            fail "Unsupported policy '$POLICY_PRESET'. Choices: $SUPPORTED_POLICIES"
            ;;
    esac

    if [[ "$POLICY_ALLOWLIST" == *offline* && "$POLICY_ALLOWLIST" != "offline" ]]; then
        fail "Use either offline or a service allowlist, not both: $POLICY_ALLOWLIST"
    fi
    case "$LANDLOCK_COMPATIBILITY" in
        hard_requirement|best_effort)
            ;;
        *)
            fail "Unsupported Landlock compatibility '$LANDLOCK_COMPATIBILITY'; use hard_requirement or best_effort"
            ;;
    esac
}

validate_service() {
    case "$1" in
        offline|github|pypi|nvidia|tavily|serper|huggingface|arxiv|semantic-scholar|npm)
            ;;
        *)
            fail "Unsupported policy service '$1'. Supported: $SUPPORTED_SERVICES"
            ;;
    esac
}

emit_policy_header() {
    cat >"$POLICY_FILE" <<EOF
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated by scripts/openshell/setup_openshell.sh.

version: 1

filesystem_policy:
  include_workdir: true
  # Declare the proxy baseline up front so OpenShell does not create an enriched
  # revision whose content/hash differs from the policy AI-Q submitted.
  read_only:
    - /usr
    - /lib
    - /etc
    - /var/log
    - /proc
    - /dev/urandom
  read_write:
    - /sandbox
    - /workspace
    - /tmp
    - /dev/null

# hard_requirement is the production default: a missing Landlock LSM fails closed. Set
# best_effort only for an explicit local demo that accepts loss of filesystem confinement.
landlock:
  compatibility: $LANDLOCK_COMPATIBILITY

process:
  run_as_user: sandbox
  run_as_group: sandbox

EOF
}

emit_policy_entry() {
    local key="$1"
    local name="$2"
    shift 2
    {
        echo "  $key:"
        echo "    name: $name"
        echo "    endpoints:"
        local host
        for host in "$@"; do
            echo "      - host: $host"
            echo "        port: 443"
            echo "        protocol: rest"
            echo "        enforcement: enforce"
            echo "        access: read-only"
        done
        echo "    binaries:"
        echo "      - { path: /usr/bin/curl }"
        echo "      - { path: /usr/local/bin/python3 }"
        echo "      - { path: /usr/local/bin/python }"
        echo "      - { path: /usr/local/bin/pip }"
        echo "      - { path: /usr/local/bin/pip3 }"
    } >>"$POLICY_FILE"
}

emit_policy_service() {
    case "$1" in
        github)
            emit_policy_entry github github-readonly api.github.com github.com
            ;;
        pypi)
            emit_policy_entry pypi pypi-readonly pypi.org files.pythonhosted.org
            ;;
        nvidia)
            emit_policy_entry nvidia nvidia-api-readonly integrate.api.nvidia.com
            ;;
        tavily)
            emit_policy_entry tavily tavily-api-readonly api.tavily.com
            ;;
        serper)
            emit_policy_entry serper serper-api-readonly google.serper.dev
            ;;
        huggingface)
            emit_policy_entry huggingface huggingface-readonly huggingface.co cdn-lfs.huggingface.co
            ;;
        arxiv)
            emit_policy_entry arxiv arxiv-readonly export.arxiv.org
            ;;
        semantic-scholar)
            emit_policy_entry semantic_scholar semantic-scholar-readonly api.semanticscholar.org
            ;;
        npm)
            emit_policy_entry npm npm-readonly registry.npmjs.org
            ;;
    esac
}

generate_policy() {
    log "Generating OpenShell policy"
    resolve_policy
    mkdir -p "$(dirname "$POLICY_FILE")"
    emit_policy_header
    if [[ "$POLICY_ALLOWLIST" == "offline" ]]; then
        echo "network_policies: {}" >>"$POLICY_FILE"
    else
        echo "network_policies:" >>"$POLICY_FILE"
        IFS=',' read -r -a services <<<"$POLICY_ALLOWLIST"
        local service
        for service in "${services[@]}"; do
            validate_service "$service"
            emit_policy_service "$service"
        done
    fi
    echo "Policy file: $POLICY_FILE"
    echo "Policy: $POLICY_PRESET"
    echo "Allowed services: $POLICY_ALLOWLIST"
    echo "Landlock compatibility: $LANDLOCK_COMPATIBILITY"
}

build_image() {
    if [[ "$BUILD_IMAGE" != "true" ]]; then
        return
    fi
    log "Building sandbox image: $IMAGE_NAME (sandbox log level: $SANDBOX_LOG_LEVEL)"
    "$DOCKER_BIN" build -t "$IMAGE_NAME" \
        --build-arg OPENSHELL_SANDBOX_LOG_LEVEL="$SANDBOX_LOG_LEVEL" \
        -f "$REPO_ROOT/deploy/openshell/Dockerfile.aiq-demo" "$REPO_ROOT/deploy/openshell"
}

diagnose_gateway_components() {
    log "Inspecting packaged gateway components"
    if ! "$VENV_DIR/bin/python" "$SCRIPT_DIR/check_versions.py" \
        --gateway-name "${AIQ_OPENSHELL_GATEWAY_NAME:-openshell}" --skip-live; then
        echo "Provisioning completed, but gateway remediation is required before readiness or AI-Q startup."
    fi
}

print_next_steps() {
    local runtime_config="configs/config_openshell.yml"
    local runtime_env=""
    local landlock_note="Production defaults require hard Landlock; no runtime override is needed."
    if [[ "$LANDLOCK_COMPATIBILITY" == "best_effort" ]]; then
        runtime_env="AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false "
        landlock_note="This is a local best_effort policy. Prefix validation, CLI, and E2E commands with
AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false."
    fi
    cat <<EOF

OpenShell dependencies, policy, and image are provisioned for AI-Q.

The default local gateway, image, policy path, and expected version are already
wired into the launcher, config, and live suite. These are optional overrides
for custom shells or remote gateways:

  export AIQ_OPENSHELL_GATEWAY_NAME="openshell"
  export AIQ_OPENSHELL_WORKSPACE="${AIQ_OPENSHELL_WORKSPACE:-default}"
  export AIQ_OPENSHELL_IMAGE="$IMAGE_NAME"
  export AIQ_OPENSHELL_POLICY_FILE="$POLICY_FILE"
  export AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION="$OPENSHELL_VERSION"

$landlock_note

Validate the AI-Q config:

  ${runtime_env}.venv/bin/nat validate --config_file $runtime_config

Start or verify an authenticated gateway and run its strict capability probe:

  source .venv/bin/activate
  ./scripts/openshell/start_openshell_gateway.sh

Start CLI mode after the gateway probe succeeds:

  ${runtime_env}./scripts/start_cli.sh --config_file $runtime_config --verbose

Start E2E mode after the gateway probe succeeds:

  ${runtime_env}./scripts/start_e2e.sh --config_file $runtime_config

Or combine the gateway probe with E2E startup:

  ${runtime_env}./scripts/start_e2e.sh --start-openshell-gateway --config_file $runtime_config

EOF
}

main() {
    detect_os
    require_uv
    if [[ "$LIST_OPENSHELL_VERSIONS" == "true" ]]; then
        echo "$DEFAULT_OPENSHELL_VERSION"
        exit 0
    fi
    resolve_openshell_version
    resolve_policy
    ensure_aiq_env
    install_openshell_python
    resolve_openshell_cli
    resolve_docker
    configure_docker_host
    verify_docker_runtime
    generate_policy
    build_image
    diagnose_gateway_components
    print_next_steps
}

main
