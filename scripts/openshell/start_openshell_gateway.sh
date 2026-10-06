#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start or reuse an authenticated OpenShell gateway and prove strict Deep Researcher Agent capabilities.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

GATEWAY_NAME="${DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME:-openshell}"
WORKSPACE_NAME="${DEEP_RESEARCHER_OPENSHELL_WORKSPACE:-default}"
export OPENSHELL_WORKSPACE="$WORKSPACE_NAME"
IMAGE_NAME="${DEEP_RESEARCHER_OPENSHELL_IMAGE:-deep-researcher-openshell-demo:latest}"
POLICY_FILE="${DEEP_RESEARCHER_OPENSHELL_POLICY_FILE:-$REPO_ROOT/configs/openshell/generated/deep-researcher-openshell-policy.yaml}"
SANDBOX_NAME="${DEEP_RESEARCHER_OPENSHELL_SANDBOX_NAME:-deep-researcher-openshell-demo}"
OPENSHELL_BIN="${OPENSHELL_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
START_SERVICE=true
CREATE_SHARED_DEBUG_SANDBOX=false
STATUS_ATTEMPTS="${DEEP_RESEARCHER_OPENSHELL_STATUS_ATTEMPTS:-60}"
POLL_DELAY="${DEEP_RESEARCHER_OPENSHELL_POLL_DELAY:-1}"
READY_TIMEOUT_SECONDS="${DEEP_RESEARCHER_OPENSHELL_READY_TIMEOUT_SECONDS:-120}"
POLICY_LOAD_TIMEOUT_SECONDS="${DEEP_RESEARCHER_OPENSHELL_POLICY_LOAD_TIMEOUT_SECONDS:-30}"
READINESS_CHECKER="$SCRIPT_DIR/check_openshell_readiness.py"
VERSION_INSPECTOR="$SCRIPT_DIR/check_versions.py"

usage() {
    cat <<'EOF'
Usage: scripts/openshell/start_openshell_gateway.sh [options]

Starts or reuses the official packaged OpenShell gateway service, validates the
selected registration is authenticated, and performs a mandatory disposable
policy/selector/execution/cleanup probe. The script never launches the raw gateway binary.
Long-running service and credential ownership remains with Homebrew, systemd, or
the external operator.

Canonical operator guide: docs/source/deployment/openshell.md

Options:
  --gateway-name NAME           Registered gateway name (default: openshell).
  --workspace NAME              OpenShell workspace (default: default).
  --image-name NAME             Image used for the readiness probe.
  --policy-file PATH            Policy used for the readiness probe.
  --reuse-existing, --no-start  Do not start a local packaged service.
  --create-shared-debug-sandbox Create/reuse a persistent named debug sandbox.
  --sandbox-name NAME           Shared debug sandbox name.
  -h, --help                    Show this help.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gateway-name)
            GATEWAY_NAME="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE_NAME="$2"
            export OPENSHELL_WORKSPACE="$WORKSPACE_NAME"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --policy-file)
            POLICY_FILE="$2"
            shift 2
            ;;
        --reuse-existing|--no-start)
            START_SERVICE=false
            shift
            ;;
        --create-shared-debug-sandbox)
            CREATE_SHARED_DEBUG_SANDBOX=true
            shift
            ;;
        --sandbox-name)
            SANDBOX_NAME="$2"
            shift 2
            ;;
        --gateway-bin)
            fail "Raw openshell-gateway launch is unsupported; use the packaged service or --reuse-existing"
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

resolve_dependencies() {
    if [[ -n "${OPENSHELL_GATEWAY_LAUNCH_BIN:-}" ]]; then
        fail "OPENSHELL_GATEWAY_LAUNCH_BIN is unsupported; raw gateway processes are not an authenticated lifecycle"
    fi
    if [[ -n "${OPENSHELL_GATEWAY_ENDPOINT:-}" ]]; then
        fail "Direct OPENSHELL_GATEWAY_ENDPOINT bypasses registration authentication checks"
    fi
    case "${OPENSHELL_GATEWAY_INSECURE:-}" in
        1|true|TRUE|yes|YES|on|ON)
            fail "OPENSHELL_GATEWAY_INSECURE is forbidden for Deep Researcher Agent OpenShell lifecycle checks"
            ;;
    esac
    if [[ ! -f "$POLICY_FILE" ]]; then
        fail "Policy file not found: $POLICY_FILE; run scripts/openshell/setup_openshell.sh first"
    fi

    if [[ -z "$OPENSHELL_BIN" ]]; then
        if [[ -x "$VENV_DIR/bin/openshell" ]]; then
            OPENSHELL_BIN="$VENV_DIR/bin/openshell"
        else
            OPENSHELL_BIN="$(command -v openshell || true)"
        fi
    fi
    if [[ -z "$OPENSHELL_BIN" || ! -x "$OPENSHELL_BIN" ]]; then
        fail "OpenShell CLI not found; run scripts/openshell/setup_openshell.sh first"
    fi

    if [[ -z "$PYTHON_BIN" ]]; then
        if [[ -x "$VENV_DIR/bin/python" ]]; then
            PYTHON_BIN="$VENV_DIR/bin/python"
        else
            PYTHON_BIN="$(command -v python3 || true)"
        fi
    fi
    if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
        fail "Python 3 is required to validate structured gateway metadata"
    fi
    if [[ ! -f "$READINESS_CHECKER" ]]; then
        fail "OpenShell readiness checker not found"
    fi
    if [[ ! -f "$VERSION_INSPECTOR" ]]; then
        fail "OpenShell version inspector not found"
    fi
}

validate_gateway_registration() {
    local gateway_json
    gateway_json="$("$OPENSHELL_BIN" gateway list -o json)"
    GATEWAY_TYPE="$(GATEWAY_JSON="$gateway_json" "$PYTHON_BIN" - "$GATEWAY_NAME" <<'PY'
import json
import os
import sys

name = sys.argv[1]
try:
    payload = json.loads(os.environ["GATEWAY_JSON"])
except (KeyError, json.JSONDecodeError):
    raise SystemExit("gateway metadata is not valid JSON")
if not isinstance(payload, list):
    raise SystemExit("gateway metadata must be a JSON list")
record = next((item for item in payload if isinstance(item, dict) and item.get("name") == name), None)
if record is None:
    raise SystemExit(f"registered gateway not found: {name}")
endpoint = record.get("endpoint")
auth = record.get("auth")
if not isinstance(endpoint, str) or not endpoint.lower().startswith("https://"):
    raise SystemExit("gateway registration must use HTTPS")
if auth not in {"mtls", "oidc", "cloudflare_jwt"}:
    raise SystemExit("gateway registration must use mTLS, OIDC, or edge authentication")
print(record.get("type", "remote"))
PY
)" || fail "Gateway '$GATEWAY_NAME' is missing or not authenticated"
    "$OPENSHELL_BIN" gateway select "$GATEWAY_NAME" >/dev/null
}

start_packaged_service() {
    if [[ "$GATEWAY_TYPE" != "local" ]]; then
        fail "Remote gateway '$GATEWAY_NAME' is unreachable; this script will not start a local replacement"
    fi
    case "$(uname -s)" in
        Darwin)
            command -v brew >/dev/null 2>&1 || fail "Homebrew is required to start the packaged OpenShell service"
            brew services start nvidia/openshell/openshell >/dev/null
            ;;
        Linux)
            command -v systemctl >/dev/null 2>&1 || fail "systemctl is required to start the packaged OpenShell service"
            systemctl --user start openshell-gateway
            ;;
        *)
            fail "Unsupported operating system for packaged gateway startup"
            ;;
    esac
}

wait_for_gateway() {
    local attempt
    if "$OPENSHELL_BIN" status >/dev/null 2>&1; then
        return
    fi
    if [[ "$START_SERVICE" != "true" ]]; then
        fail "Gateway '$GATEWAY_NAME' is not reachable and --reuse-existing forbids service startup"
    fi
    start_packaged_service
    for attempt in $(seq 1 "$STATUS_ATTEMPTS"); do
        if "$OPENSHELL_BIN" status >/dev/null 2>&1; then
            return
        fi
        sleep "$POLL_DELAY"
    done
    fail "Authenticated gateway '$GATEWAY_NAME' did not become ready"
}

check_component_versions() {
    if [[ "${1:-}" == "skip-live" ]]; then
        "$PYTHON_BIN" "$VERSION_INSPECTOR" --gateway-name "$GATEWAY_NAME" --skip-live \
            || fail "OpenShell component version check failed"
    else
        "$PYTHON_BIN" "$VERSION_INSPECTOR" --gateway-name "$GATEWAY_NAME" \
            || fail "OpenShell component version check failed"
    fi
}

sandbox_is_listed() {
    "$OPENSHELL_BIN" sandbox list | grep -F "$1" >/dev/null 2>&1
}

sandbox_is_ready() {
    "$OPENSHELL_BIN" sandbox list | grep -F "$1" | grep -F "Ready" >/dev/null 2>&1
}

run_strict_readiness_check() {
    if ! "$PYTHON_BIN" "$READINESS_CHECKER" \
        --gateway-name "$GATEWAY_NAME" \
        --workspace "$WORKSPACE_NAME" \
        --image-name "$IMAGE_NAME" \
        --policy-file "$POLICY_FILE" \
        --openshell-bin "$OPENSHELL_BIN" \
        --ready-timeout-seconds "$READY_TIMEOUT_SECONDS" \
        --policy-load-timeout-seconds "$POLICY_LOAD_TIMEOUT_SECONDS"; then
        fail "OpenShell strict readiness check failed"
    fi
}

create_shared_debug_sandbox() {
    if [[ "$CREATE_SHARED_DEBUG_SANDBOX" != "true" ]]; then
        return
    fi
    if sandbox_is_ready "$SANDBOX_NAME"; then
        echo "Reusing existing shared debug sandbox: $SANDBOX_NAME"
        return
    fi
    if sandbox_is_listed "$SANDBOX_NAME"; then
        fail "Shared debug sandbox exists but is not Ready: $SANDBOX_NAME"
    fi
    "$OPENSHELL_BIN" sandbox create \
        --name "$SANDBOX_NAME" \
        --from "$IMAGE_NAME" \
        --policy "$POLICY_FILE" \
        --label deep-researcher=shared-debug \
        --no-auto-providers \
        --no-tty \
        -- true >/dev/null
    sandbox_is_ready "$SANDBOX_NAME" || fail "Shared debug sandbox did not become Ready"
    echo "Created shared debug sandbox: $SANDBOX_NAME"
}

main() {
    resolve_dependencies
    validate_gateway_registration
    check_component_versions skip-live
    wait_for_gateway
    check_component_versions
    run_strict_readiness_check
    create_shared_debug_sandbox
}

main
