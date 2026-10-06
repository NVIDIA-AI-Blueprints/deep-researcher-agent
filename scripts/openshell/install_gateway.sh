#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Explicitly install the certified packaged OpenShell gateway for local Apple Silicon demos.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DRY_RUN=false
ASSUME_YES=false
USE_COLIMA=false
DOCKER_HOST_VALUE=""
GATEWAY_NAME="${DEEP_RESEARCHER_OPENSHELL_GATEWAY_NAME:-openshell}"

usage() {
    cat <<'EOF'
Usage: scripts/openshell/install_gateway.sh [options]

Installs or reinstalls Deep Researcher Agent's certified OpenShell release through OpenShell's
official tagged installer. This is local-demo tooling for Apple Silicon macOS;
Linux and remote gateways remain operator-owned.

Options:
  --dry-run                  Print the planned release and operations only.
  --yes                      Skip the interactive confirmation.
  --colima                   Persist the Docker driver and default Colima socket.
  --docker-host unix://PATH  Persist a specific local Docker socket (implies --colima).
  --gateway-name NAME        Registered local gateway name (default: openshell).
  -h, --help                 Show this help.

Canonical operator guide: docs/source/deployment/openshell.md
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes)
            ASSUME_YES=true
            shift
            ;;
        --colima)
            USE_COLIMA=true
            shift
            ;;
        --docker-host)
            [[ $# -ge 2 ]] || fail "--docker-host requires a unix:// path"
            DOCKER_HOST_VALUE="$2"
            USE_COLIMA=true
            shift 2
            ;;
        --gateway-name)
            [[ $# -ge 2 ]] || fail "--gateway-name requires a name"
            GATEWAY_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "Unknown option: $1"
            ;;
    esac
done

[[ "$(id -u)" -ne 0 ]] || fail "Run this installer as the logged-in macOS user, not root"
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] \
    || fail "The Deep Researcher Agent gateway installer supports Apple Silicon macOS local gateways only"
command -v brew >/dev/null 2>&1 || fail "Homebrew is required"
[[ -x "$PYTHON_BIN" ]] || fail "Deep Researcher Agent Python was not found; run ./scripts/setup.sh first"

RELEASE_TAG="$($PYTHON_BIN "$SCRIPT_DIR/version_contract.py" --field release-tag)"
EXPECTED_SHA256="$($PYTHON_BIN "$SCRIPT_DIR/version_contract.py" --field installer-sha256)"
INSTALLER_URL="https://raw.githubusercontent.com/NVIDIA/OpenShell/$RELEASE_TAG/install.sh"

if [[ -n "$DOCKER_HOST_VALUE" && ! "$DOCKER_HOST_VALUE" =~ ^unix://[^[:space:]]+$ ]]; then
    fail "--docker-host must be a unix:// path"
fi
if [[ "$USE_COLIMA" == "true" && -z "$DOCKER_HOST_VALUE" ]]; then
    DOCKER_HOST_VALUE="unix://$HOME/.colima/default/docker.sock"
fi

installed_formulas="$(brew list --formula --full-name 2>/dev/null \
    | awk -F/ '$NF == "openshell" {print}' \
    | sort -u || true)"
formula_count="$(printf '%s\n' "$installed_formulas" | awk 'NF {count++} END {print count+0}')"
if [[ "$formula_count" -gt 1 ]]; then
    fail "ambiguous_gateway_installation: multiple OpenShell formulas are installed"
fi
if [[ "$formula_count" -eq 1 ]]; then
    installed_formula="$(printf '%s\n' "$installed_formulas" | awk 'NF {print; exit}')"
    if [[ "$installed_formula" != "nvidia/openshell/openshell" ]]; then
        fail "ambiguous_gateway_installation: remove the non-official OpenShell formula before continuing"
    fi
fi
services_json="$(brew services list --json 2>/dev/null || printf '[]')"
service_names="$(BREW_SERVICES_JSON="$services_json" "$PYTHON_BIN" - <<'PY'
import json
import os

try:
    services = json.loads(os.environ.get("BREW_SERVICES_JSON", "[]"))
except json.JSONDecodeError:
    services = []
names = {
    item["name"]
    for item in services
    if isinstance(item, dict)
    and isinstance(item.get("name"), str)
    and item["name"].split("/")[-1] == "openshell"
}
print("\n".join(sorted(names)))
PY
)"
service_count="$(printf '%s\n' "$service_names" | awk 'NF {count++} END {print count+0}')"
if [[ "$service_count" -gt 1 ]]; then
    fail "ambiguous_gateway_installation: multiple OpenShell services are registered"
fi

cat <<EOF
Certified release: $RELEASE_TAG
Installer: $INSTALLER_URL
Gateway owner: official nvidia/openshell Homebrew service
EOF
if [[ "$USE_COLIMA" == "true" ]]; then
    echo "Persistent driver configuration: ~/.config/openshell/gateway.env"
fi
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete; no files, taps, packages, or services were changed."
    exit 0
fi

if [[ "$ASSUME_YES" != "true" ]]; then
    [[ -t 0 ]] || fail "Interactive confirmation is unavailable; rerun with --yes"
    read -r -p "Install or reinstall packaged OpenShell $RELEASE_TAG? [y/N] " answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) echo "Installation cancelled."; exit 0 ;;
    esac
fi

installer_file="$(mktemp -t deep-researcher-openshell-installer.XXXXXX)"
trap 'rm -f "$installer_file"' EXIT
if ! curl -fLsS --retry 3 --max-redirs 5 -o "$installer_file" "$INSTALLER_URL"; then
    fail "gateway_installer_download_failed"
fi
actual_sha256="$(shasum -a 256 "$installer_file" | awk '{print $1}')"
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || fail "gateway_installer_checksum_mismatch"

if [[ "$USE_COLIMA" == "true" ]]; then
    "$PYTHON_BIN" - "$HOME/.config/openshell/gateway.env" "$DOCKER_HOST_VALUE" <<'PY'
from pathlib import Path
import os
import stat
import sys

path = Path(sys.argv[1])
docker_host = sys.argv[2]
updates = {"OPENSHELL_DRIVERS": "docker", "DOCKER_HOST": docker_host}
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
result = []
written = set()
for line in lines:
    key = line.split("=", 1)[0].strip() if "=" in line else ""
    if key in updates:
        if key not in written:
            result.append(f"{key}={updates[key]}")
            written.add(key)
        continue
    result.append(line)
for key, value in updates.items():
    if key not in written:
        result.append(f"{key}={value}")
path.parent.mkdir(parents=True, exist_ok=True)
mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text("\n".join(result) + "\n", encoding="utf-8")
temporary.chmod(mode)
os.replace(temporary, path)
PY
fi

OPENSHELL_VERSION="$RELEASE_TAG" sh "$installer_file"

if ! "$PYTHON_BIN" "$SCRIPT_DIR/check_versions.py" --gateway-name "$GATEWAY_NAME"; then
    fail "OpenShell installation completed but component verification failed"
fi
echo "Packaged OpenShell gateway $RELEASE_TAG is installed and verified."
