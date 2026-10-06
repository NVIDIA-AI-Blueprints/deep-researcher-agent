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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"
CONFIG_FILE="configs/config_cli_default.yml"
CLI_VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --verbose|-v)
            CLI_VERBOSE="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config_file PATH  Config file (default: configs/config_cli_default.yml)"
            echo "  -v, --verbose       Show detailed console logs"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Available configs in configs/:"
            echo "  config_cli_default.yml  - CLI mode (default)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

if [ -f "$REPO_ROOT/deploy/.env" ]; then
    set -a
    source "$REPO_ROOT/deploy/.env"
    set +a
else
    echo "No .env file found. Copy deploy/.env.example to .env"
fi

export DEEP_RESEARCHER_DEV_ENV=cli

echo "============================================"
echo "  Deep Researcher Agent Blueprint - CLI Mode"
echo "============================================"
echo ""
echo "Config: $(basename "$CONFIG_FILE")"
[ "$CLI_VERBOSE" = "true" ] && echo "Verbose: ON" || echo "Verbose: OFF (use -v to enable)"
echo ""
echo "Type 'exit' or 'quit' to exit"
echo "--------------------------------------------"
echo ""

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"
if [ "$CLI_VERBOSE" = "true" ]; then
    "$VENV_DIR/bin/deep-researcher-research" --config_file "$CONFIG_FILE" --verbose
else
    "$VENV_DIR/bin/deep-researcher-research" --config_file "$CONFIG_FILE"
fi
