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
CONFIG_FILE="configs/config_web_default_llamaindex.yml"
PORT=8000

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Start the NAT FastAPI server for deep research."
            echo ""
            echo "Options:"
            echo "  --config_file PATH  Config file (default: configs/config_web_default_llamaindex.yml)"
            echo "  --port PORT         Server port (default: 8000)"
            echo "  -h, --help          Show this help"
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
    echo "Warning: No .env file found. Copy deploy/.env.example to deploy/.env"
fi

# Suppress Python warnings unless overridden by .env
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

# Validate that config has front_end (API mode) - required for server/debug mode
if ! grep -q "front_end:" "$REPO_ROOT/$CONFIG_FILE" 2>/dev/null; then
    echo ""
    echo "Error: Config file '$CONFIG_FILE' does not have front_end configured."
    echo "This script requires a web-enabled config (e.g., config_web_default_llamaindex.yml)"
    echo ""
    echo "For CLI mode, use: ./scripts/start_cli.sh --config_file $CONFIG_FILE"
    exit 1
fi

echo ""
echo "============================================"
echo "  Deep Researcher Agent Blueprint - Server Mode"
echo "============================================"
echo ""
echo "Config: $(basename "$CONFIG_FILE")"
echo "Port:   $PORT"
echo ""
echo "--------------------------------------------"
echo "  Available Endpoints:"
echo "--------------------------------------------"
echo ""
echo "  API Documentation:"
echo "    http://localhost:$PORT/docs"
echo ""
echo "  Debug Console (Deep Research):"
echo "    http://localhost:$PORT/debug"
echo ""
echo "  Health Check:"
echo "    http://localhost:$PORT/health"
echo ""
echo "--------------------------------------------"
echo ""
echo "Starting server..."
echo ""

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"
nat serve --config_file "$CONFIG_FILE" --port $PORT
