#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

CONFIG_FILE="configs/config_web_default_llamaindex.yml"
HOST="0.0.0.0"
PORT=8000

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Start the Deep Researcher Agent API backend for Agent Skill use."
            echo ""
            echo "Options:"
            echo "  --config_file PATH  Config file (default: configs/config_web_default_llamaindex.yml)"
            echo "  --host HOST         Server host (default: 0.0.0.0)"
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

if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if ! python3 - "$PROJECT_ROOT/$CONFIG_FILE" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
for line in config_path.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if re.match(r"front_end\s*:", stripped):
        sys.exit(0)
sys.exit(1)
PY
then
    echo ""
    echo "Error: Config file '$CONFIG_FILE' does not have front_end configured."
    echo "Agent Skill mode requires an API-enabled config such as configs/config_web_default_llamaindex.yml."
    echo ""
    echo "For CLI mode, use: ./scripts/start_cli.sh --config_file $CONFIG_FILE"
    exit 1
fi

cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/deploy/.env" ]; then
    set -a
    source "$PROJECT_ROOT/deploy/.env"
    set +a
    echo "Backend environment file loaded (deploy/.env)"
else
    echo "Warning: No deploy/.env file found. Copy deploy/.env.example to deploy/.env"
fi

export DEEP_RESEARCHER_DEV_ENV=skill
export DEEP_RESEARCHER_ENABLE_DEBUG=false
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

DISPLAY_HOST="$HOST"
if [[ "$DISPLAY_HOST" == "0.0.0.0" || "$DISPLAY_HOST" == "::" ]]; then
    DISPLAY_HOST="localhost"
elif [[ "$DISPLAY_HOST" == *:* && "$DISPLAY_HOST" != \[*\] ]]; then
    DISPLAY_HOST="[$DISPLAY_HOST]"
fi
SKILL_SERVER_URL="http://$DISPLAY_HOST:$PORT"

source "$VENV_DIR/bin/activate"

if ! python -c "import nat" 2>/dev/null; then
    echo "NAT is not installed in .venv. Run ./scripts/setup.sh first."
    exit 1
fi

echo ""
echo "============================================"
echo "  Deep Researcher Agent Blueprint - Agent Skill Backend"
echo "============================================"
echo ""
echo "Config:      $CONFIG_FILE"
echo "Bind Host:   $HOST"
echo "API Server:  $SKILL_SERVER_URL"
echo "Skill URL:   DEEP_RESEARCHER_SERVER_URL=$SKILL_SERVER_URL"
echo "Debug UI:    disabled"
echo ""
echo "Starting server..."
echo ""

nat serve --config_file "$CONFIG_FILE" --host "$HOST" --port "$PORT"
