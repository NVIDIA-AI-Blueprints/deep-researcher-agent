#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate the installable AI-Q artifact without relying on checkout metadata or
# editable imports. CI intentionally runs this against the committed HEAD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
SOURCE_ROOT="$TMP_ROOT/source"
WHEEL_DIR="$TMP_ROOT/wheel"
VENV_DIR="$TMP_ROOT/venv"
RUNTIME_ROOT="$TMP_ROOT/runtime"

cleanup() {
    rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT

mkdir -p "$SOURCE_ROOT" "$WHEEL_DIR" "$RUNTIME_ROOT"
git -C "$REPO_ROOT" archive HEAD | tar -xf - -C "$SOURCE_ROOT"
if [[ -e "$SOURCE_ROOT/.git" ]]; then
    echo "Release source export unexpectedly contains .git metadata" >&2
    exit 1
fi

uv build --wheel --out-dir "$WHEEL_DIR" "$SOURCE_ROOT"
wheel_count=$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'aiq_agent-*.whl' | wc -l | tr -d ' ')
if [[ "$wheel_count" != "1" ]]; then
    echo "Expected exactly one aiq-agent wheel, found $wheel_count" >&2
    exit 1
fi
WHEEL_PATH=$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'aiq_agent-*.whl')

UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
    uv sync --project "$SOURCE_ROOT" --frozen --group dev --no-editable --no-install-package aiq-agent
uv pip install --python "$VENV_DIR/bin/python" --no-deps --reinstall "$WHEEL_PATH"
uv pip check --python "$VENV_DIR/bin/python"

cp -R "$SOURCE_ROOT/configs" "$RUNTIME_ROOT/configs"
cd "$RUNTIME_ROOT"

"$VENV_DIR/bin/python" -I - "$SOURCE_ROOT" "$WHEEL_PATH" <<'PY'
import importlib.metadata
import sys
import zipfile
from pathlib import Path

import aiq_agent

source_root = Path(sys.argv[1]).resolve()
wheel_path = Path(sys.argv[2]).resolve()
expected = {
    "aiq_agent/agents/chat_researcher/prompts/context_aware_intent_router.j2",
    "aiq_agent/agents/chat_researcher/prompts/intent_classification.j2",
    "aiq_agent/agents/clarifier/prompts/research_clarification.j2",
    "aiq_agent/agents/data_science/prompts/agent.j2",
    "aiq_agent/agents/deep_researcher/prompts/orchestrator.j2",
    "aiq_agent/agents/deep_researcher/prompts/planner.j2",
    "aiq_agent/agents/deep_researcher/prompts/researcher.j2",
    "aiq_agent/agents/deep_researcher/prompts/source_registry.j2",
    "aiq_agent/agents/deep_researcher/prompts/source_router.j2",
    "aiq_agent/agents/deep_researcher/prompts/writer.j2",
    "aiq_agent/agents/report_rewriter/prompts/edit.j2",
    "aiq_agent/agents/shallow_researcher/prompts/researcher.j2",
    "aiq_agent/agents/deep_researcher/skills/research/data-table-analysis/SKILL.md",
    "aiq_agent/agents/deep_researcher/skills/research/forecast-analysis/SKILL.md",
    "aiq_agent/agents/deep_researcher/skills/research/lightweight-calculation/SKILL.md",
    "aiq_agent/agents/deep_researcher/skills/synthesis/long-form-report-writer/SKILL.md",
    "aiq_agent/agents/deep_researcher/skills/synthesis/prediction-report-writer/SKILL.md",
    "aiq_agent/agents/deep_researcher/skills/visualization/chart-generation/SKILL.md",
}

source_package = source_root / "src" / "aiq_agent"
actual_source = {
    path.relative_to(source_root / "src").as_posix()
    for path in (source_package / "agents").glob("*/prompts/*.j2")
}
actual_source.update(
    path.relative_to(source_root / "src").as_posix()
    for path in (source_package / "agents" / "deep_researcher" / "skills").rglob("SKILL.md")
)
if actual_source != expected:
    raise SystemExit(
        "Runtime asset manifest is stale: "
        f"missing={sorted(expected - actual_source)}, unexpected={sorted(actual_source - expected)}"
    )

for relative in expected:
    source_path = source_root / "src" / relative
    if source_path.stat().st_size == 0:
        raise SystemExit(f"Source runtime asset is empty: {relative}")

with zipfile.ZipFile(wheel_path) as wheel:
    wheel_entries = {entry.filename: entry.file_size for entry in wheel.infolist()}
for relative in expected:
    if wheel_entries.get(relative, 0) == 0:
        raise SystemExit(f"Wheel runtime asset is missing or empty: {relative}")

package_root = Path(aiq_agent.__file__).resolve().parent
venv_root = Path(sys.prefix).resolve()
try:
    package_root.relative_to(venv_root)
except ValueError as exc:
    raise SystemExit(f"aiq_agent imported outside the isolated environment: {package_root}") from exc
if source_root in package_root.parents:
    raise SystemExit(f"aiq_agent imported from the source export: {package_root}")

for relative in expected:
    installed_path = package_root / Path(relative).relative_to("aiq_agent")
    if not installed_path.is_file() or installed_path.stat().st_size == 0:
        raise SystemExit(f"Installed runtime asset is missing or empty: {relative}")

direct_url = importlib.metadata.distribution("aiq-agent").read_text("direct_url.json") or ""
if '"editable": true' in direct_url:
    raise SystemExit("aiq-agent was installed editable")

print(f"Verified {len(expected)} non-empty runtime assets in source, wheel, and installed distribution")
print(f"Verified isolated aiq_agent import: {aiq_agent.__file__}")
PY

export NVIDIA_API_KEY="ci-not-a-real-key"  # pragma: allowlist secret
export OPENAI_API_KEY="ci-not-a-real-key"  # pragma: allowlist secret
export INFERENCE_NVIDIA_API_KEY="ci-not-a-real-key"  # pragma: allowlist secret
export AIQ_DATA_SCIENCE_MODEL="openai/ci-test-model"
export AIQ_INFERENCE_BASE_URL="https://inference.invalid/v1"
export TAVILY_API_KEY="ci-not-a-real-key"  # pragma: allowlist secret
export SERPER_API_KEY="ci-not-a-real-key"  # pragma: allowlist secret
export GSF_BASE_URL="https://gsf.invalid"
export GSF_EMAIL="ci@example.invalid"
export GSF_PASSWORD="ci-not-a-real-password"  # pragma: allowlist secret
export COLLECTION_NAME="ci-test-collection"
export RAG_SERVER_URL="https://rag.invalid/v1"
export REDIS_PASSWORD="ci-not-a-real-password"  # pragma: allowlist secret
export NAT_JOB_STORE_DB_URL="sqlite+aiosqlite:///$RUNTIME_ROOT/jobs.db"
export AIQ_CHECKPOINT_DB="$RUNTIME_ROOT/checkpoints.db"
export AIQ_SUMMARY_DB="sqlite+aiosqlite:///$RUNTIME_ROOT/summaries.db"
export AIQ_CHROMA_DIR="$RUNTIME_ROOT/chroma"
export AZURE_SEARCH_ENDPOINT="https://azure-search.invalid"
export NRL_SCOPE="ci"
export MCP_TOKEN_DB="$RUNTIME_ROOT/mcp_tokens.db"
export AIQ_OPENSHELL_POLICY_FILE="$RUNTIME_ROOT/configs/openshell/aiq-research-policy.yaml"

config_count=0
while IFS= read -r config_path; do
    config_count=$((config_count + 1))
    echo "Validating installed workflow: ${config_path#"$RUNTIME_ROOT/"}"
    "$VENV_DIR/bin/nat" validate --config_file "$config_path"
done < <(find "$RUNTIME_ROOT/configs" -maxdepth 1 -type f -name 'config_*.yml' | sort)

if [[ "$config_count" -eq 0 ]]; then
    echo "No shipped top-level workflow configs were discovered" >&2
    exit 1
fi
echo "Validated $config_count shipped top-level workflow configs from the installed release artifact"
