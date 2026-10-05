# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LIGHTNING_AGENT_MAX_TOKENS = 32768
CONFIG_PATHS = [
    REPO_ROOT / "configs" / "config_cli_default.yml",
    REPO_ROOT / "configs" / "config_mcp.yml",
    REPO_ROOT / "configs" / "config_openshell.yml",
    REPO_ROOT / "configs" / "config_web_azure_ai_search.yml",
    REPO_ROOT / "configs" / "config_web_default_guardrails.yml",
    REPO_ROOT / "configs" / "config_web_default_llamaindex.yml",
    REPO_ROOT / "configs" / "config_web_frag.yml",
    REPO_ROOT / "configs" / "config_web_frag_mcp_auth.yml",
    REPO_ROOT / "configs" / "config_web_nemo_retriever.yml",
    REPO_ROOT / "configs" / "config_web_opensearch.yml",
    REPO_ROOT / "configs" / "nemo_relay" / "config_web_default_with_pricing.yml",
    REPO_ROOT / ".agents" / "skills" / "aiq-configure-workflow" / "assets" / "config-scaffold.yml",
]


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_lightning_agent_llm_uses_full_output_cap(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lightning_agent_llm = config["llms"]["nemotron_lightning_agent_llm"]

    assert lightning_agent_llm.get("max_tokens") == EXPECTED_LIGHTNING_AGENT_MAX_TOKENS
