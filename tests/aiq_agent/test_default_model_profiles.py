# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from aiq_agent.agents.shallow_researcher.agent import ShallowResearcherAgent
from aiq_agent.common import LLMProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SHALLOW_PROMPT = REPO_ROOT / "src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2"
BREV_GETTING_STARTED_NOTEBOOK = REPO_ROOT / "docs/notebooks/0_Getting_Started_with_Deep_Researcher_Agent.ipynb"

ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
LIGHTNING_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
BUILD_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LIGHTNING_AGENT_MAX_TOKENS = 32768
BENCHMARK_LIGHTNING_AGENT_MAX_TOKENS = 8192

CONFIG_GLOBS = (
    ".agents/skills/aiq-configure-workflow/assets/config-scaffold.yml",
    "configs/config_*.yml",
    "frontends/benchmarks/**/configs/*.yml",
)
CONFIG_PATHS = tuple(sorted(path for pattern in CONFIG_GLOBS for path in REPO_ROOT.glob(pattern)))
FRESHQA_CONFIG_PATHS = tuple(sorted(REPO_ROOT.glob("frontends/benchmarks/freshqa/configs/*.yml")))
SHALLOW_PROFILE_PATHS = (
    REPO_ROOT / "configs/config_web_default_guardrails.yml",
    REPO_ROOT / "configs/config_frontier_models.yml",
)

DEPRECATED_REFERENCES = (
    "/".join(("nvidia", "nemotron-3-super-120b-a12b")),
    "/".join(("nvidia", "nemotron-3-nano-30b-a3b")),
    "/".join(("nvidia", "nemotron-mini-4b-instruct")),
    "/".join(("nvidia", "llama-nemotron-embed-vl-1b-v2")),
    "/".join(("nvidia", "nemotron-nano-12b-v2-vl")),
    "/".join(("openai", "gpt-oss-120b")),
    ".".join(("inference-api", "nvidia", "com")),
)
SCANNED_SUFFIXES = {
    ".baseline",
    ".example",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "results",
}


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_for_alias(config: dict, alias: str) -> str:
    return config["llms"][alias]["model_name"]


def _thinking_enabled(config: dict, alias: str) -> bool:
    return bool(config["llms"][alias].get("chat_template_kwargs", {}).get("enable_thinking", False))


def _registered_source_tools(config: dict) -> set[str]:
    functions = config.get("functions", {})
    registries = (
        function
        for function in functions.values()
        if isinstance(function, dict) and function.get("_type") == "data_source_registry"
    )
    return {
        tool for registry in registries for source in registry.get("sources", []) for tool in source.get("tools", [])
    }


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_default_profiles_use_role_appropriate_models(config_path: Path):
    config = _load_config(config_path)
    functions = config.get("functions", {})
    is_frontier_profile = config_path.name == "config_frontier_models.yml"
    is_benchmark_profile = REPO_ROOT / "frontends" / "benchmarks" in config_path.parents

    for function in functions.values():
        if not isinstance(function, dict):
            continue

        function_type = function.get("_type")
        if function_type == "intent_classifier":
            if is_frontier_profile:
                continue
            alias = function["llm"]
            assert alias == "nemotron_lightning_intent_llm"
            assert _model_for_alias(config, alias) == LIGHTNING_MODEL
            assert config["llms"][alias]["base_url"] == BUILD_BASE_URL
            assert config["llms"][alias]["api_key"] == "${NVIDIA_API_KEY}"
            assert config["llms"][alias]["temperature"] == 0.1
            assert config["llms"][alias]["top_p"] == 0.9
            assert config["llms"][alias]["max_tokens"] == 1024
            assert not config["llms"][alias]["parallel_tool_calls"]
            assert not _thinking_enabled(config, alias)
        elif function_type == "shallow_research_agent":
            if is_frontier_profile:
                continue
            alias = function["llm"]
            assert alias == "nemotron_lightning_agent_llm"
            assert _model_for_alias(config, alias) == LIGHTNING_MODEL
            assert config["llms"][alias]["base_url"] == BUILD_BASE_URL
            assert config["llms"][alias]["api_key"] == "${NVIDIA_API_KEY}"
            assert config["llms"][alias]["temperature"] == 0.2
            assert config["llms"][alias]["top_p"] == 0.7
            expected_max_tokens = (
                BENCHMARK_LIGHTNING_AGENT_MAX_TOKENS if is_benchmark_profile else DEFAULT_LIGHTNING_AGENT_MAX_TOKENS
            )
            assert config["llms"][alias]["max_tokens"] == expected_max_tokens
            assert not config["llms"][alias]["parallel_tool_calls"]
            assert _thinking_enabled(config, alias)
        elif not is_frontier_profile and function_type == "clarifier_agent":
            assert _model_for_alias(config, function["llm"]) == ULTRA_MODEL
        elif not is_frontier_profile and function_type == "deep_research_agent":
            assert function["writer_llm"] == "nemotron_ultra_writer_llm"
            for role in (
                "orchestrator_llm",
                "source_router_llm",
                "researcher_llm",
                "planner_llm",
                "writer_llm",
            ):
                assert _model_for_alias(config, function[role]) == ULTRA_MODEL


@pytest.mark.parametrize("config_path", FRESHQA_CONFIG_PATHS, ids=lambda path: path.name)
def test_freshqa_research_tools_are_registered_data_sources(config_path: Path):
    config = _load_config(config_path)
    source_tools = _registered_source_tools(config)

    for function in config.get("functions", {}).values():
        if isinstance(function, dict) and function.get("_type") in {
            "shallow_research_agent",
            "deep_research_agent",
        }:
            assert set(function.get("tools", [])) <= source_tools


@pytest.mark.parametrize("config_path", SHALLOW_PROFILE_PATHS, ids=lambda path: path.name)
def test_shallow_profiles_use_the_shared_citation_prompt(config_path: Path):
    """Default Lightning and frontier Luna must share the hardened prompt and runtime path."""
    config = _load_config(config_path)
    shallow = config["functions"]["shallow_research_agent"]
    agent = ShallowResearcherAgent(llm_provider=MagicMock(spec=LLMProvider), tools=[])

    assert shallow["_type"] == "shallow_research_agent"
    assert "system_prompt" not in shallow
    assert agent.system_prompt == SHARED_SHALLOW_PROMPT.read_text(encoding="utf-8")


def test_brev_getting_started_uses_ultra_for_shallow_research():
    """The Brev launchable avoids the hosted Lightning shallow-serving limitation."""
    notebook = json.loads(BREV_GETTING_STARTED_NOTEBOOK.read_text(encoding="utf-8"))
    config_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and cell.get("source", [""])[0].startswith("%%writefile config_simple_researcher.yml")
    ]

    assert len(config_cells) == 1
    config = yaml.safe_load("".join(config_cells[0]["source"][2:]))
    shallow = config["functions"]["shallow_research_agent"]
    shallow_alias = shallow["llm"]
    web_search = config["functions"]["web_search_tool"]

    assert config["functions"]["intent_classifier"]["llm"] == "nemotron_lightning_intent_llm"
    assert shallow_alias == "nemotron_ultra_shallow_llm"
    assert _model_for_alias(config, shallow_alias) == ULTRA_MODEL
    assert config["llms"][shallow_alias]["max_tokens"] == 8192
    assert not config["llms"][shallow_alias]["parallel_tool_calls"]
    assert _thinking_enabled(config, shallow_alias)
    assert "nemotron_lightning_agent_llm" not in config["llms"]
    assert shallow["max_llm_turns"] == 20
    assert shallow["max_tool_iterations"] == 5
    assert web_search["max_results"] == 5
    assert web_search["max_retries"] == 3
    assert not web_search["advanced_search"]


def test_deprecated_model_and_endpoint_references_are_absent():
    violations: list[str] = []

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES or IGNORED_PARTS.intersection(path.parts):
            continue

        text = path.read_text(encoding="utf-8")
        for reference in DEPRECATED_REFERENCES:
            if reference in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {reference}")

    assert not violations, "Deprecated references remain:\n" + "\n".join(violations)
