# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 public configuration and packaging contract tests."""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "config_mcp.yml"
_MCP_MANIFEST_PATH = _REPO_ROOT / "mcp" / "pyproject.toml"
_ROOT_MANIFEST_PATH = _REPO_ROOT / "pyproject.toml"
_MCP_LOCK_PATH = _REPO_ROOT / "mcp" / "uv.lock"
_ROOT_LOCK_PATH = _REPO_ROOT / "uv.lock"
_SETUP_SCRIPT_PATH = _REPO_ROOT / "scripts" / "setup.sh"
_CANONICAL_SECURITY_POLICY_PATHS = (
    _REPO_ROOT / "AGENTS.md",
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "source" / "deployment" / "docker-build.md",
    _REPO_ROOT / "docs" / "source" / "integration" / "mcp-server.md",
    _REPO_ROOT / "mcp" / "README.md",
    _REPO_ROOT / "mcp" / "SECURITY.md",
)


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_strings(nested)


def _requirements_named(values: list[str], name: str) -> list[Requirement]:
    return [requirement for value in values if (requirement := Requirement(value)).name == name]


def _locked_versions(lock: dict[str, Any], name: str) -> set[Version]:
    return {Version(package["version"]) for package in lock["package"] if package["name"] == name}


def test_public_mcp_config_preserves_reference_orchestration_choices() -> None:
    config = yaml.safe_load(_CONFIG_PATH.read_text())

    assert set(config) == {"functions", "general", "llms", "workflow"}
    assert "front_end" not in config["general"]
    assert config.get("authentication") is None
    assert config.get("function_groups") is None
    assert config.get("object_stores") is None

    workflow = config["workflow"]
    assert workflow == {
        "_type": "chat_deepresearcher_agent",
        "enable_clarifier": False,
        "enable_escalation": True,
        "use_async_deep_research": False,
        "checkpoint_db": "${DEEP_RESEARCHER_CHECKPOINT_DB}",
    }

    functions = config["functions"]
    assert set(functions) == {
        "advanced_web_search_tool",
        "clarifier_agent",
        "data_sources",
        "deep_research_agent",
        "intent_classifier",
        "shallow_research_agent",
        "web_search_tool",
    }
    assert functions["intent_classifier"]["_type"] == "intent_classifier"
    clarifier = functions["clarifier_agent"]
    assert clarifier["_type"] == "clarifier_agent"
    assert clarifier["llm"] == "nemotron_ultra_llm"
    assert clarifier["max_turns"] == 3
    assert clarifier["log_response_max_chars"] == 2000
    assert "verbose" not in clarifier
    assert functions["shallow_research_agent"]["exclude_tools"] == ["advanced_web_search_tool"]
    assert functions["deep_research_agent"]["exclude_tools"] == ["web_search_tool"]

    sources = functions["data_sources"]["sources"]
    assert sources == [
        {
            "id": "web_search",
            "name": "Web Search",
            "description": "Search the public web for current information.",
            "tools": ["web_search_tool", "advanced_web_search_tool"],
        }
    ]


def test_public_mcp_config_uses_only_public_models_sources_and_environment_names() -> None:
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    text = _CONFIG_PATH.read_text().lower()

    assert {entry["base_url"] for entry in config["llms"].values()} == {"https://integrate.api.nvidia.com/v1"}
    assert all(entry["model_name"].startswith("nvidia/") for entry in config["llms"].values())
    assert config["functions"]["web_search_tool"]["_type"] == "tavily_web_search"
    assert config["functions"]["advanced_web_search_tool"]["_type"] == "tavily_web_search"
    assert "nvidia_api_key" in text
    assert "tavily_api_key" in text
    assert {value for value in _iter_strings(config) if "://" in value} == {"https://integrate.api.nvidia.com/v1"}


def test_mcp_manifest_declares_public_direct_runtime_dependencies() -> None:
    manifest = tomllib.loads(_MCP_MANIFEST_PATH.read_text())
    dependency_names = {Requirement(value).name for value in manifest["project"]["dependencies"]}

    assert dependency_names == {
        "deep-researcher-agent",
        "asyncpg",
        "langchain-core",
        "mcp",
        "msgpack",
        "nvidia-nat-core",
        "python-dotenv",
        "starlette",
        "tavily-web-search",
        "uvicorn",
    }
    assert manifest["project"]["name"] == "deep-researcher-mcp-server"
    assert manifest["project"]["license"] == "Apache-2.0"
    assert manifest["project"]["license-files"] == ["LICENSE"]
    assert "Private :: Do Not Upload" in manifest["project"]["classifiers"]
    assert manifest["project"]["scripts"] == {"deep-researcher-mcp-server": "deep_researcher_mcp.server:main"}


def test_root_workspace_excludes_the_independent_mcp_project() -> None:
    manifest = tomllib.loads(_ROOT_MANIFEST_PATH.read_text())
    workspace = manifest["tool"]["uv"]["workspace"]
    sources = manifest["tool"]["uv"]["sources"]

    assert "mcp-tests" not in manifest["dependency-groups"]
    assert "deep-researcher-mcp-server" not in manifest["dependency-groups"]["dev"]
    assert "mcp" not in workspace["members"]
    assert "mcp" in workspace["exclude"]
    assert "deep-researcher-mcp-server" not in sources
    assert sources["tavily-web-search"] == {"workspace": True}
    assert _MCP_LOCK_PATH.is_file()

    lock = tomllib.loads(_ROOT_LOCK_PATH.read_text())
    assert not any(package["name"] == "deep-researcher-mcp-server" for package in lock["package"])
    for package in lock["package"]:
        source = package["source"]
        assert len(source) == 1
        if "registry" in source:
            assert source == {"registry": "https://pypi.org/simple"}
            continue
        if "url" in source:
            assert package["name"] == "en-core-web-lg"
            assert package["version"] == "3.8.0"
            assert source == {
                "url": (
                    "https://github.com/explosion/spacy-models/releases/download/"
                    "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
                )
            }
            continue
        if "git" in source:
            assert package["name"] == "nemo-relay"
            continue
        editable = source.get("editable")
        assert isinstance(editable, str)
        assert not Path(editable).is_absolute()
        assert ".." not in Path(editable).parts


def test_root_deep_researcher_resolution_keeps_cryptography_in_the_nat_supported_range() -> None:
    manifest = tomllib.loads(_ROOT_MANIFEST_PATH.read_text())
    direct_names = {Requirement(value).name for value in manifest["project"]["dependencies"]}
    cryptography_policy = _requirements_named(manifest["tool"]["uv"]["override-dependencies"], "cryptography")

    # The published deep-researcher-agent metadata must not create a third MCP release
    # incompatibility. NAT owns the runtime requirement; the root uv policy
    # keeps Deep Researcher Agent on the compatible security floor.
    assert "cryptography" not in direct_names
    assert len(cryptography_policy) == 1
    assert Version("46.0.6") in cryptography_policy[0].specifier
    assert Version("47") not in cryptography_policy[0].specifier
    assert Version("50.0.0") not in cryptography_policy[0].specifier

    lock = tomllib.loads(_ROOT_LOCK_PATH.read_text())
    locked_versions = _locked_versions(lock, "cryptography")
    assert locked_versions
    assert all(Version("46.0.6") <= locked_version < Version("47") for locked_version in locked_versions)


def test_mcp_project_owns_its_sources_lock_and_scoped_cryptography_override() -> None:
    manifest = tomllib.loads(_MCP_MANIFEST_PATH.read_text())
    required_uv = SpecifierSet(manifest["tool"]["uv"]["required-version"])
    assert Version("0.11.24") not in required_uv
    assert Version("0.11.25") in required_uv
    assert Version("0.11.26") in required_uv
    assert 'UV_MIN_VERSION="0.11.25"' in _SETUP_SCRIPT_PATH.read_text()

    assert manifest["tool"]["uv"]["sources"] == {
        "deep-researcher-agent": {"path": "..", "editable": True},
        "knowledge-layer": {"path": "../sources/knowledge_layer", "editable": True},
        "tavily-web-search": {"path": "../sources/tavily_web_search", "editable": True},
    }

    cryptography_overrides = []
    for override in manifest["tool"]["uv"]["override-dependencies"]:
        if isinstance(override, str):
            assert Requirement(override).name != "cryptography"
            continue
        dependencies = [Requirement(value) for value in override["dependencies"]]
        if any(dependency.name == "cryptography" for dependency in dependencies):
            cryptography_overrides.append(override)

    assert cryptography_overrides == [
        {
            "package": {"name": "langchain-litellm", "version": "0.6.6"},
            "dependencies": ["cryptography>=50.0.0,<51"],
        },
        {
            "package": {"name": "nvidia-nat-core", "version": "1.8.0"},
            "dependencies": ["cryptography>=50.0.0,<51"],
        },
        {
            "package": {"name": "oci", "version": "2.178.0"},
            "dependencies": ["cryptography>=50.0.0,<51"],
        },
    ]

    lock = tomllib.loads(_MCP_LOCK_PATH.read_text())
    assert _locked_versions(lock, "cryptography") == {Version("50.0.0")}
    assert any(package["name"] == "deep-researcher-mcp-server" for package in lock["package"])

    local_sources = {
        package["name"]: package["source"]["editable"] for package in lock["package"] if "editable" in package["source"]
    }
    assert local_sources == {
        "deep-researcher-agent": "../",
        "deep-researcher-mcp-server": ".",
        "knowledge-layer": "../sources/knowledge_layer",
        "tavily-web-search": "../sources/tavily_web_search",
    }


def test_mcp_cryptography_policy_is_consistent_across_canonical_docs() -> None:
    for path in _CANONICAL_SECURITY_POLICY_PATHS:
        text = path.read_text()
        assert "cryptography==50.0.0" in text, path
        assert "cryptography==48.0.1" not in text, path

    security_policy = (_REPO_ROOT / "mcp" / "SECURITY.md").read_text()
    for platform_contract in (
        "Linux x86_64 with CPython 3.13",
        "x86_64 macOS",
        "32-bit Windows",
    ):
        assert platform_contract in security_policy
