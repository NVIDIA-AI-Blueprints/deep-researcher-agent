# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from nat.utils.io.yaml_tools import yaml_load

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "deploy" / "compose" / "docker-compose.yaml"
COMPOSE_INIT_DB_PATH = REPO_ROOT / "deploy" / "compose" / "init-db.sql"
COMPOSE_README_PATH = REPO_ROOT / "deploy" / "compose" / "README.md"
HELM_INIT_DB_PATH = REPO_ROOT / "deploy" / "helm" / "helm-charts-k8s" / "deep-researcher" / "files" / "init-db.sql"
DOCS_COMPOSE_PATH = REPO_ROOT / "docs" / "source" / "deployment" / "docker-compose.md"
DOCS_PROJECT_PATH = REPO_ROOT / "docs" / "source" / "project.json"
DOCS_VERSIONS_PATH = REPO_ROOT / "docs" / "source" / "versions1.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
PER_USER_AUTH_COMPOSE_PATH = REPO_ROOT / "deploy" / "compose" / "docker-compose.per-user-auth.yaml"
MCP_CONFIG_PATH = REPO_ROOT / "configs" / "config_web_frag_mcp_auth.yml"
EXPECTED_RELEASE_VERSION = "2.2.0"


def load_compose() -> dict[str, Any]:
    with COMPOSE_PATH.open(encoding="utf-8") as compose_file:
        return yaml.safe_load(compose_file)


def test_release_version_matches_default_compose_images():
    with PYPROJECT_PATH.open("rb") as pyproject_file:
        package_version = tomllib.load(pyproject_file)["project"]["version"]

    compose = load_compose()

    assert package_version == EXPECTED_RELEASE_VERSION
    assert compose["services"]["deep-researcher-agent"]["image"] == (
        f"${{BACKEND_IMAGE:-nvcr.io/nvidia/blueprint/aiq-agent:{package_version}}}"
    )
    assert compose["services"]["frontend"]["image"] == (
        f"${{FRONTEND_IMAGE:-nvcr.io/nvidia/blueprint/aiq-frontend:{package_version}}}"
    )

    expected_backend_image = f"nvcr.io/nvidia/blueprint/aiq-agent:{package_version}"
    expected_frontend_image = f"nvcr.io/nvidia/blueprint/aiq-frontend:{package_version}"
    for release_path in (COMPOSE_PATH, COMPOSE_README_PATH, DOCS_COMPOSE_PATH):
        release_text = release_path.read_text(encoding="utf-8")
        backend_images = set(re.findall(r"nvcr\.io/nvidia/blueprint/deep-researcher-agent:[\w.-]+", release_text))
        frontend_images = set(re.findall(r"nvcr\.io/nvidia/blueprint/deep-researcher-frontend:[\w.-]+", release_text))
        assert backend_images == {expected_backend_image}
        assert frontend_images == {expected_frontend_image}


def test_release_version_matches_published_docs():
    project_metadata = json.loads(DOCS_PROJECT_PATH.read_text(encoding="utf-8"))
    published_versions = json.loads(DOCS_VERSIONS_PATH.read_text(encoding="utf-8"))
    preferred_versions = [entry for entry in published_versions if entry.get("preferred")]

    assert project_metadata["version"] == EXPECTED_RELEASE_VERSION
    assert preferred_versions == [
        {
            "preferred": True,
            "version": EXPECTED_RELEASE_VERSION,
            "url": f"https://docs.nvidia.com/aiq-blueprint/{EXPECTED_RELEASE_VERSION}/",
        }
    ]


def test_default_compose_does_not_provision_or_configure_redis():
    compose = load_compose()
    services = compose["services"]
    backend = services["deep-researcher-agent"]

    assert "redis" not in services
    assert "redis-data" not in compose.get("volumes", {})
    assert "redis" not in backend.get("depends_on", {})

    backend_env = {
        entry.split("=", maxsplit=1)[0] for entry in backend.get("environment", []) if isinstance(entry, str)
    }
    assert backend_env.isdisjoint({"MCP_TOKEN_STORE_TYPE", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"})


def test_upload_limits_are_aligned_between_backend_and_frontend():
    compose = load_compose()

    def environment(service_name: str) -> dict[str, str]:
        return {
            name: value
            for entry in compose["services"][service_name]["environment"]
            for name, value in [entry.split("=", maxsplit=1)]
        }

    backend = environment("deep-researcher-agent")
    frontend = environment("frontend")
    upload_variables = {
        "FILE_UPLOAD_ACCEPTED_TYPES",
        "FILE_UPLOAD_MAX_SIZE_MB",
        "FILE_UPLOAD_MAX_FILE_COUNT",
    }

    assert {name: backend[name] for name in upload_variables} == {name: frontend[name] for name in upload_variables}


def test_backend_wires_default_on_deep_research_admission_limits():
    compose = load_compose()
    backend_env = {
        name: value
        for entry in compose["services"]["deep-researcher-agent"]["environment"]
        for name, value in [entry.split("=", maxsplit=1)]
    }

    assert (
        backend_env["DEEP_RESEARCHER_MAX_DEEP_RESEARCH_INPUT_CHARS"]
        == "${DEEP_RESEARCHER_MAX_DEEP_RESEARCH_INPUT_CHARS:-32768}"
    )
    assert backend_env["DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_PER_PRINCIPAL"] == (
        "${DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_PER_PRINCIPAL:-5}"
    )
    assert backend_env["DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_GLOBAL"] == (
        "${DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_GLOBAL:-50}"
    )
    assert backend_env["DEEP_RESEARCHER_MAX_DEEP_RESEARCH_SUBMISSIONS_PER_MINUTE"] == (
        "${DEEP_RESEARCHER_MAX_DEEP_RESEARCH_SUBMISSIONS_PER_MINUTE:-20}"
    )


def test_database_initializers_precreate_deep_research_admission_schema():
    for init_db_path in (COMPOSE_INIT_DB_PATH, HELM_INIT_DB_PATH):
        init_sql = init_db_path.read_text(encoding="utf-8")
        assert "CREATE INDEX IF NOT EXISTS idx_job_access_conversation" in init_sql
        assert "CREATE TABLE IF NOT EXISTS deep_research_admission" in init_sql
        assert "CREATE INDEX IF NOT EXISTS idx_deep_research_admission_owner" in init_sql


def test_per_user_auth_compose_adds_private_redis_token_store(tmp_path: Path):
    if shutil.which("docker") is None:
        pytest.skip("docker is required to validate Compose merge behavior")

    compose_version = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if compose_version.returncode != 0:
        pytest.skip("docker compose is required to validate Compose merge behavior")

    # The runtime env file is intentionally untracked. Render from a temporary
    # copy so this merge test works in a clean checkout without developer secrets.
    base_compose = load_compose()
    base_compose["services"]["deep-researcher-agent"].pop("env_file", None)
    test_compose_path = tmp_path / "docker-compose.yaml"
    test_compose_path.write_text(yaml.safe_dump(base_compose), encoding="utf-8")

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(REPO_ROOT / "deploy" / ".env.example"),
            "-f",
            str(test_compose_path),
            "-f",
            str(PER_USER_AUTH_COMPOSE_PATH),
            "config",
            "--format",
            "json",
            "--no-env-resolution",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    compose = json.loads(result.stdout)
    backend = compose["services"]["deep-researcher-agent"]
    redis = compose["services"]["redis"]

    assert "redis-data" in compose["volumes"]
    assert backend["depends_on"]["redis"]["condition"] == "service_healthy"
    assert backend["environment"]["CONFIG_FILE"] == "/app/configs/config_web_frag_mcp_auth.yml"
    assert backend["environment"]["MCP_TOKEN_STORE_TYPE"] == "redis"
    assert backend["environment"]["REDIS_HOST"] == "redis"
    assert backend["environment"]["REDIS_PORT"] == "6379"
    assert "ports" not in redis


def test_mcp_example_config_accepts_an_optional_external_redis_password(monkeypatch):
    monkeypatch.delenv("MCP_TOKEN_STORE_TYPE", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    without_password = yaml_load(MCP_CONFIG_PATH)["object_stores"]["mcp_token_store"]
    assert without_password["_type"] == "deep_researcher_sqlite"
    assert without_password["password"] is None

    monkeypatch.setenv("MCP_TOKEN_STORE_TYPE", "redis")
    test_password = "managed-redis-password"  # pragma: allowlist secret
    monkeypatch.setenv("REDIS_PASSWORD", test_password)
    with_password = yaml_load(MCP_CONFIG_PATH)["object_stores"]["mcp_token_store"]
    assert with_password["_type"] == "redis"
    assert with_password["password"] == test_password
