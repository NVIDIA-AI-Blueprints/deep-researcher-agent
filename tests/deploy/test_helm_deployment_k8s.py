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

import re
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "deploy" / "helm" / "deployment-k8s"
CHILD_CHART_DIR = REPO_ROOT / "deploy" / "helm" / "helm-charts-k8s" / "deep-researcher"
CHILD_CHART_PATH = CHILD_CHART_DIR / "Chart.yaml"
PACKAGED_CHILD_CHART_PATH = CHART_PATH / "charts" / "deep-researcher-0.0.5.tgz"
HELM_README_PATH = REPO_ROOT / "deploy" / "helm" / "README.md"
KUBERNETES_DOCS_PATH = REPO_ROOT / "docs" / "source" / "deployment" / "kubernetes.md"
OPENSEARCH_VALUES_PATH = REPO_ROOT / "deploy" / "helm" / "examples" / "aws-opensearch-serverless-values.yaml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
EXPECTED_RELEASE_VERSION = "2.2.0"
EXPECTED_CHILD_CHART_VERSION = "0.0.5"


def render_chart(*extra_args: str, namespace: str = "ns-deep-researcher") -> list[dict[str, Any]]:
    result = subprocess.run(
        ["helm", "template", "deep-researcher", str(CHART_PATH), "-n", namespace, *extra_args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def walk_values(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_values(item)


def test_release_version_matches_helm_chart_images_and_docs():
    with PYPROJECT_PATH.open("rb") as pyproject_file:
        package_version = tomllib.load(pyproject_file)["project"]["version"]

    parent_chart = yaml.safe_load((CHART_PATH / "Chart.yaml").read_text(encoding="utf-8"))
    child_chart = yaml.safe_load(CHILD_CHART_PATH.read_text(encoding="utf-8"))
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text(encoding="utf-8"))
    opensearch_values = yaml.safe_load(OPENSEARCH_VALUES_PATH.read_text(encoding="utf-8"))

    with tarfile.open(PACKAGED_CHILD_CHART_PATH, "r:gz") as archive:
        packaged_files = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            packaged_file = archive.extractfile(member)
            assert packaged_file is not None
            relative_path = Path(member.name).relative_to("deep-researcher").as_posix()
            packaged_files[relative_path] = packaged_file.read()

    source_files = {
        path.relative_to(CHILD_CHART_DIR).as_posix(): path.read_bytes()
        for path in CHILD_CHART_DIR.rglob("*")
        if path.is_file()
    }
    packaged_child_chart = yaml.safe_load(packaged_files["Chart.yaml"])

    assert package_version == EXPECTED_RELEASE_VERSION
    assert parent_chart["version"] == EXPECTED_RELEASE_VERSION
    assert parent_chart["appVersion"] == EXPECTED_RELEASE_VERSION
    assert child_chart["appVersion"] == EXPECTED_RELEASE_VERSION
    assert child_chart["version"] == EXPECTED_CHILD_CHART_VERSION
    assert packaged_child_chart == child_chart
    assert packaged_files.keys() == source_files.keys()
    assert {path: content for path, content in packaged_files.items() if path != "Chart.yaml"} == {
        path: content for path, content in source_files.items() if path != "Chart.yaml"
    }
    assert parent_chart["dependencies"][0]["version"] == EXPECTED_CHILD_CHART_VERSION
    assert values["deep-researcher"]["apps"]["backend"]["image"]["tag"] == EXPECTED_RELEASE_VERSION
    assert values["deep-researcher"]["apps"]["frontend"]["image"]["tag"] == EXPECTED_RELEASE_VERSION
    assert opensearch_values["deep-researcher"]["apps"]["backend"]["image"]["tag"] == EXPECTED_RELEASE_VERSION

    rendered_images = {
        manifest["metadata"]["name"]: manifest["spec"]["template"]["spec"]["containers"][0]["image"]
        for manifest in render_chart()
        if manifest.get("kind") == "Deployment"
    }
    assert (
        rendered_images["deep-researcher-backend"] == f"nvcr.io/nvidia/blueprint/aiq-agent:{EXPECTED_RELEASE_VERSION}"
    )
    assert (
        rendered_images["deep-researcher-frontend"]
        == f"nvcr.io/nvidia/blueprint/aiq-frontend:{EXPECTED_RELEASE_VERSION}"
    )

    expected_chart_archive = f"deep-researcher-web-{EXPECTED_RELEASE_VERSION}.tgz"
    chart_archive_pattern = re.compile(r"deep-researcher-web-[\w.-]+\.tgz")
    for documentation_path in (HELM_README_PATH, KUBERNETES_DOCS_PATH):
        documentation = documentation_path.read_text(encoding="utf-8")
        assert set(chart_archive_pattern.findall(documentation)) == {expected_chart_archive}
        assert f"**deep-researcher-web** version **{EXPECTED_RELEASE_VERSION}**" in documentation


def test_default_chart_renders_referenced_configmaps_and_uses_user_supplied_secret():
    manifests = render_chart()

    rendered_configmaps = {
        manifest["metadata"]["name"] for manifest in manifests if manifest.get("kind") == "ConfigMap"
    }
    rendered_secrets = {manifest["metadata"]["name"] for manifest in manifests if manifest.get("kind") == "Secret"}

    referenced_configmaps = set()
    referenced_secrets = set()

    for manifest in manifests:
        for node in walk_values(manifest):
            if isinstance(node.get("configMap"), dict) and node["configMap"].get("name"):
                referenced_configmaps.add(node["configMap"]["name"])
            if isinstance(node.get("secretRef"), dict) and node["secretRef"].get("name"):
                referenced_secrets.add(node["secretRef"]["name"])
            if isinstance(node.get("secretKeyRef"), dict) and node["secretKeyRef"].get("name"):
                referenced_secrets.add(node["secretKeyRef"]["name"])

    assert referenced_configmaps <= rendered_configmaps
    assert referenced_secrets == {"deep-researcher-credentials"}
    assert "deep-researcher-credentials" not in rendered_secrets


def test_all_namespaced_resources_honor_release_namespace():
    """Regression test for #290: resources must use the Helm release namespace
    (``helm install -n <ns>``) instead of a hardcoded ``ns-deep-researcher``, so ``helm
    install -n`` and GitOps operators (ArgoCD, Fleet) target the right namespace.
    """
    release_namespace = "my-namespace"
    manifests = render_chart(namespace=release_namespace)

    namespaced = [manifest for manifest in manifests if manifest.get("metadata", {}).get("namespace") is not None]

    # The chart must render at least one namespaced resource, otherwise this
    # test would pass vacuously if templating silently stopped emitting them.
    assert namespaced, "expected the chart to render namespaced resources"

    offenders = {
        (manifest.get("kind"), manifest["metadata"].get("name")): manifest["metadata"]["namespace"]
        for manifest in namespaced
        if manifest["metadata"]["namespace"] != release_namespace
    }
    assert not offenders, f"resources not pinned to release namespace {release_namespace!r}: {offenders}"


def test_chart_renders_app_host_aliases(tmp_path: Path):
    values_file = tmp_path / "host-aliases.yaml"
    values_file.write_text(
        """
deep-researcher:
  apps:
    backend:
      hostAliases:
        - ip: "127.0.0.1"
          hostnames:
            - "deep-researcher.local"
""",
        encoding="utf-8",
    )

    manifests = render_chart("-f", str(values_file))

    backend_deployment = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "deep-researcher-backend"
    )

    assert backend_deployment["spec"]["template"]["spec"]["hostAliases"] == [
        {"ip": "127.0.0.1", "hostnames": ["deep-researcher.local"]}
    ]


def test_default_chart_does_not_provision_or_configure_redis():
    manifests = render_chart()

    redis_resources = {
        (manifest.get("kind"), manifest.get("metadata", {}).get("name"))
        for manifest in manifests
        if manifest.get("metadata", {}).get("name", "").startswith("deep-researcher-redis")
    }
    assert redis_resources == set()

    backend_deployment = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "deep-researcher-backend"
    )
    backend_env = {
        item["name"] for item in backend_deployment["spec"]["template"]["spec"]["containers"][0].get("env", [])
    }
    assert backend_env.isdisjoint({"MCP_TOKEN_STORE_TYPE", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"})


def test_upload_limits_are_aligned_between_backend_and_frontend():
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text(encoding="utf-8"))
    backend_env = values["deep-researcher"]["apps"]["backend"]["env"]
    frontend_env = values["deep-researcher"]["apps"]["frontend"]["env"]
    upload_variables = {
        "FILE_UPLOAD_ACCEPTED_TYPES",
        "FILE_UPLOAD_MAX_SIZE_MB",
        "FILE_UPLOAD_MAX_FILE_COUNT",
    }

    assert {name: backend_env[name] for name in upload_variables} == {
        name: frontend_env[name] for name in upload_variables
    }


def test_backend_wires_default_on_deep_research_admission_limits():
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text(encoding="utf-8"))
    backend_env = values["deep-researcher"]["apps"]["backend"]["env"]

    assert backend_env["DEEP_RESEARCHER_MAX_DEEP_RESEARCH_INPUT_CHARS"] == "32768"
    assert backend_env["DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_PER_PRINCIPAL"] == "5"
    assert backend_env["DEEP_RESEARCHER_MAX_ACTIVE_DEEP_RESEARCH_JOBS_GLOBAL"] == "50"
    assert backend_env["DEEP_RESEARCHER_MAX_DEEP_RESEARCH_SUBMISSIONS_PER_MINUTE"] == "20"


def test_backend_uses_separate_liveness_and_readiness_endpoints():
    manifests = render_chart()

    backend_deployment = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "deep-researcher-backend"
    )
    backend_container = backend_deployment["spec"]["template"]["spec"]["containers"][0]

    assert backend_container["livenessProbe"]["httpGet"]["path"] == "/live"
    assert backend_container["readinessProbe"]["httpGet"]["path"] == "/health"


def test_shared_dask_example_renders_secured_external_scheduler_and_workers():
    manifests = render_chart("-f", str(REPO_ROOT / "deploy" / "helm" / "examples" / "shared-dask-values.yaml"))
    deployments = {
        manifest["metadata"]["name"]: manifest for manifest in manifests if manifest.get("kind") == "Deployment"
    }
    services = {manifest["metadata"]["name"]: manifest for manifest in manifests if manifest.get("kind") == "Service"}
    network_policies = {
        manifest["metadata"]["name"]: manifest for manifest in manifests if manifest.get("kind") == "NetworkPolicy"
    }

    backend = deployments["deep-researcher-backend"]["spec"]["template"]["spec"]["containers"][0]
    scheduler = deployments["deep-researcher-dask-scheduler"]["spec"]["template"]["spec"]["containers"][0]
    worker_deployment = deployments["deep-researcher-dask-worker"]
    worker = worker_deployment["spec"]["template"]["spec"]["containers"][0]

    backend_env = {item["name"]: item["value"] for item in backend["env"] if "value" in item}
    scheduler_env = {item["name"]: item["value"] for item in scheduler["env"] if "value" in item}
    worker_env = {item["name"]: item["value"] for item in worker["env"] if "value" in item}
    for deployment_name, container, volume_name, secret_name in (
        ("deep-researcher-backend", backend, "dask-client-tls", "deep-researcher-dask-client-tls"),
        ("deep-researcher-dask-scheduler", scheduler, "dask-scheduler-tls", "deep-researcher-dask-scheduler-tls"),
        ("deep-researcher-dask-worker", worker, "dask-worker-tls", "deep-researcher-dask-worker-tls"),
    ):
        pod_spec = deployments[deployment_name]["spec"]["template"]["spec"]
        tls_volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == volume_name)
        tls_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == volume_name)
        assert tls_volume["secret"]["secretName"] == secret_name
        assert tls_mount == {"name": volume_name, "mountPath": "/etc/dask-tls", "readOnly": True}

    assert backend_env["NAT_DASK_SCHEDULER_ADDRESS"] == "tls://deep-researcher-dask-scheduler:8786"
    assert backend_env["DASK_DISTRIBUTED__COMM__REQUIRE_ENCRYPTION"] == "true"
    assert scheduler["command"] == ["dask-scheduler"]
    assert scheduler["args"][scheduler["args"].index("--protocol") + 1] == "tls"
    assert scheduler["readinessProbe"]["tcpSocket"]["port"] == 8786
    assert scheduler_env["DASK_DISTRIBUTED__COMM__REQUIRE_ENCRYPTION"] == "true"
    assert "envFrom" not in scheduler
    assert worker_deployment["spec"]["replicas"] == 4
    assert worker["command"] == ["dask-worker"]
    assert worker["args"][0] == "tls://deep-researcher-dask-scheduler:8786"
    assert "--tls-ca-file" in worker["args"]
    assert worker_env["NAT_DASK_SCHEDULER_ADDRESS"] == "tls://deep-researcher-dask-scheduler:8786"
    assert worker_env["CONFIG_FILE"] == "configs/config_web_default_llamaindex.yml"
    assert "deep-researcher-dask-scheduler" in services
    assert "deep-researcher-dask-worker" not in services

    scheduler_policy = network_policies["deep-researcher-dask-scheduler"]["spec"]
    allowed_apps = {peer["podSelector"]["matchLabels"]["app"] for peer in scheduler_policy["ingress"][0]["from"]}
    assert scheduler_policy["podSelector"]["matchLabels"]["app"] == "deep-researcher-dask-scheduler"
    assert allowed_apps == {"deep-researcher-backend", "deep-researcher-dask-worker"}
    assert scheduler_policy["ingress"][0]["ports"] == [{"protocol": "TCP", "port": 8786}]


def test_network_policy_rejects_missing_port_allowlist():
    result = subprocess.run(
        [
            "helm",
            "template",
            "deep-researcher",
            str(CHART_PATH),
            "-f",
            str(REPO_ROOT / "deploy" / "helm" / "examples" / "shared-dask-values.yaml"),
            "--set-json",
            "deep-researcher.apps.dask-scheduler.networkPolicy.ports=[]",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "apps.dask-scheduler.networkPolicy.ports must contain at least one port" in result.stderr


def test_shared_dask_scheduler_excludes_inline_shared_secrets():
    manifests = render_chart(
        "-f",
        str(REPO_ROOT / "deploy" / "helm" / "examples" / "shared-dask-values.yaml"),
        "--set",
        "deep-researcher.secretEnvAsEnv=true",
        "--set-string",
        "deep-researcher.secretEnv.TEST_SHARED_SECRET=test-only",
    )
    deployments = {
        manifest["metadata"]["name"]: manifest for manifest in manifests if manifest.get("kind") == "Deployment"
    }
    scheduler = deployments["deep-researcher-dask-scheduler"]["spec"]["template"]["spec"]["containers"][0]
    worker = deployments["deep-researcher-dask-worker"]["spec"]["template"]["spec"]["containers"][0]
    scheduler_env_names = {item["name"] for item in scheduler["env"]}
    worker_env = {item["name"]: item.get("value") for item in worker["env"]}

    assert "TEST_SHARED_SECRET" not in scheduler_env_names
    assert worker_env["TEST_SHARED_SECRET"] == "test-only"  # pragma: allowlist secret
