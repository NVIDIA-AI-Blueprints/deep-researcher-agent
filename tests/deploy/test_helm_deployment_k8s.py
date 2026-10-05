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

import hashlib
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "deploy" / "helm" / "deployment-k8s"
CHILD_CHART_DIR = REPO_ROOT / "deploy" / "helm" / "helm-charts-k8s" / "aiq"
CHILD_CHART_PATH = CHILD_CHART_DIR / "Chart.yaml"
PACKAGED_CHILD_CHART_PATH = CHART_PATH / "charts" / "aiq-0.0.5.tgz"
HELM_README_PATH = REPO_ROOT / "deploy" / "helm" / "README.md"
KUBERNETES_DOCS_PATH = REPO_ROOT / "docs" / "source" / "deployment" / "kubernetes.md"
OPENSEARCH_VALUES_PATH = REPO_ROOT / "deploy" / "helm" / "examples" / "aws-opensearch-serverless-values.yaml"
CUSTOM_CONFIG_EXAMPLE_PATH = REPO_ROOT / "deploy" / "helm" / "examples" / "custom-config-values.yaml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
EXPECTED_RELEASE_VERSION = "2.2.0"
EXPECTED_CHILD_CHART_VERSION = "0.0.5"


def render_chart(*extra_args: str, namespace: str = "ns-aiq") -> list[dict[str, Any]]:
    result = subprocess.run(
        ["helm", "template", "aiq", str(CHART_PATH), "-n", namespace, *extra_args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def render_chart_error(*extra_args: str, namespace: str = "ns-aiq") -> str:
    """Render the chart expecting failure and return the helm stderr output."""
    result = subprocess.run(
        ["helm", "template", "aiq", str(CHART_PATH), "-n", namespace, *extra_args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "expected helm template to fail"
    return result.stderr


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
            relative_path = Path(member.name).relative_to("aiq").as_posix()
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
    assert values["aiq"]["apps"]["backend"]["image"]["tag"] == EXPECTED_RELEASE_VERSION
    assert values["aiq"]["apps"]["frontend"]["image"]["tag"] == EXPECTED_RELEASE_VERSION
    assert opensearch_values["aiq"]["apps"]["backend"]["image"]["tag"] == EXPECTED_RELEASE_VERSION

    rendered_images = {
        manifest["metadata"]["name"]: manifest["spec"]["template"]["spec"]["containers"][0]["image"]
        for manifest in render_chart()
        if manifest.get("kind") == "Deployment"
    }
    assert rendered_images["aiq-backend"] == f"nvcr.io/nvidia/blueprint/aiq-agent:{EXPECTED_RELEASE_VERSION}"
    assert rendered_images["aiq-frontend"] == f"nvcr.io/nvidia/blueprint/aiq-frontend:{EXPECTED_RELEASE_VERSION}"

    expected_chart_archive = f"aiq2-web-{EXPECTED_RELEASE_VERSION}.tgz"
    chart_archive_pattern = re.compile(r"aiq2-web-[\w.-]+\.tgz")
    for documentation_path in (HELM_README_PATH, KUBERNETES_DOCS_PATH):
        documentation = documentation_path.read_text(encoding="utf-8")
        assert set(chart_archive_pattern.findall(documentation)) == {expected_chart_archive}
        assert f"**aiq2-web** version **{EXPECTED_RELEASE_VERSION}**" in documentation


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
    assert referenced_secrets == {"aiq-credentials"}
    assert "aiq-credentials" not in rendered_secrets


def test_all_namespaced_resources_honor_release_namespace():
    """Regression test for #290: resources must use the Helm release namespace
    (``helm install -n <ns>``) instead of a hardcoded ``ns-aiq``, so ``helm
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
aiq:
  apps:
    backend:
      hostAliases:
        - ip: "127.0.0.1"
          hostnames:
            - "aiq.local"
""",
        encoding="utf-8",
    )

    manifests = render_chart("-f", str(values_file))

    backend_deployment = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "aiq-backend"
    )

    assert backend_deployment["spec"]["template"]["spec"]["hostAliases"] == [
        {"ip": "127.0.0.1", "hostnames": ["aiq.local"]}
    ]


def test_default_chart_does_not_provision_or_configure_redis():
    manifests = render_chart()

    redis_resources = {
        (manifest.get("kind"), manifest.get("metadata", {}).get("name"))
        for manifest in manifests
        if manifest.get("metadata", {}).get("name", "").startswith("aiq-redis")
    }
    assert redis_resources == set()

    backend_deployment = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "aiq-backend"
    )
    backend_env = {
        item["name"] for item in backend_deployment["spec"]["template"]["spec"]["containers"][0].get("env", [])
    }
    assert backend_env.isdisjoint({"MCP_TOKEN_STORE_TYPE", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"})


def test_upload_limits_are_aligned_between_backend_and_frontend():
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text(encoding="utf-8"))
    backend_env = values["aiq"]["apps"]["backend"]["env"]
    frontend_env = values["aiq"]["apps"]["frontend"]["env"]
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
    backend_env = values["aiq"]["apps"]["backend"]["env"]

    assert backend_env["AIQ_MAX_DEEP_RESEARCH_INPUT_CHARS"] == "32768"
    assert backend_env["AIQ_MAX_ACTIVE_DEEP_RESEARCH_JOBS_PER_PRINCIPAL"] == "5"
    assert backend_env["AIQ_MAX_ACTIVE_DEEP_RESEARCH_JOBS_GLOBAL"] == "50"
    assert backend_env["AIQ_MAX_DEEP_RESEARCH_SUBMISSIONS_PER_MINUTE"] == "20"


def test_backend_uses_separate_liveness_and_readiness_endpoints():
    manifests = render_chart()

    backend_deployment = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "aiq-backend"
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

    backend = deployments["aiq-backend"]["spec"]["template"]["spec"]["containers"][0]
    scheduler = deployments["aiq-dask-scheduler"]["spec"]["template"]["spec"]["containers"][0]
    worker_deployment = deployments["aiq-dask-worker"]
    worker = worker_deployment["spec"]["template"]["spec"]["containers"][0]

    backend_env = {item["name"]: item["value"] for item in backend["env"] if "value" in item}
    scheduler_env = {item["name"]: item["value"] for item in scheduler["env"] if "value" in item}
    worker_env = {item["name"]: item["value"] for item in worker["env"] if "value" in item}
    for deployment_name, container, volume_name, secret_name in (
        ("aiq-backend", backend, "dask-client-tls", "aiq-dask-client-tls"),
        ("aiq-dask-scheduler", scheduler, "dask-scheduler-tls", "aiq-dask-scheduler-tls"),
        ("aiq-dask-worker", worker, "dask-worker-tls", "aiq-dask-worker-tls"),
    ):
        pod_spec = deployments[deployment_name]["spec"]["template"]["spec"]
        tls_volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == volume_name)
        tls_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == volume_name)
        assert tls_volume["secret"]["secretName"] == secret_name
        assert tls_mount == {"name": volume_name, "mountPath": "/etc/dask-tls", "readOnly": True}

    assert backend_env["NAT_DASK_SCHEDULER_ADDRESS"] == "tls://aiq-dask-scheduler:8786"
    assert backend_env["DASK_DISTRIBUTED__COMM__REQUIRE_ENCRYPTION"] == "true"
    assert scheduler["command"] == ["dask-scheduler"]
    assert scheduler["args"][scheduler["args"].index("--protocol") + 1] == "tls"
    assert scheduler["readinessProbe"]["tcpSocket"]["port"] == 8786
    assert scheduler_env["DASK_DISTRIBUTED__COMM__REQUIRE_ENCRYPTION"] == "true"
    assert "envFrom" not in scheduler
    assert worker_deployment["spec"]["replicas"] == 4
    assert worker["command"] == ["dask-worker"]
    assert worker["args"][0] == "tls://aiq-dask-scheduler:8786"
    assert "--tls-ca-file" in worker["args"]
    assert worker_env["NAT_DASK_SCHEDULER_ADDRESS"] == "tls://aiq-dask-scheduler:8786"
    assert worker_env["CONFIG_FILE"] == "configs/config_web_default_llamaindex.yml"
    assert "aiq-dask-scheduler" in services
    assert "aiq-dask-worker" not in services

    scheduler_policy = network_policies["aiq-dask-scheduler"]["spec"]
    allowed_apps = {peer["podSelector"]["matchLabels"]["app"] for peer in scheduler_policy["ingress"][0]["from"]}
    assert scheduler_policy["podSelector"]["matchLabels"]["app"] == "aiq-dask-scheduler"
    assert allowed_apps == {"aiq-backend", "aiq-dask-worker"}
    assert scheduler_policy["ingress"][0]["ports"] == [{"protocol": "TCP", "port": 8786}]


def test_network_policy_rejects_missing_port_allowlist():
    result = subprocess.run(
        [
            "helm",
            "template",
            "aiq",
            str(CHART_PATH),
            "-f",
            str(REPO_ROOT / "deploy" / "helm" / "examples" / "shared-dask-values.yaml"),
            "--set-json",
            "aiq.apps.dask-scheduler.networkPolicy.ports=[]",
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
        "aiq.secretEnvAsEnv=true",
        "--set-string",
        "aiq.secretEnv.TEST_SHARED_SECRET=test-only",
    )
    deployments = {
        manifest["metadata"]["name"]: manifest for manifest in manifests if manifest.get("kind") == "Deployment"
    }
    scheduler = deployments["aiq-dask-scheduler"]["spec"]["template"]["spec"]["containers"][0]
    worker = deployments["aiq-dask-worker"]["spec"]["template"]["spec"]["containers"][0]
    scheduler_env_names = {item["name"] for item in scheduler["env"]}
    worker_env = {item["name"]: item.get("value") for item in worker["env"]}

    assert "TEST_SHARED_SECRET" not in scheduler_env_names
    assert worker_env["TEST_SHARED_SECRET"] == "test-only"  # pragma: allowlist secret


def _backend_deployment(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the ``aiq-backend`` Deployment manifest from rendered chart output."""
    return next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "aiq-backend"
    )


def _backend_env(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the backend container's ``env`` list from rendered chart output."""
    return _backend_deployment(manifests)["spec"]["template"]["spec"]["containers"][0].get("env", [])


def test_backend_config_inline_renders_configmap_and_mounts(tmp_path: Path):
    """Inline ``config.content`` renders a ConfigMap, mounts it, and sets CONFIG_FILE."""
    values_file = tmp_path / "inline-config.yaml"
    values_file.write_text(
        """
aiq:
  apps:
    backend:
      config:
        enabled: true
        content: |
          general:
            x: 1
""",
        encoding="utf-8",
    )

    manifests = render_chart("-f", str(values_file))

    configmap = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "ConfigMap" and manifest["metadata"]["name"] == "aiq-backend-config"
    )
    assert configmap["data"]["config.yml"] == "general:\n  x: 1\n"

    backend_deployment = _backend_deployment(manifests)
    pod_spec = backend_deployment["spec"]["template"]["spec"]

    app_config_volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == "app-config")
    assert app_config_volume["configMap"]["name"] == "aiq-backend-config"

    container = pod_spec["containers"][0]
    app_config_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == "app-config")
    assert app_config_mount["mountPath"] == "/app/custom-configs"
    assert app_config_mount["readOnly"] is True

    env = {item["name"]: item.get("value") for item in container.get("env", [])}
    assert env["CONFIG_FILE"] == "/app/custom-configs/config.yml"

    # The backend reads CONFIG_FILE once at startup, so a content change must roll the pods.
    annotations = backend_deployment["spec"]["template"]["metadata"]["annotations"]
    assert annotations["checksum/app-config"] == hashlib.sha256(b"general:\n  x: 1\n").hexdigest()


def test_backend_config_inline_change_rolls_pods(tmp_path: Path):
    """Editing inline ``config.content`` changes the pod-template checksum annotation."""

    def checksum(model_name: str) -> str:
        values_file = tmp_path / f"inline-{model_name}.yaml"
        values_file.write_text(
            f"""
aiq:
  apps:
    backend:
      config:
        enabled: true
        content: |
          llms:
            default_llm:
              model_name: {model_name}
""",
            encoding="utf-8",
        )
        manifests = render_chart("-f", str(values_file))
        return _backend_deployment(manifests)["spec"]["template"]["metadata"]["annotations"]["checksum/app-config"]

    assert checksum("model-a") != checksum("model-b")


def test_backend_config_existing_configmap_reference(tmp_path: Path):
    """``config.existingConfigMap`` mounts the referenced CM and renders none of its own."""
    values_file = tmp_path / "existing-config.yaml"
    values_file.write_text(
        """
aiq:
  apps:
    backend:
      config:
        enabled: true
        existingConfigMap: my-cm
""",
        encoding="utf-8",
    )

    manifests = render_chart("-f", str(values_file))

    rendered_configmaps = {
        manifest["metadata"]["name"] for manifest in manifests if manifest.get("kind") == "ConfigMap"
    }
    assert "aiq-backend-config" not in rendered_configmaps

    backend_deployment = _backend_deployment(manifests)
    pod_spec = backend_deployment["spec"]["template"]["spec"]

    app_config_volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == "app-config")
    assert app_config_volume["configMap"]["name"] == "my-cm"

    container = pod_spec["containers"][0]
    app_config_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == "app-config")
    assert app_config_mount["mountPath"] == "/app/custom-configs"

    env = {item["name"]: item.get("value") for item in container.get("env", [])}
    assert env["CONFIG_FILE"] == "/app/custom-configs/config.yml"

    # The chart cannot see the content of a user-owned ConfigMap, so it adds no checksum
    # annotation; editing that ConfigMap requires a manual restart.
    annotations = backend_deployment["spec"]["template"]["metadata"].get("annotations", {})
    assert "checksum/app-config" not in annotations


def test_backend_config_honors_custom_mount_path_and_file_name(tmp_path: Path):
    """``mountPath`` and ``fileName`` drive the ConfigMap key, the mount, and CONFIG_FILE."""
    values_file = tmp_path / "custom-paths.yaml"
    values_file.write_text(
        """
aiq:
  apps:
    backend:
      config:
        enabled: true
        fileName: my-workflow.yml
        mountPath: /app/overrides
        content: |
          general:
            x: 1
""",
        encoding="utf-8",
    )

    manifests = render_chart("-f", str(values_file))

    configmap = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "ConfigMap" and manifest["metadata"]["name"] == "aiq-backend-config"
    )
    assert set(configmap["data"]) == {"my-workflow.yml"}

    container = _backend_deployment(manifests)["spec"]["template"]["spec"]["containers"][0]
    app_config_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == "app-config")
    assert app_config_mount["mountPath"] == "/app/overrides"

    env = {item["name"]: item.get("value") for item in container.get("env", [])}
    assert env["CONFIG_FILE"] == "/app/overrides/my-workflow.yml"


def test_backend_config_rejects_enabled_without_a_source():
    """``config.enabled`` without content or existingConfigMap would mount a ConfigMap that is never created."""
    stderr = render_chart_error("--set", "aiq.apps.backend.config.enabled=true")

    assert "apps.backend.config.enabled is true but neither config.content nor config.existingConfigMap is set" in (
        stderr
    )


def test_backend_config_rejects_both_content_and_existing_configmap():
    """Supplying both sources is ambiguous, so the chart fails instead of silently picking one."""
    stderr = render_chart_error(
        "--set",
        "aiq.apps.backend.config.enabled=true",
        "--set-string",
        "aiq.apps.backend.config.content=general: {}",
        "--set-string",
        "aiq.apps.backend.config.existingConfigMap=my-cm",
    )

    assert "apps.backend.config sets both config.content and config.existingConfigMap" in stderr


def test_backend_config_rejects_mounting_over_baked_in_configs():
    """``/app/configs`` holds the image's own configs; mounting over it hides them."""
    for mount_path in ("/app/configs", "/app/configs/"):
        stderr = render_chart_error(
            "--set",
            "aiq.apps.backend.config.enabled=true",
            "--set-string",
            "aiq.apps.backend.config.content=general: {}",
            "--set-string",
            f"aiq.apps.backend.config.mountPath={mount_path}",
        )

        assert "apps.backend.config.mountPath must not be /app/configs" in stderr


def test_custom_config_example_mounts_the_documented_configmap():
    """The shipped example overlay renders and mounts the ConfigMap its comments tell users to create."""
    manifests = render_chart("-f", str(CUSTOM_CONFIG_EXAMPLE_PATH))

    rendered_configmaps = {
        manifest["metadata"]["name"] for manifest in manifests if manifest.get("kind") == "ConfigMap"
    }
    assert "aiq-backend-config" not in rendered_configmaps

    pod_spec = _backend_deployment(manifests)["spec"]["template"]["spec"]
    app_config_volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == "app-config")
    assert app_config_volume["configMap"]["name"] == "my-custom-aiq-config"

    env = {item["name"]: item.get("value") for item in pod_spec["containers"][0].get("env", [])}
    assert env["CONFIG_FILE"] == "/app/custom-configs/config.yml"


def test_backend_config_accepts_a_complete_shipped_config_via_set_file():
    """A whole shipped config passed with --set-file survives the round trip into the ConfigMap."""
    shipped_config_path = REPO_ROOT / "configs" / "config_web_default_llamaindex.yml"
    manifests = render_chart(
        "--set",
        "aiq.apps.backend.config.enabled=true",
        "--set-file",
        f"aiq.apps.backend.config.content={shipped_config_path}",
    )

    configmap = next(
        manifest
        for manifest in manifests
        if manifest.get("kind") == "ConfigMap" and manifest["metadata"]["name"] == "aiq-backend-config"
    )
    mounted_config = yaml.safe_load(configmap["data"]["config.yml"])
    shipped_config = yaml.safe_load(shipped_config_path.read_text(encoding="utf-8"))
    assert mounted_config == shipped_config

    env = {item["name"]: item.get("value") for item in _backend_env(manifests)}
    assert env["CONFIG_FILE"] == "/app/custom-configs/config.yml"


def test_backend_config_disabled_by_default():
    """With no custom config, no app-config volume/mount is added and CONFIG_FILE is the baked default."""
    manifests = render_chart()

    backend_deployment = _backend_deployment(manifests)
    pod_spec = backend_deployment["spec"]["template"]["spec"]

    volume_names = {volume["name"] for volume in pod_spec.get("volumes", [])}
    assert "app-config" not in volume_names

    mount_names = {mount["name"] for mount in pod_spec["containers"][0].get("volumeMounts", [])}
    assert "app-config" not in mount_names

    env = {item["name"]: item.get("value") for item in pod_spec["containers"][0].get("env", [])}
    assert env["CONFIG_FILE"] == "configs/config_web_default_llamaindex.yml"


def test_backend_config_overrides_baked_config_file(tmp_path: Path):
    """Enabling a custom config emits exactly one CONFIG_FILE pointing at the mount."""
    values_file = tmp_path / "override-config.yaml"
    values_file.write_text(
        """
aiq:
  apps:
    backend:
      config:
        enabled: true
        content: |
          general:
            x: 1
""",
        encoding="utf-8",
    )

    manifests = render_chart("-f", str(values_file))

    config_file_entries = [item for item in _backend_env(manifests) if item["name"] == "CONFIG_FILE"]
    assert len(config_file_entries) == 1
    assert config_file_entries[0]["value"] == "/app/custom-configs/config.yml"
