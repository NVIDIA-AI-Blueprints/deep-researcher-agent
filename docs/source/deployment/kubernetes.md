<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes (Helm)

Deploy the AI-Q blueprint to a Kubernetes cluster using the Helm charts included in the repository.

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or a local cluster such as Kind or Minikube).
- `kubectl` configured with cluster access.
- `helm` v3.x installed.
- API keys for the models and tools you plan to use (refer to [Installation -- API Key Setup](../get-started/installation.md#api-key-setup)).

## Chart Structure

The Helm deployment lives under `deploy/helm/`:

```
deploy/helm/
├── README.md                  # NGC chart install & shared configuration
├── deployment-k8s/            # Source chart wrapper (for repo-based deployments)
│   ├── Chart.yaml             #   Depends on helm-charts-k8s/aiq
│   ├── values.yaml            #   Deployment values
│   └── README.md              #   Source chart instructions & Kind local dev guide
└── helm-charts-k8s/
    └── aiq/                   # Base application Helm chart (templates, helpers)
```

## Setup

### 1. Create a namespace

```bash
kubectl create namespace ns-aiq --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Create the credentials secret

The deployment reads API keys from a Kubernetes Secret named `aiq-credentials`:

```bash
kubectl create secret generic aiq-credentials -n ns-aiq \
  --from-literal=NVIDIA_API_KEY="$NGC_API_KEY" \
  --from-literal=TAVILY_API_KEY="$TAVILY_API_KEY" \
  --from-literal=DB_USER_NAME="aiq" \
  --from-literal=DB_USER_PASSWORD="aiq_dev"
```

### 3. Create an image pull secret (NGC registry)

If you are pulling pre-built images from the NGC container registry, create a Docker registry secret:

```bash
kubectl create secret docker-registry ngc-secret -n ns-aiq \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY
```

## Deploy

### Using the NGC Helm chart

Install the pre-built chart **aiq2-web** version **2.2.0** from the NGC Helm repository (`https://helm.ngc.nvidia.com/nvidia/blueprint/charts/`).

**1. Pull the chart and install from local file:**

```bash
helm pull https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq2-web-2.2.0.tgz \
  --username='$oauthtoken' \
  --password=<YOUR_NGC_API_KEY>

# Verify the chart was pulled correctly
helm show chart aiq2-web-2.2.0.tgz

# Install from the local file
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq --create-namespace \
  --wait --timeout 10m \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret'
```

Replace `<YOUR_NGC_API_KEY>` with your NGC API key (or use `$NGC_API_KEY`). Prefer `--password=$NGC_API_KEY` to avoid exposing the key in shell history.

**2. Optional — Install directly from the chart URL** (without pulling first):

```bash
helm upgrade --install aiq https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq2-web-2.2.0.tgz \
  --username='$oauthtoken' \
  --password=$NGC_API_KEY \
  -n ns-aiq --create-namespace \
  --wait --timeout 10m \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret'
```

For the full NGC chart workflow (value overrides, upgrades, troubleshooting), refer to `deploy/helm/README.md` in the source repository.

### Using the source chart

If you cloned the repository, you can deploy from the local source chart. Refer to the source chart README at `deploy/helm/deployment-k8s/README.md` in the source repository for full details including Kind local development.

The source chart on `develop` derives every namespaced resource from Helm's
`.Release.Namespace`, which is the value passed with `-n`. The examples use `ns-aiq`,
but you can choose another namespace by replacing it consistently in Helm, `kubectl`,
Secret creation, and external bindings such as EKS Pod Identity. The
`aiq.namespace.create` value controls whether the chart renders a Namespace object; it
does not override `-n`. This behavior applies to the repository source chart and does not
change the published NGC 2.2.0 instructions above.

```bash
cd deploy/helm

helm dependency update deployment-k8s/

helm install aiq deployment-k8s/ -n ns-aiq --create-namespace
```

To use NGC container images with the source chart:

```bash
helm install aiq deployment-k8s/ -n ns-aiq --create-namespace \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set aiq.apps.backend.image.repository=nvcr.io/nvidia/blueprint/aiq-agent \
  --set aiq.apps.frontend.image.repository=nvcr.io/nvidia/blueprint/aiq-frontend
```

To use locally built images (refer to [Docker Build System](./docker-build.md)):

```bash
helm install aiq deployment-k8s/ -n ns-aiq --create-namespace \
  --set aiq.apps.backend.image.repository=aiq-agent \
  --set aiq.apps.backend.image.tag=dev \
  --set aiq.apps.backend.image.pullPolicy=IfNotPresent \
  --set aiq.apps.frontend.image.repository=aiq-frontend \
  --set aiq.apps.frontend.image.tag=dev \
  --set aiq.apps.frontend.image.pullPolicy=IfNotPresent
```

### Verify

```bash
kubectl get pods -n ns-aiq
```

Expected output:

```
NAME                            READY   STATUS    RESTARTS   AGE
aiq-backend-xxx                 1/1     Running   0          30s
aiq-frontend-xxx                1/1     Running   0          30s
aiq-postgres-xxx                1/1     Running   0          30s
```

### Health check

Once all pods are running, verify the backend is responding:

```bash
kubectl port-forward -n ns-aiq svc/aiq-backend 8000:8000 &
curl http://localhost:8000/live
curl http://localhost:8000/health
```

The chart uses `/live` for the liveness probe and `/health` for readiness. The
liveness endpoint checks only that the API process responds; database or content-
encryption outages make the pod unready without causing a restart loop.

The backend API docs are available at `http://localhost:8000/docs` while the port-forward is active.

### Access the application

```bash
# Frontend UI
kubectl port-forward -n ns-aiq svc/aiq-frontend 3000:3000

# Backend API
kubectl port-forward -n ns-aiq svc/aiq-backend 8000:8000
```

Open [http://localhost:3000](http://localhost:3000) to access the web UI.

## Configuration

### Select a built-in config

The backend loads a workflow config at startup. Switch configs with `--set`:

| Config file | Description |
|-------------|-------------|
| `configs/config_web_default_llamaindex.yml` | Default -- LlamaIndex backend (no external RAG required) |
| `configs/config_web_frag.yml` | Foundational RAG mode (requires a running RAG service) |

```bash
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set aiq.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml
```

For source chart deployments, use `deployment-k8s/` instead of the `.tgz` file (refer to [Deploy](#deploy)).

### Provide a custom config

Custom config is off by default. Enable it via `aiq.apps.backend.config` to
supply your own workflow YAML instead of one of the baked-in configs -- for
example, to point at a self-hosted NIM (see
[Swapping models](../customization/swapping-models.md)). When enabled, the
chart mounts your config at `/app/custom-configs/config.yml` and automatically
sets `CONFIG_FILE` to that path, overriding any `env.CONFIG_FILE` you also
set.

Start from a complete shipped config. A workflow config needs `general`,
`llms`, `functions`, and `workflow` sections; a fragment that only redefines an
LLM will not boot. Copy the profile you want and edit it:

```bash
cp configs/config_web_default_llamaindex.yml my-config.yml
```

| Field | Description |
|-------|-------------|
| `enabled` | Set `true` to mount a custom config and override `CONFIG_FILE`. Default `false`. |
| `content` | Inline YAML rendered into a chart-managed ConfigMap. |
| `existingConfigMap` | Name of a ConfigMap you created yourself, keyed by `fileName`. |
| `fileName` | Key in the ConfigMap and the mounted filename. Default `config.yml`. |
| `mountPath` | Directory the config is mounted into. Default `/app/custom-configs`. |

The chart rejects invalid combinations at render time rather than producing a
Deployment that crash-loops:

- `enabled: true` with neither `content` nor `existingConfigMap` set.
- Both `content` and `existingConfigMap` set -- pick one.
- `mountPath: /app/configs`, which would hide the image's baked-in configs.

**Option A -- Inline (chart renders the ConfigMap):**

Pass the whole file with `--set-file` instead of pasting YAML into a values
file:

```bash
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  --set aiq.apps.backend.config.enabled=true \
  --set-file aiq.apps.backend.config.content=my-config.yml
```

The chart hashes the rendered content into a `checksum/app-config` pod
annotation, so a later `helm upgrade` with an edited `my-config.yml` restarts
the backend and the new workflow takes effect.

**Option B -- Existing ConfigMap:**

Create the ConfigMap yourself first, keyed by the `fileName` you configure
(`config.yml` by default):

```bash
kubectl create configmap aiq-custom-config \
  --from-file=config.yml=my-config.yml -n ns-aiq
```

Then reference it in your values file:

```yaml
# aiq-custom-config-values.yaml
aiq:
  apps:
    backend:
      config:
        enabled: true
        existingConfigMap: aiq-custom-config
```

```bash
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  -f aiq-custom-config-values.yaml
```

A ready-to-edit version of this overlay ships at
`deploy/helm/examples/custom-config-values.yaml`.

GitOps users can manage the `aiq-custom-config` ConfigMap out-of-band (for
example, with Kustomize or a separate manifest) and just reference its name
here.

```{note}
The backend reads `CONFIG_FILE` once at startup, and the chart cannot see
inside a ConfigMap it does not render. After editing an `existingConfigMap`,
restart the backend yourself:

    kubectl rollout restart deployment/aiq-backend -n ns-aiq
```

## FRAG Integration

To use the Foundational RAG (FRAG) config, you need a running NVIDIA RAG Blueprint deployment. Refer to the [RAG Blueprint Helm deployment guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/docs/deploy-helm.md) for setup instructions.

### Same-cluster RAG connection

If the RAG Blueprint is deployed in the same Kubernetes cluster, use internal service DNS:

```bash
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set aiq.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml \
  --set aiq.apps.backend.env.RAG_SERVER_URL=http://rag-server.<rag-namespace>.svc.cluster.local:8081/v1 \
  --set aiq.apps.backend.env.RAG_INGEST_URL=http://ingestor-server.<rag-namespace>.svc.cluster.local:8082/v1
```

Replace `<rag-namespace>` with the namespace where the RAG Blueprint is deployed.

### External RAG connection

If the RAG service is running outside the cluster:

```bash
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set aiq.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml \
  --set aiq.apps.backend.env.RAG_SERVER_URL=http://<rag-host>:8081/v1 \
  --set aiq.apps.backend.env.RAG_INGEST_URL=http://<rag-ingest-host>:8082/v1
```

### Values file approach

For complex overrides, create a values file instead of passing many `--set` flags:

```yaml
# aiq-frag-values.yaml
aiq:
  apps:
    backend:
      env:
        CONFIG_FILE: configs/config_web_frag.yml
        RAG_SERVER_URL: http://rag-server.rag-namespace.svc.cluster.local:8081/v1
        RAG_INGEST_URL: http://ingestor-server.rag-namespace.svc.cluster.local:8082/v1
```

```bash
helm upgrade --install aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  -f aiq-frag-values.yaml
```

For complete examples with NGC-specific flags, refer to `deploy/helm/README.md` in the source repository.

## Secrets Reference

### Required

| Key | Description |
|-----|-------------|
| `NVIDIA_API_KEY` | API key for NIM inference models |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `DB_USER_NAME` | PostgreSQL username (default: `aiq`) |
| `DB_USER_PASSWORD` | PostgreSQL password (default: `aiq_dev`) |

### Optional

| Key | Description |
|-----|-------------|
| `EXA_API_KEY` | Exa API key for web search |
| `NIMBLE_API_KEY` | Nimble API key for web search |
| `SERPER_API_KEY` | Serper API key for Google search |
| `JINA_API_KEY` | Jina API key |

### Updating secrets

```bash
kubectl delete secret aiq-credentials -n ns-aiq
kubectl create secret generic aiq-credentials -n ns-aiq \
  --from-literal=NVIDIA_API_KEY="new-key" \  # pragma: allowlist secret
  --from-literal=TAVILY_API_KEY="new-key" \  # pragma: allowlist secret
  --from-literal=DB_USER_NAME="aiq" \
  --from-literal=DB_USER_PASSWORD="aiq_dev"

kubectl rollout restart deployment -n ns-aiq aiq-backend aiq-frontend
```

## Upgrade

For NGC Helm chart releases, pull the new chart archive (same NGC URL pattern with the new version) if needed, then run:

```bash
helm upgrade aiq aiq2-web-2.2.0.tgz -n ns-aiq \
  --wait --timeout 10m \
  --set 'aiq.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'aiq.apps.frontend.imagePullSecrets[0].name=ngc-secret'
```

For source chart deployments:

```bash
helm upgrade aiq deployment-k8s/ -n ns-aiq
```

## Uninstall

```bash
helm uninstall aiq -n ns-aiq

# Optionally remove namespace and secrets
kubectl delete namespace ns-aiq
```

## Troubleshooting

### Pod status

```bash
kubectl get pods -n ns-aiq
kubectl describe pod <pod-name> -n ns-aiq
kubectl get events -n ns-aiq --sort-by='.lastTimestamp'
```

### Logs

```bash
# Backend logs
kubectl logs -n ns-aiq -l component=backend -f

# Frontend logs
kubectl logs -n ns-aiq -l component=frontend -f

# Database init container logs
kubectl logs -n ns-aiq <backend-pod> -c db-init
```

### PVC inspection

```bash
kubectl get pvc -n ns-aiq
kubectl describe pvc aiq-postgres-data -n ns-aiq
```

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImagePullBackOff` | Missing or incorrect image pull secret | Verify `ngc-secret` exists and credentials are valid. Check `kubectl describe pod <pod>`. |
| `CrashLoopBackOff` | Missing credentials or bad config | Check `kubectl logs <pod> -n ns-aiq`. Verify `aiq-credentials` secret has all required keys. |
| Pod stuck in `Pending` | Insufficient cluster resources or PVC not bound | Check `kubectl describe pod <pod>` for scheduling errors. Verify PVC status with `kubectl get pvc -n ns-aiq`. |
| FRAG mode: RAG connection refused | RAG service not reachable | Verify RAG pods are running and service DNS resolves. Test with `kubectl exec` into the backend pod and `curl` the RAG URL. |
| Health check fails | Backend not fully started | Wait for init containers to complete. Check `kubectl logs <pod> -c db-init -n ns-aiq` for database init issues. |
