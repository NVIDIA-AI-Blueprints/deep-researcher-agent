<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes (Helm)

Deploy the Deep Researcher Agent blueprint to a Kubernetes cluster using the Helm charts included in the repository.

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
│   ├── Chart.yaml             #   Depends on helm-charts-k8s/deep-researcher
│   ├── values.yaml            #   Deployment values
│   └── README.md              #   Source chart instructions & Kind local dev guide
└── helm-charts-k8s/
    └── deep-researcher/                   # Base application Helm chart (templates, helpers)
```

## Setup

### 1. Create a namespace

```bash
kubectl create namespace ns-deep-researcher --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Create the credentials secret

The deployment reads API keys from a Kubernetes Secret named `deep-researcher-credentials`:

```bash
kubectl create secret generic deep-researcher-credentials -n ns-deep-researcher \
  --from-literal=NVIDIA_API_KEY="$NGC_API_KEY" \
  --from-literal=TAVILY_API_KEY="$TAVILY_API_KEY" \
  --from-literal=DB_USER_NAME="deep_researcher" \
  --from-literal=DB_USER_PASSWORD="deep_researcher_dev"
```

### 3. Create an image pull secret (NGC registry)

If you are pulling pre-built images from the NGC container registry, create a Docker registry secret:

```bash
kubectl create secret docker-registry ngc-secret -n ns-deep-researcher \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY
```

## Deploy

### Using the NGC Helm chart

Install the pre-built chart **deep-researcher-web** version **2.2.0** from the NGC Helm repository (`https://helm.ngc.nvidia.com/nvidia/blueprint/charts/`).

**1. Pull the chart and install from local file:**

```bash
helm pull https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq2-web-2.2.0.tgz \
  --username='$oauthtoken' \
  --password=<YOUR_NGC_API_KEY>

# Verify the chart was pulled correctly
helm show chart deep-researcher-web-2.2.0.tgz

# Install from the local file
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher --create-namespace \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret'
```

Replace `<YOUR_NGC_API_KEY>` with your NGC API key (or use `$NGC_API_KEY`). Prefer `--password=$NGC_API_KEY` to avoid exposing the key in shell history.

**2. Optional — Install directly from the chart URL** (without pulling first):

```bash
helm upgrade --install deep-researcher https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq2-web-2.2.0.tgz \
  --username='$oauthtoken' \
  --password=$NGC_API_KEY \
  -n ns-deep-researcher --create-namespace \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret'
```

For the full NGC chart workflow (value overrides, upgrades, troubleshooting), refer to `deploy/helm/README.md` in the source repository.

### Using the source chart

If you cloned the repository, you can deploy from the local source chart. Refer to the source chart README at `deploy/helm/deployment-k8s/README.md` in the source repository for full details including Kind local development.

The source chart on `develop` derives every namespaced resource from Helm's
`.Release.Namespace`, which is the value passed with `-n`. The examples use `ns-deep-researcher`,
but you can choose another namespace by replacing it consistently in Helm, `kubectl`,
Secret creation, and external bindings such as EKS Pod Identity. The
`deep-researcher.namespace.create` value controls whether the chart renders a Namespace object; it
does not override `-n`. This behavior applies to the repository source chart and does not
change the published NGC 2.2.0 instructions above.

```bash
cd deploy/helm

helm dependency update deployment-k8s/

helm install deep-researcher deployment-k8s/ -n ns-deep-researcher --create-namespace
```

To use NGC container images with the source chart:

```bash
helm install deep-researcher deployment-k8s/ -n ns-deep-researcher --create-namespace \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.image.repository=nvcr.io/nvidia/blueprint/aiq-agent \
  --set deep-researcher.apps.frontend.image.repository=nvcr.io/nvidia/blueprint/aiq-frontend
```

To use locally built images (refer to [Docker Build System](./docker-build.md)):

```bash
helm install deep-researcher deployment-k8s/ -n ns-deep-researcher --create-namespace \
  --set deep-researcher.apps.backend.image.repository=deep-researcher-agent \
  --set deep-researcher.apps.backend.image.tag=dev \
  --set deep-researcher.apps.backend.image.pullPolicy=IfNotPresent \
  --set deep-researcher.apps.frontend.image.repository=deep-researcher-frontend \
  --set deep-researcher.apps.frontend.image.tag=dev \
  --set deep-researcher.apps.frontend.image.pullPolicy=IfNotPresent
```

### Verify

```bash
kubectl get pods -n ns-deep-researcher
```

Expected output:

```
NAME                            READY   STATUS    RESTARTS   AGE
deep-researcher-backend-xxx                 1/1     Running   0          30s
deep-researcher-frontend-xxx                1/1     Running   0          30s
deep-researcher-postgres-xxx                1/1     Running   0          30s
```

### Health check

Once all pods are running, verify the backend is responding:

```bash
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-backend 8000:8000 &
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
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-frontend 3000:3000

# Backend API
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-backend 8000:8000
```

Open [http://localhost:3000](http://localhost:3000) to access the web UI.

## Configuration

The backend loads a workflow config at startup. Switch configs with `--set`:

| Config file | Description |
|-------------|-------------|
| `configs/config_web_default_llamaindex.yml` | Default -- LlamaIndex backend (no external RAG required) |
| `configs/config_web_frag.yml` | Foundational RAG mode (requires a running RAG service) |

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml
```

For source chart deployments, use `deployment-k8s/` instead of the `.tgz` file (refer to [Deploy](#deploy)).

## FRAG Integration

To use the Foundational RAG (FRAG) config, you need a running NVIDIA RAG Blueprint deployment. Refer to the [RAG Blueprint Helm deployment guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/docs/deploy-helm.md) for setup instructions.

### Same-cluster RAG connection

If the RAG Blueprint is deployed in the same Kubernetes cluster, use internal service DNS:

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml \
  --set deep-researcher.apps.backend.env.RAG_SERVER_URL=http://rag-server.<rag-namespace>.svc.cluster.local:8081/v1 \
  --set deep-researcher.apps.backend.env.RAG_INGEST_URL=http://ingestor-server.<rag-namespace>.svc.cluster.local:8082/v1
```

Replace `<rag-namespace>` with the namespace where the RAG Blueprint is deployed.

### External RAG connection

If the RAG service is running outside the cluster:

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml \
  --set deep-researcher.apps.backend.env.RAG_SERVER_URL=http://<rag-host>:8081/v1 \
  --set deep-researcher.apps.backend.env.RAG_INGEST_URL=http://<rag-ingest-host>:8082/v1
```

### Values file approach

For complex overrides, create a values file instead of passing many `--set` flags:

```yaml
# deep-researcher-frag-values.yaml
deep-researcher:
  apps:
    backend:
      env:
        CONFIG_FILE: configs/config_web_frag.yml
        RAG_SERVER_URL: http://rag-server.rag-namespace.svc.cluster.local:8081/v1
        RAG_INGEST_URL: http://ingestor-server.rag-namespace.svc.cluster.local:8082/v1
```

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  -f deep-researcher-frag-values.yaml
```

For complete examples with NGC-specific flags, refer to `deploy/helm/README.md` in the source repository.

## Secrets Reference

### Required

| Key | Description |
|-----|-------------|
| `NVIDIA_API_KEY` | API key for NIM inference models |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `DB_USER_NAME` | PostgreSQL username (default: `deep_researcher`) |
| `DB_USER_PASSWORD` | PostgreSQL password (default: `deep_researcher_dev`) |

### Optional

| Key | Description |
|-----|-------------|
| `EXA_API_KEY` | Exa API key for web search |
| `NIMBLE_API_KEY` | Nimble API key for web search |
| `SERPER_API_KEY` | Serper API key for Google search |
| `JINA_API_KEY` | Jina API key |

### Updating secrets

```bash
kubectl delete secret deep-researcher-credentials -n ns-deep-researcher
kubectl create secret generic deep-researcher-credentials -n ns-deep-researcher \
  --from-literal=NVIDIA_API_KEY="new-key" \  # pragma: allowlist secret
  --from-literal=TAVILY_API_KEY="new-key" \  # pragma: allowlist secret
  --from-literal=DB_USER_NAME="deep_researcher" \
  --from-literal=DB_USER_PASSWORD="deep_researcher_dev"

kubectl rollout restart deployment -n ns-deep-researcher deep-researcher-backend deep-researcher-frontend
```

## Upgrade

For NGC Helm chart releases, pull the new chart archive (same NGC URL pattern with the new version) if needed, then run:

```bash
helm upgrade deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret'
```

For source chart deployments:

```bash
helm upgrade deep-researcher deployment-k8s/ -n ns-deep-researcher
```

## Uninstall

```bash
helm uninstall deep-researcher -n ns-deep-researcher

# Optionally remove namespace and secrets
kubectl delete namespace ns-deep-researcher
```

## Troubleshooting

### Pod status

```bash
kubectl get pods -n ns-deep-researcher
kubectl describe pod <pod-name> -n ns-deep-researcher
kubectl get events -n ns-deep-researcher --sort-by='.lastTimestamp'
```

### Logs

```bash
# Backend logs
kubectl logs -n ns-deep-researcher -l component=backend -f

# Frontend logs
kubectl logs -n ns-deep-researcher -l component=frontend -f

# Database init container logs
kubectl logs -n ns-deep-researcher <backend-pod> -c db-init
```

### PVC inspection

```bash
kubectl get pvc -n ns-deep-researcher
kubectl describe pvc deep-researcher-postgres-data -n ns-deep-researcher
```

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImagePullBackOff` | Missing or incorrect image pull secret | Verify `ngc-secret` exists and credentials are valid. Check `kubectl describe pod <pod>`. |
| `CrashLoopBackOff` | Missing credentials or bad config | Check `kubectl logs <pod> -n ns-deep-researcher`. Verify `deep-researcher-credentials` secret has all required keys. |
| Pod stuck in `Pending` | Insufficient cluster resources or PVC not bound | Check `kubectl describe pod <pod>` for scheduling errors. Verify PVC status with `kubectl get pvc -n ns-deep-researcher`. |
| FRAG mode: RAG connection refused | RAG service not reachable | Verify RAG pods are running and service DNS resolves. Test with `kubectl exec` into the backend pod and `curl` the RAG URL. |
| Health check fails | Backend not fully started | Wait for init containers to complete. Check `kubectl logs <pod> -c db-init -n ns-deep-researcher` for database init issues. |
