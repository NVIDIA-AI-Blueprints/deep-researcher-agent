<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Researcher Agent Helm Deployment

This directory contains Helm charts for deploying Deep Researcher Agent to a Kubernetes cluster.

## Directory Structure

```
deploy/helm/
├── README.md                  # This file — NGC chart install & shared configuration
├── deployment-k8s/            # Source chart wrapper (for repo-based deployments)
│   ├── Chart.yaml             #   Depends on helm-charts-k8s/deep-researcher
│   ├── values.yaml            #   Deployment values
│   └── README.md              #   Source chart instructions & Kind local dev guide
└── helm-charts-k8s/
    └── deep-researcher/                   # Base application Helm chart (templates, helpers)
```

## Deployment Methods

| Method | When to use |
|--------|-------------|
| [NGC Helm chart](#install-from-ngc-helm-repository) | Install a pre-built chart from the NGC Helm repository |
| [Source chart](deployment-k8s/README.md) | You cloned the repository and want to build/deploy from source |

All examples use `ns-deep-researcher`. The repository source chart derives every namespaced
resource from Helm's `.Release.Namespace`, supplied with `-n`; use a different namespace
consistently across Helm, Secrets, `kubectl`, and external identity bindings. The
`deep-researcher.namespace.create` value controls whether the source chart renders a Namespace object
and does not override `-n`. The NGC instructions below install published chart 2.2.0. Use
the source chart when you need behavior from the checked-out repository.

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or a local cluster such as Kind or Minikube)
- `kubectl` configured with cluster access
- `helm` v3.x installed
- NGC API key (`NGC_API_KEY` environment variable)
- Required API keys (refer to [Secrets](#secrets) below)

## Install from NGC Helm Repository

This section installs the pre-built chart **deep-researcher-web** version **2.2.0** from the NGC Helm repository (`https://helm.ngc.nvidia.com/nvidia/blueprint/charts/`).

### 1. Create the namespace

```bash
kubectl create namespace ns-deep-researcher --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Create secrets

API credentials for the application:

```bash
kubectl create secret generic deep-researcher-credentials -n ns-deep-researcher \
  --from-literal=NVIDIA_API_KEY="$NGC_API_KEY" \
  --from-literal=TAVILY_API_KEY="$TAVILY_API_KEY" \
  --from-literal=DB_USER_NAME="deep_researcher" \
  --from-literal=DB_USER_PASSWORD="deep_researcher_dev"
```

Image pull secret for the NGC container registry:

```bash
kubectl create secret docker-registry ngc-secret -n ns-deep-researcher \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY
```

### 3. Pull the chart and install

**Pull, verify, then install from local file:**

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
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret'
```

- Replace `<YOUR_NGC_API_KEY>` with your NGC API key (or use `$NGC_API_KEY` if set in your environment).
- To avoid exposing the key in shell history, use a variable: `--password=$NGC_API_KEY`.

**Optional — Install directly from the chart URL** (without pulling first):

```bash
helm upgrade --install deep-researcher https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq2-web-2.2.0.tgz \
  --username='$oauthtoken' \
  --password=$NGC_API_KEY \
  -n ns-deep-researcher --create-namespace \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret'
```

### Override values

Pass additional `--set` flags to customize the deployment:

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.image.tag=<tag>
```

Or supply a custom values file:

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  -f custom-values.yaml
```

### Inspect default values

To see what values the chart supports before installing:

```bash
helm show values deep-researcher-web-2.2.0.tgz
```

### Amazon OpenSearch Serverless

The backend image can be overridden through values without forking the chart:

```yaml
deep-researcher:
  apps:
    backend:
      image:
        repository: <registry>/<deep-researcher-agent-image>
        tag: <tag>
```

For Amazon OpenSearch Serverless, set the backend workflow config to `configs/config_web_opensearch.yml` and configure
SigV4 through environment values:

```yaml
deep-researcher:
  apps:
    backend:
      env:
        CONFIG_FILE: configs/config_web_opensearch.yml
        OPENSEARCH_URL: https://abc123.us-west-2.aoss.amazonaws.com
        OPENSEARCH_AUTH_TYPE: sigv4
        OPENSEARCH_AWS_SERVICE: aoss
        OPENSEARCH_INDEX_PREFIX: deep-researcher
        AWS_REGION: us-west-2
        OPENSEARCH_INGESTION_MODE: auto
        OPENSEARCH_DASK_FILE_TRANSFER: bytes
```

A complete example is available at
[`deploy/helm/examples/aws-opensearch-serverless-values.yaml`](examples/aws-opensearch-serverless-values.yaml).

For EKS Pod Identity, associate the IAM role with the backend service account for this release. These examples install
into `ns-deep-researcher`, and the backend service account is `deep-researcher-backend`. If you change `-n`, create the association in that same
release namespace. EKS Pod Identity associations are created through EKS, not by annotating the service account. The role
also needs OpenSearch Serverless IAM access and a data access policy for the target collection/index pattern.

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

```bash
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-backend 8000:8000 &
curl http://localhost:8000/live
curl http://localhost:8000/health
```

The backend liveness probe uses `/live`, which checks only that the API process
responds. The readiness probe uses `/health`, which checks required dependencies
and can return HTTP 503 without causing Kubernetes to restart the process.

The backend API docs are available at `http://localhost:8000/docs` while the port-forward is active.

### Access the application

```bash
# Frontend UI
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-frontend 3000:3000

# Backend API
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-backend 8000:8000
```

Open [http://localhost:3000](http://localhost:3000) to access the web UI.

### Upgrade

To upgrade an existing release to a newer chart version, pull the new chart archive (same NGC URL pattern with the new version, e.g. `deep-researcher-web-<version>.tgz`) or use the new chart URL for direct install, then run:

```bash
helm upgrade deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret'
```

### Uninstall

```bash
helm uninstall deep-researcher -n ns-deep-researcher

# Optionally remove namespace and secrets
kubectl delete namespace ns-deep-researcher
```

## Configuration

The backend loads a workflow config at startup. Switch configs with `--set`:

| Config file | Description |
|-------------|-------------|
| `configs/config_web_default_llamaindex.yml` | Default — LlamaIndex backend (no external RAG required) |
| `configs/config_web_frag.yml` | Foundational RAG mode (requires a running RAG service) |
| `configs/config_web_frag_mcp_auth.yml` | Foundational RAG with optional per-user MCP authentication |

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml
```

### Per-user MCP authentication with external Redis

The chart does not install Redis. When selecting the per-user authentication config, provide a Redis service that the backend and its workers can both reach. The deployer owns its availability, persistence, networking, and backup.

Add `REDIS_PASSWORD` to the existing `deep-researcher-credentials` Secret when the Redis service requires authentication, then create a values file:

```yaml
# deep-researcher-per-user-auth-values.yaml
deep-researcher:
  apps:
    backend:
      env:
        CONFIG_FILE: configs/config_web_frag_mcp_auth.yml
        MCP_TOKEN_STORE_TYPE: redis
        REDIS_HOST: redis.example.com
        REDIS_PORT: "6379"
      secretEnv:
        REDIS_PASSWORD: REDIS_PASSWORD
```

Apply it to either the downloaded chart or a source-chart installation:

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  -f deep-researcher-per-user-auth-values.yaml
```

Configure `MCP_GDRIVE_URL`, `DEEP_RESEARCHER_PUBLIC_URL`, and any OAuth client credentials required by the protected MCP source through the same `env` and `secretEnv` maps. NAT 1.8's Redis object store in this image supports host, port, database, and optional password; this example does not support TLS, ACL usernames, Sentinel, or Redis Cluster.

## FRAG Integration

To use the Foundational RAG (FRAG) config, you need a running NVIDIA RAG Blueprint deployment. Refer to the [RAG Blueprint Helm deployment guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/docs/deploy-helm.md) for setup instructions.

### Same-cluster RAG connection

If the RAG Blueprint is deployed in the same Kubernetes cluster, use internal service DNS:

```bash
helm upgrade --install deep-researcher deep-researcher-web-2.2.0.tgz -n ns-deep-researcher \
  --wait --timeout 10m \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret' \
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
  --set 'deep-researcher.apps.postgres.imagePullSecrets[0].name=ngc-secret' \
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

## Secrets

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
| `SERPER_API_KEY` | Serper API key for Google search |
| `JINA_API_KEY` | Jina API key |
| `WANDB_API_KEY` | Weights & Biases API key |
| `MODAL_TOKEN_ID` | Modal sandbox token ID |
| `MODAL_TOKEN_SECRET` | Modal sandbox token secret |
| `DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER` | Artifact byte storage provider; unset or `sql` keeps bytes in SQL, `s3` uses S3-compatible storage |
| `DEEP_RESEARCHER_ARTIFACT_S3_BUCKET` | Required when `DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER=s3` |
| `DEEP_RESEARCHER_ARTIFACT_S3_ENDPOINT_URL` | Unset for AWS S3; set for MinIO or compatible storage |
| `DEEP_RESEARCHER_ARTIFACT_S3_REGION` | Optional S3 region |
| `DEEP_RESEARCHER_ARTIFACT_S3_PREFIX` | Optional object-key prefix; defaults to `artifacts/v1` |
| `AWS_ACCESS_KEY_ID` | Development-only S3-compatible access key; use workload identity in production |
| `AWS_SECRET_ACCESS_KEY` | Development-only S3-compatible secret key; use workload identity in production |

Production artifact storage is operator-managed infrastructure. Configure the
service account or workload with a short-lived role, restrict object operations to
the Deep Researcher Agent worker identity and configured bucket prefix, enable Block Public Access
and TLS-only access, and enable storage-layer encryption such as SSE-KMS. Deep Researcher Agent API
ownership checks do not protect direct bucket access, and Deep Researcher Agent does not
application-encrypt artifact blob bytes.

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

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImagePullBackOff` | Missing or incorrect image pull secret | Verify `ngc-secret` exists and credentials are valid. Check `kubectl describe pod <pod>`. |
| `CrashLoopBackOff` | Missing credentials or bad config | Check `kubectl logs <pod> -n ns-deep-researcher`. Verify `deep-researcher-credentials` secret has all required keys. |
| Pod stuck in `Pending` | Insufficient cluster resources or PVC not bound | Check `kubectl describe pod <pod>` for scheduling errors. Verify PVC status with `kubectl get pvc -n ns-deep-researcher`. |
| FRAG mode: RAG connection refused | RAG service not reachable | Verify RAG pods are running and service DNS resolves. Test with `kubectl exec` into the backend pod and `curl` the RAG URL. |
| Health check fails | Backend not fully started | Wait for init containers to complete. Check `kubectl logs <pod> -c db-init -n ns-deep-researcher` for database init issues. |

## Source Chart Deployment

For deploying from the cloned repository (building from source charts, local images, NGC images), refer to the [deployment-k8s README](deployment-k8s/README.md).
