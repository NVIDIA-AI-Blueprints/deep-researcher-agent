# Deep Researcher Agent Kubernetes Deployment — Source Chart

Deploy Deep Researcher Agent from the cloned repository using the local Helm chart.

> **Looking to install from the NGC Helm repository?** Refer to the [Helm README](../README.md#install-from-ngc-helm-repository) instead.

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, etc.) or local cluster (Kind, Minikube)
- `kubectl` configured with cluster access
- `helm` v3.x installed
- An `deep-researcher-credentials` Secret with database credentials and required API keys.

## Deploy

Create the namespace and required user-supplied credentials first:

The source chart derives every namespaced resource from Helm's
`.Release.Namespace`, supplied with `-n`. This guide uses `ns-deep-researcher`; if you choose a
different namespace, use it consistently for the Helm release, Secrets, `kubectl`
commands, and external identity bindings. Setting `deep-researcher.namespace.create=true` controls
whether the chart renders a Namespace object; it does not override the release namespace.

```bash
kubectl create namespace ns-deep-researcher --dry-run=client -o yaml | kubectl apply -f -

DB_USER_NAME="${DB_USER_NAME:-deep_researcher}"
DB_USER_PASSWORD="${DB_USER_PASSWORD:-deep_researcher_dev}" # pragma: allowlist secret

kubectl create secret generic deep-researcher-credentials -n ns-deep-researcher \
  --from-literal=DB_USER_NAME="$DB_USER_NAME" \
  --from-literal=DB_USER_PASSWORD="$DB_USER_PASSWORD" \
  --from-literal=NVIDIA_API_KEY="$NVIDIA_API_KEY" \
  --from-literal=TAVILY_API_KEY="$TAVILY_API_KEY"
```

Then install the chart:

```bash
helm dependency update deployment-k8s/
helm install deep-researcher deployment-k8s/ -n ns-deep-researcher
```

The default source install creates the PostgreSQL init ConfigMap. The credentials
Secret is user-supplied so API keys are provided explicitly.

If pulling images from NGC, include the image pull secret and repository overrides:

```bash
helm dependency update deployment-k8s/
helm install deep-researcher deployment-k8s/ -n ns-deep-researcher --create-namespace \
  --set 'deep-researcher.apps.backend.imagePullSecrets[0].name=ngc-secret' \
  --set 'deep-researcher.apps.frontend.imagePullSecrets[0].name=ngc-secret' \
  --set deep-researcher.apps.backend.image.repository=nvcr.io/nvidia/blueprint/aiq-agent \
  --set deep-researcher.apps.frontend.image.repository=nvcr.io/nvidia/blueprint/aiq-frontend
```

### Shared Dask scheduler and workers

For multiple backend replicas, use one scheduler service and a dedicated worker
deployment instead of starting a scheduler and worker inside every backend pod.
Create client, scheduler, and worker TLS Secrets before applying the example.
All certificates must be signed by the same CA, and the scheduler certificate
must be valid for the `deep-researcher-dask-scheduler` Service name:

```bash
kubectl create secret generic deep-researcher-dask-client-tls -n ns-deep-researcher \
  --from-file=ca.crt=./tls/ca.crt \
  --from-file=tls.crt=./tls/client.crt \
  --from-file=tls.key=./tls/client.key
kubectl create secret generic deep-researcher-dask-scheduler-tls -n ns-deep-researcher \
  --from-file=ca.crt=./tls/ca.crt \
  --from-file=tls.crt=./tls/scheduler.crt \
  --from-file=tls.key=./tls/scheduler.key
kubectl create secret generic deep-researcher-dask-worker-tls -n ns-deep-researcher \
  --from-file=ca.crt=./tls/ca.crt \
  --from-file=tls.crt=./tls/worker.crt \
  --from-file=tls.key=./tls/worker.key

helm upgrade --install deep-researcher deployment-k8s/ -n ns-deep-researcher \
  -f ../examples/shared-dask-values.yaml
```

The example sets `NAT_DASK_SCHEDULER_ADDRESS=tls://deep-researcher-dask-scheduler:8786`
on the API pods, creates `deep-researcher-dask-scheduler`, and starts four
`deep-researcher-dask-worker` pods. All API replicas can therefore submit work to the same
worker pool. Change `deep-researcher.apps.dask-worker.replicas` and its resource requests to
match the expected research concurrency.

Workers use the backend workflow configuration and application credentials
because they execute research functions. The scheduler mounts only its TLS
identity; `mountSharedSecrets: false` keeps application credentials off that
workload. The worker Service is disabled because only the scheduler needs a
stable network endpoint.

The scheduler accepts executable task payloads. The example therefore requires
mutual TLS and creates an ingress NetworkPolicy that permits scheduler traffic
on port 8786 only from the backend and worker pods. Do not expose the scheduler
or dashboard through an Ingress or public load balancer.

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

### Health Check

```bash
kubectl port-forward -n ns-deep-researcher svc/deep-researcher-backend 8000:8000 &
curl http://localhost:8000/live
curl http://localhost:8000/health
```

The backend liveness probe uses `/live`, which checks only that the API process
responds. The readiness probe uses `/health`, so database or content-encryption
outages remove the pod from service without triggering a restart loop.

## Configuration, FRAG, Secrets, and Access

These topics apply to all deployment methods and are documented in the [Helm README](../README.md):

- [Configuration](../README.md#configuration) — switching workflow configs (LlamaIndex vs FRAG)
- [FRAG Integration](../README.md#frag-integration) — connecting to a RAG Blueprint deployment
- [Secrets](../README.md#secrets) — required API keys and database credentials (`DB_USER_NAME=deep_researcher`, `DB_USER_PASSWORD=deep_researcher_dev` by default)
- [Access the application](../README.md#access-the-application) — port-forwarding to the UI and API

## Upgrade

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

Refer to the [Helm README — Troubleshooting](../README.md#troubleshooting) for common issues (pod status, logs, image pull errors, FRAG connection issues).

---

## Getting Started with Minimal K8s (Kind)

For local development with a [Kind](https://kind.sigs.k8s.io/) cluster, you can build images locally and load them directly into the cluster — no registry push required.

### 1. Build the images

Run the following from the **repository root**:

```bash
# Backend
docker build --platform linux/amd64 \
  -f deploy/Dockerfile \
  -t deep-researcher-agent:dev \
  .

# Frontend
docker build --platform linux/amd64 \
  -f frontends/ui/deploy/Dockerfile \
  -t deep-researcher-frontend:dev \
  frontends/ui
```

This produces two local images:
- `deep-researcher-agent:dev` — backend (Python / FastAPI)
- `deep-researcher-frontend:dev` — frontend (Next.js)

### 2. Load the images into Kind

```bash
kind load docker-image deep-researcher-agent:dev --name <your-cluster-name>
kind load docker-image deep-researcher-frontend:dev --name <your-cluster-name>
```

> Run `kind get clusters` if you are unsure of the cluster name.

### 3. Configure values to use the local images

Edit `deployment-k8s/values.yaml` (or pass `--set` flags at deploy time) so the image references match what you loaded:

```yaml
deep-researcher:
  apps:
    backend:
      image:
        repository: deep-researcher-agent
        tag: dev
        pullPolicy: IfNotPresent
    frontend:
      image:
        repository: deep-researcher-frontend
        tag: dev
        pullPolicy: IfNotPresent
```

Or pass them inline during deployment:

```bash
helm upgrade --install deep-researcher deployment-k8s/ -n ns-deep-researcher --create-namespace \
  --set deep-researcher.apps.backend.image.repository=deep-researcher-agent \
  --set deep-researcher.apps.backend.image.tag=dev \
  --set deep-researcher.apps.backend.image.pullPolicy=IfNotPresent \
  --set deep-researcher.apps.frontend.image.repository=deep-researcher-frontend \
  --set deep-researcher.apps.frontend.image.tag=dev \
  --set deep-researcher.apps.frontend.image.pullPolicy=IfNotPresent
```

### 4. Create credentials and deploy

Create the `deep-researcher-credentials` Secret shown above, then deploy.

After a rebuild, reload the updated image with `kind load docker-image ...` and restart the affected deployment:

```bash
kubectl rollout restart deployment -n ns-deep-researcher deep-researcher-backend   # or deep-researcher-frontend
```
