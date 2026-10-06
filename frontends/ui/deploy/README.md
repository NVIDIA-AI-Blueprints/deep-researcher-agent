# Deep Researcher Agent Blueprint UI - Docker Deployment

This directory contains the Dockerfile to build and deploy the Deep Researcher Agent Blueprint UI as a Docker container.

## Architecture

The UI container acts as a **full proxy** between the browser and backend:

```
Browser  -->  UI Container (HTTP + WebSocket Proxy)  -->  Backend
```

**All traffic flows through the UI container:**
- HTTP API requests -> `/api/*` routes -> Backend
- WebSocket connections -> `/websocket` proxy -> Backend

**Benefits:**
- Backend does not need public exposure
- Single ingress point for security
- Runtime configurable backend URL

## Quick Start

### 1. Build the Image

From the **`frontends/ui/`** directory:

```bash
docker build -f deploy/Dockerfile -t deep-researcher-blueprint-ui:latest .
```

### 2. Run the Container

**Without authentication (development/testing):**

```bash
docker run -p 3000:3000 \
  -e BACKEND_URL=http://backend:8000 \
  -e REQUIRE_AUTH=false \
  deep-researcher-blueprint-ui:latest
```

**With authentication:**

```bash
docker run -p 3000:3000 \
  -e BACKEND_URL=http://backend:8000 \
  -e REQUIRE_AUTH=true \
  -e NEXTAUTH_SECRET=$(openssl rand -base64 32) \
  -e NEXTAUTH_URL=http://localhost:3000 \
  -e OAUTH_CLIENT_ID=your-client-id \
  -e OAUTH_CLIENT_SECRET=your-client-secret \
  -e OAUTH_ISSUER=https://your-oidc-provider.com \
  deep-researcher-blueprint-ui:latest
```

## Environment Variables

All environment variables are **runtime configurable** - no rebuild needed when changed.

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | Backend API URL |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_AUTH` | `false` | Set to `true` to require OAuth login (default user when false) |
| `NEXTAUTH_SECRET` | - | Session encryption secret (required if auth enabled) |
| `NEXTAUTH_URL` | - | Public URL where app is hosted (required if auth enabled) |
| `SECURE_COOKIES` | - | Explicit cookie security override (optional) |

> **Cookie Security:** The `NEXTAUTH_URL` protocol determines cookie security:
> - `http://...` -> non-secure cookies (works over HTTP)
> - `https://...` -> secure cookies only (required for HTTPS)
>
> For reverse proxy setups (TLS terminated at proxy), set `NEXTAUTH_URL` to the external HTTPS URL.

### OAuth Provider (required when `REQUIRE_AUTH=true`)

| Variable | Description |
|----------|-------------|
| `OAUTH_CLIENT_ID` | OAuth client ID from your OIDC provider |
| `OAUTH_CLIENT_SECRET` | OAuth client secret |
| `OAUTH_ISSUER` | OIDC issuer URL (enables auto-discovery) |

## Docker Compose Example

```yaml
services:
  frontend:
    image: deep-researcher-blueprint-ui:latest
    environment:
      - BACKEND_URL=http://backend:8000
      - REQUIRE_AUTH=${REQUIRE_AUTH:-false}
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
      - NEXTAUTH_URL=${NEXTAUTH_URL:-http://localhost:3000}
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
      - OAUTH_ISSUER=${OAUTH_ISSUER}
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

## Networking

### Connecting to Host Services

When running in Docker and connecting to services on the host machine:

- **macOS/Windows:** Use `host.docker.internal`
- **Linux:** Use `--network=host` or configure Docker networking

```bash
docker run -p 3000:3000 \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  -e REQUIRE_AUTH=false \
  deep-researcher-blueprint-ui:latest
```

## Troubleshooting

### Cannot connect to backend

1. Verify `BACKEND_URL` is correct
2. Use `host.docker.internal` for host services (macOS/Windows)
3. Ensure backend is bound to `0.0.0.0`, not `127.0.0.1`
4. Check Docker network configuration

### Authentication not working

1. Verify `REQUIRE_AUTH` is set to `true`
2. Check `NEXTAUTH_SECRET` is set
3. Check `NEXTAUTH_URL` matches your public URL (and protocol)
4. Verify OAuth credentials are correct

### Container won't start

1. Check logs: `docker logs <container-id>`
2. Verify all required environment variables are set
3. Ensure port 3000 is not in use

## Image Details

- **Base:** `nvcr.io/nvidia/base/ubuntu:jammy-20251013`
- **Node.js:** 22.x (via NodeSource)
- **User:** `nextjs` (uid 1001)
- **Port:** 3000
