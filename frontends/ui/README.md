# NVIDIA Deep Researcher Agent Blueprint UI

A modern research assistant interface built with Next.js, React, TypeScript, TailwindCSS, and NVIDIA KUI Foundations.

## Overview

The Deep Researcher Agent Blueprint UI provides an accessible, feature-rich frontend for the Deep Researcher Agent backend. It features:

- **Next.js** with App Router and Turbopack
- **React** with TypeScript (strict mode)
- **[KUI Foundations](https://www.npmjs.com/package/@nvidia/foundations-react-core)** NVIDIA design components
- **TailwindCSS** for layout utilities
- **Adapter-based architecture** for clean separation of concerns
- **Optional OAuth authentication** (disabled by default)

## Prerequisites

- Node.js
- npm
- Deep Researcher Agent Blueprint running (default: `http://localhost:8000`)

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Review the `.env` config in the project root to ensure values are set correctly for local development.

Key variables for local development:

```bash
# Backend URL (must match where your backend is running)
BACKEND_URL=http://localhost:8000

# Skip authentication for local development (uses Default User)
REQUIRE_AUTH=false

# File upload settings (should match backend limits)
FILE_UPLOAD_ACCEPTED_TYPES=.pdf,.txt,.md,.docx,.pptx
FILE_UPLOAD_MAX_SIZE_MB=100
FILE_UPLOAD_MAX_FILE_COUNT=10
FILE_EXPIRATION_CHECK_INTERVAL_HOURS=24
```

See `.env.example` for the full list of available frontend variables including authentication and file upload configuration.


### 3. Start Servers

#### Running the Services

**Start e2e** (from monorepo root)
```bash
cd ../../
./scripts/start_e2e.sh
```
>**NOTE:** For UI development it may be more useful to use `./scripts/start_server_in_debug_mode.sh` with `npm run dev` in separate terminals.

#### Separate terminal env setup

When running the backend with `./scripts/start_server_in_debug_mode.sh` and the UI with `npm run dev` in a separate terminal, load the root env file in the UI terminal first:

```bash
set -a; source ../../deploy/.env; set +a
npm run dev
```

**URLs:**

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

## NPM Scripts

| Script               | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `npm run dev`        | Start gateway + Next.js dev server (with HMR)          |
| `npm run build`      | Build for production                                   |
| `npm run start`      | Start production server (gateway with WebSocket proxy) |
| `npm run lint`       | Run ESLint                                             |
| `npm run lint:fix`   | Run ESLint with auto-fix                               |
| `npm run format`     | Format code with Prettier                              |
| `npm run type-check` | Run TypeScript type checking                           |
| `npm run test`       | Run tests once (Vitest)                                |
| `npm run test:watch` | Run tests in watch mode                                |
| `npm run test:ci`    | Run tests with coverage                                |

## Project Structure

```
src/
├── adapters/           # External interface boundaries
│   ├── api/            # Backend API clients (chat, documents, websocket, deep-research)
│   ├── auth/           # NextAuth configuration, session, and types
│   ├── datadog/        # Real User Monitoring integration (optional)
│   └── ui/             # KUI component re-exports, icons, logo
├── app/                # Next.js App Router pages and API routes
│   ├── api/            # Route handlers (auth, chat, health, proxy, jobs)
│   └── auth/           # Sign-in and error pages
├── features/           # Business logic modules
│   ├── chat/           # Chat functionality (components, hooks, store, types)
│   ├── documents/      # File upload, validation, and persistence
│   └── layout/         # App layout components (panels, tabs, navigation)
├── hooks/              # Shared React hooks (PDF download, session URL)
├── lib/                # Utilities (PDF generation)
├── mocks/              # MSW mock handlers and database for testing
├── pages/              # API routes (PDF generation)
├── shared/             # Shared components, config, context, hooks, and utilities
│   ├── components/     # MarkdownRenderer, StarfieldAnimation
│   ├── config/         # File upload configuration
│   ├── context/        # AppConfigContext
│   ├── hooks/          # Backend health check hook
│   └── utils/          # Shared utilities (time formatting)
├── styles/             # KUI-generated CSS and safelist
├── test-utils/         # Test helper utilities
└── utils/              # General utilities (markdown download)
```

## Architecture

The UI acts as a **gateway/proxy** between the browser and backend:

- All HTTP API requests go through Next.js API routes (`/api/*`)
- WebSocket connections are proxied through the custom server (`/websocket`)
- Backend URL is runtime configurable via `BACKEND_URL` environment variable

This architecture ensures the backend doesn't need public exposure - only the UI container needs ingress. See the [Docker Deployment](#docker-deployment) section for details.

## Session Storage Management

The Deep Researcher Agent UI uses localStorage to persist chat sessions across page refreshes. To prevent quota exceeded errors and ensure optimal performance, the app implements automatic storage management.

### Storage Limits

- **localStorage Quota**: ~5MB (browser-dependent)
- **Warning Threshold**: 4MB (80% of quota)
- **Target After Cleanup**: <3MB (60% of quota)

### What Gets Stored

Sessions are stored with optimized data to minimize storage usage:

**Stored (Essential for UI):**
- Session metadata (id, title, timestamps)
- Message content and timestamps
- Thinking steps (for ChatThinking display)
- Plan messages (cannot be refetched from backend)
- Job IDs for deep research restoration

**Not Stored (Fetched from backend on demand):**
- Report content (loaded via API)
- Citations, tasks, tool calls (replayed from SSE stream)
- Agent traces and file artifacts

### Automatic Cleanup

When creating a new session, if storage exceeds 4MB:

1. **Auto-cleanup triggers** - Deletes oldest sessions (by `updatedAt` timestamp)
2. **Current session protected** - Never deletes the active session
3. **Stops at 3MB** - Cleanup continues until storage is healthy
4. **Console warnings** - Logs deleted sessions for debugging

### Manual Cleanup

To manually clear sessions:
1. Open SessionsPanel (left sidebar)
2. Click "Delete All Sessions" button
3. Or delete individual sessions one at a time


### How Research Data Loading Works

When you reopen a session after a page refresh:

1. **ChatArea** - Displays immediately (messages, thinking steps loaded from localStorage)
2. **PlanTab** - Displays immediately (plan messages loaded from localStorage)
3. **Tasks/Thinking/Report tabs** - Shows loading state, then fetches data from the backend

The lazy loading is automatic and seamless - you don't need to do anything special.

### Deep Research Progress

The Tasks tab presents an append-only timeline of workflow phases that actually started.
Repeated phase attempts share one row, concurrent researchers are aggregated with a
completed/observed count, and missing phases are never inferred from later activity. The
Thinking tab retains the detailed execution traces. For inactive legacy jobs without a
recognized workflow trace, Tasks falls back to the root todo artifact. This projection is
frontend-only and does not add model calls or change the backend event contract.

## Docker Deployment

### Build

From the **UI directory** (`frontends/ui/`):

```bash
docker build -f deploy/Dockerfile -t deep-researcher-blueprint-ui:latest .
```

### Run

**Without authentication (default):**

```bash
docker run -p 3000:3000 \
  -e BACKEND_URL=http://localhost:8000 \
  -e REQUIRE_AUTH=false \
  deep-researcher-blueprint-ui:latest
```

**With OAuth authentication:**

```bash
docker run -p 3000:3000 \
  -e BACKEND_URL=http://localhost:8000 \
  -e REQUIRE_AUTH=true \
  -e NEXTAUTH_SECRET=$(openssl rand -base64 32) \
  -e NEXTAUTH_URL=https://your-domain.com \
  -e OAUTH_CLIENT_ID=your-client-id \
  -e OAUTH_CLIENT_SECRET=your-client-secret \
  -e OAUTH_ISSUER=https://your-oidc-provider.com \
  deep-researcher-blueprint-ui:latest
```

### Docker Compose Example

```yaml
services:
  frontend:
    image: deep-researcher-blueprint-ui:latest
    environment:
      # Backend
      - BACKEND_URL=http://backend:8000

      # Authentication (auth is disabled by default)
      - REQUIRE_AUTH=${REQUIRE_AUTH:-false}
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
      - NEXTAUTH_URL=${NEXTAUTH_URL:-http://localhost:3000}

      # OAuth (required when REQUIRE_AUTH=true)
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
      - OAUTH_ISSUER=${OAUTH_ISSUER}
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

### Networking

#### Connecting to Host Services

When running in Docker and connecting to services on the host machine:

- **macOS/Windows:** Use `host.docker.internal`
- **Linux:** Use `--network=host` or configure Docker networking

```bash
# Connect to backend running on host
docker run -p 3000:3000 \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  -e REQUIRE_AUTH=false \
  deep-researcher-blueprint-ui:latest
```

#### Docker Network

When using docker-compose or custom networks, use service names:

```bash
-e BACKEND_URL=http://backend:8000
```

### Health Check

The container includes a health check that polls the root endpoint:

```
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3
    CMD curl -f http://localhost:3000/ || exit 1
```

## Environment Variables

All environment variables are **runtime configurable** - no container rebuild needed.

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | Backend API URL |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_AUTH` | `false` | Set to `true` to require OAuth login |
| `NEXTAUTH_SECRET` | - | Session encryption secret (required if auth enabled) |
| `NEXTAUTH_URL` | - | Public URL where app is hosted (required if auth enabled) |
| `SESSION_MAX_AGE_HOURS` | `24` | Session and idToken cookie lifetime in hours |
| `TOKEN_REFRESH_BUFFER_MINUTES` | `15` | Minutes before token expiry to trigger refresh (set to 30 for long-running jobs) |

> **Cookie Security:** `NEXTAUTH_URL` determines cookie security:
> - `http://...` -> non-secure cookies (local dev over HTTP)
> - `https://...` -> secure cookies (production over HTTPS)

### OAuth (required when `REQUIRE_AUTH=true`)

Provider-specific env vars depend on your provider implementation. See `src/adapters/auth/providers/auth-example.ts` for a template/checklist and the [Authentication](#authentication) section for setup steps.


## API Communication

The UI supports two communication patterns with the backend:

### HTTP Streaming (SSE)

OpenAI-compatible chat completions via `/chat/stream`:

```typescript
import { streamChat } from '@/adapters/api'

await streamChat(
  { messages, sessionId },
  {
    onChunk: (content) => console.log(content),
    onComplete: () => console.log('Done'),
    onError: (error) => console.error(error),
  }
)
```

### WebSocket

Custom protocol for real-time agent communication:

```typescript
import { createWebSocketClient } from '@/adapters/api'

const ws = createWebSocketClient({
  sessionId: 'abc123',
  callbacks: {
    onAgentText: (content, isFinal) => {},
    onStatus: (status, message) => {},
    onToolCall: (name, input, output) => {},
    onError: (code, message) => {},
  },
})

ws.connect()
ws.sendMessage('Hello!')
```

## Authentication

Authentication is **disabled by default**. All users are assigned a "Default User" identity with no login required. The auth system uses a **plugin architecture** where `src/adapters/auth/providers/index.ts` is the sole file that controls whether auth is enabled and which provider is active.

### Architecture

```
src/adapters/auth/
├── providers/
│   ├── types.ts           # AuthProviderConfig interface (contract)
│   ├── index.ts           # SWAP-POINT: returns null (disabled) or a real provider
│   └── auth-example.ts    # Provider template/checklist (not imported by default)
├── config.ts              # NextAuth config (provider-agnostic, never needs editing)
├── session.ts             # useAuth() hook (provider-agnostic)
├── types.ts               # NextAuth type extensions
└── index.ts               # Re-exports
```

- `providers/index.ts` exports `getAuthProviderConfig()` which returns the active provider configuration. By default it returns `{ provider: null }` (auth disabled).
- `config.ts` imports from `providers/index.ts` and wires the provider into NextAuth. It never needs to be edited when adding a new provider.
- `session.ts` provides the `useAuth()` hook that components use. It reads `authProviderId` from `AppConfig` and adapts dynamically.

### Provider Contract

Every auth provider must conform to the `AuthProviderConfig` interface defined in `providers/types.ts`:

```typescript
interface AuthProviderConfig {
  provider: Record<string, unknown> | null  // NextAuth-compatible provider object, or null
  providerId: string                        // ID used in signIn(providerId) -- must match provider.id
  refreshToken: (refreshToken: string) => Promise<TokenRefreshResult>
}

interface TokenRefreshResult {
  access_token: string
  id_token?: string
  expires_in: number
  refresh_token?: string
}
```

### Enabling Authentication (step-by-step)

To enable OAuth/OIDC authentication, follow these steps:

#### Step 1: Create a provider file

Create a new file in `src/adapters/auth/providers/` (e.g. `my-sso.ts`). See `auth-example.ts` in the same directory for a template/checklist. Your file should export:

1. A NextAuth-compatible provider object (OAuth/OIDC config)
2. A token refresh function matching the `TokenRefreshResult` return type

Example minimal provider:

```typescript
// src/adapters/auth/providers/my-sso.ts
import type { TokenRefreshResult } from './types'

export const MySSOProvider = {
  id: 'my-sso',
  name: 'My SSO',
  type: 'oauth' as const,
  wellKnown: `${process.env.MY_SSO_ISSUER}/.well-known/openid-configuration`,
  authorization: {
    params: { scope: 'openid profile email', response_type: 'code' },
  },
  clientId: process.env.MY_SSO_CLIENT_ID,
  clientSecret: process.env.MY_SSO_CLIENT_SECRET || '',
  checks: ['pkce', 'state'] as ('pkce' | 'state' | 'nonce')[],
  idToken: true,
  profile(profile: { sub: string; email: string; name: string; picture?: string }) {
    return { id: profile.sub, email: profile.email, name: profile.name, image: profile.picture }
  },
}

export const refreshMySSOToken = async (refreshToken: string): Promise<TokenRefreshResult> => {
  const response = await fetch(process.env.MY_SSO_TOKEN_URL!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
      client_id: process.env.MY_SSO_CLIENT_ID || '',
    }),
  })
  const tokens = await response.json()
  if (!response.ok) throw tokens
  return tokens
}
```

#### Step 2: Wire it into providers/index.ts

Replace the default `getAuthProviderConfig()` to return your provider:

```typescript
// src/adapters/auth/providers/index.ts
import type { AuthProviderConfig } from './types'
import { MySSOProvider, refreshMySSOToken } from './my-sso'

export type { AuthProviderConfig, TokenRefreshResult } from './types'

export const getAuthProviderConfig = (): AuthProviderConfig => ({
  provider: MySSOProvider,
  providerId: 'my-sso',
  refreshToken: refreshMySSOToken,
})
```

#### Step 3: Set environment variables

```bash
REQUIRE_AUTH=true
NEXTAUTH_SECRET=<generate-with-openssl-rand-base64-32>
NEXTAUTH_URL=http://localhost:3000

# Provider-specific (names depend on your provider file)
MY_SSO_ISSUER=https://sso.example.com
MY_SSO_CLIENT_ID=<your-client-id>
MY_SSO_CLIENT_SECRET=<your-client-secret>
MY_SSO_TOKEN_URL=https://sso.example.com/token
```

That's it. No other files need to change -- `config.ts`, `session.ts`, `proxy.ts`, and all components automatically adapt to the new provider via `getAuthProviderConfig()`.

### Disabling Authentication

To disable auth (the default), ensure `providers/index.ts` returns `{ provider: null }` and either unset `REQUIRE_AUTH` or set `REQUIRE_AUTH=false`. The app will use a "Default User" identity with no login required.

### Using the Auth Hook

```typescript
import { useAuth } from '@/adapters/auth'

const MyComponent = () => {
  const { user, isAuthenticated, isLoading, idToken, signIn, signOut } = useAuth()

  if (isLoading) return <Spinner />
  if (!isAuthenticated) return <Button onClick={signIn}>Sign In</Button>

  return <Text>Welcome, {user?.name}</Text>
}
```

When auth is disabled, `useAuth()` returns `isAuthenticated: true` with a default user -- no sign-in flow is triggered.


## Development

### Adding a New Feature

1. Create a directory under `src/features/[feature-name]/`
2. Add subdirectories: `components/`, `hooks/`
3. Create `types.ts` for feature-specific types
4. Create `store.ts` for Zustand state (if needed)

### Adding a New API Endpoint

1. Add Zod schema in `src/adapters/api/schemas.ts`
2. Create client function in appropriate adapter file
3. Export from `src/adapters/api/index.ts`

## Import Rules

Features should **never** import external packages directly. All external calls go through adapters:

```typescript
// Correct
import { Button, Flex, Text } from '@/adapters/ui'
import { streamChat } from '@/adapters/api'
import { useSession } from '@/adapters/auth'

// Wrong
import { Button } from '@nvidia/foundations-react-core'
import { signIn } from 'next-auth/react'
```


## Styling

This project uses KUI Foundations for styling:

- Use KUI component props for visual styling (`kind`, `size`, etc.)
- Use Tailwind only for layout (`flex`, `grid`, `mt-4`, `px-6`)
- Never override KUI colors with Tailwind
- Dark mode is handled automatically by ThemeProvider

```tsx
// Correct
<Flex className="mt-4 px-6">
  <Button kind="primary" size="medium">Submit</Button>
</Flex>

// Wrong
<Button className="bg-blue-500 text-white">Submit</Button>
```

## Testing

The project uses **Vitest** with **Testing Library** and **MSW** (Mock Service Worker):

```bash
# Run tests once
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:ci
```

- **Vitest** -- Test runner with coverage via `@vitest/coverage-v8`
- **Testing Library** -- `@testing-library/react` and `@testing-library/user-event` for component testing
- **MSW** -- Mock Service Worker for API mocking in tests (handlers in `src/mocks/`)
- **happy-dom** -- DOM environment for tests

Test utilities are in `src/test-utils/` and MSW mock handlers/database are in `src/mocks/`.

## Troubleshooting

### Backend connection fails

1. Verify backend is running: `curl http://localhost:8000/docs`
2. Check `BACKEND_URL` in `.env.local`
3. Check browser console for CORS errors


### Port already in use

Kill existing processes:

```bash
lsof -ti :8000 | xargs kill -9  # Backend
lsof -ti :3000 | xargs kill -9  # Frontend
```

### Docker: Cannot connect to backend

- Use `host.docker.internal` instead of `localhost` to reach host machine services
- Ensure backend is bound to `0.0.0.0`, not just `127.0.0.1`
- Check Docker network configuration if using docker-compose
