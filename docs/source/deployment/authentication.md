<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Authentication

Deep Researcher Agent authentication is disabled by default for local development. That mode represents
one trusted user or trust domain; it is not a multi-user authorization boundary and
must not be exposed directly to a shared or untrusted network. Customers operating a
multi-user or externally reachable deployment must enable Deep Researcher Agent authentication or place
Deep Researcher Agent behind a customer-managed authenticated gateway, and must own the corresponding
authorization policy, network isolation, and edge rate limits.

When Deep Researcher Agent authentication is configured, the web UI signs users in with an OAuth/OIDC
provider, and the backend validates any presented bearer token or `idToken` cookie.
External requests require a valid token when `REQUIRE_AUTH=true` and the request
hostname is classified through `DEEP_RESEARCHER_EXTERNAL_HOSTNAMES`; requests outside that
boundary remain on the trusted/internal path.

Use this guide when you need to:

- Require users to sign in before using DeepResearcher.
- Add an OAuth/OIDC provider to the Deep Researcher Agent UI.
- Configure backend JWT validation.
- Gate data sources that need an authenticated Deep Researcher Agent user.
- Forward the Deep Researcher Agent user token to custom tools or MCP pass-through integrations.

## How Deep Researcher Agent Auth Works

Deep Researcher Agent has two auth layers that must be configured together:

| Layer | Responsibility |
|---|---|
| Frontend UI | Runs the OAuth/OIDC sign-in flow with NextAuth, stores the session, and sets an `idToken` cookie after login. |
| Backend API | Runs `AuthMiddleware`, validates bearer tokens or the `idToken` cookie with registered validators, and exposes the verified principal to tools and jobs. |

The same `REQUIRE_AUTH=true` setting is used by both services, but each service also needs its own
configuration. The frontend needs an OAuth provider. The backend needs at least one token validator.

## Step 1: Add a UI OAuth Provider

Create a provider file in:

```text
frontends/ui/src/adapters/auth/providers/
```

For example:

```typescript
// frontends/ui/src/adapters/auth/providers/my-sso.ts
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
  // MY_SSO_CLIENT_SECRET must be set; an empty string causes silent OAuth failures.
  clientSecret: process.env.MY_SSO_CLIENT_SECRET || '',
  checks: ['pkce', 'state'] as ('pkce' | 'state' | 'nonce')[],
  idToken: true,
  profile(profile: { sub: string; email: string; name: string; picture?: string }) {
    return {
      id: profile.sub,
      email: profile.email,
      name: profile.name,
      image: profile.picture,
    }
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
      client_secret: process.env.MY_SSO_CLIENT_SECRET || '',
    }),
  })

  const tokens = await response.json()
  if (!response.ok) {
    throw tokens
  }
  return tokens
}
```

Then update the provider registry:

```typescript
// frontends/ui/src/adapters/auth/providers/index.ts
import type { AuthProviderConfig } from './types'
import { MySSOProvider, refreshMySSOToken } from './my-sso'

export type { AuthProviderConfig, TokenRefreshResult } from './types'

export const getAuthProviderConfig = (): AuthProviderConfig => ({
  provider: MySSOProvider,
  providerId: 'my-sso',
  refreshToken: refreshMySSOToken,
})
```

Use `frontends/ui/src/adapters/auth/providers/auth-example.ts` as the implementation checklist for a
new provider.

## Step 2: Configure UI Auth Environment

Set these environment variables for the frontend:

```bash
REQUIRE_AUTH=true
NEXTAUTH_SECRET=<generate-with-openssl-rand-base64-32>
NEXTAUTH_URL=https://deep-researcher.example.com
SESSION_MAX_AGE_HOURS=24

MY_SSO_ISSUER=https://sso.example.com
MY_SSO_CLIENT_ID=<client-id>
MY_SSO_CLIENT_SECRET=<client-secret>
MY_SSO_TOKEN_URL=https://sso.example.com/token
```

Set `NEXTAUTH_URL` to the public URL users open in their browser. If the public URL is HTTPS, cookies
are secure by default. For reverse proxies that terminate TLS, still use the external HTTPS URL.

## Step 3: Add a Backend Token Validator

The backend only enforces auth when `REQUIRE_AUTH=true` and a request is classified as external. Set
`DEEP_RESEARCHER_EXTERNAL_HOSTNAMES` to the hostnames that should be treated as externally reachable:

```bash
REQUIRE_AUTH=true
DEEP_RESEARCHER_EXTERNAL_HOSTNAMES=deep-researcher-api.example.com
```

Register a validator before the backend starts. The recommended approach for deployment packages is
an entry point:

```toml
# pyproject.toml of your deployment package
[project.entry-points."deep_researcher_api.validators"]
my_sso = "my_deep_researcher_auth.validators:get_validators"
```

```python
# my_deep_researcher_auth/validators.py
import os

from deep_researcher_api.auth.jwt_validator import JWTValidator


def get_validators() -> list:
    issuer = os.environ["DEEP_RESEARCHER_JWT_ISSUER"]
    audience = os.environ.get("DEEP_RESEARCHER_JWT_AUDIENCE")
    return [JWTValidator(issuer_url=issuer, audience=audience)]
```

Then set the validator environment:

```bash
DEEP_RESEARCHER_JWT_ISSUER=https://sso.example.com
DEEP_RESEARCHER_JWT_AUDIENCE=<optional-api-audience>
```

Install the deployment package into the backend environment before starting DeepResearcher. At startup, Deep Researcher Agent
loads validators from the `deep_researcher_api.validators` entry point group. If `REQUIRE_AUTH=true` and no
validators are registered, the backend fails fast.

For simple embedded deployments, you can also register programmatically before `nat serve` starts:

```python
from deep_researcher_api.auth.jwt_validator import JWTValidator
from deep_researcher_api.plugin import register_validator

register_validator(JWTValidator(issuer_url="https://sso.example.com", audience="api://deep-researcher"))
```

## Step 4: Mark Authenticated Data Sources

Use `requires_auth: true` for sources that require a signed-in Deep Researcher Agent user:

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: internal_mcp
        name: "Internal MCP"
        description: "Call internal MCP tools using your Deep Researcher Agent sign-in."
        requires_auth: true
        tools:
          - internal_mcp
```

The UI disables these sources until the user signs in to DeepResearcher. This flag does not automatically
authenticate to an upstream MCP server or API. Tools still need service-account credentials, native
MCP auth, or Deep Researcher Agent token pass-through.

## Step 5: Use the Current User Token in Tools

Custom tools can read the current Deep Researcher Agent request token with `get_auth_token()`:

```python
from deep_researcher_agent.auth import get_auth_token
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class InternalLookupConfig(FunctionBaseConfig, name="internal_lookup"):
    endpoint: str


@register_function(config_type=InternalLookupConfig)
async def internal_lookup(config: InternalLookupConfig, builder):
    async def _lookup(query: str) -> str:
        token = get_auth_token()
        if not token:
            return "Sign in before using Internal Lookup."

        # `call_internal_service` is a placeholder for your own async HTTP call
        # (e.g. via httpx.AsyncClient) — replace with the real client invocation.
        return await call_internal_service(
            endpoint=config.endpoint,
            query=query,
            headers={"Authorization": f"Bearer {token}"},
        )

    yield FunctionInfo.from_fn(
        _lookup,
        description="Look up internal information using the signed-in Deep Researcher Agent user's token.",
    )
```

Deep Researcher Agent also propagates the request token into async Dask jobs, so custom tools should use
`get_auth_token()` instead of reading HTTP headers directly.

Use `get_current_principal()` when you need trusted identity metadata:

```python
from deep_researcher_agent.auth import get_current_principal

principal = get_current_principal()
if principal:
    user_id = principal.sub
```

Do not use unverified JWT payloads for authorization decisions.

## Headless API Callers

API clients can send a bearer token directly:

```bash
curl -H "Authorization: Bearer ${DEEP_RESEARCHER_ID_TOKEN}" \
     -H "X-DeepResearcher-Mode: headless" \
     https://deep-researcher-api.example.com/v1/data_sources
```

`X-DeepResearcher-Mode: headless` tells Deep Researcher Agent the caller cannot participate in interactive clarifier back-and-forth.

## Troubleshooting

### Backend Fails at Startup

If `REQUIRE_AUTH=true`, make sure at least one validator is registered. The backend will fail with a
clear error if no validators are available.

### Requests Are Not Rejected

Set `DEEP_RESEARCHER_EXTERNAL_HOSTNAMES` to the external backend hostname. Deep Researcher Agent treats requests to other hostnames
as internal traffic for cluster-to-cluster communication.

### UI Shows Default User

Confirm `frontends/ui/src/adapters/auth/providers/index.ts` returns your provider, not the default
`provider: null` configuration.

### Authenticated Data Source Is Disabled

The source has `requires_auth: true`, but the UI does not have an `idToken`. Confirm the user is
signed in and that the NextAuth callback sets the `idToken` cookie.

### Tool Cannot See the Token

Use `get_auth_token()` inside the tool call body. Do not capture the token at startup, because startup
does not run in a user request context.

## Related Documentation

- [MCP Tools and Authentication](../customization/mcp-tools.md)
- [Tools and Sources](../customization/tools-and-sources.md)
- [Production Considerations](./production.md)
- [Observability](./observability.md)
