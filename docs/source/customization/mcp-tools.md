<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# MCP Tools and Authentication

Model Context Protocol (MCP) is an open protocol that standardizes how applications expose tools and
context to LLM applications. The Deep Researcher Agent Blueprint is built on the NVIDIA NeMo Agent toolkit (NAT), so
Deep Researcher Agent can use MCP servers as data sources through NAT function groups.

This guide targets Deep Researcher Agent deployments pinned to NAT `1.8.0`. Verify your installed version with
`uv pip show nvidia-nat`.

## What this guide covers

**Supported:**

- Connect Deep Researcher Agent to an unauthenticated MCP server.
- Connect Deep Researcher Agent to an MCP server with backend service-account credentials.
- Connect each signed-in user to a protected MCP server through the Deep Researcher Agent UI using MCP OAuth.
- Forward the signed-in Deep Researcher Agent user's identity to a downstream service from a custom Deep Researcher Agent tool.

**Not yet first-party:**

- A first-party DeepResearcher-token pass-through MCP auth provider. Today, if your MCP server trusts the Deep Researcher Agent
  user's bearer token, you must implement and register a custom NAT auth provider in your
  deployment package.

For the full NAT MCP reference:

- [NAT MCP client guide](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html)
- [NAT MCP service-account auth guide](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/auth/mcp-auth/mcp-service-account-auth.html)

This page documents Deep Researcher Agent as an MCP client/data-source consumer. The Deep Researcher Agent reference API does not expose the research
workflow as a public MCP server.

## Choose an Integration Pattern

| Scenario | Pattern | Section |
|---|---|---|
| MCP server has no per-user auth | `mcp_client` function group | [Connect Deep Researcher Agent to an MCP Server](#connect-deep-researcher-agent-to-an-mcp-server) |
| MCP server uses backend / app credentials | `mcp_client` + `mcp_service_account` | [Service-Account MCP Servers](#service-account-mcp-servers) |
| Downstream API trusts the Deep Researcher Agent user's bearer token | Custom Deep Researcher Agent tool using `get_auth_token()` | [Forwarding Deep Researcher Agent User Identity](#forwarding-deep-researcher-agent-user-identity-from-a-tool) |
| MCP server requires per-user OAuth consent | `per_user_mcp_client` + `mcp_oauth2` | [Per-User MCP OAuth](#per-user-mcp-oauth) |

## Prerequisites

Install NAT and the MCP package on the same release line as your Deep Researcher Agent deployment:

```bash
uv pip install "nvidia-nat[mcp]==1.8.0" nvidia-nat-mcp==1.8.0 nvidia-nat-redis==1.8.0
```

Keep `nvidia-nat`, `nvidia-nat-core`, `nvidia-nat-eval`, and `nvidia-nat-mcp` on the same release
line. If you are pinning newer NAT minor releases, bump all four together.

You can inspect the installed MCP component schemas with:

```bash
nat info components -t function_group -q mcp_client
nat info components -t auth_provider -q mcp_service_account
```

## Connect Deep Researcher Agent to an MCP Server

Use `mcp_client` to connect to an MCP server and make its tools available to Deep Researcher Agent agents. The
`mcp_client` function group discovers remote tools and registers them as NAT functions.

```yaml
function_groups:
  mcp_financial_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${MCP_SERVER_URL:-http://localhost:9901/mcp}
```

Supported transports:

- `streamable-http`: recommended for new deployments and required for protected MCP servers.
- `stdio`: useful for local MCP servers started as subprocesses.
- `sse`: backward-compatible. Avoid it for production auth scenarios.

### Register the MCP Group as a Data Source

Add the function group to `data_source_registry`. The registry is Deep Researcher Agent's source of truth for UI
toggles, per-message data source filtering, and default tool inheritance.

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        description: "Search the web for real-time information."
        tools:
          - web_search_tool
          - advanced_web_search_tool
      - id: financial_data
        name: "Financial Data"
        description: "Query financial reports and market data through MCP."
        tools:
          - mcp_financial_tools
```

When an Deep Researcher Agent has no explicit `tools` list, it inherits all tools from `data_source_registry`.
The registry auto-detects function groups and maps every discovered tool back to the source using
NAT function-group prefixes, such as `mcp_financial_tools__get_stock_quote`.

Use `exclude_tools` to specialize individual agents:

```yaml
functions:
  shallow_research_agent:
    _type: shallow_research_agent
    llm: nemotron_lightning_agent_llm
    exclude_tools:
      - mcp_financial_tools__expensive_long_running_tool
```

### Limit and Rename MCP Tools

Use `include`, `exclude`, and `tool_overrides` when the MCP server exposes more tools than Deep Researcher Agent
should use, or when the upstream descriptions are too generic for reliable tool routing.

```yaml
function_groups:
  mcp_financial_tools:
    _type: mcp_client
    include:
      - get_stock_quote
      - get_earnings_report
    server:
      transport: streamable-http
      url: ${MCP_SERVER_URL:-http://localhost:9901/mcp}
    tool_overrides:
      get_stock_quote:
        alias: stock_price
        description: "Returns the current stock price for a ticker symbol."
      get_earnings_report:
        description: "Returns the latest quarterly earnings report for a company."
```

## Service-Account MCP Servers

Use service-account authentication when the MCP server should be accessed with an application or
backend identity, not an individual Deep Researcher Agent user's identity. This is the preferred pattern for CI,
batch jobs, shared enterprise data sources, and container deployments.

```yaml
function_groups:
  mcp_enterprise_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${ENTERPRISE_MCP_URL}
      auth_provider: enterprise_service_account

authentication:
  enterprise_service_account:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes:
      - enterprise.read
```

For MCP servers that require both an OAuth2 service-account token and a service-specific delegation
token, add a `service_token` block:

```yaml
authentication:
  enterprise_dual_auth:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes:
      - enterprise.read
    service_token:
      token: ${ENTERPRISE_SERVICE_TOKEN}
      header: X-Service-Account-Token
```

Register the function group in `data_source_registry` the same way as unauthenticated MCP tools. If
the source does not require the end user to sign in to Deep Researcher Agent, leave `requires_auth` unset or `false`.

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: enterprise_mcp
        name: "Enterprise MCP"
        description: "Search enterprise systems using service-account credentials."
        tools:
          - mcp_enterprise_tools
```

## Forwarding Deep Researcher Agent User Identity from a Tool

When a downstream API or MCP gateway already trusts the Deep Researcher Agent user's bearer token, the supported
Deep Researcher Agent 2.1 pattern is a small custom Deep Researcher Agent tool that reads the request token with
`deep_researcher_agent.auth.get_auth_token()` and forwards it on the outbound call. This works whether the
downstream service is a real MCP server, an HTTP API, or a gateway in front of one.

Deep Researcher Agent exposes:

- `deep_researcher_agent.auth.get_auth_token()` — returns the current request token when available.
- `deep_researcher_agent.auth.get_current_principal()` — returns verified identity metadata from Deep Researcher Agent auth
  middleware (use this for authorization decisions; do not trust unverified JWT payloads).
- Async job token propagation — Deep Researcher Agent captures the request token at submit time and makes it
  available in Dask workers through the same `get_auth_token()` helper. The token is **not**
  refreshed inside the worker, so jobs that outlive the access token's TTL will fail mid-execution
  on auth-required tool calls. There is currently no in-worker refresh guarantee; configure an adequate token TTL or
  reconnect and resubmit the job after expiry.

Example custom tool that forwards the Deep Researcher Agent user's token to an internal search service:

```python
from pydantic import Field

from deep_researcher_agent.auth import get_auth_token
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class InternalSearchConfig(FunctionBaseConfig, name="internal_search"):
    endpoint: str = Field(..., description="Internal search endpoint")


@register_function(config_type=InternalSearchConfig)
async def internal_search(config: InternalSearchConfig, builder):
    async def _search(query: str) -> str:
        token = get_auth_token()
        if not token:
            return "Sign in before using Internal Search."

        # Use an async HTTP client in production code.
        # Forward only to trusted services over HTTPS.
        # `call_internal_search` is a placeholder for your own async HTTP call, e.g.:
        #   async with httpx.AsyncClient() as client:
        #       resp = await client.get(config.endpoint, params={"q": query},
        #                               headers={"Authorization": f"Bearer {token}"})
        #       return resp.text
        return await call_internal_search(
            endpoint=config.endpoint,
            query=query,
            headers={"Authorization": f"Bearer {token}"},
        )

    yield FunctionInfo.from_fn(
        _search,
        description="Search internal systems using the signed-in Deep Researcher Agent user's token.",
    )
```

Register the tool as an auth-required data source so the UI disables it until the user signs in:

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: internal_search
        name: "Internal Search"
        description: "Search internal systems using your Deep Researcher Agent sign-in."
        requires_auth: true
        tools:
          - internal_search

  internal_search:
    _type: internal_search
    endpoint: ${INTERNAL_SEARCH_URL}
```

This is the supported DeepResearcher-user-identity MCP pattern. The two alternatives —
protocol-level pass-through via a custom NAT auth provider, and an auth-forwarding MCP proxy — are
viable in NAT but are not first-class in Deep Researcher Agent; treat them as deployment-side extensions.

For the broader auth context (UI sign-in flow, validator registration, headless API callers), refer to
[Authentication](../deployment/authentication.md).

## Per-User MCP OAuth

Use per-user MCP OAuth when each Deep Researcher Agent user must authorize the upstream MCP server with their own
identity. The reference configuration is
[`configs/config_web_frag_mcp_auth.yml`](../../../configs/config_web_frag_mcp_auth.yml). It combines
an OAuth-protected data source (declared with a `per_user_auth` block), an `mcp_oauth2` authentication
provider, and a shared token object store. The config deliberately does **not** declare a
`per_user_mcp_client` function group: Deep Researcher Agent builds the per-user MCP client in code, per job, from the
`mcp_oauth2` provider's server URL and the signed-in user's stored token. A config-declared
`per_user_mcp_client` is built by NAT's interactive-session builder, which fails for a user with no
token and breaks the interactive WebSocket chat path.

Set these values before starting Deep Researcher Agent:

- `MCP_GDRIVE_URL`: protected streamable-HTTP MCP endpoint. The example calls the source `gdrive`,
  but the mechanism is not Google Drive-specific.
- `DEEP_RESEARCHER_PUBLIC_URL`: externally reachable Deep Researcher Agent origin used to construct the OAuth callback URL.
- `MCP_GDRIVE_CLIENT_ID` and `MCP_GDRIVE_CLIENT_SECRET`: only when the MCP authorization server
  requires a pre-registered OAuth client rather than dynamic client registration.
- `MCP_TOKEN_STORE_TYPE`: `deep_researcher_sqlite` for a single-host example or `redis` for multi-process and
  multi-host deployments.

The UI reads connection status from `/v1/data_sources` and presents Connect or Reconnect for the protected source. Deep Researcher Agent
owns the connect flow and OAuth callback and stores the resulting token under the current Deep Researcher Agent user identity. Disconnect
is not currently exposed by the reference API or UI. Submitting a job with a disconnected protected source
fails with `409 mcp_auth_required`; Deep Researcher Agent does not silently run the job without that source.

Both interactive WebSocket sessions and REST-submitted async jobs resolve the user's MCP tools.
Async workers open their own MCP client for the job and read the same token store as the API, so
the API and workers must share that store:

- `deep_researcher_sqlite` requires the API and worker processes to share the same absolute database path on
  one host.
- `redis` is the supported example for multiple processes, hosts, or Kubernetes pods. Each
  protected source must reference an object store with a distinct bucket or namespace;
  configuration fails closed when two sources reference the same object-store configuration.

This example targets the Deep Researcher Agent web/API deployment, which owns the connect and callback routes and
supplies user identity to jobs. Raw NAT CLI runs do not provide that browser OAuth lifecycle; use
an unauthenticated or service-account MCP configuration for standalone CLI execution.

For a local Redis-backed stack, use the
[per-user-auth Compose override](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/blob/develop/deploy/compose/README.md#per-user-mcp-authentication).
For a released chart, provide an external Redis service as described in the
[Helm deployment guide](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/blob/develop/deploy/helm/README.md#per-user-mcp-authentication-with-external-redis).
The default Compose and Helm deployments remain Redis-free when this example is not selected.

Refer to the
[NAT MCP authentication guide](https://docs.nvidia.com/nemo/agent-toolkit/latest/components/auth/mcp-auth/index.html)
for protocol details.

## Security Guidance

- Prefer `streamable-http` for MCP servers, especially protected servers. Avoid `sse` for
  production authentication scenarios.
- Store secrets in environment variables or a secret manager, not in YAML checked into source
  control.
- Use service-account MCP auth only when shared app-level access is acceptable.
- Use a nested `per_user_auth` block (the `PerUserAuthConfig` object, not a bare boolean) for upstream
  MCP OAuth; `requires_auth: true` only gates a source on Deep Researcher Agent sign-in and does not authorize the
  upstream MCP server:

  ```yaml
  per_user_auth:
    required: true                     # gate job submission until the user connects
    provider: google                   # provider identifier
    mcp_server_id: gdrive              # key used to look up the user's token in NAT token storage
    auth_provider: mcp_oauth2_gdrive   # the `authentication` (mcp_oauth2) provider for this source
  ```

- Keep token forwarding scoped to trusted internal services and HTTPS endpoints.
- Use `requires_auth: true` for sources that depend on Deep Researcher Agent sign-in but do not have a separate
  upstream OAuth connection.

## Troubleshooting

### MCP Tools Do Not Appear in the UI

Confirm the function group is listed in `data_source_registry`. The Deep Researcher Agent UI gets its connection
list from `GET /v1/data_sources`; if a source is missing from the registry, the UI has no toggle
for it.

### The Agent Does Not Use the MCP Tool

Check the remote tool descriptions:

```bash
nat mcp client tool list --url http://localhost:9901/mcp --tool <tool_name> --detail
```

If descriptions are vague, add `tool_overrides` with task-specific descriptions. You may also need
to update prompts so agents know when to prefer the new source.

### Data Source Filtering Does Not Match MCP Tools

Deep Researcher Agent maps function groups by prefix. A group named `mcp_financial_tools` maps tools such as
`mcp_financial_tools__stock_price` back to the source containing `mcp_financial_tools`. If you
list individual tool references instead of the group name, list the exact exposed tool names.

### Authenticated Source Is Disabled in the UI

If a source has `requires_auth: true`, the UI disables it until the user has an auth token. Verify
the frontend auth provider is configured and that requests include an `idToken` cookie or
`Authorization: Bearer <token>` header.

### Async Deep Research Loses User Auth Mid-Job

Use `get_auth_token()` inside custom Deep Researcher Agent tools rather than reading request headers directly — Deep Researcher Agent
captures the request token at job submit and restores it in the async worker context (refer to
[Authentication → Use the current user token in tools](../deployment/authentication.md#step-5-use-the-current-user-token-in-tools)).
Note that the access token is **not** refreshed inside the worker, so jobs that outlive the
token's TTL will fail mid-execution. There is currently no in-worker refresh guarantee; reconnect and resubmit after
expiry, or configure token lifetimes appropriate for the longest expected job.

## Related Documentation

- [Authentication](../deployment/authentication.md)
- [Tools and Sources](./tools-and-sources.md)
- [Adding a Tool](../extending/adding-a-tool.md)
- [Adding a Data Source](../extending/adding-a-data-source.md)
- [Prompts](./prompts.md)
