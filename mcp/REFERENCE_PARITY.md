<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MCP compatibility contract

This document freezes the public behavior of the Deep Researcher Agent MCP component and records
its intentional design decisions. It is a self-contained behavioral
specification for compatibility reviews and future changes.

## Validated baseline

The public implementation validates MCP 1.28.1, NAT Core 1.8.0, and Deep Researcher Agent 2.0.
The ordered contract over tool name, input schema, output schema, annotations,
title, metadata, icons, and execution fields has the frozen SHA-256
`81eba67fadd56e64b58a84b700b202841f8636c93c6cbf63752507c8bf5ca96a`
and is enforced by executable tests. Descriptions are intentionally excluded
from that hash and instead have semantic golden assertions covering polling,
state, todo, report, and anonymous-capability instructions.

## Preserved behavior

| Area | Public parity contract |
|---|---|
| Tool surface | Exactly `submit_query(query)`, `poll_query(job_id)`, and `get_final_report(job_id)`, in that order. Each argument is one required unconstrained string and each output is a generic object. |
| Meta submit | Synchronous classification, persisted UUIDv4 job, and exact `{job_id, depth: "meta", state: "complete", result}`; missing meta text falls back to `I'm here to help.` |
| Shallow submit | Initial estimate/first-poll values are 10/5 seconds. The server waits inline; terminal work returns complete/result or failed/error. A timeout leaves the task alive, returns first-poll 0, and returns `max(0, int(10 - inline_wait_seconds))` as the estimate (the default 30-second wait therefore returns zero). An inline-wait infrastructure failure returns the original queued capability and cadence unchanged. |
| Deep submit | Always queued and never inline-waited, with estimate/first-poll values 180/180 seconds. |
| Poll | Status only. Found rows always include `todos`. Shallow queued/running cadence is 3 seconds; deep cadence is fixed at 180 seconds. Complete never includes `result`; failed includes `error`; terminal responses omit cadence. |
| Final report | Queued/running becomes `not_ready/job_not_ready` without todos or cadence. Complete includes result; failed includes error; missing/hidden jobs return the exact not-found shape. |
| Background work | The job UUID is also the NAT conversation/checkpoint thread ID. Submit creates a process-owned task; request completion and inline timeout do not cancel it. |
| Persistence | The Postgres schema, migrations, UUIDv4 IDs, 86,400-second job TTL, 30-second heartbeats, 300-second reconciliation interval, 600-second stale threshold, poll count, and cross-manager visibility are retained. |
| Todos | Only top-level deep-research checkpoints are read. Queued, shallow, and unavailable progress returns `[]`; running/complete/failed deep jobs may return normalized `{content,status,id?}` items. Reads fail soft. |
| Errors | Generic workflow exceptions, cancellation, swallowed no-source failures, stale jobs, not-ready, and not-found responses retain stable client-safe messages. Submission failures before a capability is returned, plus poll and report infrastructure exceptions, become sanitized MCP tool errors; internal exception text never crosses the transport boundary. |

The persisted state graph is:

```text
meta classification ───────────────────────────────> complete
research classification -> queued -> running -> complete | failed
lookup-only synthetic states: not_ready | not_found
```

## Public design decisions

| Decision | Public behavior and rationale |
|---|---|
| Authentication and principal | The standalone server has no authentication provider or actor-token propagation. Every call uses the constant `anonymous` principal; the UUID is the bearer capability. Authorization-like headers are ignored. |
| Malformed or noncanonical capability IDs | The public server requires exact lowercase canonical UUID spelling and returns the same stable `not_found/job_not_found` shape used for unknown UUIDs. |
| Transport path and health | The public endpoint defaults to configurable `/mcp`, with explicit `/live` and `/health`. The optional standalone SSE GET is rejected with 405. Host/Origin checks are DNS-rebinding/browser safeguards, not identity. |
| Lifecycle ownership | MCP 1.28.1 is mounted beneath one outer Starlette lifespan that owns the NAT workflow, job manager, and MCP session manager once per worker. Stateless request teardown cannot stop process-owned work. |
| Startup and environment | Startup validates the public workflow file and `DEEP_RESEARCHER_CHECKPOINT_DB`; public model/search credentials are `NVIDIA_API_KEY` and `TAVILY_API_KEY`, while transport settings use the `DEEP_RESEARCHER_MCP_*` namespace. |
| Public workflow surface | The default workflow uses public NIM and Tavily configuration. Enterprise integrations and paper search are not part of this standalone profile. |
| Certificates | No CA bundle or certificate-mount behavior is bundled. Ordinary platform trust and deployment-level TLS termination are outside this component. |
| Logging | Capability UUIDs and exception text that might echo them are replaced by opaque references in normal logs. Client-visible sanitized error shapes remain compatible. |

Changes to the tool schema, state machine, response fields, polling cadence,
todo normalization, or persistence contract require an explicit update to
this document and its golden tests.

## Compatibility adaptations

- The destination validates `mcp==1.28.1`, `nvidia-nat==1.8.0`, and
  `nvidia-nat-core==1.8.0`.
- MCP protocol `2025-11-25` is exercised with the supported
  `streamable_http_client` and `ClientSession`, including stateless
  initialization and structured tool results.
- NAT 1.8 compatibility covers `load_workflow(config_file)`,
  `SessionManager.shared_builder`, `get_function("intent_classifier")`,
  `session(conversation_id=...)`, `session.run(query)`,
  `runner.result(to_type=ChatResearcherResponse)`, explicit `success`/`failed`
  workflow outcomes, and `Context.scope`.
- The public NAT workflow is loaded in a PostgreSQL-backed compatibility test
  without invoking external model or search calls.

## Executable evidence

| Contract | Tests |
|---|---|
| Tool schemas, descriptions, no-auth settings, lifecycle | `test_server_runtime.py` |
| Exact state/response/cadence/error/todo matrix | `test_jobs.py`, `test_checkpoint_todos.py`, `test_server_runtime.py` |
| Shared ledger, heartbeats, poll counts, TTL, reconciliation | `test_postgres_job_store.py` |
| Real supported-client submit/poll/report flow | `test_client_session_integration.py` |
| NAT/MCP versions and API surface | `test_dependency_compatibility.py`, `test_workflow_runner.py` |
