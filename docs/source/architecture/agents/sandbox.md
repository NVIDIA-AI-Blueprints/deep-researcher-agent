<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Research Sandbox Notes

Deep research can optionally run DeepAgents `execute` calls through a sandbox provider
(Modal, OpenShell, or any registered provider). Modal and OpenShell create a fresh physical
sandbox per job. OpenShell binds the configured policy at creation and attests the authoritative
effective policy source, content, hash, and active revision before exposing the execution backend.
OpenShell lifecycle calls are explicitly scoped to the configured workspace (default: `default`).
On the supported OpenShell `0.0.88` stack, current, active, revision, and effective-config versions
must all be positive and agree with the exact submitted policy/hash; missing capabilities or any
version disagreement fails closed.

The sandbox is an internal execution detail. There are no sandbox-specific API
endpoints, and job-level auth remains responsible for submit, stream, status,
cancel, state, and report access. User-visible artifact surfaces are the Files tab
and the artifact runtime (`.../job/{job_id}/artifacts`), which follows the same job
access policy and is owner-scoped when `REQUIRE_AUTH=true`.

> **Developer reference:** the full architecture, provider contract, config schema,
> artifact pipeline, and troubleshooting live next to the code in
> [`src/aiq_agent/agents/deep_researcher/sandbox/README.md`](https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/blob/release/2.2/src/aiq_agent/agents/deep_researcher/sandbox/README.md).
> Operators should use the canonical [OpenShell deployment guide](../../deployment/openshell.md)
> for setup, authenticated gateway lifecycle, supported platforms, acceptance, and cleanup.

## Current Behavior

- Modal and OpenShell use one physical sandbox per deep research job. OpenShell shared
  attachment is available only through an explicit debug-only opt-in and is not job-isolated.
- Synchronous sandbox-enabled runs use an internal per-agent runtime ID.
- Providers are selected by config (`sandbox.provider` + `providers.<name>`); the
  provider is validated against the registry and gated by its declared capabilities.
  OpenShell policy YAML is parsed strictly against the installed SDK schema, applied in the
  job's creation spec, checked against the declared network upper bound (hostless/CIDR overrides
  are rejected), and attested through the gateway status/config RPCs before use.
- Job IDs must satisfy each provider's object-name rules (Modal: 64 chars or fewer,
  alphanumeric plus dash/period/underscore).
- `timeout` bounds individual execution. Other lifecycle controls are provider-dependent.
- Files written inside the workdir are temporary scratch state. Durable text should
  be written through DeepAgents virtual paths such as `/shared/`; durable binaries
  (charts, CSVs) are captured by the artifact runtime.

## Artifact Storage

Metadata for generated files such as charts and CSVs is stored in SQL. File content can
also be stored in SQL, but object storage such as AWS S3 or MinIO is recommended for
production deployments. The selected artifact storage provider determines where the file
content is stored.

Each captured file emits a metadata-only `artifact.update` event. Stored events populate
the Files tab during both live execution and replay; file bytes remain behind the
job-scoped artifact content endpoint. Rejected candidates emit `artifact.warning`.

For configuration variables and examples, refer to [Docker Compose](../../deployment/docker-compose.md#artifact-storage)
and [Production Considerations](../../deployment/production.md#artifact-storage).

## Operational Notes

- High-concurrency Modal and OpenShell runs create one sandbox per job. Optional submit-path
  caps (`AIQ_MAX_SANDBOXES_PER_PRINCIPAL` / `AIQ_MAX_SANDBOXES_GLOBAL`, default-off) bound
  concurrency and cost.
- Custom client-supplied job IDs must not be reused for a new job.
- Manifest checkpoints preserve completed artifacts after successful sandbox commands. The
  terminal finalizer harvests once before cleanup on success/failure; cancellation harvests
  only when the provider is idle and otherwise terminates immediately.
- The runtime closes provider sessions on success, failure, cancellation, and timeout.
- Per-job OpenShell mode requires `delete_on_exit: true`. A persistent shared sandbox is
  possible only through the explicit debug attachment settings.
- OpenShell provisioning and authenticated gateway lifecycle have separate owners. See the
  [OpenShell responsibility table](../../deployment/openshell.md#responsibility-and-lifecycle-ownership)
  rather than duplicating operational commands here.

## Current Safeguards

The following safeguards are in place:

- Explicit sandbox cleanup on success, failure, cancellation, and timeout.
- Idempotency-gated retry-on-stale-container handling.
- Artifact capture for generated charts/binaries (validate -> store -> serve/embed),
  with MIME-from-bytes spoof rejection, SVG sanitization, and an inline-render allowlist.
- Sandbox quota and concurrency controls, and periodic time-based artifact retention cleanup.
- Structured lifecycle logging for sandbox create, reuse, failure, and cleanup.
- Structured `sandbox.attestation` and `sandbox.cleanup` events.
