<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Async Job Content Encryption

Deep Researcher Agent can encrypt sensitive async job content before it is persisted by the
Deep Researcher Agent async jobs API. This is application-level envelope encryption for final
report output and selected artifact event payload fields.

Content encryption is disabled by default. Operators must explicitly set
`DEEP_RESEARCHER_CONTENT_ENCRYPTION=key` or `vault` on every API and worker process to
enable it.

This feature is intentionally narrow in its first milestone. It protects the
serialized final output payload returned by
`GET /v1/jobs/async/job/{job_id}/report`, such as `{"report": "..."}`, and
the `content` field of `artifact.update` events for `output` and `file`
artifacts.

## Scope and Limitations

Encrypted when enabled:

- `job_info.output` for jobs submitted through `/v1/jobs/async/submit` or
  `deep_researcher_api.jobs.submit.submit_agent_job`.
- `job_events.event_data` field `artifact.update.data.content` when the
  artifact `data.type` is `output` or `file`.

Still plaintext:

- Other `job_events.event_data` fields, including artifact metadata, tool
  events, heartbeat events, cancellation events, error events, citation source
  artifacts, citation use artifacts, and todo artifacts.
- Job status, ownership metadata, timestamps, event type, and other control
  plane fields.
- `job_info.error`.
- PostgreSQL notification payloads.
- `summaries.summary`.
- LangGraph checkpoints in `deep_researcher_checkpoints`.
- Historical `job_info.output` rows written before encryption was enabled.
- Inline CLI and local NeMo Agent Toolkit runs that do not use the Deep Researcher Agent async
  API job runner.

Because checkpoints, errors, citations, todos, and other event fields can
contain equivalent research content, this phase does not provide full
database-level job-content confidentiality.

This feature protects the configured fields from storage readers and detects
ciphertext tampering within the authenticated job and field context. It is not
a database integrity or event-freshness control. A principal with database
write access can still delete or reorder rows, replay an entire encrypted row,
or swap encrypted artifact content between events in the same job when the
authenticated field path is identical.

## Modes

Set `DEEP_RESEARCHER_CONTENT_ENCRYPTION` on every API and worker process.

| Mode | Behavior |
|------|----------|
| `off` | Default. Preserves existing plaintext behavior and never attempts to decrypt `aiqenc:` values. |
| `key` | Uses one operator-managed static 32-byte key to wrap per-job data encryption keys. Intended only for development, testing, or deployments that cannot use Vault. |
| `vault` | Uses HashiCorp Vault Transit to generate and wrap per-job data encryption keys. Recommended for production. |

Encrypted values are stored as `aiqenc:` envelopes. Encrypted event fields are
stored as marker objects containing the envelope so the surrounding event JSON
remains inspectable. The envelope contains non-secret metadata, the wrapped data
encryption key, nonce, ciphertext, tag, algorithm, key id, and an AAD hint that
binds the value to either `job_info.output:{job_id}` or the encrypted
`job_events.event_data` field path for that job.

## Static Key Configuration

Static key mode requires a base64 or base64url value that decodes to exactly
32 raw bytes.

```bash
DEEP_RESEARCHER_CONTENT_ENCRYPTION=key
DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY=<base64url-encoded-32-byte-key>
DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID=<operator-managed-key-id>
```

`DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID` is optional, but it is cryptographic identity,
not cosmetic metadata: it is authenticated while wrapping each data encryption
key. If omitted, envelopes use `static-key` as the key id. Keep the configured
key id unchanged while encrypted jobs are retained. The first implementation
supports one active static key only; changing the key or key id makes existing
envelopes unreadable unless those jobs have expired or a future rewrap process
has migrated them.

Invalid static-key configuration fails startup.

## Vault Transit Configuration

Vault mode uses AppRole authentication and Transit data keys. Token fallback is
not supported in the first implementation.

```bash
DEEP_RESEARCHER_CONTENT_ENCRYPTION=vault
VAULT_ADDR=<vault-address>
VAULT_ROLE_ID=<approle-role-id>
VAULT_SECRET_ID=<approle-secret-id>
DEEP_RESEARCHER_ENCRYPTION_TRANSIT_KEY=<transit-key-name>
VAULT_TRANSIT_MOUNT=<transit-mount>
DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID=<logical-key-id>
```

`VAULT_TRANSIT_MOUNT` defaults to `transit` if omitted.
`DEEP_RESEARCHER_CONTENT_ENCRYPTION_KEY_ID` is optional; if omitted, envelopes use
`<transit-mount>/<transit-key-name>`.

Set `VAULT_NAMESPACE=<vault-namespace>` only when your Vault deployment
requires a namespace.

Missing Vault configuration fails startup. If Vault configuration is present
but Vault is temporarily unreachable, unauthorized, or otherwise operationally
unready, the API starts unhealthy instead of exiting.

The application relies on Vault Transit versioned ciphertext for decrypting
after Transit key rotation. Do not disable or destroy old Transit key versions
until corresponding encrypted jobs have expired or have been rewrapped by a
future migration process.

## Vault Client and Retry Behavior

Vault mode uses the synchronous `hvac.Client`. API routes that may call Vault
offload that work to worker threads so Vault latency does not block the FastAPI
event loop. Each API or worker process keeps one process-local Vault client and
guards client creation, AppRole login, and Transit calls with a lock. This
serializes Vault Transit operations per process and avoids concurrent mutation
of the shared `hvac.Client`.

Vault Transit operations use bounded retries for transient failures only:

- request connection failures and timeouts,
- Vault rate limiting,
- Vault 5xx-style operational failures, and
- unauthorized responses after forcing one AppRole re-login.

The retry policy does not retry missing configuration, invalid requests,
permission-denied responses, missing Transit paths, malformed Vault responses,
or decrypt denials. Tune `VAULT_TIMEOUT_SECONDS` to bound individual Vault
requests and `DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS` to control how
often health and submit readiness recheck Vault.

## Rollout Behavior

The feature defaults to `off`, so upgrading Deep Researcher Agent does not change persisted job
content unless an operator explicitly enables encryption. The first
implementation is forward-only after enablement:

- New `job_info.output` writes are encrypted after enablement.
- New `artifact.update.data.content` event fields are encrypted after
  enablement for `output` and `file` artifacts.
- Existing plaintext `job_info.output` rows are intentionally unreadable while
  `DEEP_RESEARCHER_CONTENT_ENCRYPTION=key` or `vault`.
- Existing plaintext event rows remain readable in encrypted modes.
- No historical plaintext backfill is included.
- No rewrap tooling is included.

Migration, backfill, application-managed key rotation, and decrypt-on-rollback
tooling are outside the scope of this release. Enable encryption only for a new
or empty job history, after existing jobs have expired, or after operators
accept that old plaintext final-report rows in `job_info.output` cannot be read
in encrypted modes.

Keep the encryption mode, static key and key id, or Vault Transit mount and key
name stable while encrypted jobs are retained. Switching the mode to `off`
does not decrypt existing values: report reads can expose the stored `aiqenc:`
value and encrypted event fields remain marker objects. Restore the original
encryption configuration to read those jobs.

## Health and Failure Behavior

`/live` reports process liveness without calling Vault or the database. `/health`
reports dependency readiness and includes encryption status. When encryption is
configured but unready, `/health` returns HTTP 503 and new async submissions are
rejected with HTTP 503, while `/live` remains successful so an orchestrator does
not restart an otherwise live API process during a dependency outage.

Workers independently validate encryption before marking a job `RUNNING`. If
encryption is unavailable at worker startup, the job is marked `FAILURE` and
the agent does not run.

Each submission also carries a non-secret encryption policy identity from the
API process to the worker. The identity covers the mode and key id, a static-key
fingerprint in `key` mode, or the Vault address, namespace, Transit mount, and
Transit key in `vault` mode. The worker fails the job before `RUNNING` if its
local policy does not match, including `key` or `vault` submissions received by
a worker configured with `off`.

If final-report encryption, artifact event-content encryption, or encrypted
persistence fails after an agent has completed, the job is marked `FAILURE`.
Timer-triggered event flush failures are surfaced before the worker can mark
the job `SUCCESS`. The worker does not fall back to writing plaintext output.
When encryption is `off`, event persistence retains its legacy best-effort
behavior: database write failures are logged without changing the job status.

Report reads fail closed:

- Vault or crypto unavailability returns HTTP 503.
- Plaintext, malformed, or undecryptable `job_info.output` in encrypted mode
  returns HTTP 500.
- Job access is authorized before decryption is attempted.

## Cache

Decrypt paths use an in-memory plaintext data encryption key cache per process.
The default TTL is 15 minutes with a maximum of 1024 entries. Set
`DEEP_RESEARCHER_CONTENT_ENCRYPTION_DEK_CACHE_TTL_SECONDS=0` to disable this cache.

The readiness cache defaults to 60 seconds. Health checks and submit requests
reuse the cached state until it becomes stale. Set
`DEEP_RESEARCHER_CONTENT_ENCRYPTION_READINESS_TTL_SECONDS` to override the default.
