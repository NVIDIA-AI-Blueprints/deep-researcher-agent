<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenShell Deployment

This is the canonical operator guide for running AI-Q deep-research code in
[NVIDIA OpenShell](https://docs.nvidia.com/openshell/latest/about/installation).
It owns the setup, deployment, acceptance, and troubleshooting contract. The
architecture and implementation pages describe invariants and extension points;
they intentionally link here for operational steps.

OpenShell is the primary path for current sandbox-enabled deep-research
validation. It does not replace AI-Q's non-sandbox workflow defaults, and it
remains experimental until the Linux hard-Landlock acceptance gate passes.

## AI-Q Environment Prerequisite

Run all commands in this guide from the AI-Q repository root using the standard
AI-Q virtual environment:

```bash
./scripts/setup.sh
source .venv/bin/activate
```

`setup.sh` installs the locked AI-Q dependencies, API frontends, data sources,
pre-commit hooks, and UI packages. A new shell must run
`source .venv/bin/activate` again before validation, tests, or E2E startup.

### Setup Command Overview

| Command | Purpose |
|---|---|
| `./scripts/setup.sh` | Create `.venv` and install the standard AI-Q backend, frontend, data-source, and development dependencies |
| `source .venv/bin/activate` | Select the repository Python environment in the current shell |
| `./scripts/openshell/setup_openshell.sh --policy offline` | Install the certified OpenShell SDK/adapter, generate a production hard-Landlock policy, and build the sandbox image |
| `./scripts/openshell/setup_openshell.sh --local-demo --policy offline` | Generate the explicit `best_effort` local-demo policy instead of the production policy |
| `./scripts/openshell/install_gateway.sh` | Explicitly install or repair the certified packaged gateway on Apple Silicon macOS |
| `./scripts/openshell/check_versions.py` | Report safe CLI, SDK, package, and live-gateway version diagnostics |
| `./scripts/openshell/start_openshell_gateway.sh` | Start or reuse the packaged authenticated gateway and run the disposable strict capability probe |
| `nat validate --config_file configs/config_openshell.yml` | Validate the AI-Q workflow and its OpenShell policy/config pairing before startup |
| `./scripts/start_e2e.sh --start-openshell-gateway --config_file configs/config_openshell.yml` | Re-run gateway readiness, then start the AI-Q backend and UI |

## Purpose and Security Boundary

OpenShell executes code generated during deep research. AI-Q orchestration,
inference, retrieval tools, credentials, checkpoints, and report state remain on
the host side. AI-Q does not copy the host environment or model credentials into
the sandbox creation specification.

In production, every job receives a distinct physical sandbox bound to the
submitted policy. AI-Q verifies the authoritative policy source, content, hash,
and revision before exposing the execution adapter. Successful, failed,
timed-out, and cancelled jobs must delete the sandboxes they own.

AI-Q does not reproduce OpenShell's private policy-hash algorithm. It requires
the effective-config policy and revision policy to both equal the submitted
protobuf, requires both authoritative hashes to be non-empty and equal, and
emits that OpenShell-provided hash in the sanitized attestation event.

Attaching to an existing shared sandbox is an explicit debug escape hatch. It is
not job-isolated and is not a production mode. OpenShell is an external runtime
and authentication boundary, not merely a Python dependency: an operator must
own the gateway service, registration, credentials, version, and availability.

## Supported Platforms

| Path | Intended use | Landlock | Gateway owner | AI-Q status |
|---|---|---|---|---|
| Linux + Docker | Production acceptance | `hard_requirement` | systemd or external operator | Tested path after the live suite passes |
| macOS + Docker Desktop | Local demo | `best_effort` permitted explicitly | Homebrew | Demo only |
| Linux + Podman | Operator-managed evaluation | `hard_requirement` | external operator/system service | Supported upstream, not automated or certified by AI-Q acceptance |
| Remote authenticated gateway | Managed deployment | Gateway-host dependent | external operator | Accepted only after the AI-Q live suite passes |
| Windows/WSL | -- | -- | -- | Outside current AI-Q setup-script support |

OpenShell supports Docker and rootless Podman upstream. AI-Q's provisioning and
acceptance automation certify only the exercised Docker path. Follow the
[official OpenShell installation documentation](https://docs.nvidia.com/openshell/latest/about/installation)
for gateway-host prerequisites and upstream runtime support.

## Known Limitations

- OpenShell integration remains experimental until the Linux/Docker live suite
  passes with Landlock `hard_requirement`; macOS `best_effort` results are local
  functional evidence only.
- Docker is the only AI-Q-automated runtime path. Podman is supported upstream
  but not automated or production-accepted by AI-Q; Windows/WSL is unsupported.
- `best_effort` permits execution when Landlock is unavailable, so macOS/Docker
  Desktop does not provide the production filesystem-confinement guarantee.
- OpenShell does not currently satisfy AI-Q's optional CPU/memory resource-limit
  capability. Configuring those limits fails closed rather than silently
  ignoring them.
- Shared named-sandbox attachment is debug-only and is not a tenant or job
  isolation boundary.
- The authenticated gateway is an operator-owned external service. AI-Q can
  validate or start a packaged local service, but E2E shutdown intentionally
  does not stop the gateway.

## Version Compatibility

`scripts/openshell/setup_openshell.sh` accepts only the certified OpenShell
version. `latest` and other exact releases are rejected so an evaluation upgrade
cannot silently become a production stack. The strict gateway launcher then
requires the virtual-environment CLI, SDK, packaged local CLI, and live gateway
to match that version.

The published `langchain-nvidia-openshell==0.1.0` metadata still declares
`deepagents<0.6`, while AI-Q uses DeepAgents 0.6.x. The OpenShell setup therefore
installs the optional adapter and then restores AI-Q's required DeepAgents and
OpenShell versions. This is intentionally isolated from `scripts/setup.sh`; it
may repeat package installation, and it remains an upstream metadata limitation
until a compatible adapter release is published. Do not move an unreleased
adapter into the AI-Q lockfile or hide the conflict with a dependency override.
Until that release exists, `pip check` reports the adapter's declared
DeepAgents-range conflict; this PR must not describe that metadata state as
clean even though the tested adapter surface works with AI-Q's locked runtime.

AI-Q currently requires and defaults to OpenShell `0.0.88`. It retains the
policy-revision and request-label capabilities AI-Q requires, adds explicit
workspace scoping to the SDK, and assigns file-compatible Landlock rights to
non-directory paths. That path handling allows the shipped `/dev/urandom` and
`/dev/null` rules to work on Linux while Landlock remains a
`hard_requirement`; earlier releases can reject those rules during sandbox
startup.

Landlock mode never downgrades automatically. Linux production acceptance uses
`hard_requirement`, so a gateway or host that cannot enforce the required
filesystem policy fails sandbox readiness or creation closed, before research
commands can run. `best_effort` is intended primarily for local macOS/Docker
Desktop functional demos, where Landlock is unavailable: it requires an
explicit AI-Q opt-in, permits the demo to run without the production
filesystem-confinement guarantee, and cannot be reported as Linux production
acceptance.

The certified release is necessary but not sufficient: runtime security decisions
remain capability-based. On the supported `0.0.88` stack, the current, active,
revision, and effective-config versions must all be positive and agree. Any
missing capability or version/content/hash/source mismatch fails closed.

## Responsibility and Lifecycle Ownership

| Component | Owns | Must not do |
|---|---|---|
| `scripts/openshell/setup_openshell.sh` | SDK/adapter install, policy generation, image build | Start, stop, register, select, probe, or kill gateways |
| `scripts/openshell/install_gateway.sh` | Explicit installation or repair of the official packaged local macOS gateway | Install Linux/remote gateways, create custom taps, weaken TLS, or launch raw binaries |
| Homebrew/systemd/external operator | Long-running gateway service and credentials | Delegate process ownership to AI-Q setup |
| `scripts/openshell/start_openshell_gateway.sh` | Validate registration/auth, optionally start a packaged service, select the gateway, and run the strict disposable capability probe | Install or upgrade gateways, launch raw binaries, stop externally managed services, or persist credentials |
| AI-Q runtime | Per-job create, readiness, attestation, execution, and terminal deletion | Reuse a shared sandbox without explicit debug opt-in |
| Live pytest fixtures | Acceptance-test resources and verified teardown | Leave resources for manual cleanup |

Provisioning and long-running service lifecycle are deliberately separate. E2E
shutdown never stops a Homebrew-, systemd-, or operator-managed gateway.

## Policy and AI-Q Config Pairing

Both policy layers are enforced:

- The OpenShell policy is authoritative at the gateway.
- `network` and `network_allow` in the AI-Q config are an upper bound on that
  policy. They never grant additional access.
- `network: blocked` permits no policy endpoint.
- `network: allowlist` requires every endpoint to have a non-empty normalized
  host, and every host must appear in `network_allow`.
- Hostless endpoints and `allowed_ips` or CIDR exceptions are rejected because
  the public AI-Q policy does not model those exceptions.
- Production requires both `landlock.compatibility: hard_requirement` in the
  policy and `require_hard_landlock: true` in the AI-Q config.
- A local demo using `best_effort` requires both the policy value
  `best_effort` and `AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false` when validating
  or running the standard OpenShell config.
- Custom policies must explicitly include OpenShell's proxy filesystem baseline,
  including read-only `/proc`. Otherwise the supervisor creates an enriched
  revision whose content and hash correctly fail AI-Q's exact attestation. The
  generated policy from `scripts/openshell/setup_openshell.sh` already includes this baseline.

Any mismatch fails closed before the execution adapter is available. Keep
environment-specific generated policies out of commits, and never put
credentials in policy or workflow configuration files.

## Environment Contract

The gateway launcher, AI-Q runtime, and live suite use these non-secret settings:

| Variable | Default | Purpose |
|---|---|---|
| `AIQ_OPENSHELL_LIVE_TESTS` | unset | Must equal `1` to enable live tests |
| `AIQ_OPENSHELL_GATEWAY_NAME` | active gateway | Registered gateway name |
| `AIQ_OPENSHELL_WORKSPACE` | `default` | Optional non-default workspace override for sandbox lifecycle operations |
| `AIQ_OPENSHELL_POLICY_FILE` | `configs/openshell/generated/aiq-openshell-policy.yaml` | Policy submitted and attested |
| `AIQ_OPENSHELL_IMAGE` | `aiq-openshell-demo:latest` | Prebuilt sandbox image |
| `AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION` | installed SDK version | Optional exact live-test override |
| `AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK` | `true` | Set `false` only for an explicit local `best_effort` demo |
| `AIQ_OPENSHELL_LIVE_ALLOW_BEST_EFFORT` | unset | Explicit non-production macOS/demo opt-in |

## Data Science Python Sandbox

The opt-in `sandboxed_python` tool uses the same attested OpenShell provider but
requires an offline policy: GSF, knowledge retrieval, web search, model calls,
and credentials remain host-side. Generate a separate policy file so enabling
the Python tool never reuses the network-enabled deep-research policy:

```bash
./scripts/openshell/setup_openshell.sh \
  --policy offline \
  --policy-file configs/openshell/generated/aiq-ds-python-policy.yaml \
  --image-name aiq-openshell-demo:latest
```

Then select the generated policy for the Data Science profile:

```bash
export AIQ_DS_OPENSHELL_IMAGE=aiq-openshell-demo:latest
export AIQ_DS_OPENSHELL_POLICY_FILE="$PWD/configs/openshell/generated/aiq-ds-python-policy.yaml"
```

`sandboxed_python` accepts no host-process backend and rejects OpenShell configs
that enable network access or attach to a shared sandbox. It uploads only its
version-matched one-shot runner, model-authored script request, and bounded
request-local GSF receipt JSON. It re-uploads the authoritative receipts before
every script. Each fresh Python process receives hard Unix limits for memory,
CPU time, process count, open files, and output-file size before model-authored
code runs. The request's terminal cleanup deletes the whole sandbox and its
process tree.
The macOS `best_effort` setup remains a functional demo only; production
acceptance still requires Linux with hard Landlock.

## Linux Production Acceptance

First install and register an authenticated packaged gateway, or arrange an
externally operated gateway, using the official OpenShell documentation. The
registration must use HTTPS and mTLS, OIDC, or trusted edge authentication.
Do not launch a raw `openshell-gateway` process.

From the AI-Q repository root, provision the pinned SDK, hard policy, and image:

```bash
./scripts/openshell/setup_openshell.sh \
  --openshell-version 0.0.88 \
  --policy offline \
  --landlock-compatibility hard_requirement
```

Select the authenticated registration and prove version, policy, selector,
execution, and deletion capabilities. Omit `--reuse-existing` only when the gateway is a local packaged
service that the launcher may start through systemd.

```bash
./scripts/openshell/start_openshell_gateway.sh \
  --gateway-name openshell \
  --image-name aiq-openshell-demo:latest \
  --policy-file configs/openshell/generated/aiq-openshell-policy.yaml
```

The readiness probe proves that the submitted policy is effective, but it does
not independently require hard Landlock. Production acceptance therefore uses
the `hard_requirement` policy generated above and the matching
`require_hard_landlock: true` AI-Q config.

Export the same image and policy for the AI-Q process:

```bash
export AIQ_OPENSHELL_GATEWAY_NAME=openshell
export AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest
export AIQ_OPENSHELL_POLICY_FILE="$PWD/configs/openshell/generated/aiq-openshell-policy.yaml"
export AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION=0.0.88
```

Validate and start AI-Q with the production pairing in
`configs/config_openshell.yml`:

```bash
source .venv/bin/activate
nat validate --config_file configs/config_openshell.yml
./scripts/start_e2e.sh --config_file configs/config_openshell.yml
```

Because the gateway was already verified above, `--start-openshell-gateway` is
not needed here. It is an optional convenience that reruns the same strict probe
before E2E startup.

In a separate shell with the same exported settings, run the required acceptance
suite:

```bash
AIQ_OPENSHELL_LIVE_TESTS=1 \
AIQ_OPENSHELL_GATEWAY_NAME=openshell \
AIQ_OPENSHELL_POLICY_FILE=configs/openshell/generated/aiq-openshell-policy.yaml \
AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest \
AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION=0.0.88 \
.venv/bin/python -m pytest -m integration -vv \
  tests/aiq_agent/agents/deep_researcher/sandbox/test_openshell_live.py
```

Only this Linux, hard-Landlock run can be recorded as production acceptance.

## macOS Local Demo

Use Docker Desktop and the official `nvidia/openshell` Homebrew service. First
provision AI-Q's optional Python components, policy, and image. This step also
prints safe component diagnostics when the packaged gateway is missing or stale.

macOS ships Bash 3.2. Install Bash 5 when the setup script reports unsupported
Bash behavior:

```bash
brew install bash
/opt/homebrew/bin/bash ./scripts/openshell/setup_openshell.sh \
  --openshell-version 0.0.88 \
  --local-demo \
  --policy offline
```

If setup reports `packaged_gateway_missing` or
`component_version_mismatch`, inspect the explicit operation and then run it as
the logged-in user:

```bash
./scripts/openshell/install_gateway.sh --dry-run
./scripts/openshell/install_gateway.sh
```

For Colima, persist the driver configuration in OpenShell's service environment
instead of the caller's transient launchd environment:

```bash
./scripts/openshell/install_gateway.sh --colima
# Or select a specific local socket:
./scripts/openshell/install_gateway.sh \
  --docker-host "unix://$HOME/.colima/default/docker.sock"
```

The wrapper downloads the installer from the certified OpenShell tag to a
temporary file, verifies its checked-in SHA-256, and invokes the official
installer with `OPENSHELL_VERSION=v0.0.88`. It refuses root, non-Apple-Silicon
hosts, and ambiguous OpenShell installations. It never pipes a download into a
shell, creates an AI-Q tap, launches a raw gateway, disables TLS, or stores
credentials.

OpenShell's release installer stages its formula in a local `nvidia/openshell`
tap created with Homebrew's `--no-git` mode. Because that tap has no Git remote,
`brew upgrade openshell` cannot fetch a newly released formula; it can leave the
packaged gateway on `0.0.72` while AI-Q's virtual-environment CLI/SDK is `0.0.88`.
Rerun the explicit pinned installer wrapper when directed. Do not copy a formula
manually, create a custom AI-Q tap, use `launchctl setenv`, or keep multiple
OpenShell service identities.

You can inspect the safe version report directly:

```bash
.venv/bin/python scripts/openshell/check_versions.py
.venv/bin/python scripts/openshell/check_versions.py --json
```

Then validate the standard config, validate the gateway, and start AI-Q with the
explicit local-demo environment override:

```bash
source .venv/bin/activate
AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false \
  nat validate --config_file configs/config_openshell.yml

./scripts/openshell/start_openshell_gateway.sh \
  --gateway-name openshell \
  --image-name aiq-openshell-demo:latest \
  --policy-file configs/openshell/generated/aiq-openshell-policy.yaml

AIQ_OPENSHELL_POLICY_FILE=configs/openshell/generated/aiq-openshell-policy.yaml \
AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest \
AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION=0.0.88 \
AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false \
./scripts/start_e2e.sh --config_file configs/config_openshell.yml \
  2>&1 | tee e2e-openshell-0.0.88.log
```

Run the same mechanics through the convenience wrapper with the explicit demo
opt-in:

```bash
.venv/bin/python scripts/openshell/smoke_openshell_isolation.py \
  --gateway openshell \
  --policy configs/openshell/generated/aiq-openshell-policy.yaml \
  --image aiq-openshell-demo:latest \
  --expected-gateway-version 0.0.88 \
  --allow-best-effort-landlock
```

A passing macOS run is useful local evidence, but it does not satisfy Linux
production acceptance.

## Existing Remote Gateway

The remote gateway must already be registered over HTTPS with mTLS, OIDC, or
trusted edge authentication. The launcher validates the registration and refuses
to substitute a local gateway if the remote service is unavailable:

```bash
./scripts/openshell/start_openshell_gateway.sh \
  --gateway-name enterprise \
  --reuse-existing \
  --image-name aiq-openshell-demo:latest \
  --policy-file configs/openshell/generated/aiq-openshell-policy.yaml
```

The disposable strict capability probe is mandatory. After it passes, export
`AIQ_OPENSHELL_GATEWAY_NAME=enterprise` and run the live suite. Set
`AIQ_OPENSHELL_WORKSPACE` only when the registered gateway uses a non-default
workspace. Never fall back to a plaintext registration, insecure TLS, or a
local raw gateway.

## Shared Debug Attachment

Create a named shared sandbox only when debugging requires it:

```bash
./scripts/openshell/start_openshell_gateway.sh \
  --gateway-name openshell \
  --create-shared-debug-sandbox \
  --sandbox-name aiq-openshell-demo
```

Attachment also requires `allow_shared_sandbox: true` and an explicit
`existing_sandbox_name` in a local AI-Q config. When a policy is supplied, AI-Q
requires strict effective-policy attestation and rejects `attest: false`. Without
a policy, the attachment still requires READY/loaded-version checks and emits
`assurance=reduced`. The attaching job never owns or deletes the shared sandbox;
the operator or test fixture that created it remains responsible.

## Expected Runtime Behavior

The human-readable contract is:

- One running deep-research job creates one physical OpenShell sandbox.
- Two concurrent jobs create two distinct sandbox names and physical IDs.
- Active jobs are discoverable with `--selector aiq=deep-research`, with a
  distinct `aiq-job-id` gateway label for each job.
- Attestation succeeds before the execution adapter is exposed.
- Success, command failure, timeout, and cancellation delete owned sandboxes.
- Cancelling one job does not delete or replace another job's sandbox.
- `sandbox.attestation` reports sanitized status, policy version, hash, source,
  assurance, and reason code.
- `sandbox.cleanup` reports `started`, `succeeded`, or `failed`, with stable
  `reason_codes` on failure.
- Credentials, policy contents, SDK response bodies, and exception messages are
  not emitted in lifecycle events or failure logs.

The final job state is separate from physical cleanup: verify the cleanup event
and absence from the gateway rather than assuming a terminal job status deleted
the resource.

## Artifact Capture

`configs/config_openshell.yml` enables durable sandbox artifact capture. Successful
`execute` calls checkpoint declared artifacts, and terminal finalization performs
one idempotent scan before sandbox cleanup. Metadata is stored in the job database;
bytes use the configured SQL or S3-compatible artifact blob provider. Clients
receive `artifact.update` metadata with a `content_url`, never raw bytes in SSE.

For validation, storage configuration, event payloads, and report rendering, see
the developer [artifact runtime](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/src/aiq_agent/agents/deep_researcher/sandbox/README.md#artifact-runtime)
and [production artifact storage](./production.md#artifact-storage) guides.

## Acceptance Tests

The canonical acceptance entry point is pytest:

```bash
AIQ_OPENSHELL_LIVE_TESTS=1 \
AIQ_OPENSHELL_GATEWAY_NAME=openshell \
AIQ_OPENSHELL_POLICY_FILE=configs/openshell/generated/aiq-openshell-policy.yaml \
AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest \
AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION=0.0.88 \
.venv/bin/python -m pytest -m integration -vv \
  tests/aiq_agent/agents/deep_researcher/sandbox/test_openshell_live.py
```

The suite contains three independently reported tests:

- `test_live_per_job_isolation_attestation_and_cancellation` proves distinct
  sandboxes, authoritative source/content/hash/revision attestation, isolated
  cancellation, selector membership, continued execution, and terminal deletion.
- `test_live_failure_cleanup_and_log_redaction` proves cleanup after a deterministic
  failed command and verifies that a credential-shaped exception canary reaches
  neither logs nor events.
- `test_live_shared_policy_mismatch_is_rejected` proves that a structurally
  different claimed policy cannot attach successfully, while the directly owned
  shared sandbox remains usable until fixture teardown.

Every fixture registers resources immediately, tears them down in reverse order,
and verifies deletion through the gateway. A teardown failure fails the test even
when the test body also failed. Without `AIQ_OPENSHELL_LIVE_TESTS=1`, all three
tests are collected and skipped before optional OpenShell imports or gateway
connections.

`scripts/openshell/smoke_openshell_isolation.py` is a convenience wrapper only. It maps its
arguments to the environment contract, enables the live gate, and returns pytest's
exit code unchanged. Pytest owns every assertion and cleanup fixture.

Record the non-secret gateway version, policy path, image tag, platform, and
Landlock mode with acceptance results. Do not record registrations, environment
values, policy contents, response bodies, or credentials.

## Inspection and Troubleshooting

Inspect only registered resources and sanitized AI-Q lifecycle events:

```bash
.venv/bin/openshell status
.venv/bin/openshell gateway list -o json
.venv/bin/openshell sandbox list
.venv/bin/openshell sandbox list --selector aiq=deep-research -o json
```

The selected gateway registration must be HTTPS and report `mtls`, `oidc`, or
trusted edge authentication. During a job, the selector must show one owned
sandbox per active deep-research job. After termination, each owned name must be
absent from direct and selector listings. Use the sandbox name from sanitized `sandbox.attestation` or
`sandbox.cleanup` events; do not expose full SDK payloads to logs.

| Failure | Safe action |
|---|---|
| Generated policy or image is missing | Run `scripts/openshell/setup_openshell.sh` and reuse the exact paths it prints. |
| `packaged_gateway_missing` on Apple Silicon macOS | Run `./scripts/openshell/install_gateway.sh --dry-run`, then explicitly approve the installer. |
| `component_version_mismatch` on local macOS | Run `.venv/bin/python scripts/openshell/check_versions.py`, then use the exact local installer remediation it prints. |
| `ambiguous_gateway_installation` | Remove the obsolete OpenShell formula/service identity through Homebrew before retrying. Do not let AI-Q guess which service to replace. |
| `remote_gateway_version_mismatch` | Coordinate an upgrade with the registered gateway owner. AI-Q never replaces a remote service with a local one. |
| `gateway_unavailable` | Start or verify the packaged local service with `scripts/openshell/start_openshell_gateway.sh`, or contact the remote gateway owner. Do not reinstall a matching stack merely because the service is stopped. |
| CLI, SDK, and gateway versions differ | Do not start AI-Q or create a probe sandbox. Align all reported components to the certified version and rerun the launcher. |
| `request_labels_unsupported` | The installed Python SDK cannot persist gateway labels required for AI-Q ownership and selectors; install a supported release. |
| `policy_status_inconsistent` | The effective policy matches but its revision never became `LOADED`. This is an OpenShell lifecycle failure, not a Landlock-mode mismatch. Do not disable attestation. |
| `policy_content_mismatch` | Regenerate the policy with `scripts/openshell/setup_openshell.sh`, or add the required OpenShell proxy filesystem baseline (including read-only `/proc`) to a custom policy. Do not weaken exact attestation. |
| `selector_mismatch` | The probe was not discoverable through gateway metadata. Do not rely on Docker/template labels as a substitute. |
| Registration is plaintext or unauthenticated | Register an HTTPS gateway with mTLS, OIDC, or trusted edge authentication. Do not bypass the launcher check. |
| Docker daemon is unavailable | Start the operator-owned Docker service and rerun provisioning/probe. |
| Podman is selected | Follow upstream OpenShell guidance; do not report the path as AI-Q production-accepted. |
| Landlock policy/config mismatch | Pair `hard_requirement` with the default config, or set `AIQ_OPENSHELL_REQUIRE_HARD_LANDLOCK=false` only with an explicit `best_effort` demo policy. |
| Policy is broader than `network_allow` | Remove the endpoint or add its exact normalized hostname to the declared upper bound. Do not add CIDR exceptions. |
| Sandbox never becomes Ready | Inspect the owning gateway/runtime service, image availability, and sanitized sandbox status; do not dump SDK bodies. |
| Probe or job deletion cannot be verified | Treat acceptance as failed, identify the exact sandbox, and retry explicit deletion through the registered gateway. |
| macOS reports Bash 3.2 incompatibility | Install Bash 5 and invoke the setup script with its absolute path. |

For a named sandbox that this operator owns, use explicit cleanup and verify its
absence:

```bash
.venv/bin/openshell sandbox delete <identified-sandbox-name>
.venv/bin/openshell sandbox list
```

Manage a packaged gateway only through its owner:

```bash
brew services restart nvidia/openshell/openshell
systemctl --user restart openshell-gateway
```

Never use broad `pkill`, launch the raw gateway binary, enable insecure TLS, or
perform destructive cleanup without first identifying the owned resource. Do not
stop an externally managed gateway from AI-Q shutdown logic.
