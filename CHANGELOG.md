# Change Log

Release v2.2.0 (2026-08-17)

Published NVIDIA NGC artifacts:

- Backend container: [`nvcr.io/nvidia/blueprint/aiq-agent:2.2.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/containers/aiq-agent/2.2.0)
- Frontend container: [`nvcr.io/nvidia/blueprint/aiq-frontend:2.2.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/containers/aiq-frontend/2.2.0)
- Helm chart: [`nvidia/blueprint/aiq2-web:2.2.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/blueprint/helm-charts/aiq2-web/2.2.0)

**Research and reports**

- Routed deep research now uses explicit source-router, structured planner, concurrent researcher, and writer roles, with bounded source-tool batching and no research-plan approval step
- Report follow-up supports answers over a completed report, child-job cosmetic rewrites, and delta research that carries the parent report forward as context
- Clarification is more targeted: it can search for context before asking the user to narrow scope or choose an output shape
- The writer produces figures through an on-demand `chart-generation` skill in the new `visualization` skill collection, rendering a sandbox PNG artifact when a sandbox is available and an inline chart spec otherwise; the skill ships enabled only in the skills and sandbox example configs (`config_domain_routing_and_skills`, `config_openshell`), while every other config presents chart-worthy data as a Markdown table, and a writer that wants inline charts must be assigned the `visualization` collection (charts are not sandbox-gated)

**Sources and integrations**

- OpenSearch is a first-class knowledge backend for self-hosted, Amazon OpenSearch Service, and Amazon OpenSearch Serverless deployments
- Azure AI Search is a managed knowledge backend with API-key or Azure identity authentication, namespaced index ownership, and hybrid retrieval
- Paper search adds SerpAPI and SearchAPI providers alongside Serper; the routed-research profile adds DuckDuckGo news and Polymarket sources
- You.com adds configurable web search, page-content extraction, cited open-domain research, and finance-focused research tools
- Nimble adds configurable web search with lite and deep modes, plus an Enterprise-only fast mode and optional focus, country, and locale controls
- Per-user MCP OAuth adds status, connect, callback, and reconnect flows backed by a token store shared by the API and workers; disconnect and in-worker token refresh are not included
- A standalone public MCP server exposes stateless submit, poll, and final-report tools over Streamable HTTP with PostgreSQL-backed job state

**Sandboxes, artifacts, and policy**

- DeepAgents execution uses a provider-neutral, job-scoped sandbox contract: Modal and OpenShell create one physical sandbox per deep-research job; OpenShell additionally applies and attests the configured policy before execution
- Opt-in durable artifact capture checkpoints manifest-declared files after successful sandbox `execute` calls, performs one final manifest-plus-directory scan on success/failure, and preserves earlier checkpoints without delaying cancellation when the provider is busy
- Captured files store metadata in SQL and bytes in SQL or S3-compatible storage, emit metadata-only `artifact.update` events for live and replayed Files-tab access, and remain available through job-scoped list/content endpoints that enforce ownership when `REQUIRE_AUTH=true`
- Opt-in NeMo Guardrails middleware covers selected workflow and agent input/output boundaries; defining middleware does not activate every boundary
- Opt-in content encryption protects final async output and selected artifact event content only; it is off by default, forward-only, and does not encrypt checkpoints or most job/event metadata
- Summary Store database logging masks URL passwords and removes query parameters so credentials and query-string secrets are not written to initialization or lifecycle logs

**Deployment and observability**

- The repository source Helm chart honors `helm install -n <namespace>` for every namespaced resource, including GitOps-rendered deployments; chart metadata advances to `deep-researcher-web` 2.2.0 with the `deep-researcher` 0.0.5 dependency
- NAT-exported async-job traces preserve configured workflow, task/batch, named-agent, and model/tool hierarchy across concurrent researchers without copying graph-state content into structural agent spans
- Deep-research intake uses atomic per-principal, deployment-wide, and per-minute admission controls before Dask enqueue; each job also enforces hard input, runtime, plan, report, shared-state, query, note, todo, and source-tool budgets
- Document ingestion enforces server-side file-count, per-file, aggregate-size, declared-type, and content validation using the same upload settings as the UI
- The embedded Dask scheduler, dashboard, and worker listeners bind to loopback; production guidance defines the customer-operated security boundaries for external Dask, provider quotas, authentication, and S3-compatible artifact storage

**Agent Skills, UX, and developer workflow**

- Consumer Agent Skills now include `deep-researcher-deploy` and `deep-researcher-research`; maintainer skills cover workflow configuration, data sources, tools, release QA, PR preparation, prompt/model customization, and CI maintenance
- The UI surfaces batched researcher activity and improves research-session recovery, expiry handling, and WebSocket delivery reliability
- Contributor governance and product-level Agent Skill evaluation checks expand release and contribution tooling
- Pinned to NeMo Agent Toolkit (NAT) v1.8.0

The eleven checked-in workflow configurations are focused profiles; no single profile enables every 2.2 capability.

Release v2.1.0

- Deep Researcher Agent REST API with pluggable auth middleware, entry-point-registered token validators, and async job ownership enforcement
- Auth extensibility hooks (`register_token_fetcher`, provider lifecycle) and auth refactor eliminating the refresh race
- Data source registry driving UI toggles, per-message filtering, and agent tool inheritance
- New `exa_web_search` data source with `full_text` and `highlights` controls
- Deep researcher consumes DeepAgents skills with a job-scoped Modal sandbox; built-in `data-table-analysis` skill and `configs/config_skills.yml` example
- Deep Researcher Agent is consumable as a portable Agent Skill (`.agents/skills/deep-researcher-research/`), with `.claude/skills/deep-researcher-research/` retained as a Claude Code compatibility symlink for routed `/chat` and async job lifecycle against a local Deep Researcher Agent server
- Cost analysis tool with pricing configs and profiling example
- Documented MCP client patterns scoped for 2.1: `mcp_client`, `mcp_service_account`, and user-identity tools
- Prompt restructure across all agents for KV cache prefix reuse
- Operability: idempotent DB init, tuned Dask/Postgres defaults, request tracing into NAT spans, UI stream-failure hardening
- New authentication and MCP tools guides; new skills-and-sandbox example
- Pinned to NeMo Agent Toolkit (NAT) v1.6.0; CVE bumps for Pillow, cryptography, pygments, authlib, pyopenssl, and pytest

Release v2.0.0

Ground-up rewrite of the NVIDIA Deep Researcher Agent Blueprint, built on the NVIDIA NeMo Agent Toolkit (NAT).

- Two-tier research architecture with automatic routing between shallow (fast, bounded) and deep (multi-phase, report-grade) research via a single-call Intent Classifier
- Deep Researcher rebuilt with a three-role subagent architecture (Orchestrator, Planner, Researcher) using the `deepagents` library, with configurable research loops and per-role LLM assignment
- New Shallow Researcher agent with tool-call budgets, context compaction, and synthesis anchors for citation-backed answers
- Clarifier agent with human-in-the-loop plan generation, approval, and feedback before deep research
- Shallow-to-deep escalation when the shallow researcher detects insufficient results
- Async Jobs REST API (`/v1/jobs/async/`) with SSE streaming, event replay, reconnection support, and cooperative cancellation
- Dask-based distributed execution with configurable workers, heartbeats, and stale job reaping
- PostgreSQL persistence for job store, event store, LangGraph checkpoints, and document summaries
- Pluggable Knowledge Layer with factory/registry pattern — swap between LlamaIndex (local ChromaDB) and Foundational RAG (hosted NVIDIA RAG Blueprint) without code changes
- Multimodal document extraction (VLM-powered image captioning and chart data extraction)
- Document summaries injected into agent prompts for file-aware research
- Deterministic citation verification pipeline with five-level URL matching, report sanitization, and audit trail
- New Next.js frontend with conversational UI, document upload, collection management, and real-time progress streaming
- Optional OAuth/OIDC authentication with configurable providers
- Multi-backend observability: Phoenix, LangSmith, W&B Weave, and OpenTelemetry Collector with privacy redaction
- FreshQA benchmark for shallow researcher factuality evaluation via `nat eval`
- Docker Compose and Helm chart deployments with distroless runtime images, non-root execution, and horizontal scaling
- Native NAT integration — all configuration through YAML with `nat run` / `nat serve` / `nat eval`
- Four pre-built configs: CLI default, Web + LlamaIndex, Web + Foundational RAG, Hybrid Frontier Model
- uv workspace monorepo, Jupyter notebook tutorial series, and debug console at `/debug`
- Pinned to NeMo Agent Toolkit (NAT) v1.4.0; Python 3.11–3.13; Node.js 22+
- Deep Researcher Agent holds top positions on both DeepResearch Bench and DeepResearch Bench II leaderboards (see `drb1` and `drb2` branches)

Release v1.2.1
- Upgraded llama-3.3-70b-instruct NIM from version 1.13.1 to 1.14.0
- Aligned Helm values and referenced Docker image tags with the new nim-llm version
- Adopted RAG 2.3.2
- Removed manual NIM_MODEL_PROFILE configuration from Helm values and Docker Compose to rely on automatic profile detection, updated documentation accordingly

Release v1.2.0
- Added support for Helm deployments
- Add support and documentation for evaluation
- Simplified the configuration and integration with RAG, removing nginx
- Adopted RAG 2.3.0
- Tested for compatability with RTX Pro 6000

Release v1.1.0
- Tested for compatability with RAG 2.2.0 release and B200
- Adds support for NVIDIA Workbench

Release v1.0.0

Initial release of the NVIDIA Deep Researcher Agent Research Assistant Blueprint featuring:
- Multi-modal PDF document upload and processing, compatible with the NVIDIA RAG 2.1 blueprint release
- Demo web application
- Deep research report writing including human-in-the-loop feedback
