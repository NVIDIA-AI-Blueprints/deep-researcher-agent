# Basic Validation

These checks confirm the deployed Deep Researcher Agent system is reachable and minimally usable. They are not report-quality scoring.

## Determine Server URL

Default:

```bash
PORT="${PORT:-8000}"
DEEP_RESEARCHER_SERVER_URL="${DEEP_RESEARCHER_SERVER_URL:-http://localhost:$PORT}"
echo "DEEP_RESEARCHER_SERVER_URL=$DEEP_RESEARCHER_SERVER_URL"
```

If the user configured a custom `PORT` or external host, use that URL.

## Backend API

```bash
curl -sf "$DEEP_RESEARCHER_SERVER_URL/health" >/dev/null && echo "backend=healthy"
```

If `/health` is unavailable, try `/v1/health` before failing:

```bash
curl -sf "$DEEP_RESEARCHER_SERVER_URL/v1/health" >/dev/null && echo "backend=healthy"
```

## UI When Applicable

Run this only for deployment modes that intentionally start the browser UI:

```bash
curl -sf "http://localhost:${FRONTEND_PORT:-3000}" >/dev/null && echo "frontend=reachable"
```

## PostgreSQL When Using Docker Compose

Run this only for Docker Compose deployments. It is not required for local process or CLI modes unless the selected config explicitly uses a local PostgreSQL service.

```bash
docker exec deep-researcher-postgres pg_isready -U deep_researcher -d deep_researcher_jobs
docker exec deep-researcher-postgres pg_isready -U deep_researcher -d deep_researcher_checkpoints
```

## Async Agent API

Use the installed `deep-researcher-research` helper from the skill checkout when available:

```bash
DEEP_RESEARCHER_SERVER_URL="$DEEP_RESEARCHER_SERVER_URL" python3 skills/deep-researcher-research/scripts/deep_researcher.py health
DEEP_RESEARCHER_SERVER_URL="$DEEP_RESEARCHER_SERVER_URL" python3 skills/deep-researcher-research/scripts/deep_researcher.py agents
```

## Shallow End-To-End Check

Run a shallow `/chat` check when required model/search credentials are present. If credentials are missing, report that deploy validation reached infrastructure/API readiness but could not prove model-backed response generation.

```bash
DEEP_RESEARCHER_SERVER_URL="$DEEP_RESEARCHER_SERVER_URL" python3 skills/deep-researcher-research/scripts/deep_researcher.py chat "Briefly confirm Deep Researcher Agent is responding."
```

Do not run deep research as part of basic deploy validation. Deep research belongs to `deep-researcher-research` when requested, and broader integration validation belongs to `end-to-end-validation.md`.

## Optional Deep Research Completion Validation

Basic deploy validation does not prove that deep research can complete. It confirms that services are reachable and, when credentials are present, that a shallow model-backed request can run. Use `end-to-end-validation.md` for the optional deeper check: submit an explicit `deep_researcher` job, poll it to completion, and fetch the final report.

## Handoff

When validation passes, tell the user:

- backend URL
- frontend URL when applicable, or that the UI was intentionally not started
- PostgreSQL readiness when using Docker Compose
- whether `deep-researcher-research` can use its default `DEEP_RESEARCHER_SERVER_URL`
- the exact `export DEEP_RESEARCHER_SERVER_URL=...` command when not using the default backend URL
- whether only basic deploy validation was run or deep research completion validation also passed

Then ask:

```text
Basic deployment validation passed. Would you like me to run deep research completion validation now? This submits a `deep_researcher` job and commonly takes 7-20 minutes with substantial model/search quota. Otherwise, you can skip validation and try Deep Researcher Agent yourself.
```

Only start deep research completion validation if the user confirms.
