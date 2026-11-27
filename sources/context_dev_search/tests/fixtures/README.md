# Recorded Context.dev response fixtures

Real `POST /v1/web/search` responses from the Context.dev Search API, replayed by
`test_context_dev_recorded_replay.py` through the HTTP transport (the exact seam the unit tests
mock) to exercise the full tool pipeline with no network and no credentials.

## Provenance

| File | Captured | Query | Request |
|---|---|---|---|
| `recorded_web_search_response.json` | 2026-11-27 | `NVIDIA CUDA Toolkit documentation` | `numResults=5`, highlights on, `maxCharacters=4000` (web search defaults) |
| `recorded_paper_search_response.json` | 2026-11-27 | `paper that replaced recurrence with self-attention for machine translation` | `numResults=20`, highlights on, `maxCharacters=1500`, no domain filter. The tool itself requests `3 × max_results` and keeps scholarly hosts; 20 results are enough to include news and blog results for the filter to drop. |

The queries match the live integration test's canned queries, so the recorded and
live layers certify the same scenarios.

## Redaction contract

Only the response fields the tools read are kept: `results[].url`, `title`,
`description`, and `highlights.highlights`. The web fixture keeps at most two passages
per result, each truncated to 800 characters; the paper fixture keeps one passage per
result and shortens descriptions and passages to at most 400 characters. Request and response headers, the API key, and
account metadata (`key_metadata`, `cache_metadata`) are never stored. The
`_request_id` field is the API's public request identifier and carries no
credentials.

## Refreshing a fixture

```python
import json, os, httpx

KEY = os.environ["CONTEXT_DEV_API_KEY"]
body = {
    "query": "NVIDIA CUDA Toolkit documentation",
    "numResults": 5,
    "highlightsOptions": {"enabled": True, "maxCharacters": 4000},
    "descriptionMaxCharacters": 4000,
}
response = httpx.post(
    "https://api.context.dev/v1/web/search",
    json=body,
    headers={"Authorization": f"Bearer {KEY}"},
    timeout=60,
).json()
payload = {
    "_description": "Recorded Context.dev POST /v1/web/search response",
    "_captured": "YYYY-MM-DD",
    "_request_id": response["request_id"],
    "_query": body["query"],
    "_config": {"max_results": 5, "max_content_length": 4000},
    "_redaction": "only results[].url/title/description/highlights.highlights kept; "
    "at most 2 passages per result, each truncated to 800 chars; no headers, auth, or account metadata",
    "results": [
        {
            "url": r["url"],
            "title": r["title"],
            "description": r["description"],
            "highlights": {
                "highlights": [p[:800] for p in (r["highlights"]["highlights"] or [])[:2]],
                "code": r["highlights"]["code"],
            },
        }
        for r in response["results"]
    ],
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
```

For the paper search fixture, use the paper query, `numResults=20`, and
`maxCharacters=1500`, with no `includeDomains` (the API leaves its indexed research
papers out of domain-filtered requests). Update `_captured` and re-run the replay tests after
refreshing.

Licensed under the Apache License, Version 2.0 (SPDX-License-Identifier: Apache-2.0);
JSON cannot carry a license header, so this note covers the fixture files.
