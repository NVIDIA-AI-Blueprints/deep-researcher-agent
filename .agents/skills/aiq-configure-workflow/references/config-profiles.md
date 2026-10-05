# Pick a starting profile

There is **no single all-features profile**. Use the decision tree below, then copy
the file from `configs/`. For the full nine-profile table, see
`docs/source/customization/configuration-reference.md` ("Provided Config Files").

1. **Run mode**
   - CLI only (`nat run`, `start_cli.sh`) → `config_cli_default.yml` (no `front_end`)
   - Direct DS Agent development → `config_cli_data_science.yml` (no router or `front_end`)
   - Web UI / REST / async jobs / `aiq-research` → `config_web_*` or frontier/domain/skills profile (`front_end._type: aiq_api`)

2. **Knowledge backend**
   - None → `config_cli_default.yml`
   - LlamaIndex → `config_web_default_llamaindex.yml`
   - Foundational RAG → `config_web_frag.yml` (`RAG_SERVER_URL`, `RAG_INGEST_URL`)
   - OpenSearch → `config_web_opensearch.yml`

3. **Model family**
   - Nemotron → most profiles
   - GPT-5.2 orchestration/planning/writing → `config_frontier_models.yml` (`OPENAI_API_KEY`)
   - Default split → Nemotron 3.5 Lightning for intent/shallow; Ultra for clarification/deep research, with a larger writer budget

4. **Optional features** — copy blocks from:
   - Guardrails → `config_web_default_guardrails.yml`
   - MCP OAuth source → `config_web_frag_mcp_auth.yml`
   - Domain routing + skills + Modal → `config_domain_routing_and_skills.yml`
   - OpenShell sandbox → `config_openshell.yml`
