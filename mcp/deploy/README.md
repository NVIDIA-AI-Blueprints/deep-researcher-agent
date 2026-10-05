# MCP deployment assets

`init-mcp-db.sql` initializes the Postgres-backed submit/poll ledger used by the public MCP server. The script is
idempotent and is mounted by `deploy/compose/docker-compose.mcp.yaml` for first-time database initialization.

The `aiq_maas_mcp` value in `mcp_schema_migrations` is deliberately retained as a database migration identifier.
It lets an existing deployment upgrade in place; it is opaque storage metadata, not a runtime dependency, and is
not sent to clients.

Application startup also applies the same schema under a Postgres advisory lock. The SQL file remains necessary
so a fresh Compose deployment has the expected schema before the MCP process starts.
