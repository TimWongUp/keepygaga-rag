# Use keepygaga_rag as the MCP host ID

**Status:** superseded by ADR-0004

Keepygaga Knowledge uses `keepygaga_rag` as its MCP client registration key, producing `mcp__keepygaga_rag__search`. This supersedes ADR-0002: retaining `rag` distinguishes retrieval from core memory, while the full Keepygaga product name makes cross-agent tool traces recognizable; MCP `serverInfo.name` and the raw `search` Tool remain unchanged.
