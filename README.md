# Keepygaga RAG

[English](README.md) | [简体中文](README.zh-CN.md)

Keepygaga RAG is the independently runnable Knowledge/RAG subproject of
Keepygaga. It indexes authorized local Markdown and plain-text sources, combines
SQLite FTS5 and LanceDB vector recall with reciprocal-rank fusion, and applies
online reranking while preserving traceable source coordinates.

> **Project status: personal software in active development (pre-alpha).** This
> repository is published for early testing and collaboration. Interfaces, data
> formats, and behavior may change without notice. There is no support, uptime, or
> compatibility commitment, and issues or pull requests may be handled at the
> maintainer's discretion.

Its only public MCP tool is raw `search`. When the MCP host registration ID is
`keepygaga_rag`, the complete host tool name is `mcp__keepygaga_rag__search`. The MCP
Server display name is `Keepygaga RAG`.

## Requirements

- Python 3.12+
- A local directory containing Markdown or plain-text material
- Configured and user-authorized embedding and reranking APIs
- [`uv`](https://docs.astral.sh/uv/) is recommended

## Install and use

```bash
git clone https://github.com/TimWongUp/keepygaga-rag.git
cd keepygaga-rag
uv sync --extra dashboard
cp keepygaga-rag.example.toml keepygaga-rag.toml
uv run keepygaga-rag doctor
uv run python mcp_server.py
uv run keepygaga-rag dashboard
uv run keepygaga-rag indexer
```

The Dashboard listens on `127.0.0.1:8765` by default. Runtime overrides use
`KEEPYGAGA_RAG_CONFIG`, `KEEPYGAGA_RAG_DASHBOARD_PORT`, and
`KEEPYGAGA_RAG_DASHBOARD_AUTO_CLOSE`.

## Safety boundaries

- Original files remain authoritative; indexes are rebuildable derived data.
- `agents-memory/**` and `_context-backups/**` are always excluded at every depth.
- Indexed text is sent only to the authorized embedding provider. Query text and
  candidate chunks are sent only to the authorized reranker.
- Provider, model, or source-scope changes require renewed user consent.
- Hybrid results locate sources; read the returned source file before treating a
  match as authoritative.

For the exact local/external data boundary, credential handling, and provider
responsibilities, see [Privacy and data flow](PRIVACY.md).

## Contributing and security

Contributions are welcome within the project's pre-alpha scope. Before opening
an issue or pull request, read [Contributing](CONTRIBUTING.md).

Do not report vulnerabilities in a public issue. Follow the private reporting
instructions in [Security](SECURITY.md).

## License

[MIT](LICENSE)
