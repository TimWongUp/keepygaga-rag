# Privacy and data flow

[English](PRIVACY.md) | [简体中文](PRIVACY.zh-CN.md)

Keepygaga RAG is local-first personal software in active development. This
document describes the current technical data flow; it is not a legal privacy
policy or a promise about third-party providers.

## Data that stays local

- Source Markdown and plain-text files remain in their original locations.
- SQLite, FTS, and LanceDB indexes are stored under the configured local store.
- Runtime configuration and API keys are read from local configuration and
  environment variables. They are not intended to be committed to this
  repository.
- The Dashboard binds to `127.0.0.1` by default.
- The application does not contain a separate telemetry, analytics, or automatic
  update-reporting client.

`agents-memory/**` and `_context-backups/**` are hard-excluded from source
scanning at every depth. The default configuration also excludes hidden
descendants, `.obsidian/`, `.git/`, `.keepygaga/`, `.venv/`, and
`node_modules/`. These defaults do not reject a source root merely because the
root itself is hidden, and a custom index store is not automatically added to
the scan exclusions.

The local `.env`, `keepygaga-rag.toml`, default `.keepygaga/` store, and
`.venv/` paths are excluded from Git by default. A custom store path is not
automatically added to `.gitignore`.

## Data sent to configured providers

External transfer is disabled until the user authorizes the source scope,
provider, and model binding. Once authorized:

- source chunks in the authorized scope are sent to the configured embedding
  provider when building an index; embedding input may include the chunk text,
  document title, heading path, and filename;
- search query text is sent to the configured embedding provider;
- search query text and recalled candidate chunks, including the same metadata,
  are sent to the configured reranking provider; and
- the relevant API key is sent only as authentication to its configured provider
  endpoint.

Changing the source scope, provider, or model requires renewed consent. Provider
terms, logging, retention, and regional processing policies remain outside this
project's control.

## Data returned through MCP

The `search` tool returns matched chunk text, headings, scores, and absolute
source paths to the connected MCP host or client. Treat that host or client as a
data recipient and configure access accordingly.

## User responsibilities

Before indexing, review the selected provider, endpoint, model, and source scope.
Do not include material that you are not permitted to process, and remove or
redact sensitive content that should not be transmitted to an external provider.
Use trusted HTTPS endpoints: custom `base_url` values are not forced to use
HTTPS. If you customize the store path, exclude it from both source scanning and
Git as appropriate. Protect local configuration, environment variables, and
index files according to the sensitivity of the source material.
