# Contributing to Keepygaga RAG

Keepygaga RAG is personal software in active pre-alpha development. Focused bug
reports and pull requests are welcome, but response times, compatibility, and
acceptance are not guaranteed.

## Before opening an issue

- Search existing issues first.
- Include the smallest reproducible example, expected behavior, actual behavior,
  operating system, Python version, and the output of `keepygaga-rag doctor
  --json` with paths and sensitive values redacted.
- Do not include API keys, source documents, index contents, private paths, or
  other personal data.
- Report suspected vulnerabilities privately as described in
  [SECURITY.md](SECURITY.md).

## Development setup

Use Python 3.12 or later and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/TimWongUp/keepygaga-rag.git
cd keepygaga-rag
uv sync --locked --extra dashboard
```

Runtime configuration, API keys, source documents, and generated indexes must
remain outside Git. Start from `keepygaga-rag.example.toml`; never commit a
populated `keepygaga-rag.toml` or `.env` file.

## Making a change

- Keep the change focused on one problem and preserve the local-first security
  and authorization boundaries documented in `README.md` and `PRIVACY.md`.
- Add or update the smallest relevant test when behavior changes.
- Do not make normal validation depend on real source collections or external
  embedding and reranking services.
- Run the project checks before opening a pull request:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
uv run python scripts/smoke_mcp_server.py
uv build
```

## Pull requests

Explain the user-visible result, the reason for the change, and the validation
performed. Keep unrelated refactors out of the same pull request. By submitting
a contribution, you agree that it is licensed under this repository's
[MIT License](LICENSE).
