# Split core memory and knowledge retrieval into separate repositories

The open-source-ready core-memory contract remains in `keepygaga`, while Knowledge/RAG evolves in the independent `keepygaga-knowledge` repository. Separate dependencies, release readiness and operational lifecycles outweigh the convenience of one package, so integration happens through parallel MCP Server registrations rather than Python imports.
