from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import Icon, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from keepygaga_rag.config import DEFAULT_CONFIG_PATH
from keepygaga_rag.knowledge.api import knowledge_search as run_search

CONFIG_PATH = (
    Path(
        os.environ.get(
            "KEEPYGAGA_RAG_CONFIG",
            str(DEFAULT_CONFIG_PATH),
        )
    )
    .expanduser()
    .resolve()
)


class SearchResultItem(BaseModel):
    """One source chunk returned by knowledge search."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Absolute path to the source file")
    heading_path: str = Field(description="Heading hierarchy containing the chunk")
    text: str = Field(description="Matched source text")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="1-based first source line at indexing time; null for older chunks",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Inclusive last source line at indexing time; null for older chunks",
    )
    score: float = Field(
        description="Reranker score, or RRF score when reranking degraded"
    )


class SearchResultGroup(BaseModel):
    """Search results from one retrieval table."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(description="Retrieval table identifier")
    results: list[SearchResultItem]


class SearchResult(BaseModel):
    """Successful knowledge search response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "no_results"]
    query: str
    groups: list[SearchResultGroup]
    warnings: list[str] = Field(default_factory=list)


class StrictMCPServer(MCPServer):
    """MCPServer with closed top-level argument models."""

    def add_tool(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: ToolAnnotations | None = None,
        icons: list[Icon] | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
    ) -> None:
        super().add_tool(
            fn,
            name=name,
            title=title,
            description=description,
            annotations=annotations,
            icons=icons,
            meta=meta,
            structured_output=structured_output,
        )
        tool_name = name or fn.__name__
        tool = self._tool_manager.get_tool(tool_name)
        if tool is None:  # pragma: no cover
            raise RuntimeError(f"tool registration failed: {tool_name}")
        arguments = tool.fn_metadata.arg_model
        arguments.model_config["extra"] = "forbid"
        arguments.model_rebuild(force=True)
        tool.parameters = arguments.model_json_schema(by_alias=True)


mcp = StrictMCPServer("Keepygaga RAG")


@mcp.tool(
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    )
)
def search(
    query: Annotated[
        str,
        Field(
            min_length=1,
            description="Natural-language or exact-text knowledge query",
        ),
    ],
    top_k: Annotated[
        int,
        Field(ge=1, le=20, description="Maximum results to return"),
    ] = 5,
    table_ids: Annotated[
        tuple[str, ...],
        Field(max_length=20, description="Optional retrieval-table filters"),
    ] = (),
    source_ids: Annotated[
        tuple[str, ...],
        Field(max_length=20, description="Optional source filters"),
    ] = (),
) -> SearchResult:
    """
    Search ordinary local knowledge with hybrid FTS and vector recall, reciprocal
    rank fusion, and online reranking. Results are grouped by text table and include
    source paths, headings, chunk text, scores, and indexed source line ranges.

    This tool never searches core Agent memory or context-backup trees, even when
    either appears below a configured source root. Indexing text is sent to the
    configured Embedding provider; query text and candidate chunks are sent to the
    configured Reranker provider only for sources with current user consent.

    Args:
        query: Natural-language or exact-text knowledge query.
        top_k: Results to return, from 1 through 20.
        table_ids: Optional text-table filters, at most 20.
        source_ids: Optional source filters, at most 20.

    Results locate candidate source material. Read the returned source file before
    treating a match as authoritative. Line ranges describe the indexed version;
    subsequent source edits can shift them.
    """
    try:
        result = run_search(
            query=query,
            top_k=top_k,
            table_ids=table_ids,
            source_ids=source_ids,
            config_path=CONFIG_PATH,
        )
        status = result.get("status")
        if status not in {"ok", "no_results"}:
            if status == "unavailable":
                raise ToolError(
                    "knowledge search is temporarily unavailable; retry after checking "
                    "the authorization guard"
                )
            message = result.get("message")
            if isinstance(message, str) and message:
                raise ToolError(message)
            raise RuntimeError("knowledge search returned an invalid failure response")
        return SearchResult.model_validate(result)
    except ToolError:
        raise
    except Exception:
        raise ToolError("knowledge search failed unexpectedly") from None


if __name__ == "__main__":
    mcp.run()
