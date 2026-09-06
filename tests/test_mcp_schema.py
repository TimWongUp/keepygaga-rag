from __future__ import annotations

import asyncio

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

import mcp_server


def _search_tool():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert [tool.name for tool in tools] == ["search"]
    return tools[0]


def _call_search(
    arguments: dict[str, object],
) -> tuple[str | None, CallToolResult]:
    async def call():
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool("search", arguments)
            return client.protocol_version, result

    return asyncio.run(call())


def _result_text(result: CallToolResult) -> str:
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


def test_public_mcp_surface_and_display_name() -> None:
    tool = _search_tool()
    assert mcp_server.mcp.name == "Keepygaga RAG"
    assert tool.input_schema["additionalProperties"] is False
    assert set(tool.input_schema["properties"]) == {
        "query",
        "top_k",
        "table_ids",
        "source_ids",
    }


def test_search_schema_keeps_bounded_contract() -> None:
    tool = _search_tool()
    schema = tool.input_schema
    description = tool.description or ""
    assert "context-backup trees" in description
    assert "hybrid FTS and vector recall" in description
    assert "online reranking" in description
    assert "Embedding provider" in description
    assert schema["properties"]["query"]["minLength"] == 1
    assert schema["properties"]["top_k"]["minimum"] == 1
    assert schema["properties"]["top_k"]["maximum"] == 20
    assert schema["properties"]["table_ids"]["maxItems"] == 20
    assert schema["properties"]["source_ids"]["maxItems"] == 20
    output = tool.output_schema
    assert output is not None
    assert output["additionalProperties"] is False
    assert output["properties"]["status"]["enum"] == ["ok", "no_results"]
    assert output["properties"]["groups"]["type"] == "array"


def test_tool_annotation_is_read_only_and_non_destructive() -> None:
    annotations = _search_tool().annotations
    assert annotations is not None
    assert annotations.read_only_hint is True
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is True
    assert annotations.open_world_hint is True


def test_runtime_rejects_unexpected_top_level_arguments() -> None:
    protocol_version, result = _call_search({"query": "test", "unexpected": True})
    assert protocol_version == "2026-07-28"
    assert result.is_error is True
    assert "Extra inputs are not permitted" in _result_text(result)


def test_search_returns_typed_structured_output(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "run_search",
        lambda **_kwargs: {
            "status": "ok",
            "query": "test",
            "groups": [
                {
                    "table": "text_chunks_v1",
                    "results": [
                        {
                            "source": "/tmp/source.md",
                            "heading_path": "Heading",
                            "text": "matched text",
                            "start_line": 4,
                            "end_line": 6,
                            "score": 0.9,
                        }
                    ],
                }
            ],
            "warnings": [],
        },
    )

    protocol_version, result = _call_search({"query": "test"})

    assert protocol_version == "2026-07-28"
    assert result.is_error is False
    assert result.structured_content == {
        "status": "ok",
        "query": "test",
        "groups": [
            {
                "table": "text_chunks_v1",
                "results": [
                    {
                        "source": "/tmp/source.md",
                        "heading_path": "Heading",
                        "text": "matched text",
                        "start_line": 4,
                        "end_line": 6,
                        "score": 0.9,
                    }
                ],
            }
        ],
        "warnings": [],
    }


def test_search_expected_failure_is_a_tool_error(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "run_search",
        lambda **_kwargs: {
            "status": "consent_required",
            "message": "renew provider consent before searching this source",
            "groups": [],
        },
    )

    _, result = _call_search({"query": "test"})

    assert result.is_error is True
    assert result.structured_content is None
    assert "renew provider consent" in _result_text(result)


def test_search_unexpected_failure_hides_internal_details(
    monkeypatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    def fail(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("secret token at /private/internal/path")

    monkeypatch.setattr(mcp_server, "run_search", fail)

    _, result = _call_search({"query": "test"})

    text = _result_text(result)
    assert result.is_error is True
    assert result.structured_content is None
    assert text == "Error executing tool search: knowledge search failed unexpectedly"
    assert "secret" not in text
    assert "/private/internal/path" not in text
    stderr = capfd.readouterr().err
    assert "secret" not in stderr
    assert "/private/internal/path" not in stderr
