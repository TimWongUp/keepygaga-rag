#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from keepygaga_rag.diagnostics import run_doctor  # noqa: E402

REQUIRED_TOOLS = {"search"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 Keepygaga RAG stdio、Tool schema 与只读 Doctor。"
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def _text(result: CallToolResult) -> str:
    for block in result.content:
        if isinstance(block, TextContent):
            return block.text
    raise RuntimeError("MCP tool did not return text content")


async def run_smoke(timeout: float) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="keepygaga-rag-smoke-") as directory:
        workspace = Path(directory)
        config_path = workspace / "keepygaga-rag.toml"
        config_path.write_text(
            f"""
[knowledge]
enabled = true
store = "{workspace / ".keepygaga/knowledge-test"}"
stability_seconds = 1

[embedding.profiles.qwen3_4b_2560]
provider = "openai_compatible"
base_url = "https://example.invalid/v1"
api_key_env = "KEEPYGAGA_SMOKE_EMBED_KEY"
model = "embedding-model"
dimensions = 3

[rerank.profiles.qwen3_reranker_4b]
provider = "api"
protocol = "cohere_compatible"
base_url = "https://example.invalid/v1"
api_key_env = "KEEPYGAGA_SMOKE_RERANK_KEY"
model = "rerank-model"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["KEEPYGAGA_RAG_CONFIG"] = str(config_path)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server.py")],
            cwd=ROOT,
            env=environment,
        )
        async with asyncio.timeout(timeout):
            async with Client(parameters) as client:
                listed = await client.list_tools()
                tool_names = sorted(tool.name for tool in listed.tools)
                searched = await client.call_tool("search", {"query": "smoke"})
                invalid = await client.call_tool(
                    "search", {"query": "smoke", "unexpected": True}
                )
                server_info = client.server_info
                protocol_version = client.protocol_version
        doctor = run_doctor(config_path, project_root=ROOT)
        search_text = _text(searched)
        status = (
            "ok"
            if set(tool_names) == REQUIRED_TOOLS
            and server_info is not None
            and server_info.name == "Keepygaga RAG"
            and protocol_version == "2026-07-28"
            and searched.is_error is True
            and "not initialized" in search_text
            and invalid.is_error is True
            and doctor.get("status") in {"ok", "warning"}
            else "error"
        )
        return {
            "schema": "keepygaga-rag-mcp-smoke-v2",
            "status": status,
            "server": server_info.name if server_info is not None else None,
            "protocol_version": protocol_version,
            "tool_count": len(tool_names),
            "tools": tool_names,
            "search_status": "not_initialized"
            if "not initialized" in search_text
            else "error",
            "invalid_arguments_rejected": invalid.is_error,
            "doctor_status": doctor.get("status"),
            "external_model_called": False,
        }


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(run_smoke(args.timeout))
    except Exception as exc:
        report = {
            "schema": "keepygaga-rag-mcp-smoke-v2",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
