from __future__ import annotations

import asyncio
import re
import sqlite3
from contextlib import contextmanager
from datetime import timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from filelock import FileLock

import keepygaga_rag.dashboard.web as dashboard_web
from keepygaga_rag.config import load_config, update_knowledge_chunk_settings
from keepygaga_rag.dashboard.__main__ import _raise_open_file_limit
from keepygaga_rag.dashboard.knowledge_service import (
    KnowledgeDashboardError,
    KnowledgeDashboardService,
)
from keepygaga_rag.dashboard.presence import DashboardPresence
from keepygaga_rag.dashboard.web import (
    _format_local_datetime,
    _source_status_label,
    create_app,
)
from keepygaga_rag.knowledge.authorization import AuthorizationGuard
from keepygaga_rag.knowledge.db import (
    ChunkRecord,
    KnowledgeDB,
    read_schema_version,
)
from keepygaga_rag.knowledge.indexer import (
    DiscoveredFile,
    SourceSafetyError,
    discover_files,
)
from keepygaga_rag.knowledge.indexer_cli import build_indexer
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.vectors import LanceVectorStore


def dashboard_fixture(tmp_path: Path) -> Path:
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text("", encoding="utf-8")
    return config_path


def enable_knowledge(config_path: Path) -> None:
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[knowledge]
enabled = true
store = ".keepygaga/knowledge-test"
stability_seconds = 1

[embedding.profiles.qwen3_4b_2560]
provider = "openai_compatible"
base_url = "https://example.test/v1"
api_key_env = "TEST_EMBED_KEY"
model = "embedding-model"
dimensions = 3

[rerank.profiles.qwen3_reranker_4b]
provider = "api"
protocol = "cohere_compatible"
base_url = "https://example.test/v1"
api_key_env = "TEST_RERANK_KEY"
model = "rerank-model"
""",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def dashboard_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_EMBED_KEY", "test-embedding-key")
    monkeypatch.setenv("TEST_RERANK_KEY", "test-rerank-key")


def test_dashboard_describes_live_knowledge_backend(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        overview = client.get("/")
        settings = client.get("/settings")
    assert overview.status_code == 200
    assert settings.status_code == 404
    assert "mcp__keepygaga_rag__search" in overview.text
    assert "FTS" in overview.text
    assert "Reranker" in overview.text
    assert re.search(
        r"检查于 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        overview.text,
    )
    assert "本地知识检索" in overview.text
    assert "Keepygaga RAG · 本机服务" in overview.text
    assert "Obsidian CLI" not in overview.text


def test_dashboard_switches_between_chinese_and_english(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        chinese = client.get("/knowledge")
        switch = client.get(
            "/language/en?next=/knowledge",
            follow_redirects=False,
        )
        english = client.get("/knowledge")
        english_overview = client.get("/")
        context_redirect = client.get("/context", follow_redirects=False)
    assert '<html lang="zh-CN">' in chinese.text
    assert "MCP 已接入" in chinese.text
    assert switch.status_code == 303
    assert "keepygaga_rag_lang=en" in switch.headers["set-cookie"]
    assert '<html lang="en">' in english.text
    assert "MCP connected" in english.text
    assert "the filename becomes the title" in english.text
    assert "Markdown headings are fixed boundaries" in english.text
    assert "force a character split only when one sentence exceeds the maximum" in english.text
    assert re.search(
        r'<time datetime="[^"]+">Checked \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}</time>',
        english_overview.text,
    )
    assert "Local knowledge retrieval · traceable source" in english_overview.text
    assert "Keepygaga RAG · Local service" in english_overview.text
    assert context_redirect.status_code == 303
    assert context_redirect.headers["location"] == "/#guide"


def test_dashboard_restarts_launchd_indexer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    calls: list[str] = []
    monkeypatch.setattr(
        dashboard_web,
        "_indexer_restart_supported",
        lambda: True,
    )
    monkeypatch.setattr(dashboard_web, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        dashboard_web,
        "_restart_indexer",
        lambda: calls.append("restart"),
    )
    app = create_app(config_path=config_path, project_root=tmp_path)

    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        restarted = client.post(
            "/knowledge/indexer/restart",
            data={"csrf_token": token.group(1)},
            headers={
                "host": "127.0.0.1",
                "origin": "http://127.0.0.1",
            },
            follow_redirects=False,
        )
        repeated = client.post(
            "/knowledge/indexer/restart",
            data={"csrf_token": token.group(1)},
            headers={
                "host": "127.0.0.1",
                "origin": "http://127.0.0.1",
            },
        )
        result_page = client.get(restarted.headers["location"])

    assert 'action="/knowledge/indexer/restart"' in page.text
    assert restarted.status_code == 303
    assert restarted.headers["location"] == (
        "/knowledge?message=indexer-restarted"
    )
    assert calls == ["restart"]
    assert repeated.status_code == 429
    assert "Indexer 刚刚已请求重启" in repeated.text
    assert "Indexer 已请求重启" in result_page.text


def test_dashboard_restart_indexer_requires_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    calls: list[str] = []
    monkeypatch.setattr(dashboard_web, "_indexer_restart_supported", lambda: True)
    monkeypatch.setattr(dashboard_web, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        dashboard_web,
        "_restart_indexer",
        lambda: calls.append("restart"),
    )
    app = create_app(config_path=config_path, project_root=tmp_path)

    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        rejected = client.post(
            "/knowledge/indexer/restart",
            data={"csrf_token": token.group(1)},
        )

    assert rejected.status_code == 403
    assert calls == []


def test_dashboard_hides_restart_for_custom_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    calls: list[str] = []
    monkeypatch.setattr(dashboard_web, "_indexer_restart_supported", lambda: True)
    monkeypatch.setattr(
        dashboard_web,
        "_restart_indexer",
        lambda: calls.append("restart"),
    )
    app = create_app(config_path=config_path, project_root=tmp_path)

    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        rejected = client.post(
            "/knowledge/indexer/restart",
            data={"csrf_token": token.group(1)},
            headers={
                "host": "127.0.0.1",
                "origin": "http://127.0.0.1",
            },
        )

    assert 'action="/knowledge/indexer/restart"' not in page.text
    assert rejected.status_code == 422
    assert "当前 Dashboard 配置没有可管理的 indexer 服务" in rejected.text
    assert calls == []


def test_restart_indexer_uses_fixed_launchd_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> object:
        commands.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(dashboard_web, "_indexer_restart_supported", lambda: True)
    monkeypatch.setattr(
        dashboard_web,
        "_indexer_launchd_job_registered",
        lambda: True,
    )
    monkeypatch.setattr(dashboard_web.os, "getuid", lambda: 501)
    monkeypatch.setattr(dashboard_web.subprocess, "run", run)

    dashboard_web._restart_indexer()

    assert commands == [
        [
            "/usr/bin/osascript",
            "-e",
            (
                    'display dialog "Keepygaga RAG Dashboard 请求重启索引协调器。" '
                    'with title "重启 Keepygaga RAG Indexer" '
                'buttons {"取消", "重启"} default button "重启" '
                'cancel button "取消" with icon caution'
            ),
        ],
        [
            "/bin/launchctl",
            "kickstart",
            "-k",
            "gui/501/ai.keepygaga.knowledge.indexer",
        ]
    ]


def test_restart_cooldown_restarts_after_slow_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moments = iter([100.0, 161.0, 161.0])
    monkeypatch.setattr(
        dashboard_web.time,
        "monotonic",
        lambda: next(moments),
    )
    controller = dashboard_web.IndexerRestartController(
        lambda: None,
        cooldown_seconds=60,
    )

    controller.restart()

    with pytest.raises(dashboard_web.IndexerRestartCooldown):
        controller.restart()


def test_dashboard_get_and_post_chunk_settings(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token_match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token_match is not None
        saved = client.post(
            "/knowledge/chunk-settings",
            data={
                "csrf_token": token_match.group(1),
                "chunk_target_chars": "1800",
                "chunk_max_chars": "2600",
                "chunk_mode": "length",
                "chunk_overlap_chars": "12",
            },
            follow_redirects=False,
        )
        result_page = client.get(saved.headers["location"])

    assert page.status_code == 200
    assert "文字分块设置" in page.text
    assert 'name="chunk_target_chars"' in page.text
    assert 'name="chunk_mode"' in page.text
    assert 'name="chunk_overlap_chars"' in page.text
    assert 'value="2400"' in page.text
    assert 'value="3200"' in page.text
    assert 'value="structure" selected' in page.text
    assert re.search(
        r'name="chunk_overlap_chars"[^>]*value="0"',
        page.text,
    )
    assert "2400 字符" in page.text
    assert "3200 字符" in page.text
    assert "当前生效设置" in page.text
    assert "结构优先（推荐）" in page.text
    assert "字数优先（语义边界）" in page.text
    assert "当前设置：<strong>结构优先</strong>" in page.text
    assert "当前设置：<strong>2400 字符</strong>" in page.text
    assert "当前设置：<strong>3200 字符</strong>" in page.text
    assert "当前设置：<strong>0 字符（不重叠）</strong>" in page.text
    assert "无标题时按段落、标点处理，文件名作为标题" in page.text
    assert "Markdown（标记文本）标题是固定边界" in page.text
    assert "查看完整切块规则" in page.text
    assert "当前全局值" not in page.text
    assert "<code>chunk_mode</code>" not in page.text
    assert "already-authorized Embedding Provider" not in page.text
    assert saved.status_code == 303
    assert "message=chunk-settings-updated" in saved.headers["location"]
    assert "文字分块设置已保存" in result_page.text
    assert "不需要重启 Agent" in result_page.text
    assert 'value="length" selected' in result_page.text
    assert re.search(
        r'name="chunk_overlap_chars"[^>]*value="12"',
        result_page.text,
    )
    config = load_config(config_path)
    assert config.knowledge.chunk_target_chars == 1800
    assert config.knowledge.chunk_max_chars == 2600
    assert config.knowledge.chunk_mode == "length"
    assert config.knowledge.chunk_overlap_chars == 12


def test_dashboard_get_and_post_retrieval_settings(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token_match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token_match is not None
        saved = client.post(
            "/knowledge/retrieval-settings",
            data={
                "csrf_token": token_match.group(1),
                "vector_recall_limit": "32",
                "keyword_recall_limit": "16",
                "rerank_candidate_limit": "24",
                "max_chunks_per_source_file": "2",
            },
            follow_redirects=False,
        )
        result_page = client.get(saved.headers["location"])

    assert page.status_code == 200
    assert "检索质量设置" in page.text
    assert 'name="vector_recall_limit"' in page.text
    assert 'name="keyword_recall_limit"' in page.text
    assert 'name="rerank_candidate_limit"' in page.text
    assert 'name="max_chunks_per_source_file"' in page.text
    assert re.search(
        r'name="vector_recall_limit"[^>]*value="40"',
        page.text,
    )
    assert re.search(
        r'name="keyword_recall_limit"[^>]*value="20"',
        page.text,
    )
    assert re.search(
        r'name="rerank_candidate_limit"[^>]*value="20"',
        page.text,
    )
    assert re.search(
        r'name="max_chunks_per_source_file"[^>]*value="2"',
        page.text,
    )
    assert "向量召回数量（Top-K，候选数）" in page.text
    assert "重排候选数量（Reranker，重排模型）" in page.text
    assert saved.status_code == 303
    assert "message=retrieval-settings-updated" in saved.headers["location"]
    assert "检索设置已保存" in result_page.text
    assert "不需要重启 Agent" in result_page.text
    assert re.search(
        r'name="vector_recall_limit"[^>]*value="32"',
        result_page.text,
    )
    assert re.search(
        r'name="keyword_recall_limit"[^>]*value="16"',
        result_page.text,
    )
    assert re.search(
        r'name="rerank_candidate_limit"[^>]*value="24"',
        result_page.text,
    )
    config = load_config(config_path)
    assert config.knowledge.vector_recall_limit == 32
    assert config.knowledge.keyword_recall_limit == 16
    assert config.knowledge.max_chunks_per_source_file == 2
    assert config.rerank_profiles["qwen3_reranker_4b"].candidate_limit == 24


def test_dashboard_runs_readonly_retrieval_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    calls: list[dict[str, object]] = []

    def diagnose(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        candidate = {
            "rank": 1,
            "source": "/vault/note.md",
            "heading_path": "Knowledge",
            "excerpt": "LanceDB keeps vectors separate from SQLite.",
            "metric_label": "SCORE",
            "metric": 0.94,
            "channels": ["fts", "vector"],
        }
        return {
            "status": "ok",
            "query": "为什么使用 LanceDB？",
            "groups": [],
            "warnings": [],
            "diagnostics": {
                "settings": {
                    "vector_recall_limit": 40,
                    "keyword_recall_limit": 20,
                    "rerank_candidate_limit": 20,
                    "max_chunks_per_source_file": 2,
                },
                "lexical_query": "为什么 OR 使用 OR lancedb",
                "stages": [
                    {
                        "id": stage,
                        "count": 1,
                        "elapsed_ms": 1.2,
                        "candidates": [candidate],
                    }
                    for stage in ("keyword", "vector", "fusion", "rerank")
                ],
                "final": [candidate],
                "total_elapsed_ms": 8.4,
            },
        }

    monkeypatch.setattr(
        "keepygaga_rag.dashboard.knowledge_service.knowledge_search_diagnostics",
        diagnose,
    )
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token is not None
        result = client.post(
            "/knowledge/diagnostics",
            data={
                "csrf_token": token.group(1),
                "query": "为什么使用 LanceDB？",
            },
        )

    assert result.status_code == 200
    assert "看见一次查询如何抵达答案" in result.text
    assert "关键词召回" in result.text
    assert "向量召回" in result.text
    assert "RRF 融合" in result.text
    assert "最终返回" in result.text
    assert "/vault/note.md" in result.text
    assert "为什么使用 LanceDB？" in result.text
    assert calls == [
        {
            "query": "为什么使用 LanceDB？",
            "top_k": 5,
            "config_path": config_path.resolve(),
        }
    ]


def test_dashboard_rejects_invalid_retrieval_settings_without_writing(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    original = config_path.read_text(encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token is not None
        rejected = client.post(
            "/knowledge/retrieval-settings",
            data={
                "csrf_token": token.group(1),
                "vector_recall_limit": "101",
                "keyword_recall_limit": "20",
                "rerank_candidate_limit": "20",
                "max_chunks_per_source_file": "2",
            },
        )

    assert rejected.status_code == 422
    assert "向量召回数必须在 1 到 100 之间" in rejected.text
    assert 'value="101"' in rejected.text
    assert config_path.read_text(encoding="utf-8") == original


def test_dashboard_rejects_invalid_chunk_settings_without_writing(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    original = config_path.read_text(encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token is not None
        rejected = client.post(
            "/knowledge/chunk-settings",
            data={
                "csrf_token": token.group(1),
                "chunk_target_chars": "3000",
                "chunk_max_chars": "2000",
                "chunk_mode": "structure",
                "chunk_overlap_chars": "0",
            },
        )

    assert rejected.status_code == 422
    assert "软目标不能大于硬上限" in rejected.text
    assert 'value="3000"' in rejected.text
    assert config_path.read_text(encoding="utf-8") == original


def test_dashboard_rejects_overlap_at_or_above_hard_limit(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    original = config_path.read_text(encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token is not None
        rejected = client.post(
            "/knowledge/chunk-settings",
            data={
                "csrf_token": token.group(1),
                "chunk_target_chars": "1000",
                "chunk_max_chars": "2000",
                "chunk_mode": "length",
                "chunk_overlap_chars": "2000",
            },
        )

    assert rejected.status_code == 422
    assert "重叠字符数必须小于硬上限" in rejected.text
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("mode_value", "overlap_value", "message"),
    [
        ("invalid", "0", "切块模式无效，请选择结构优先或字数优先"),
        ("structure", "-1", "重叠字符数必须是非负整数"),
        ("structure", "abc", "重叠字符数必须是非负整数"),
        ("structure", "", "重叠字符数必须是非负整数"),
    ],
)
def test_dashboard_rejects_invalid_mode_and_overlap_values(
    tmp_path: Path,
    mode_value: str,
    overlap_value: str,
    message: str,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    original = config_path.read_text(encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token is not None
        rejected = client.post(
            "/knowledge/chunk-settings",
            data={
                "csrf_token": token.group(1),
                "chunk_target_chars": "1000",
                "chunk_max_chars": "2000",
                "chunk_mode": mode_value,
                "chunk_overlap_chars": overlap_value,
            },
        )

    assert rejected.status_code == 422
    assert message in rejected.text
    assert config_path.read_text(encoding="utf-8") == original


def test_dashboard_rolls_back_all_chunk_settings_when_rebuild_queue_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    service = KnowledgeDashboardService(config_path)

    def fail_prepare(*args: object, **kwargs: object) -> dict[str, int]:
        raise RuntimeError("queue failed")

    monkeypatch.setattr(
        "keepygaga_rag.knowledge.db.KnowledgeDB.prepare_chunk_rebuild",
        fail_prepare,
    )

    with pytest.raises(RuntimeError, match="queue failed"):
        service.update_chunk_settings(
            target_value="1200",
            max_value="1800",
            mode_value="length",
            overlap_value="12",
        )

    restored = load_config(config_path)
    assert restored.knowledge.chunk_target_chars == 2400
    assert restored.knowledge.chunk_max_chars == 3200
    assert restored.knowledge.chunk_mode == "structure"
    assert restored.knowledge.chunk_overlap_chars == 0


def test_dashboard_rolls_back_when_safe_source_enumeration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    service = KnowledgeDashboardService(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    note = source_root / "note.md"
    note.write_text("alpha body", encoding="utf-8")
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    stat = note.stat()
    record = runtime.database.get_or_create_source_file(
        source_id=source_id,
        relative_path="note.md",
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )
    runtime.database.stage_generation(
        file_id=record.id,
        generation=1,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        chunks=[
            ChunkRecord(
                chunk_id=f"{record.id}:1:0",
                source_file_id=record.id,
                generation=1,
                ordinal=0,
                text="alpha body",
                search_text="alpha body",
                heading_path="",
                title="Note",
                content_hash="content",
                embedding_input_hash="embedding",
            )
        ],
    )
    runtime.database.activate_generation(
        record.id,
        1,
        sha256(note.read_bytes()).hexdigest(),
    )
    with sqlite3.connect(runtime.database.path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_chunk_settings(
                singleton, target_chars, max_chars, chunk_mode,
                overlap_chars, updated_at
            ) VALUES (1, 2400, 3200, 'structure', 0, 'before')
            """
        )

    def fail_safe_source_enumeration(_path: str) -> bool:
        raise RuntimeError("safe source enumeration failed")

    monkeypatch.setattr(
        "keepygaga_rag.dashboard.knowledge_service.source_root_is_safe",
        fail_safe_source_enumeration,
    )

    with pytest.raises(RuntimeError, match="safe source enumeration failed"):
        service.update_chunk_settings(
            target_value="1200",
            max_value="1800",
            mode_value="length",
            overlap_value="12",
        )

    restored = load_config(config_path)
    assert restored.knowledge.chunk_target_chars == 2400
    assert restored.knowledge.chunk_max_chars == 3200
    assert restored.knowledge.chunk_mode == "structure"
    assert restored.knowledge.chunk_overlap_chars == 0
    with sqlite3.connect(runtime.database.path) as connection:
        settings = connection.execute(
            """
            SELECT target_chars, max_chars, chunk_mode, overlap_chars
            FROM knowledge_chunk_settings
            WHERE singleton = 1
            """
        ).fetchone()
    assert settings == (2400, 3200, "structure", 0)
    assert not runtime.database.list_source_files(source_id)[0].rechunk_required
    source = runtime.database.get_source(source_id)
    assert source is not None
    assert source.status == "idle"
    with sqlite3.connect(runtime.database.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sync_jobs"
        ).fetchone()[0] == 0


def test_chunk_setting_change_marks_active_files_and_queues_rebuild(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    service = KnowledgeDashboardService(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    note = source_root / "note.md"
    note.write_text("alpha body", encoding="utf-8")
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    stat = note.stat()
    record = runtime.database.get_or_create_source_file(
        source_id=source_id,
        relative_path="note.md",
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )
    chunk = ChunkRecord(
        chunk_id=f"{record.id}:1:0",
        source_file_id=record.id,
        generation=1,
        ordinal=0,
        text="alpha body",
        search_text="alpha body",
        heading_path="",
        title="Note",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    runtime.database.stage_generation(
        file_id=record.id,
        generation=1,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        chunks=[chunk],
    )
    runtime.database.activate_generation(
        record.id,
        1,
        sha256(note.read_bytes()).hexdigest(),
    )

    result = service.update_chunk_settings(
        target_value="1200",
        max_value="1800",
    )
    repeated = service.update_chunk_settings(
        target_value="1200",
        max_value="1800",
    )
    mode_changed = service.update_chunk_settings(
        target_value="1200",
        max_value="1800",
        mode_value="length",
        overlap_value="0",
    )
    with sqlite3.connect(runtime.database.path) as connection:
        mode_settings = connection.execute(
            """
            SELECT target_chars, max_chars, chunk_mode, overlap_chars
            FROM knowledge_chunk_settings
            WHERE singleton = 1
            """
        ).fetchone()
    overlap_changed = service.update_chunk_settings(
        target_value="1200",
        max_value="1800",
        mode_value="length",
        overlap_value="5",
    )

    marked = runtime.database.list_source_files(source_id)[0]
    assert marked.rechunk_required
    assert marked.active_generation == 1
    assert result["marked_files"] == 1
    assert result["queued_sources"] == 1
    assert repeated["changed"] == 0
    assert repeated["marked_files"] == 0
    assert repeated["queued_sources"] == 0
    assert mode_changed["changed"] == 1
    assert mode_changed["marked_files"] == 1
    assert mode_changed["queued_sources"] == 1
    assert overlap_changed["changed"] == 1
    assert overlap_changed["marked_files"] == 1
    assert overlap_changed["queued_sources"] == 1
    assert mode_settings == (1200, 1800, "length", 0)
    with sqlite3.connect(runtime.database.path) as connection:
        overlap_settings = connection.execute(
            """
            SELECT target_chars, max_chars, chunk_mode, overlap_chars
            FROM knowledge_chunk_settings
            WHERE singleton = 1
            """
        ).fetchone()
    assert overlap_settings == (1200, 1800, "length", 5)
    with sqlite3.connect(runtime.database.path) as connection:
        reason = connection.execute(
            "SELECT reason FROM sync_jobs WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        job_count = connection.execute(
            "SELECT COUNT(*) FROM sync_jobs WHERE source_id = ?",
            (source_id,),
        ).fetchone()[0]
    assert reason == ("chunk-settings-changed",)
    assert job_count == 1


def test_dashboard_rereads_chunk_baseline_after_exclusive_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    original_acquire = AuthorizationGuard.acquire_exclusive

    @contextmanager
    def acquire_then_change(
        guard: AuthorizationGuard,
        *,
        timeout: float = 0,
    ):
        with original_acquire(guard, timeout=timeout):
            update_knowledge_chunk_settings(
                config_path,
                target_chars=2000,
                max_chars=2800,
                chunk_mode="length",
                overlap_chars=7,
            )
            yield

    monkeypatch.setattr(
        AuthorizationGuard,
        "acquire_exclusive",
        acquire_then_change,
    )

    service = KnowledgeDashboardService(config_path)
    service.update_chunk_settings(
        target_value="2000",
        max_value="2800",
    )

    current = load_config(config_path)
    assert current.knowledge.chunk_target_chars == 2000
    assert current.knowledge.chunk_max_chars == 2800
    assert current.knowledge.chunk_mode == "length"
    assert current.knowledge.chunk_overlap_chars == 7
    runtime = KnowledgeRuntime.load(config_path)
    with sqlite3.connect(runtime.database.path) as connection:
        settings = connection.execute(
            """
            SELECT target_chars, max_chars, chunk_mode, overlap_chars
            FROM knowledge_chunk_settings
            WHERE singleton = 1
            """
        ).fetchone()
    assert settings == (2000, 2800, "length", 7)


def test_long_running_indexer_refreshes_chunk_settings(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    service = KnowledgeDashboardService(config_path)
    source_root = tmp_path / "Empty"
    source_root.mkdir()
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Empty",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    indexer = build_indexer(runtime)
    assert indexer.target_chars == 2400

    service.update_chunk_settings(
        target_value="900",
        max_value="1500",
        mode_value="length",
        overlap_value="8",
    )
    source = runtime.database.get_source(source_id)
    assert source is not None
    indexer.sync_source(source)

    assert indexer.target_chars == 900
    assert indexer.max_chars == 1500
    assert indexer.chunk_mode == "length"
    assert indexer.overlap_chars == 8


def test_dashboard_registers_previewed_knowledge_source(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    (source_root / "guide.md").write_text("# Guide\n\nalpha", encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token_match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token_match is not None
        token = token_match.group(1)
        preview = client.post(
            "/knowledge/preview",
            data={"csrf_token": token, "root": str(source_root)},
        )
        added = client.post(
            "/knowledge/sources",
            data={
                "csrf_token": token,
                "root": str(source_root),
                "display_name": "Documents",
                "consent": "on",
            },
            follow_redirects=False,
        )
    assert page.status_code == 200
    assert "INDEX OBSERVATORY" in page.text
    assert preview.status_code == 200
    assert "预计分块" in preview.text
    assert "选择生效范围" in preview.text
    assert "隐藏文件、隐藏目录" in preview.text
    assert added.status_code == 303
    snapshot = KnowledgeDashboardService(config_path).snapshot()
    sources = snapshot["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 1
    assert sources[0]["display_name"] == "Documents"
    assert sources[0]["auto_sync"] is False


def test_dashboard_preflight_scan_error_returns_422_and_keeps_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    root = tmp_path / "Documents"
    root.mkdir()

    def fail_scan(_source: object) -> object:
        raise SourceSafetyError("selected file could not be read")

    monkeypatch.setattr(
        "keepygaga_rag.knowledge.indexer.discover_files",
        fail_scan,
    )
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        response = client.post(
            "/knowledge/preview",
            data={"csrf_token": token.group(1), "root": str(root)},
        )

    assert response.status_code == 422
    assert "selected file could not be read" in response.text
    assert f'value="{root}"' in response.text


def test_dashboard_add_error_preserves_empty_scope_root_and_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_EMBED_KEY", raising=False)
    monkeypatch.delenv("TEST_RERANK_KEY", raising=False)
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    root = tmp_path / "Documents"
    root.mkdir()
    (root / "note.md").write_text("note", encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        response = client.post(
            "/knowledge/sources",
            data={
                "csrf_token": token.group(1),
                "root": str(root),
                "display_name": "Kept name",
                "scope_selection": "[]",
                "consent": "on",
            },
        )

    assert response.status_code == 422
    assert f'value="{root}"' in response.text
    assert 'value="Kept name"' in response.text
    assert "value='[]'" in response.text
    assert 'name="consent" required checked' in response.text


def test_dashboard_add_error_preserves_nonempty_scope_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_EMBED_KEY", raising=False)
    monkeypatch.delenv("TEST_RERANK_KEY", raising=False)
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    root = tmp_path / "Documents"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "note.md").write_text("note", encoding="utf-8")
    (root / "other.md").write_text("other", encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        response = client.post(
            "/knowledge/sources",
            data={
                "csrf_token": token.group(1),
                "root": str(root),
                "display_name": "Kept name",
                "scope_selection": '["nested/note.md"]',
                "consent": "on",
            },
        )

    assert response.status_code == 422
    assert "value='[\"nested/note.md\"]'" in response.text


def test_dashboard_snapshot_bounds_scope_scan_per_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    root = tmp_path / "Documents"
    root.mkdir()
    service = KnowledgeDashboardService(config_path)
    service.add_source(
        root_value=str(root),
        display_name="Documents",
        consent=True,
    )
    yielded = 0

    def many_files(_source: object):
        nonlocal yielded
        for index in range(10_000):
            yielded += 1
            yield DiscoveredFile(
                path=root / f"note-{index}.md",
                relative_path=f"note-{index}.md",
                mtime_ns=1,
                size=1,
                device=1,
                inode=index,
            )

    monkeypatch.setattr(
        "keepygaga_rag.dashboard.knowledge_service.discover_files",
        many_files,
    )
    snapshot = service.snapshot()

    assert yielded <= 1_001
    source = cast(list[dict[str, object]], snapshot["sources"])[0]
    scope = cast(dict[str, object], source["scope"])
    assert scope["truncated"] is True
    assert scope["node_limit"] == 1_000


def test_dashboard_does_not_mark_unavailable_lance_as_ready(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    root = tmp_path / "Documents"
    root.mkdir()
    service = KnowledgeDashboardService(config_path)
    service.add_source(
        root_value=str(root),
        display_name="Documents",
        consent=True,
    )

    snapshot = service.snapshot()

    assert snapshot["backend_status"] == "degraded"
    assert snapshot["vector_status"] == "unavailable"


def test_dashboard_scope_excludes_hidden_paths_and_persists_selection(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    folder = source_root / "folder"
    hidden_folder = source_root / ".private"
    folder.mkdir(parents=True)
    hidden_folder.mkdir()
    (source_root / "visible.md").write_text("visible", encoding="utf-8")
    (source_root / ".hidden.md").write_text("hidden", encoding="utf-8")
    (hidden_folder / "secret.md").write_text("secret", encoding="utf-8")
    (folder / "a.md").write_text("a", encoding="utf-8")
    (folder / "b.txt").write_text("b", encoding="utf-8")
    (folder / "image.png").write_bytes(b"png")

    service = KnowledgeDashboardService(config_path)
    preview = service.preview(str(source_root))
    scope = preview["scope"]
    assert isinstance(scope, dict)
    assert scope["total_files"] == 3
    items = scope["items"]
    assert isinstance(items, list)
    scope_paths = {
        str(item["path"]) for item in items if isinstance(item, dict)
    }
    assert ".hidden.md" not in scope_paths
    assert ".private/secret.md" not in scope_paths

    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
        selected_paths=["folder/a.md"],
    )
    runtime = KnowledgeRuntime.load(config_path)
    source = runtime.database.get_source(source_id)
    assert source is not None
    assert source.include_patterns == ("folder/a.md",)
    assert service.update_source_scope(source_id, ["folder/a.md"]) == 0
    unchanged = runtime.database.get_source(source_id)
    assert unchanged is not None
    assert unchanged.consent_identity == runtime.consent_identity

    service.update_source_scope(source_id, ["folder"])
    updated = runtime.database.get_source(source_id)
    assert updated is not None
    assert updated.include_patterns == (
        "folder/**/*.md",
        "folder/*.md",
        "folder/**/*.txt",
        "folder/*.txt",
    )
    assert updated.consent_identity == ""
    service.renew_consent(source_id, consent=True)
    renewed = runtime.database.get_source(source_id)
    assert renewed is not None
    assert renewed.consent_identity == runtime.consent_identity
    assert runtime.database.claim_next_job() is not None


def test_dashboard_nested_directory_scope_persists_and_indexes_all_depths(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    selected = source_root / "folder"
    (selected / "first" / "second").mkdir(parents=True)
    (source_root / "outside").mkdir()
    (selected / "direct.md").write_text("direct", encoding="utf-8")
    (selected / "first" / "note.md").write_text("first", encoding="utf-8")
    (selected / "first" / "second" / "deep.md").write_text(
        "deep", encoding="utf-8"
    )
    (selected / "wrong.json").write_text("wrong", encoding="utf-8")
    (source_root / "outside" / "outside.md").write_text(
        "outside", encoding="utf-8"
    )
    (source_root / "root.md").write_text("root", encoding="utf-8")

    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    assert service.update_source_scope(source_id, ["folder"]) == 0

    runtime = KnowledgeRuntime.load(config_path)
    source = runtime.database.get_source(source_id)
    assert source is not None
    assert source.include_patterns == (
        "folder/**/*.md",
        "folder/*.md",
        "folder/**/*.txt",
        "folder/*.txt",
    )
    assert {
        item.relative_path for item in discover_files(source)
    } == {
        "folder/direct.md",
        "folder/first/note.md",
        "folder/first/second/deep.md",
    }


def test_dashboard_snapshot_handles_legacy_unsafe_root_and_allows_removal(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    unsafe_root = tmp_path / ".git"
    unsafe_root.mkdir()
    runtime = KnowledgeRuntime.load(config_path)
    source = runtime.database.add_source(
        display_name="Legacy repository metadata",
        absolute_path=str(unsafe_root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=runtime.consent_identity,
    )
    service = KnowledgeDashboardService(config_path)

    snapshot = service.snapshot()

    sources = snapshot["sources"]
    assert isinstance(sources, list)
    legacy = next(item for item in sources if item["id"] == source.id)
    scope = legacy["scope"]
    assert isinstance(scope, dict)
    assert scope["available"] is False
    assert scope["items"] == []
    assert scope["selection"] == []

    service.remove_source(source.id)

    assert runtime.database.get_source(source.id) is None


def test_chinese_source_statuses_never_fall_back_to_internal_codes() -> None:
    expected = {
        "idle": "待命",
        "queued": "排队中",
        "syncing": "同步中",
        "paused": "已暂停",
        "error": "异常",
        "preview": "预检中",
    }
    assert {
        status: _source_status_label(status, "zh") for status in expected
    } == expected


def test_knowledge_page_renders_chinese_source_status(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    (source_root / "note.md").write_text("note", encoding="utf-8")
    nested = source_root / "nested"
    nested.mkdir()
    (nested / "child.md").write_text("child", encoding="utf-8")
    KnowledgeDashboardService(config_path).add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
    assert ">待命<" in page.text
    assert ">idle<" not in page.text
    assert "选择文件夹与文件" in page.text
    assert ">分块<" in page.text
    assert 'data-scope-toggle' in page.text
    assert 'class="scope-node-check"' in page.text
    assert '<button type="button" class="scope-node-main"' in page.text
    assert '<label class="scope-node scope-node-directory"' not in page.text
    assert 'aria-expanded="false"' in page.text
    assert re.search(
        r'data-scope-path="nested/child\.md"[^>]* hidden',
        page.text,
    )


def test_dashboard_updates_source_scope_through_local_route(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    (source_root / "keep.md").write_text("keep", encoding="utf-8")
    (source_root / "drop.md").write_text("drop", encoding="utf-8")
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token_match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token_match is not None
        updated = client.post(
            f"/knowledge/sources/{source_id}/scope",
            data={
                "csrf_token": token_match.group(1),
                "scope_selection": '["keep.md"]',
            },
            follow_redirects=False,
        )
    assert updated.status_code == 303
    assert updated.headers["location"] == "/knowledge?message=scope-updated"
    source = KnowledgeRuntime.load(config_path).database.get_source(source_id)
    assert source is not None
    assert source.include_patterns == ("keep.md",)
    assert source.consent_identity == ""


def test_dashboard_rejects_hard_excluded_source_root(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    excluded = tmp_path / "Agents-Memory"
    excluded.mkdir()
    service = KnowledgeDashboardService(config_path)
    with pytest.raises(KnowledgeDashboardError, match="cannot include"):
        service.add_source(
            root_value=str(excluded),
            display_name="Excluded",
            consent=True,
        )


def test_dashboard_remove_source_uses_indexer_write_lock(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    lock = FileLock(runtime.store_root / "indexer.lock")
    with lock, pytest.raises(KnowledgeDashboardError, match="索引器正在写入"):
        service.remove_source(source_id)
    assert runtime.database.get_source(source_id) is not None
    job_id = runtime.database.queue_sync(source_id, "manual")
    with pytest.raises(KnowledgeDashboardError, match="同步任务"):
        service.remove_source(source_id)
    runtime.database.finish_job(job_id)
    service.remove_source(source_id)
    assert runtime.database.get_source(source_id) is None


def test_dashboard_chooses_source_directory_with_native_picker(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    selected = tmp_path / "Selected"
    selected.mkdir()
    monkeypatch.setattr(
        "keepygaga_rag.dashboard.web._choose_directory",
        lambda: str(selected),
    )
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token_match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token_match is not None
        response = client.post(
            "/knowledge/choose-directory",
            data={"csrf_token": token_match.group(1)},
        )
    assert 'data-directory-path' in page.text
    assert 'readonly required data-directory-path' not in page.text
    assert 'required data-directory-path' in page.text
    assert "选择目录" in page.text
    assert response.json() == {"status": "ok", "path": str(selected)}


def test_dashboard_renders_source_operation_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)

    def fail_queue_sync(
        _self: KnowledgeDashboardService,
        _source_id: str,
        *,
        require_credentials: bool = False,
    ) -> None:
        del require_credentials
        raise OSError("knowledge database is unavailable")

    monkeypatch.setattr(KnowledgeDashboardService, "queue_sync", fail_queue_sync)
    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token_match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.text,
        )
        assert token_match is not None
        response = client.post(
            "/knowledge/sources/source-id/sync",
            data={"csrf_token": token_match.group(1)},
        )

    assert response.status_code == 422
    assert "knowledge database is unavailable" in response.text


def test_renew_consent_is_serialized_by_indexer_lock(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    runtime.database.set_source_consent(source_id, "")
    lock = FileLock(runtime.store_root / "indexer.lock")

    with lock, pytest.raises(KnowledgeDashboardError, match="索引器正在写入"):
        service.renew_consent(source_id, consent=True)

    current = runtime.database.get_source(source_id)
    assert current is not None
    assert current.consent_identity == ""
    assert runtime.database.claim_next_job() is None


def test_scope_update_is_serialized_by_indexer_lock(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    lock = FileLock(runtime.store_root / "indexer.lock")

    with lock, pytest.raises(KnowledgeDashboardError, match="索引器正在写入"):
        service.update_source_scope(source_id, [])

    current = runtime.database.get_source(source_id)
    assert current is not None
    assert current.include_patterns == tuple(
        runtime.config.knowledge.include_patterns
    )
    assert not current.scope_cleanup_pending


def test_toggle_source_is_serialized_by_indexer_lock(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    lock = FileLock(runtime.store_root / "indexer.lock")

    with lock, pytest.raises(KnowledgeDashboardError, match="索引器正在写入"):
        service.toggle_source(source_id, enabled=False)

    current = runtime.database.get_source(source_id)
    assert current is not None
    assert current.enabled


def test_runtime_registers_exact_historical_vector_tables(tmp_path: Path) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    first = KnowledgeRuntime.load(config_path)
    first_table = first.vectors.table_name
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'model = "embedding-model"',
            'model = "embedding-model-v2"',
        ),
        encoding="utf-8",
    )

    second = KnowledgeRuntime.load(config_path)

    assert second.vectors.table_name != first_table
    assert second.database.list_vector_tables(second.table_id) == [
        first_table,
        second.vectors.table_name,
    ]


def test_runtime_backfills_only_exact_physical_history_tables(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    first = KnowledgeRuntime.load(config_path)
    with sqlite3.connect(first.database.path) as connection:
        connection.execute("DROP TABLE knowledge_vector_tables")
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (3, 'now')"
        )
    old = LanceVectorStore(
        first.store_root / "lancedb",
        table_name="text_chunks_v1_0123456789ab",
        dimensions=3,
        embedding_space_id="oldspace",
    )
    similar = LanceVectorStore(
        first.store_root / "lancedb",
        table_name="text_chunks_v1_0123456789ab_extra",
        dimensions=3,
        embedding_space_id="similar",
    )
    old.ensure_table()
    similar.ensure_table()
    caplog.clear()

    second = KnowledgeRuntime.load(config_path)

    registered = second.database.list_vector_tables(second.table_id)
    assert "text_chunks_v1_0123456789ab" in registered
    assert "text_chunks_v1_0123456789ab_extra" not in registered
    assert "ignoring unregistered LanceDB table" in caplog.text


def test_dashboard_formats_utc_timestamps_in_local_timezone() -> None:
    value = _format_local_datetime(
        "2026-07-30T14:10:55.227140+00:00",
        timezone(timedelta(hours=8)),
    )
    assert value == "2026-07-30 22:10:55"


def test_dashboard_raises_its_open_file_soft_limit(monkeypatch) -> None:
    import resource

    applied: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (256, 100_000))
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda _kind, limits: applied.append(limits),
    )
    _raise_open_file_limit()
    assert applied == [(65_536, 100_000)]


def test_dashboard_auto_closes_after_last_tab_disconnects(
    tmp_path: Path,
) -> None:
    shutdown_calls: list[str] = []
    app = create_app(
        config_path=dashboard_fixture(tmp_path),
        project_root=tmp_path,
        auto_close=True,
        shutdown_callback=lambda: shutdown_calls.append("shutdown"),
        auto_close_grace_seconds=0.02,
    )
    with TestClient(app) as client:
        page = client.get("/")
        assert 'data-dashboard-auto-close="true"' in page.text

    async def exercise_presence() -> None:
        presence = DashboardPresence(
            shutdown=lambda: shutdown_calls.append("shutdown"),
            grace_seconds=0.02,
        )
        presence.connected(1)
        presence.connected(2)
        presence.disconnected(2)
        await asyncio.sleep(0.04)
        assert shutdown_calls == []
        presence.disconnected(1)
        await asyncio.sleep(0.04)
        await presence.close()

    asyncio.run(exercise_presence())
    assert shutdown_calls == ["shutdown"]


def test_dashboard_snapshot_is_readonly_and_exposes_indexer_failures(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    store = tmp_path / ".keepygaga/knowledge-test"
    service = KnowledgeDashboardService(config_path)

    assert service.snapshot()["backend_status"] == "not_initialized"
    assert not store.exists()

    source_root = tmp_path / "Documents"
    source_root.mkdir()
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    source_file = runtime.database.get_or_create_source_file(
        source_id=source_id,
        relative_path="broken.md",
        mtime_ns=0,
        size=1,
    )
    runtime.database.mark_file_error(source_file.id, "embedding failed")
    runtime.database.queue_sync(source_id, "manual")
    runtime.database.begin_run(source_id)

    snapshot = service.snapshot()
    queue = snapshot["queue"]
    assert isinstance(queue, dict)
    assert queue["queued"] == 1
    assert queue["running"] == 0
    recent_runs = cast(list[dict[str, object]], snapshot["recent_runs"])
    failed_files = cast(list[dict[str, object]], snapshot["failed_files"])
    sources = cast(list[dict[str, object]], snapshot["sources"])
    assert recent_runs[0]["status"] == "running"
    assert failed_files[0]["relative_path"] == "broken.md"
    assert cast(list[dict[str, object]], sources[0]["failed_files"])[0]["last_error"] == (
        "embedding failed"
    )

    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        retry = client.post(
            f"/knowledge/source-files/{source_file.id}/retry",
            data={"csrf_token": token.group(1)},
            follow_redirects=False,
        )
    assert "Indexer 实时状态" in page.text
    assert "broken.md" in page.text
    assert "embedding failed" in page.text
    assert retry.status_code == 303
    assert retry.headers["location"] == "/knowledge?message=retry-queued"


def test_dashboard_snapshot_reads_existing_store_while_coordinator_is_held(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    service = KnowledgeDashboardService(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    runtime.vectors.ensure_table()
    database_path = runtime.database.path
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    wal_path = database_path.with_name(f"{database_path.name}-wal")
    vectors_root = runtime.store_root / "lancedb"
    before_vectors = tuple(
        (
            path.relative_to(vectors_root).as_posix(),
            path.read_bytes(),
        )
        for path in sorted(vectors_root.rglob("*"))
        if path.is_file()
    )
    before = (
        database_path.read_bytes(),
        wal_path.read_bytes() if wal_path.is_file() else None,
    )
    coordinator = FileLock(runtime.store_root / "coordinator.lock")
    with coordinator:
        snapshot = service.snapshot()

    assert snapshot["sources"]
    indexer = cast(dict[str, object], snapshot["indexer"])
    assert indexer["status"] == "running"
    assert database_path.read_bytes() == before[0]
    assert (
        wal_path.read_bytes() if wal_path.is_file() else None
    ) == before[1]
    assert before_vectors == tuple(
        (
            path.relative_to(vectors_root).as_posix(),
            path.read_bytes(),
        )
        for path in sorted(vectors_root.rglob("*"))
        if path.is_file()
    )


def test_dashboard_snapshot_and_page_show_failed_jobs_and_runs(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    service = KnowledgeDashboardService(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    job_id = runtime.database.queue_sync(source_id, "manual")
    runtime.database.finish_job(job_id, error="embedding failed")
    run_id = runtime.database.begin_run(source_id)
    runtime.database.finish_run(
        run_id,
        counters={"scanned_files": 1},
        error="source scan failed",
    )

    snapshot = service.snapshot()
    queue = cast(dict[str, object], snapshot["queue"])
    jobs = cast(list[dict[str, object]], queue["jobs"])
    runs = cast(list[dict[str, object]], snapshot["recent_runs"])
    assert queue["failed"] == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["error"] == "embedding failed"
    assert runs[0]["status"] == "failed"
    assert runs[0]["error"] == "source scan failed"

    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
    assert page.status_code == 200
    assert "embedding failed" in page.text
    assert "source scan failed" in page.text


def test_dashboard_writable_current_schema_does_not_add_history_indexes(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    KnowledgeDB(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            DROP INDEX IF EXISTS sync_jobs_open_order_idx;
            DROP INDEX IF EXISTS sync_jobs_recent_order_idx;
            DROP INDEX IF EXISTS sync_runs_recent_order_idx;
            """
        )
        connection.commit()
    assert read_schema_version(database_path) == 7

    with sqlite3.connect(database_path) as connection:
        before = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    KnowledgeRuntime.load(config_path)
    with sqlite3.connect(database_path) as connection:
        after = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    assert after == before
    assert not {
        "sync_jobs_open_order_idx",
        "sync_jobs_recent_order_idx",
        "sync_runs_recent_order_idx",
    } & after


def test_dashboard_task_page_uses_bounded_recent_closed_jobs(
    tmp_path: Path,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    for index in range(12):
        job_id = runtime.database.queue_sync(source_id, f"closed-{index}")
        runtime.database.finish_job(job_id)
    runtime.database.queue_sync(source_id, "open")

    snapshot = service.snapshot()
    queue = cast(dict[str, object], snapshot["queue"])
    jobs = cast(list[dict[str, object]], queue["jobs"])
    assert queue["queued"] == 1
    assert queue["succeeded"] == 10
    assert queue["failed"] == 0
    assert queue["closed_counts_scope"] == "recent_jobs"
    assert queue["closed_counts_limit"] == 10
    assert len(jobs) == 11
    assert jobs[1]["reason"] == "closed-11"
    assert jobs[-1]["reason"] == "closed-2"

    app = create_app(config_path=config_path, project_root=tmp_path)
    with TestClient(app) as client:
        page = client.get("/knowledge")
    assert page.status_code == 200
    assert "最近 10 条记录" in page.text
    assert "closed-11" in page.text
    assert "closed-0" not in page.text


def test_dashboard_blocks_missing_provider_credentials_and_keeps_enrollment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_EMBED_KEY", raising=False)
    monkeypatch.delenv("TEST_RERANK_KEY", raising=False)
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    (source_root / "note.md").write_text("note", encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)

    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        preview = client.post(
            "/knowledge/preview",
            data={"csrf_token": token.group(1), "root": str(source_root)},
        )
        rejected = client.post(
            "/knowledge/sources",
            data={
                "csrf_token": token.group(1),
                "root": str(source_root),
                "display_name": "Documents kept",
                "scope_selection": '["note.md"]',
                "consent": "on",
            },
        )

    assert preview.status_code == 200
    assert rejected.status_code == 422
    assert "TEST_EMBED_KEY" in rejected.text
    assert "TEST_RERANK_KEY" in rejected.text
    assert "选择生效范围" in rejected.text
    assert 'value="Documents kept"' in rejected.text


def test_dashboard_sync_retry_and_renew_require_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    service = KnowledgeDashboardService(config_path)
    source_id = service.add_source(
        root_value=str(source_root),
        display_name="Documents",
        consent=True,
    )
    runtime = KnowledgeRuntime.load(config_path)
    source_file = runtime.database.get_or_create_source_file(
        source_id=source_id,
        relative_path="broken.md",
        mtime_ns=0,
        size=1,
    )
    runtime.database.mark_file_error(source_file.id, "embedding failed")
    monkeypatch.delenv("TEST_EMBED_KEY", raising=False)
    monkeypatch.delenv("TEST_RERANK_KEY", raising=False)

    with pytest.raises(KnowledgeDashboardError, match="缺少 Provider 凭据"):
        service.queue_sync(source_id)
    with pytest.raises(KnowledgeDashboardError, match="缺少 Provider 凭据"):
        service.retry_failed_file(source_file.id)
    with pytest.raises(KnowledgeDashboardError, match="缺少 Provider 凭据"):
        service.renew_consent(source_id, consent=True)

    with runtime.database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sync_jobs WHERE source_id = ?",
            (source_id,),
        ).fetchone()[0] == 0


def test_dashboard_allows_manual_linux_directory_without_zenity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard_web.sys, "platform", "linux")
    monkeypatch.setattr(dashboard_web.shutil, "which", lambda _name: None)
    config_path = dashboard_fixture(tmp_path)
    enable_knowledge(config_path)
    source_root = tmp_path / "Documents"
    source_root.mkdir()
    (source_root / "note.md").write_text("note", encoding="utf-8")
    app = create_app(config_path=config_path, project_root=tmp_path)

    with TestClient(app) as client:
        page = client.get("/knowledge")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert token is not None
        preview = client.post(
            "/knowledge/preview",
            data={"csrf_token": token.group(1), "root": str(source_root)},
        )

    assert "readonly required data-directory-path" not in page.text
    assert preview.status_code == 200
    assert "预计分块" in preview.text
