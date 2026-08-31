from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import cast

import pytest
from filelock import FileLock

from keepygaga_rag.config import load_config
from keepygaga_rag.diagnostics import run_doctor
from keepygaga_rag.knowledge.db import SCHEMA_VERSION, ChunkRecord, KnowledgeDB
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.vectors import LanceVectorStore


def _config(tmp_path: Path, *, enabled: bool) -> Path:
    path = tmp_path / "keepygaga-rag.toml"
    path.write_text(
        f"""
[knowledge]
enabled = {str(enabled).lower()}
store = ".keepygaga/knowledge-test"

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
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_doctor_treats_disabled_knowledge_as_error(tmp_path: Path) -> None:
    result = run_doctor(_config(tmp_path, enabled=False), project_root=tmp_path)
    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    tools = cast(list[str], result["tools"])
    assert result["schema"] == "keepygaga-rag-doctor-v1"
    assert tools == ["search"]
    assert knowledge["status"] == "error"
    assert "mcp_cutover" not in details


def test_doctor_reports_missing_knowledge_database(tmp_path: Path) -> None:
    result = run_doctor(_config(tmp_path, enabled=True), project_root=tmp_path)
    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    assert knowledge["status"] == "warning"
    assert "knowledge database does not exist" in str(knowledge["message"])
    assert details["backend"] == "sqlite_fts5_lancedb"


@pytest.mark.parametrize(
    "error",
    [
        "ProviderActionRequiredError: embedding provider request failed with HTTP 402",
        "HTTPStatusError: 401 Unauthorized",
        "HTTPStatusError: 403 Forbidden",
    ],
)
def test_doctor_identifies_provider_action_required_file_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    config_path = _config(tmp_path, enabled=True)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    database = KnowledgeDB(database_path)
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="test-consent",
    )
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=1,
        size=1,
    )
    database.mark_file_error(source_file.id, error)
    monkeypatch.setenv("TEST_EMBED_KEY", "secret")
    monkeypatch.setenv("TEST_RERANK_KEY", "secret")

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    assert details["provider_action_required_file_errors"] == 1
    assert "account balance" in str(knowledge["message"])


def test_doctor_reports_knowledge_schema_upgrade_required(tmp_path: Path) -> None:
    config_path = _config(tmp_path, enabled=True)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    KnowledgeDB(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            (SCHEMA_VERSION - 1,),
        )

    result = run_doctor(config_path, project_root=tmp_path)

    assert result["status"] == "warning"
    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    assert knowledge["status"] == "warning"
    assert "upgrade required" in str(knowledge["message"])
    assert details["schema_version"] == SCHEMA_VERSION - 1
    assert details["expected_schema_version"] == SCHEMA_VERSION
    assert details["migration_required"] is True


def test_doctor_rejects_newer_knowledge_schema(tmp_path: Path) -> None:
    config_path = _config(tmp_path, enabled=True)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    KnowledgeDB(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            (SCHEMA_VERSION + 1,),
        )

    result = run_doctor(config_path, project_root=tmp_path)

    assert result["status"] == "error"
    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    assert knowledge["status"] == "error"
    assert "newer than this code" in str(knowledge["message"])
    assert details["schema_version"] == SCHEMA_VERSION + 1
    assert details["migration_required"] is False


def test_doctor_marks_corrupt_knowledge_database_as_error(tmp_path: Path) -> None:
    config_path = _config(tmp_path, enabled=True)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    database_path.parent.mkdir(parents=True)
    database_path.write_bytes(b"this is not a sqlite database")

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    assert knowledge["status"] == "error"
    assert result["status"] == "error"
    assert details["database_corrupt"] is True
    assert "database" in str(details["database_error"])


def test_doctor_reports_coordinator_and_stuck_tasks(tmp_path: Path) -> None:
    config_path = _config(tmp_path, enabled=True)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    database = KnowledgeDB(database_path)
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="test-consent",
    )
    job_id = database.queue_sync(source.id, "test")
    run_id = database.begin_run(source.id)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE sync_jobs
            SET status = 'running', started_at = '2000-01-01T00:00:00+00:00'
            WHERE id = ?
            """,
            (job_id,),
        )
        connection.execute(
            """
            UPDATE sync_runs
            SET started_at = '2000-01-01T00:00:00+00:00'
            WHERE id = ?
            """,
            (run_id,),
        )

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    tasks = cast(dict[str, object], details["tasks"])
    assert knowledge["status"] == "error"
    assert details["indexer_status"] == "stopped"
    assert details["indexer_running"] is False
    assert details["open_task_count"] == 2
    assert details["stuck_task_count"] == 2
    assert len(cast(list[object], tasks["stuck_jobs"])) == 1
    assert len(cast(list[object], tasks["stuck_runs"])) == 1
    assert "stuck" in str(knowledge["message"])


def test_doctor_detects_held_coordinator_lock(tmp_path: Path) -> None:
    config_path = _config(tmp_path, enabled=True)
    store = tmp_path / ".keepygaga/knowledge-test"
    KnowledgeDB(store / "knowledge.sqlite3")
    coordinator = FileLock(store / "coordinator.lock")

    with coordinator:
        result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    coordinator_details = cast(dict[str, object], details["coordinator"])
    assert coordinator_details["status"] == "running"
    assert coordinator_details["lock_held"] is True
    assert details["indexer_running"] is True


def test_doctor_reports_missing_active_lance_vectors(tmp_path: Path) -> None:
    config_path = _config(tmp_path, enabled=True)
    database_path = tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    database = KnowledgeDB(database_path)
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="test-consent",
    )
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=1,
        size=10,
    )
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=1,
        size=10,
        chunks=[
            ChunkRecord(
                chunk_id="note:1:0",
                source_file_id=source_file.id,
                generation=1,
                ordinal=0,
                text="active text",
                search_text="active text",
                heading_path="",
                title="Note",
                content_hash="content",
                embedding_input_hash="embedding",
            )
        ],
    )
    database.activate_generation(source_file.id, 1, "digest")

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    vector_integrity = cast(dict[str, object], details["vector_integrity"])
    assert knowledge["status"] == "error"
    assert vector_integrity["status"] == "error"
    assert "vector" in str(vector_integrity["reason"]).casefold()


def test_doctor_reports_invalid_active_lance_vector_metadata(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path, enabled=True)
    runtime = KnowledgeRuntime.from_config(
        load_config(config_path),
        config_path,
    )
    root = tmp_path / "source"
    root.mkdir()
    source = runtime.database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=runtime.consent_identity,
    )
    source_file = runtime.database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=1,
        size=10,
    )
    chunk = ChunkRecord(
        chunk_id="note:1:0",
        source_file_id=source_file.id,
        generation=1,
        ordinal=0,
        text="active text",
        search_text="active text",
        heading_path="",
        title="Note",
        content_hash="content",
        embedding_input_hash="expected-hash",
    )
    runtime.database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=1,
        size=10,
        chunks=[chunk],
    )
    runtime.database.activate_generation(source_file.id, 1, "digest")
    embedding_space_id = hashlib.sha256(
        runtime.embedding.identity.encode("utf-8")
    ).hexdigest()
    vectors = LanceVectorStore(
        runtime.store_root / "lancedb",
        table_name=runtime.vectors.table_name,
        dimensions=runtime.embedding.dimensions,
        embedding_space_id=embedding_space_id,
    )
    vectors.add(
        [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": source.id,
                "source_file_id": source_file.id,
                "embedding_input_hash": "wrong-hash",
                "embedding_space_id": embedding_space_id,
                "generation": 1,
                "vector": [1.0, 0.0, 0.0],
            }
        ]
    )

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    vector_integrity = cast(dict[str, object], details["vector_integrity"])
    assert vector_integrity["status"] == "error"
    assert vector_integrity["invalid_vector_count"] == 1
    assert vector_integrity["invalid_chunk_ids"] == [chunk.chunk_id]


def test_doctor_excludes_stale_source_chunks_from_vector_check(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path, enabled=True)
    database = KnowledgeDB(
        tmp_path / ".keepygaga/knowledge-test/knowledge.sqlite3"
    )
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Stale",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="stale-consent",
    )
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=1,
        size=1,
    )
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=1,
        size=1,
        chunks=[
            ChunkRecord(
                chunk_id="stale:1:0",
                source_file_id=source_file.id,
                generation=1,
                ordinal=0,
                text="stale",
                search_text="stale",
                heading_path="",
                title="Note",
                content_hash="content",
                embedding_input_hash="hash",
            )
        ],
    )
    database.activate_generation(source_file.id, 1, "digest")

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    vector_integrity = cast(dict[str, object], details["vector_integrity"])
    assert vector_integrity["expected_active_vectors"] == 0
    assert vector_integrity["missing_chunk_ids"] == []


def test_doctor_limits_vector_failure_samples_to_fifty(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path, enabled=True)
    store = tmp_path / ".keepygaga/knowledge-test"
    database = KnowledgeDB(store / "knowledge.sqlite3")
    runtime = KnowledgeRuntime.from_config(
        load_config(config_path),
        config_path,
    )
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=runtime.consent_identity,
    )
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=1,
        size=1,
    )
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=1,
        size=1,
        chunks=[
            ChunkRecord(
                chunk_id=f"missing:{index}",
                source_file_id=source_file.id,
                generation=1,
                ordinal=index,
                text="missing",
                search_text="missing",
                heading_path="",
                title="Note",
                content_hash=f"content-{index}",
                embedding_input_hash=f"hash-{index}",
            )
            for index in range(60)
        ],
    )
    database.activate_generation(source_file.id, 1, "digest")
    embedding_space_id = hashlib.sha256(
        runtime.embedding.identity.encode("utf-8")
    ).hexdigest()
    empty_vectors = LanceVectorStore(
        store / "lancedb",
        table_name=runtime.vectors.table_name,
        dimensions=runtime.embedding.dimensions,
        embedding_space_id=embedding_space_id,
    )
    empty_vectors.ensure_table()

    result = run_doctor(config_path, project_root=tmp_path)

    checks = cast(list[dict[str, object]], result["checks"])
    knowledge = next(item for item in checks if item["id"] == "knowledge")
    details = cast(dict[str, object], knowledge["details"])
    vector_integrity = cast(dict[str, object], details["vector_integrity"])
    assert vector_integrity["missing_vector_count"] == 60
    assert len(cast(list[str], vector_integrity["missing_chunk_ids"])) == 50
