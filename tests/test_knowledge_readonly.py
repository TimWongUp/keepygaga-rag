from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from filelock import FileLock

import keepygaga_rag.knowledge.authorization as authorization_module
from keepygaga_rag.config import load_config
from keepygaga_rag.knowledge.api import knowledge_search
from keepygaga_rag.knowledge.authorization import AuthorizationGuard
from keepygaga_rag.knowledge.chunking import fts_query
from keepygaga_rag.knowledge.db import (
    SCHEMA_VERSION,
    ChunkRecord,
    KnowledgeDB,
    read_schema_version,
)
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.searcher import KnowledgeSearcher
from keepygaga_rag.knowledge.vectors import LanceVectorStore


def _knowledge_config(store: str) -> str:
    return f"""
[knowledge]
enabled = true
store = "{store}"

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


class _ReadonlyEmbedding:
    identity = "test-embedding"
    dimensions = 1

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class _ReadonlyReranker:
    identity = "test-reranker"

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        return [(0, 1.0)] if documents else []


def _sqlite_snapshot(path: Path) -> tuple[bytes, bytes | None]:
    def contents(candidate: Path) -> bytes | None:
        return candidate.read_bytes() if candidate.is_file() else None

    return (
        path.read_bytes(),
        contents(path.with_name(f"{path.name}-wal")),
    )


def test_runtime_blocks_schema_upgrade_while_indexer_is_running(
    tmp_path: Path,
) -> None:
    store = tmp_path / "knowledge"
    database_path = store / "knowledge.sqlite3"
    KnowledgeDB(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            (SCHEMA_VERSION - 1,),
        )
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(
        _knowledge_config(str(store)) + "\n",
        encoding="utf-8",
    )
    before = _sqlite_snapshot(database_path)

    coordinator = FileLock(store / "coordinator.lock")
    with coordinator, pytest.raises(
        RuntimeError, match="stop the running keepygaga-rag indexer"
    ):
        KnowledgeRuntime.load(config_path)

    assert read_schema_version(database_path) == SCHEMA_VERSION - 1
    assert _sqlite_snapshot(database_path) == before
    with coordinator:
        KnowledgeRuntime.from_config(
            load_config(config_path),
            config_path,
            coordinator_lock=coordinator,
        )
    assert read_schema_version(database_path) == SCHEMA_VERSION
    with coordinator.acquire(timeout=0):
        pass


def test_current_schema_runtime_skips_initialization_while_coordinator_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "knowledge"
    database_path = store / "knowledge.sqlite3"
    KnowledgeDB(database_path)
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(
        _knowledge_config(str(store)) + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(database_path) as connection:
        before_schema = tuple(
            connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE type IN ('index', 'table', 'trigger', 'view')
                ORDER BY type, name
                """
            ).fetchall()
        )

    def fail_initialize(_database: KnowledgeDB) -> None:
        raise AssertionError("current schema must not initialize")

    monkeypatch.setattr(KnowledgeDB, "_initialize", fail_initialize)
    coordinator = FileLock(store / "coordinator.lock")
    with coordinator:
        runtime = KnowledgeRuntime.from_config(
            load_config(config_path),
            config_path,
            coordinator_lock=coordinator,
        )

    assert not runtime.database.readonly
    assert read_schema_version(database_path) == SCHEMA_VERSION
    with sqlite3.connect(database_path) as connection:
        after_schema = tuple(
            connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE type IN ('index', 'table', 'trigger', 'view')
                ORDER BY type, name
                """
            ).fetchall()
        )
    assert after_schema == before_schema


def test_readonly_database_rejects_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.sqlite3"
    KnowledgeDB(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            (SCHEMA_VERSION + 1,),
        )

    with pytest.raises(RuntimeError, match="supports up to"):
        KnowledgeDB(path, readonly=True)


def test_knowledge_search_reads_older_schema_without_migrating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_EMBED_KEY", raising=False)
    monkeypatch.delenv("TEST_RERANK_KEY", raising=False)
    store = tmp_path / "knowledge"
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(_knowledge_config(str(store)), encoding="utf-8")
    runtime = KnowledgeRuntime.load(config_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = runtime.database.add_source(
        display_name="Source",
        absolute_path=str(source_root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=runtime.consent_identity,
    )
    source_file = runtime.database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=0,
        size=10,
    )
    runtime.database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=0,
        size=10,
        chunks=[
            ChunkRecord(
                chunk_id="note:1:0",
                source_file_id=source_file.id,
                generation=1,
                ordinal=0,
                text="alpha legacy body",
                search_text="alpha legacy body",
                heading_path="",
                title="Legacy Note",
                content_hash="content",
                embedding_input_hash="embedding",
            )
        ],
    )
    runtime.database.activate_generation(source_file.id, 1, "digest")
    with sqlite3.connect(runtime.database.path) as connection:
        connection.execute("DROP TABLE chunks_fts")
        connection.execute("ALTER TABLE chunks DROP COLUMN filename")
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                search_text,
                tokenize = 'unicode61'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunks_fts(chunk_id, search_text)
            SELECT chunk_id, search_text FROM chunks
            """
        )
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            (SCHEMA_VERSION - 1,),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = _sqlite_snapshot(runtime.database.path)

    result = knowledge_search(query="alpha", config_path=config_path)

    assert result["status"] == "ok"
    groups = cast(list[dict[str, object]], result["groups"])
    assert cast(list[dict[str, object]], groups[0]["results"])[0]["text"] == (
        "alpha legacy body"
    )
    assert read_schema_version(runtime.database.path) == SCHEMA_VERSION - 1
    assert _sqlite_snapshot(runtime.database.path) == before


def test_writable_runtime_rejects_newer_schema_without_modifying_database(
    tmp_path: Path,
) -> None:
    store = tmp_path / "knowledge"
    path = store / "knowledge.sqlite3"
    KnowledgeDB(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE knowledge_chunk_settings")
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
            (SCHEMA_VERSION + 1,),
        )
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(_knowledge_config(str(store)), encoding="utf-8")
    before = _sqlite_snapshot(path)

    with pytest.raises(RuntimeError, match="supports up to"):
        KnowledgeRuntime.load(config_path)

    assert _sqlite_snapshot(path) == before
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "knowledge_chunk_settings" not in tables


def test_writable_runtime_rejects_existing_database_without_schema(
    tmp_path: Path,
) -> None:
    store = tmp_path / "knowledge"
    store.mkdir()
    path = store / "knowledge.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE user_data(value TEXT NOT NULL)")
        connection.execute("INSERT INTO user_data(value) VALUES ('preserve-me')")
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(_knowledge_config(str(store)), encoding="utf-8")
    before = _sqlite_snapshot(path)

    with pytest.raises(RuntimeError, match="schema version is unavailable"):
        KnowledgeRuntime.load(config_path)

    assert _sqlite_snapshot(path) == before


def test_readonly_guard_uses_existing_filelock_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "indexer.lock"
    guard = AuthorizationGuard(path)
    guard.ensure_writable()
    before = path.read_bytes()
    locked_descriptors: list[int] = []

    def lock_descriptor(descriptor: int, *, blocking: bool) -> bool:
        assert not blocking
        locked_descriptors.append(descriptor)
        return True

    def unlock_descriptor(descriptor: int) -> None:
        assert descriptor == locked_descriptors[-1]

    monkeypatch.setattr(authorization_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(authorization_module, "lock_descriptor", lock_descriptor)
    monkeypatch.setattr(authorization_module, "unlock_descriptor", unlock_descriptor)

    with guard.acquire_exclusive():
        pass
    assert path.is_file()

    with guard.acquire_readonly():
        assert path.is_file()

    assert path.read_bytes() == before
    assert len(locked_descriptors) == 1
    with guard.acquire_readonly():
        pass
    assert path.is_file()
    assert len(locked_descriptors) == 2


def test_exclusive_guard_initializes_missing_lock_file(tmp_path: Path) -> None:
    path = tmp_path / "indexer.lock"
    guard = AuthorizationGuard(path)

    with guard.acquire_exclusive():
        assert path.is_file()


def test_readonly_guard_does_not_initialize_missing_lock_file(tmp_path: Path) -> None:
    path = tmp_path / "indexer.lock"
    guard = AuthorizationGuard(path)

    with pytest.raises(
        authorization_module.AuthorizationGuardUnavailable,
        match="has not been initialized",
    ), guard.acquire_readonly():
        pass

    assert not path.exists()


def test_readonly_search_reports_unavailable_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.sqlite3"
    KnowledgeDB(path)
    readonly = KnowledgeDB(path, readonly=True)
    guard_path = tmp_path / "indexer.lock"
    result = KnowledgeSearcher(
        database=readonly,
        vectors=cast(Any, object()),
        embedding=_ReadonlyEmbedding(),
        reranker=_ReadonlyReranker(),
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
        authorization_guard=AuthorizationGuard(guard_path),
    ).search(query="alpha")

    assert result["status"] == "unavailable"
    assert "not been initialized" in str(result["message"])
    assert not guard_path.exists()


def test_readonly_search_does_not_initialize_missing_storage(tmp_path: Path) -> None:
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(_knowledge_config(".keepygaga/knowledge"), encoding="utf-8")

    result = knowledge_search(query="alpha", config_path=config_path)

    assert result == {
        "status": "not_initialized",
        "message": "knowledge index is not initialized",
        "groups": [],
    }
    assert not (tmp_path / ".keepygaga").exists()


def test_readonly_vector_store_does_not_create_missing_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lancedb"

    try:
        LanceVectorStore(
            root,
            table_name="text_chunks_v1",
            dimensions=3,
            embedding_space_id="space",
            readonly=True,
        )
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected a missing read-only vector store to fail")

    assert not root.exists()


def test_schema_v1_migrates_scope_cleanup_and_vector_queue(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES (1, 'now');
            CREATE TABLE sources (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                absolute_path TEXT NOT NULL UNIQUE,
                enabled INTEGER NOT NULL DEFAULT 1,
                auto_sync INTEGER NOT NULL DEFAULT 1,
                include_patterns_json TEXT NOT NULL,
                exclude_patterns_json TEXT NOT NULL,
                consent_identity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'idle',
                last_scan_at TEXT NOT NULL DEFAULT '',
                last_success_at TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    KnowledgeDB(path)

    with sqlite3.connect(path) as connection:
        source_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sources)")
        }
        vector_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(vector_deletions)")
        }
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 7
    assert "scope_cleanup_pending" in source_columns
    assert {"source_id", "scope_cleanup"} <= vector_columns


def test_readonly_v1_source_is_compatible_without_migrating(
    tmp_path: Path,
) -> None:
    store = tmp_path / "knowledge"
    path = store / "knowledge.sqlite3"
    (tmp_path / "legacy").mkdir()
    store.mkdir()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            f"""
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES (1, 'now');
            CREATE TABLE sources (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                absolute_path TEXT NOT NULL UNIQUE,
                enabled INTEGER NOT NULL DEFAULT 1,
                auto_sync INTEGER NOT NULL DEFAULT 1,
                include_patterns_json TEXT NOT NULL,
                exclude_patterns_json TEXT NOT NULL,
                consent_identity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'idle',
                last_scan_at TEXT NOT NULL DEFAULT '',
                last_success_at TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO sources(
                id, display_name, absolute_path, include_patterns_json,
                exclude_patterns_json, consent_identity, created_at, updated_at
            ) VALUES (
                'legacy-source', 'Legacy', '{tmp_path / "legacy"}',
                '["**/*.md"]', '[]', 'legacy-consent', 'now', 'now'
            );
            """
        )
    (store / "lancedb").mkdir()
    AuthorizationGuard(store / "indexer.lock").ensure_writable()
    config_path = tmp_path / "keepygaga-rag.toml"
    config_path.write_text(_knowledge_config(str(store)), encoding="utf-8")
    before = sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns

    readonly = KnowledgeDB(path, readonly=True)
    listed = readonly.list_sources()
    fetched = readonly.get_source("legacy-source")
    result = knowledge_search(query="alpha", config_path=config_path)

    assert len(listed) == 1
    assert not listed[0].scope_cleanup_pending
    assert fetched is not None
    assert not fetched.scope_cleanup_pending
    assert result["status"] == "consent_required"
    assert result["source_ids"] == ["legacy-source"]
    assert sha256(path.read_bytes()).hexdigest() == before
    assert path.stat().st_mtime_ns == before_mtime
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sources)")
        }
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert "scope_cleanup_pending" not in columns
    assert version == 1


def test_schema_v3_migrates_exact_vector_table_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES (3, 'now');
            CREATE TABLE knowledge_tables (
                table_id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                lance_table TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO knowledge_tables(
                table_id, content_type, lance_table, schema_version,
                enabled, created_at, updated_at
            ) VALUES (
                'text_chunks_v1', 'text', 'text_chunks_v1_oldspace',
                1, 1, 'now', 'now'
            );
            """
        )

    database = KnowledgeDB(path)

    assert database.list_vector_tables("text_chunks_v1") == [
        "text_chunks_v1_oldspace"
    ]
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 7


def test_schema_v4_migrates_rechunk_flag_and_readonly_falls_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES (4, 'now');
            CREATE TABLE source_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'discovered',
                missing_count INTEGER NOT NULL DEFAULT 0,
                active_generation INTEGER NOT NULL DEFAULT 0,
                pending_generation INTEGER,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_id, relative_path)
            );
            INSERT INTO source_files(
                source_id, relative_path, mtime_ns, size, created_at, updated_at
            ) VALUES ('source', 'note.md', 0, 4, 'now', 'now');
            """
        )

    legacy = KnowledgeDB(path, readonly=True).list_source_files("source")
    assert len(legacy) == 1
    assert not legacy[0].rechunk_required

    migrated = KnowledgeDB(path)

    assert not migrated.list_source_files("source")[0].rechunk_required
    assert migrated.chunk_settings_state() is None
    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(source_files)")
        }
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert "rechunk_required" in columns
    assert version == 7


def test_schema_v5_migrates_chunk_settings_defaults_to_v7(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES (5, 'now');
            CREATE TABLE knowledge_chunk_settings (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                target_chars INTEGER NOT NULL,
                max_chars INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO knowledge_chunk_settings(
                singleton, target_chars, max_chars, updated_at
            ) VALUES (1, 123, 456, 'now');
            """
        )

    database = KnowledgeDB(path)

    assert database.chunk_settings_state() == (123, 456)
    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(knowledge_chunk_settings)"
            )
        }
        settings = connection.execute(
            """
            SELECT target_chars, max_chars, chunk_mode, overlap_chars
            FROM knowledge_chunk_settings
            WHERE singleton = 1
            """
        ).fetchone()
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert {"chunk_mode", "overlap_chars"} <= columns
    assert settings == (123, 456, "structure", 0)
    assert version == 7


def test_schema_v6_rebuilds_four_field_fts_and_marks_vectors_stale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES (6, 'now');
            CREATE TABLE sources (
                id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1
            );
            INSERT INTO sources(id) VALUES ('source');
            CREATE TABLE source_files (
                id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'indexed',
                active_generation INTEGER NOT NULL DEFAULT 0,
                pending_generation INTEGER
            );
            INSERT INTO source_files(
                id, source_id, relative_path, active_generation
            ) VALUES (1, 'source', 'nested/guide.md', 1);
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                source_file_id INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_input_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO chunks(
                chunk_id, source_file_id, generation, ordinal, text,
                search_text, heading_path, title, content_hash,
                embedding_input_hash, created_at
            ) VALUES (
                'chunk:1:0', 1, 1, 0, 'TextOnly', 'old tokens',
                'HeadingOnly', 'TitleOnly', 'content', 'embedding', 'now'
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                search_text,
                tokenize = 'unicode61'
            );
            INSERT INTO chunks_fts(chunk_id, search_text)
            VALUES ('chunk:1:0', 'old tokens');
            """
        )

    before = _sqlite_snapshot(path)
    readonly = KnowledgeDB(path, readonly=True)
    assert readonly.fts_search(fts_query("old"), limit=1)[0]["chunk_id"] == (
        "chunk:1:0"
    )
    assert _sqlite_snapshot(path) == before

    database = KnowledgeDB(path)

    with sqlite3.connect(path) as connection:
        chunk_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chunks)")
        }
        fts_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chunks_fts)")
        }
        row = connection.execute(
            """
            SELECT filename, search_text
            FROM chunks
            WHERE chunk_id = 'chunk:1:0'
            """
        ).fetchone()
        source_file = connection.execute(
            "SELECT rechunk_required FROM source_files WHERE id = 1"
        ).fetchone()
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]

    assert "filename" in chunk_columns
    assert {"text", "title", "heading_path", "filename"} <= fts_columns
    assert row == ("guide.md", "textonly")
    assert source_file == (1,)
    assert version == 7
    for query in ("TextOnly", "TitleOnly", "HeadingOnly", "guide"):
        assert database.fts_search(fts_query(query), limit=1)[0][
            "chunk_id"
        ] == "chunk:1:0"
    assert database.fts_search(fts_query("old"), limit=1) == []


def test_valid_readonly_search_does_not_write_derived_storage(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "knowledge"
    database = KnowledgeDB(store_root / "knowledge.sqlite3")
    (tmp_path / "source").mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=0,
        size=5,
    )
    chunk = ChunkRecord(
        chunk_id="note:1:0",
        source_file_id=source_file.id,
        generation=1,
        ordinal=0,
        text="alpha body",
        search_text="alpha body",
        heading_path="",
        title="Note",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=0,
        size=5,
        chunks=[chunk],
    )
    database.activate_generation(source_file.id, 1, "digest")
    vectors_root = store_root / "lancedb"
    writable_vectors = LanceVectorStore(
        vectors_root,
        table_name="text_chunks_v1_current",
        dimensions=1,
        embedding_space_id="space",
    )
    writable_vectors.add(
        [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": source.id,
                "source_file_id": source_file.id,
                "embedding_input_hash": "embedding",
                "embedding_space_id": "space",
                "generation": 1,
                "vector": [1.0],
            }
        ]
    )
    guard_path = store_root / "indexer.lock"
    AuthorizationGuard(guard_path).ensure_writable()
    with database.connect() as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before_db = database.path.read_bytes()
    before_store = tuple(
        (path.relative_to(vectors_root).as_posix(), path.stat().st_size)
        for path in sorted(vectors_root.rglob("*"))
        if path.is_file()
    )
    before_guard = guard_path.read_bytes()

    result = KnowledgeSearcher(
        database=KnowledgeDB(database.path, readonly=True),
        vectors=LanceVectorStore(
            vectors_root,
            table_name="text_chunks_v1_current",
            dimensions=1,
            embedding_space_id="space",
            readonly=True,
        ),
        embedding=_ReadonlyEmbedding(),
        reranker=_ReadonlyReranker(),
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    ).search(query="alpha")

    assert result["status"] == "ok"
    assert database.path.read_bytes() == before_db
    assert before_store == tuple(
        (path.relative_to(vectors_root).as_posix(), path.stat().st_size)
        for path in sorted(vectors_root.rglob("*"))
        if path.is_file()
    )
    assert guard_path.read_bytes() == before_guard
