from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keepygaga_rag.knowledge.chunking import (
    CHUNKER_VERSION,
    FTS_BM25_WEIGHTS,
    TOKENIZER_VERSION,
    filename_for_path,
    lexical_field_values,
    lexical_fields,
)

SCHEMA_VERSION = 7
FTS_BM25_ARGUMENTS = ", ".join(str(value) for value in FTS_BM25_WEIGHTS)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def timestamp_at_or_before(value: str, upper_bound: str) -> bool:
    if not value:
        return True
    try:
        parsed = datetime.fromisoformat(value)
        bound = datetime.fromisoformat(upper_bound)
    except ValueError:
        return value <= upper_bound
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if bound.tzinfo is None:
        bound = bound.replace(tzinfo=UTC)
    return parsed <= bound


def read_schema_version(path: Path) -> int | None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    try:
        with sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro",
            uri=True,
            timeout=5,
        ) as connection:
            row = connection.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).casefold():
            return None
        raise
    return int(row[0]) if row is not None and row[0] is not None else None


@dataclass(frozen=True)
class SourceRecord:
    id: str
    display_name: str
    absolute_path: str
    enabled: bool
    auto_sync: bool
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    consent_identity: str
    status: str
    last_scan_at: str
    last_success_at: str
    last_error: str
    created_at: str
    updated_at: str
    scope_cleanup_pending: bool = False


@dataclass(frozen=True)
class SourceFileRecord:
    id: int
    source_id: str
    relative_path: str
    mtime_ns: int
    size: int
    sha256: str
    status: str
    missing_count: int
    active_generation: int
    pending_generation: int | None
    last_error: str
    rechunk_required: bool


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_file_id: int
    generation: int
    ordinal: int
    text: str
    search_text: str
    heading_path: str
    title: str
    content_hash: str
    embedding_input_hash: str
    filename: str = ""
    fts_fields: tuple[str, str, str, str] | None = None


class KnowledgeDB:
    def __init__(
        self,
        path: Path,
        *,
        readonly: bool = False,
        initialize: bool = True,
    ):
        self.path = path.expanduser().resolve()
        self.readonly = readonly
        current = read_schema_version(self.path)
        if current is not None and current > SCHEMA_VERSION:
            raise RuntimeError(
                "unsupported knowledge schema version: "
                f"{current}; this code supports up to {SCHEMA_VERSION}"
            )
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(f"knowledge database does not exist: {self.path}")
        elif initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def connect(self) -> sqlite3.Connection:
        if self.readonly:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro",
                uri=True,
                timeout=5,
            )
        else:
            connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            raise RuntimeError("knowledge database is read-only")
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_tables (
                    table_id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    lance_table TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_profiles (
                    profile_id TEXT PRIMARY KEY,
                    table_id TEXT NOT NULL REFERENCES knowledge_tables(table_id),
                    embedding_identity TEXT NOT NULL,
                    rerank_identity TEXT NOT NULL,
                    tokenizer_version TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    retrieval_limits_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_vector_tables (
                    table_id TEXT NOT NULL REFERENCES knowledge_tables(table_id),
                    lance_table TEXT NOT NULL UNIQUE,
                    registered_at TEXT NOT NULL,
                    PRIMARY KEY(table_id, lance_table)
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    absolute_path TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    auto_sync INTEGER NOT NULL DEFAULT 1,
                    include_patterns_json TEXT NOT NULL,
                    exclude_patterns_json TEXT NOT NULL,
                    consent_identity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    scope_cleanup_pending INTEGER NOT NULL DEFAULT 0,
                    last_scan_at TEXT NOT NULL DEFAULT '',
                    last_success_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    relative_path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'discovered',
                    missing_count INTEGER NOT NULL DEFAULT 0,
                    active_generation INTEGER NOT NULL DEFAULT 0,
                    pending_generation INTEGER,
                    last_error TEXT NOT NULL DEFAULT '',
                    rechunk_required INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_id, relative_path)
                );

                CREATE INDEX IF NOT EXISTS source_files_status_idx
                    ON source_files(source_id, status);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_file_id INTEGER NOT NULL
                        REFERENCES source_files(id) ON DELETE CASCADE,
                    generation INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_input_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_file_id, generation, ordinal)
                );

                CREATE INDEX IF NOT EXISTS chunks_active_idx
                    ON chunks(source_file_id, generation);
                CREATE INDEX IF NOT EXISTS chunks_embedding_hash_idx
                    ON chunks(embedding_input_hash);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    title,
                    heading_path,
                    filename,
                    tokenize = 'unicode61'
                );

                CREATE TABLE IF NOT EXISTS sync_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    requested_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    reason TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT ''
                );

                CREATE UNIQUE INDEX IF NOT EXISTS sync_jobs_one_open_per_source
                    ON sync_jobs(source_id)
                    WHERE status IN ('queued', 'running');

                CREATE TABLE IF NOT EXISTS sync_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    scanned_files INTEGER NOT NULL DEFAULT 0,
                    indexed_files INTEGER NOT NULL DEFAULT 0,
                    reused_vectors INTEGER NOT NULL DEFAULT 0,
                    embedded_vectors INTEGER NOT NULL DEFAULT 0,
                    missing_files INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS knowledge_chunk_settings (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    target_chars INTEGER NOT NULL,
                    max_chars INTEGER NOT NULL,
                    chunk_mode TEXT NOT NULL DEFAULT 'structure',
                    overlap_chars INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vector_deletions (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL DEFAULT '',
                    scope_cleanup INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_rebuild_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    revision INTEGER NOT NULL DEFAULT 0
                );

                INSERT OR IGNORE INTO knowledge_rebuild_state(singleton, revision)
                VALUES (1, 0);

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_sources_insert
                AFTER INSERT ON sources
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_sources_update
                AFTER UPDATE ON sources
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_sources_delete
                AFTER DELETE ON sources
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_source_files_insert
                AFTER INSERT ON source_files
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_source_files_update
                AFTER UPDATE ON source_files
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_source_files_delete
                AFTER DELETE ON source_files
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_chunks_insert
                AFTER INSERT ON chunks
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_chunks_update
                AFTER UPDATE ON chunks
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_rebuild_chunks_delete
                AFTER DELETE ON chunks
                BEGIN
                    UPDATE knowledge_rebuild_state
                    SET revision = revision + 1
                    WHERE singleton = 1;
                END;
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = connection.execute(
                    "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0]
                if current is None:
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) "
                        "VALUES (?, ?)",
                        (SCHEMA_VERSION, utc_now()),
                    )
                elif current < SCHEMA_VERSION:
                    self._migrate(connection, int(current))
                elif current != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"unsupported knowledge schema version: {current}"
                    )
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _migrate(connection: sqlite3.Connection, current: int) -> None:
        if current not in {1, 2, 3, 4, 5, 6}:
            raise RuntimeError(f"unsupported knowledge schema version: {current}")
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(sources)").fetchall()
        }
        if "scope_cleanup_pending" not in columns:
            connection.execute(
                "ALTER TABLE sources ADD COLUMN scope_cleanup_pending "
                "INTEGER NOT NULL DEFAULT 0"
            )
        vector_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(vector_deletions)"
            ).fetchall()
        }
        if "source_id" not in vector_columns:
            connection.execute(
                "ALTER TABLE vector_deletions ADD COLUMN source_id "
                "TEXT NOT NULL DEFAULT ''"
            )
        if "scope_cleanup" not in vector_columns:
            connection.execute(
                "ALTER TABLE vector_deletions ADD COLUMN scope_cleanup "
                "INTEGER NOT NULL DEFAULT 0"
            )
        source_file_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(source_files)").fetchall()
        }
        if "rechunk_required" not in source_file_columns:
            connection.execute(
                "ALTER TABLE source_files ADD COLUMN rechunk_required "
                "INTEGER NOT NULL DEFAULT 0"
            )
        chunk_setting_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(knowledge_chunk_settings)"
            ).fetchall()
        }
        if "chunk_mode" not in chunk_setting_columns:
            connection.execute(
                "ALTER TABLE knowledge_chunk_settings ADD COLUMN "
                "chunk_mode TEXT NOT NULL DEFAULT 'structure'"
            )
        if "overlap_chars" not in chunk_setting_columns:
            connection.execute(
                "ALTER TABLE knowledge_chunk_settings ADD COLUMN "
                "overlap_chars INTEGER NOT NULL DEFAULT 0"
            )
        chunk_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "filename" not in chunk_columns:
            connection.execute(
                "ALTER TABLE chunks ADD COLUMN filename "
                "TEXT NOT NULL DEFAULT ''"
            )
        connection.execute("DROP TABLE IF EXISTS chunks_fts")
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                title,
                heading_path,
                filename,
                tokenize = 'unicode61'
            )
            """
        )
        chunk_cursor = connection.execute(
            """
            SELECT c.chunk_id, c.text, c.title, c.heading_path, c.filename,
                   sf.relative_path
            FROM chunks c
            JOIN source_files sf ON sf.id = c.source_file_id
            """
        )
        while rows := chunk_cursor.fetchmany(256):
            chunk_updates: list[tuple[str, str, str]] = []
            fts_rows: list[tuple[str, str, str, str, str]] = []
            for row in rows:
                filename = str(row["filename"] or "") or filename_for_path(
                    str(row["relative_path"])
                )
                fields = lexical_fields(
                    text=str(row["text"]),
                    title=str(row["title"]),
                    heading_path=str(row["heading_path"]),
                    filename=filename,
                )
                chunk_updates.append(
                    (filename, fields["text"], str(row["chunk_id"]))
                )
                fts_rows.append(
                    (
                        str(row["chunk_id"]),
                        fields["text"],
                        fields["title"],
                        fields["heading_path"],
                        fields["filename"],
                    )
                )
            connection.executemany(
                """
                UPDATE chunks SET filename = ?, search_text = ?
                WHERE chunk_id = ?
                """,
                chunk_updates,
            )
            connection.executemany(
                """
                INSERT INTO chunks_fts(
                    chunk_id, text, title, heading_path, filename
                ) VALUES (?, ?, ?, ?, ?)
                """,
                fts_rows,
            )
        connection.execute(
            """
            UPDATE source_files
            SET rechunk_required = 1
            WHERE active_generation > 0 OR pending_generation IS NOT NULL
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO knowledge_vector_tables(
                table_id, lance_table, registered_at
            )
            SELECT table_id, lance_table, ? FROM knowledge_tables
            """,
            (utc_now(),),
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utc_now()),
        )

    def register_table(
        self,
        *,
        table_id: str,
        lance_table: str,
        embedding_identity: str,
        rerank_identity: str,
        retrieval_limits: dict[str, int],
    ) -> None:
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT kt.lance_table, rp.embedding_identity
                FROM knowledge_tables kt
                LEFT JOIN retrieval_profiles rp
                  ON rp.profile_id = ? AND rp.table_id = kt.table_id
                WHERE kt.table_id = ?
                """,
                (f"{table_id}:default", table_id),
            ).fetchone()
            active_lance_table = lance_table
            if (
                existing is not None
                and str(existing["embedding_identity"] or "")
                == embedding_identity
            ):
                active_lance_table = str(existing["lance_table"])
            connection.execute(
                """
                INSERT INTO knowledge_tables(
                    table_id, content_type, lance_table, schema_version,
                    enabled, created_at, updated_at
                ) VALUES (?, 'text', ?, 1, 1, ?, ?)
                ON CONFLICT(table_id) DO UPDATE SET
                    lance_table = excluded.lance_table,
                    updated_at = excluded.updated_at
                """,
                (table_id, active_lance_table, now, now),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_vector_tables(
                    table_id, lance_table, registered_at
                ) VALUES (?, ?, ?)
                """,
                (table_id, lance_table, now),
            )
            connection.execute(
                """
                INSERT INTO retrieval_profiles(
                    profile_id, table_id, embedding_identity, rerank_identity,
                    tokenizer_version, chunker_version, retrieval_limits_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    embedding_identity = excluded.embedding_identity,
                    rerank_identity = excluded.rerank_identity,
                    tokenizer_version = excluded.tokenizer_version,
                    chunker_version = excluded.chunker_version,
                    retrieval_limits_json = excluded.retrieval_limits_json,
                    updated_at = excluded.updated_at
                """,
                (
                    f"{table_id}:default",
                    table_id,
                    embedding_identity,
                    rerank_identity,
                    TOKENIZER_VERSION,
                    CHUNKER_VERSION,
                    json.dumps(retrieval_limits, sort_keys=True),
                    now,
                    now,
                ),
            )

    def active_lance_table(
        self,
        table_id: str,
        *,
        embedding_identity: str,
    ) -> str | None:
        try:
            with self.connect() as connection:
                row = connection.execute(
                    """
                    SELECT kt.lance_table
                    FROM knowledge_tables kt
                    JOIN retrieval_profiles rp
                      ON rp.profile_id = ? AND rp.table_id = kt.table_id
                    WHERE kt.table_id = ?
                      AND rp.embedding_identity = ?
                    """,
                    (f"{table_id}:default", table_id, embedding_identity),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        return str(row[0]) if row is not None else None

    def vector_rebuild_revision(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT revision FROM knowledge_rebuild_state WHERE singleton = 1"
            ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("knowledge rebuild state is not initialized")
        return int(row[0])

    def activate_vector_table_if_revision_matches(
        self,
        table_id: str,
        lance_table: str,
        *,
        rebuild_revision: int,
    ) -> bool:
        with self.transaction() as connection:
            table = connection.execute(
                "SELECT 1 FROM knowledge_tables WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            if table is None:
                raise KeyError(table_id)
            current = connection.execute(
                "SELECT revision FROM knowledge_rebuild_state WHERE singleton = 1"
            ).fetchone()
            if current is None:  # pragma: no cover
                raise RuntimeError("knowledge rebuild state is not initialized")
            if int(current[0]) != rebuild_revision:
                return False
            connection.execute(
                """
                UPDATE knowledge_tables
                SET lance_table = ?, updated_at = ?
                WHERE table_id = ?
                """,
                (lance_table, utc_now(), table_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_vector_tables(
                    table_id, lance_table, registered_at
                ) VALUES (?, ?, ?)
                """,
                (table_id, lance_table, utc_now()),
            )
            return True

    def activate_vector_table(self, table_id: str, lance_table: str) -> None:
        with self.transaction() as connection:
            table = connection.execute(
                "SELECT 1 FROM knowledge_tables WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            if table is None:
                raise KeyError(table_id)
            connection.execute(
                """
                UPDATE knowledge_tables
                SET lance_table = ?, updated_at = ?
                WHERE table_id = ?
                """,
                (lance_table, utc_now(), table_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_vector_tables(
                    table_id, lance_table, registered_at
                ) VALUES (?, ?, ?)
                """,
                (table_id, lance_table, utc_now()),
            )

    def list_vector_tables(self, table_id: str) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT lance_table FROM knowledge_vector_tables
                WHERE table_id = ?
                ORDER BY registered_at, lance_table
                """,
                (table_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def register_vector_table(self, table_id: str, lance_table: str) -> None:
        with self.transaction() as connection:
            table = connection.execute(
                "SELECT 1 FROM knowledge_tables WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            if table is None:
                raise KeyError(table_id)
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_vector_tables(
                    table_id, lance_table, registered_at
                ) VALUES (?, ?, ?)
                """,
                (table_id, lance_table, utc_now()),
            )

    def list_sources(self) -> list[SourceRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sources ORDER BY display_name, absolute_path"
            ).fetchall()
        return [self._source(row) for row in rows]

    def get_source(self, source_id: str) -> SourceRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
        return self._source(row) if row is not None else None

    def list_sync_jobs(
        self,
        *,
        limit: int = 50,
        statuses: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        if not statuses:
            open_jobs = self.list_sync_jobs(
                limit=limit,
                statuses=("queued", "running"),
            )
            if len(open_jobs) >= limit:
                return open_jobs[:limit]
            return open_jobs + self.list_sync_jobs(
                limit=limit - len(open_jobs),
                statuses=("succeeded", "failed"),
            )
        parameters: list[object] = []
        placeholders = ", ".join("?" for _ in statuses)
        status_clause = f"j.status IN ({placeholders})"
        parameters.extend(statuses)
        parameters.append(limit)
        open_statuses = all(status in {"queued", "running"} for status in statuses)
        closed_statuses = all(
            status in {"succeeded", "failed"} for status in statuses
        )
        if open_statuses:
            order_clause = "j.requested_at, j.id"
        elif closed_statuses:
            order_clause = "j.id DESC"
        else:
            order_clause = "j.id DESC"
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    j.id, j.source_id, s.display_name AS source_name,
                    s.absolute_path AS source_path, j.requested_at,
                    j.started_at, j.finished_at, j.status, j.reason, j.error
                FROM sync_jobs AS j
                JOIN sources AS s ON s.id = j.source_id
                WHERE {status_clause}
                ORDER BY {order_clause}
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def sync_job_counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM sync_jobs
                WHERE status IN ('queued', 'running')
                GROUP BY status
                """
            ).fetchall()
        result = {"queued": 0, "running": 0}
        result.update({str(row[0]): int(row[1]) for row in rows})
        return result

    def list_sync_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.id, r.source_id, s.display_name AS source_name,
                    s.absolute_path AS source_path, r.started_at,
                    r.finished_at, r.status, r.scanned_files,
                    r.indexed_files, r.reused_vectors, r.embedded_vectors,
                    r.missing_files, r.error
                FROM sync_runs AS r
                JOIN sources AS s ON s.id = r.source_id
                ORDER BY r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_source_file(self, file_id: int) -> SourceFileRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_files WHERE id = ?", (file_id,)
            ).fetchone()
        return self._source_file(row) if row is not None else None

    def list_failed_source_files(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    sf.id, sf.source_id, s.display_name AS source_name,
                    s.absolute_path AS source_path, sf.relative_path,
                    sf.status, sf.last_error, sf.active_generation,
                    sf.pending_generation, sf.updated_at
                FROM source_files AS sf
                JOIN sources AS s ON s.id = sf.source_id
                WHERE sf.status = 'error'
                ORDER BY sf.updated_at DESC, sf.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_source(
        self,
        *,
        display_name: str,
        absolute_path: str,
        include_patterns: Sequence[str],
        exclude_patterns: Sequence[str],
        consent_identity: str,
        auto_sync: bool = True,
    ) -> SourceRecord:
        source_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sources(
                    id, display_name, absolute_path, enabled, auto_sync,
                    include_patterns_json, exclude_patterns_json,
                    consent_identity, status, created_at, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, 'idle', ?, ?)
                """,
                (
                    source_id,
                    display_name,
                    absolute_path,
                    int(auto_sync),
                    json.dumps(list(include_patterns), ensure_ascii=False),
                    json.dumps(list(exclude_patterns), ensure_ascii=False),
                    consent_identity,
                    now,
                    now,
                ),
            )
        result = self.get_source(source_id)
        if result is None:  # pragma: no cover
            raise RuntimeError("source insert did not persist")
        return result

    def set_source_enabled(self, source_id: str, enabled: bool) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE sources
                SET enabled = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(enabled), "idle" if enabled else "paused", utc_now(), source_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(source_id)

    def set_source_auto_sync(self, source_id: str, enabled: bool) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE sources
                SET auto_sync = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(enabled), utc_now(), source_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(source_id)

    def set_source_consent(self, source_id: str, consent_identity: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE sources
                SET consent_identity = ?, updated_at = ?
                WHERE id = ?
                """,
                (consent_identity, utc_now(), source_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(source_id)

    def renew_source_consent(
        self,
        source_id: str,
        consent_identity: str,
        *,
        reason: str,
    ) -> int | None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE sources
                SET consent_identity = ?, updated_at = ?
                WHERE id = ? AND scope_cleanup_pending = 0
                """,
                (consent_identity, utc_now(), source_id),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT scope_cleanup_pending FROM sources WHERE id = ?",
                    (source_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(source_id)
                raise ValueError("source scope cleanup is pending")
            source = connection.execute(
                "SELECT enabled FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
            if source is None:  # pragma: no cover
                raise KeyError(source_id)
            if not bool(source["enabled"]):
                return None
            existing = connection.execute(
                """
                SELECT id FROM sync_jobs
                WHERE source_id = ? AND status IN ('queued', 'running')
                """,
                (source_id,),
            ).fetchone()
            if existing is not None:
                return int(existing[0])
            queued = connection.execute(
                """
                INSERT INTO sync_jobs(source_id, requested_at, status, reason)
                VALUES (?, ?, 'queued', ?)
                """,
                (source_id, utc_now(), reason),
            )
            connection.execute(
                """
                UPDATE sources
                SET status = 'queued', updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), source_id),
            )
            if queued.lastrowid is None:  # pragma: no cover
                raise RuntimeError("sync job insert did not return an id")
            return int(queued.lastrowid)

    def set_source_scope(
        self,
        source_id: str,
        *,
        include_patterns: Sequence[str],
        consent_identity: str,
    ) -> None:
        with self.transaction() as connection:
            open_job = connection.execute(
                """
                SELECT 1 FROM sync_jobs
                WHERE source_id = ? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (source_id,),
            ).fetchone()
            if open_job is not None:
                raise ValueError("source has a queued or running sync job")
            cursor = connection.execute(
                """
                UPDATE sources
                SET include_patterns_json = ?, consent_identity = ?,
                    scope_cleanup_pending = 1, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(list(include_patterns), ensure_ascii=False),
                    consent_identity,
                    utc_now(),
                    source_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(source_id)

    def complete_source_scope_cleanup(self, source_id: str) -> None:
        with self.transaction() as connection:
            pending = connection.execute(
                """
                SELECT 1 FROM vector_deletions
                WHERE source_id = ? AND scope_cleanup = 1
                LIMIT 1
                """,
                (source_id,),
            ).fetchone()
            if pending is not None:
                raise RuntimeError("source scope vector cleanup is still pending")
            cursor = connection.execute(
                """
                UPDATE sources
                SET scope_cleanup_pending = 0, updated_at = ?
                WHERE id = ? AND scope_cleanup_pending = 1
                """,
                (utc_now(), source_id),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT 1 FROM sources WHERE id = ?", (source_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(source_id)
                raise RuntimeError("source scope cleanup is not pending")

    @staticmethod
    def _queue_vector_deletions(
        connection: sqlite3.Connection,
        chunk_ids: Iterable[str],
        *,
        source_id: str,
        scope_cleanup: bool = False,
    ) -> None:
        values = tuple(dict.fromkeys(chunk_ids))
        if not values:
            return
        connection.executemany(
            """
            INSERT INTO vector_deletions(
                chunk_id, source_id, scope_cleanup, queued_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                source_id = CASE WHEN excluded.scope_cleanup = 1
                    THEN excluded.source_id ELSE vector_deletions.source_id END,
                scope_cleanup = MAX(
                    vector_deletions.scope_cleanup, excluded.scope_cleanup
                )
            """,
            (
                (chunk_id, source_id, int(scope_cleanup), utc_now())
                for chunk_id in values
            ),
        )

    def pending_vector_deletions(self, *, limit: int = 1_000) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id FROM vector_deletions
                ORDER BY queued_at, chunk_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def complete_vector_deletions(self, chunk_ids: Iterable[str]) -> None:
        values = tuple(dict.fromkeys(chunk_ids))
        if not values:
            return
        with self.transaction() as connection:
            connection.executemany(
                "DELETE FROM vector_deletions WHERE chunk_id = ?",
                ((chunk_id,) for chunk_id in values),
            )

    def remove_source(self, source_id: str) -> list[str]:
        with self.transaction() as connection:
            open_job = connection.execute(
                """
                SELECT 1 FROM sync_jobs
                WHERE source_id = ? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (source_id,),
            ).fetchone()
            if open_job is not None:
                raise ValueError("source has a queued or running sync job")
            chunk_ids = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT c.chunk_id
                    FROM chunks c
                    JOIN source_files sf ON sf.id = c.source_file_id
                    WHERE sf.source_id = ?
                    """,
                    (source_id,),
                )
            ]
            if chunk_ids:
                connection.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    ((chunk_id,) for chunk_id in chunk_ids),
                )
                self._queue_vector_deletions(
                    connection,
                    chunk_ids,
                    source_id=source_id,
                )
            cursor = connection.execute(
                "DELETE FROM sources WHERE id = ?", (source_id,)
            )
            if cursor.rowcount != 1:
                raise KeyError(source_id)
        return chunk_ids

    def source_stats(self, source_id: str) -> dict[str, int]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(DISTINCT sf.id) AS files,
                    COUNT(c.chunk_id) AS chunks,
                    COUNT(DISTINCT CASE WHEN sf.status = 'error' THEN sf.id END)
                        AS errors,
                    COUNT(DISTINCT CASE WHEN sf.status = 'missing_pending'
                        THEN sf.id END) AS missing_pending,
                    COUNT(DISTINCT CASE WHEN sf.status = 'waiting_stable'
                        THEN sf.id END) AS waiting_stable,
                    COUNT(DISTINCT CASE WHEN sf.status = 'pending'
                        THEN sf.id END) AS pending
                FROM source_files sf
                LEFT JOIN chunks c
                    ON c.source_file_id = sf.id
                    AND c.generation = sf.active_generation
                WHERE sf.source_id = ?
                """,
                (source_id,),
            ).fetchone()
        if row is None:
            return {
                "files": 0,
                "chunks": 0,
                "errors": 0,
                "missing_pending": 0,
                "waiting_stable": 0,
                "pending": 0,
            }
        keys = (
            "files",
            "chunks",
            "errors",
            "missing_pending",
            "waiting_stable",
            "pending",
        )
        return {key: int(row[key] or 0) for key in keys}

    def queue_sync(
        self,
        source_id: str,
        reason: str,
        *,
        expected_consent_identity: str | None = None,
    ) -> int:
        with self.transaction() as connection:
            source = connection.execute(
                """
                SELECT enabled, scope_cleanup_pending, consent_identity
                FROM sources WHERE id = ?
                """,
                (source_id,),
            ).fetchone()
            if source is None:
                raise KeyError(source_id)
            if not bool(source["enabled"]):
                raise ValueError("source is disabled")
            if bool(source["scope_cleanup_pending"]):
                raise ValueError("source scope cleanup is pending")
            if (
                expected_consent_identity is not None
                and str(source["consent_identity"])
                != expected_consent_identity
            ):
                raise ValueError("source provider consent changed")
            existing = connection.execute(
                """
                SELECT id FROM sync_jobs
                WHERE source_id = ? AND status IN ('queued', 'running')
                """,
                (source_id,),
            ).fetchone()
            if existing is not None:
                return int(existing[0])
            cursor = connection.execute(
                """
                INSERT INTO sync_jobs(source_id, requested_at, status, reason)
                VALUES (?, ?, 'queued', ?)
                """,
                (source_id, utc_now(), reason),
            )
            connection.execute(
                """
                UPDATE sources
                SET status = 'queued', updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), source_id),
            )
            if cursor.lastrowid is None:  # pragma: no cover
                raise RuntimeError("sync job insert did not return an id")
        return cursor.lastrowid

    def prepare_chunk_rebuild(
        self,
        *,
        target_chars: int,
        max_chars: int,
        chunk_mode: str = "structure",
        overlap_chars: int = 0,
        consent_identity: str,
        safe_source_ids: Sequence[str],
        reason: str,
    ) -> dict[str, int]:
        safe_ids = set(safe_source_ids)
        result = {
            "marked_files": 0,
            "queued_sources": 0,
            "new_jobs": 0,
            "skipped_disabled": 0,
            "skipped_consent": 0,
            "skipped_scope_cleanup": 0,
            "skipped_unsafe": 0,
            "deferred_sources": 0,
            "changed": 0,
        }
        with self.transaction() as connection:
            current = connection.execute(
                """
                SELECT target_chars, max_chars, chunk_mode, overlap_chars
                FROM knowledge_chunk_settings
                WHERE singleton = 1
                """
            ).fetchone()
            if current is not None and (
                int(current["target_chars"]),
                int(current["max_chars"]),
                str(current["chunk_mode"]),
                int(current["overlap_chars"]),
            ) == (target_chars, max_chars, chunk_mode, overlap_chars):
                return result
            result["changed"] = 1
            marked = connection.execute(
                """
                UPDATE source_files
                SET rechunk_required = 1, updated_at = ?
                WHERE active_generation > 0
                """,
                (utc_now(),),
            )
            result["marked_files"] = marked.rowcount
            sources = connection.execute(
                """
                SELECT id, enabled, consent_identity, scope_cleanup_pending
                FROM sources
                ORDER BY id
                """
            ).fetchall()
            for source in sources:
                source_id = str(source["id"])
                if not bool(source["enabled"]):
                    result["skipped_disabled"] += 1
                    continue
                if bool(source["scope_cleanup_pending"]):
                    result["skipped_scope_cleanup"] += 1
                    continue
                if str(source["consent_identity"]) != consent_identity:
                    result["skipped_consent"] += 1
                    continue
                if source_id not in safe_ids:
                    result["skipped_unsafe"] += 1
                    continue
                existing = connection.execute(
                    """
                    SELECT id FROM sync_jobs
                    WHERE source_id = ? AND status IN ('queued', 'running')
                    """,
                    (source_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO sync_jobs(
                            source_id, requested_at, status, reason
                        ) VALUES (?, ?, 'queued', ?)
                        """,
                        (source_id, utc_now(), reason),
                    )
                    result["new_jobs"] += 1
                connection.execute(
                    """
                    UPDATE sources
                    SET status = 'queued', updated_at = ?
                    WHERE id = ?
                    """,
                    (utc_now(), source_id),
                )
                result["queued_sources"] += 1
            result["deferred_sources"] = sum(
                result[key]
                for key in (
                    "skipped_disabled",
                    "skipped_consent",
                    "skipped_scope_cleanup",
                    "skipped_unsafe",
                )
            )
            connection.execute(
                """
                INSERT INTO knowledge_chunk_settings(
                    singleton, target_chars, max_chars, chunk_mode,
                    overlap_chars, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    target_chars = excluded.target_chars,
                    max_chars = excluded.max_chars,
                    chunk_mode = excluded.chunk_mode,
                    overlap_chars = excluded.overlap_chars,
                    updated_at = excluded.updated_at
                """,
                (
                    target_chars,
                    max_chars,
                    chunk_mode,
                    overlap_chars,
                    utc_now(),
                ),
            )
        return result

    def chunk_settings_state(self) -> tuple[int, int] | None:
        with self.connect() as connection:
            try:
                row = connection.execute(
                    """
                    SELECT target_chars, max_chars
                    FROM knowledge_chunk_settings
                    WHERE singleton = 1
                    """
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        if row is None:
            return None
        return int(row["target_chars"]), int(row["max_chars"])

    def requeue_job(self, job_id: int, *, error: str = "") -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT source_id FROM sync_jobs WHERE id = ? AND status = 'running'",
                (job_id,),
            ).fetchone()
            if row is None:
                exists = connection.execute(
                    "SELECT 1 FROM sync_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                raise ValueError("sync job is not running")
            connection.execute(
                """
                UPDATE sync_jobs
                SET status = 'queued', started_at = '', finished_at = '', error = ?
                WHERE id = ? AND status = 'running'
                """,
                (error, job_id),
            )
            connection.execute(
                """
                UPDATE sources
                SET status = 'queued', updated_at = ?
                WHERE id = ? AND enabled = 1 AND scope_cleanup_pending = 0
                """,
                (utc_now(), str(row["source_id"])),
            )

    def claim_next_job(self) -> tuple[int, SourceRecord] | None:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT j.id, j.source_id
                FROM sync_jobs j
                JOIN sources s ON s.id = j.source_id
                WHERE j.status = 'queued' AND s.enabled = 1
                  AND s.scope_cleanup_pending = 0
                ORDER BY j.requested_at, j.id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE sync_jobs
                SET status = 'running', started_at = ?
                WHERE id = ?
                """,
                (utc_now(), int(row["id"])),
            )
            source_row = connection.execute(
                "SELECT * FROM sources WHERE id = ?", (str(row["source_id"]),)
            ).fetchone()
        if source_row is None:  # pragma: no cover
            return None
        return int(row["id"]), self._source(source_row)

    def recover_interrupted_jobs(self) -> int:
        with self.transaction() as connection:
            now = utc_now()
            valid_source_ids = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT j.source_id
                    FROM sync_jobs AS j
                    JOIN sources AS s ON s.id = j.source_id
                    WHERE j.status = 'running'
                      AND s.enabled = 1
                      AND s.scope_cleanup_pending = 0
                    """
                )
            }
            run_source_ids = {
                str(row[0])
                for row in connection.execute(
                    "SELECT source_id FROM sync_runs WHERE status = 'running'"
                )
            }
            cursor = connection.execute(
                """
                UPDATE sync_jobs
                SET status = 'queued', started_at = '', finished_at = '',
                    error = ''
                WHERE status = 'running'
                  AND source_id IN (
                      SELECT id FROM sources
                      WHERE enabled = 1 AND scope_cleanup_pending = 0
                  )
                """
            )
            invalid_source_rows = connection.execute(
                """
                SELECT DISTINCT j.source_id, s.enabled, s.scope_cleanup_pending
                FROM sync_jobs AS j
                JOIN sources AS s ON s.id = j.source_id
                WHERE j.status IN ('queued', 'running')
                  AND (s.enabled = 0 OR s.scope_cleanup_pending = 1)
                """
            ).fetchall()
            connection.execute(
                """
                UPDATE sync_jobs
                SET status = 'failed', finished_at = ?,
                    error = CASE
                        WHEN source_id IN (
                            SELECT id FROM sources WHERE enabled = 0
                        ) THEN 'source is disabled during indexer recovery'
                        ELSE 'source scope cleanup is pending during indexer recovery'
                    END
                WHERE status IN ('queued', 'running')
                  AND source_id IN (
                      SELECT id FROM sources
                      WHERE enabled = 0 OR scope_cleanup_pending = 1
                  )
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE sync_runs
                SET finished_at = ?, status = 'failed',
                    error = 'indexer interrupted before sync run completed'
                WHERE status = 'running'
                """,
                (now,),
            )
            invalid_source_ids = {
                str(row["source_id"]) for row in invalid_source_rows
            }
            orphan_run_source_ids = run_source_ids - valid_source_ids
            if orphan_run_source_ids:
                connection.executemany(
                    """
                    UPDATE sources
                    SET status = CASE
                            WHEN enabled = 0 THEN 'paused'
                            ELSE 'error'
                        END,
                        last_error = CASE
                            WHEN enabled = 0 THEN 'source is disabled during indexer recovery'
                            WHEN scope_cleanup_pending = 1 THEN 'source scope cleanup is pending during indexer recovery'
                            ELSE 'indexer interrupted before sync run completed'
                        END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    ((now, source_id) for source_id in orphan_run_source_ids),
                )
            if valid_source_ids:
                connection.executemany(
                    """
                    UPDATE sources
                    SET status = 'queued', last_error = '', updated_at = ?
                    WHERE id = ? AND enabled = 1 AND scope_cleanup_pending = 0
                    """,
                    ((now, source_id) for source_id in valid_source_ids),
                )
            if invalid_source_ids:
                connection.executemany(
                    """
                    UPDATE sources
                    SET status = CASE WHEN enabled = 0 THEN 'paused' ELSE 'error' END,
                        last_error = CASE
                            WHEN enabled = 0 THEN 'source is disabled during indexer recovery'
                            ELSE 'source scope cleanup is pending during indexer recovery'
                        END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    ((now, source_id) for source_id in invalid_source_ids),
                )
            return cursor.rowcount

    def finish_job(self, job_id: int, *, error: str = "") -> None:
        status = "failed" if error else "succeeded"
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT source_id FROM sync_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            connection.execute(
                """
                UPDATE sync_jobs
                SET status = ?, finished_at = ?, error = ?
                WHERE id = ?
                """,
                (status, utc_now(), error, job_id),
            )
            if error:
                connection.execute(
                    """
                    UPDATE sources
                    SET status = 'error', last_error = ?, updated_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (error, utc_now(), str(row[0])),
                )

    def queue_due_sources(
        self,
        *,
        before: str,
        consent_identity: str | None = None,
        source_root_predicate: Callable[[str], bool] | None = None,
        source_predicate: Callable[[SourceRecord], bool] | None = None,
    ) -> int:
        queued = 0
        for source in self.list_sources():
            if (
                source.enabled
                and source.auto_sync
                and not source.scope_cleanup_pending
                and (
                    consent_identity is None
                    or source.consent_identity == consent_identity
                )
                and timestamp_at_or_before(source.last_scan_at, before)
                and (
                    source_root_predicate is None
                    or source_root_predicate(source.absolute_path)
                )
                and (source_predicate is None or source_predicate(source))
            ):
                try:
                    self.queue_sync(
                        source.id,
                        "periodic",
                        expected_consent_identity=consent_identity,
                    )
                except (KeyError, ValueError):
                    continue
                else:
                    queued += 1
        return queued

    def list_source_files(self, source_id: str) -> list[SourceFileRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_files
                WHERE source_id = ?
                ORDER BY relative_path
                """,
                (source_id,),
            ).fetchall()
        return [self._source_file(row) for row in rows]

    def get_or_create_source_file(
        self,
        *,
        source_id: str,
        relative_path: str,
        mtime_ns: int,
        size: int,
    ) -> SourceFileRecord:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO source_files(
                    source_id, relative_path, mtime_ns, size, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, relative_path) DO NOTHING
                """,
                (source_id, relative_path, mtime_ns, size, now, now),
            )
            row = connection.execute(
                """
                SELECT * FROM source_files
                WHERE source_id = ? AND relative_path = ?
                """,
                (source_id, relative_path),
            ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("source file insert did not persist")
        return self._source_file(row)

    def mark_file_waiting(
        self, file_id: int, *, mtime_ns: int, size: int
    ) -> None:
        self._update_file(
            file_id,
            mtime_ns=mtime_ns,
            size=size,
            status="waiting_stable",
            last_error="",
        )

    def mark_file_unchanged(
        self, file_id: int, *, mtime_ns: int, size: int
    ) -> None:
        self._update_file(
            file_id,
            mtime_ns=mtime_ns,
            size=size,
            status="indexed",
            missing_count=0,
            last_error="",
        )

    def mark_file_error(self, file_id: int, message: str) -> None:
        self._update_file(file_id, status="error", last_error=message)

    def mark_file_missing(self, file_id: int) -> tuple[int, list[str]]:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT missing_count FROM source_files WHERE id = ?", (file_id,)
            ).fetchone()
            if row is None:
                raise KeyError(file_id)
            missing_count = int(row[0]) + 1
            connection.execute(
                """
                UPDATE source_files
                SET missing_count = ?, status = 'missing_pending', updated_at = ?
                WHERE id = ?
                """,
                (missing_count, utc_now(), file_id),
            )
            chunk_ids = [
                str(item[0])
                for item in connection.execute(
                    "SELECT chunk_id FROM chunks WHERE source_file_id = ?",
                    (file_id,),
                )
            ]
        return missing_count, chunk_ids

    def delete_source_file(
        self,
        file_id: int,
        *,
        scope_cleanup: bool = False,
    ) -> list[str]:
        with self.transaction() as connection:
            source_row = connection.execute(
                "SELECT source_id FROM source_files WHERE id = ?", (file_id,)
            ).fetchone()
            if source_row is None:
                raise KeyError(file_id)
            chunk_ids = [
                str(item[0])
                for item in connection.execute(
                    "SELECT chunk_id FROM chunks WHERE source_file_id = ?",
                    (file_id,),
                )
            ]
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                ((chunk_id,) for chunk_id in chunk_ids),
            )
            self._queue_vector_deletions(
                connection,
                chunk_ids,
                source_id=str(source_row["source_id"]),
                scope_cleanup=scope_cleanup,
            )
            connection.execute("DELETE FROM source_files WHERE id = ?", (file_id,))
        return chunk_ids

    def stage_generation(
        self,
        *,
        file_id: int,
        generation: int,
        mtime_ns: int,
        size: int,
        chunks: Sequence[ChunkRecord],
    ) -> None:
        with self.transaction() as connection:
            source_row = connection.execute(
                """
                SELECT source_id, relative_path FROM source_files WHERE id = ?
                """,
                (file_id,),
            ).fetchone()
            if source_row is None:
                raise KeyError(file_id)
            default_filename = filename_for_path(str(source_row["relative_path"]))
            old_pending = connection.execute(
                """
                SELECT chunk_id FROM chunks
                WHERE source_file_id = ? AND generation = ?
                """,
                (file_id, generation),
            ).fetchall()
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                ((str(row[0]),) for row in old_pending),
            )
            self._queue_vector_deletions(
                connection,
                (str(row[0]) for row in old_pending),
                source_id=str(source_row["source_id"]),
            )
            connection.execute(
                """
                DELETE FROM chunks
                WHERE source_file_id = ? AND generation = ?
                """,
                (file_id, generation),
            )
            now = utc_now()
            for offset in range(0, len(chunks), 256):
                batch = chunks[offset : offset + 256]
                chunk_rows = []
                fts_rows = []
                for chunk in batch:
                    fields = (
                        chunk.fts_fields
                        if chunk.fts_fields is not None
                        else lexical_field_values(
                            text=chunk.text,
                            title=chunk.title,
                            heading_path=chunk.heading_path,
                            filename=chunk.filename or default_filename,
                        )
                    )
                    chunk_rows.append(
                        (
                            chunk.chunk_id,
                            chunk.source_file_id,
                            chunk.generation,
                            chunk.ordinal,
                            chunk.text,
                            fields[0],
                            chunk.heading_path,
                            chunk.title,
                            chunk.filename or default_filename,
                            chunk.content_hash,
                            chunk.embedding_input_hash,
                            now,
                        )
                    )
                    fts_rows.append((chunk.chunk_id, *fields))
                connection.executemany(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_file_id, generation, ordinal, text,
                        search_text, heading_path, title, filename, content_hash,
                        embedding_input_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )
                connection.executemany(
                    """
                    INSERT INTO chunks_fts(
                        chunk_id, text, title, heading_path, filename
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    fts_rows,
                )
            connection.execute(
                """
                UPDATE source_files
                SET pending_generation = ?, mtime_ns = ?, size = ?,
                    status = 'pending', missing_count = 0, last_error = '',
                    updated_at = ?
                WHERE id = ?
                """,
                (generation, mtime_ns, size, now, file_id),
            )

    def activate_generation(
        self, file_id: int, generation: int, sha256: str
    ) -> list[str]:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT source_id, pending_generation FROM source_files
                WHERE id = ?
                """,
                (file_id,),
            ).fetchone()
            if row is None or row["pending_generation"] != generation:
                raise RuntimeError("pending generation changed before activation")
            old_ids = [
                str(item[0])
                for item in connection.execute(
                    """
                    SELECT chunk_id FROM chunks
                    WHERE source_file_id = ? AND generation != ?
                    """,
                    (file_id, generation),
                )
            ]
            connection.execute(
                """
                UPDATE source_files
                SET active_generation = ?, pending_generation = NULL,
                    sha256 = ?, status = 'indexed', rechunk_required = 0,
                    updated_at = ?
                WHERE id = ?
                """,
                (generation, sha256, utc_now(), file_id),
            )
            self._queue_vector_deletions(
                connection,
                old_ids,
                source_id=str(row["source_id"]),
            )
        return old_ids

    def cleanup_old_chunks(
        self, file_id: int, generation: int, chunk_ids: Iterable[str]
    ) -> None:
        values = tuple(dict.fromkeys(chunk_ids))
        if not values:
            return
        with self.transaction() as connection:
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                ((chunk_id,) for chunk_id in values),
            )
            source_row = connection.execute(
                "SELECT source_id FROM source_files WHERE id = ?", (file_id,)
            ).fetchone()
            if source_row is None:
                raise KeyError(file_id)
            self._queue_vector_deletions(
                connection,
                values,
                source_id=str(source_row["source_id"]),
            )
            connection.execute(
                """
                DELETE FROM chunks
                WHERE source_file_id = ? AND generation != ?
                """,
                (file_id, generation),
            )

    def cleanup_inactive_chunks(self) -> list[str]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, sf.source_id
                FROM chunks c
                JOIN source_files sf ON sf.id = c.source_file_id
                WHERE c.generation != sf.active_generation
                """
            ).fetchall()
            if not rows:
                return []
            chunk_ids = [str(row["chunk_id"]) for row in rows]
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                ((chunk_id,) for chunk_id in chunk_ids),
            )
            for source_id in {str(row["source_id"]) for row in rows}:
                self._queue_vector_deletions(
                    connection,
                    (
                        str(row["chunk_id"])
                        for row in rows
                        if str(row["source_id"]) == source_id
                    ),
                    source_id=source_id,
                )
            connection.executemany(
                "DELETE FROM chunks WHERE chunk_id = ?",
                ((chunk_id,) for chunk_id in chunk_ids),
            )
        return chunk_ids

    def active_chunks_by_hashes(
        self, hashes: Sequence[str]
    ) -> dict[str, str]:
        if not hashes:
            return {}
        placeholders = ", ".join("?" for _ in hashes)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.embedding_input_hash, c.chunk_id
                FROM chunks c
                JOIN source_files sf ON sf.id = c.source_file_id
                WHERE c.embedding_input_hash IN ({placeholders})
                  AND c.generation = sf.active_generation
                """,
                tuple(hashes),
            ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def active_chunk_ids(self, file_id: int) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id
                FROM chunks
                WHERE source_file_id = ?
                  AND generation = (
                      SELECT active_generation
                      FROM source_files
                      WHERE id = ?
                  )
                ORDER BY ordinal
                """,
                (file_id, file_id),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def active_chunk_vector_metadata(self, file_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.embedding_input_hash
                FROM chunks AS c
                JOIN source_files AS sf ON sf.id = c.source_file_id
                WHERE c.source_file_id = ?
                  AND c.generation = sf.active_generation
                ORDER BY c.ordinal
                """,
                (file_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def active_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    c.*, sf.relative_path, s.id AS source_id,
                    s.display_name AS source_name, s.absolute_path
                FROM chunks c
                JOIN source_files sf ON sf.id = c.source_file_id
                JOIN sources s ON s.id = sf.source_id
                WHERE c.chunk_id = ?
                  AND c.generation = sf.active_generation
                  AND s.enabled = 1
                """,
                (chunk_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def active_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    c.*, sf.relative_path, s.id AS source_id,
                    s.display_name AS source_name, s.absolute_path
                FROM chunks c
                JOIN source_files sf ON sf.id = c.source_file_id
                JOIN sources s ON s.id = sf.source_id
                WHERE c.chunk_id IN ({placeholders})
                  AND c.generation = sf.active_generation
                  AND s.enabled = 1
                """,
                tuple(chunk_ids),
            ).fetchall()
        by_id = {str(row["chunk_id"]): dict(row) for row in rows}
        return [by_id[item] for item in chunk_ids if item in by_id]

    def active_chunks_for_vector_rebuild(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.*, sf.relative_path, s.id AS source_id
                FROM chunks c
                JOIN source_files sf ON sf.id = c.source_file_id
                JOIN sources s ON s.id = sf.source_id
                WHERE c.generation = sf.active_generation
                  AND s.enabled = 1
                ORDER BY s.id, sf.id, c.ordinal
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def iter_active_chunks_for_vector_rebuild(
        self,
        *,
        page_size: int = 256,
    ) -> Iterator[list[dict[str, Any]]]:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        last_source_id = ""
        last_source_file_id = -1
        last_ordinal = -1
        while True:
            with self.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        c.*, sf.relative_path, s.id AS source_id
                    FROM chunks c
                    JOIN source_files sf ON sf.id = c.source_file_id
                    JOIN sources s ON s.id = sf.source_id
                    WHERE c.generation = sf.active_generation
                      AND s.enabled = 1
                      AND (
                          s.id > ?
                          OR (s.id = ? AND sf.id > ?)
                          OR (
                              s.id = ?
                              AND sf.id = ?
                              AND c.ordinal > ?
                          )
                      )
                    ORDER BY s.id, sf.id, c.ordinal
                    LIMIT ?
                    """,
                    (
                        last_source_id,
                        last_source_id,
                        last_source_file_id,
                        last_source_id,
                        last_source_file_id,
                        last_ordinal,
                        page_size,
                    ),
                ).fetchall()
            if not rows:
                return
            page = [dict(row) for row in rows]
            yield page
            last = rows[-1]
            last_source_id = str(last["source_id"])
            last_source_file_id = int(last["source_file_id"])
            last_ordinal = int(last["ordinal"])

    def fts_search(
        self,
        query: str,
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        source_clause = ""
        parameters: list[object] = [query]
        if source_ids:
            placeholders = ", ".join("?" for _ in source_ids)
            source_clause = f" AND s.id IN ({placeholders})"
            parameters.extend(source_ids)
        parameters.append(limit)
        with self.connect() as connection:
            fts_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(chunks_fts)"
                ).fetchall()
            }
            score_expression = (
                f"bm25(chunks_fts, {FTS_BM25_ARGUMENTS})"
                if "text" in fts_columns
                else "bm25(chunks_fts)"
            )
            rows = connection.execute(
                f"""
                SELECT
                    c.chunk_id,
                    {score_expression} AS lexical_score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                JOIN source_files sf ON sf.id = c.source_file_id
                JOIN sources s ON s.id = sf.source_id
                WHERE chunks_fts MATCH ?
                  AND c.generation = sf.active_generation
                  AND s.enabled = 1
                  {source_clause}
                ORDER BY lexical_score
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def begin_run(self, source_id: str) -> int:
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sync_runs(source_id, started_at, status)
                VALUES (?, ?, 'running')
                """,
                (source_id, now),
            )
            connection.execute(
                """
                UPDATE sources
                SET status = 'syncing', last_scan_at = ?, last_error = '',
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, source_id),
            )
            if cursor.lastrowid is None:  # pragma: no cover
                raise RuntimeError("sync run insert did not return an id")
            return cursor.lastrowid

    def finish_run(
        self,
        run_id: int,
        *,
        counters: dict[str, int],
        error: str = "",
    ) -> None:
        status = "failed" if error else "succeeded"
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT source_id FROM sync_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            now = utc_now()
            connection.execute(
                """
                UPDATE sync_runs
                SET finished_at = ?, status = ?, scanned_files = ?,
                    indexed_files = ?, reused_vectors = ?,
                    embedded_vectors = ?, missing_files = ?, error = ?
                WHERE id = ?
                """,
                (
                    now,
                    status,
                    counters.get("scanned_files", 0),
                    counters.get("indexed_files", 0),
                    counters.get("reused_vectors", 0),
                    counters.get("embedded_vectors", 0),
                    counters.get("missing_files", 0),
                    error,
                    run_id,
                ),
            )
            connection.execute(
                """
                UPDATE sources
                SET status = ?, last_success_at = CASE WHEN ? = ''
                    THEN ? ELSE last_success_at END,
                    last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "error" if error else "idle",
                    error,
                    now,
                    error,
                    now,
                    str(row[0]),
                ),
            )

    def _update_file(self, file_id: int, **values: object) -> None:
        if not values:
            return
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE source_files SET {assignments} WHERE id = ?",
                (*values.values(), file_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(file_id)

    @staticmethod
    def _source(row: sqlite3.Row) -> SourceRecord:
        columns = set(row.keys())
        return SourceRecord(
            id=str(row["id"]),
            display_name=str(row["display_name"]),
            absolute_path=str(row["absolute_path"]),
            enabled=bool(row["enabled"]),
            auto_sync=bool(row["auto_sync"]),
            include_patterns=tuple(json.loads(row["include_patterns_json"])),
            exclude_patterns=tuple(json.loads(row["exclude_patterns_json"])),
            consent_identity=str(row["consent_identity"]),
            status=str(row["status"]),
            last_scan_at=str(row["last_scan_at"]),
            last_success_at=str(row["last_success_at"]),
            last_error=str(row["last_error"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            scope_cleanup_pending=(
                bool(row["scope_cleanup_pending"])
                if "scope_cleanup_pending" in columns
                else False
            ),
        )

    @staticmethod
    def _source_file(row: sqlite3.Row) -> SourceFileRecord:
        columns = set(row.keys())
        pending = row["pending_generation"]
        return SourceFileRecord(
            id=int(row["id"]),
            source_id=str(row["source_id"]),
            relative_path=str(row["relative_path"]),
            mtime_ns=int(row["mtime_ns"]),
            size=int(row["size"]),
            sha256=str(row["sha256"]),
            status=str(row["status"]),
            missing_count=int(row["missing_count"]),
            active_generation=int(row["active_generation"]),
            pending_generation=int(pending) if pending is not None else None,
            last_error=str(row["last_error"]),
            rechunk_required=(
                bool(row["rechunk_required"])
                if "rechunk_required" in columns
                else False
            ),
        )
