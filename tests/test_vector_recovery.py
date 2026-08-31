from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from keepygaga_rag.knowledge import indexer_cli
from keepygaga_rag.knowledge.db import ChunkRecord, KnowledgeDB
from keepygaga_rag.knowledge.indexer import (
    IndexerLockUnavailable,
    KnowledgeIndexer,
)
from keepygaga_rag.knowledge.indexer_cli import repair_vector_store, run_once
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.vectors import LanceVectorStore


class RecordingVectors:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.history_calls: list[tuple[list[str], list[str]]] = []

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self.deleted.extend(chunk_ids)

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
        self.history_calls.append((list(chunk_ids), list(table_names)))
        self.delete(chunk_ids)

    def ensure_table(self) -> None:
        return

    def add(self, rows: Sequence[dict[str, object]]) -> None:
        return

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        return {}

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        return []


class UnusedEmbedding:
    identity = "test-embedding"
    dimensions = 1

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("recovery must not embed text")


class RebuildEmbedding:
    identity = "repair-embedding"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def _chunk(
    file_id: int,
    generation: int,
    text: str,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk:{generation}:0",
        source_file_id=file_id,
        generation=generation,
        ordinal=0,
        text=text,
        search_text=text,
        heading_path="",
        title="Note",
        content_hash=f"content-{generation}",
        embedding_input_hash=f"embedding-{generation}",
    )


def test_recovery_removes_vectors_left_after_generation_activation_crash(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    database.register_table(
        table_id="text_chunks_v1",
        lance_table="text_chunks_v1_current",
        embedding_identity="embedding",
        rerank_identity="rerank",
        retrieval_limits={},
    )
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=0,
        size=1,
    )
    first = _chunk(file.id, 1, "first")
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=1,
        chunks=[first],
    )
    database.activate_generation(file.id, 1, "first")

    second = _chunk(file.id, 2, "second")
    database.stage_generation(
        file_id=file.id,
        generation=2,
        mtime_ns=1,
        size=2,
        chunks=[second],
    )
    old_ids = database.activate_generation(file.id, 2, "second")
    assert old_ids == [first.chunk_id]
    assert database.pending_vector_deletions() == [first.chunk_id]

    vectors = RecordingVectors()
    indexer = KnowledgeIndexer(
        database=database,
        vectors=vectors,
        embedding=UnusedEmbedding(),
        table_id="text_chunks_v1",
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )

    indexer.reconcile_derived_state()

    assert vectors.deleted == [first.chunk_id]
    assert vectors.history_calls == [
        ([first.chunk_id], ["text_chunks_v1_current"])
    ]
    assert database.pending_vector_deletions() == []
    assert database.source_stats(source.id)["chunks"] == 1
    assert [
        str(chunk["chunk_id"])
        for chunk in database.active_chunks([first.chunk_id, second.chunk_id])
    ] == [second.chunk_id]


def test_startup_recovery_marks_interrupted_sync_runs_failed(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    run_id = database.begin_run(source.id)

    assert database.recover_interrupted_jobs() == 0

    with database.connect() as connection:
        row = connection.execute(
            "SELECT status, finished_at, error FROM sync_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "failed"
    assert row[1]
    assert row[2] == "indexer interrupted before sync run completed"
    recovered_source = database.get_source(source.id)
    assert recovered_source is not None
    assert recovered_source.status == "error"
    assert recovered_source.last_error == row[2]


def test_startup_recovery_requeues_source_with_running_job_and_run(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    job_id = database.queue_sync(source.id, "manual")
    claimed = database.claim_next_job()
    assert claimed is not None
    assert claimed[0] == job_id
    run_id = database.begin_run(source.id)

    assert database.recover_interrupted_jobs() == 1

    with database.connect() as connection:
        job = connection.execute(
            "SELECT status FROM sync_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        run = connection.execute(
            "SELECT status, error FROM sync_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert job is not None
    assert job[0] == "queued"
    assert run is not None
    assert run[0] == "failed"
    assert run[1] == "indexer interrupted before sync run completed"
    recovered_source = database.get_source(source.id)
    assert recovered_source is not None
    assert recovered_source.status == "queued"


def test_run_once_rebuilds_and_switches_a_broken_vector_table(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    table_id = "text_chunks_v1"
    good_table = f"{table_id}_good"
    broken_table = f"{table_id}_broken"
    embedding = RebuildEmbedding()
    embedding_space_id = hashlib.sha256(
        embedding.identity.encode("utf-8")
    ).hexdigest()
    database.register_table(
        table_id=table_id,
        lance_table=good_table,
        embedding_identity=embedding.identity,
        rerank_identity="repair-reranker",
        retrieval_limits={},
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "nested").mkdir()
    note = root / "nested" / "note.md"
    note.write_text("alpha body", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    good_vectors = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=good_table,
        dimensions=3,
        embedding_space_id=embedding_space_id,
    )
    indexer = KnowledgeIndexer(
        database=database,
        vectors=good_vectors,
        embedding=embedding,
        table_id=table_id,
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    indexer.sync_source(source)

    active = database.active_chunks_for_vector_rebuild()[0]
    broken_writer = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=broken_table,
        dimensions=2,
        embedding_space_id=embedding_space_id,
    )
    broken_writer.add(
        [
            {
                "chunk_id": str(active["chunk_id"]),
                "source_id": str(active["source_id"]),
                "source_file_id": int(active["source_file_id"]),
                "embedding_input_hash": str(active["embedding_input_hash"]),
                "embedding_space_id": embedding_space_id,
                "generation": int(active["generation"]),
                "vector": [1.0, 0.0],
            }
        ]
    )
    database.activate_vector_table(table_id, broken_table)
    broken_vectors = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=broken_table,
        dimensions=3,
        embedding_space_id=embedding_space_id,
    )
    indexer.vectors = broken_vectors
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(
            database=database,
            vectors=broken_vectors,
            embedding=embedding,
            store_root=tmp_path,
            table_id=table_id,
            consent_identity="consent",
        ),
    )

    run_once(runtime, indexer)

    repair_documents = [
        document
        for batch in embedding.calls
        for document in batch
        if "Filename: note.md" in document
    ]
    assert repair_documents[-1] == (
        "Title: note\n"
        "Filename: note.md\n"
        "Text:\nalpha body"
    )

    active_table = database.active_lance_table(
        table_id,
        embedding_identity=embedding.identity,
    )
    assert active_table is not None
    assert active_table not in {good_table, broken_table}
    repaired = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=active_table,
        dimensions=3,
        embedding_space_id=embedding_space_id,
        readonly=True,
    )
    assert repaired.get_vectors([str(active["chunk_id"])])


def test_health_check_repairs_missing_vector_for_active_chunk(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    table_id = "text_chunks_v1"
    current_table = f"{table_id}_current"
    embedding = RebuildEmbedding()
    database.register_table(
        table_id=table_id,
        lance_table=current_table,
        embedding_identity=embedding.identity,
        rerank_identity="repair-reranker",
        retrieval_limits={},
    )
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("alpha body", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    stat = note.stat()
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )
    chunk = _chunk(source_file.id, 1, "alpha body")
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        chunks=[chunk],
    )
    database.activate_generation(source_file.id, 1, "digest")
    vectors = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=current_table,
        dimensions=3,
        embedding_space_id=hashlib.sha256(
            embedding.identity.encode("utf-8")
        ).hexdigest(),
    )
    indexer = KnowledgeIndexer(
        database=database,
        vectors=vectors,
        embedding=embedding,
        table_id=table_id,
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(
            database=database,
            vectors=vectors,
            embedding=embedding,
            store_root=tmp_path,
            table_id=table_id,
            consent_identity="consent",
        ),
    )

    assert run_once(runtime, indexer) == 0

    active_table = database.active_lance_table(
        table_id,
        embedding_identity=embedding.identity,
    )
    assert active_table is not None
    assert active_table != current_table
    repaired = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=active_table,
        dimensions=3,
        embedding_space_id=hashlib.sha256(
            embedding.identity.encode("utf-8")
        ).hexdigest(),
        readonly=True,
    )
    assert repaired.get_vectors([chunk.chunk_id])


@pytest.mark.parametrize("field", ["embedding_space_id", "embedding_input_hash"])
def test_health_check_repairs_vector_with_invalid_identity_metadata(
    tmp_path: Path,
    field: str,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    table_id = "text_chunks_v1"
    current_table = f"{table_id}_current"
    embedding = RebuildEmbedding()
    embedding_space_id = hashlib.sha256(
        embedding.identity.encode("utf-8")
    ).hexdigest()
    database.register_table(
        table_id=table_id,
        lance_table=current_table,
        embedding_identity=embedding.identity,
        rerank_identity="repair-reranker",
        retrieval_limits={},
    )
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("alpha body", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    stat = note.stat()
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )
    chunk = _chunk(source_file.id, 1, "alpha body")
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        chunks=[chunk],
    )
    database.activate_generation(source_file.id, 1, "digest")
    vectors = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=current_table,
        dimensions=3,
        embedding_space_id=embedding_space_id,
    )
    invalid_row = {
        "chunk_id": chunk.chunk_id,
        "source_id": source.id,
        "source_file_id": source_file.id,
        "embedding_input_hash": chunk.embedding_input_hash,
        "embedding_space_id": embedding_space_id,
        "generation": 1,
        "vector": [1.0, 0.0, 0.0],
    }
    invalid_row[field] = "wrong"
    vectors.add([invalid_row])
    indexer = KnowledgeIndexer(
        database=database,
        vectors=vectors,
        embedding=embedding,
        table_id=table_id,
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(
            database=database,
            vectors=vectors,
            embedding=embedding,
            store_root=tmp_path,
            table_id=table_id,
            consent_identity="consent",
        ),
    )

    run_once(runtime, indexer)

    active_table = database.active_lance_table(
        table_id,
        embedding_identity=embedding.identity,
    )
    assert active_table is not None
    assert active_table != current_table
    repaired = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=active_table,
        dimensions=3,
        embedding_space_id=embedding_space_id,
        readonly=True,
    )
    metadata = repaired.get_vector_metadata([chunk.chunk_id])[chunk.chunk_id]
    assert metadata["embedding_space_id"] == embedding_space_id
    assert metadata["embedding_input_hash"] != "wrong"


def test_vector_repair_drops_new_table_when_source_scope_changes(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    table_id = "text_chunks_v1"
    current_table = f"{table_id}_current"
    database.register_table(
        table_id=table_id,
        lance_table=current_table,
        embedding_identity="repair-embedding",
        rerank_identity="repair-reranker",
        retrieval_limits={},
    )
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=0,
        size=1,
    )
    chunk = _chunk(file.id, 1, "alpha")
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=1,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "alpha")

    class ScopeChangingEmbedding(RebuildEmbedding):
        changed = False

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            if not self.changed:
                database.set_source_scope(
                    source.id,
                    include_patterns=["other.md"],
                    consent_identity="",
                )
                self.changed = True
            return super().embed(texts)

    embedding = ScopeChangingEmbedding()
    vectors = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=current_table,
        dimensions=3,
        embedding_space_id=hashlib.sha256(
            embedding.identity.encode("utf-8")
        ).hexdigest(),
    )
    vectors.ensure_table()
    indexer = KnowledgeIndexer(
        database=database,
        vectors=vectors,
        embedding=embedding,
        table_id=table_id,
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(
            database=database,
            vectors=vectors,
            embedding=embedding,
            store_root=tmp_path,
            table_id=table_id,
            consent_identity="consent",
            authorization_is_current=lambda: True,
        ),
    )

    with pytest.raises(RuntimeError, match="source authorization or scope"):
        repair_vector_store(runtime, indexer)

    assert database.active_lance_table(
        table_id,
        embedding_identity=embedding.identity,
    ) == current_table
    assert vectors.list_tables() == [current_table]


@pytest.mark.parametrize("pending_scope_cleanup", [True, False])
def test_vector_repair_only_activates_empty_table_for_pending_scope_cleanup(
    tmp_path: Path, pending_scope_cleanup: bool
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    table_id = "text_chunks_v1"
    current_table = f"{table_id}_current"
    database.register_table(
        table_id=table_id,
        lance_table=current_table,
        embedding_identity="repair-embedding",
        rerank_identity="repair-reranker",
        retrieval_limits={},
    )
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=0,
        size=1,
    )
    chunk = _chunk(source_file.id, 1, "alpha")
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=0,
        size=1,
        chunks=[chunk],
    )
    database.activate_generation(source_file.id, 1, "alpha")
    if pending_scope_cleanup:
        database.set_source_scope(
            source.id,
            include_patterns=["other.md"],
            consent_identity="",
        )
        database.set_source_consent(source.id, "consent")
    else:
        database.set_source_consent(source.id, "stale-consent")

    embedding = RebuildEmbedding()
    embedding_space_id = hashlib.sha256(
        embedding.identity.encode("utf-8")
    ).hexdigest()
    vectors = LanceVectorStore(
        tmp_path / "lancedb",
        table_name=current_table,
        dimensions=3,
        embedding_space_id=embedding_space_id,
    )
    vectors.ensure_table()
    indexer = KnowledgeIndexer(
        database=database,
        vectors=vectors,
        embedding=embedding,
        table_id=table_id,
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(
            database=database,
            vectors=vectors,
            embedding=embedding,
            store_root=tmp_path,
            table_id=table_id,
            consent_identity="consent",
            authorization_is_current=lambda: True,
        ),
    )

    assert repair_vector_store(runtime, indexer) is pending_scope_cleanup
    assert embedding.calls == []
    active_table = database.active_lance_table(
        table_id,
        embedding_identity=embedding.identity,
    )
    assert active_table is not None
    if pending_scope_cleanup:
        assert active_table != current_table
    else:
        assert active_table == current_table


def test_coordinator_retries_failed_reconciliation_and_clears_queue(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=0,
        size=1,
    )
    chunk = _chunk(file.id, 1, "stale")
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=1,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    database.delete_source_file(file.id, scope_cleanup=True)

    class FailingVectors(RecordingVectors):
        failures = 1

        def delete_across_history(
            self,
            chunk_ids: Sequence[str],
            *,
            table_names: Sequence[str],
        ) -> None:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("temporary vector failure")
            super().delete_across_history(
                chunk_ids, table_names=table_names
            )

    indexer = KnowledgeIndexer(
        database=database,
        vectors=FailingVectors(),
        embedding=UnusedEmbedding(),
        table_id="text_chunks_v1",
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )

    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )

    with pytest.raises(RuntimeError, match="temporary vector failure"):
        run_once(runtime, indexer)

    assert database.pending_vector_deletions() == [chunk.chunk_id]

    processed = run_once(runtime, indexer)

    assert processed == 0
    assert database.pending_vector_deletions() == []


def test_reconciled_scope_is_queued_and_processed_in_same_cycle(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    root = tmp_path / "source"
    root.mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    database.set_source_scope(
        source.id,
        include_patterns=["keep.md"],
        consent_identity="",
    )
    database.set_source_consent(source.id, "consent")
    indexer = KnowledgeIndexer(
        database=database,
        vectors=RecordingVectors(),
        embedding=UnusedEmbedding(),
        table_id="text_chunks_v1",
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )

    processed = run_once(runtime, indexer)

    assert processed == 1
    current = database.get_source(source.id)
    assert current is not None
    assert not current.scope_cleanup_pending
    with database.connect() as connection:
        job = connection.execute(
            "SELECT status, reason FROM sync_jobs WHERE source_id = ?",
            (source.id,),
        ).fetchone()
    assert job is not None
    assert tuple(job) == ("succeeded", "scope-cleanup-reconciled")


def test_run_once_reports_failed_sync_job(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    job_id = database.queue_sync(source.id, "manual")

    class FailingIndexer:
        def reconcile_chunk_settings(self) -> None:
            return None

        def reconcile_derived_state(self) -> int:
            return 0

        def sync_source(self, _source: object) -> dict[str, int]:
            raise RuntimeError("sync failed")

    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )

    with pytest.raises(RuntimeError, match="knowledge synchronization failed"):
        run_once(runtime, cast(KnowledgeIndexer, FailingIndexer()))

    with database.connect() as connection:
        job = connection.execute(
            "SELECT status, error FROM sync_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    assert job is not None
    assert job[0] == "failed"
    assert job[1] == "RuntimeError: sync failed"


def test_claimed_job_is_requeued_after_indexer_lock_failure(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
        auto_sync=False,
    )
    job_id = database.queue_sync(source.id, "manual")

    class LockThenSuccess:
        attempts = 0

        def reconcile_chunk_settings(self) -> None:
            return None

        def reconcile_derived_state(self) -> int:
            return 0

        def sync_source(self, _source: object) -> dict[str, int]:
            self.attempts += 1
            if self.attempts == 1:
                raise IndexerLockUnavailable("busy")
            return {}

    indexer = LockThenSuccess()
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )

    with pytest.raises(IndexerLockUnavailable):
        run_once(runtime, cast(KnowledgeIndexer, indexer))
    with database.connect() as connection:
        first = connection.execute(
            "SELECT status FROM sync_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert first is not None
    assert first["status"] == "queued"

    assert run_once(runtime, cast(KnowledgeIndexer, indexer)) == 1
    with database.connect() as connection:
        second = connection.execute(
            "SELECT status FROM sync_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert second is not None
    assert second["status"] == "succeeded"


def test_once_returns_nonzero_when_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / "store"
    config = SimpleNamespace(
        knowledge=SimpleNamespace(
            store=str(store_root),
            scan_interval_seconds=300,
        )
    )
    runtime = SimpleNamespace(
        database=SimpleNamespace(recover_interrupted_jobs=lambda: 0),
    )

    monkeypatch.setattr(sys, "argv", ["keepygaga-rag indexer", "--once"])
    monkeypatch.setattr(indexer_cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        indexer_cli,
        "resolve_knowledge_store",
        lambda _config, _path: store_root,
    )
    monkeypatch.setattr(
        indexer_cli.KnowledgeRuntime,
        "from_config",
        lambda *_args, **_kwargs: runtime,
    )
    monkeypatch.setattr(indexer_cli, "build_indexer", lambda _runtime: object())
    monkeypatch.setattr(
        indexer_cli,
        "queue_automatic_sources",
        lambda *_args, **_kwargs: 0,
    )

    def fail_once(*_args, **_kwargs) -> int:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(indexer_cli, "run_once", fail_once)

    assert indexer_cli.main() == 1
