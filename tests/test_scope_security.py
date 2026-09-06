from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from filelock import FileLock, Timeout

from keepygaga_rag.dashboard.knowledge_service import (
    KnowledgeDashboardError,
    KnowledgeDashboardService,
)
from keepygaga_rag.knowledge.authorization import AuthorizationGuard
from keepygaga_rag.knowledge.db import (
    ChunkRecord,
    KnowledgeDB,
    SourceRecord,
    timestamp_at_or_before,
)
from keepygaga_rag.knowledge.indexer import KnowledgeIndexer
from keepygaga_rag.knowledge.indexer_cli import queue_automatic_sources
from keepygaga_rag.knowledge.retry import (
    automatic_retry_is_due,
    provider_action_retry_delay_seconds,
)
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.searcher import KnowledgeSearcher


class FailingVectors:
    def delete(self, _chunk_ids: Sequence[str]) -> None:
        raise RuntimeError("vector delete failed")

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
        self.delete(chunk_ids)


class RecordingVectors:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self.deleted.extend(chunk_ids)

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
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


class EmptyVectors:
    def ensure_table(self) -> None:
        return

    def add(self, rows: Sequence[dict[str, object]]) -> None:
        return

    def delete(self, chunk_ids: Sequence[str]) -> None:
        return

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
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


class TestEmbedding:
    identity = "test-embedding"
    dimensions = 1

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0]]


class TrackingReranker:
    identity = "test-reranker"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        self.calls.append(list(documents))
        return []


def test_scope_cleanup_failure_stays_pending_and_blocks_renewal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    root = tmp_path / "source"
    root.mkdir()
    (root / "keep.md").write_text("keep", encoding="utf-8")
    (root / "drop.md").write_text("drop", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="drop.md",
        mtime_ns=0,
        size=0,
    )
    chunk = ChunkRecord(
        chunk_id="drop:1:0",
        source_file_id=file.id,
        generation=1,
        ordinal=0,
        text="drop",
        search_text="drop",
        heading_path="",
        title="Drop",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=0,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    runtime = SimpleNamespace(
        database=database,
        vectors=FailingVectors(),
        store_root=tmp_path,
        authorization_guard=AuthorizationGuard(tmp_path / "indexer.lock"),
        consent_identity="consent",
        table_id="text_chunks_v1",
        config=SimpleNamespace(
            knowledge=SimpleNamespace(
                include_patterns=("**/*.md",),
                exclude_patterns=(),
            )
        ),
    )
    monkeypatch.setattr(
        "keepygaga_rag.dashboard.knowledge_service.KnowledgeRuntime.load",
        lambda _path: runtime,
    )
    service = KnowledgeDashboardService(tmp_path / "keepygaga-rag.toml")

    with pytest.raises(RuntimeError, match="vector delete failed"):
        service.update_source_scope(source.id, ["keep.md"])

    pending = database.get_source(source.id)
    assert pending is not None
    assert pending.scope_cleanup_pending
    assert pending.consent_identity == ""
    assert database.pending_vector_deletions() == ["drop:1:0"]
    database.set_source_enabled(source.id, False)
    with pytest.raises(KnowledgeDashboardError, match="清理尚未完成"):
        service.renew_consent(source.id, consent=True)

    recovering_vectors = RecordingVectors()
    indexer = KnowledgeIndexer(
        database=database,
        vectors=recovering_vectors,
        embedding=TestEmbedding(),
        table_id="text_chunks_v1",
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="consent",
    )
    indexer.reconcile_derived_state()

    recovered = database.get_source(source.id)
    assert recovered is not None
    assert not recovered.scope_cleanup_pending
    assert recovering_vectors.deleted == ["drop:1:0"]
    service.renew_consent(source.id, consent=True)


def test_search_drops_out_of_scope_fts_chunk_before_reranking(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    (tmp_path / "source").mkdir()
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="drop.md",
        mtime_ns=0,
        size=6,
    )
    chunk = ChunkRecord(
        chunk_id="drop:1:0",
        source_file_id=file.id,
        generation=1,
        ordinal=0,
        text="secret out of scope",
        search_text="secret out of scope",
        heading_path="",
        title="Drop",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=6,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    database.set_source_scope(
        source.id,
        include_patterns=["keep.md"],
        consent_identity="",
    )
    database.set_source_consent(source.id, "consent")
    reranker = TrackingReranker()
    searcher = KnowledgeSearcher(
        database=database,
        vectors=EmptyVectors(),
        embedding=TestEmbedding(),
        reranker=reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    )

    pending = searcher.search(query="secret")

    assert pending["status"] == "consent_required"
    assert reranker.calls == []
    database.complete_source_scope_cleanup(source.id)
    result = searcher.search(query="secret")

    assert result["status"] == "no_results"
    assert reranker.calls == []


def test_search_rejects_legacy_hard_excluded_source_root(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Legacy sensitive root",
        absolute_path=str(tmp_path / "agents-memory"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="secret.md",
        mtime_ns=0,
        size=6,
    )
    chunk = ChunkRecord(
        chunk_id="legacy-secret:1:0",
        source_file_id=file.id,
        generation=1,
        ordinal=0,
        text="secret body",
        search_text="secret body",
        heading_path="",
        title="Secret",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=6,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    reranker = TrackingReranker()
    result = KnowledgeSearcher(
        database=database,
        vectors=EmptyVectors(),
        embedding=TestEmbedding(),
        reranker=reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    ).search(query="secret")

    assert result["status"] == "no_results"
    assert reranker.calls == []


def test_scope_cleanup_completion_keeps_disabled_source_paused(
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
    database.set_source_enabled(source.id, False)
    database.set_source_scope(
        source.id,
        include_patterns=["keep.md"],
        consent_identity="",
    )

    database.complete_source_scope_cleanup(source.id)

    completed = database.get_source(source.id)
    assert completed is not None
    assert completed.status == "paused"
    assert not completed.scope_cleanup_pending


def test_database_sync_gate_defends_pending_scope_cleanup(tmp_path: Path) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    database.set_source_scope(
        source.id,
        include_patterns=["keep.md"],
        consent_identity="",
    )

    with pytest.raises(ValueError, match="cleanup is pending"):
        database.queue_sync(source.id, "manual")
    with pytest.raises(ValueError, match="cleanup is pending"):
        database.renew_source_consent(
            source.id, "consent", reason="renewed"
        )
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO sync_jobs(source_id, requested_at, status, reason)
            VALUES (?, 'now', 'queued', 'race')
            """,
            (source.id,),
        )
    assert database.claim_next_job() is None
    current = database.get_source(source.id)
    assert current is not None
    assert current.consent_identity == ""


def test_due_queue_preserves_job_when_cleanup_becomes_pending(
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
    job_id = database.queue_sync(source.id, "startup")
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE sources SET scope_cleanup_pending = 1
            WHERE id = ?
            """,
            (source.id,),
        )

    queued = database.queue_due_sources(
        before="9999-12-31T23:59:59+00:00",
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )
    startup_queued = queue_automatic_sources(runtime, reason="startup")

    assert queued == 0
    assert startup_queued == 0
    with database.connect() as connection:
        job = connection.execute(
            "SELECT status FROM sync_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert job is not None
    assert job["status"] == "queued"
    current = database.get_source(source.id)
    assert current is not None
    assert current.status == "queued"


def test_queue_sync_atomically_rejects_changed_consent(tmp_path: Path) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="new-consent",
    )

    with pytest.raises(ValueError, match="consent changed"):
        database.queue_sync(
            source.id,
            "manual",
            expected_consent_identity="old-consent",
        )

    with database.connect() as connection:
        jobs = connection.execute(
            "SELECT COUNT(*) FROM sync_jobs WHERE source_id = ?", (source.id,)
        ).fetchone()
    assert jobs is not None
    assert jobs[0] == 0


def test_queue_sync_atomically_rejects_disabled_source(
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
    database.set_source_enabled(source.id, False)

    with pytest.raises(ValueError, match="disabled"):
        database.queue_sync(
            source.id,
            "manual",
            expected_consent_identity="consent",
        )

    with database.connect() as connection:
        jobs = connection.execute(
            "SELECT COUNT(*) FROM sync_jobs WHERE source_id = ?", (source.id,)
        ).fetchone()
    assert jobs is not None
    assert jobs[0] == 0


@pytest.mark.parametrize("state", ["disabled", "cleanup"])
@pytest.mark.parametrize("job_status", ["queued", "running"])
def test_recovery_fails_open_task_for_unconsumable_source(
    tmp_path: Path,
    state: str,
    job_status: str,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    job_id = database.queue_sync(source.id, "manual")
    if job_status == "running":
        claimed = database.claim_next_job()
        assert claimed is not None
    if state == "disabled":
        database.set_source_enabled(source.id, False)
    else:
        with database.transaction() as connection:
            connection.execute(
                "UPDATE sources SET scope_cleanup_pending = 1 WHERE id = ?",
                (source.id,),
            )
    if job_status == "queued":
        assert database.claim_next_job() is None

    assert database.recover_interrupted_jobs() == 0

    with database.connect() as connection:
        job = connection.execute(
            "SELECT status, error FROM sync_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert job is not None
    assert job[0] == "failed"
    assert state in str(job[1])
    recovered = database.get_source(source.id)
    assert recovered is not None
    assert recovered.status == ("paused" if state == "disabled" else "error")


def test_search_rechecks_scope_before_sending_documents_to_reranker(
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
        relative_path="secret.md",
        mtime_ns=0,
        size=6,
    )
    chunk = ChunkRecord(
        chunk_id="secret:1:0",
        source_file_id=file.id,
        generation=1,
        ordinal=0,
        text="secret body",
        search_text="secret body",
        heading_path="",
        title="Secret",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=6,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")

    class ScopeChangingEmbedding(TestEmbedding):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            database.set_source_scope(
                source.id,
                include_patterns=["keep.md"],
                consent_identity="",
            )
            return super().embed(texts)

    reranker = TrackingReranker()
    searcher = KnowledgeSearcher(
        database=database,
        vectors=EmptyVectors(),
        embedding=ScopeChangingEmbedding(),
        reranker=reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    )

    result = searcher.search(query="secret")

    assert result["status"] == "no_results"
    assert reranker.calls == []


def test_readonly_search_rechecks_consent_without_creating_lock_file(
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
        relative_path="secret.md",
        mtime_ns=0,
        size=6,
    )
    chunk = ChunkRecord(
        chunk_id="secret:1:0",
        source_file_id=file.id,
        generation=1,
        ordinal=0,
        text="secret body",
        search_text="secret body",
        heading_path="",
        title="Secret",
        content_hash="content",
        embedding_input_hash="embedding",
    )
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=6,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    guard_path = tmp_path / "indexer.lock"
    AuthorizationGuard(guard_path).ensure_writable()
    guard_before = guard_path.read_bytes()

    class ConsentRevokingEmbedding(TestEmbedding):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            database.set_source_consent(source.id, "")
            return super().embed(texts)

    reranker = TrackingReranker()
    searcher = KnowledgeSearcher(
        database=KnowledgeDB(database.path, readonly=True),
        vectors=EmptyVectors(),
        embedding=ConsentRevokingEmbedding(),
        reranker=reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    )

    result = searcher.search(query="secret")

    assert result["status"] == "no_results"
    assert reranker.calls == []
    current = database.get_source(source.id)
    assert current is not None
    assert not current.scope_cleanup_pending
    assert guard_path.exists()
    assert guard_path.read_bytes() == guard_before


def test_readonly_policy_failure_is_closed_but_reranker_failure_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    (tmp_path / "source").mkdir()
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
        size=4,
    )
    chunk = ChunkRecord(
        chunk_id="note:1:0",
        source_file_id=file.id,
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
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=4,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    AuthorizationGuard(tmp_path / "indexer.lock").ensure_writable()
    readonly = KnowledgeDB(database.path, readonly=True)
    original_list_sources = readonly.list_sources
    calls = 0

    def fail_final_policy_read():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("policy unavailable")
        return original_list_sources()

    monkeypatch.setattr(readonly, "list_sources", fail_final_policy_read)
    blocked_reranker = TrackingReranker()
    blocked = KnowledgeSearcher(
        database=readonly,
        vectors=EmptyVectors(),
        embedding=TestEmbedding(),
        reranker=blocked_reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    ).search(query="alpha")

    assert blocked["status"] == "no_results"
    assert blocked_reranker.calls == []

    class FailingReranker(TrackingReranker):
        def rerank(
            self,
            query: str,
            documents: Sequence[str],
            *,
            top_n: int,
        ) -> list[tuple[int, float]]:
            self.calls.append(list(documents))
            raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(readonly, "list_sources", original_list_sources)
    failing_reranker = FailingReranker()
    fallback = KnowledgeSearcher(
        database=readonly,
        vectors=EmptyVectors(),
        embedding=TestEmbedding(),
        reranker=failing_reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    ).search(query="alpha")

    assert fallback["status"] == "ok"
    groups = fallback["groups"]
    assert isinstance(groups, list)
    results = groups[0]["results"]
    assert set(results[0]) == {"source", "heading_path", "text", "score", "start_line", "end_line"}
    assert failing_reranker.calls == [[
        "Title: Note\nFilename: note.md\nText:\nalpha body"
    ]]
    assert (tmp_path / "indexer.lock").exists()

    class InvalidReranker(TrackingReranker):
        def rerank(
            self,
            query: str,
            documents: Sequence[str],
            *,
            top_n: int,
        ) -> list[tuple[int, float]]:
            self.calls.append(list(documents))
            return [(len(documents) + 1, 1.0)]

    invalid_reranker = InvalidReranker()
    invalid = KnowledgeSearcher(
        database=readonly,
        vectors=EmptyVectors(),
        embedding=TestEmbedding(),
        reranker=invalid_reranker,
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    ).search(query="alpha")

    assert invalid["status"] == "ok"
    invalid_groups = invalid["groups"]
    assert isinstance(invalid_groups, list)
    assert set(invalid_groups[0]["results"][0]) == set(results[0])
    invalid_warnings = invalid["warnings"]
    assert isinstance(invalid_warnings, list)
    assert any(
        "no valid result indexes" in str(warning)
        for warning in invalid_warnings
    )


def test_provider_callback_observes_readonly_authorization_guard(
    tmp_path: Path,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    (tmp_path / "source").mkdir()
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
        size=4,
    )
    chunk = ChunkRecord(
        chunk_id="note:1:0",
        source_file_id=file.id,
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
        file_id=file.id,
        generation=1,
        mtime_ns=0,
        size=4,
        chunks=[chunk],
    )
    database.activate_generation(file.id, 1, "digest")
    guard_path = tmp_path / "indexer.lock"
    AuthorizationGuard(guard_path).ensure_writable()
    observed = []
    database_before = database.path.read_bytes()
    guard_before = guard_path.read_bytes()

    class LockCheckingEmbedding(TestEmbedding):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            with pytest.raises(Timeout), FileLock(guard_path).acquire(timeout=0):
                pass
            observed.append(True)
            return super().embed(texts)

    result = KnowledgeSearcher(
        database=KnowledgeDB(database.path, readonly=True),
        vectors=EmptyVectors(),
        embedding=LockCheckingEmbedding(),
        reranker=TrackingReranker(),
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity="consent",
    ).search(query="alpha")

    assert result["status"] == "ok"
    assert observed == [True]
    assert database.path.read_bytes() == database_before
    assert guard_path.read_bytes() == guard_before


def test_automatic_queue_skips_source_deleted_during_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="consent",
    )
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )
    original_queue = database.queue_sync

    def delete_before_queue(
        source_id: str,
        reason: str,
        *,
        expected_consent_identity: str | None = None,
    ) -> int:
        database.remove_source(source.id)
        return original_queue(
            source_id,
            reason,
            expected_consent_identity=expected_consent_identity,
        )

    monkeypatch.setattr(database, "queue_sync", delete_before_queue)

    assert queue_automatic_sources(runtime, reason="startup") == 0


def test_provider_action_required_error_backs_off_automatic_but_not_manual_sync(
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
    last_scan_at = datetime(2026, 8, 26, 3, 0, tzinfo=UTC)
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE sources
            SET status = 'error', last_scan_at = ?, last_error = ?
            WHERE id = ?
            """,
            (
                last_scan_at.isoformat(),
                "ProviderActionRequiredError: embedding provider request failed "
                "with HTTP 402",
                source.id,
            ),
        )
    current = database.get_source(source.id)
    assert current is not None
    now = last_scan_at + timedelta(minutes=10)

    def source_predicate(item: SourceRecord) -> bool:
        return automatic_retry_is_due(
            item,
            regular_interval_seconds=300,
            now=now,
        )

    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity="consent"),
    )

    assert not source_predicate(current)
    assert (
        database.queue_due_sources(
            before=now.isoformat(),
            consent_identity="consent",
            source_predicate=source_predicate,
        )
        == 0
    )
    assert (
        queue_automatic_sources(
            runtime,
            reason="startup",
            source_predicate=source_predicate,
        )
        == 0
    )
    assert database.queue_sync(source.id, "manual") > 0
    assert automatic_retry_is_due(
        current,
        regular_interval_seconds=300,
        now=last_scan_at + timedelta(hours=1),
    )


@pytest.mark.parametrize(
    "error",
    [
        "HTTPStatusError: 401 Unauthorized",
        "HTTPStatusError: 402 Payment Required",
        "HTTPStatusError: 403 Forbidden",
    ],
)
def test_legacy_provider_action_required_errors_receive_backoff(error: str) -> None:
    assert provider_action_retry_delay_seconds(error) == 3600


def test_retry_timestamps_normalize_offsets_and_legacy_naive_values(
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
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE sources
            SET status = 'error', last_scan_at = ?, last_error = ?
            WHERE id = ?
            """,
            (
                "2026-08-26T08:00:00",
                "HTTPStatusError: 403 Forbidden",
                source.id,
            ),
        )
    current = database.get_source(source.id)
    assert current is not None

    assert timestamp_at_or_before(
        "2026-08-26T08:00:00+08:00",
        "2026-08-26T00:10:00+00:00",
    )
    assert not automatic_retry_is_due(
        current,
        regular_interval_seconds=300,
        now=datetime(2026, 8, 26, 8, 10, tzinfo=UTC),
    )
    assert automatic_retry_is_due(
        current,
        regular_interval_seconds=300,
        now=datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
    )
