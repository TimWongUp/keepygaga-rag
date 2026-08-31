from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from keepygaga_rag.knowledge import indexer as indexer_module
from keepygaga_rag.knowledge.db import KnowledgeDB, SourceRecord
from keepygaga_rag.knowledge.indexer import (
    KnowledgeIndexer,
    SourceSafetyError,
    validate_source_root,
)


class RecordingEmbedding:
    identity = "test|embedding|v1"
    dimensions = 1

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] for _ in texts]


class MemoryVectors:
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


def build_indexer(
    tmp_path: Path,
) -> tuple[KnowledgeDB, RecordingEmbedding, KnowledgeIndexer, str]:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    embedding = RecordingEmbedding()
    consent = "test-consent"
    indexer = KnowledgeIndexer(
        database=database,
        vectors=MemoryVectors(),
        embedding=embedding,
        table_id="text_chunks_v1",
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity=consent,
    )
    return database, embedding, indexer, consent


def add_source(
    database: KnowledgeDB,
    root: Path,
    consent: str,
) -> SourceRecord:
    return database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )


def test_sync_rejects_registered_root_replaced_with_symlink(
    tmp_path: Path,
) -> None:
    database, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    source = add_source(database, root, consent)
    outside = tmp_path / "agents-memory"
    outside.mkdir()
    (outside / "secret.md").write_text("must not embed", encoding="utf-8")
    root.rename(tmp_path / "original-source")
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SourceSafetyError, match="symbolic link"):
        indexer.sync_source(source)

    assert embedding.calls == []


def test_incomplete_directory_scan_does_not_mark_existing_files_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("safe text", encoding="utf-8")
    source = add_source(database, root, consent)
    indexer.sync_source(source)

    def incomplete_walk(
        _root: Path,
        *,
        followlinks: bool,
        onerror,
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        del followlinks
        onerror(PermissionError(13, "permission denied", str(root / "nested")))
        yield str(root), [], []

    monkeypatch.setattr(indexer_module.os, "walk", incomplete_walk)

    with pytest.raises(SourceSafetyError, match="source scan failed"):
        indexer.sync_source(database.get_source(source.id) or source)

    record = database.list_source_files(source.id)[0]
    assert record.relative_path == "note.md"
    assert record.status == "indexed"
    assert record.missing_count == 0


def test_scan_skips_hidden_and_unselected_unreadable_entries_before_lstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "selected.md").write_text("selected", encoding="utf-8")
    hidden = root / ".hidden.md"
    hidden.write_text("hidden", encoding="utf-8")
    ignored = root / "ignored.md"
    ignored.write_text("ignored", encoding="utf-8")
    ignored_directory = root / "ignored-directory"
    ignored_directory.mkdir()
    (ignored_directory / "nested.md").write_text(
        "ignored", encoding="utf-8"
    )
    hard_excluded = root / ".git"
    hard_excluded.mkdir()
    (hard_excluded / "nested.md").write_text("ignored", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["selected.md"],
        exclude_patterns=["ignored-directory"],
        consent_identity=consent,
    )
    original_lstat = Path.lstat
    blocked = {hidden, ignored, ignored_directory, hard_excluded}

    def fail_for_ignored(path: Path):
        if path in blocked:
            raise PermissionError(13, "permission denied", str(path))
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_for_ignored)

    result = indexer.sync_source(source)

    assert result["scanned_files"] == 1
    assert [record.relative_path for record in database.list_source_files(source.id)] == [
        "selected.md"
    ]


def test_invalid_utf8_file_fails_with_a_locatable_error(
    tmp_path: Path,
) -> None:
    database, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "note.md").write_bytes(b"# title\n\xff")
    source = add_source(database, root, consent)

    with pytest.raises(RuntimeError, match=r"note\.md.*UnicodeDecodeError"):
        indexer.sync_source(source)

    record = database.list_source_files(source.id)[0]
    assert record.status == "error"
    assert "UnicodeDecodeError" in record.last_error
    failed_source = database.get_source(source.id)
    assert failed_source is not None
    assert "note.md" in failed_source.last_error
    assert database.source_stats(source.id)["chunks"] == 0


def test_sync_rejects_registered_root_redirected_by_ancestor_symlink(
    tmp_path: Path,
) -> None:
    database, embedding, indexer, consent = build_indexer(tmp_path)
    registered_parent = tmp_path / "registered-parent"
    registered_parent.mkdir()
    root = registered_parent / "source"
    root.mkdir()
    (root / "note.md").write_text("safe text", encoding="utf-8")
    source = add_source(database, root, consent)
    canonical_parent = tmp_path / "canonical-parent"
    registered_parent.rename(canonical_parent)
    registered_parent.symlink_to(canonical_parent, target_is_directory=True)

    with pytest.raises(SourceSafetyError, match="redirected"):
        indexer.sync_source(source)

    assert embedding.calls == []


def test_sync_rejects_root_replaced_with_symlink_after_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "note.md").write_text("safe text", encoding="utf-8")
    source = add_source(database, root, consent)
    outside = tmp_path / "agents-memory"
    outside.mkdir()
    (outside / "secret.md").write_text("must not embed", encoding="utf-8")
    original_discover = indexer_module.discover_files

    def replace_root_after_discovery(
        discovered_source: SourceRecord,
        *,
        expected_root_stat: os.stat_result | None = None,
    ) -> Iterator[indexer_module.DiscoveredFile]:
        for item in original_discover(
            discovered_source,
            expected_root_stat=expected_root_stat,
        ):
            yield item
            root.rename(tmp_path / "original-source")
            root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        indexer_module,
        "discover_files",
        replace_root_after_discovery,
    )

    with pytest.raises(SourceSafetyError, match="symbolic link"):
        indexer.sync_source(source)

    assert embedding.calls == []


def test_file_replaced_with_symlink_after_discovery_is_not_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("safe text", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("must not embed", encoding="utf-8")
    source = add_source(database, root, consent)
    original_discover = indexer_module.discover_files

    def replace_after_discovery(
        discovered_source: SourceRecord,
        *,
        expected_root_stat: os.stat_result | None = None,
    ) -> Iterator[indexer_module.DiscoveredFile]:
        for item in original_discover(
            discovered_source,
            expected_root_stat=expected_root_stat,
        ):
            yield item
            note.unlink()
            note.symlink_to(outside)

    monkeypatch.setattr(indexer_module, "discover_files", replace_after_discovery)

    with pytest.raises(RuntimeError, match="source file"):
        indexer.sync_source(source)

    assert embedding.calls == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires FIFO support")
def test_fifo_is_ignored_without_opening_or_embedding(tmp_path: Path) -> None:
    database, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    os.mkfifo(root / "blocked.md")
    source = add_source(database, root, consent)

    result = indexer.sync_source(source)

    assert result["scanned_files"] == 0
    assert embedding.calls == []


@pytest.mark.parametrize("name", [".git", ".venv", ".keepygaga", "node_modules"])
def test_hard_excluded_directory_cannot_be_source_root(
    tmp_path: Path,
    name: str,
) -> None:
    root = tmp_path / name
    root.mkdir()
    (root / "secret.md").write_text("must not embed", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot include"):
        validate_source_root(root)

    database, embedding, indexer, consent = build_indexer(tmp_path)
    source = add_source(database, root, consent)
    with pytest.raises(SourceSafetyError, match="hard-excluded"):
        indexer.sync_source(source)
    assert embedding.calls == []


def test_claimed_source_paused_before_sync_is_not_embedded(
    tmp_path: Path,
) -> None:
    database, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "note.md").write_text("secret", encoding="utf-8")
    source = add_source(database, root, consent)
    job_id = database.queue_sync(source.id, "manual")
    claimed = database.claim_next_job()
    assert claimed is not None
    assert claimed[0] == job_id
    database.set_source_enabled(source.id, False)

    with pytest.raises(RuntimeError, match="consent is missing or stale"):
        indexer.sync_source(claimed[1])

    assert embedding.calls == []
