from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from keepygaga_rag.config import EmbeddingProfileConfig, RerankProfileConfig
from keepygaga_rag.knowledge.api import knowledge_search
from keepygaga_rag.knowledge.chunking import (
    chunk_text,
    fts_query,
    sha256_text,
    strip_frontmatter,
)
from keepygaga_rag.knowledge.db import ChunkRecord, KnowledgeDB
from keepygaga_rag.knowledge.indexer import (
    KnowledgeIndexer,
    _reported_sync_errors,
    path_matches_patterns,
    preview_source,
)
from keepygaga_rag.knowledge.indexer_cli import run_once
from keepygaga_rag.knowledge.providers import (
    CohereCompatibleRerankProvider,
    OpenAICompatibleEmbeddingProvider,
    ProviderActionRequiredError,
)
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.searcher import KnowledgeSearcher
from keepygaga_rag.knowledge.vectors import LanceVectorStore


class FakeEmbedding:
    identity = "fake|embedding|v1"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed")
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        values = [
            float(lowered.count("alpha")),
            float(lowered.count("beta")),
            max(1.0, len(text) / 100.0),
        ]
        length = math.sqrt(sum(value * value for value in values))
        return [value / length for value in values]


class FakeReranker:
    identity = "fake|reranker|v1"

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        token = query.casefold()
        ranked = sorted(
            enumerate(documents),
            key=lambda item: (-item[1].casefold().count(token), item[0]),
        )
        return [
            (index, 1.0 - position / 100)
            for position, (index, _) in enumerate(ranked[:top_n])
        ]


def test_frontmatter_strips_bom_and_crlf_markdown_delimiters(
    tmp_path: Path,
) -> None:
    value = "\ufeff---\r\nlayout: note\r\n---\r\n# Topic\r\n\r\nbody\r\n"

    assert strip_frontmatter(value) == "# Topic\n\nbody\n"
    chunks = chunk_text(
        value,
        source_path=tmp_path / "note.md",
        target_chars=100,
        max_chars=200,
        embedding_identity="test",
    )
    assert [chunk.text for chunk in chunks] == ["body"]
    assert chunks[0].heading_path == "Topic"


def test_structure_chunking_keeps_paragraphs_and_headings_separate(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "# Alpha\n\nalpha paragraph\n\nsecond paragraph\n\n## Beta\n\nbeta paragraph",
        source_path=tmp_path / "guide.md",
        target_chars=10,
        max_chars=40,
        embedding_identity="test",
    )

    assert [chunk.heading_path for chunk in chunks] == [
        "Alpha",
        "Alpha",
        "Alpha > Beta",
    ]
    assert [chunk.text for chunk in chunks] == [
        "alpha paragraph",
        "second paragraph",
        "beta paragraph",
    ]
    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert all(chunk.title == "Alpha" for chunk in chunks)
    assert all(chunk.filename == "guide.md" for chunk in chunks)
    assert "Filename: guide.md" in chunks[0].embedding_text
    assert "Heading path: Alpha" in chunks[0].embedding_text
    assert "Text:\nalpha paragraph" in chunks[0].embedding_text
    assert "guide.md" not in chunks[0].search_text


def test_structure_plain_text_fallback_uses_sentences_or_hard_limit(
    tmp_path: Path,
) -> None:
    sentence_chunks = chunk_text(
        "first sentence. second sentence! third sentence?",
        source_path=tmp_path / "plain.txt",
        target_chars=18,
        max_chars=40,
        embedding_identity="test",
    )
    hard_chunks = chunk_text(
        "abcdefghijabcdefghij",
        source_path=tmp_path / "plain.txt",
        target_chars=5,
        max_chars=10,
        embedding_identity="test",
    )

    assert [chunk.heading_path for chunk in sentence_chunks] == ["", ""]
    assert all(chunk.title == "plain" for chunk in sentence_chunks)
    assert [chunk.text for chunk in sentence_chunks] == [
        "first sentence. second sentence!",
        " third sentence?",
    ]
    assert [chunk.text for chunk in hard_chunks] == [
        "abcdefghij",
        "abcdefghij",
    ]


def test_embedding_input_hash_includes_title_heading_and_filename(
    tmp_path: Path,
) -> None:
    base = chunk_text(
        "# Title A\n\n## Section A\n\nbody",
        source_path=tmp_path / "one.md",
        target_chars=100,
        max_chars=200,
        embedding_identity="test",
    )[0]
    renamed = chunk_text(
        "# Title A\n\n## Section A\n\nbody",
        source_path=tmp_path / "two.md",
        target_chars=100,
        max_chars=200,
        embedding_identity="test",
    )[0]
    reheaded = chunk_text(
        "# Title A\n\n## Section B\n\nbody",
        source_path=tmp_path / "one.md",
        target_chars=100,
        max_chars=200,
        embedding_identity="test",
    )[0]

    assert base.embedding_input_hash != renamed.embedding_input_hash
    assert base.embedding_input_hash != reheaded.embedding_input_hash


@pytest.mark.parametrize("chunk_mode", ["structure", "length"])
def test_plain_text_blocks_and_fallbacks_preserve_exact_source(
    tmp_path: Path,
    chunk_mode: str,
) -> None:
    with_blank_lines = chunk_text(
        "alpha paragraph\n\nbeta paragraph",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=40,
        chunk_mode=chunk_mode,
        embedding_identity="test",
    )
    without_blank_lines = chunk_text(
        "first line\nsecond line",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=40,
        chunk_mode=chunk_mode,
        embedding_identity="test",
    )
    without_punctuation = chunk_text(
        "plain text without punctuation",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=40,
        chunk_mode=chunk_mode,
        embedding_identity="test",
    )

    assert [chunk.text for chunk in with_blank_lines] == [
        "alpha paragraph",
        "beta paragraph",
    ]
    assert [chunk.text for chunk in without_blank_lines] == [
        "first line\nsecond line",
    ]
    assert [chunk.text for chunk in without_punctuation] == [
        "plain text without punctuation",
    ]
    assert all(chunk.title == "plain" for chunk in with_blank_lines)


@pytest.mark.parametrize("chunk_mode", ["structure", "length"])
def test_short_plain_punctuated_text_without_blank_lines_is_exact(
    tmp_path: Path,
    chunk_mode: str,
) -> None:
    chunks = chunk_text(
        "First sentence. Second sentence!",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=100,
        chunk_mode=chunk_mode,
        embedding_identity="test",
    )

    assert [chunk.text for chunk in chunks] == [
        "First sentence.",
        " Second sentence!",
    ]


def test_length_boundaries_ignore_decimals_urls_and_single_newlines(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "Version 3.14 and https://example.com/a.b stay together. Next sentence.",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=100,
        chunk_mode="length",
        embedding_identity="test",
    )
    newline_chunks = chunk_text(
        "first line\nsecond line without punctuation",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=100,
        chunk_mode="length",
        embedding_identity="test",
    )

    assert [chunk.text for chunk in chunks] == [
        "Version 3.14 and https://example.com/a.b stay together.",
        " Next sentence.",
    ]
    url_sentence_chunks = chunk_text(
        "See https://example.com. Next sentence.",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=100,
        chunk_mode="length",
        embedding_identity="test",
    )
    protected_chunks = chunk_text(
        "Version 3.14 and https://example.com/a.b?x=1.2 plus `code.value` stay. Next.",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=120,
        chunk_mode="length",
        embedding_identity="test",
    )
    cjk_url_chunks = chunk_text(
        "参见 https://example.com/a.b。下一句。",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=100,
        chunk_mode="length",
        embedding_identity="test",
    )
    assert [chunk.text for chunk in newline_chunks] == [
        "first line\nsecond line without punctuation",
    ]
    assert [chunk.text for chunk in cjk_url_chunks] == [
        "参见 https://example.com/a.b。",
        "下一句。",
    ]
    assert [chunk.text for chunk in url_sentence_chunks] == [
        "See https://example.com.",
        " Next sentence.",
    ]
    assert [chunk.text for chunk in protected_chunks] == [
        "Version 3.14 and https://example.com/a.b?x=1.2 plus `code.value` stay.",
        " Next.",
    ]


def test_length_boundaries_preserve_chinese_punctuation_and_source_spacing(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "第一句。第二句；第三句，第四句",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=40,
        chunk_mode="length",
        embedding_identity="test",
    )
    spaced = chunk_text(
        "English sentence.  Next sentence。",
        source_path=tmp_path / "plain.txt",
        target_chars=1,
        max_chars=40,
        chunk_mode="length",
        embedding_identity="test",
    )

    assert [chunk.text for chunk in chunks] == [
        "第一句。",
        "第二句；",
        "第三句，",
        "第四句",
    ]
    assert [chunk.text for chunk in spaced] == [
        "English sentence.",
        "  Next sentence。",
    ]
    assert all(
        chunk.content_hash == sha256_text(chunk.text) for chunk in chunks + spaced
    )


def test_length_keeps_fenced_code_newlines_as_one_semantic_unit(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "```python\nfirst line.\nsecond line.\n```\n\nAfter.",
        source_path=tmp_path / "code.md",
        target_chars=1,
        max_chars=100,
        chunk_mode="length",
        embedding_identity="test",
    )

    assert [chunk.text for chunk in chunks] == [
        "```python\nfirst line.\nsecond line.\n```",
        "After.",
    ]


@pytest.mark.parametrize("chunk_mode", ["structure", "length"])
def test_heading_boundary_closes_before_target_in_both_modes(
    tmp_path: Path,
    chunk_mode: str,
) -> None:
    chunks = chunk_text(
        "# Alpha\n\nalpha body\n\n## Beta\n\nbeta body",
        source_path=tmp_path / "guide.md",
        target_chars=100,
        max_chars=200,
        chunk_mode=chunk_mode,
        embedding_identity="test",
    )

    assert [chunk.heading_path for chunk in chunks] == [
        "Alpha",
        "Alpha > Beta",
    ]
    assert [chunk.text for chunk in chunks] == ["alpha body", "beta body"]


def test_structure_overlap_only_applies_inside_forced_paragraph_windows(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "# Alpha\n\nabcdefghijklmnopqrst\n\n# Beta\n\nshort",
        source_path=tmp_path / "guide.md",
        target_chars=5,
        max_chars=10,
        chunk_mode="structure",
        overlap_chars=3,
        embedding_identity="test",
    )

    assert [chunk.heading_path for chunk in chunks] == [
        "Alpha",
        "Alpha",
        "Alpha",
        "Beta",
    ]
    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrst",
        "short",
    ]
    assert [chunk.text[:3] for chunk in chunks[1:3]] == [
        chunk.text[-3:] for chunk in chunks[:2]
    ]
    assert not chunks[-1].text.startswith(chunks[-2].text[-3:])


def test_length_chunking_prefers_sentence_boundaries_and_overlaps_only_length_splits(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "# Topic\n\nAlpha phrase. Beta phrase. Gamma phrase.\n\nNatural paragraph.",
        source_path=tmp_path / "guide.md",
        target_chars=15,
        max_chars=25,
        chunk_mode="length",
        overlap_chars=4,
        embedding_identity="test",
    )

    assert [chunk.text for chunk in chunks] == [
        "Alpha phrase.",
        "ase. Beta phrase.",
        "ase. Gamma phrase.",
        "Natural paragraph.",
    ]
    assert all(len(chunk.text) <= 25 for chunk in chunks)
    assert chunks[1].text[:4] == chunks[0].text[-4:]
    assert chunks[2].text[:4] == chunks[1].text[-4:]
    assert not chunks[-1].text.startswith(chunks[-2].text[-4:])
    assert all(chunk.heading_path == "Topic" for chunk in chunks)


def test_oversized_semantic_unit_uses_hard_windows_with_overlap(
    tmp_path: Path,
) -> None:
    chunks = chunk_text(
        "abcdefghijklmnopqrstuv",
        source_path=tmp_path / "plain.txt",
        target_chars=5,
        max_chars=10,
        chunk_mode="length",
        overlap_chars=3,
        embedding_identity="test",
    )

    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuv",
    ]
    assert [chunk.text[:3] for chunk in chunks[1:]] == [
        chunk.text[-3:] for chunk in chunks[:-1]
    ]
    no_overlap = chunk_text(
        "abcdefghijklmnopqrstuv",
        source_path=tmp_path / "plain.txt",
        target_chars=5,
        max_chars=10,
        chunk_mode="length",
        embedding_identity="test",
    )
    assert [chunk.text for chunk in no_overlap] == [
        "abcdefghij",
        "klmnopqrst",
        "uv",
    ]
    assert "".join(chunk.text for chunk in no_overlap) == (
        "abcdefghijklmnopqrstuv"
    )


class FakeVectors:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.search_limits: list[int] = []

    def ensure_table(self) -> None:
        return

    def add(self, rows: Sequence[dict[str, object]]) -> None:
        for row in rows:
            self.rows[str(row["chunk_id"])] = dict(row)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self.rows.pop(chunk_id, None)

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
        self.delete(chunk_ids)

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        return {
            chunk_id: [
                float(value)
                for value in cast(Sequence[float], self.rows[chunk_id]["vector"])
            ]
            for chunk_id in chunk_ids
            if chunk_id in self.rows
        }

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        self.search_limits.append(limit)
        allowed = set(source_ids)
        rows: list[tuple[str, float]] = []
        for chunk_id, row in self.rows.items():
            if allowed and str(row["source_id"]) not in allowed:
                continue
            stored = [
                float(value)
                for value in cast(
                    Sequence[float], row["vector"]
                )
            ]
            similarity = sum(a * b for a, b in zip(vector, stored, strict=True))
            rows.append((chunk_id, 1.0 - similarity))
        rows.sort(key=lambda item: (item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "distance": distance}
            for chunk_id, distance in rows[:limit]
        ]


class ReopeningVectors(FakeVectors):
    def __init__(self, delegate: FakeVectors) -> None:
        super().__init__()
        self.rows = delegate.rows
        self.fail = True
        self.reopen_calls = 0

    def reopen(self) -> None:
        self.reopen_calls += 1
        self.fail = False

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        if self.fail:
            raise RuntimeError("stale vector handle")
        return super().search(
            vector,
            limit=limit,
            source_ids=source_ids,
        )


def build_indexer(
    tmp_path: Path,
) -> tuple[KnowledgeDB, FakeVectors, FakeEmbedding, KnowledgeIndexer, str]:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    vectors = FakeVectors()
    embedding = FakeEmbedding()
    consent = "accepted-provider-identity"
    indexer = KnowledgeIndexer(
        database=database,
        vectors=vectors,
        embedding=embedding,
        table_id="text_chunks_v1",
        lock_path=tmp_path / "indexer.lock",
        target_chars=45,
        max_chars=90,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity=consent,
    )
    return database, vectors, embedding, indexer, consent


def test_scoped_directory_patterns_match_all_markdown_depths_only() -> None:
    recursive_pattern = ("folder/**/*.md",)

    assert path_matches_patterns("folder/direct.md", recursive_pattern)
    assert path_matches_patterns("folder/first/note.md", recursive_pattern)
    assert path_matches_patterns("folder/first/second/deep.md", recursive_pattern)
    assert not path_matches_patterns("other/folder/first/second/deep.md", recursive_pattern)
    assert not path_matches_patterns("folder/first/second/deep.txt", recursive_pattern)
    assert path_matches_patterns(
        "folder/direct.md", ("folder/*.md",)
    )
    assert not path_matches_patterns(
        "folder/first/note.md", ("folder/*.md",)
    )
    assert not path_matches_patterns(
        "other/folder/direct.md", ("folder/*.md",)
    )
    assert path_matches_patterns("root.md", ("**/*.md",))
    assert path_matches_patterns("nested/root.md", ("**/*.md",))
    assert path_matches_patterns("folder/private/deep.md", ("folder/**/private/**",))


def test_indexer_selects_scoped_directory_recursively(tmp_path: Path) -> None:
    database, _, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    selected = root / "folder"
    (selected / "first" / "second").mkdir(parents=True)
    (selected / "first" / "private").mkdir()
    (root / "other").mkdir()
    (selected / "direct.md").write_text("direct", encoding="utf-8")
    (selected / "first" / "note.md").write_text("first", encoding="utf-8")
    (selected / "first" / "second" / "deep.md").write_text(
        "deep", encoding="utf-8"
    )
    (selected / "first" / "second" / "deep.txt").write_text(
        "deep text", encoding="utf-8"
    )
    (selected / "first" / "private" / "secret.md").write_text(
        "secret", encoding="utf-8"
    )
    (selected / "wrong.json").write_text("wrong", encoding="utf-8")
    (root / "other" / "outside.md").write_text("outside", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=[
            "folder/**/*.md",
            "folder/*.md",
            "folder/**/*.txt",
            "folder/*.txt",
        ],
        exclude_patterns=["folder/**/private/**"],
        consent_identity=consent,
    )

    result = indexer.sync_source(source)

    assert result["scanned_files"] == 4
    assert {
        record.relative_path for record in database.list_source_files(source.id)
    } == {
        "folder/direct.md",
        "folder/first/note.md",
        "folder/first/second/deep.md",
        "folder/first/second/deep.txt",
    }


def test_scoped_pattern_matching_avoids_recursive_call_stack_growth() -> None:
    pattern = "/".join(("**",) * 1_100 + ("note.md",))

    assert path_matches_patterns("note.md", (pattern,))


def test_indexer_generation_reuse_failure_and_missing_confirmation(
    tmp_path: Path,
) -> None:
    database, vectors, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    note = root / "notes.md"
    note.write_text(
        "# Alpha\n\nalpha " + "steady " * 18 + "\n\n## Beta\n\nbeta " + "old " * 18,
        encoding="utf-8",
    )
    private = root / "nested" / "Agents-Memory"
    private.mkdir(parents=True)
    (private / "secret.md").write_text("must never index", encoding="utf-8")
    (root / ".hidden.md").write_text("must stay hidden", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md", "**/*.txt"],
        exclude_patterns=["**/.*/**"],
        consent_identity=consent,
    )

    first = indexer.sync_source(source)
    assert first["scanned_files"] == 1
    assert first["indexed_files"] == 1
    assert first["embedded_vectors"] >= 2
    initial_embedding_count = sum(len(batch) for batch in embedding.calls)
    assert database.source_stats(source.id)["files"] == 1

    note.write_text(
        "# Alpha\n\nalpha "
        + "steady " * 18
        + "\n\n## Beta\n\nbeta "
        + "changed " * 18,
        encoding="utf-8",
    )
    second = indexer.sync_source(database.get_source(source.id) or source)
    assert second["reused_vectors"] >= 1
    assert sum(len(batch) for batch in embedding.calls) - initial_embedding_count < (
        second["reused_vectors"] + second["embedded_vectors"]
    )

    active_before_failure = {
        str(row["chunk_id"])
        for row in database.active_chunks(list(vectors.rows))
    }
    embedding.fail = True
    note.write_text("# Alpha\n\nalpha replacement", encoding="utf-8")
    try:
        indexer.sync_source(database.get_source(source.id) or source)
    except RuntimeError as exc:
        assert "embedding failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected indexing failure")
    active_after_failure = {
        str(row["chunk_id"])
        for row in database.active_chunks(list(vectors.rows))
    }
    assert active_after_failure == active_before_failure

    embedding.fail = False
    note.unlink()
    indexer.sync_source(database.get_source(source.id) or source)
    assert database.source_stats(source.id)["missing_pending"] == 1
    indexer.sync_source(database.get_source(source.id) or source)
    assert database.source_stats(source.id)["files"] == 0


def test_indexer_rechunks_unchanged_active_file_when_explicitly_marked(
    tmp_path: Path,
) -> None:
    database, _, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("alpha " + "body " * 30, encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    indexer.sync_source(source)
    before = database.list_source_files(source.id)[0]
    assert before.active_generation == 1

    result = database.prepare_chunk_rebuild(
        target_chars=20,
        max_chars=35,
        consent_identity=consent,
        safe_source_ids=[source.id],
        reason="chunk-settings-changed",
    )
    indexer.target_chars = 20
    indexer.max_chars = 35
    synced = indexer.sync_source(database.get_source(source.id) or source)

    after = database.list_source_files(source.id)[0]
    assert result["marked_files"] == 1
    assert synced["indexed_files"] == 1
    assert after.active_generation == 2
    assert not after.rechunk_required


def test_chunk_settings_job_is_consumed_with_new_chunk_limits(
    tmp_path: Path,
) -> None:
    database, _, _, indexer, consent = build_indexer(tmp_path)
    indexer.target_chars = 1_000
    indexer.max_chars = 1_200
    database.prepare_chunk_rebuild(
        target_chars=1_000,
        max_chars=1_200,
        consent_identity=consent,
        safe_source_ids=[],
        reason="chunk-settings-changed",
    )
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("alpha " * 120, encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    indexer.sync_source(source)
    before = database.list_source_files(source.id)[0]
    before_chunks = database.active_chunk_ids(before.id)
    assert len(before_chunks) == 1

    # Simulate a crash after keepygaga-rag.toml changed but before the Dashboard
    # could mark files or queue the matching SQLite job.
    indexer.chunk_settings_loader = lambda: (40, 60)
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity=consent),
    )

    processed = run_once(runtime, indexer)

    after = database.list_source_files(source.id)[0]
    active_ids = database.active_chunk_ids(after.id)
    active = [database.active_chunk(chunk_id) for chunk_id in active_ids]
    assert processed == 1
    assert after.active_generation == 2
    assert len(active_ids) > len(before_chunks)
    assert all(row is not None and len(str(row["text"])) <= 60 for row in active)
    with sqlite3.connect(database.path) as connection:
        job = connection.execute(
            "SELECT status, reason FROM sync_jobs WHERE source_id = ?",
            (source.id,),
        ).fetchone()
    assert job == ("succeeded", "chunk-settings-changed")


def test_long_running_indexer_applies_mode_and_overlap_to_real_markdown(
    tmp_path: Path,
) -> None:
    database, vectors, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text(
        "# Topic\n\nalpha phrase. beta phrase. gamma phrase.\n\nfinal",
        encoding="utf-8",
    )
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    database.prepare_chunk_rebuild(
        target_chars=15,
        max_chars=25,
        chunk_mode="length",
        overlap_chars=4,
        consent_identity=consent,
        safe_source_ids=[source.id],
        reason="chunk-settings-changed",
    )
    indexer.chunk_settings_loader = lambda: (15, 25, "length", 4)
    runtime = cast(
        KnowledgeRuntime,
        SimpleNamespace(database=database, consent_identity=consent),
    )

    assert run_once(runtime, indexer) == 1

    source_file = database.list_source_files(source.id)[0]
    active = [
        database.active_chunk(chunk_id)
        for chunk_id in database.active_chunk_ids(source_file.id)
    ]
    texts = [str(row["text"]) for row in active if row is not None]
    assert indexer.chunk_mode == "length"
    assert indexer.overlap_chars == 4
    assert texts == [
        "alpha phrase.",
        "ase. beta phrase.",
        "ase. gamma phrase.",
        "final",
    ]
    assert all(
        str(row["heading_path"]) == "Topic"
        for row in active
        if row is not None
    )
    assert texts[1][:4] == texts[0][-4:]
    assert texts[2][:4] == texts[1][-4:]
    assert not texts[-1].startswith(texts[-2][-4:])


@pytest.mark.parametrize("same_stat", [True, False])
def test_indexer_rebuilds_missing_vectors_after_embedding_table_rotation(
    tmp_path: Path,
    same_stat: bool,
) -> None:
    database = KnowledgeDB(tmp_path / "knowledge.sqlite3")
    table_id = "text_chunks_v1"
    current_table = "text_chunks_v1_current"
    history_table = "text_chunks_v1_history"
    database.register_table(
        table_id=table_id,
        lance_table=current_table,
        embedding_identity="fake|embedding|v2",
        rerank_identity="fake|reranker|v1",
        retrieval_limits={},
    )
    database.register_vector_table(table_id, history_table)
    root = tmp_path / "source"
    root.mkdir()
    note = root / "note.md"
    note.write_text("alpha body", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="accepted-provider-identity",
    )
    stat = note.stat()
    file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="note.md",
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )
    old_chunk = ChunkRecord(
        chunk_id=f"{file.id}:1:0",
        source_file_id=file.id,
        generation=1,
        ordinal=0,
        text="alpha body",
        search_text="alpha body",
        heading_path="",
        title="Note",
        content_hash="old-content",
        embedding_input_hash="old-embedding-input",
    )
    database.stage_generation(
        file_id=file.id,
        generation=1,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        chunks=[old_chunk],
    )
    database.activate_generation(
        file.id,
        1,
        sha256(note.read_bytes()).hexdigest(),
    )
    if not same_stat:
        database.mark_file_unchanged(
            file.id,
            mtime_ns=stat.st_mtime_ns - 1,
            size=stat.st_size,
        )

    vector_root = tmp_path / "lance"
    old_vectors = LanceVectorStore(
        vector_root,
        table_name=history_table,
        dimensions=3,
        embedding_space_id="old-space",
    )
    current_vectors = LanceVectorStore(
        vector_root,
        table_name=current_table,
        dimensions=3,
        embedding_space_id="new-space",
    )
    old_vectors.add(
        [
            {
                "chunk_id": old_chunk.chunk_id,
                "source_id": source.id,
                "source_file_id": file.id,
                "embedding_input_hash": old_chunk.embedding_input_hash,
                "embedding_space_id": "old-space",
                "generation": 1,
                "vector": [1.0, 0.0, 0.0],
            }
        ]
    )

    class RotatedEmbedding(FakeEmbedding):
        identity = "fake|embedding|v2"

    embedding = RotatedEmbedding()
    indexer = KnowledgeIndexer(
        database=database,
        vectors=current_vectors,
        embedding=embedding,
        table_id=table_id,
        lock_path=tmp_path / "indexer.lock",
        target_chars=100,
        max_chars=200,
        stability_seconds=0,
        missing_confirmations=2,
        consent_identity="accepted-provider-identity",
    )

    result = indexer.sync_source(source)

    assert result["indexed_files"] == 1
    assert result["embedded_vectors"] == 1
    assert embedding.calls
    assert current_vectors.get_vectors([f"{file.id}:2:0"])
    assert old_vectors.get_vectors([old_chunk.chunk_id]) == {}
    assert database.pending_vector_deletions() == []


def test_scope_change_removes_existing_file_without_missing_grace(
    tmp_path: Path,
) -> None:
    database, vectors, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "keep.md").write_text("alpha keep", encoding="utf-8")
    (root / "drop.md").write_text("beta drop", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    indexer.sync_source(source)
    assert database.source_stats(source.id)["files"] == 2
    database.set_source_scope(
        source.id,
        include_patterns=["keep.md"],
        consent_identity=consent,
    )
    updated = database.get_source(source.id)
    assert updated is not None
    result = indexer.sync_source(updated)
    assert result["missing_files"] == 0
    assert database.source_stats(source.id)["files"] == 1
    assert {
        record.relative_path for record in database.list_source_files(source.id)
    } == {"keep.md"}
    assert all(
        str(row["source_id"]) == source.id for row in vectors.rows.values()
    )


def test_preview_rejects_case_variant_of_hard_excluded_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_Context-Backups" / "notes"
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot include"):
        preview_source(
            root,
            include_patterns=["**/*.md"],
            exclude_patterns=[],
            target_chars=100,
        )


def test_hybrid_search_returns_grouped_traceable_results(tmp_path: Path) -> None:
    database, vectors, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "alpha.md").write_text(
        "# Alpha handbook\n\nalpha calibration procedure and traceable evidence",
        encoding="utf-8",
    )
    source = database.add_source(
        display_name="Lab notes",
        absolute_path=str(root),
        include_patterns=["**/*.md", "**/*.txt"],
        exclude_patterns=["**/.*/**"],
        consent_identity=consent,
    )
    indexer.sync_source(source)
    assert embedding.calls == [[
        "Title: Alpha handbook\n"
        "Heading path: Alpha handbook\n"
        "Filename: alpha.md\n"
        "Text:\nalpha calibration procedure and traceable evidence"
    ]]
    searcher = KnowledgeSearcher(
        database=database,
        vectors=vectors,
        embedding=embedding,
        reranker=FakeReranker(),
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity=consent,
    )
    result = searcher.search(query="alpha", top_k=3)
    assert result["status"] == "ok"
    groups = result["groups"]
    assert isinstance(groups, list)
    assert groups[0]["table"] == "text_chunks_v1"
    row = groups[0]["results"][0]
    assert set(row) == {"source", "heading_path", "text", "score"}
    assert row["source"].endswith("alpha.md")
    assert row["heading_path"] == "Alpha handbook"
    assert row["text"].startswith("alpha calibration")
    assert isinstance(row["score"], float)
    assert "diagnostics" not in result
    traced = searcher.search(
        query="alpha",
        top_k=3,
        include_diagnostics=True,
    )
    assert {
        key: value for key, value in traced.items() if key != "diagnostics"
    } == result
    diagnostics = cast(dict[str, object], traced["diagnostics"])
    stages = cast(list[dict[str, object]], diagnostics["stages"])
    assert [stage["id"] for stage in stages] == [
        "keyword",
        "vector",
        "fusion",
        "rerank",
    ]
    for stage in stages:
        elapsed_ms = stage["elapsed_ms"]
        count = stage["count"]
        assert isinstance(elapsed_ms, (int, float))
        assert isinstance(count, int)
        assert elapsed_ms >= 0
        assert count >= 1
    assert len(cast(list[dict[str, object]], diagnostics["final"])) == 1
    settings = cast(dict[str, object], diagnostics["settings"])
    assert settings["max_chunks_per_source_file"] == 2
    database.set_source_enabled(source.id, False)
    disabled = searcher.search(
        query="alpha", top_k=3, source_ids=[source.id]
    )
    assert disabled["status"] == "invalid_request"
    assert "disabled" in str(disabled["message"])
    database.set_source_enabled(source.id, True)
    database.set_source_consent(source.id, "changed-provider")
    embedding.fail = True
    blocked = searcher.search(
        query="alpha", top_k=3, source_ids=[source.id]
    )
    assert blocked["status"] == "consent_required"
    blocked_default = searcher.search(query="alpha", top_k=3)
    assert blocked_default["status"] == "consent_required"


def test_reranker_document_contains_all_retrieval_fields() -> None:
    document = KnowledgeSearcher._rerank_document(
        {
            "text": "正文内容",
            "title": "总标题",
            "heading_path": "总标题 > 子标题",
            "filename": "guide.md",
            "relative_path": "nested/guide.md",
        }
    )

    assert document == (
        "Title: 总标题\n"
        "Heading path: 总标题 > 子标题\n"
        "Filename: guide.md\n"
        "Text:\n正文内容"
    )


def test_fts_searches_text_title_heading_and_filename_fields(
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
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="target.md",
        mtime_ns=0,
        size=4,
    )
    chunk = ChunkRecord(
        chunk_id="target:1:0",
        source_file_id=source_file.id,
        generation=1,
        ordinal=0,
        text="正文 only",
        search_text="正文 only",
        heading_path="Heading only",
        title="Title only",
        content_hash="content",
        embedding_input_hash="embedding",
        filename="target.md",
    )
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=0,
        size=4,
        chunks=[chunk],
    )
    database.activate_generation(source_file.id, 1, "digest")

    for query in ("正文", "Title", "Heading", "target"):
        rows = database.fts_search(fts_query(query), limit=1)
        assert rows[0]["chunk_id"] == chunk.chunk_id


def test_fts_bm25_prefers_title_heading_filename_over_body(
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
    source_file = database.get_or_create_source_file(
        source_id=source.id,
        relative_path="rank.md",
        mtime_ns=0,
        size=4,
    )
    field_rows = (
        ("text", "needle", "context", "context", "text.md"),
        ("title", "body", "needle", "context", "title.md"),
        ("heading", "body", "context", "needle", "heading.md"),
        ("filename", "body", "context", "context", "needle.md"),
    )
    chunks = [
        ChunkRecord(
            chunk_id=f"{kind}:1:0",
            source_file_id=source_file.id,
            generation=1,
            ordinal=ordinal,
            text=text,
            search_text=text,
            heading_path=heading_path,
            title=title,
            content_hash=kind,
            embedding_input_hash=kind,
            filename=filename,
        )
        for ordinal, (kind, text, title, heading_path, filename) in enumerate(
            field_rows
        )
    ]
    database.stage_generation(
        file_id=source_file.id,
        generation=1,
        mtime_ns=0,
        size=4,
        chunks=chunks,
    )
    database.activate_generation(source_file.id, 1, "digest")

    rows = database.fts_search(fts_query("needle"), limit=4)

    assert [row["chunk_id"] for row in rows] == [
        "title:1:0",
        "heading:1:0",
        "filename:1:0",
        "text:1:0",
    ]


def test_indexer_populates_fts_fields_from_real_chunks(
    tmp_path: Path,
) -> None:
    database, _, _, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    note = root / "nested" / "field-note.md"
    note.write_text(
        "# Unique title\n\n## Unique heading\n\nbody only",
        encoding="utf-8",
    )
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )

    indexer.sync_source(source)

    for query in ("Unique title", "Unique heading", "field-note"):
        rows = database.fts_search(fts_query(query), limit=1)
        assert rows
        chunk = database.active_chunk(str(rows[0]["chunk_id"]))
        assert chunk is not None
        assert chunk["filename"] == "field-note.md"
        assert "nested" not in str(chunk["filename"])


def test_search_applies_recall_limits_and_per_note_chunk_cap(
    tmp_path: Path,
) -> None:
    database, vectors, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.md").write_text(
        "\n\n".join(
            [
                "alpha one paragraph with enough words to become a chunk.",
                "alpha two paragraph with enough words to become a chunk.",
                "alpha three paragraph with enough words to become a chunk.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "two.md").write_text(
        "alpha another note with enough words to become a chunk.",
        encoding="utf-8",
    )
    source = database.add_source(
        display_name="Lab notes",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    indexer.sync_source(source)
    searcher = KnowledgeSearcher(
        database=database,
        vectors=vectors,
        embedding=embedding,
        reranker=FakeReranker(),
        table_id="text_chunks_v1",
        consent_identity=consent,
        vector_recall_limit=7,
        keyword_recall_limit=6,
        rerank_candidate_limit=8,
        max_chunks_per_source_file=2,
    )

    result = searcher.search(query="alpha", top_k=5)

    assert result["status"] == "ok"
    groups = cast(list[dict[str, object]], result["groups"])
    rows = cast(list[dict[str, object]], groups[0]["results"])
    assert vectors.search_limits[-1] == 7
    counts: dict[str, int] = {}
    for row in rows:
        path = Path(str(row["source"])).name
        counts[path] = counts.get(path, 0) + 1
    assert counts.get("one.md", 0) <= 2
    assert counts.get("two.md", 0) >= 1


def test_search_reopens_vector_backend_once_after_vector_failure(
    tmp_path: Path,
) -> None:
    database, vectors, embedding, indexer, consent = build_indexer(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    (root / "note.md").write_text("alpha body", encoding="utf-8")
    source = database.add_source(
        display_name="Source",
        absolute_path=str(root),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    indexer.sync_source(source)
    flaky = ReopeningVectors(vectors)

    searcher = KnowledgeSearcher(
        database=database,
        vectors=flaky,
        embedding=embedding,
        reranker=FakeReranker(),
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity=consent,
    )

    result = searcher.search(query="alpha", top_k=1)

    assert result["status"] == "ok"
    assert flaky.reopen_calls == 1
    warnings = result["warnings"]
    assert isinstance(warnings, list)
    assert "vector recall recovered after reopening the backend" in warnings


def test_search_warns_when_some_enabled_sources_have_stale_consent(
    tmp_path: Path,
) -> None:
    database, vectors, embedding, _, consent = build_indexer(tmp_path)
    stale = database.add_source(
        display_name="Stale",
        absolute_path=str(tmp_path / "stale"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity="old-provider",
    )
    (tmp_path / "stale").mkdir()
    database.add_source(
        display_name="Current",
        absolute_path=str(tmp_path / "current"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    (tmp_path / "current").mkdir()
    embedding.fail = True
    searcher = KnowledgeSearcher(
        database=database,
        vectors=vectors,
        embedding=embedding,
        reranker=FakeReranker(),
        table_id="text_chunks_v1",
        candidate_limit=10,
        consent_identity=consent,
    )
    result = searcher.search(query="alpha")
    assert result["status"] == "no_results"
    warnings = result["warnings"]
    assert isinstance(warnings, list)
    assert any(stale.id in warning for warning in warnings)
    assert "vector recall unavailable: RuntimeError" in warnings
    assert all("embedding failed" not in warning for warning in warnings)


def test_failed_job_restores_source_status_from_queued(tmp_path: Path) -> None:
    database, _, _, _, consent = build_indexer(tmp_path)
    source = database.add_source(
        display_name="Source",
        absolute_path=str(tmp_path / "source"),
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        consent_identity=consent,
    )
    job_id = database.queue_sync(source.id, "manual")
    claimed = database.claim_next_job()
    assert claimed is not None
    database.finish_job(job_id, error="RuntimeError: lock unavailable")
    failed = database.get_source(source.id)
    assert failed is not None
    assert failed.status == "error"
    assert failed.last_error == "RuntimeError: lock unavailable"


def test_task_history_uses_bounded_id_windows_and_open_counts(
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
        for job_id in range(1, 14):
            status = "queued" if job_id == 1 else (
                "failed" if job_id % 2 else "succeeded"
            )
            connection.execute(
                """
                INSERT INTO sync_jobs(
                    id, source_id, requested_at, started_at, finished_at,
                    status, reason, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    source.id,
                    f"2026-08-05T00:{job_id:02d}:00+00:00",
                    "2026-08-05T00:00:00+00:00",
                    f"2020-01-01T00:{job_id:02d}:00+00:00",
                    status,
                    f"job-{job_id}",
                    "failed" if status == "failed" else "",
                ),
            )
        connection.execute(
            """
            INSERT INTO sync_runs(id, source_id, started_at, status)
            VALUES (1, ?, '2026-08-05T00:02:00+00:00', 'succeeded')
            """,
            (source.id,),
        )
        connection.execute(
            """
            INSERT INTO sync_runs(id, source_id, started_at, status)
            VALUES (2, ?, '2020-01-01T00:01:00+00:00', 'failed')
            """,
            (source.id,),
        )

    closed = database.list_sync_jobs(
        limit=3,
        statuses=("succeeded", "failed"),
    )
    combined = database.list_sync_jobs(limit=3)
    assert [int(job["id"]) for job in closed] == [13, 12, 11]
    assert [int(job["id"]) for job in combined] == [1, 13, 12]
    assert database.sync_job_counts() == {"queued": 1, "running": 0}
    assert int(database.list_sync_runs(limit=1)[0]["id"]) == 2


def test_knowledge_api_does_not_expose_runtime_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_config_path: object, *, readonly: bool = False) -> None:
        assert readonly
        raise RuntimeError("secret provider URL and local config path")

    monkeypatch.setattr(
        "keepygaga_rag.knowledge.api.KnowledgeRuntime.load",
        fail_load,
    )
    result = knowledge_search(query="alpha")
    assert result == {
        "status": "invalid_source",
        "message": "knowledge backend unavailable: RuntimeError",
        "groups": [],
    }


def test_lancedb_vector_store_round_trip(tmp_path: Path) -> None:
    store = LanceVectorStore(
        tmp_path / "lance",
        table_name="text_test",
        dimensions=3,
        embedding_space_id="space",
    )
    store.add(
        [
            {
                "chunk_id": "1:1:0",
                "source_id": "source",
                "source_file_id": 1,
                "embedding_input_hash": "hash",
                "embedding_space_id": "space",
                "generation": 1,
                "vector": [1.0, 0.0, 0.0],
            }
        ]
    )
    assert store.get_vectors(["1:1:0"]) == {"1:1:0": [1.0, 0.0, 0.0]}
    result = store.search(
        [1.0, 0.0, 0.0], limit=3, source_ids=["source"]
    )
    assert result == [{"chunk_id": "1:1:0", "distance": 0.0}]
    store.delete(["1:1:0"])
    assert store.get_vectors(["1:1:0"]) == {}


def test_lancedb_delete_cleans_matching_historical_embedding_tables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lance"
    old = LanceVectorStore(
        root,
        table_name="text_chunks_v1_oldspace",
        dimensions=3,
        embedding_space_id="oldspace",
    )
    current = LanceVectorStore(
        root,
        table_name="text_chunks_v1_newspace",
        dimensions=3,
        embedding_space_id="newspace",
    )
    old.add(
        [
            {
                "chunk_id": "stale:1:0",
                "source_id": "source",
                "source_file_id": 1,
                "embedding_input_hash": "old-hash",
                "embedding_space_id": "oldspace",
                "generation": 1,
                "vector": [1.0, 0.0, 0.0],
            }
        ]
    )
    current.add(
        [
            {
                "chunk_id": "stale:1:0",
                "source_id": "source",
                "source_file_id": 1,
                "embedding_input_hash": "current-hash",
                "embedding_space_id": "newspace",
                "generation": 1,
                "vector": [1.0, 0.0, 0.0],
            }
        ]
    )
    unrelated = LanceVectorStore(
        root,
        table_name="text_chunks_v1_unrelated",
        dimensions=3,
        embedding_space_id="unrelated",
    )
    unrelated.add(
        [
            {
                "chunk_id": "stale:1:0",
                "source_id": "other-source",
                "source_file_id": 2,
                "embedding_input_hash": "other-hash",
                "embedding_space_id": "unrelated",
                "generation": 1,
                "vector": [0.0, 1.0, 0.0],
            }
        ]
    )

    current.delete_across_history(
        ["stale:1:0"],
        table_names=["text_chunks_v1_oldspace", "text_chunks_v1_newspace"],
    )

    assert old.get_vectors(["stale:1:0"]) == {}
    assert current.get_vectors(["stale:1:0"]) == {}
    assert unrelated.get_vectors(["stale:1:0"]) != {}


def test_online_provider_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    class Response:
        def __init__(self, payload: dict[str, object]):
            self.payload = payload

        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return self.payload

    class Client:
        def __init__(self, **_kwargs: object):
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return

        def post(self, path: str, *, json: dict[str, object]) -> Response:
            requests.append((path, json))
            if path == "/embeddings":
                return Response(
                    {
                        "data": [
                            {"index": 0, "embedding": [3.0, 4.0, 0.0]}
                        ]
                    }
                )
            return Response(
                {"results": [{"index": 0, "relevance_score": 0.9}]}
            )

    monkeypatch.setattr("keepygaga_rag.knowledge.providers.httpx.Client", Client)
    monkeypatch.setenv("EMBED_KEY", "secret")
    monkeypatch.setenv("RERANK_KEY", "secret")
    embedding = OpenAICompatibleEmbeddingProvider(
        EmbeddingProfileConfig(
            provider="openai_compatible",
            base_url="https://example.test/v1",
            api_key_env="EMBED_KEY",
            model="embedding-model",
            dimensions=3,
        )
    )
    reranker = CohereCompatibleRerankProvider(
        RerankProfileConfig(
            provider="api",
            protocol="cohere_compatible",
            base_url="https://example.test/v1",
            api_key_env="RERANK_KEY",
            model="rerank-model",
        )
    )
    assert embedding.embed(["alpha"]) == [[0.6, 0.8, 0.0]]
    assert reranker.rerank("alpha", ["document"], top_n=1) == [(0, 0.9)]
    assert requests[0] == (
        "/embeddings",
        {
            "model": "embedding-model",
            "input": ["alpha"],
            "dimensions": 3,
        },
    )
    assert requests[1][0] == "/rerank"
    assert requests[1][1]["return_documents"] is False


def test_online_providers_classify_action_required_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            request = httpx.Request("POST", "https://example.test/v1/request")
            response = httpx.Response(402, request=request)
            raise httpx.HTTPStatusError(
                "payment required",
                request=request,
                response=response,
            )

    class Client:
        def __init__(self, **_kwargs: object):
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return

        def post(self, _path: str, *, json: dict[str, object]) -> Response:
            del json
            return Response()

    monkeypatch.setattr("keepygaga_rag.knowledge.providers.httpx.Client", Client)
    monkeypatch.setenv("EMBED_KEY", "secret")
    monkeypatch.setenv("RERANK_KEY", "secret")
    embedding = OpenAICompatibleEmbeddingProvider(
        EmbeddingProfileConfig(
            provider="openai_compatible",
            base_url="https://example.test/v1",
            api_key_env="EMBED_KEY",
            model="embedding-model",
            dimensions=3,
        )
    )
    reranker = CohereCompatibleRerankProvider(
        RerankProfileConfig(
            provider="api",
            protocol="cohere_compatible",
            base_url="https://example.test/v1",
            api_key_env="RERANK_KEY",
            model="rerank-model",
        )
    )

    with pytest.raises(
        ProviderActionRequiredError,
        match="embedding provider request failed with HTTP 402",
    ):
        embedding.embed(["alpha"])
    with pytest.raises(
        ProviderActionRequiredError,
        match="rerank provider request failed with HTTP 402",
    ):
        reranker.rerank("alpha", ["document"], top_n=1)


def test_sync_error_summary_preserves_provider_action_required_error() -> None:
    errors = [f"{index}.md: RuntimeError: ordinary" for index in range(5)]
    provider_error = (
        "provider.md: ProviderActionRequiredError: embedding provider request "
        "failed with HTTP 402"
    )

    reported = _reported_sync_errors(
        [*errors, provider_error],
        provider_action_error=provider_error,
    )

    assert reported == [*errors[:4], provider_error]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0, 0.0]},
            ],
            "indices do not match",
        ),
        (
            [{"index": 0, "embedding": [float("nan"), 0.0, 0.0]}],
            "non-finite value",
        ),
        (
            [{"index": 0.9, "embedding": [1.0, 0.0, 0.0]}],
            "invalid index",
        ),
    ],
)
def test_embedding_provider_rejects_invalid_response_rows(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return {"data": rows}

    class Client:
        def __init__(self, **_kwargs: object):
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return

        def post(self, _path: str, *, json: dict[str, object]) -> Response:
            del json
            return Response()

    monkeypatch.setattr("keepygaga_rag.knowledge.providers.httpx.Client", Client)
    monkeypatch.setenv("EMBED_KEY", "secret")
    provider = OpenAICompatibleEmbeddingProvider(
        EmbeddingProfileConfig(
            provider="openai_compatible",
            base_url="https://example.test/v1",
            api_key_env="EMBED_KEY",
            model="embedding-model",
            dimensions=3,
        )
    )
    texts = ["first", "second"] if len(rows) == 2 else ["first"]

    with pytest.raises(RuntimeError, match=message):
        provider.embed(texts)
