from __future__ import annotations

from pathlib import Path

import pytest

from keepygaga_rag.config import (
    load_config,
    update_knowledge_chunk_settings,
    update_knowledge_retrieval_settings,
)


def test_config_uses_knowledge_defaults(tmp_path: Path) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    path.write_text("", encoding="utf-8")
    config = load_config(path)
    assert config.knowledge.enabled is False
    assert config.knowledge.scan_interval_seconds == 300
    assert config.knowledge.chunk_target_chars == 2400
    assert config.knowledge.chunk_max_chars == 3200
    assert config.knowledge.chunk_mode == "structure"
    assert config.knowledge.chunk_overlap_chars == 0
    assert config.knowledge.vector_recall_limit == 40
    assert config.knowledge.keyword_recall_limit == 20
    assert config.knowledge.max_chunks_per_source_file == 2


def test_update_knowledge_chunk_settings_preserves_toml_format(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    path.write_text(
        """# keep this header
[knowledge] # keep this table comment
enabled = false
chunk_target_chars = 2400 # soft

""",
        encoding="utf-8",
    )

    updated = update_knowledge_chunk_settings(
        path,
        target_chars=1800,
        max_chars=2600,
        chunk_mode="length",
        overlap_chars=12,
    )

    text = path.read_text(encoding="utf-8")
    assert "# keep this header" in text
    assert "chunk_target_chars = 1800 # soft" in text
    assert "chunk_max_chars = 2600" in text
    assert 'chunk_mode = "length"' in text
    assert "chunk_overlap_chars = 12" in text
    assert updated.knowledge.chunk_target_chars == 1800
    assert updated.knowledge.chunk_max_chars == 2600
    assert updated.knowledge.chunk_mode == "length"
    assert updated.knowledge.chunk_overlap_chars == 12


def test_update_knowledge_retrieval_settings_preserves_profile_format(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    path.write_text(
        """# keep this header
[knowledge] # keep this table comment
enabled = false
vector_recall_limit = 40 # vector

[rerank.profiles.qwen3_reranker_4b]
provider = "api"
protocol = "cohere_compatible"
base_url = "https://example.test/v1"
api_key_env = "RERANK_KEY"
model = "rerank-model"
candidate_limit = 20 # rerank
""",
        encoding="utf-8",
    )

    updated = update_knowledge_retrieval_settings(
        path,
        vector_recall_limit=32,
        keyword_recall_limit=16,
        rerank_candidate_limit=24,
        max_chunks_per_source_file=2,
        rerank_profile="qwen3_reranker_4b",
    )

    text = path.read_text(encoding="utf-8")
    assert "# keep this header" in text
    assert "vector_recall_limit = 32 # vector" in text
    assert "keyword_recall_limit = 16" in text
    assert "max_chunks_per_source_file = 2" in text
    assert "candidate_limit = 24 # rerank" in text
    assert updated.knowledge.vector_recall_limit == 32
    assert updated.knowledge.keyword_recall_limit == 16
    assert updated.knowledge.max_chunks_per_source_file == 2
    assert updated.rerank_profiles["qwen3_reranker_4b"].candidate_limit == 24


@pytest.mark.parametrize(
    ("target_chars", "max_chars"),
    [(0, 100), (100, 0), (201, 200)],
)
def test_update_knowledge_chunk_settings_rejects_invalid_values_without_writing(
    tmp_path: Path,
    target_chars: int,
    max_chars: int,
) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    original = "[knowledge]\nenabled = false\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        update_knowledge_chunk_settings(
            path,
            target_chars=target_chars,
            max_chars=max_chars,
        )

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("chunk_mode", "overlap_chars"),
    [("unknown", 0), ("length", -1), ("length", 200)],
)
def test_update_knowledge_chunk_settings_rejects_mode_and_overlap(
    tmp_path: Path,
    chunk_mode: str,
    overlap_chars: int,
) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    original = "[knowledge]\nenabled = false\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        update_knowledge_chunk_settings(
            path,
            target_chars=100,
            max_chars=200,
            chunk_mode=chunk_mode,
            overlap_chars=overlap_chars,
        )

    assert path.read_text(encoding="utf-8") == original


def test_knowledge_profiles_are_required_when_enabled(tmp_path: Path) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    path.write_text("[knowledge]\nenabled = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embedding_profile"):
        load_config(path)


def test_knowledge_config_loads_independent_profiles(tmp_path: Path) -> None:
    path = tmp_path / "keepygaga-rag.toml"
    path.write_text(
        """
[knowledge]
enabled = true
scan_interval_seconds = 600

[embedding.profiles.qwen3_4b_2560]
provider = "openai_compatible"
base_url = "https://example.test/v1"
api_key_env = "EMBED_KEY"
model = "embedding-model"
dimensions = 1024

[rerank.profiles.qwen3_reranker_4b]
provider = "api"
protocol = "cohere_compatible"
base_url = "https://example.test/v1"
api_key_env = "RERANK_KEY"
model = "rerank-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.knowledge.scan_interval_seconds == 600
    assert config.embedding_profiles["qwen3_4b_2560"].dimensions == 1024
    assert config.rerank_profiles["qwen3_reranker_4b"].model == "rerank-model"
