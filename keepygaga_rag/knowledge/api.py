from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from keepygaga_rag.config import DEFAULT_CONFIG_PATH
from keepygaga_rag.knowledge.runtime import KnowledgeRuntime
from keepygaga_rag.knowledge.searcher import KnowledgeSearcher


def knowledge_search(
    *,
    query: str,
    top_k: int = 5,
    table_ids: Sequence[str] = (),
    source_ids: Sequence[str] = (),
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, object]:
    return _run_search(
        query=query,
        top_k=top_k,
        table_ids=table_ids,
        source_ids=source_ids,
        config_path=config_path,
        include_diagnostics=False,
    )


def knowledge_search_diagnostics(
    *,
    query: str,
    top_k: int = 5,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, object]:
    return _run_search(
        query=query,
        top_k=top_k,
        table_ids=(),
        source_ids=(),
        config_path=config_path,
        include_diagnostics=True,
    )


def _run_search(
    *,
    query: str,
    top_k: int,
    table_ids: Sequence[str],
    source_ids: Sequence[str],
    config_path: Path | str,
    include_diagnostics: bool,
) -> dict[str, object]:
    try:
        runtime = KnowledgeRuntime.load(config_path, readonly=True)
    except Exception as exc:
        disabled = "disabled" in str(exc).casefold()
        uninitialized = isinstance(exc, FileNotFoundError)
        return {
            "status": (
                "disabled"
                if disabled
                else "not_initialized"
                if uninitialized
                else "invalid_source"
            ),
            "message": (
                "knowledge backend is disabled"
                if disabled
                else "knowledge index is not initialized"
                if uninitialized
                else f"knowledge backend unavailable: {type(exc).__name__}"
            ),
            "groups": [],
        }
    profile = runtime.config.rerank_profiles[
        runtime.config.knowledge.rerank_profile
    ]
    searcher = KnowledgeSearcher(
        database=runtime.database,
        vectors=runtime.vectors,
        embedding=runtime.embedding,
        reranker=runtime.reranker,
        table_id=runtime.table_id,
        consent_identity=runtime.consent_identity,
        candidate_limit=profile.candidate_limit,
        vector_recall_limit=runtime.config.knowledge.vector_recall_limit,
        keyword_recall_limit=runtime.config.knowledge.keyword_recall_limit,
        rerank_candidate_limit=profile.candidate_limit,
        max_chunks_per_source_file=(
            runtime.config.knowledge.max_chunks_per_source_file
        ),
        authorization_guard=runtime.authorization_guard,
    )
    return searcher.search(
        query=query,
        top_k=top_k,
        table_ids=table_ids,
        source_ids=source_ids,
        include_diagnostics=include_diagnostics,
    )
