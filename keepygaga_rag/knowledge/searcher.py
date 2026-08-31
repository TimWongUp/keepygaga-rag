from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

from keepygaga_rag.knowledge.authorization import (
    AuthorizationGuard,
    AuthorizationGuardUnavailable,
)
from keepygaga_rag.knowledge.chunking import (
    build_retrieval_text,
    filename_for_path,
    fts_query,
)
from keepygaga_rag.knowledge.db import KnowledgeDB, SourceRecord
from keepygaga_rag.knowledge.indexer import (
    source_path_is_selected,
    source_root_is_safe,
)
from keepygaga_rag.knowledge.providers import EmbeddingProvider, RerankProvider
from keepygaga_rag.knowledge.vectors import VectorStore

MAX_TOP_K = 20
RRF_K = 60


class RRFEntry(TypedDict):
    rrf_score: float
    sources: list[str]
    vector_distance: float | None


class KnowledgeSearcher:
    def __init__(
        self,
        *,
        database: KnowledgeDB,
        vectors: VectorStore,
        embedding: EmbeddingProvider,
        reranker: RerankProvider,
        table_id: str,
        consent_identity: str,
        candidate_limit: int | None = None,
        vector_recall_limit: int | None = None,
        keyword_recall_limit: int | None = None,
        rerank_candidate_limit: int | None = None,
        max_chunks_per_source_file: int = 2,
        authorization_guard: AuthorizationGuard | None = None,
    ):
        self.database = database
        self.vectors = vectors
        self.embedding = embedding
        self.reranker = reranker
        self.table_id = table_id
        legacy_candidate_limit = candidate_limit if candidate_limit is not None else 20
        legacy_recall_settings = (
            candidate_limit is not None
            and vector_recall_limit is None
            and keyword_recall_limit is None
            and rerank_candidate_limit is None
        )
        self.vector_recall_limit = (
            vector_recall_limit
            if vector_recall_limit is not None
            else legacy_candidate_limit * 4
            if legacy_recall_settings
            else 40
        )
        self.keyword_recall_limit = (
            keyword_recall_limit
            if keyword_recall_limit is not None
            else legacy_candidate_limit
        )
        self.rerank_candidate_limit = (
            rerank_candidate_limit
            if rerank_candidate_limit is not None
            else legacy_candidate_limit
        )
        self.max_chunks_per_source_file = max_chunks_per_source_file
        self.consent_identity = consent_identity
        self.authorization_guard = authorization_guard

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        table_ids: Sequence[str] = (),
        source_ids: Sequence[str] = (),
        include_diagnostics: bool = False,
    ) -> dict[str, object]:
        guard = self.authorization_guard
        if self.database.readonly:
            guard = guard or AuthorizationGuard(
                self.database.path.parent / "indexer.lock"
            )
            guard_context = guard.acquire_readonly()
        else:
            guard_context = nullcontext()
        try:
            with guard_context:
                return self._search_locked(
                    query=query,
                    top_k=top_k,
                    table_ids=table_ids,
                    source_ids=source_ids,
                    include_diagnostics=include_diagnostics,
                )
        except AuthorizationGuardUnavailable as exc:
            return {
                "status": "unavailable",
                "query": query.strip(),
                "groups": [],
                "message": f"authorization guard unavailable: {exc}",
                "warnings": [
                    f"authorization guard unavailable: {type(exc).__name__}: {exc}"
                ],
            }

    def _search_locked(
        self,
        *,
        query: str,
        top_k: int = 5,
        table_ids: Sequence[str] = (),
        source_ids: Sequence[str] = (),
        include_diagnostics: bool = False,
    ) -> dict[str, object]:
        started_at = perf_counter()
        query = query.strip()
        if not query:
            return {"status": "invalid_request", "message": "query must not be empty"}
        if not 1 <= top_k <= MAX_TOP_K:
            return {
                "status": "invalid_request",
                "message": f"top_k must be between 1 and {MAX_TOP_K}",
            }
        unknown_tables = sorted(set(table_ids) - {self.table_id})
        if unknown_tables:
            return {
                "status": "invalid_request",
                "message": f"unknown table_ids: {', '.join(unknown_tables)}",
            }
        if table_ids and self.table_id not in table_ids:
            return {"status": "no_results", "query": query, "groups": []}
        sources = self.database.list_sources()
        sources_by_id = {source.id: source for source in sources}
        known_source_ids = {source.id for source in sources}
        unknown_sources = sorted(set(source_ids) - known_source_ids)
        if unknown_sources:
            return {
                "status": "invalid_request",
                "message": f"unknown source_ids: {', '.join(unknown_sources)}",
            }
        requested = set(source_ids)
        disabled_requested = sorted(
            source.id
            for source in sources
            if source.id in requested and not source.enabled
        )
        if disabled_requested:
            return {
                "status": "invalid_request",
                "message": (
                    "requested source_ids are disabled: "
                    + ", ".join(disabled_requested)
                ),
                "source_ids": disabled_requested,
                "groups": [],
            }
        consented_source_ids = {
            source.id
            for source in sources
            if source.enabled
            and not source.scope_cleanup_pending
            and source.consent_identity == self.consent_identity
            and source_root_is_safe(source.absolute_path)
        }
        stale_requested = sorted(requested - consented_source_ids)
        if stale_requested:
            return {
                "status": "consent_required",
                "message": (
                    "one or more requested sources require renewed provider consent "
                    "or scope cleanup"
                ),
                "source_ids": stale_requested,
                "groups": [],
            }
        enabled_source_ids = {
            source.id
            for source in sources
            if source.enabled and source_root_is_safe(source.absolute_path)
        }
        stale_enabled = sorted(enabled_source_ids - consented_source_ids)
        if not source_ids and not consented_source_ids and stale_enabled:
            return {
                "status": "consent_required",
                "message": (
                    "all enabled sources require renewed provider consent "
                    "or scope cleanup"
                ),
                "source_ids": stale_enabled,
                "groups": [],
            }
        effective_source_ids = (
            list(source_ids) if source_ids else sorted(consented_source_ids)
        )
        initial_policy = self._policy_snapshot(sources, effective_source_ids)
        warnings: list[str] = []
        if not source_ids and stale_enabled and consented_source_ids:
            warnings.append(
                "some enabled sources were skipped because provider consent is stale "
                "or scope cleanup is pending: "
                + ", ".join(stale_enabled)
            )
        if not effective_source_ids:
            return {
                "status": "no_results",
                "query": query,
                "groups": [],
                "warnings": warnings,
            }

        lexical: list[dict[str, Any]] = []
        lexical_query = fts_query(query)
        lexical_started_at = perf_counter()
        if lexical_query:
            lexical = self.database.fts_search(
                lexical_query,
                limit=self.keyword_recall_limit,
                source_ids=effective_source_ids,
            )
        lexical_elapsed_ms = (perf_counter() - lexical_started_at) * 1000

        vector_rows: list[dict[str, object]] = []
        vector_started_at = perf_counter()
        try:
            query_vectors = self.embedding.embed([query])
        except Exception as exc:
            warnings.append(
                f"vector recall unavailable: {type(exc).__name__}"
            )
        else:
            if query_vectors:
                try:
                    vector_rows = self.vectors.search(
                        query_vectors[0],
                        limit=self.vector_recall_limit,
                        source_ids=effective_source_ids,
                    )
                except Exception as exc:
                    reopen = getattr(self.vectors, "reopen", None)
                    if not callable(reopen):
                        warnings.append(
                            f"vector recall unavailable: {type(exc).__name__}"
                        )
                    else:
                        try:
                            reopen()
                            vector_rows = self.vectors.search(
                                query_vectors[0],
                                limit=self.vector_recall_limit,
                                source_ids=effective_source_ids,
                            )
                            warnings.append(
                                "vector recall recovered after reopening the backend"
                            )
                        except Exception as retry_exc:
                            warnings.append(
                                "vector recall unavailable after one retry: "
                                f"{type(retry_exc).__name__}"
                            )
        vector_elapsed_ms = (perf_counter() - vector_started_at) * 1000

        fusion_started_at = perf_counter()
        allowed = set(effective_source_ids)
        recall_ids = list(
            dict.fromkeys(
                str(row["chunk_id"]) for row in [*lexical, *vector_rows]
            )
        )
        recalled_chunks = {
            str(chunk["chunk_id"]): chunk
            for chunk in self.database.active_chunks(recall_ids)
            if str(chunk["source_id"]) in allowed
            and (source := sources_by_id.get(str(chunk["source_id"]))) is not None
            and source_root_is_safe(source.absolute_path)
            and not source.scope_cleanup_pending
            and source_path_is_selected(str(chunk["relative_path"]), source)
        }
        lexical = [
            row for row in lexical if str(row["chunk_id"]) in recalled_chunks
        ]
        vector_rows = [
            row for row in vector_rows if str(row["chunk_id"]) in recalled_chunks
        ]
        ranked = self._rrf(lexical, vector_rows)
        chunks = [
            recalled_chunks[chunk_id]
            for chunk_id in ranked
            if chunk_id in recalled_chunks
        ]
        chunks = chunks[: self.rerank_candidate_limit]
        fusion_elapsed_ms = (perf_counter() - fusion_started_at) * 1000
        if not chunks:
            result: dict[str, object] = {
                "status": "no_results",
                "query": query,
                "groups": [],
                "warnings": warnings,
            }
            if include_diagnostics:
                result["diagnostics"] = self._diagnostics(
                    lexical_query=lexical_query,
                    lexical=lexical,
                    vectors=vector_rows,
                    ranked=ranked,
                    recalled_chunks=recalled_chunks,
                    rerank_chunks=[],
                    reranked=[],
                    final=[],
                    timings={
                        "keyword": lexical_elapsed_ms,
                        "vector": vector_elapsed_ms,
                        "fusion": fusion_elapsed_ms,
                        "rerank": 0.0,
                        "total": (perf_counter() - started_at) * 1000,
                    },
                )
            return result

        try:
            current_sources_list = self.database.list_sources()
            current_policy = self._policy_snapshot(
                current_sources_list, effective_source_ids
            )
            current_sources = {
                source.id: source for source in current_sources_list
            }
            verified_chunks = [
                chunk
                for chunk in chunks
                if (
                    source := current_sources.get(str(chunk["source_id"]))
                )
                is not None
                and source.enabled
                and not source.scope_cleanup_pending
                and source.consent_identity == self.consent_identity
                and source_root_is_safe(source.absolute_path)
                and source_path_is_selected(
                    str(chunk["relative_path"]), source
                )
            ]
        except Exception as exc:
            warnings.append(
                f"authorization policy recheck unavailable: {type(exc).__name__}"
            )
            return {
                "status": "no_results",
                "query": query,
                "groups": [],
                "warnings": warnings,
            }
        if current_policy != initial_policy:
            warnings.append("authorization policy changed during search")
            return {
                "status": "no_results",
                "query": query,
                "groups": [],
                "warnings": warnings,
            }
        chunks = verified_chunks
        if not chunks:
            return {
                "status": "no_results",
                "query": query,
                "groups": [],
                "warnings": warnings,
            }
        reranked: list[tuple[int, float]] = []
        rerank_started_at = perf_counter()
        try:
            reranked = self.reranker.rerank(
                query,
                [self._rerank_document(chunk) for chunk in chunks],
                top_n=min(self.rerank_candidate_limit, len(chunks)),
            )
        except Exception as exc:
            warnings.append(f"rerank unavailable: {type(exc).__name__}")
        reranker_had_valid_index = any(
            0 <= index < len(chunks) for index, _ in reranked
        )
        final = self._select_final_chunks(
            chunks,
            reranked,
            top_k=top_k,
        )
        if reranked and not reranker_had_valid_index:
            warnings.append(
                "reranker returned no valid result indexes; using RRF"
            )
            final = self._select_final_chunks(chunks, [], top_k=top_k)
        rerank_elapsed_ms = (perf_counter() - rerank_started_at) * 1000

        results = [
            self._result(
                chunk,
                rerank_score,
                ranked[str(chunk["chunk_id"])],
            )
            for chunk, rerank_score in final
        ]
        result = {
            "status": "ok",
            "query": query,
            "groups": [{"table": self.table_id, "results": results}],
            "warnings": warnings,
        }
        if include_diagnostics:
            result["diagnostics"] = self._diagnostics(
                lexical_query=lexical_query,
                lexical=lexical,
                vectors=vector_rows,
                ranked=ranked,
                recalled_chunks=recalled_chunks,
                rerank_chunks=chunks,
                reranked=reranked,
                final=results,
                timings={
                    "keyword": lexical_elapsed_ms,
                    "vector": vector_elapsed_ms,
                    "fusion": fusion_elapsed_ms,
                    "rerank": rerank_elapsed_ms,
                    "total": (perf_counter() - started_at) * 1000,
                },
            )
        return result

    def _diagnostics(
        self,
        *,
        lexical_query: str,
        lexical: Sequence[dict[str, Any]],
        vectors: Sequence[dict[str, object]],
        ranked: dict[str, RRFEntry],
        recalled_chunks: dict[str, dict[str, Any]],
        rerank_chunks: Sequence[dict[str, Any]],
        reranked: Sequence[tuple[int, float]],
        final: Sequence[dict[str, object]],
        timings: dict[str, float],
    ) -> dict[str, object]:
        def numeric(value: object) -> float:
            return float(value) if isinstance(value, (int, float)) else 0.0

        def candidate(
            chunk: dict[str, Any],
            *,
            rank: int,
            metric_label: str,
            metric: float,
            channels: Sequence[str] = (),
        ) -> dict[str, object]:
            source = Path(str(chunk["absolute_path"])) / str(
                chunk["relative_path"]
            )
            text = " ".join(str(chunk["text"]).split())
            return {
                "rank": rank,
                "source": str(source),
                "heading_path": str(chunk["heading_path"]),
                "excerpt": text[:220],
                "metric_label": metric_label,
                "metric": metric,
                "channels": list(channels),
            }

        keyword_candidates = [
            candidate(
                recalled_chunks[str(row["chunk_id"])],
                rank=rank,
                metric_label="BM25",
                metric=float(row.get("lexical_score", 0.0)),
            )
            for rank, row in enumerate(lexical[:10], start=1)
            if str(row["chunk_id"]) in recalled_chunks
        ]
        vector_candidates = [
            candidate(
                recalled_chunks[str(row["chunk_id"])],
                rank=rank,
                metric_label="DIST",
                metric=numeric(row.get("distance", 0.0)),
            )
            for rank, row in enumerate(vectors[:10], start=1)
            if str(row["chunk_id"]) in recalled_chunks
        ]
        fusion_candidates = [
            candidate(
                recalled_chunks[chunk_id],
                rank=rank,
                metric_label="RRF",
                metric=entry["rrf_score"],
                channels=entry["sources"],
            )
            for rank, (chunk_id, entry) in enumerate(
                list(ranked.items())[:10], start=1
            )
            if chunk_id in recalled_chunks
        ]
        rerank_candidates = [
            candidate(
                rerank_chunks[index],
                rank=rank,
                metric_label="SCORE",
                metric=float(score),
            )
            for rank, (index, score) in enumerate(reranked[:10], start=1)
            if 0 <= index < len(rerank_chunks)
        ]
        final_candidates = [
            {
                "rank": rank,
                "source": str(row["source"]),
                "heading_path": str(row["heading_path"]),
                "excerpt": " ".join(str(row["text"]).split())[:260],
                "metric_label": "FINAL",
                "metric": numeric(row["score"]),
                "channels": [],
            }
            for rank, row in enumerate(final, start=1)
        ]
        return {
            "settings": {
                "vector_recall_limit": self.vector_recall_limit,
                "keyword_recall_limit": self.keyword_recall_limit,
                "rerank_candidate_limit": self.rerank_candidate_limit,
                "max_chunks_per_source_file": self.max_chunks_per_source_file,
            },
            "lexical_query": lexical_query,
            "stages": [
                {
                    "id": "keyword",
                    "count": len(lexical),
                    "elapsed_ms": timings["keyword"],
                    "candidates": keyword_candidates,
                },
                {
                    "id": "vector",
                    "count": len(vectors),
                    "elapsed_ms": timings["vector"],
                    "candidates": vector_candidates,
                },
                {
                    "id": "fusion",
                    "count": len(ranked),
                    "elapsed_ms": timings["fusion"],
                    "candidates": fusion_candidates,
                },
                {
                    "id": "rerank",
                    "count": len(reranked),
                    "elapsed_ms": timings["rerank"],
                    "candidates": rerank_candidates,
                },
            ],
            "final": final_candidates,
            "total_elapsed_ms": timings["total"],
        }

    def _select_final_chunks(
        self,
        chunks: Sequence[dict[str, Any]],
        reranked: Sequence[tuple[int, float]],
        *,
        top_k: int,
    ) -> list[tuple[dict[str, Any], float | None]]:
        ordered: list[tuple[dict[str, Any], float | None]] = []
        reranked_indexes: set[int] = set()
        for index, score in reranked:
            if 0 <= index < len(chunks) and index not in reranked_indexes:
                reranked_indexes.add(index)
                ordered.append((chunks[index], score))
        if not ordered:
            ordered = [(chunk, None) for chunk in chunks]
        else:
            ordered.extend(
                (chunk, None)
                for index, chunk in enumerate(chunks)
                if index not in reranked_indexes
            )

        selected: list[tuple[dict[str, Any], float | None]] = []
        counts: dict[tuple[str, str], int] = {}
        for chunk, score in ordered:
            source_key = (
                str(chunk["source_id"]),
                str(chunk["relative_path"]),
            )
            if counts.get(source_key, 0) >= self.max_chunks_per_source_file:
                continue
            counts[source_key] = counts.get(source_key, 0) + 1
            selected.append((chunk, score))
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _policy_snapshot(
        sources: Sequence[SourceRecord], source_ids: Sequence[str]
    ) -> tuple[tuple[object, ...], ...]:
        requested = set(source_ids)
        return tuple(
            sorted(
                (
                    source.id,
                    source.enabled,
                    source.scope_cleanup_pending,
                    source.consent_identity,
                    source.include_patterns,
                    source.exclude_patterns,
                )
                for source in sources
                if source.id in requested
            )
        )

    @staticmethod
    def _rrf(
        lexical: Sequence[dict[str, Any]],
        vectors: Sequence[dict[str, object]],
    ) -> dict[str, RRFEntry]:
        scores: dict[str, RRFEntry] = {}
        for source_name, rows in (("fts", lexical), ("vector", vectors)):
            for rank, row in enumerate(rows, start=1):
                chunk_id = str(row["chunk_id"])
                entry = scores.setdefault(
                    chunk_id,
                    {
                        "rrf_score": 0.0,
                        "sources": [],
                        "vector_distance": None,
                    },
                )
                entry["rrf_score"] += 1.0 / (RRF_K + rank)
                entry["sources"].append(source_name)
                if source_name == "vector":
                    distance = row.get("distance")
                    if isinstance(distance, (int, float)):
                        entry["vector_distance"] = float(distance)
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1]["rrf_score"], item[0]),
        )
        return dict(ordered)

    @staticmethod
    def _rerank_document(chunk: dict[str, Any]) -> str:
        filename = str(chunk.get("filename") or "") or filename_for_path(
            str(chunk["relative_path"])
        )
        return build_retrieval_text(
            text=str(chunk["text"]),
            title=str(chunk["title"]),
            heading_path=str(chunk["heading_path"]),
            filename=filename,
        )

    @staticmethod
    def _result(
        chunk: dict[str, Any],
        rerank_score: float | None,
        score: RRFEntry,
    ) -> dict[str, object]:
        source = Path(str(chunk["absolute_path"])) / str(chunk["relative_path"])
        return {
            "source": str(source),
            "heading_path": str(chunk["heading_path"]),
            "text": str(chunk["text"]),
            "score": rerank_score
            if rerank_score is not None
            else score["rrf_score"],
        }
