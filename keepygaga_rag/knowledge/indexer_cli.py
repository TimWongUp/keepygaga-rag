from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from keepygaga_rag.config import DEFAULT_CONFIG_PATH, load_config
from keepygaga_rag.knowledge.authorization import AuthorizationGuardUnavailable
from keepygaga_rag.knowledge.chunking import (
    build_embedding_input_hash,
    build_retrieval_text,
    filename_for_path,
)
from keepygaga_rag.knowledge.db import SourceRecord
from keepygaga_rag.knowledge.indexer import (
    IndexerLockUnavailable,
    KnowledgeIndexer,
    source_path_is_selected,
    source_root_is_safe,
)
from keepygaga_rag.knowledge.retry import automatic_retry_is_due
from keepygaga_rag.knowledge.runtime import (
    KnowledgeRuntime,
    resolve_knowledge_store,
)
from keepygaga_rag.knowledge.vectors import LanceVectorStore

LOGGER = logging.getLogger(__name__)
VECTOR_HEALTH_CHECK_INTERVAL_SECONDS = 5 * 60


class VectorBackendUnavailable(RuntimeError):
    """Raised when reconciliation must wait for vector backend recovery."""


def load_current_chunk_settings(
    runtime: KnowledgeRuntime,
) -> tuple[int, int, str, int] | None:
    try:
        current = load_config(runtime.config_path)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "keeping previous chunk settings because config refresh failed: %s",
            type(exc).__name__,
        )
        return None
    if not current.knowledge.enabled:
        return None
    return (
        current.knowledge.chunk_target_chars,
        current.knowledge.chunk_max_chars,
        current.knowledge.chunk_mode,
        current.knowledge.chunk_overlap_chars,
    )


def queue_automatic_sources(
    runtime: KnowledgeRuntime,
    *,
    reason: str,
    source_ids: set[str] | None = None,
    source_predicate: Callable[[SourceRecord], bool] | None = None,
) -> int:
    queued = 0
    for source in runtime.database.list_sources():
        if source_ids is not None and source.id not in source_ids:
            continue
        if source_predicate is not None and not source_predicate(source):
            continue
        if not (
            source.enabled
            and source.auto_sync
            and not source.scope_cleanup_pending
            and source.consent_identity == runtime.consent_identity
            and source_root_is_safe(source.absolute_path)
        ):
            continue
        try:
            runtime.database.queue_sync(
                source.id,
                reason,
                expected_consent_identity=runtime.consent_identity,
            )
        except (KeyError, ValueError):
            continue
        queued += 1
    return queued


def build_indexer(runtime: KnowledgeRuntime) -> KnowledgeIndexer:
    indexer = KnowledgeIndexer(
        database=runtime.database,
        vectors=runtime.vectors,
        embedding=runtime.embedding,
        table_id=runtime.table_id,
        lock_path=runtime.store_root / "indexer.lock",
        target_chars=runtime.config.knowledge.chunk_target_chars,
        max_chars=runtime.config.knowledge.chunk_max_chars,
        chunk_mode=runtime.config.knowledge.chunk_mode,
        overlap_chars=runtime.config.knowledge.chunk_overlap_chars,
        stability_seconds=runtime.config.knowledge.stability_seconds,
        missing_confirmations=runtime.config.knowledge.missing_confirmations,
        consent_identity=runtime.consent_identity,
        distance_metric=runtime.config.embedding_profiles[
            runtime.config.knowledge.embedding_profile
        ].distance_metric,
        authorization_is_current=runtime.authorization_is_current,
        chunk_settings_loader=lambda: load_current_chunk_settings(runtime),
    )
    return indexer


def _rebuild_embedding_text(row: dict[str, object]) -> str:
    filename = str(row.get("filename") or "") or filename_for_path(
        str(row["relative_path"])
    )
    return build_retrieval_text(
        text=str(row["text"]),
        title=str(row["title"]),
        heading_path=str(row["heading_path"]),
        filename=filename,
    )


def _source_rebuild_signature(source: SourceRecord) -> tuple[object, ...]:
    return (
        source.id,
        source.absolute_path,
        source.enabled,
        source.include_patterns,
        source.exclude_patterns,
        source.consent_identity,
        source.scope_cleanup_pending,
        source_root_is_safe(source.absolute_path),
    )


def repair_vector_store(
    runtime: KnowledgeRuntime,
    indexer: KnowledgeIndexer,
) -> bool:
    """Build a fresh vector table and switch to it only after validation."""
    vectors: LanceVectorStore | None = None
    activated = False
    try:
        with indexer.authorization_guard.acquire_exclusive(timeout=0):
            source_records = runtime.database.list_sources()
            sources = {source.id: source for source in source_records}
            source_snapshot = {
                source.id: _source_rebuild_signature(source)
                for source in source_records
            }
            eligible_sources = {
                source_id
                for source_id, source in sources.items()
                if not source.scope_cleanup_pending
                and source.consent_identity == runtime.consent_identity
                and source_root_is_safe(source.absolute_path)
            }
            rebuild_revision = runtime.database.vector_rebuild_revision()

        embedding_space_id = hashlib.sha256(
            runtime.embedding.identity.encode("utf-8")
        ).hexdigest()
        suffix = hashlib.sha256(
            f"{embedding_space_id}\0{time.time_ns()}".encode()
        ).hexdigest()[:12]
        table_name = f"{runtime.table_id}_{suffix}"
        vectors = LanceVectorStore(
            runtime.store_root / "lancedb",
            table_name=table_name,
            dimensions=runtime.embedding.dimensions,
            embedding_space_id=embedding_space_id,
            distance_metric=indexer.distance_metric,
        )
        vectors.ensure_table()

        seen_active_chunks = False
        all_active_chunks_pending_cleanup = True
        rebuilt = 0
        batch_size = 32
        for page in runtime.database.iter_active_chunks_for_vector_rebuild():
            if page:
                seen_active_chunks = True
            for row in page:
                source = sources.get(str(row["source_id"]))
                if source is None or not source.scope_cleanup_pending:
                    all_active_chunks_pending_cleanup = False
                    break
            candidates = [
                row
                for row in page
                if str(row["source_id"]) in eligible_sources
                and source_path_is_selected(
                    str(row["relative_path"]),
                    sources[str(row["source_id"])],
                )
            ]
            for offset in range(0, len(candidates), batch_size):
                batch = candidates[offset : offset + batch_size]
                embedding_texts = [
                    _rebuild_embedding_text(row) for row in batch
                ]
                embeddings = runtime.embedding.embed(
                    embedding_texts
                )
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        "embedding provider returned an unexpected count"
                    )
                rows = [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "source_id": str(row["source_id"]),
                        "source_file_id": int(row["source_file_id"]),
                        "embedding_input_hash": build_embedding_input_hash(
                            retrieval_text=embedding_text,
                            embedding_identity=runtime.embedding.identity,
                        ),
                        "embedding_space_id": embedding_space_id,
                        "generation": int(row["generation"]),
                        "vector": vector,
                    }
                    for row, vector, embedding_text in zip(
                        batch, embeddings, embedding_texts, strict=True
                    )
                ]
                vectors.add(rows)
                rebuilt += len(rows)

        if (
            seen_active_chunks
            and not rebuilt
            and not all_active_chunks_pending_cleanup
        ):
            LOGGER.warning(
                "vector table repair deferred: no active chunks have current consent"
            )
            vectors.drop()
            vectors = None
            return False

        current_source_records = runtime.database.list_sources()
        current_source_snapshot = {
            source.id: _source_rebuild_signature(source)
            for source in current_source_records
        }
        if current_source_snapshot != source_snapshot:
            raise RuntimeError(
                "source authorization or scope changed during vector repair"
            )
        authorization_is_current = getattr(
            runtime, "authorization_is_current", None
        )
        if callable(authorization_is_current) and not authorization_is_current():
            raise RuntimeError(
                "knowledge provider configuration changed during vector repair"
            )
        if not runtime.database.activate_vector_table_if_revision_matches(
            runtime.table_id,
            table_name,
            rebuild_revision=rebuild_revision,
        ):
            raise RuntimeError(
                "knowledge sources or active chunks changed during vector repair"
            )
        activated = True
        runtime.vectors = vectors
        indexer.vectors = vectors
        LOGGER.info(
            "vector table repaired: table=%s rebuilt_chunks=%d",
            table_name,
            rebuilt,
        )
        return True
    except (Timeout, AuthorizationGuardUnavailable) as exc:
        if vectors is not None and not activated:
            with contextlib.suppress(Exception):
                vectors.drop()
        raise IndexerLockUnavailable(
            "another knowledge indexer holds the write lock"
        ) from exc
    except Exception:
        if vectors is not None and not activated:
            with contextlib.suppress(Exception):
                vectors.drop()
        raise


def ensure_vector_store(
    runtime: KnowledgeRuntime,
    indexer: KnowledgeIndexer,
) -> bool:
    vectors = getattr(runtime, "vectors", None)
    if vectors is None:
        return True
    now = time.monotonic()
    if now < indexer.vector_repair_backoff_until:
        return False
    revision_reader = getattr(indexer.database, "vector_rebuild_revision", None)
    current_revision = (
        cast(int | None, revision_reader())
        if callable(revision_reader)
        else None
    )
    if (
        indexer.vector_repair_failures == 0
        and now < indexer.vector_health_check_at
        and current_revision == getattr(indexer, "vector_health_revision", None)
    ):
        return True
    try:
        vectors.ensure_table()
        sources = {source.id: source for source in runtime.database.list_sources()}
        eligible_sources = {
            source_id
            for source_id, source in sources.items()
            if not source.scope_cleanup_pending
            and source.consent_identity == runtime.consent_identity
            and source_root_is_safe(source.absolute_path)
        }
        metadata_reader = getattr(vectors, "get_vector_metadata", None)
        expected_space_id = hashlib.sha256(
            indexer.embedding.identity.encode("utf-8")
        ).hexdigest()
        for page in runtime.database.iter_active_chunks_for_vector_rebuild():
            eligible = [
                row
                for row in page
                if str(row["source_id"]) in eligible_sources
                and source_path_is_selected(
                    str(row["relative_path"]),
                    sources[str(row["source_id"])],
                )
            ]
            if not eligible:
                continue
            chunk_ids = [str(row["chunk_id"]) for row in eligible]
            if callable(metadata_reader):
                metadata = cast(
                    dict[str, dict[str, str]], metadata_reader(chunk_ids)
                )
                valid = all(
                    metadata.get(str(row["chunk_id"])) == {
                        "embedding_space_id": expected_space_id,
                        "embedding_input_hash": str(row["embedding_input_hash"]),
                    }
                    for row in eligible
                )
            else:
                valid = set(vectors.get_vectors(chunk_ids)) == set(chunk_ids)
            if not valid:
                raise RuntimeError(
                    "active chunks are missing or have invalid vectors"
                )
        indexer.vector_repair_failures = 0
        indexer.vector_repair_backoff_until = 0.0
        indexer.vector_health_check_at = now + VECTOR_HEALTH_CHECK_INTERVAL_SECONDS
        indexer.vector_health_revision = current_revision
        return True
    except Exception as exc:
        LOGGER.warning(
            "vector backend health check failed; attempting isolated rebuild: %s: %s",
            type(exc).__name__,
            exc,
        )
        try:
            repaired = repair_vector_store(runtime, indexer)
        except Exception:
            indexer.vector_repair_failures += 1
            delay = min(
                300.0,
                5.0 * (2 ** min(indexer.vector_repair_failures - 1, 5)),
            )
            indexer.vector_repair_backoff_until = time.monotonic() + delay
            if indexer.vector_repair_failures == 1:
                LOGGER.exception(
                    "vector table repair failed; retrying after %.0f seconds",
                    delay,
                )
            else:
                LOGGER.warning(
                    "vector table repair still unavailable; retrying after %.0f seconds",
                    delay,
                )
            return False
        if repaired:
            indexer.vector_repair_failures = 0
            indexer.vector_repair_backoff_until = 0.0
            indexer.vector_health_check_at = (
                time.monotonic() + VECTOR_HEALTH_CHECK_INTERVAL_SECONDS
            )
            revision_reader = getattr(
                indexer.database, "vector_rebuild_revision", None
            )
            indexer.vector_health_revision = (
                cast(int | None, revision_reader())
                if callable(revision_reader)
                else None
            )
            return True
        indexer.vector_repair_failures += 1
        delay = min(
            300.0,
            5.0 * (2 ** min(indexer.vector_repair_failures - 1, 5)),
        )
        indexer.vector_repair_backoff_until = time.monotonic() + delay
        return False


def run_once(runtime: KnowledgeRuntime, indexer: KnowledgeIndexer) -> int:
    if not ensure_vector_store(runtime, indexer):
        raise VectorBackendUnavailable("vector backend health check failed")
    indexer.reconcile_chunk_settings()
    cleanup_pending = {
        source.id
        for source in runtime.database.list_sources()
        if source.scope_cleanup_pending
    }
    indexer.reconcile_derived_state()
    queue_automatic_sources(
        runtime,
        reason="scope-cleanup-reconciled",
        source_ids=cleanup_pending,
    )
    processed = 0
    failures: list[str] = []
    while claimed := runtime.database.claim_next_job():
        job_id, source = claimed
        error = ""
        try:
            indexer.sync_source(source)
        except IndexerLockUnavailable as exc:
            with contextlib.suppress(KeyError, ValueError):
                runtime.database.requeue_job(job_id, error=str(exc))
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failures.append(error)
        try:
            runtime.database.finish_job(job_id, error=error)
        except KeyError:
            continue
        processed += 1
    if failures:
        raise RuntimeError(
            "knowledge synchronization failed: " + "; ".join(failures[:5])
        )
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Keepygaga RAG index coordinator."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args(argv)

    config_path = Path(arguments.config).expanduser()
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    config = load_config(config_path)
    store_root = resolve_knowledge_store(config, config_path)
    store_root.mkdir(parents=True, exist_ok=True)
    coordinator = FileLock(store_root / "coordinator.lock")
    try:
        with coordinator.acquire(timeout=0):
            runtime = KnowledgeRuntime.from_config(
                config,
                config_path,
                coordinator_lock=coordinator,
            )
            indexer = build_indexer(runtime)
            runtime.database.recover_interrupted_jobs()
            interval = config.knowledge.scan_interval_seconds
            startup_retry_due = partial(
                automatic_retry_is_due,
                regular_interval_seconds=interval,
                now=datetime.now(UTC),
            )
            queue_automatic_sources(
                runtime,
                reason="startup",
                source_predicate=startup_retry_due,
            )
            if arguments.once:
                try:
                    run_once(runtime, indexer)
                except Exception:
                    LOGGER.exception("knowledge reconciliation failed")
                    return 1
                return 0

            while True:
                if not runtime.authorization_is_current():
                    raise SystemExit(
                        "knowledge provider configuration changed; restart "
                        "keepygaga-rag indexer"
                    )
                now = datetime.now(UTC)
                before = (now - timedelta(seconds=interval)).isoformat()
                retry_due = partial(
                    automatic_retry_is_due,
                    regular_interval_seconds=interval,
                    now=now,
                )
                runtime.database.queue_due_sources(
                    before=before,
                    consent_identity=runtime.consent_identity,
                    source_root_predicate=source_root_is_safe,
                    source_predicate=retry_due,
                )
                try:
                    processed = run_once(runtime, indexer)
                except IndexerLockUnavailable as exc:
                    LOGGER.warning("knowledge reconciliation deferred: %s", exc)
                    processed = 0
                except VectorBackendUnavailable as exc:
                    LOGGER.debug(
                        "vector reconciliation deferred during backend backoff: %s",
                        exc,
                    )
                    processed = 0
                except Exception:
                    LOGGER.exception(
                        "knowledge reconciliation failed; it will be retried"
                    )
                    processed = 0
                if processed == 0:
                    time.sleep(min(5, interval))
    except Timeout as exc:
        raise SystemExit(
            "another keepygaga-rag indexer process is already running"
        ) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
