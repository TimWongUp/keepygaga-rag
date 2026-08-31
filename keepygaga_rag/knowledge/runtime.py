from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from keepygaga_rag.config import (
    DEFAULT_CONFIG_PATH,
    EmbeddingProfileConfig,
    KnowledgeAppConfig,
    RerankProfileConfig,
    load_config,
)
from keepygaga_rag.knowledge.authorization import AuthorizationGuard
from keepygaga_rag.knowledge.db import (
    SCHEMA_VERSION,
    KnowledgeDB,
    read_schema_version,
)
from keepygaga_rag.knowledge.providers import (
    CohereCompatibleRerankProvider,
    OpenAICompatibleEmbeddingProvider,
)
from keepygaga_rag.knowledge.vectors import (
    LanceVectorStore,
    UnavailableVectorStore,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class KnowledgeRuntime:
    config: KnowledgeAppConfig
    config_path: Path
    store_root: Path
    database: KnowledgeDB
    authorization_guard: AuthorizationGuard
    vectors: LanceVectorStore | UnavailableVectorStore
    embedding: OpenAICompatibleEmbeddingProvider
    reranker: CohereCompatibleRerankProvider

    @property
    def consent_identity(self) -> str:
        value = f"{self.embedding.identity}\0{self.reranker.identity}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def table_id(self) -> str:
        return self.config.knowledge.table_id

    def authorization_is_current(self) -> bool:
        try:
            current = load_config(self.config_path)
            if not current.knowledge.enabled:
                return False
            embedding = _profile(
                current.embedding_profiles,
                current.knowledge.embedding_profile,
                "embedding",
            )
            reranker = _profile(
                current.rerank_profiles,
                current.knowledge.rerank_profile,
                "rerank",
            )
        except (OSError, RuntimeError, ValueError):
            return False
        value = f"{embedding.identity}\0{reranker.identity}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest() == (
            self.consent_identity
        )

    @classmethod
    def load(
        cls,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        *,
        readonly: bool = False,
    ) -> KnowledgeRuntime:
        resolved_config = Path(config_path).expanduser()
        if not resolved_config.is_absolute():
            resolved_config = resolved_config.resolve()
        config = load_config(resolved_config)
        return cls.from_config(config, resolved_config, readonly=readonly)

    @classmethod
    def from_config(
        cls,
        config: KnowledgeAppConfig,
        config_path: Path,
        *,
        readonly: bool = False,
        coordinator_lock: FileLock | None = None,
    ) -> KnowledgeRuntime:
        if not config.knowledge.enabled:
            raise RuntimeError("knowledge backend is disabled")
        embedding_profile = _profile(
            config.embedding_profiles,
            config.knowledge.embedding_profile,
            "embedding",
        )
        rerank_profile = _profile(
            config.rerank_profiles,
            config.knowledge.rerank_profile,
            "rerank",
        )
        store_root = resolve_knowledge_store(config, config_path)
        database_path = store_root / "knowledge.sqlite3"
        database = _open_database(
            database_path,
            readonly=readonly,
            coordinator_lock=coordinator_lock,
        )
        authorization_guard = AuthorizationGuard(store_root / "indexer.lock")
        if not readonly:
            authorization_guard.ensure_writable()
        embedding = OpenAICompatibleEmbeddingProvider(embedding_profile)
        reranker = CohereCompatibleRerankProvider(rerank_profile)
        embedding_space_id = hashlib.sha256(
            embedding.identity.encode("utf-8")
        ).hexdigest()
        computed_lance_table = (
            f"{config.knowledge.table_id}_"
            f"{embedding_space_id[:12]}"
        )
        lance_table = database.active_lance_table(
            config.knowledge.table_id,
            embedding_identity=embedding.identity,
        ) or computed_lance_table
        try:
            vectors: LanceVectorStore | UnavailableVectorStore = LanceVectorStore(
                store_root / "lancedb",
                table_name=lance_table,
                dimensions=embedding.dimensions,
                embedding_space_id=embedding_space_id,
                distance_metric=embedding_profile.distance_metric,
                readonly=readonly,
            )
        except Exception as exc:
            if not readonly:
                raise
            LOGGER.debug(
                "vector backend unavailable; keeping lexical search available: %s: %s",
                type(exc).__name__,
                exc,
            )
            vectors = UnavailableVectorStore(exc)
        if not readonly:
            database.register_table(
                table_id=config.knowledge.table_id,
                lance_table=computed_lance_table,
                embedding_identity=embedding.identity,
                rerank_identity=reranker.identity,
                retrieval_limits={
                    "fts_candidates": config.knowledge.keyword_recall_limit,
                    "vector_candidates": config.knowledge.vector_recall_limit,
                    "rerank_candidates": rerank_profile.candidate_limit,
                    "max_chunks_per_source_file": (
                        config.knowledge.max_chunks_per_source_file
                    ),
                },
            )
            valid_prefix = re.compile(
                rf"^{re.escape(config.knowledge.table_id)}_[0-9a-f]{{12}}$"
            )
            if isinstance(vectors, LanceVectorStore):
                for table_name in vectors.list_tables():
                    if valid_prefix.fullmatch(table_name):
                        database.register_vector_table(
                            config.knowledge.table_id, table_name
                        )
                    else:
                        LOGGER.warning(
                            "ignoring unregistered LanceDB table outside the "
                            "known embedding table rule: %s",
                            table_name,
                        )
        return cls(
            config=config,
            config_path=config_path,
            store_root=store_root,
            database=database,
            authorization_guard=authorization_guard,
            vectors=vectors,
            embedding=embedding,
            reranker=reranker,
        )


def resolve_knowledge_store(
    config: KnowledgeAppConfig,
    config_path: Path,
) -> Path:
    store_root = Path(config.knowledge.store).expanduser()
    if not store_root.is_absolute():
        store_root = config_path.parent / store_root
    return store_root.resolve()


def _open_database(
    database_path: Path,
    *,
    readonly: bool,
    coordinator_lock: FileLock | None,
) -> KnowledgeDB:
    exists = database_path.is_file()
    current_schema = read_schema_version(database_path)
    if current_schema is not None and current_schema > SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported knowledge schema version: "
            f"{current_schema}; this code supports up to {SCHEMA_VERSION}"
        )
    if readonly:
        return KnowledgeDB(database_path, readonly=True)
    if exists and current_schema is None:
        raise RuntimeError(
            "knowledge database schema version is unavailable; refusing to "
            "modify an existing database"
        )
    if exists and current_schema == SCHEMA_VERSION:
        return KnowledgeDB(database_path, initialize=False)

    database_path.parent.mkdir(parents=True, exist_ok=True)
    expected_lock_path = (database_path.parent / "coordinator.lock").resolve()
    if coordinator_lock is not None:
        actual_lock_path = Path(coordinator_lock.lock_file).resolve()
        if not coordinator_lock.is_locked or actual_lock_path != expected_lock_path:
            raise RuntimeError("knowledge database coordinator lock is not held")
        return _initialize_database_locked(database_path)

    coordinator = FileLock(expected_lock_path)
    try:
        with coordinator.acquire(timeout=0):
            return _initialize_database_locked(database_path)
    except Timeout as exc:
        source = "uninitialized" if current_schema is None else str(current_schema)
        raise RuntimeError(
            "knowledge database upgrade required from schema "
            f"{source} to {SCHEMA_VERSION}; stop the running "
            "keepygaga-rag indexer and retry"
        ) from exc


def _initialize_database_locked(database_path: Path) -> KnowledgeDB:
    current_schema = read_schema_version(database_path)
    if database_path.is_file() and current_schema is None:
        raise RuntimeError(
            "knowledge database schema version is unavailable; refusing to "
            "modify an existing database"
        )
    if current_schema is not None and current_schema > SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported knowledge schema version: "
            f"{current_schema}; this code supports up to {SCHEMA_VERSION}"
        )
    if current_schema == SCHEMA_VERSION:
        return KnowledgeDB(database_path, initialize=False)
    return KnowledgeDB(database_path, initialize=True)


def _profile[T](
    profiles: dict[str, T], name: str, kind: str
) -> T:
    try:
        return profiles[name]
    except KeyError as exc:
        raise RuntimeError(f"{kind} profile is not configured: {name}") from exc


def provider_summary(
    embedding: EmbeddingProfileConfig,
    rerank: RerankProfileConfig,
) -> dict[str, object]:
    return {
        "embedding": {
            "provider": embedding.provider,
            "model": embedding.model,
            "dimensions": embedding.dimensions,
            "api_key_env": embedding.api_key_env,
        },
        "rerank": {
            "provider": rerank.provider,
            "model": rerank.model,
            "api_key_env": rerank.api_key_env,
        },
    }
