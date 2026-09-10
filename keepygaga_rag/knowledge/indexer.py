from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import cast

from filelock import Timeout

from keepygaga_rag.config import KNOWLEDGE_HARD_EXCLUDED_DIRS
from keepygaga_rag.knowledge.authorization import (
    AuthorizationGuard,
    AuthorizationGuardUnavailable,
)
from keepygaga_rag.knowledge.chunking import TextChunk, chunk_text
from keepygaga_rag.knowledge.db import (
    ChunkRecord,
    KnowledgeDB,
    SourceFileRecord,
    SourceRecord,
)
from keepygaga_rag.knowledge.providers import (
    EmbeddingProvider,
    ProviderActionRequiredError,
)
from keepygaga_rag.knowledge.vectors import VectorStore

DEFAULT_HARD_EXCLUDED_DIRS = frozenset(
    {
        *KNOWLEDGE_HARD_EXCLUDED_DIRS,
        ".obsidian",
        ".git",
        ".keepygaga",
        ".venv",
        "node_modules",
    }
)
_CASEFOLDED_HARD_EXCLUDED_DIRS = frozenset(
    name.casefold() for name in DEFAULT_HARD_EXCLUDED_DIRS
)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _reported_sync_errors(
    errors: Sequence[str],
    *,
    provider_action_error: str = "",
    limit: int = 5,
) -> list[str]:
    reported = list(errors[:limit])
    if provider_action_error and provider_action_error not in reported:
        reported[-1] = provider_action_error
    return reported


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    relative_path: str
    mtime_ns: int
    size: int
    device: int
    inode: int


class SourceSafetyError(RuntimeError):
    """Raised when a registered source changes into an unsafe filesystem object."""


class IndexerLockUnavailable(RuntimeError):
    """Raised when another process currently owns the index write lock."""


def _raise_scan_error(error: OSError) -> None:
    path = getattr(error, "filename", None) or "source"
    raise SourceSafetyError(
        f"source scan failed for {path}: {error}"
    ) from error


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _is_redirect(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or _is_reparse_point(file_stat)


def _matches_discovered_file(
    file_stat: os.stat_result,
    discovered: DiscoveredFile,
) -> bool:
    return (
        file_stat.st_dev == discovered.device
        and file_stat.st_ino == discovered.inode
        and file_stat.st_mtime_ns == discovered.mtime_ns
        and file_stat.st_size == discovered.size
    )


def _source_root_stat(root: Path) -> os.stat_result:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise SourceSafetyError("source root is unavailable") from exc
    if _is_redirect(root_stat):
        raise SourceSafetyError("source root is a symbolic link or reparse point")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SourceSafetyError("source root is not a directory")
    return root_stat


def _canonical_source_root(root: Path) -> Path:
    registered = Path(os.path.abspath(os.fspath(root.expanduser())))
    try:
        canonical = registered.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SourceSafetyError("source root is unavailable") from exc
    if os.path.normcase(os.fspath(registered)) != os.path.normcase(
        os.fspath(canonical)
    ):
        raise SourceSafetyError(
            "source root path is redirected through a symbolic link or junction"
        )
    if any(
        part.casefold() in _CASEFOLDED_HARD_EXCLUDED_DIRS
        for part in canonical.parts
    ):
        raise SourceSafetyError("source root is a hard-excluded directory")
    return canonical


def _verify_source_root(root: Path, expected: os.stat_result) -> None:
    current = _source_root_stat(root)
    if not _same_file_identity(current, expected):
        raise SourceSafetyError("source root changed during sync")


def _safe_read_file(
    path: Path,
    discovered: DiscoveredFile,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceSafetyError("source file is unavailable or unsafe") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not _matches_discovered_file(opened, discovered)
        ):
            raise SourceSafetyError("source file changed or is unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(descriptor)
        if (
            not _same_file_identity(final, opened)
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_size != opened.st_size
        ):
            raise SourceSafetyError("source file changed while reading")
        return b"".join(chunks)
    except SourceSafetyError:
        raise
    except OSError as exc:
        raise SourceSafetyError("source file is unavailable or unsafe") from exc
    finally:
        os.close(descriptor)


def _path_is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _matches_pattern(path: PurePosixPath, pattern: str) -> bool:
    if "/" not in pattern:
        return path.match(pattern)

    path_parts = path.parts
    pattern_parts = PurePosixPath(pattern).parts
    pending = [(0, 0)]
    visited: set[tuple[int, int]] = set()
    while pending:
        path_index, pattern_index = pending.pop()
        state = (path_index, pattern_index)
        if state in visited:
            continue
        visited.add(state)
        if pattern_index == len(pattern_parts):
            if path_index == len(path_parts):
                return True
            continue
        if pattern_parts[pattern_index] == "**":
            pending.append((path_index, pattern_index + 1))
            if path_index < len(path_parts):
                pending.append((path_index + 1, pattern_index))
            continue
        if (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], pattern_parts[pattern_index])
        ):
            pending.append((path_index + 1, pattern_index + 1))
    return False


def _matches(path: PurePosixPath, patterns: Sequence[str]) -> bool:
    return any(_matches_pattern(path, pattern) for pattern in patterns)


def path_matches_patterns(
    relative_path: str,
    patterns: Sequence[str],
) -> bool:
    return _matches(PurePosixPath(relative_path), patterns)


def source_path_is_selected(
    relative_path: str,
    source: SourceRecord,
) -> bool:
    pure = PurePosixPath(relative_path)
    return (
        bool(pure.parts)
        and not any(part.startswith(".") for part in pure.parts)
        and not any(
            part.casefold() in _CASEFOLDED_HARD_EXCLUDED_DIRS
            for part in pure.parts
        )
        and not _matches(pure, source.exclude_patterns)
        and _matches(pure, source.include_patterns)
    )


def source_root_is_safe(root: str | Path) -> bool:
    path = Path(root).expanduser()
    try:
        _source_root_stat(path)
        _canonical_source_root(path)
    except SourceSafetyError:
        return False
    return True


def validate_source_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"source root does not exist: {resolved}")
    excluded = next(
        (
            part
            for part in resolved.parts
            if part.casefold() in _CASEFOLDED_HARD_EXCLUDED_DIRS
        ),
        None,
    )
    if excluded is not None:
        raise ValueError(f"source root cannot include {excluded}")
    return resolved


def discover_files(
    source: SourceRecord,
    *,
    expected_root_stat: os.stat_result | None = None,
) -> Iterator[DiscoveredFile]:
    root = Path(source.absolute_path)
    root_stat = _source_root_stat(root)
    _canonical_source_root(root)
    if expected_root_stat is not None and not _same_file_identity(
        root_stat, expected_root_stat
    ):
        raise SourceSafetyError("source root changed during sync")
    try:
        walker = os.walk(
            root,
            followlinks=False,
            onerror=_raise_scan_error,
        )
        for directory, directory_names, file_names in walker:
            _verify_source_root(root, root_stat)
            current = Path(directory)
            relative_directory = current.relative_to(root)
            kept_directories: list[str] = []
            for name in directory_names:
                candidate = relative_directory / name
                pure = PurePosixPath(candidate.as_posix())
                if (
                    name.casefold() in _CASEFOLDED_HARD_EXCLUDED_DIRS
                    or name.startswith(".")
                    or _matches(pure, source.exclude_patterns)
                ):
                    continue
                try:
                    candidate_stat = (current / name).lstat()
                except OSError as exc:
                    raise SourceSafetyError(
                        f"source scan failed for {current / name}: {exc}"
                    ) from exc
                if _is_redirect(candidate_stat) or not stat.S_ISDIR(
                    candidate_stat.st_mode
                ):
                    continue
                kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in file_names:
                path = current / name
                if name.startswith("."):
                    continue
                relative = path.relative_to(root)
                pure = PurePosixPath(relative.as_posix())
                if not source_path_is_selected(pure.as_posix(), source):
                    continue
                try:
                    file_stat = path.lstat()
                except OSError as exc:
                    raise SourceSafetyError(
                        f"source scan failed for {path}: {exc}"
                    ) from exc
                if _is_redirect(file_stat) or not stat.S_ISREG(file_stat.st_mode):
                    continue
                yield DiscoveredFile(
                    path=path,
                    relative_path=pure.as_posix(),
                    mtime_ns=file_stat.st_mtime_ns,
                    size=file_stat.st_size,
                    device=file_stat.st_dev,
                    inode=file_stat.st_ino,
                )
    except SourceSafetyError:
        raise
    except OSError as exc:
        raise SourceSafetyError(f"source scan failed: {exc}") from exc
    _verify_source_root(root, root_stat)


def preview_source(
    root: Path,
    *,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    target_chars: int,
) -> dict[str, int | str]:
    resolved = validate_source_root(root)
    source = SourceRecord(
        id="preview",
        display_name=resolved.name,
        absolute_path=str(resolved),
        enabled=True,
        auto_sync=True,
        include_patterns=tuple(include_patterns),
        exclude_patterns=tuple(exclude_patterns),
        consent_identity="",
        status="preview",
        last_scan_at="",
        last_success_at="",
        last_error="",
        created_at="",
        updated_at="",
    )
    files = list(discover_files(source))
    total_bytes = sum(item.size for item in files)
    estimated_chunks = sum(
        max(1, (item.size + target_chars - 1) // target_chars) for item in files
    )
    return {
        "root": str(resolved),
        "files": len(files),
        "bytes": total_bytes,
        "estimated_chunks": estimated_chunks,
        "estimated_tokens": total_bytes // 2,
    }


class KnowledgeIndexer:
    def __init__(
        self,
        *,
        database: KnowledgeDB,
        vectors: VectorStore,
        embedding: EmbeddingProvider,
        table_id: str,
        lock_path: Path,
        target_chars: int,
        max_chars: int,
        chunk_mode: str = "structure",
        overlap_chars: int = 0,
        stability_seconds: int,
        missing_confirmations: int,
        consent_identity: str,
        distance_metric: str = "cosine",
        authorization_is_current: Callable[[], bool] | None = None,
        chunk_settings_loader: (
            Callable[
                [], tuple[int, int] | tuple[int, int, str, int] | None
            ]
            | None
        ) = None,
    ):
        self.database = database
        self.vectors = vectors
        self.embedding = embedding
        self.table_id = table_id
        self.authorization_guard = AuthorizationGuard(lock_path)
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.chunk_mode = chunk_mode
        self.overlap_chars = overlap_chars
        self.stability_seconds = stability_seconds
        self.missing_confirmations = missing_confirmations
        self.consent_identity = consent_identity
        self.distance_metric = distance_metric
        self.authorization_is_current = authorization_is_current
        self.chunk_settings_loader = chunk_settings_loader
        self.vector_repair_failures = 0
        self.vector_repair_backoff_until = 0.0
        self.vector_health_check_at = 0.0
        self.vector_health_revision: int | None = None

    def _apply_chunk_settings(
        self,
        settings: tuple[int, int] | tuple[int, int, str, int],
    ) -> None:
        if len(settings) == 2:
            self.target_chars, self.max_chars = settings
            return
        if len(settings) == 4:
            (
                self.target_chars,
                self.max_chars,
                self.chunk_mode,
                self.overlap_chars,
            ) = settings
            return
        raise ValueError("invalid chunk settings")

    def sync_source(self, source: SourceRecord) -> dict[str, int]:
        counters = {
            "scanned_files": 0,
            "indexed_files": 0,
            "reused_vectors": 0,
            "embedded_vectors": 0,
            "missing_files": 0,
        }
        try:
            self.authorization_guard.ensure_writable()
            with self.authorization_guard.acquire_readonly():
                if self.chunk_settings_loader is not None:
                    settings = self.chunk_settings_loader()
                    if settings is not None:
                        self._apply_chunk_settings(settings)
                self._reconcile_derived_state_locked()
                current = self.database.get_source(source.id)
                if current is None:
                    raise KeyError(source.id)
                self._assert_source_authorized(current.id)
                return self._sync_source(current, counters)
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise IndexerLockUnavailable(
                "another knowledge indexer holds the write lock"
            ) from exc

    def reconcile_derived_state(self) -> int:
        try:
            with self.authorization_guard.acquire_exclusive(timeout=0):
                return self._reconcile_derived_state_locked()
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise IndexerLockUnavailable(
                "another knowledge indexer holds the write lock"
            ) from exc

    def reconcile_chunk_settings(self) -> dict[str, int] | None:
        if self.chunk_settings_loader is None:
            return None
        try:
            with self.authorization_guard.acquire_exclusive(timeout=0):
                settings = self.chunk_settings_loader()
                if settings is None:
                    return None
                self._apply_chunk_settings(settings)
                safe_source_ids = [
                    source.id
                    for source in self.database.list_sources()
                    if source_root_is_safe(source.absolute_path)
                ]
                return self.database.prepare_chunk_rebuild(
                    target_chars=self.target_chars,
                    max_chars=self.max_chars,
                    chunk_mode=self.chunk_mode,
                    overlap_chars=self.overlap_chars,
                    consent_identity=self.consent_identity,
                    safe_source_ids=safe_source_ids,
                    reason="chunk-settings-changed",
                )
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise IndexerLockUnavailable(
                "another knowledge indexer holds the write lock"
            ) from exc

    def _reconcile_derived_state_locked(self) -> int:
        scope_pending = [
            source
            for source in self.database.list_sources()
            if source.scope_cleanup_pending
        ]
        for source in scope_pending:
            self._delete_out_of_scope_files(source)
        self.database.cleanup_inactive_chunks()
        deleted = self._reconcile_vector_deletions_locked()
        for source in scope_pending:
            refreshed = self.database.get_source(source.id)
            if refreshed is not None and refreshed.scope_cleanup_pending:
                self.database.complete_source_scope_cleanup(source.id)
        return deleted

    def _reconcile_vector_deletions_locked(self) -> int:
        deleted = 0
        while chunk_ids := self.database.pending_vector_deletions():
            self._delete_vectors_across_history(chunk_ids)
            self.database.complete_vector_deletions(chunk_ids)
            deleted += len(chunk_ids)
        return deleted

    def _delete_source_file(
        self,
        file_id: int,
        *,
        scope_cleanup: bool = False,
    ) -> None:
        chunk_ids = self.database.delete_source_file(
            file_id,
            scope_cleanup=scope_cleanup,
        )
        self._delete_vectors_across_history(chunk_ids)
        self.database.complete_vector_deletions(chunk_ids)

    def _delete_vectors_across_history(self, chunk_ids: Sequence[str]) -> None:
        self.vectors.delete_across_history(
            chunk_ids,
            table_names=self.database.list_vector_tables(self.table_id),
        )

    def _delete_out_of_scope_files(self, source: SourceRecord) -> None:
        for record in self.database.list_source_files(source.id):
            if not source_path_is_selected(record.relative_path, source):
                self._delete_source_file(record.id, scope_cleanup=True)

    def _sync_source(
        self, source: SourceRecord, counters: dict[str, int]
    ) -> dict[str, int]:
        self._assert_source_authorized(source.id)
        run_id = self.database.begin_run(source.id)
        error = ""
        try:
            root = Path(source.absolute_path)
            root_stat = _source_root_stat(root)
            self.vectors.ensure_table()
            discovered = {
                item.relative_path: item
                for item in discover_files(
                    source,
                    expected_root_stat=root_stat,
                )
            }
            _verify_source_root(root, root_stat)
            manifest = {
                item.relative_path: item
                for item in self.database.list_source_files(source.id)
            }
            errors: list[str] = []
            provider_action_error = ""
            counters["scanned_files"] = len(discovered)
            for relative_path, item in discovered.items():
                record = manifest.get(relative_path)
                if record is None:
                    record = self.database.get_or_create_source_file(
                        source_id=source.id,
                        relative_path=relative_path,
                        mtime_ns=item.mtime_ns,
                        size=item.size,
                    )
                try:
                    self._process_file(
                        source,
                        record,
                        item,
                        counters,
                        root=root,
                        root_stat=root_stat,
                    )
                except Exception as exc:
                    counters["failed_files"] = counters.get("failed_files", 0) + 1
                    message = f"{relative_path}: {type(exc).__name__}: {exc}"
                    errors.append(message)
                    if (
                        not provider_action_error
                        and isinstance(exc, ProviderActionRequiredError)
                    ):
                        provider_action_error = message
            _verify_source_root(root, root_stat)
            for relative_path, record in manifest.items():
                if relative_path in discovered:
                    continue
                if (
                    _path_is_regular_file(root / relative_path)
                    and not source_path_is_selected(relative_path, source)
                ):
                    self._delete_source_file(record.id)
                    continue
                counters["missing_files"] += 1
                missing_count, chunk_ids = self.database.mark_file_missing(record.id)
                if missing_count >= self.missing_confirmations:
                    self._delete_source_file(record.id)
            _verify_source_root(root, root_stat)
            if errors:
                reported_errors = _reported_sync_errors(
                    errors,
                    provider_action_error=provider_action_error,
                )
                raise RuntimeError("; ".join(reported_errors))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.database.finish_run(run_id, counters=counters, error=error)
        return counters

    @property
    def _consent_identity(self) -> str:
        return self.consent_identity

    @property
    def _embedding_space_id(self) -> str:
        return hashlib.sha256(
            self.embedding.identity.encode("utf-8")
        ).hexdigest()

    def _process_file(
        self,
        source: SourceRecord,
        record: SourceFileRecord,
        discovered: DiscoveredFile,
        counters: dict[str, int],
        *,
        root: Path,
        root_stat: os.stat_result,
    ) -> None:
        try:
            force_rechunk = record.rechunk_required
            unchanged_stat = (
                not force_rechunk
                and record.active_generation > 0
                and record.mtime_ns == discovered.mtime_ns
                and record.size == discovered.size
                and record.status in {"indexed", "missing_pending"}
            )
            vector_rebuild_required = False
            if unchanged_stat:
                if self._active_vectors_complete(record):
                    self.database.mark_file_unchanged(
                        record.id,
                        mtime_ns=discovered.mtime_ns,
                        size=discovered.size,
                    )
                    return
                vector_rebuild_required = True
            age_seconds = time.time() - (discovered.mtime_ns / 1_000_000_000)
            if (
                not vector_rebuild_required
                and age_seconds < self.stability_seconds
            ):
                self.database.mark_file_waiting(
                    record.id,
                    mtime_ns=discovered.mtime_ns,
                    size=discovered.size,
                )
                return
            _verify_source_root(root, root_stat)
            payload = _safe_read_file(discovered.path, discovered)
            _verify_source_root(root, root_stat)
            digest = hashlib.sha256(payload).hexdigest()
            if (
                not force_rechunk
                and record.active_generation > 0
                and digest == record.sha256
                and self._active_vectors_complete(record)
            ):
                self.database.mark_file_unchanged(
                    record.id,
                    mtime_ns=discovered.mtime_ns,
                    size=discovered.size,
                )
                return
            text = payload.decode("utf-8")
            parsed = chunk_text(
                text,
                source_path=Path(discovered.relative_path),
                target_chars=self.target_chars,
                max_chars=self.max_chars,
                chunk_mode=self.chunk_mode,
                overlap_chars=self.overlap_chars,
                embedding_identity=self.embedding.identity,
            )
            generation = record.active_generation + 1
            chunks = [
                ChunkRecord(
                    chunk_id=f"{record.id}:{generation}:{item.ordinal}",
                    source_file_id=record.id,
                    generation=generation,
                    ordinal=item.ordinal,
                    text=item.text,
                    search_text=item.search_text,
                    heading_path=item.heading_path,
                    title=item.title,
                    filename=item.filename,
                    content_hash=item.content_hash,
                    fts_fields=item.fts_fields,
                    embedding_input_hash=item.embedding_input_hash,
                    start_line=item.start_line,
                    end_line=item.end_line,
                )
                for item in parsed
            ]
            self.database.stage_generation(
                file_id=record.id,
                generation=generation,
                mtime_ns=discovered.mtime_ns,
                size=discovered.size,
                chunks=chunks,
            )
            vector_rows, reused, embedded = self._vectors_for_chunks(
                source, chunks, parsed
            )
            self.vectors.add(vector_rows)
            self.database.complete_vector_deletions(
                str(row["chunk_id"]) for row in vector_rows
            )
            old_chunk_ids = self.database.activate_generation(
                record.id, generation, digest
            )
            self.database.cleanup_old_chunks(
                record.id, generation, old_chunk_ids
            )
            self._delete_vectors_across_history(old_chunk_ids)
            self.database.complete_vector_deletions(old_chunk_ids)
            counters["indexed_files"] += 1
            counters["reused_vectors"] += reused
            counters["embedded_vectors"] += embedded
        except Exception as exc:
            self.database.mark_file_error(
                record.id, f"{type(exc).__name__}: {exc}"
            )
            raise

    def _active_vectors_complete(self, record: SourceFileRecord) -> bool:
        rows = self.database.active_chunk_vector_metadata(record.id)
        if not rows:
            return True
        chunk_ids = [str(row["chunk_id"]) for row in rows]
        metadata_reader = getattr(self.vectors, "get_vector_metadata", None)
        if callable(metadata_reader):
            metadata = cast(
                dict[str, dict[str, str]], metadata_reader(chunk_ids)
            )
            return all(
                metadata.get(chunk_id) == {
                    "embedding_space_id": self._embedding_space_id,
                    "embedding_input_hash": str(row["embedding_input_hash"]),
                }
                for chunk_id, row in zip(chunk_ids, rows, strict=True)
            )
        vectors = self.vectors.get_vectors(chunk_ids)
        return set(vectors) == set(chunk_ids)

    def _vectors_for_chunks(
        self,
        source: SourceRecord,
        chunks: Sequence[ChunkRecord],
        parsed: Sequence[TextChunk],
    ) -> tuple[list[dict[str, object]], int, int]:
        hashes = [chunk.embedding_input_hash for chunk in chunks]
        reusable_ids = self.database.active_chunks_by_hashes(hashes)
        reusable_vector_ids = list(reusable_ids.values())
        metadata_reader = getattr(self.vectors, "get_vector_metadata", None)
        if callable(metadata_reader):
            metadata = cast(
                dict[str, dict[str, str]], metadata_reader(reusable_vector_ids)
            )
            expected_hashes = {
                old_id: embedding_hash
                for embedding_hash, old_id in reusable_ids.items()
            }
            reusable_vector_ids = [
                old_id
                for old_id in reusable_vector_ids
                if (
                    metadata.get(old_id, {}).get("embedding_space_id")
                    == self._embedding_space_id
                    and metadata.get(old_id, {}).get("embedding_input_hash")
                    == expected_hashes.get(old_id, "")
                )
            ]
        old_vectors = self.vectors.get_vectors(reusable_vector_ids)
        vectors: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_indexes: list[int] = []
        for index, chunk in enumerate(chunks):
            old_id = reusable_ids.get(chunk.embedding_input_hash)
            vector = old_vectors.get(old_id, []) if old_id else []
            if vector:
                vectors.append(vector)
            else:
                vectors.append(None)
                missing_indexes.append(index)
                missing_texts.append(parsed[index].embedding_text)
        if missing_texts:
            self._assert_source_authorized(source.id)
            embedded = self.embedding.embed(missing_texts)
            if len(embedded) != len(missing_indexes):
                raise RuntimeError("embedding provider returned an unexpected count")
            for index, vector in zip(missing_indexes, embedded, strict=True):
                vectors[index] = vector
        rows: list[dict[str, object]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if vector is None:  # pragma: no cover
                raise RuntimeError("missing vector after embedding")
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": source.id,
                    "source_file_id": chunk.source_file_id,
                    "embedding_input_hash": chunk.embedding_input_hash,
                    "embedding_space_id": self._embedding_space_id,
                    "generation": chunk.generation,
                    "vector": vector,
                }
            )
        return rows, len(chunks) - len(missing_indexes), len(missing_indexes)

    def _assert_source_authorized(self, source_id: str) -> None:
        if (
            self.authorization_is_current is not None
            and not self.authorization_is_current()
        ):
            raise RuntimeError("knowledge provider configuration changed")
        source = self.database.get_source(source_id)
        if (
            source is None
            or not source.enabled
            or source.scope_cleanup_pending
            or source.consent_identity != self._consent_identity
        ):
            raise RuntimeError("source provider consent is missing or stale")
