from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from filelock import FileLock, Timeout

from keepygaga_rag.config import (
    DEFAULT_CONFIG_PATH,
    KnowledgeAppConfig,
    load_config,
    update_knowledge_chunk_settings,
    update_knowledge_retrieval_settings,
)
from keepygaga_rag.knowledge.api import knowledge_search_diagnostics
from keepygaga_rag.knowledge.authorization import AuthorizationGuardUnavailable
from keepygaga_rag.knowledge.db import SourceRecord
from keepygaga_rag.knowledge.indexer import (
    SourceSafetyError,
    discover_files,
    path_matches_patterns,
    preview_source,
    source_path_is_selected,
    source_root_is_safe,
    validate_source_root,
)
from keepygaga_rag.knowledge.retry import automatic_retry_at
from keepygaga_rag.knowledge.runtime import (
    KnowledgeRuntime,
    UnavailableVectorStore,
    provider_summary,
    resolve_knowledge_store,
)


class KnowledgeDashboardError(ValueError):
    pass


MAX_SCOPE_NODES = 1_000
RECENT_JOB_LIMIT = 10


def _missing_provider_env(config: KnowledgeAppConfig) -> list[str]:
    embedding_profiles = config.embedding_profiles
    rerank_profiles = config.rerank_profiles
    knowledge = config.knowledge
    names: list[str] = []
    for profiles, profile_name in (
        (embedding_profiles, knowledge.embedding_profile),
        (rerank_profiles, knowledge.rerank_profile),
    ):
        profile = profiles.get(profile_name)
        api_key_env = getattr(profile, "api_key_env", "") if profile else ""
        if api_key_env and not os.environ.get(api_key_env, "").strip():
            names.append(str(api_key_env))
    return list(dict.fromkeys(names))


def _require_provider_credentials(config: KnowledgeAppConfig) -> None:
    missing = _missing_provider_env(config)
    if missing:
        raise KnowledgeDashboardError(
            "缺少 Provider 凭据，请先配置环境变量：" + "、".join(missing)
        )


def _coordinator_state(store_root: Path) -> dict[str, object]:
    lock_path = store_root / "coordinator.lock"
    state: dict[str, object] = {
        "path": str(lock_path),
        "status": "stopped",
        "lock_held": False,
        "running": False,
    }
    if not lock_path.is_file():
        state["message"] = "coordinator.lock 不存在，Indexer 当前未运行。"
        return state
    lock = FileLock(lock_path)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        state.update(
            status="running",
            lock_held=True,
            running=True,
            message="Indexer coordinator 正在持有 coordinator.lock。",
        )
    except OSError as exc:
        state.update(
            status="unknown",
            message=f"无法检查 coordinator.lock：{type(exc).__name__}: {exc}",
        )
    else:
        lock.release()
        state["message"] = "coordinator.lock 当前未被持有。"
    return state


def _vector_backend_state(runtime: KnowledgeRuntime) -> tuple[bool, str]:
    if isinstance(runtime.vectors, UnavailableVectorStore):
        return False, f"{type(runtime.vectors.error).__name__}: {runtime.vectors.error}"
    try:
        runtime.vectors.ensure_table()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _is_within(path: str, directory: str) -> bool:
    return directory == "." or path == directory or path.startswith(f"{directory}/")


def _scope_tree(
    source: SourceRecord,
    *,
    available_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    max_nodes: int = MAX_SCOPE_NODES,
) -> dict[str, object]:
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    candidate_source = replace(
        source,
        include_patterns=tuple(available_patterns),
        exclude_patterns=tuple(exclude_patterns),
    )
    try:
        files = []
        directories = {"."}
        truncated = False
        for item in discover_files(candidate_source):
            parents = {
                parent.as_posix()
                for parent in PurePosixPath(item.relative_path).parents
                if parent.as_posix() != "."
            }
            if len(files) + len(directories | parents) + 1 > max_nodes:
                truncated = True
                break
            files.append(item)
            directories.update(parents)
    except SourceSafetyError:
        return {
            "items": [],
            "selection": [],
            "selected_files": 0,
            "total_files": 0,
            "available": False,
            "truncated": False,
            "node_limit": max_nodes,
            "error": "数据源根目录当前不可安全读取；索引已停用。",
        }
    selected_files = {
        item.relative_path
        for item in files
        if path_matches_patterns(item.relative_path, source.include_patterns)
        and not path_matches_patterns(item.relative_path, source.exclude_patterns)
    }
    directories = {"."}
    total_by_directory: dict[str, int] = {".": 0}
    selected_by_directory: dict[str, int] = {".": 0}
    for item in files:
        parents = [
            parent.as_posix()
            for parent in PurePosixPath(item.relative_path).parents
            if parent.as_posix() != "."
        ]
        for directory in (".", *parents):
            directories.add(directory)
            total_by_directory[directory] = (
                total_by_directory.get(directory, 0) + 1
            )
            if item.relative_path in selected_files:
                selected_by_directory[directory] = (
                    selected_by_directory.get(directory, 0) + 1
                )

    items: list[dict[str, object]] = []
    for directory in directories:
        total = total_by_directory.get(directory, 0)
        selected = selected_by_directory.get(directory, 0)
        state = (
            "full"
            if total > 0 and selected == total
            else "partial"
            if selected
            else "off"
        )
        items.append(
            {
                "path": directory,
                "name": (
                    Path(source.absolute_path).name
                    if directory == "."
                    else PurePosixPath(directory).name
                ),
                "kind": "directory",
                "depth": 0 if directory == "." else len(PurePosixPath(directory).parts),
                "state": state,
                "files": total,
            }
        )
    for item in files:
        items.append(
            {
                "path": item.relative_path,
                "name": PurePosixPath(item.relative_path).name,
                "kind": "file",
                "depth": len(PurePosixPath(item.relative_path).parts),
                "state": (
                    "full" if item.relative_path in selected_files else "off"
                ),
                "files": 1,
            }
        )
    items.sort(
        key=lambda item: (
            () if item["path"] == "." else PurePosixPath(str(item["path"])).parts,
            0 if item["kind"] == "directory" else 1,
        )
    )

    full_directories = sorted(
        (
            str(item["path"])
            for item in items
            if item["kind"] == "directory" and item["state"] == "full"
        ),
        key=lambda path: (0 if path == "." else len(PurePosixPath(path).parts), path),
    )
    selection: list[str] = []
    for directory in full_directories:
        if any(_is_within(directory, selected) for selected in selection):
            continue
        selection.append(directory)
    for relative_path in sorted(selected_files):
        if any(_is_within(relative_path, selected) for selected in selection):
            continue
        selection.append(relative_path)
    if truncated:
        selection = []
    return {
        "items": items,
        "selection": selection,
        "selected_files": len(selected_files),
        "total_files": len(files),
        "available": True,
        "truncated": truncated,
        "node_limit": max_nodes,
        "error": (
            f"目录节点超过 {max_nodes} 个显示上限；请缩小预检目录后再选择范围。"
            "当前结果不能保存，以免误写入不完整范围。"
            if truncated
            else ""
        ),
    }


def _scope_patterns(
    scope: dict[str, object],
    selected_paths: Sequence[str],
    *,
    available_patterns: Sequence[str],
) -> tuple[str, ...]:
    items = scope.get("items", [])
    if not isinstance(items, list):
        raise KnowledgeDashboardError("无法读取数据源范围。")
    if scope.get("truncated") is True:
        raise KnowledgeDashboardError(
            str(scope.get("error") or "数据源范围超过当前显示上限，请缩小目录后重试。")
        )
    by_path = {
        str(item["path"]): str(item["kind"])
        for item in items
        if isinstance(item, dict) and "path" in item and "kind" in item
    }
    normalized = tuple(dict.fromkeys(str(path) for path in selected_paths))
    invalid = sorted(set(normalized) - set(by_path))
    if invalid:
        raise KnowledgeDashboardError(
            f"范围中包含无效路径：{', '.join(invalid[:3])}"
        )
    if "." in normalized:
        return tuple(available_patterns)
    directories = sorted(
        (path for path in normalized if by_path[path] == "directory"),
        key=lambda path: (len(PurePosixPath(path).parts), path),
    )
    selected_directories: list[str] = []
    for directory in directories:
        if any(_is_within(directory, parent) for parent in selected_directories):
            continue
        selected_directories.append(directory)
    patterns: list[str] = []
    for directory in selected_directories:
        for pattern in available_patterns:
            relative_pattern = pattern.lstrip("/")
            patterns.append(f"{directory}/{relative_pattern}")
            if relative_pattern.startswith("**/"):
                patterns.append(
                    f"{directory}/{relative_pattern.removeprefix('**/')}"
                )
    for path in normalized:
        if by_path[path] != "file":
            continue
        if any(_is_within(path, directory) for directory in selected_directories):
            continue
        patterns.append(path)
    return tuple(dict.fromkeys(patterns))


class KnowledgeDashboardService:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path.expanduser().resolve()

    def snapshot(self) -> dict[str, object]:
        config = load_config(self.config_path)
        embedding = config.embedding_profiles.get(config.knowledge.embedding_profile)
        rerank = config.rerank_profiles.get(config.knowledge.rerank_profile)
        configured_rerank_limit = rerank.candidate_limit if rerank else 20
        provider = (
            provider_summary(embedding, rerank)
            if embedding is not None and rerank is not None
            else {
                "embedding": {
                    "api_key_env": getattr(embedding, "api_key_env", "")
                },
                "rerank": {"api_key_env": getattr(rerank, "api_key_env", "")},
            }
        )
        missing_provider_env = _missing_provider_env(config)
        base: dict[str, object] = {
            "enabled": bool(config.knowledge.enabled),
            "backend_status": "disabled"
            if not config.knowledge.enabled
            else "not_initialized",
            "sources": [],
            "table_id": config.knowledge.table_id,
            "scan_interval_seconds": config.knowledge.scan_interval_seconds,
            "chunk_target_chars": config.knowledge.chunk_target_chars,
            "chunk_max_chars": config.knowledge.chunk_max_chars,
            "chunk_mode": config.knowledge.chunk_mode,
            "chunk_overlap_chars": config.knowledge.chunk_overlap_chars,
            "vector_recall_limit": config.knowledge.vector_recall_limit,
            "keyword_recall_limit": config.knowledge.keyword_recall_limit,
            "rerank_candidate_limit": configured_rerank_limit,
            "max_chunks_per_source_file": (
                config.knowledge.max_chunks_per_source_file
            ),
            "provider": provider,
            "embedding_key_ready": bool(
                embedding
                and os.environ.get(embedding.api_key_env, "").strip()
            ),
            "rerank_key_ready": bool(
                rerank and os.environ.get(rerank.api_key_env, "").strip()
            ),
            "credentials_ready": not missing_provider_env,
            "missing_provider_env": missing_provider_env,
            "vector_ready": False,
            "vector_status": "not_initialized",
            "vector_error": "",
            "queue": {
                "queued": 0,
                "running": 0,
                "succeeded": 0,
                "failed": 0,
                "jobs": [],
            },
            "recent_runs": [],
            "failed_files": [],
        }
        store_root = resolve_knowledge_store(config, self.config_path)
        base["indexer"] = _coordinator_state(store_root)
        if not config.knowledge.enabled:
            base["message"] = (
                "knowledge backend is disabled in keepygaga-rag.toml; "
                "search is unavailable"
            )
            return base
        if embedding is None or rerank is None:
            base["message"] = "knowledge provider profiles are not configured"
            return base
        try:
            runtime = KnowledgeRuntime.load(self.config_path, readonly=True)
        except FileNotFoundError:
            base["message"] = "knowledge index is not initialized"
            return base
        vector_ready, vector_error = _vector_backend_state(runtime)
        base["vector_ready"] = vector_ready
        base["vector_status"] = "ready" if vector_ready else "unavailable"
        base["vector_error"] = vector_error
        sources: list[dict[str, object]] = []
        failed_files = runtime.database.list_failed_source_files(limit=100)
        failed_by_source: dict[str, list[dict[str, object]]] = {}
        for item in failed_files:
            failed_by_source.setdefault(str(item["source_id"]), []).append(item)
        for source in runtime.database.list_sources():
            next_scan_at = ""
            if source.auto_sync and source.last_scan_at:
                next_scan = automatic_retry_at(
                    source,
                    regular_interval_seconds=config.knowledge.scan_interval_seconds,
                )
                next_scan_at = next_scan.isoformat() if next_scan is not None else ""
            sources.append(
                {
                    "id": source.id,
                    "display_name": source.display_name,
                    "absolute_path": source.absolute_path,
                    "enabled": source.enabled,
                    "auto_sync": source.auto_sync,
                    "status": source.status,
                    "scope_cleanup_pending": source.scope_cleanup_pending,
                    "last_scan_at": source.last_scan_at,
                    "last_success_at": source.last_success_at,
                    "next_scan_at": next_scan_at,
                    "last_error": source.last_error,
                    "consent_current": (
                        source.consent_identity == runtime.consent_identity
                    ),
                    "scope": _scope_tree(
                        source,
                        available_patterns=config.knowledge.include_patterns,
                        exclude_patterns=config.knowledge.exclude_patterns,
                        max_nodes=MAX_SCOPE_NODES,
                    ),
                    "stats": runtime.database.source_stats(source.id),
                    "failed_files": failed_by_source.get(source.id, []),
                }
            )
        open_jobs = runtime.database.list_sync_jobs(
            limit=20,
            statuses=("queued", "running"),
        )
        recent_jobs = runtime.database.list_sync_jobs(
            limit=RECENT_JOB_LIMIT,
            statuses=("succeeded", "failed"),
        )
        open_job_counts = runtime.database.sync_job_counts()
        recent_closed_counts = {
            "succeeded": sum(
                str(job["status"]) == "succeeded" for job in recent_jobs
            ),
            "failed": sum(
                str(job["status"]) == "failed" for job in recent_jobs
            ),
        }
        base.update(
            {
                "backend_status": "ready" if vector_ready else "degraded",
                "message": (
                    "knowledge backend is ready"
                    if vector_ready
                    else "knowledge backend is unavailable: vector store is not ready"
                ),
                "sources": sources,
                "consent_identity": runtime.consent_identity,
                "queue": {
                    **open_job_counts,
                    **recent_closed_counts,
                    "closed_counts_scope": "recent_jobs",
                    "closed_counts_limit": RECENT_JOB_LIMIT,
                    "jobs": open_jobs + recent_jobs,
                },
                "recent_runs": runtime.database.list_sync_runs(limit=30),
                "failed_files": failed_files,
            }
        )
        return base

    def update_chunk_settings(
        self,
        *,
        target_value: str,
        max_value: str,
        mode_value: str | None = None,
        overlap_value: str | None = None,
    ) -> dict[str, int]:
        target_chars = self._positive_integer(
            target_value,
            "软目标",
        )
        max_chars = self._positive_integer(max_value, "硬上限")
        if target_chars > max_chars:
            raise KnowledgeDashboardError("软目标不能大于硬上限。")
        requested_mode = (
            None if mode_value is None else self._chunk_mode(mode_value)
        )
        requested_overlap = (
            None
            if overlap_value is None
            else self._nonnegative_integer(overlap_value, "重叠字符数")
        )
        if requested_overlap is not None and requested_overlap >= max_chars:
            raise KnowledgeDashboardError("重叠字符数必须小于硬上限。")

        initial = load_config(self.config_path)
        if initial.knowledge.enabled:
            _require_provider_credentials(initial)
        if not initial.knowledge.enabled:
            current = load_config(self.config_path)
            chunk_mode = (
                current.knowledge.chunk_mode
                if requested_mode is None
                else requested_mode
            )
            overlap_chars = (
                current.knowledge.chunk_overlap_chars
                if requested_overlap is None
                else requested_overlap
            )
            if overlap_chars >= max_chars:
                raise KnowledgeDashboardError("重叠字符数必须小于硬上限。")
            changed = (
                target_chars != current.knowledge.chunk_target_chars
                or max_chars != current.knowledge.chunk_max_chars
                or chunk_mode != current.knowledge.chunk_mode
                or overlap_chars != current.knowledge.chunk_overlap_chars
            )
            update_knowledge_chunk_settings(
                self.config_path,
                target_chars=target_chars,
                max_chars=max_chars,
                chunk_mode=chunk_mode,
                overlap_chars=overlap_chars,
            )
            return {
                "changed": int(changed),
                "marked_files": 0,
                "queued_sources": 0,
                "new_jobs": 0,
                "skipped_disabled": 0,
                "skipped_consent": 0,
                "skipped_scope_cleanup": 0,
                "skipped_unsafe": 0,
                "deferred_sources": 0,
            }

        bootstrap_runtime = KnowledgeRuntime.from_config(
            initial,
            self.config_path,
        )
        try:
            with bootstrap_runtime.authorization_guard.acquire_exclusive(timeout=0):
                current = load_config(self.config_path)
                chunk_mode = (
                    current.knowledge.chunk_mode
                    if requested_mode is None
                    else requested_mode
                )
                overlap_chars = (
                    current.knowledge.chunk_overlap_chars
                    if requested_overlap is None
                    else requested_overlap
                )
                if overlap_chars >= max_chars:
                    raise KnowledgeDashboardError("重叠字符数必须小于硬上限。")
                changed = (
                    target_chars != current.knowledge.chunk_target_chars
                    or max_chars != current.knowledge.chunk_max_chars
                    or chunk_mode != current.knowledge.chunk_mode
                    or overlap_chars != current.knowledge.chunk_overlap_chars
                )
                empty_result = {
                    "changed": int(changed),
                    "marked_files": 0,
                    "queued_sources": 0,
                    "new_jobs": 0,
                    "skipped_disabled": 0,
                    "skipped_consent": 0,
                    "skipped_scope_cleanup": 0,
                    "skipped_unsafe": 0,
                    "deferred_sources": 0,
                }
                if not current.knowledge.enabled:
                    update_knowledge_chunk_settings(
                        self.config_path,
                        target_chars=target_chars,
                        max_chars=max_chars,
                        chunk_mode=chunk_mode,
                        overlap_chars=overlap_chars,
                    )
                    return empty_result

                runtime = KnowledgeRuntime.from_config(
                    current,
                    self.config_path,
                )
                try:
                    update_knowledge_chunk_settings(
                        self.config_path,
                        target_chars=target_chars,
                        max_chars=max_chars,
                        chunk_mode=chunk_mode,
                        overlap_chars=overlap_chars,
                    )
                    safe_source_ids = [
                        source.id
                        for source in runtime.database.list_sources()
                        if source_root_is_safe(source.absolute_path)
                    ]
                    result = runtime.database.prepare_chunk_rebuild(
                        target_chars=target_chars,
                        max_chars=max_chars,
                        chunk_mode=chunk_mode,
                        overlap_chars=overlap_chars,
                        consent_identity=runtime.consent_identity,
                        safe_source_ids=safe_source_ids,
                        reason="chunk-settings-changed",
                    )
                except Exception:
                    update_knowledge_chunk_settings(
                        self.config_path,
                        target_chars=current.knowledge.chunk_target_chars,
                        max_chars=current.knowledge.chunk_max_chars,
                        chunk_mode=current.knowledge.chunk_mode,
                        overlap_chars=current.knowledge.chunk_overlap_chars,
                    )
                    raise
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise KnowledgeDashboardError(
                "索引器正在写入，请稍后再修改文字分块设置。"
            ) from exc
        return result

    def update_retrieval_settings(
        self,
        *,
        vector_value: str,
        keyword_value: str,
        rerank_value: str,
        max_chunks_value: str,
    ) -> dict[str, int]:
        vector_limit = self._bounded_integer(
            vector_value,
            "向量召回数",
            minimum=1,
            maximum=100,
        )
        keyword_limit = self._bounded_integer(
            keyword_value,
            "关键词召回数",
            minimum=1,
            maximum=100,
        )
        rerank_limit = self._bounded_integer(
            rerank_value,
            "重排候选数",
            minimum=1,
            maximum=100,
        )
        max_chunks = self._bounded_integer(
            max_chunks_value,
            "单篇笔记分块上限",
            minimum=1,
            maximum=10,
        )
        current = load_config(self.config_path)
        if current.knowledge.rerank_profile not in current.rerank_profiles:
            raise KnowledgeDashboardError("当前重排配置不存在，无法保存检索设置。")
        changed = (
            vector_limit != current.knowledge.vector_recall_limit
            or keyword_limit != current.knowledge.keyword_recall_limit
            or rerank_limit
            != current.rerank_profiles[
                current.knowledge.rerank_profile
            ].candidate_limit
            or max_chunks != current.knowledge.max_chunks_per_source_file
        )
        update_knowledge_retrieval_settings(
            self.config_path,
            vector_recall_limit=vector_limit,
            keyword_recall_limit=keyword_limit,
            rerank_candidate_limit=rerank_limit,
            max_chunks_per_source_file=max_chunks,
            rerank_profile=current.knowledge.rerank_profile,
        )
        return {
            "changed": int(changed),
            "vector_recall_limit": vector_limit,
            "keyword_recall_limit": keyword_limit,
            "rerank_candidate_limit": rerank_limit,
            "max_chunks_per_source_file": max_chunks,
        }

    def diagnose_search(self, query_value: str) -> dict[str, object]:
        query = query_value.strip()
        if not query:
            raise KnowledgeDashboardError("请输入要测试的查询。")
        return knowledge_search_diagnostics(
            query=query,
            top_k=5,
            config_path=self.config_path,
        )

    @staticmethod
    def _positive_integer(value: str, label: str) -> int:
        stripped = value.strip()
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise KnowledgeDashboardError(f"{label}必须是正整数。") from exc
        if not stripped or parsed <= 0 or str(parsed) != stripped:
            raise KnowledgeDashboardError(f"{label}必须是正整数。")
        return parsed

    @classmethod
    def _bounded_integer(
        cls,
        value: str,
        label: str,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        parsed = cls._positive_integer(value, label)
        if parsed < minimum or parsed > maximum:
            raise KnowledgeDashboardError(
                f"{label}必须在 {minimum} 到 {maximum} 之间。"
            )
        return parsed

    @staticmethod
    def _nonnegative_integer(value: str, label: str) -> int:
        stripped = value.strip()
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise KnowledgeDashboardError(
                f"{label}必须是非负整数。"
            ) from exc
        if not stripped or parsed < 0 or str(parsed) != stripped:
            raise KnowledgeDashboardError(f"{label}必须是非负整数。")
        return parsed

    @staticmethod
    def _chunk_mode(value: str) -> str:
        mode = value.strip()
        if mode not in {"structure", "length"}:
            raise KnowledgeDashboardError(
                "切块模式无效，请选择结构优先或字数优先。"
            )
        return mode

    def preview(self, root_value: str) -> dict[str, object]:
        if not root_value.strip():
            raise KnowledgeDashboardError("必须提供知识目录路径。")
        config = load_config(self.config_path)
        try:
            preview: dict[str, object] = dict(
                preview_source(
                    Path(root_value),
                    include_patterns=config.knowledge.include_patterns,
                    exclude_patterns=config.knowledge.exclude_patterns,
                    target_chars=config.knowledge.chunk_target_chars,
                )
            )
        except (OSError, SourceSafetyError, ValueError) as exc:
            raise KnowledgeDashboardError(str(exc)) from exc
        profile = config.embedding_profiles.get(
            config.knowledge.embedding_profile
        )
        tokens = int(str(preview["estimated_tokens"]))
        price = (
            profile.input_cost_per_million_tokens
            if profile is not None
            else None
        )
        preview["estimated_cost_cny"] = (
            f"{tokens / 1_000_000 * price:.4f}" if price is not None else ""
        )
        preview["price_as_of"] = profile.price_as_of if profile is not None else ""
        preview_source_record = SourceRecord(
            id="preview",
            display_name=Path(str(preview["root"])).name,
            absolute_path=str(preview["root"]),
            enabled=True,
            auto_sync=False,
            include_patterns=tuple(config.knowledge.include_patterns),
            exclude_patterns=tuple(config.knowledge.exclude_patterns),
            consent_identity="",
            status="preview",
            last_scan_at="",
            last_success_at="",
            last_error="",
            created_at="",
            updated_at="",
        )
        preview["scope"] = _scope_tree(
            preview_source_record,
            available_patterns=config.knowledge.include_patterns,
            exclude_patterns=config.knowledge.exclude_patterns,
        )
        scope = preview["scope"]
        if isinstance(scope, dict) and not scope.get("available", False):
            raise KnowledgeDashboardError(
                str(scope.get("error") or "数据源根目录当前不可安全读取。")
            )
        return preview

    def add_source(
        self,
        *,
        root_value: str,
        display_name: str,
        consent: bool,
        selected_paths: Sequence[str] | None = None,
        require_credentials: bool = True,
    ) -> str:
        if not consent:
            raise KnowledgeDashboardError(
                "必须确认在线模型的数据传输范围后才能添加数据源。"
            )
        config = load_config(self.config_path)
        if require_credentials:
            _require_provider_credentials(config)
        runtime = KnowledgeRuntime.load(self.config_path)
        try:
            root = validate_source_root(Path(root_value))
        except ValueError as exc:
            raise KnowledgeDashboardError(str(exc)) from exc
        for source in runtime.database.list_sources():
            existing = Path(source.absolute_path)
            if root == existing or root.is_relative_to(existing) or existing.is_relative_to(
                root
            ):
                raise KnowledgeDashboardError(
                    f"数据源不能重叠：{root} 与 {existing}"
                )
        label = display_name.strip() or root.name or str(root)
        preview_source_record = SourceRecord(
            id="new",
            display_name=label,
            absolute_path=str(root),
            enabled=True,
            auto_sync=False,
            include_patterns=tuple(runtime.config.knowledge.include_patterns),
            exclude_patterns=tuple(runtime.config.knowledge.exclude_patterns),
            consent_identity=runtime.consent_identity,
            status="preview",
            last_scan_at="",
            last_success_at="",
            last_error="",
            created_at="",
            updated_at="",
        )
        scope = _scope_tree(
            preview_source_record,
            available_patterns=runtime.config.knowledge.include_patterns,
            exclude_patterns=runtime.config.knowledge.exclude_patterns,
        )
        if not scope.get("available", False):
            raise KnowledgeDashboardError(
                str(scope.get("error") or "数据源根目录当前不可安全读取。")
            )
        if scope.get("truncated") is True and selected_paths is None:
            raise KnowledgeDashboardError(
                str(scope.get("error") or "数据源范围超过当前显示上限，请缩小目录后重试。")
            )
        include_patterns = (
            tuple(runtime.config.knowledge.include_patterns)
            if selected_paths is None
            else _scope_patterns(
                scope,
                selected_paths,
                available_patterns=runtime.config.knowledge.include_patterns,
            )
        )
        source = runtime.database.add_source(
            display_name=label,
            absolute_path=str(root),
            include_patterns=include_patterns,
            exclude_patterns=runtime.config.knowledge.exclude_patterns,
            consent_identity=runtime.consent_identity,
            auto_sync=False,
        )
        return source.id

    def queue_sync(
        self,
        source_id: str,
        *,
        require_credentials: bool = True,
    ) -> int:
        config = load_config(self.config_path)
        if require_credentials:
            _require_provider_credentials(config)
        runtime = KnowledgeRuntime.load(self.config_path)
        source = runtime.database.get_source(source_id)
        if source is None:
            raise KnowledgeDashboardError("数据源不存在。")
        if not source.enabled:
            raise KnowledgeDashboardError("数据源已停用，请先启用后再同步。")
        if source.scope_cleanup_pending:
            raise KnowledgeDashboardError("数据源范围清理尚未完成，暂不能同步。")
        if source.consent_identity != runtime.consent_identity:
            raise KnowledgeDashboardError(
                "Provider 或模型已经变化，请重新确认数据传输授权。"
            )
        try:
            return runtime.database.queue_sync(
                source_id,
                "manual",
                expected_consent_identity=runtime.consent_identity,
            )
        except ValueError as exc:
            raise KnowledgeDashboardError(
                "数据源授权或范围已经变化，请刷新后重试。"
            ) from exc

    def retry_failed_file(self, file_id: int) -> int:
        config = load_config(self.config_path)
        _require_provider_credentials(config)
        runtime = KnowledgeRuntime.load(self.config_path)
        record = runtime.database.get_source_file(file_id)
        if record is None:
            raise KnowledgeDashboardError("失败文件不存在。")
        if record.status != "error":
            raise KnowledgeDashboardError("该文件当前没有可重试的失败状态。")
        source = runtime.database.get_source(record.source_id)
        if source is None:
            raise KnowledgeDashboardError("失败文件所属数据源不存在。")
        if not source.enabled:
            raise KnowledgeDashboardError("数据源已停用，请先启用后再重试。")
        if source.scope_cleanup_pending:
            raise KnowledgeDashboardError(
                "数据源范围清理尚未完成，暂不能重试失败文件。"
            )
        if source.consent_identity != runtime.consent_identity:
            raise KnowledgeDashboardError(
                "Provider 或模型已经变化，请重新确认数据传输授权。"
            )
        try:
            return runtime.database.queue_sync(
                source.id,
                "retry-failed-file",
                expected_consent_identity=runtime.consent_identity,
            )
        except ValueError as exc:
            raise KnowledgeDashboardError(
                "数据源已有等待或运行中的任务，请刷新后重试。"
            ) from exc

    def toggle_source(self, source_id: str, *, enabled: bool) -> None:
        runtime = KnowledgeRuntime.load(self.config_path)
        try:
            with runtime.authorization_guard.acquire_exclusive(timeout=0):
                runtime.database.set_source_enabled(source_id, enabled)
        except Timeout as exc:
            raise KnowledgeDashboardError(
                "索引器正在写入，请稍后修改数据源状态。"
            ) from exc
        except AuthorizationGuardUnavailable as exc:
            raise KnowledgeDashboardError(
                "授权保护不可用，暂不能修改数据源状态。"
            ) from exc

    def toggle_auto_sync(self, source_id: str, *, enabled: bool) -> None:
        runtime = KnowledgeRuntime.load(self.config_path)
        try:
            with runtime.authorization_guard.acquire_exclusive(timeout=0):
                runtime.database.set_source_auto_sync(source_id, enabled)
        except Timeout as exc:
            raise KnowledgeDashboardError(
                "索引器正在写入，请稍后修改自动同步状态。"
            ) from exc

    def renew_consent(
        self,
        source_id: str,
        *,
        consent: bool,
        require_credentials: bool = True,
    ) -> None:
        if not consent:
            raise KnowledgeDashboardError("必须明确确认新的 Provider 与模型范围。")
        config = load_config(self.config_path)
        if require_credentials:
            _require_provider_credentials(config)
        runtime = KnowledgeRuntime.load(self.config_path)
        try:
            with runtime.authorization_guard.acquire_exclusive(timeout=0):
                try:
                    runtime.database.renew_source_consent(
                        source_id,
                        runtime.consent_identity,
                        reason="scope-consent-renewed",
                    )
                except ValueError as exc:
                    raise KnowledgeDashboardError(
                        "数据源范围清理尚未完成，暂不能重新授权。"
                    ) from exc
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise KnowledgeDashboardError(
                "索引器正在写入，请稍后再重新授权。"
            ) from exc

    def update_source_scope(
        self,
        source_id: str,
        selected_paths: Sequence[str],
    ) -> int:
        runtime = KnowledgeRuntime.load(self.config_path)
        try:
            with runtime.authorization_guard.acquire_exclusive(timeout=0):
                source = runtime.database.get_source(source_id)
                if source is None:
                    raise KnowledgeDashboardError("数据源不存在。")
                scope = _scope_tree(
                    source,
                    available_patterns=runtime.config.knowledge.include_patterns,
                    exclude_patterns=runtime.config.knowledge.exclude_patterns,
                )
                if not scope.get("available", False):
                    raise KnowledgeDashboardError(
                        str(scope.get("error") or "数据源根目录当前不可安全读取。")
                    )
                include_patterns = _scope_patterns(
                    scope,
                    selected_paths,
                    available_patterns=runtime.config.knowledge.include_patterns,
                )
                consent_identity = (
                    source.consent_identity
                    if include_patterns == source.include_patterns
                    else ""
                )
                try:
                    runtime.database.set_source_scope(
                        source_id,
                        include_patterns=include_patterns,
                        consent_identity=consent_identity,
                    )
                except ValueError as exc:
                    raise KnowledgeDashboardError(
                        "数据源仍有等待或运行中的同步任务，请等待任务结束后再修改范围。"
                    ) from exc
                updated = runtime.database.get_source(source_id)
                if updated is None:  # pragma: no cover
                    raise KnowledgeDashboardError("数据源不存在。")
                removed = 0
                for record in runtime.database.list_source_files(source_id):
                    if source_path_is_selected(record.relative_path, updated):
                        continue
                    chunk_ids = runtime.database.delete_source_file(
                        record.id,
                        scope_cleanup=True,
                    )
                    runtime.vectors.delete_across_history(
                        chunk_ids,
                        table_names=runtime.database.list_vector_tables(
                            runtime.table_id
                        ),
                    )
                    runtime.database.complete_vector_deletions(chunk_ids)
                    removed += 1
                runtime.database.complete_source_scope_cleanup(source_id)
                return removed
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise KnowledgeDashboardError(
                "索引器正在写入，请稍后再修改数据源范围。"
            ) from exc

    def remove_source(self, source_id: str) -> None:
        runtime = KnowledgeRuntime.load(self.config_path)
        try:
            with runtime.authorization_guard.acquire_exclusive(timeout=0):
                try:
                    chunk_ids = runtime.database.remove_source(source_id)
                except ValueError as exc:
                    raise KnowledgeDashboardError(
                        "数据源仍有等待或运行中的同步任务，请等待任务结束后再移除。"
                    ) from exc
                runtime.vectors.delete_across_history(
                    chunk_ids,
                    table_names=runtime.database.list_vector_tables(
                        runtime.table_id
                    ),
                )
                runtime.database.complete_vector_deletions(chunk_ids)
        except (Timeout, AuthorizationGuardUnavailable) as exc:
            raise KnowledgeDashboardError(
                "索引器正在写入，请稍后再移除数据源。"
            ) from exc
