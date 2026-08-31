from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from keepygaga_rag.config import KnowledgeAppConfig, load_config
from keepygaga_rag.knowledge.db import SCHEMA_VERSION, KnowledgeDB
from keepygaga_rag.knowledge.indexer import (
    source_path_is_selected,
    source_root_is_safe,
)
from keepygaga_rag.knowledge.runtime import (
    KnowledgeRuntime,
    resolve_knowledge_store,
)

PUBLIC_MCP_TOOLS = ("search",)
DOCTOR_SCHEMA = "keepygaga-rag-doctor-v1"
DOCTOR_STUCK_TASK_SECONDS = 15 * 60


def _check(
    checks: list[dict[str, object]],
    *,
    check_id: str,
    status: str,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": status,
            "message": message,
            "details": details or {},
        }
    )


def run_doctor(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, object]:
    del project_root
    checks: list[dict[str, object]] = []
    try:
        config = load_config(config_path)
    except Exception as exc:
        return {
            "schema": DOCTOR_SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "error",
            "tools": list(PUBLIC_MCP_TOOLS),
            "checks": [
                {
                    "id": "config",
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {},
                }
            ],
        }
    _check(
        checks,
        check_id="config",
        status="ok",
        message="configuration loaded",
        details={"path": str(config_path)},
    )
    _knowledge_checks(config, config_path, checks)

    statuses = {str(item["status"]) for item in checks}
    overall = (
        "error" if "error" in statuses else "warning" if "warning" in statuses else "ok"
    )
    return {
        "schema": DOCTOR_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": overall,
        "tools": list(PUBLIC_MCP_TOOLS),
        "checks": checks,
    }


def _probe_coordinator_lock(store: Path) -> dict[str, object]:
    lock_path = store / "coordinator.lock"
    result: dict[str, object] = {
        "path": str(lock_path),
        "status": "stopped",
        "lock_held": False,
        "running": False,
    }
    if not lock_path.is_file():
        result["message"] = "coordinator.lock does not exist"
        return result
    lock = FileLock(lock_path)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        result.update(
            status="running",
            lock_held=True,
            running=True,
            message="coordinator.lock is held by the indexer",
        )
    except OSError as exc:
        result.update(
            status="unknown",
            message=f"unable to probe coordinator.lock: {type(exc).__name__}: {exc}",
        )
    else:
        lock.release()
        result["message"] = "coordinator.lock is not held"
    return result


def _timestamp_age_seconds(value: object) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())


def _task_details(
    jobs: list[dict[str, object]],
    runs: list[dict[str, object]],
    coordinator: dict[str, object],
) -> dict[str, object]:
    open_jobs = [
        job for job in jobs if str(job.get("status")) in {"queued", "running"}
    ]
    open_runs = [run for run in runs if str(run.get("status")) == "running"]
    stuck_jobs = [
        {
            **job,
            "age_seconds": age,
        }
        for job in open_jobs
        if str(job.get("status")) == "running"
        and (age := _timestamp_age_seconds(job.get("started_at"))) is not None
        and age > DOCTOR_STUCK_TASK_SECONDS
    ]
    stuck_runs = [
        {
            **run,
            "age_seconds": age,
        }
        for run in open_runs
        if (age := _timestamp_age_seconds(run.get("started_at"))) is not None
        and age > DOCTOR_STUCK_TASK_SECONDS
    ]
    coordinator_status = str(coordinator.get("status", "unknown"))
    coordinator_running = coordinator_status == "running"
    return {
        "coordinator_running": coordinator_running,
        "open_jobs": open_jobs,
        "open_runs": open_runs,
        "stuck_jobs": stuck_jobs,
        "stuck_runs": stuck_runs,
        "open_job_count": len(open_jobs),
        "open_run_count": len(open_runs),
        "stuck_job_count": len(stuck_jobs),
        "stuck_run_count": len(stuck_runs),
        "coordinator_status": coordinator_status,
    }


def _active_vector_integrity(
    config_path: Path,
    *,
    schema_version: int | None,
) -> dict[str, object]:
    if schema_version != SCHEMA_VERSION:
        return {
            "status": "not_checked",
            "reason": "active vector check requires the current SQLite schema",
            "expected_active_vectors": 0,
            "found_active_vectors": 0,
            "missing_vector_count": 0,
            "invalid_vector_count": 0,
            "missing_chunk_ids": [],
            "invalid_chunk_ids": [],
        }
    try:
        runtime = KnowledgeRuntime.load(config_path, readonly=True)
        sources = {
            source.id: source
            for source in runtime.database.list_sources()
            if source.enabled
            and not source.scope_cleanup_pending
            and source.consent_identity == runtime.consent_identity
            and source_root_is_safe(source.absolute_path)
        }
        expected = 0
        found = 0
        missing_count = 0
        invalid_count = 0
        missing_sample: list[str] = []
        invalid_sample: list[str] = []
        expected_space_id = hashlib.sha256(
            runtime.embedding.identity.encode("utf-8")
        ).hexdigest()
        metadata_reader = getattr(runtime.vectors, "get_vector_metadata", None)
        vector_backend_checked = False
        for page in runtime.database.iter_active_chunks_for_vector_rebuild():
            eligible = [
                row
                for row in page
                if str(row["source_id"]) in sources
                and source_path_is_selected(
                    str(row["relative_path"]), sources[str(row["source_id"])]
                )
            ]
            if not eligible:
                continue
            if not vector_backend_checked:
                runtime.vectors.ensure_table()
                vector_backend_checked = True
            chunk_ids = [str(row["chunk_id"]) for row in eligible]
            expected += len(chunk_ids)
            if callable(metadata_reader):
                metadata = cast(
                    dict[str, dict[str, str]], metadata_reader(chunk_ids)
                )
            else:
                metadata = {
                    chunk_id: {}
                    for chunk_id in runtime.vectors.get_vectors(chunk_ids)
                }
            for row in eligible:
                chunk_id = str(row["chunk_id"])
                actual = metadata.get(chunk_id)
                if actual is None:
                    missing_count += 1
                    if len(missing_sample) < 50:
                        missing_sample.append(chunk_id)
                    continue
                if (
                    actual.get("embedding_space_id") != expected_space_id
                    or actual.get("embedding_input_hash")
                    != str(row["embedding_input_hash"])
                ):
                    invalid_count += 1
                    if len(invalid_sample) < 50:
                        invalid_sample.append(chunk_id)
                    continue
                found += 1
        if not expected:
            runtime.vectors.ensure_table()
    except FileNotFoundError:
        return {
            "status": "not_checked",
            "reason": "LanceDB store or SQLite database is not initialized",
            "expected_active_vectors": 0,
            "found_active_vectors": 0,
            "missing_vector_count": 0,
            "invalid_vector_count": 0,
            "missing_chunk_ids": [],
            "invalid_chunk_ids": [],
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "expected_active_vectors": 0,
            "found_active_vectors": 0,
            "missing_vector_count": 0,
            "invalid_vector_count": 0,
            "missing_chunk_ids": [],
            "invalid_chunk_ids": [],
        }
    return {
        "status": "error" if missing_count or invalid_count else "ok",
        "reason": (
            "active chunks have missing or invalid Lance vectors"
            if missing_count or invalid_count
            else "no current authorized active chunks to check"
            if not expected
            else "active vectors are complete"
        ),
        "expected_active_vectors": expected,
        "found_active_vectors": found,
        "missing_vector_count": missing_count,
        "invalid_vector_count": invalid_count,
        "missing_chunk_ids": missing_sample,
        "invalid_chunk_ids": invalid_sample,
    }


def _knowledge_checks(
    config: KnowledgeAppConfig,
    config_path: Path,
    checks: list[dict[str, object]],
) -> None:
    if not config.knowledge.enabled:
        _check(
            checks,
            check_id="knowledge",
            status="error",
            message="knowledge backend is disabled",
            details={},
        )
        return
    embedding = config.embedding_profiles[config.knowledge.embedding_profile]
    rerank = config.rerank_profiles[config.knowledge.rerank_profile]
    store = resolve_knowledge_store(config, config_path)
    missing_keys = [
        name
        for name in (embedding.api_key_env, rerank.api_key_env)
        if not os.environ.get(name, "").strip()
    ]
    database_path = store / "knowledge.sqlite3"
    indexed: dict[str, int] = {}
    database_error = ""
    database_corrupt = False
    schema_unavailable = False
    schema_version: int | None = None
    jobs: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    if database_path.is_file():
        try:
            with sqlite3.connect(
                f"{database_path.as_uri()}?mode=ro", uri=True
            ) as connection:
                quick_check = connection.execute("PRAGMA quick_check").fetchone()
                if quick_check is None or str(quick_check[0]).casefold() != "ok":
                    database_error = (
                        "knowledge database integrity check failed: "
                        f"{quick_check[0] if quick_check else 'no result'}"
                    )
                    database_corrupt = True
                schema_row = connection.execute(
                    "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()
                schema_version = (
                    int(schema_row[0])
                    if schema_row is not None and schema_row[0] is not None
                    else None
                )
                indexed = {
                    "sources": int(
                        connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                    ),
                    "enabled_sources": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM sources WHERE enabled = 1"
                        ).fetchone()[0]
                    ),
                    "auto_sync_sources": int(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM sources
                            WHERE enabled = 1 AND auto_sync = 1
                            """
                        ).fetchone()[0]
                    ),
                    "files": int(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM source_files AS sf
                            JOIN sources AS s ON s.id = sf.source_id
                            WHERE s.enabled = 1 AND sf.active_generation > 0
                            """
                        ).fetchone()[0]
                    ),
                    "active_chunks": int(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM chunks AS c
                            JOIN source_files AS sf ON sf.id = c.source_file_id
                            JOIN sources AS s ON s.id = sf.source_id
                            WHERE s.enabled = 1
                              AND c.generation = sf.active_generation
                            """
                        ).fetchone()[0]
                    ),
                    "file_errors": int(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM source_files AS sf
                            JOIN sources AS s ON s.id = sf.source_id
                            WHERE s.enabled = 1 AND sf.status = 'error'
                            """
                        ).fetchone()[0]
                    ),
                    "provider_action_required_file_errors": int(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM source_files AS sf
                            JOIN sources AS s ON s.id = sf.source_id
                            WHERE s.enabled = 1 AND sf.status = 'error'
                              AND (
                                sf.last_error LIKE '%ProviderActionRequiredError:%'
                                OR sf.last_error LIKE '%401 Unauthorized%'
                                OR sf.last_error LIKE '%402 Payment Required%'
                                OR sf.last_error LIKE '%403 Forbidden%'
                              )
                            """
                        ).fetchone()[0]
                    ),
                }
                consent_identity = hashlib.sha256(
                    f"{embedding.identity}\0{rerank.identity}".encode()
                ).hexdigest()
                indexed["consent_current_sources"] = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM sources
                        WHERE enabled = 1 AND consent_identity = ?
                        """,
                        (consent_identity,),
                    ).fetchone()[0]
                )
            if schema_version is not None and schema_version <= SCHEMA_VERSION:
                readonly_database = KnowledgeDB(database_path, readonly=True)
                jobs = readonly_database.list_sync_jobs(
                    limit=100,
                    statuses=("queued", "running"),
                )
                runs = readonly_database.list_sync_runs(limit=100)
        except RuntimeError as exc:
            if schema_version is None or schema_version <= SCHEMA_VERSION:
                database_error = f"{type(exc).__name__}: {exc}"
                database_corrupt = True
        except (OSError, sqlite3.Error) as exc:
            database_error = f"{type(exc).__name__}: {exc}"
            lowered = str(exc).casefold()
            schema_unavailable = (
                "no such table" in lowered and schema_version is None
            )
            database_corrupt = not schema_unavailable
    else:
        database_error = "knowledge database does not exist"

    warnings: list[str] = []
    errors: list[str] = []
    schema_is_newer = (
        schema_version is not None and schema_version > SCHEMA_VERSION
    )
    if missing_keys:
        warnings.append("provider credentials are missing")
    if database_error:
        if database_corrupt:
            errors.append(database_error)
        else:
            warnings.append(database_error)
    elif schema_version is None:
        warnings.append("knowledge database schema version is unavailable")
    elif schema_version < SCHEMA_VERSION:
        warnings.append(
            "knowledge database upgrade required: "
            f"schema {schema_version} -> {SCHEMA_VERSION}"
        )
    elif schema_is_newer:
        warnings.append(
            "knowledge database schema is newer than this code: "
            f"schema {schema_version} > {SCHEMA_VERSION}"
        )
    elif indexed.get("enabled_sources", 0) == 0:
        warnings.append("no enabled knowledge source")
    elif indexed.get("active_chunks", 0) == 0:
        warnings.append("no active knowledge chunks")
    elif indexed.get("consent_current_sources") != indexed.get("enabled_sources"):
        warnings.append("one or more source consents are stale")
    elif indexed.get("auto_sync_sources") != indexed.get("enabled_sources"):
        warnings.append("one or more enabled sources are not set to auto-sync")
    if indexed.get("provider_action_required_file_errors", 0):
        warnings.append(
            "provider credentials, account balance, quota, or model access "
            "requires attention before failed files can be retried"
        )
    elif indexed.get("file_errors", 0):
        warnings.append("one or more source files failed to index")

    coordinator = _probe_coordinator_lock(store)
    task_state = _task_details(jobs, runs, coordinator)
    if str(coordinator.get("status")) == "unknown":
        warnings.append("coordinator status could not be determined")
    elif str(coordinator.get("status")) != "running":
        warnings.append("knowledge coordinator is not running")
    if task_state["stuck_job_count"] or task_state["stuck_run_count"]:
        errors.append("one or more knowledge tasks appear stuck")
    elif task_state["open_job_count"] or task_state["open_run_count"]:
        if not task_state["coordinator_running"]:
            errors.append(
                "knowledge tasks remain open while the coordinator is not running"
            )
        else:
            warnings.append("knowledge tasks are still open")

    vector_integrity = _active_vector_integrity(
        config_path,
        schema_version=schema_version,
    )
    if vector_integrity["status"] == "error":
        errors.append(str(vector_integrity["reason"]))

    _check(
        checks,
        check_id="knowledge",
        status=(
            "error"
            if schema_is_newer or errors
            else "warning"
            if warnings
            else "ok"
        ),
        message=(
            "knowledge backend has errors: "
            + "; ".join(errors + warnings)
            if errors or warnings
            else "knowledge backend is ready"
        ),
        details={
            "store": str(store),
            "database": str(database_path),
            "schema_version": schema_version,
            "expected_schema_version": SCHEMA_VERSION,
            "migration_required": (
                schema_version is not None and schema_version < SCHEMA_VERSION
            ),
            "table_id": config.knowledge.table_id,
            "scan_interval_seconds": config.knowledge.scan_interval_seconds,
            "embedding_model": embedding.model,
            "embedding_dimensions": embedding.dimensions,
            "rerank_model": rerank.model,
            "missing_key_env": missing_keys,
            "backend": "sqlite_fts5_lancedb",
            "agents_memory_excluded": True,
            "context_backups_excluded": True,
            "database_error": database_error,
            "database_corrupt": database_corrupt,
            "schema_unavailable": schema_unavailable,
            "coordinator": coordinator,
            "indexer_status": coordinator["status"],
            "indexer_running": coordinator["running"],
            "coordinator_running": task_state["coordinator_running"],
            "tasks": task_state,
            "open_jobs": task_state["open_jobs"],
            "open_runs": task_state["open_runs"],
            "stuck_jobs": task_state["stuck_jobs"],
            "stuck_runs": task_state["stuck_runs"],
            "open_task_count": (
                cast(int, task_state["open_job_count"])
                + cast(int, task_state["open_run_count"])
            ),
            "stuck_task_count": (
                cast(int, task_state["stuck_job_count"])
                + cast(int, task_state["stuck_run_count"])
            ),
            "vector_integrity": vector_integrity,
            "active_vector_integrity": vector_integrity,
            "mcp_process": {
                "status": "not_checked",
                "reason": (
                    "MCP process liveness is not reliable from an independent doctor"
                ),
            },
            **indexed,
        },
    )
