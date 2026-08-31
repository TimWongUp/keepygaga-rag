from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, tzinfo
from pathlib import Path
from urllib.parse import urlencode, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from keepygaga_rag.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from keepygaga_rag.dashboard.knowledge_service import (
    KnowledgeDashboardError,
    KnowledgeDashboardService,
)
from keepygaga_rag.dashboard.presence import DashboardPresence
from keepygaga_rag.diagnostics import run_doctor

PACKAGE_ROOT = Path(__file__).resolve().parent
LAUNCHCTL_PATH = Path("/bin/launchctl")
OSASCRIPT_PATH = Path("/usr/bin/osascript")
INDEXER_LAUNCHD_LABEL = "ai.keepygaga.knowledge.indexer"
INDEXER_RESTART_COOLDOWN_SECONDS = 60.0


def _language(request: Request) -> str:
    requested = request.query_params.get("lang")
    if requested in {"zh", "en"}:
        return requested
    stored = request.cookies.get("keepygaga_rag_lang")
    return stored if stored in {"zh", "en"} else "zh"


def _status_label(status: object, lang: str = "zh") -> str:
    labels = {
        "zh": {"ok": "正常", "warning": "需留意", "error": "异常"},
        "en": {"ok": "Healthy", "warning": "Attention", "error": "Error"},
    }
    return labels.get(lang, labels["zh"]).get(str(status), str(status))


def _source_status_label(status: object, lang: str = "zh") -> str:
    labels = {
        "zh": {
            "idle": "待命",
            "queued": "排队中",
            "syncing": "同步中",
            "paused": "已暂停",
            "error": "异常",
            "preview": "预检中",
        },
        "en": {
            "idle": "Idle",
            "queued": "Queued",
            "syncing": "Syncing",
            "paused": "Paused",
            "error": "Error",
            "preview": "Preview",
        },
    }
    return labels.get(lang, labels["zh"]).get(str(status), str(status))


def _indexer_status_label(status: object, lang: str = "zh") -> str:
    labels = {
        "zh": {
            "running": "运行中",
            "stopped": "未运行",
            "unknown": "无法确认",
            "disabled": "已停用",
        },
        "en": {
            "running": "Running",
            "stopped": "Stopped",
            "unknown": "Unknown",
            "disabled": "Disabled",
        },
    }
    return labels.get(lang, labels["zh"]).get(str(status), str(status))


def _job_status_label(status: object, lang: str = "zh") -> str:
    labels = {
        "zh": {
            "queued": "排队中",
            "running": "运行中",
            "succeeded": "已完成",
            "failed": "失败",
        },
        "en": {
            "queued": "Queued",
            "running": "Running",
            "succeeded": "Succeeded",
            "failed": "Failed",
        },
    }
    return labels.get(lang, labels["zh"]).get(str(status), str(status))


def _run_status_label(status: object, lang: str = "zh") -> str:
    labels = {
        "zh": {
            "running": "运行中",
            "succeeded": "已完成",
            "failed": "失败",
        },
        "en": {
            "running": "Running",
            "succeeded": "Succeeded",
            "failed": "Failed",
        },
    }
    return labels.get(lang, labels["zh"]).get(str(status), str(status))


def _scope_selection(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise KnowledgeDashboardError("数据源范围格式无效。") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise KnowledgeDashboardError("数据源范围格式无效。")
    if len(parsed) > 2000:
        raise KnowledgeDashboardError("数据源范围项目过多。")
    return parsed


def _retain_scope_selection(
    scope: dict[str, object], selected_paths: Sequence[str]
) -> None:
    items_value = scope.get("items")
    if not isinstance(items_value, list):
        return
    items = [item for item in items_value if isinstance(item, dict)]
    selection = tuple(dict.fromkeys(str(path) for path in selected_paths))

    def selected(path: str) -> bool:
        return any(
            parent == "."
            or path == parent
            or path.startswith(f"{parent}/")
            for parent in selection
        )

    files = [item for item in items if item.get("kind") == "file"]
    for item in items:
        path = str(item.get("path", ""))
        if item.get("kind") == "file":
            item["state"] = "full" if selected(path) else "off"
            continue
        descendants = [
            file_item
            for file_item in files
            if selected(str(file_item.get("path", "")))
            and (
                path == "."
                or str(file_item.get("path", "")) == path
                or str(file_item.get("path", "")).startswith(f"{path}/")
            )
        ]
        total = sum(
            1
            for file_item in files
            if path == "."
            or str(file_item.get("path", "")) == path
            or str(file_item.get("path", "")).startswith(f"{path}/")
        )
        item["state"] = (
            "full"
            if total and len(descendants) == total
            else "partial"
            if descendants
            else "off"
        )
    scope["selection"] = list(selection)
    scope["selected_files"] = sum(
        1 for item in files if selected(str(item.get("path", "")))
    )


def _format_local_datetime(value: object, timezone: tzinfo | None = None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return parsed.astimezone(timezone).strftime("%Y-%m-%d %H:%M:%S")


def _choose_directory() -> str:
    if sys.platform == "darwin":
        command = [
            "osascript",
            "-e",
            (
                'POSIX path of (choose folder with prompt '
                '"选择要加入 Keepygaga RAG 的知识目录")'
            ),
        ]
    elif sys.platform == "win32":
        command = [
            "powershell",
            "-NoProfile",
            "-STA",
            "-Command",
            (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$dialog.Description = '选择要加入 Keepygaga RAG 的知识目录'; "
                "if ($dialog.ShowDialog() -eq 'OK') "
                "{ [Console]::Out.Write($dialog.SelectedPath) }"
            ),
        ]
    elif executable := shutil.which("zenity"):
        command = [
            executable,
            "--file-selection",
            "--directory",
            "--title=选择要加入 Keepygaga RAG 的知识目录",
        ]
    else:
        return ""
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip().rstrip("/")


def _indexer_restart_supported() -> bool:
    return (
        sys.platform == "darwin"
        and LAUNCHCTL_PATH.is_file()
        and OSASCRIPT_PATH.is_file()
    )


def _indexer_launchd_job_registered() -> bool:
    if not _indexer_restart_supported():
        return False
    try:
        completed = subprocess.run(
            [
                str(LAUNCHCTL_PATH),
                "print",
                f"gui/{os.getuid()}/{INDEXER_LAUNCHD_LABEL}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _restart_indexer() -> None:
    if not _indexer_launchd_job_registered():
        raise RuntimeError("当前没有可管理的 launchd indexer 服务。")
    confirmed = subprocess.run(
        [
            str(OSASCRIPT_PATH),
            "-e",
            (
                'display dialog "Keepygaga RAG Dashboard 请求重启索引协调器。" '
                'with title "重启 Keepygaga RAG Indexer" '
                'buttons {"取消", "重启"} default button "重启" '
                'cancel button "取消" with icon caution'
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if confirmed.returncode != 0:
        raise RuntimeError("已取消重启 indexer。")
    completed = subprocess.run(
        [
            str(LAUNCHCTL_PATH),
            "kickstart",
            "-k",
            f"gui/{os.getuid()}/{INDEXER_LAUNCHD_LABEL}",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "无法通过 launchd 重启 indexer"
            f"（退出码 {completed.returncode}）。"
        )


class IndexerRestartCooldown(RuntimeError):
    pass


class IndexerRestartController:
    def __init__(
        self,
        restart: Callable[[], None],
        *,
        cooldown_seconds: float = INDEXER_RESTART_COOLDOWN_SECONDS,
    ):
        self._restart = restart
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._last_attempt_at: float | None = None

    def restart(self) -> None:
        with self._lock:
            now = time.monotonic()
            if (
                self._last_attempt_at is not None
                and now - self._last_attempt_at < self._cooldown_seconds
            ):
                raise IndexerRestartCooldown(
                    "Indexer 刚刚已请求重启，请稍后再试。"
                )
            self._last_attempt_at = now
            self._restart()
            self._last_attempt_at = time.monotonic()


def _check_label(check_id: object, lang: str = "zh") -> str:
    labels = {
        "zh": {
            "config": "基础配置",
            "knowledge": "本地知识库",
        },
        "en": {
            "config": "Configuration",
            "knowledge": "Local knowledge",
        },
    }
    return labels.get(lang, labels["zh"]).get(str(check_id), str(check_id))


def _check_description(check: dict[str, object], lang: str = "zh") -> str:
    check_id = str(check.get("id", ""))
    status = str(check.get("status", ""))
    if check_id == "knowledge" and status != "ok":
        return (
            "Check the knowledge source, index state, consent, and provider credentials."
            if lang == "en"
            else "请检查知识数据源、索引状态、在线模型授权与密钥配置。"
        )
    return str(check.get("message", ""))


def _verify_local_request(
    request: Request,
    submitted_token: str,
    expected_token: str,
    *,
    require_origin: bool = False,
) -> None:
    if not secrets.compare_digest(submitted_token, expected_token):
        raise HTTPException(status_code=403, detail="CSRF token 无效。")
    origin = request.headers.get("origin")
    if not origin:
        if require_origin:
            raise HTTPException(status_code=403, detail="缺少同源请求信息。")
        return
    parsed = urlparse(origin)
    host = request.headers.get("host", "")
    if (
        parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.netloc != host
    ):
        raise HTTPException(status_code=403, detail="拒绝跨源配置写入。")


def create_app(
    *,
    config_path: Path | None = None,
    project_root: Path | None = None,
    auto_close: bool = False,
    shutdown_callback: Callable[[], None] | None = None,
    auto_close_grace_seconds: float = 10.0,
) -> FastAPI:
    if auto_close and shutdown_callback is None:
        raise ValueError("auto-close requires a shutdown callback")
    resolved_root = (project_root or PROJECT_ROOT).expanduser().resolve()
    resolved_config = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    default_config = DEFAULT_CONFIG_PATH.expanduser().resolve()
    indexer_restart_supported = (
        resolved_config == default_config and _indexer_restart_supported()
    )
    directory_picker_supported = (
        sys.platform in {"darwin", "win32"} or shutil.which("zenity") is not None
    )
    knowledge_service = KnowledgeDashboardService(resolved_config)
    templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
    presence = (
        DashboardPresence(
            shutdown=shutdown_callback,
            grace_seconds=auto_close_grace_seconds,
        )
        if shutdown_callback is not None and auto_close
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if presence is not None:
            await presence.close()

    app = FastAPI(
        title="Keepygaga RAG Dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_ROOT / "static")),
        name="static",
    )
    app.state.csrf_token = secrets.token_urlsafe(32)
    app.state.indexer_restart_controller = IndexerRestartController(
        _restart_indexer
    )
    templates.env.globals.update(
        dashboard_auto_close=auto_close,
        status_label=_status_label,
        source_status_label=_source_status_label,
        status_class=lambda status: {
            "ok": "good",
            "warning": "warn",
            "error": "bad",
        }.get(str(status), "muted"),
        check_label=_check_label,
        check_description=_check_description,
        format_local_datetime=_format_local_datetime,
        indexer_status_label=_indexer_status_label,
        job_status_label=_job_status_label,
        run_status_label=_run_status_label,
    )

    if presence is not None:

        @app.get("/_dashboard/presence")
        async def dashboard_presence(request: Request) -> StreamingResponse:
            client_id = id(request)

            async def events():
                presence.connected(client_id)
                try:
                    while not await request.is_disconnected():
                        yield ": keepalive\n\n"
                        await asyncio.sleep(5)
                finally:
                    presence.disconnected(client_id)

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    async def render_knowledge(
        request: Request,
        *,
        preview: Mapping[str, object] | None = None,
        root_value: str = "",
        error: str = "",
        message: str = "",
        chunk_form: Mapping[str, object] | None = None,
        retrieval_form: Mapping[str, object] | None = None,
        enrollment_form: Mapping[str, object] | None = None,
        rebuild_result: Mapping[str, object] | None = None,
        diagnostic_query: str = "",
        diagnostic_result: Mapping[str, object] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        lang = _language(request)
        try:
            snapshot = await asyncio.to_thread(knowledge_service.snapshot)
            context = {
                "page": "knowledge",
                "snapshot": snapshot,
                "preview": preview,
                "root_value": root_value,
                "error": error,
                "message": message,
                "chunk_form": chunk_form,
                "retrieval_form": retrieval_form,
                "enrollment_form": enrollment_form,
                "rebuild_result": rebuild_result,
                "diagnostic_query": diagnostic_query,
                "diagnostic_result": diagnostic_result,
                "indexer_restart_supported": indexer_restart_supported,
                "directory_picker_supported": directory_picker_supported,
                "csrf_token": app.state.csrf_token,
                "lang": lang,
            }
        except Exception as exc:
            context = {
                "page": "knowledge",
                "fatal_error": f"无法加载知识库：{exc}",
                "preview": preview,
                "root_value": root_value,
                "error": error,
                "message": message,
                "chunk_form": chunk_form,
                "retrieval_form": retrieval_form,
                "enrollment_form": enrollment_form,
                "rebuild_result": rebuild_result,
                "diagnostic_query": diagnostic_query,
                "diagnostic_result": diagnostic_result,
                "indexer_restart_supported": indexer_restart_supported,
                "directory_picker_supported": directory_picker_supported,
                "csrf_token": app.state.csrf_token,
                "lang": lang,
            }
            status_code = 500
        return templates.TemplateResponse(
            request=request,
            name="knowledge.html",
            context=context,
            status_code=status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        health = await asyncio.to_thread(
            run_doctor,
            resolved_config,
            project_root=resolved_root,
        )
        return templates.TemplateResponse(
            request=request,
            name="overview.html",
            context={
                "page": "overview",
                "health": health,
                "config_path": str(resolved_config),
                "lang": _language(request),
            },
        )

    @app.get("/language/{lang}")
    async def set_language(lang: str, next: str = "/") -> RedirectResponse:
        if lang not in {"zh", "en"}:
            raise HTTPException(status_code=404, detail="Unsupported language.")
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        response = RedirectResponse(url=target, status_code=303)
        response.set_cookie(
            "keepygaga_rag_lang",
            lang,
            max_age=31_536_000,
            httponly=True,
            samesite="lax",
        )
        return response

    @app.get("/context")
    async def legacy_context_page() -> RedirectResponse:
        return RedirectResponse(url="/#guide", status_code=303)

    @app.get("/knowledge", response_class=HTMLResponse)
    async def knowledge(
        request: Request,
        message: str = "",
        changed: int = 0,
        marked_files: int = 0,
        queued_sources: int = 0,
        deferred_sources: int = 0,
    ) -> HTMLResponse:
        rebuild_result = (
            {
                "changed": changed,
                "marked_files": marked_files,
                "queued_sources": queued_sources,
                "deferred_sources": deferred_sources,
            }
            if message == "chunk-settings-updated"
            else None
        )
        return await render_knowledge(
            request,
            message=message,
            rebuild_result=rebuild_result,
        )

    @app.post("/knowledge/chunk-settings", response_class=HTMLResponse)
    async def update_knowledge_chunk_settings(request: Request) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        raw_chunk_mode = form.get("chunk_mode")
        raw_chunk_overlap = form.get("chunk_overlap_chars")
        chunk_form = {
            "chunk_target_chars": str(form.get("chunk_target_chars", "")),
            "chunk_max_chars": str(form.get("chunk_max_chars", "")),
            "chunk_mode": str(raw_chunk_mode) if raw_chunk_mode is not None else "",
            "chunk_overlap_chars": (
                str(raw_chunk_overlap)
                if raw_chunk_overlap is not None
                else ""
            ),
        }
        try:
            result = await asyncio.to_thread(
                knowledge_service.update_chunk_settings,
                target_value=str(chunk_form["chunk_target_chars"]),
                max_value=str(chunk_form["chunk_max_chars"]),
                mode_value=(
                    str(raw_chunk_mode) if raw_chunk_mode is not None else None
                ),
                overlap_value=(
                    str(raw_chunk_overlap)
                    if raw_chunk_overlap is not None
                    else None
                ),
            )
        except (KnowledgeDashboardError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(
                request,
                error=str(exc),
                chunk_form=chunk_form,
                status_code=422,
            )
        query = urlencode(
            {
                "message": "chunk-settings-updated",
                "changed": result["changed"],
                "marked_files": result["marked_files"],
                "queued_sources": result["queued_sources"],
                "deferred_sources": result["deferred_sources"],
            }
        )
        return RedirectResponse(url=f"/knowledge?{query}", status_code=303)

    @app.post("/knowledge/indexer/restart", response_class=HTMLResponse)
    async def restart_knowledge_indexer(request: Request) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
            require_origin=True,
        )
        if not indexer_restart_supported:
            return await render_knowledge(
                request,
                error="当前 Dashboard 配置没有可管理的 indexer 服务。",
                status_code=422,
            )
        try:
            await asyncio.to_thread(
                app.state.indexer_restart_controller.restart
            )
        except IndexerRestartCooldown as exc:
            return await render_knowledge(request, error=str(exc), status_code=429)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(
            url="/knowledge?message=indexer-restarted",
            status_code=303,
        )

    @app.post("/knowledge/retrieval-settings", response_class=HTMLResponse)
    async def update_knowledge_retrieval_settings(request: Request) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        retrieval_form = {
            "vector_recall_limit": str(form.get("vector_recall_limit", "")),
            "keyword_recall_limit": str(form.get("keyword_recall_limit", "")),
            "rerank_candidate_limit": str(
                form.get("rerank_candidate_limit", "")
            ),
            "max_chunks_per_source_file": str(
                form.get("max_chunks_per_source_file", "")
            ),
        }
        try:
            await asyncio.to_thread(
                knowledge_service.update_retrieval_settings,
                vector_value=str(retrieval_form["vector_recall_limit"]),
                keyword_value=str(retrieval_form["keyword_recall_limit"]),
                rerank_value=str(retrieval_form["rerank_candidate_limit"]),
                max_chunks_value=str(
                    retrieval_form["max_chunks_per_source_file"]
                ),
            )
        except (KnowledgeDashboardError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(
                request,
                error=str(exc),
                retrieval_form=retrieval_form,
                status_code=422,
            )
        return RedirectResponse(
            url="/knowledge?message=retrieval-settings-updated",
            status_code=303,
        )

    @app.post("/knowledge/diagnostics", response_class=HTMLResponse)
    async def diagnose_knowledge_search(request: Request) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        diagnostic_query = str(form.get("query", ""))
        try:
            result = await asyncio.to_thread(
                knowledge_service.diagnose_search,
                diagnostic_query,
            )
        except (KnowledgeDashboardError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(
                request,
                diagnostic_query=diagnostic_query,
                error=str(exc),
                status_code=422,
            )
        return await render_knowledge(
            request,
            diagnostic_query=diagnostic_query,
            diagnostic_result=result,
        )

    @app.post("/knowledge/choose-directory")
    async def choose_knowledge_directory(request: Request) -> JSONResponse:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            selected = await asyncio.to_thread(_choose_directory)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return JSONResponse(
                {"status": "error", "message": str(exc)},
                status_code=422,
            )
        return JSONResponse(
            {
                "status": "ok" if selected else "cancelled",
                "path": selected,
            }
        )

    @app.post("/knowledge/preview", response_class=HTMLResponse)
    async def preview_knowledge_source(request: Request) -> HTMLResponse:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            preview = await asyncio.to_thread(
                knowledge_service.preview,
                str(form.get("root", "")),
            )
            return await render_knowledge(request, preview=preview)
        except (KnowledgeDashboardError, OSError, ValueError) as exc:
            return await render_knowledge(
                request,
                root_value=str(form.get("root", "")),
                error=str(exc),
                status_code=422,
            )

    @app.post("/knowledge/sources", response_class=HTMLResponse)
    async def add_knowledge_source(request: Request) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        root_value = str(form.get("root", ""))
        display_name = str(form.get("display_name", ""))
        consent = str(form.get("consent", "")) == "on"
        enrollment_form = {
            "display_name": display_name,
            "consent": consent,
        }
        selected_paths: list[str] | None = None
        try:
            selected_paths = _scope_selection(
                form.get("scope_selection", '["."]')
            )
            await asyncio.to_thread(
                knowledge_service.add_source,
                root_value=root_value,
                display_name=display_name,
                consent=consent,
                selected_paths=selected_paths,
                require_credentials=True,
            )
        except (KnowledgeDashboardError, OSError, RuntimeError, ValueError) as exc:
            retained_preview: dict[str, object] | None = None
            if root_value:
                try:
                    retained_preview = await asyncio.to_thread(
                        knowledge_service.preview,
                        root_value,
                    )
                    scope = retained_preview.get("scope")
                    if (
                        isinstance(scope, dict)
                        and selected_paths is not None
                    ):
                        _retain_scope_selection(scope, selected_paths)
                except (KnowledgeDashboardError, OSError, RuntimeError, ValueError):
                    retained_preview = None
            return await render_knowledge(
                request,
                preview=retained_preview,
                root_value=root_value,
                enrollment_form=enrollment_form,
                error=str(exc),
                status_code=422,
            )
        return RedirectResponse(
            url="/knowledge?message=source-added", status_code=303
        )

    @app.post(
        "/knowledge/sources/{source_id}/sync",
        response_class=HTMLResponse,
    )
    async def sync_knowledge_source(
        source_id: str, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            await asyncio.to_thread(
                knowledge_service.queue_sync,
                source_id,
                require_credentials=True,
            )
        except (KnowledgeDashboardError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(
            url="/knowledge?message=sync-queued", status_code=303
        )

    @app.post(
        "/knowledge/sources/{source_id}/enabled",
        response_class=HTMLResponse,
    )
    async def toggle_knowledge_source(
        source_id: str, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            await asyncio.to_thread(
                knowledge_service.toggle_source,
                source_id,
                enabled=str(form.get("enabled", "")) == "true",
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(url="/knowledge", status_code=303)

    @app.post(
        "/knowledge/sources/{source_id}/scope",
        response_class=HTMLResponse,
    )
    async def update_knowledge_source_scope(
        source_id: str, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            selected_paths = _scope_selection(
                form.get("scope_selection", "[]")
            )
            await asyncio.to_thread(
                knowledge_service.update_source_scope,
                source_id,
                selected_paths,
            )
        except (
            KeyError,
            KnowledgeDashboardError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(
            url="/knowledge?message=scope-updated",
            status_code=303,
        )

    @app.post(
        "/knowledge/sources/{source_id}/auto-sync",
        response_class=HTMLResponse,
    )
    async def toggle_knowledge_auto_sync(
        source_id: str, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            await asyncio.to_thread(
                knowledge_service.toggle_auto_sync,
                source_id,
                enabled=str(form.get("enabled", "")) == "true",
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(url="/knowledge", status_code=303)

    @app.post(
        "/knowledge/sources/{source_id}/remove",
        response_class=HTMLResponse,
    )
    async def remove_knowledge_source(
        source_id: str, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            await asyncio.to_thread(knowledge_service.remove_source, source_id)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(
            url="/knowledge?message=source-removed", status_code=303
        )

    @app.post(
        "/knowledge/sources/{source_id}/consent",
        response_class=HTMLResponse,
    )
    async def renew_knowledge_consent(
        source_id: str, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            await asyncio.to_thread(
                knowledge_service.renew_consent,
                source_id,
                consent=str(form.get("consent", "")) == "on",
                require_credentials=True,
            )
        except (
            KeyError,
            KnowledgeDashboardError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(
            url="/knowledge?message=consent-renewed", status_code=303
        )

    @app.post(
        "/knowledge/source-files/{file_id}/retry",
        response_class=HTMLResponse,
    )
    async def retry_failed_knowledge_file(
        file_id: int, request: Request
    ) -> Response:
        form = await request.form()
        _verify_local_request(
            request,
            str(form.get("csrf_token", "")),
            app.state.csrf_token,
        )
        try:
            await asyncio.to_thread(knowledge_service.retry_failed_file, file_id)
        except (
            KnowledgeDashboardError,
            KeyError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return await render_knowledge(request, error=str(exc), status_code=422)
        return RedirectResponse(
            url="/knowledge?message=retry-queued", status_code=303
        )

    return app
