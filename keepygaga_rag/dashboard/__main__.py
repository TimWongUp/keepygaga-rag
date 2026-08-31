from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from keepygaga_rag.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from keepygaga_rag.dashboard.web import create_app


def _raise_open_file_limit(target: int = 65_536) -> None:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(target, hard)
        if desired > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (ImportError, OSError, ValueError):
        pass


def main() -> None:
    _raise_open_file_limit()
    config_path = Path(
        os.environ.get("KEEPYGAGA_RAG_CONFIG", str(DEFAULT_CONFIG_PATH))
    ).expanduser()
    port = int(os.environ.get("KEEPYGAGA_RAG_DASHBOARD_PORT", "8765"))
    auto_close = os.environ.get(
        "KEEPYGAGA_RAG_DASHBOARD_AUTO_CLOSE", ""
    ).casefold() in {"1", "true", "yes", "on"}
    server: uvicorn.Server | None = None

    def request_shutdown() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(
        config_path=config_path,
        project_root=PROJECT_ROOT,
        auto_close=auto_close,
        shutdown_callback=request_shutdown if auto_close else None,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info",
        )
    )
    server.run()


if __name__ == "__main__":
    main()
