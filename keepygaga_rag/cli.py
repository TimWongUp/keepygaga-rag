from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Mapping
from pathlib import Path

from keepygaga_rag.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from keepygaga_rag.diagnostics import run_doctor

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_DEFAULT_PORT = 8765


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keepygaga-rag",
        description="Open the Dashboard, or run a Keepygaga RAG command.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command")

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")

    commands.add_parser("dashboard")

    indexer = commands.add_parser("indexer")
    indexer.add_argument("--once", action="store_true")

    return parser


def _print(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _dashboard_is_ready(url: str) -> bool:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Keepygaga RAG CLI"}
    )
    try:
        with urllib.request.urlopen(request, timeout=0.75) as response:
            payload = response.read(65_536)
    except (OSError, urllib.error.URLError):
        return False
    return b"Keepygaga" in payload


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((DASHBOARD_HOST, port)) == 0


def _dashboard_command() -> list[str]:
    return [sys.executable, "-m", "keepygaga_rag.dashboard"]


def _launch_dashboard(config_path: Path) -> int:
    raw_port = os.environ.get(
        "KEEPYGAGA_RAG_DASHBOARD_PORT", str(DASHBOARD_DEFAULT_PORT)
    )
    try:
        port = int(raw_port)
        if not 1 <= port <= 65_535:
            raise ValueError
    except ValueError:
        print(
            f"Invalid KEEPYGAGA_RAG_DASHBOARD_PORT: {raw_port}",
            file=sys.stderr,
        )
        return 2

    url = f"http://{DASHBOARD_HOST}:{port}/"
    if _dashboard_is_ready(url):
        webbrowser.open(url)
        print(f"Keepygaga RAG Dashboard: {url}")
        return 0
    if _port_is_open(port):
        print(
            f"Port {port} is already used by another service.",
            file=sys.stderr,
        )
        return 1

    runtime_dir = PROJECT_ROOT / ".keepygaga"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "dashboard.log"
    environment = os.environ.copy()
    environment["KEEPYGAGA_RAG_CONFIG"] = str(config_path)
    environment["KEEPYGAGA_RAG_DASHBOARD_AUTO_CLOSE"] = "1"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            _dashboard_command(),
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(50):
        if _dashboard_is_ready(url):
            webbrowser.open(url)
            print(f"Keepygaga RAG Dashboard: {url}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.2)

    print(
        f"Dashboard failed to start; inspect {log_path}",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if args.command in {None, "dashboard"}:
        return _launch_dashboard(config_path)
    if args.command == "doctor":
        payload = run_doctor(config_path, project_root=PROJECT_ROOT)
        _print(payload)
        return 1 if payload["status"] == "error" else 0
    if args.command == "indexer":
        from keepygaga_rag.knowledge.indexer_cli import main as run_indexer

        indexer_args = ["--config", str(config_path)]
        if args.once:
            indexer_args.append("--once")
        return run_indexer(indexer_args)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
