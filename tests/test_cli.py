from pathlib import Path
from types import SimpleNamespace

from keepygaga_rag import cli


def test_no_subcommand_launches_dashboard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "keepygaga-rag.toml"
    called: list[Path] = []
    monkeypatch.setattr(
        cli,
        "_launch_dashboard",
        lambda path: called.append(path) or 0,
    )

    assert cli.main(["--config", str(config_path)]) == 0
    assert called == [config_path.resolve()]


def test_running_dashboard_opens_without_starting_another_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(cli, "_dashboard_is_ready", lambda _url: True)
    monkeypatch.setattr(
        cli.webbrowser,
        "open",
        lambda url: opened.append(url) or True,
    )
    monkeypatch.setattr(
        cli,
        "_port_is_open",
        lambda _port: (_ for _ in ()).throw(
            AssertionError("port check should not run")
        ),
    )

    assert cli._launch_dashboard(tmp_path / "keepygaga-rag.toml") == 0
    assert opened == ["http://127.0.0.1:8765/"]


def test_launcher_enables_auto_close_for_background_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environments: list[dict[str, str]] = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_dashboard_is_ready", lambda _url: bool(environments))
    monkeypatch.setattr(cli, "_port_is_open", lambda _port: False)
    monkeypatch.setattr(cli, "_dashboard_command", lambda: ["/tool/dashboard"])
    monkeypatch.setattr(cli.webbrowser, "open", lambda _url: True)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *_args, **kwargs: (
            environments.append(kwargs["env"])
            or SimpleNamespace(poll=lambda: None)
        ),
    )

    assert cli._launch_dashboard(tmp_path / "keepygaga-rag.toml") == 0
    assert environments[0]["KEEPYGAGA_RAG_DASHBOARD_AUTO_CLOSE"] == "1"
    assert environments[0]["KEEPYGAGA_RAG_CONFIG"] == str(
        (tmp_path / "keepygaga-rag.toml").resolve()
    )


def test_dashboard_subcommand_uses_the_primary_cli(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "keepygaga-rag.toml"
    called: list[Path] = []
    monkeypatch.setattr(
        cli,
        "_launch_dashboard",
        lambda path: called.append(path) or 0,
    )

    assert cli.main(["--config", str(config_path), "dashboard"]) == 0
    assert called == [config_path.resolve()]


def test_indexer_subcommand_passes_config_and_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from keepygaga_rag.knowledge import indexer_cli

    config_path = tmp_path / "keepygaga-rag.toml"
    called: list[list[str] | None] = []
    monkeypatch.setattr(
        indexer_cli,
        "main",
        lambda argv=None: called.append(argv) or 0,
    )

    assert cli.main(["--config", str(config_path), "indexer", "--once"]) == 0
    assert called == [["--config", str(config_path.resolve()), "--once"]]
