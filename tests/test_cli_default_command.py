from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

import fuzzyevolve.cli as cli


def test_default_command_inserts_run_for_seed_text(monkeypatch):
    called = {}

    def _fake_execute_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "_execute_run", _fake_execute_run)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["hello", "world"])
    assert result.exit_code == 0
    assert called["seed_parts"] == ["hello", "world"]


def test_default_command_inserts_run_for_options(monkeypatch):
    called = {}

    def _fake_execute_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "_execute_run", _fake_execute_run)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["--iterations", "5", "seed"])
    assert result.exit_code == 0
    assert called["iterations"] == 5
    assert called["seed_parts"] == ["seed"]


def test_top_level_help_not_rewritten(monkeypatch):
    called = {"run": False}

    def _fake_execute_run(**_kwargs):
        called["run"] = True

    monkeypatch.setattr(cli, "_execute_run", _fake_execute_run)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert called["run"] is False
    assert "Commands" in result.output
    assert "run" in result.output
    assert "tui" in result.output


def test_tui_subcommand_not_rewritten(monkeypatch):
    called = SimpleNamespace(run=False, tui=False)

    def _fake_execute_run(**_kwargs):
        called.run = True

    def _fake_run_tui(*_args, **_kwargs):
        called.tui = True

    monkeypatch.setattr(cli, "_execute_run", _fake_execute_run)
    monkeypatch.setattr("fuzzyevolve.tui.app.run_tui", _fake_run_tui)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["tui"])
    assert result.exit_code == 0
    assert called.run is False
    assert called.tui is True
