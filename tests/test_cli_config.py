from __future__ import annotations

from pathlib import Path

import pytest
import typer

from fuzzyevolve.cli import _resolve_config_path


def test_resolve_config_prefers_explicit_path(tmp_path: Path):
    cfg = tmp_path / "my.toml"
    cfg.write_text("iterations = 1\n")
    path, msg = _resolve_config_path(cfg, cwd=tmp_path)
    assert path == cfg
    assert "Using config file:" in msg


def test_resolve_config_autodetects_toml_in_cwd(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("iterations = 1\n")
    path, msg = _resolve_config_path(None, cwd=tmp_path)
    assert path == cfg
    assert "from CWD" in msg


def test_resolve_config_prefers_toml_over_json(tmp_path: Path):
    toml_cfg = tmp_path / "config.toml"
    json_cfg = tmp_path / "config.json"
    toml_cfg.write_text("iterations = 1\n")
    json_cfg.write_text("{\"iterations\": 2}\n")
    path, _ = _resolve_config_path(None, cwd=tmp_path)
    assert path == toml_cfg


def test_resolve_config_falls_back_to_json(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{\"iterations\": 2}\n")
    path, _ = _resolve_config_path(None, cwd=tmp_path)
    assert path == cfg


def test_resolve_config_uses_defaults_when_missing(tmp_path: Path):
    path, msg = _resolve_config_path(None, cwd=tmp_path)
    assert path is None
    assert "default" in msg.lower()


def test_resolve_config_errors_when_explicit_missing(tmp_path: Path):
    missing = tmp_path / "missing.toml"
    with pytest.raises(typer.BadParameter):
        _resolve_config_path(missing, cwd=tmp_path)
