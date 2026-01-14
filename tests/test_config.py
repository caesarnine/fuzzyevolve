"""Tests for config defaults and validation."""

from fuzzyevolve.config import Config


def test_default_score_cs_are_aligned():
    cfg = Config()
    assert cfg.archive_score_c == cfg.report_score_c

