"""Tests for config defaults and validation."""

import pytest

from fuzzyevolve.config import Config, EmbeddingsConfig, MetricsConfig


def test_default_config_has_metrics():
    cfg = Config()
    assert cfg.metrics.names


def test_metrics_names_trimmed_and_nonempty():
    cfg = Config(metrics=MetricsConfig(names=["  clarity  ", ""]))
    assert cfg.metrics.names == ["clarity"]


def test_metrics_names_reject_all_empty():
    with pytest.raises(ValueError):
        Config(metrics=MetricsConfig(names=["", "   "]))


def test_embeddings_model_rejects_hash():
    with pytest.raises(ValueError):
        Config(embeddings=EmbeddingsConfig(model="hash"))
