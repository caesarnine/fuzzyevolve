"""Tests for config defaults and validation."""

import pytest

from fuzzyevolve.config import Config, EmbeddingsConfig, MetricsConfig
from fuzzyevolve.config import MutationOperatorConfig


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


def test_crossover_operator_defaults_committee_size():
    cfg = Config(
        mutation={
            "operators": [
                MutationOperatorConfig(name="x", role="crossover", min_jobs=1)
            ]
        }
    )
    op = cfg.mutation.operators[0]
    assert op.role == "crossover"
    assert op.committee_size == 3


def test_crossover_operator_rejects_too_small_farthest_k():
    with pytest.raises(ValueError):
        Config(
            mutation={
                "operators": [
                    MutationOperatorConfig(
                        name="x",
                        role="crossover",
                        committee_size=5,
                        partner_selection="far_random",
                        partner_farthest_k=2,
                        min_jobs=1,
                    )
                ]
            }
        )
