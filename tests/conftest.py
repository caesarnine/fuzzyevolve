"""Pytest configuration and shared fixtures."""

import logging
import sys
from pathlib import Path

import pytest

from fuzzyevolve.config import Config
from fuzzyevolve.core.models import Elite

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def mock_rating():
    class MockRating:
        def __init__(self, mu: float = 25.0, sigma: float = 8.333):
            self.mu = mu
            self.sigma = sigma

    return MockRating


@pytest.fixture
def sample_elite(mock_rating):
    return Elite(
        text="Sample text content",
        descriptor={"lang": "txt", "len": 21},
        ratings={
            "clarity": mock_rating(mu=20.0, sigma=5.0),
            "creativity": mock_rating(mu=25.0, sigma=7.0),
            "impact": mock_rating(mu=22.0, sigma=6.0),
        },
        age=10,
    )


@pytest.fixture
def sample_config():
    return Config(
        iterations=10,
        island_count=2,
        elites_per_cell=3,
        metrics=["clarity", "creativity", "impact"],
        anchor_injection_prob=0.0,
        axes={
            "lang": ["txt", "code"],
            "len": {"bins": [0, 100, 500, 1000, 10000]},
        },
        max_diffs=2,
    )
