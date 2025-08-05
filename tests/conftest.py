"""Pytest configuration and shared fixtures."""

import pytest
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

@pytest.fixture
def mock_rating():
    """Create a mock TrueSkill rating object."""
    class MockRating:
        def __init__(self, mu=25.0, sigma=8.333):
            self.mu = mu
            self.sigma = sigma
    
    return MockRating

@pytest.fixture
def sample_elite(mock_rating):
    """Create a sample elite for testing."""
    return {
        "txt": "Sample text content",
        "rating": {
            "clarity": mock_rating(mu=20.0, sigma=5.0),
            "creativity": mock_rating(mu=25.0, sigma=7.0),
            "impact": mock_rating(mu=22.0, sigma=6.0)
        },
        "age": 10
    }

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from fuzzyevolve.config import Config
    return Config(
        iterations=10,
        num_islands=2,
        k_top=3,
        metrics=["clarity", "creativity", "impact"],
        axes={
            "lang": ["txt", "code"],
            "len": {"bins": [0, 100, 500, 1000, 10000]}
        },
        youth_bias=0.3,
        n_diffs=2
    )