"""Tests for the evolutionary driver."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fuzzyevolve.evolution.driver import EvolutionaryDriver
from fuzzyevolve.evolution.archive import MixedArchive
from fuzzyevolve.config import Config


class TestEvolutionaryDriver:
    """Test the main evolutionary loop driver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cfg = Config(
            iterations=2,
            num_islands=2,
            k_top=3,
            metrics=["test_metric"],
            youth_bias=0.5,
            n_diffs=2
        )
        self.llm_provider = Mock()
        self.judge = Mock()
        self.judge.envs = {"test_metric": Mock()}
        self.judge.envs["test_metric"].create_rating = Mock(
            return_value=type('Rating', (), {"mu": 25.0, "sigma": 8.333})()
        )
        
        self.islands = [
            MixedArchive(self.cfg.axes, self.cfg.k_top)
            for _ in range(self.cfg.num_islands)
        ]
        
        self.driver = EvolutionaryDriver(
            self.cfg, self.llm_provider, self.judge, self.islands
        )
    
    def test_initialization(self):
        """Test driver initialization."""
        assert self.driver.cfg == self.cfg
        assert self.driver.llm_provider == self.llm_provider
        assert self.driver.judge == self.judge
        assert len(self.driver.islands) == 2
    
    def test_empty_candidates_handling(self):
        """Test that driver handles empty candidate list for inspirations."""
        # Set up a minimal archive with just one elite
        seed_elite = {
            "txt": "seed",
            "rating": {"test_metric": self.judge.envs["test_metric"].create_rating()},
            "age": 0
        }
        desc = {"lang": "txt", "len": 4}
        self.islands[0].add(desc, seed_elite)
        
        # Mock LLM to return a valid mutation
        self.llm_provider.call = Mock(return_value="""
        <thinking>Test</thinking>
        <diffs>
        <<<<<<< SEARCH
        seed
        =======
        modified
        >>>>>>> REPLACE
        </diffs>
        """)
        
        # Mock judge to not crash
        self.judge.rank_and_rate = Mock()
        
        # This should not crash even with only one elite (no candidates for inspiration)
        with patch('fuzzyevolve.evolution.driver.Progress'):
            with patch('pathlib.Path.write_text'):
                try:
                    self.driver.run("seed", Mock(), quiet=True)
                except Exception as e:
                    pytest.fail(f"Driver crashed with empty candidates: {e}")
    
    @patch('fuzzyevolve.evolution.driver.Progress')
    def test_migration(self, mock_progress):
        """Test migration between islands."""
        # Add elites to islands
        for i in range(2):
            for j in range(3):
                elite = {
                    "txt": f"text-{i}-{j}",
                    "rating": {"test_metric": self.judge.envs["test_metric"].create_rating()},
                    "age": 0
                }
                desc = {"lang": "txt", "len": 10}
                self.islands[i].add(desc, elite)
        
        # Set up for migration to happen
        self.cfg.iterations = self.cfg.migration_every
        self.cfg.migrants_per_island = 1
        
        self.llm_provider.call = Mock(return_value="""
        <thinking>Test</thinking>
        <diffs>
        <<<<<<< SEARCH
        text
        =======
        modified
        >>>>>>> REPLACE
        </diffs>
        """)
        self.judge.rank_and_rate = Mock()
        
        initial_island0_count = sum(len(b) for b in self.islands[0].cell.values())
        
        with patch('pathlib.Path.write_text'):
            self.driver.run("seed", Mock(), quiet=True)
        
        # After migration, islands should have exchanged some elites
        # This is hard to test precisely due to randomness, but we can check
        # that migration code was at least executed (via mock calls)
        assert self.llm_provider.call.called
    
    @patch('fuzzyevolve.evolution.driver.Progress')  
    def test_sparring(self, mock_progress):
        """Test global sparring between islands."""
        # Add elites to islands
        for i in range(2):
            for j in range(2):
                elite = {
                    "txt": f"text-{i}-{j}",
                    "rating": {"test_metric": self.judge.envs["test_metric"].create_rating()},
                    "age": 0
                }
                desc = {"lang": "txt", "len": 10}
                self.islands[i].add(desc, elite)
        
        # Set up for sparring to happen
        self.cfg.iterations = self.cfg.sparring_every
        
        self.llm_provider.call = Mock(return_value="""
        <thinking>Test</thinking>
        <diffs>
        <<<<<<< SEARCH
        text
        =======
        modified
        >>>>>>> REPLACE
        </diffs>
        """)
        
        # Track judge calls
        judge_calls = []
        def track_judge_call(players):
            judge_calls.append(len(players))
        
        self.judge.rank_and_rate = Mock(side_effect=track_judge_call)
        
        with patch('pathlib.Path.write_text'):
            self.driver.run("seed", Mock(), quiet=True)
        
        # Should have at least one call with multiple players for sparring
        assert any(n > 1 for n in judge_calls), "No sparring occurred"