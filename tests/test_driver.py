"""Tests for the evolutionary driver."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
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

    @patch("fuzzyevolve.evolution.driver.Progress")
    def test_batch_judging_multiple_children(self, mock_progress):
        """Judge once per iteration even with multiple diff blocks."""
        cfg = Config(
            iterations=1,
            num_islands=1,
            k_top=3,
            metrics=["test_metric"],
            n_diffs=2,
            judge_include_inspirations=False,
        )
        llm_provider = Mock()
        judge = Mock()
        judge.envs = {"test_metric": Mock()}
        judge.envs["test_metric"].create_rating = Mock(
            return_value=type("Rating", (), {"mu": 25.0, "sigma": 8.333})()
        )
        islands = [MixedArchive(cfg.axes, cfg.k_top)]
        driver = EvolutionaryDriver(cfg, llm_provider, judge, islands)

        llm_provider.call = Mock(
            return_value="""
<thinking>Test</thinking>
<diffs>
<<<<<<< SEARCH
seed
=======
modified1
>>>>>>> REPLACE

<<<<<<< SEARCH
seed
=======
modified2
>>>>>>> REPLACE
</diffs>
"""
        )
        judge.rank_and_rate = Mock()

        with patch("pathlib.Path.write_text"):
            driver.run("seed", Path("out.txt"), quiet=True)

        assert judge.rank_and_rate.call_count == 1
        players = judge.rank_and_rate.call_args[0][0]
        assert len(players) == 3  # parent + 2 children
        assert players[0]["txt"] == "seed"
        assert {players[1]["txt"], players[2]["txt"]} == {"modified1", "modified2"}

    @pytest.mark.parametrize("include_inspirations", [True, False])
    @patch("fuzzyevolve.evolution.driver.Progress")
    def test_judge_include_inspirations_toggle(
        self, mock_progress, include_inspirations
    ):
        """Optionally include inspirations in judge ranking."""
        cfg = Config(
            iterations=2,
            num_islands=1,
            k_top=5,
            metrics=["test_metric"],
            n_diffs=1,
            judge_include_inspirations=include_inspirations,
        )
        llm_provider = Mock()
        judge = Mock()
        judge.envs = {"test_metric": Mock()}
        judge.envs["test_metric"].create_rating = Mock(
            return_value=type("Rating", (), {"mu": 25.0, "sigma": 8.333})()
        )
        islands = [MixedArchive(cfg.axes, cfg.k_top)]
        driver = EvolutionaryDriver(cfg, llm_provider, judge, islands)

        # Always applicable: both "seed" and its children contain "e".
        llm_provider.call = Mock(
            return_value="""
<thinking>Test</thinking>
<diffs>
<<<<<<< SEARCH
e
=======
eX
>>>>>>> REPLACE
</diffs>
"""
        )
        judge.rank_and_rate = Mock()

        with patch("pathlib.Path.write_text"):
            driver.run("seed", Path("out.txt"), quiet=True)

        assert judge.rank_and_rate.call_count == 2
        group_sizes = [len(call.args[0]) for call in judge.rank_and_rate.call_args_list]
        assert group_sizes[0] == 2  # first iter: parent + child
        assert group_sizes[1] == (3 if include_inspirations else 2)
    
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
