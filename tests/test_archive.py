"""Tests for the MAP-Elites archive implementation."""

import pytest
from fuzzyevolve.evolution.archive import MixedArchive


class TestMixedArchive:
    """Test the MixedArchive class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.axes = {
            "lang": ["txt", "code"],
            "len": {"bins": [0, 100, 500, 1000, 10000]}
        }
        self.archive = MixedArchive(self.axes, k_top=3)
    
    def test_initialization(self):
        """Test archive initialization."""
        assert self.archive.k_top == 3
        assert self.archive.total_cells == 2 * 4  # 2 langs * 4 len bins
        assert self.archive.empty_cells == 8
        assert len(self.archive.cell) == 0
    
    def test_key_calculation(self):
        """Test cell key calculation for descriptors."""
        # Test categorical axis
        desc1 = {"lang": "txt", "len": 50}
        key1 = self.archive._key(desc1)
        assert key1 == ("txt", 0)
        
        # Test binned axis
        desc2 = {"lang": "code", "len": 250}
        key2 = self.archive._key(desc2)
        assert key2 == ("code", 1)
        
        desc3 = {"lang": "txt", "len": 999}
        key3 = self.archive._key(desc3)
        assert key3 == ("txt", 2)
    
    def test_add_elite(self):
        """Test adding elites to the archive."""
        elite1 = {
            "txt": "test content",
            "rating": {"clarity": type('Rating', (), {"mu": 25.0, "sigma": 8.333})()},
            "age": 0
        }
        desc1 = {"lang": "txt", "len": 12}
        
        self.archive.add(desc1, elite1)
        
        assert len(self.archive.cell) == 1
        assert self.archive.empty_cells == 7
        assert elite1["cell_key"] == ("txt", 0)
        assert self.archive.cell[("txt", 0)][0] == elite1
    
    def test_top_k_limit(self):
        """Test that only top-k elites are kept per cell."""
        # Add more than k_top elites to same cell
        for i in range(5):
            elite = {
                "txt": f"test {i}",
                "rating": {"metric": type('Rating', (), {"mu": float(i), "sigma": 1.0})()},
                "age": i
            }
            desc = {"lang": "txt", "len": 50}
            self.archive.add(desc, elite)
        
        # Should only keep top 3
        key = ("txt", 0)
        assert len(self.archive.cell[key]) == 3
        # Should be sorted by score (highest first)
        scores = [e["rating"]["metric"].mu for e in self.archive.cell[key]]
        assert scores == sorted(scores, reverse=True)

    def test_random_elite(self):
        """Test random elite selection from the archive."""
        for i in range(5):
            elite = {
                "txt": f"test {i}",
                "rating": {"metric": type("Rating", (), {"mu": 25.0, "sigma": 8.0})()},
                "age": i,
            }
            desc = {"lang": "txt", "len": 50}
            self.archive.add(desc, elite)

        selected = self.archive.random_elite()
        assert selected is not None
        assert "txt" in selected
    
    def test_best_property(self):
        """Test getting the best elite from archive."""
        # Add elites with different scores
        for i in range(3):
            for j in range(2):
                elite = {
                    "txt": f"test {i}-{j}",
                    "rating": {"metric": type('Rating', (), {"mu": float(i*10 + j), "sigma": 1.0})()},
                    "age": 0
                }
                desc = {"lang": "txt" if j == 0 else "code", "len": 50}
                self.archive.add(desc, elite)
        
        best = self.archive.best
        # Best should have highest mu - sigma
        assert best["rating"]["metric"].mu == 21.0
