"""Tests for LLM response parsers."""

import logging
import pytest
from fuzzyevolve.llm.parsers import parse_llm_mutation_response, parse_llm_judge_response


class TestMutationParser:
    """Test mutation response parsing."""
    
    def setup_method(self):
        """Set up test logger."""
        self.logger = logging.getLogger("test")
    
    def test_parse_valid_mutation_response(self):
        """Test parsing a well-formed mutation response."""
        response = """
        <thinking>
        I will improve this text by making it more concise.
        </thinking>
        <diffs>
        <<<<<<< SEARCH
        old text here
        =======
        new improved text
        >>>>>>> REPLACE
        </diffs>
        """
        
        thinking, diffs = parse_llm_mutation_response(response, self.logger)
        
        assert thinking == "I will improve this text by making it more concise."
        assert "<<<<<<< SEARCH" in diffs
        assert ">>>>>>> REPLACE" in diffs
        assert "</diffs>" not in diffs  # Should NOT include closing tag
    
    def test_parse_multiple_diffs(self):
        """Test parsing response with multiple diff blocks."""
        response = """
        <thinking>
        Multiple improvements needed.
        </thinking>
        <diffs>
        <<<<<<< SEARCH
        first old
        =======
        first new
        >>>>>>> REPLACE
        
        <<<<<<< SEARCH
        second old
        =======
        second new
        >>>>>>> REPLACE
        </diffs>
        """
        
        thinking, diffs = parse_llm_mutation_response(response, self.logger)
        
        assert thinking == "Multiple improvements needed."
        assert diffs.count("<<<<<<< SEARCH") == 2
        assert diffs.count(">>>>>>> REPLACE") == 2
    
    def test_parse_missing_thinking_tag(self):
        """Test parsing when thinking tag is missing."""
        response = """
        <diffs>
        <<<<<<< SEARCH
        old
        =======
        new
        >>>>>>> REPLACE
        </diffs>
        """
        
        thinking, diffs = parse_llm_mutation_response(response, self.logger)
        
        assert thinking is None
        assert diffs is not None
        assert "<<<<<<< SEARCH" in diffs
    
    def test_parse_missing_diffs_tag_with_fallback(self):
        """Test fallback when diffs tag is missing but diff content exists."""
        response = """
        <<<<<<< SEARCH
        old text
        =======
        new text
        >>>>>>> REPLACE
        """
        
        thinking, diffs = parse_llm_mutation_response(response, self.logger)
        
        assert thinking is None
        assert diffs == response  # Should use raw response as fallback (not stripped)
    
    def test_parse_case_insensitive_tags(self):
        """Test that tag matching is case-insensitive."""
        response = """
        <THINKING>
        Uppercase thinking.
        </THINKING>
        <DIFFS>
        <<<<<<< SEARCH
        old
        =======
        new
        >>>>>>> REPLACE
        </DIFFS>
        """
        
        thinking, diffs = parse_llm_mutation_response(response, self.logger)
        
        assert thinking == "Uppercase thinking."
        assert "<<<<<<< SEARCH" in diffs


class TestJudgeParser:
    """Test judge response parsing."""
    
    def setup_method(self):
        """Set up test logger."""
        self.logger = logging.getLogger("test")
        self.metrics = ["clarity", "creativity", "impact"]
    
    def test_parse_valid_judge_response(self):
        """Test parsing a well-formed judge response."""
        response = """
        <thinking>
        Candidate 2 shows the best clarity, followed by 0 and 1.
        For creativity, 1 is best, then 2, then 0.
        Impact-wise, 0 leads, then 1, then 2.
        </thinking>
        <output>
        <clarity>2, 0, 1</clarity>
        <creativity>1, 2, 0</creativity>
        <impact>0, 1, 2</impact>
        </output>
        """
        
        thinking, rankings = parse_llm_judge_response(response, self.metrics, self.logger)
        
        assert thinking is not None
        assert "Candidate 2 shows the best clarity" in thinking
        assert rankings["clarity"] == [2, 0, 1]
        assert rankings["creativity"] == [1, 2, 0]
        assert rankings["impact"] == [0, 1, 2]
    
    def test_parse_with_brackets(self):
        """Test parsing when IDs are in brackets."""
        response = """
        <thinking>
        Analysis complete.
        </thinking>
        <output>
        <clarity>[1, 0, 2]</clarity>
        <creativity>[2, 1, 0]</creativity>
        <impact>[0, 2, 1]</impact>
        </output>
        """
        
        thinking, rankings = parse_llm_judge_response(response, self.metrics, self.logger)
        
        assert rankings["clarity"] == [1, 0, 2]
        assert rankings["creativity"] == [2, 1, 0]
        assert rankings["impact"] == [0, 2, 1]
    
    def test_parse_missing_metric(self):
        """Test handling when a metric is missing."""
        response = """
        <thinking>
        Partial analysis.
        </thinking>
        <output>
        <clarity>1, 0</clarity>
        <creativity>0, 1</creativity>
        </output>
        """
        
        thinking, rankings = parse_llm_judge_response(response, self.metrics, self.logger)
        
        assert "clarity" in rankings
        assert "creativity" in rankings
        assert "impact" not in rankings  # Missing metric
    
    def test_regex_fallback(self):
        """Test regex fallback when XML parsing fails."""
        # Malformed XML that will fail lxml parsing
        response = """
        <thinking>
        Some analysis
        </thinking>
        <output>
        <clarity>1, 0, 2
        <creativity>2, 1, 0</creativity>
        <impact>0, 2, 1</impact>
        </output>
        """
        
        thinking, rankings = parse_llm_judge_response(response, self.metrics, self.logger)
        
        # Should still extract what it can via regex
        assert thinking is not None
        assert len(rankings) > 0  # Should get at least some rankings via fallback