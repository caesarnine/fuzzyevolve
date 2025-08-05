"""Tests for diff utilities."""

import pytest
from fuzzyevolve.utils.diff import split_blocks, apply_patch


class TestDiffUtilities:
    """Test diff parsing and application."""
    
    def test_split_single_block(self):
        """Test splitting a single diff block."""
        diff = """<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE"""
        
        blocks = split_blocks(diff)
        assert len(blocks) == 1
        assert blocks[0] == diff
    
    def test_split_multiple_blocks(self):
        """Test splitting multiple diff blocks."""
        diff = """<<<<<<< SEARCH
first old
=======
first new
>>>>>>> REPLACE
<<<<<<< SEARCH
second old
=======
second new
>>>>>>> REPLACE"""
        
        blocks = split_blocks(diff)
        assert len(blocks) == 2
        assert "first old" in blocks[0]
        assert "second old" in blocks[1]
    
    def test_split_blocks_with_text_between(self):
        """Test splitting when there's text between blocks."""
        diff = """Some preamble
<<<<<<< SEARCH
first old
=======
first new
>>>>>>> REPLACE
Some text in between
<<<<<<< SEARCH
second old
=======
second new
>>>>>>> REPLACE"""
        
        blocks = split_blocks(diff)
        # The function splits on <<<<<<< SEARCH, so:
        # 1. "Some preamble" (before first SEARCH)
        # 2. First diff block + "Some text in between" 
        # 3. Second diff block
        assert len(blocks) == 3
        # Preamble should be first block
        assert blocks[0] == "Some preamble"
        # First diff should be in second block
        assert "first old" in blocks[1]
        assert "Some text in between" in blocks[1]
        # Second diff should be in third block
        assert "second old" in blocks[2]
    
    def test_apply_valid_patch(self):
        """Test applying a valid patch."""
        text = "Hello world. This is a test. Goodbye."
        diff = """<<<<<<< SEARCH
This is a test.
=======
This is an improved test.
>>>>>>> REPLACE"""
        
        result = apply_patch(text, diff)
        assert result == "Hello world. This is an improved test. Goodbye."
    
    def test_apply_patch_not_found(self):
        """Test applying patch when search text not found."""
        text = "Hello world."
        diff = """<<<<<<< SEARCH
Not in text
=======
Replacement
>>>>>>> REPLACE"""
        
        result = apply_patch(text, diff)
        assert result == text  # Should return original
    
    def test_apply_patch_multiline(self):
        """Test applying patch with multiline content."""
        text = """Line 1
Line 2
Line 3"""
        diff = """<<<<<<< SEARCH
Line 2
=======
Modified Line 2
With Extra Line
>>>>>>> REPLACE"""
        
        result = apply_patch(text, diff)
        expected = """Line 1
Modified Line 2
With Extra Line
Line 3"""
        assert result == expected
    
    def test_apply_patch_malformed(self):
        """Test handling malformed diff block."""
        text = "Original text"
        
        # Missing REPLACE marker
        diff = """<<<<<<< SEARCH
old
=======
new"""
        
        result = apply_patch(text, diff)
        assert result == text  # Should return original
        
        # Missing separator
        diff2 = """<<<<<<< SEARCH
old
new
>>>>>>> REPLACE"""
        
        result2 = apply_patch(text, diff2)
        assert result2 == text  # Should return original
    
    def test_apply_patch_strips_newlines(self):
        """Test that search and replace strings are stripped of newlines."""
        text = "Hello world"
        diff = """<<<<<<< SEARCH

Hello world

=======

Goodbye world

>>>>>>> REPLACE"""
        
        result = apply_patch(text, diff)
        assert result == "Goodbye world"
    
    def test_apply_patch_only_first_occurrence(self):
        """Test that only first occurrence is replaced."""
        text = "foo bar foo baz foo"
        diff = """<<<<<<< SEARCH
foo
=======
FOO
>>>>>>> REPLACE"""
        
        result = apply_patch(text, diff)
        assert result == "FOO bar foo baz foo"  # Only first foo replaced