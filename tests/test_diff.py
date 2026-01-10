"""Tests for diff utilities."""

from fuzzyevolve.mutation.diff import extract_blocks, parse_block


class TestDiffUtilities:
    def test_extract_single_block(self):
        diff = """<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE"""

        blocks = extract_blocks(diff)
        assert len(blocks) == 1
        assert blocks[0].search == "old text"
        assert blocks[0].replace == "new text"

    def test_extract_multiple_blocks(self):
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

        blocks = extract_blocks(diff)
        assert len(blocks) == 2
        assert blocks[0].search == "first old"
        assert blocks[1].search == "second old"

    def test_extract_blocks_with_preamble(self):
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

        blocks = extract_blocks(diff)
        assert len(blocks) == 2
        assert blocks[0].search == "first old"
        assert blocks[1].search == "second old"

    def test_apply_valid_patch(self):
        text = "Hello world. This is a test. Goodbye."
        diff = """<<<<<<< SEARCH
This is a test.
=======
This is an improved test.
>>>>>>> REPLACE"""

        block = parse_block(diff)
        result = block.apply(text)
        assert result == "Hello world. This is an improved test. Goodbye."

    def test_apply_patch_not_found(self):
        text = "Hello world."
        diff = """<<<<<<< SEARCH
Not in text
=======
Replacement
>>>>>>> REPLACE"""

        block = parse_block(diff)
        result = block.apply(text)
        assert result is None

    def test_apply_patch_multiline(self):
        text = """Line 1
Line 2
Line 3"""
        diff = """<<<<<<< SEARCH
Line 2
=======
Modified Line 2
With Extra Line
>>>>>>> REPLACE"""

        block = parse_block(diff)
        result = block.apply(text)
        expected = """Line 1
Modified Line 2
With Extra Line
Line 3"""
        assert result == expected

    def test_apply_patch_malformed(self):
        diff = """<<<<<<< SEARCH
old
=======
new"""

        block = parse_block(diff)
        assert block is None

    def test_apply_patch_strips_newlines(self):
        text = "Hello world"
        diff = """<<<<<<< SEARCH

Hello world

=======

Goodbye world

>>>>>>> REPLACE"""

        block = parse_block(diff)
        result = block.apply(text)
        assert result == "Goodbye world"

    def test_apply_patch_only_first_occurrence(self):
        text = "foo bar foo baz foo"
        diff = """<<<<<<< SEARCH
foo
=======
FOO
>>>>>>> REPLACE"""

        block = parse_block(diff)
        result = block.apply(text)
        assert result == "FOO bar foo baz foo"
