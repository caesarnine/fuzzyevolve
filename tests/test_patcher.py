"""Tests for the patcher."""

from fuzzyevolve.mutation.patcher import apply_patch


def test_exact_patch_applies():
    result = apply_patch("hello world", "hello", "hi")
    assert result.success is True
    assert result.new_text == "hi world"


def test_patch_rejected_when_search_missing():
    result = apply_patch("hello world", "goodbye", "hi")
    assert result.success is False
    assert result.new_text is None


def test_patch_rejected_on_empty_search():
    result = apply_patch("hello world", "", "hi")
    assert result.success is False
    assert result.new_text is None
