"""Tests for the fuzzy patcher."""

from fuzzyevolve.mutation.patcher import PatchConfig, apply_patch


def test_exact_patch_wins():
    cfg = PatchConfig(
        fuzzy_enabled=True,
        threshold=0.9,
        margin=0.05,
        min_search_len=3,
        max_window_expansion=0.2,
    )
    result = apply_patch("hello world", "hello", "hi", cfg)
    assert result.success is True
    assert result.used_fuzzy is False
    assert result.new_text == "hi world"


def test_fuzzy_patch_applies_when_exact_missing():
    cfg = PatchConfig(
        fuzzy_enabled=True,
        threshold=0.8,
        margin=0.0,
        min_search_len=3,
        max_window_expansion=0.2,
    )
    result = apply_patch("hello world", "hello worlt", "hello world", cfg)
    assert result.success is True
    assert result.used_fuzzy is True
    assert result.new_text == "hello world"


def test_fuzzy_patch_rejected_on_short_search():
    cfg = PatchConfig(
        fuzzy_enabled=True,
        threshold=0.8,
        margin=0.0,
        min_search_len=5,
        max_window_expansion=0.2,
    )
    result = apply_patch("hello world", "helo", "hola", cfg)
    assert result.success is False
    assert result.used_fuzzy is False


def test_fuzzy_patch_rejected_on_low_margin():
    cfg = PatchConfig(
        fuzzy_enabled=True,
        threshold=0.6,
        margin=0.2,
        min_search_len=3,
        max_window_expansion=0.2,
    )
    result = apply_patch("foo bar foo", "f0o", "baz", cfg)
    assert result.success is False
    assert result.used_fuzzy is True
