"""Tests for LLM prompt builders."""

from fuzzyevolve.core.models import Elite
from fuzzyevolve.llm.prompts import build_mutation_prompt, build_rank_prompt


class DummyRating:
    def __init__(self, mu: float = 25.0, sigma: float = 8.333):
        self.mu = mu
        self.sigma = sigma


def make_elite(text: str) -> Elite:
    return Elite(
        text=text,
        descriptor={"lang": "txt", "len": len(text)},
        ratings={
            "clarity": DummyRating(),
            "creativity": DummyRating(),
        },
        age=0,
    )


class TestMutationPrompt:
    def test_no_thinking_tags(self):
        parent = make_elite("Hello world.")
        prompt = build_mutation_prompt(
            parent,
            inspirations=[],
            goal="Improve the text.",
            instructions="Make it clearer.",
            max_diffs=2,
            show_metric_stats=True,
            metric_c=1.0,
        )

        lowered = prompt.lower()
        assert "<thinking>" not in lowered
        assert "<diffs>" not in lowered
        assert "return up to 2" in lowered
        assert "`search`" in prompt


class TestRankPrompt:
    def test_no_thinking_tags(self):
        e0 = make_elite("Text A.")
        e1 = make_elite("Text B.")
        prompt = build_rank_prompt(["clarity", "creativity"], [(0, e0), (1, e1)])

        lowered = prompt.lower()
        assert "<thinking>" not in lowered
        assert "<output>" not in lowered
        assert "metrics:" in lowered
        assert "[0]" in prompt
        assert "[1]" in prompt
