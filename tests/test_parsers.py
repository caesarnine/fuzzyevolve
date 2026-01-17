"""Tests for LLM prompt builders."""

from fuzzyevolve.core.models import Elite
from fuzzyevolve.adapters.llm.prompts import build_mutation_prompt, build_rank_prompt


class DummyRating:
    def __init__(self, mu: float = 25.0, sigma: float = 8.333):
        self.mu = mu
        self.sigma = sigma


def make_elite(text: str) -> Elite:
    return Elite(
        text=text,
        descriptor={"len": len(text)},
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
            parent=parent,
            inspirations=[],
            goal="Improve the text.",
            instructions="Make it clearer.",
            max_edits=2,
            metrics=["clarity", "creativity"],
            metric_descriptions=None,
            show_metric_stats=True,
            score_lcb_c=1.0,
            inspiration_labels=None,
        )

        lowered = prompt.lower()
        assert "<thinking>" not in lowered
        assert "<diffs>" not in lowered
        assert "return up to 2" in lowered
        assert "`search`" in prompt

    def test_score_uses_metric_c(self):
        parent = Elite(
            text="Hello.",
            descriptor={"len": 6},
            ratings={"clarity": DummyRating(mu=10.0, sigma=1.0)},
            age=0,
        )
        inspiration = Elite(
            text="Inspire.",
            descriptor={"len": 8},
            ratings={"clarity": DummyRating(mu=20.0, sigma=2.0)},
            age=0,
        )
        prompt = build_mutation_prompt(
            parent=parent,
            inspirations=[inspiration],
            goal="Improve the text.",
            instructions="Make it clearer.",
            max_edits=1,
            metrics=["clarity"],
            metric_descriptions=None,
            show_metric_stats=False,
            score_lcb_c=1.0,
            inspiration_labels=None,
        )

        assert "Score: 9.000" in prompt
        assert "[1] score=18.000" in prompt

    def test_metric_definitions_included_when_provided(self):
        parent = make_elite("Hello world.")
        prompt = build_mutation_prompt(
            parent=parent,
            inspirations=[],
            goal="Improve the text.",
            instructions="Make it clearer.",
            max_edits=1,
            metrics=["clarity", "creativity"],
            metric_descriptions={
                "clarity": "Easy to follow and unambiguous.",
                "creativity": "Fresh and surprising ideas.",
            },
            show_metric_stats=False,
            score_lcb_c=1.0,
            inspiration_labels=None,
        )

        assert "Metric definitions:" in prompt
        assert "- clarity: Easy to follow and unambiguous." in prompt
        assert "- creativity: Fresh and surprising ideas." in prompt

    def test_inspiration_labels_rendered_when_provided(self):
        parent = make_elite("Hello.")
        inspiration = make_elite("Inspire.")
        prompt = build_mutation_prompt(
            parent=parent,
            inspirations=[inspiration],
            goal="Improve the text.",
            instructions="Make it clearer.",
            max_edits=1,
            metrics=["clarity", "creativity"],
            metric_descriptions=None,
            show_metric_stats=False,
            score_lcb_c=1.0,
            inspiration_labels=["CHAMPION"],
        )

        assert "[1] CHAMPION score=" in prompt


class TestRankPrompt:
    def test_no_thinking_tags(self):
        e0 = make_elite("Text A.")
        e1 = make_elite("Text B.")
        prompt = build_rank_prompt(
            metrics=["clarity", "creativity"],
            items=[(0, e0.text), (1, e1.text)],
            metric_descriptions=None,
        )

        lowered = prompt.lower()
        assert "<thinking>" not in lowered
        assert "<output>" not in lowered
        assert "metrics:" in lowered
        assert "[0]" in prompt
        assert "[1]" in prompt

    def test_metric_definitions_included_when_provided(self):
        e0 = make_elite("Text A.")
        e1 = make_elite("Text B.")
        prompt = build_rank_prompt(
            metrics=["clarity", "creativity"],
            items=[(0, e0.text), (1, e1.text)],
            metric_descriptions={
                "clarity": "Easy to follow and unambiguous.",
                "creativity": "Fresh and surprising ideas.",
            },
        )

        assert "Metric definitions:" in prompt
        assert "- clarity: Easy to follow and unambiguous." in prompt
        assert "- creativity: Fresh and surprising ideas." in prompt
