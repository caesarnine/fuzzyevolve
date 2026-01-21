"""Tests for LLM prompt builders."""

from __future__ import annotations

import numpy as np

from fuzzyevolve.adapters.llm.prompts import (
    build_critique_prompt,
    build_rank_prompt,
    build_rewrite_prompt,
)
from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite


class DummyRating:
    def __init__(self, mu: float = 25.0, sigma: float = 8.333):
        self.mu = mu
        self.sigma = sigma


def make_elite(text: str) -> Elite:
    return Elite(
        text=text,
        embedding=np.array([1.0], dtype=float),
        ratings={
            "clarity": DummyRating(),
            "creativity": DummyRating(),
        },
        age=0,
    )


class TestCritiquePrompt:
    def test_no_thinking_tags(self):
        parent = make_elite("Hello world.")
        prompt = build_critique_prompt(
            parent=parent,
            goal="Improve the text.",
            metrics=["clarity", "creativity"],
            metric_descriptions=None,
            routes=3,
            show_metric_stats=True,
            score_lcb_c=1.0,
        )

        lowered = prompt.lower()
        assert "<thinking>" not in lowered
        assert "<output>" not in lowered
        assert "routes: 3 distinct" in lowered

    def test_score_uses_metric_c(self):
        parent = Elite(
            text="Hello.",
            embedding=np.array([1.0], dtype=float),
            ratings={"clarity": DummyRating(mu=10.0, sigma=1.0)},
            age=0,
        )
        prompt = build_critique_prompt(
            parent=parent,
            goal="Improve the text.",
            metrics=["clarity"],
            metric_descriptions=None,
            routes=2,
            show_metric_stats=False,
            score_lcb_c=1.0,
        )

        assert "Score (LCB avg): 9.000" in prompt

    def test_metric_definitions_included_when_provided(self):
        parent = make_elite("Hello world.")
        prompt = build_critique_prompt(
            parent=parent,
            goal="Improve the text.",
            metrics=["clarity", "creativity"],
            metric_descriptions={
                "clarity": "Easy to follow and unambiguous.",
                "creativity": "Fresh and surprising ideas.",
            },
            routes=2,
            show_metric_stats=False,
            score_lcb_c=1.0,
        )

        assert "Metric definitions:" in prompt
        assert "- clarity: Easy to follow and unambiguous." in prompt
        assert "- creativity: Fresh and surprising ideas." in prompt


class TestRewritePrompt:
    def test_explore_prompt_omits_parent_text(self):
        parent = make_elite("Hello world.")
        critique = Critique(summary="Needs more specificity.", routes=("Go surreal.",))
        prompt = build_rewrite_prompt(
            parent=parent,
            goal="Write a story.",
            operator_name="explore",
            role="explore",
            operator_instructions="Rewrite freely.",
            critique=critique,
            focus="Go surreal.",
            metrics=["clarity", "creativity"],
            metric_descriptions=None,
            show_metric_stats=False,
            score_lcb_c=1.0,
        )

        assert "Parent text intentionally omitted for exploration." in prompt
        assert parent.text not in prompt

    def test_exploit_prompt_includes_parent_text(self):
        parent = make_elite("Hello world.")
        critique = Critique(issues=("Tighten pacing.",))
        prompt = build_rewrite_prompt(
            parent=parent,
            goal="Write a story.",
            operator_name="exploit",
            role="exploit",
            operator_instructions="Rewrite to improve.",
            critique=critique,
            focus="Tighten pacing.",
            metrics=["clarity", "creativity"],
            metric_descriptions=None,
            show_metric_stats=False,
            score_lcb_c=1.0,
        )

        assert "Operator: exploit (exploit)" in prompt
        assert "Focus (optional):" in prompt
        assert "Tighten pacing." in prompt
        assert parent.text in prompt


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
