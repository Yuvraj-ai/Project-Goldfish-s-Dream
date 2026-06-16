"""Tests for reviewer agents."""

from dataclasses import dataclass

import pytest

from open_deep_research.reviewers import (
    ContradictionReviewer,
    CoverageReviewer,
    EvidenceReviewer,
    ReviewFeedback,
    StyleReviewer,
)

SAMPLE_REPORT = """
## Executive Summary

This report covers quantum computing advances.

## Findings

Quantum computing has made significant progress in 2024.
However, there are still major challenges to overcome.

## Conclusion

Quantum computing will transform industries.
"""


@pytest.mark.asyncio
async def test_coverage_reviewer_all_covered():
    """Test coverage reviewer when all subquestions are addressed."""
    reviewer = CoverageReviewer()
    plan = {"subquestions": ["quantum computing advances", "major challenges"]}
    feedback = await reviewer.review(SAMPLE_REPORT, plan)
    assert feedback.score == 1.0
    assert feedback.reviewer_name == "CoverageReviewer"


@pytest.mark.asyncio
async def test_coverage_reviewer_missing():
    """Test coverage reviewer when subquestion is missing."""
    reviewer = CoverageReviewer()
    plan = {"subquestions": ["quantum computing advances", "blockchain technology"]}
    feedback = await reviewer.review(SAMPLE_REPORT, plan)
    assert feedback.score < 1.0
    assert len(feedback.issues) > 0


@pytest.mark.asyncio
async def test_coverage_reviewer_empty_plan():
    """Test coverage reviewer with empty plan."""
    reviewer = CoverageReviewer()
    feedback = await reviewer.review(SAMPLE_REPORT, {"subquestions": []})
    assert feedback.score == 1.0


@pytest.mark.asyncio
async def test_evidence_reviewer_citations():
    """Test evidence reviewer with cited claims."""
    reviewer = EvidenceReviewer()
    report = "Quantum computing advances (Example Press, 2024). This is a longer claim that should be checked for citation."
    feedback = await reviewer.review(report, [], [])
    assert feedback.reviewer_name == "EvidenceReviewer"


@pytest.mark.asyncio
async def test_contradiction_reviewer():
    """Test contradiction reviewer."""
    reviewer = ContradictionReviewer()
    report = "However, some sources disagree with this assessment."
    feedback = await reviewer.review(report)
    assert feedback.reviewer_name == "ContradictionReviewer"
    assert feedback.score <= 1.0


@pytest.mark.asyncio
async def test_contradiction_reviewer_with_acknowledgment():
    """Test contradiction reviewer with acknowledged disagreement."""
    reviewer = ContradictionReviewer()
    report = "There is ongoing debate about this topic. However, some sources disagree with this assessment."
    feedback = await reviewer.review(report)
    assert feedback.reviewer_name == "ContradictionReviewer"


@dataclass
class MockProfile:
    max_length: int
    tone: str


@pytest.mark.asyncio
async def test_style_reviewer_formal():
    """Test style reviewer for formal tone."""
    reviewer = StyleReviewer()
    profile = MockProfile(max_length=10, tone="formal")
    report = "This is a formal report."
    feedback = await reviewer.review(report, profile)
    assert feedback.reviewer_name == "StyleReviewer"


@pytest.mark.asyncio
async def test_style_reviewer_too_long():
    """Test style reviewer detects too-long report."""
    reviewer = StyleReviewer()
    profile = MockProfile(max_length=5, tone="formal")
    report = "This is a very long report that exceeds the maximum length."
    feedback = await reviewer.review(report, profile)
    assert any("too long" in issue.lower() for issue in feedback.issues)


@pytest.mark.asyncio
async def test_review_feedback_structure():
    """Test ReviewFeedback structure."""
    feedback = ReviewFeedback(
        reviewer_name="TestReviewer",
        score=0.8,
        issues=["issue1"],
        rewrite_instructions=["fix1"],
    )
    assert feedback.score == 0.8
    assert len(feedback.issues) == 1
    assert len(feedback.rewrite_instructions) == 1
