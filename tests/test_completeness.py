"""Tests for completeness check — placeholder detection and removal."""

import pytest
from open_deep_research.deep_researcher import _remove_placeholder_sentences, compile_report
from open_deep_research.reviewers import CompletenessReviewer


def test_remove_source_required():
    text = "Good sentence. This needs (source required) here. Another good one."
    result = _remove_placeholder_sentences(text)
    assert "Good sentence" in result
    assert "Another good one" in result
    assert "(source required)" not in result


def test_remove_citation_needed():
    text = "A claim [citation needed] is not backed up."
    result = _remove_placeholder_sentences(text)
    assert result == ""


def test_remove_insert_source():
    text = "Some fact [insert source] should have a citation."
    result = _remove_placeholder_sentences(text)
    assert result == ""


def test_remove_todo():
    text = "Good intro. The details (TODO) add later. Good outro."
    result = _remove_placeholder_sentences(text)
    assert "Good intro" in result
    assert "Good outro" in result
    assert "(TODO)" not in result


def test_no_false_positives():
    text = "The study (Smith, 2023) found significant results. This is normal text."
    result = _remove_placeholder_sentences(text)
    assert result == text


@pytest.mark.asyncio
async def test_placeholder_in_compile_report():
    state = {
        "report_outline": {
            "profile": "deep_research_report",
            "sections": [],
            "citation_index": None,
            "subquestions": [],
        },
        "written_sections": [
            {
                "section_title": "Results",
                "content": "The model achieves high accuracy. Data (source required) for the setup. Another valid result.",
            }
        ],
        "sources": [],
    }
    from langchain_core.runnables import RunnableConfig
    result = await compile_report(state, RunnableConfig())
    final = result["final_report"]
    assert "(source required)" not in final
    assert "The model achieves high accuracy" in final
    assert "Another valid result" in final


@pytest.mark.asyncio
async def test_completeness_reviewer_clean_report():
    reviewer = CompletenessReviewer()
    report = "This is a complete report with no placeholders."
    feedback = await reviewer.review(report)
    assert feedback.score == 1.0
    assert len(feedback.issues) == 0


@pytest.mark.asyncio
async def test_completeness_reviewer_detects_placeholders():
    reviewer = CompletenessReviewer()
    report = "Good intro. Data (source required) missing. Another point."
    feedback = await reviewer.review(report)
    assert feedback.score < 1.0
    assert len(feedback.issues) > 0
    assert "(source required)" in feedback.issues[0]


@pytest.mark.asyncio
async def test_completeness_reviewer_multiple_placeholders():
    reviewer = CompletenessReviewer()
    report = "(source required) and [citation needed] and (TODO) all present."
    feedback = await reviewer.review(report)
    assert feedback.score < 0.5
    assert len(feedback.issues) >= 2
    assert feedback.score >= 0.0


@pytest.mark.asyncio
async def test_completeness_reviewer_rewrite_instructions():
    reviewer = CompletenessReviewer()
    report = "Missing (source required) here."
    feedback = await reviewer.review(report)
    assert len(feedback.rewrite_instructions) > 0
    assert "placeholder" in feedback.rewrite_instructions[0].lower()
