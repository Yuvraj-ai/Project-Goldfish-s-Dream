"""Tests for source relevance filter."""

import pytest

from open_deep_research.deep_researcher import score_source_relevance


def test_relevance_high_overlap():
    query = "rate limiting in Python async"
    source = {
        "title": "Python Async Rate Limiting Guide",
        "content": "Learn how to implement rate limiting in Python async applications using asyncio and semaphores.",
    }
    score = score_source_relevance(query, source)
    assert score >= 0.5


def test_relevance_low_overlap():
    query = "rate limiting in Python async"
    source = {
        "title": "Window Installation Guide for Homes",
        "content": "How to install double-pane windows in your home for better insulation.",
    }
    score = score_source_relevance(query, source)
    assert score < 0.2


def test_relevance_empty_query():
    source = {"title": "Anything at all", "content": "Some random content."}
    score = score_source_relevance("", source)
    assert score == 1.0


def test_relevance_empty_source():
    query = "rate limiting in Python async"
    source = {"title": "", "content": ""}
    score = score_source_relevance(query, source)
    assert score == 0.0


def test_relevance_short_words_ignored():
    query = "a an the cat dog"
    source = {
        "title": "Cat Dog Training Tips",
        "content": "Training your cat and dog together at home.",
    }
    score = score_source_relevance(query, source)
    assert score >= 0.5
