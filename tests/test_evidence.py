"""Tests for evidence extraction engine."""
from datetime import datetime, timedelta

from open_deep_research.evidence import (
    _claims_conflict,
    _claims_similar,
    compress_evidence,
    compute_recency_score,
    compute_source_credibility,
    deduplicate_claims,
    detect_conflicts,
    extract_evidence,
)
from open_deep_research.state import EvidenceCard, Source


class TestSourceCredibility:
    """Test source credibility computation."""

    def test_academic_source_boost(self):
        source = Source(url="https://arxiv.org/abs/2401.00001", source_type="academic")
        score = compute_source_credibility(source)
        assert score > 0.6

    def test_gov_source_boost(self):
        source = Source(url="https://www.nih.gov/research", source_type="web")
        score = compute_source_credibility(source)
        assert score > 0.6

    def test_unknown_source_baseline(self):
        source = Source(url="https://random-blog.com/article", source_type="web")
        score = compute_source_credibility(source)
        assert 0.3 <= score <= 0.7

    def test_low_credibility_penalty(self):
        source = Source(url="https://random-blog.com/article", source_type="web", credibility_score=0.1)
        score = compute_source_credibility(source)
        assert score < 0.5


class TestRecencyScore:
    """Test recency score computation."""

    def test_recent_source(self):
        recent = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        score = compute_recency_score(recent, "news_or_current_events")
        assert score == 1.0

    def test_old_news_source(self):
        old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        score = compute_recency_score(old, "news_or_current_events")
        assert score < 1.0

    def test_academic_allows_older(self):
        old = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        score = compute_recency_score(old, "academic_literature_review")
        assert score == 1.0

    def test_unknown_date_neutral(self):
        score = compute_recency_score(None, "default")
        assert score == 0.5

    def test_invalid_date_neutral(self):
        score = compute_recency_score("not-a-date", "default")
        assert score == 0.5


class TestClaimsSimilar:
    """Test claim similarity detection."""

    def test_identical_claims(self):
        assert _claims_similar(
            "Python is a popular language",
            "Python is a popular language"
        ) is True

    def test_similar_claims(self):
        assert _claims_similar(
            "React is used for web development",
            "React is commonly used for web development"
        ) is True

    def test_different_claims(self):
        assert _claims_similar(
            "Python is a programming language",
            "Dogs are mammals"
        ) is False

    def test_empty_claims(self):
        assert _claims_similar("", "something") is False


class TestClaimsConflict:
    """Test claim conflict detection."""

    def test_negation_conflict(self):
        assert _claims_conflict(
            "Python is fast",
            "Python is not fast"
        ) is True

    def test_opposite_directions(self):
        assert _claims_conflict(
            "Performance increases with caching",
            "Performance decreases with caching"
        ) is True

    def test_no_conflict(self):
        assert _claims_conflict(
            "Python is popular",
            "JavaScript is popular"
        ) is False


class TestExtractEvidence:
    """Test evidence extraction from raw results."""

    def test_extracts_cards_and_sources(self):
        raw_results = [
            {"url": "https://example.com/1", "title": "Test 1", "content": "Test content 1"},
            {"url": "https://example.com/2", "title": "Test 2", "content": "Test content 2"},
        ]
        cards, sources = extract_evidence(
            raw_results, subquestion_id="q1", researcher_id="r1"
        )
        assert len(cards) == 2
        assert len(sources) == 2

    def test_confidence_computed_deterministically(self):
        raw_results = [
            {"url": "https://arxiv.org/paper", "title": "Paper", "content": "Academic content"},
        ]
        cards, _ = extract_evidence(raw_results)
        assert 0.0 <= cards[0].confidence <= 1.0

    def test_empty_results(self):
        cards, sources = extract_evidence([])
        assert len(cards) == 0
        assert len(sources) == 0


class TestDeduplicateClaims:
    """Test claim deduplication."""

    def test_removes_duplicates(self):
        cards = [
            EvidenceCard(id="1", claim="Python is popular and fast", confidence=0.8),
            EvidenceCard(id="2", claim="Python is popular and fast today", confidence=0.9),
            EvidenceCard(id="3", claim="JavaScript is popular", confidence=0.7),
        ]
        unique, dupes = deduplicate_claims(cards)
        assert len(unique) == 2
        assert len(dupes) == 1

    def test_keeps_unique_claims(self):
        cards = [
            EvidenceCard(id="1", claim="Python is fast", confidence=0.8),
            EvidenceCard(id="2", claim="Ruby is elegant", confidence=0.7),
        ]
        unique, dupes = deduplicate_claims(cards)
        assert len(unique) == 2
        assert len(dupes) == 0

    def test_empty_list(self):
        unique, dupes = deduplicate_claims([])
        assert len(unique) == 0
        assert len(dupes) == 0


class TestDetectConflicts:
    """Test conflict detection."""

    def test_detects_negation(self):
        cards = [
            EvidenceCard(id="1", claim="Python is fast for web servers"),
            EvidenceCard(id="2", claim="Python is not fast for web servers"),
        ]
        conflicts = detect_conflicts(cards)
        assert len(conflicts) == 1

    def test_no_conflicts(self):
        cards = [
            EvidenceCard(id="1", claim="Python is popular"),
            EvidenceCard(id="2", claim="JavaScript is popular"),
        ]
        conflicts = detect_conflicts(cards)
        assert len(conflicts) == 0


class TestCompressEvidence:
    """Test evidence compression."""

    def test_merges_similar_claims(self):
        cards = [
            EvidenceCard(id="1", claim="Python is popular", confidence=0.8,
                        supporting_source_ids=["url1"], exact_excerpts=["ex1"]),
            EvidenceCard(id="2", claim="Python is popular in ML", confidence=0.9,
                        supporting_source_ids=["url2"], exact_excerpts=["ex2"]),
        ]
        compressed = compress_evidence(cards)
        assert len(compressed) <= 2

    def test_empty_input(self):
        compressed = compress_evidence([])
        assert len(compressed) == 0
