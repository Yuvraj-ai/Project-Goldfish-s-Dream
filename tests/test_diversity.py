"""Tests for source diversity and temporal relevance."""
from open_deep_research.utils import (
    _parse_date,
    enforce_source_diversity,
    get_domain,
    get_domain_histogram,
    temporal_relevance_boost,
)


class TestParseDate:
    def test_iso_format(self):
        from datetime import datetime
        assert _parse_date("2026-06-15") == datetime(2026, 6, 15)

    def test_iso_with_time(self):
        from datetime import datetime
        assert _parse_date("2026-06-15T12:00:00") == datetime(2026, 6, 15, 12, 0, 0)

    def test_crossref_date_parts(self):
        from datetime import datetime
        assert _parse_date([2024, 6, 15]) == datetime(2024, 6, 15)

    def test_crossref_date_parts_two(self):
        from datetime import datetime
        assert _parse_date([2024, 6]) == datetime(2024, 6, 1)

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_none(self):
        assert _parse_date(None) is None

    def test_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_empty_list(self):
        assert _parse_date([]) is None


class TestGetDomain:
    def test_extracts_domain(self):
        assert get_domain("https://example.com/path") == "example.com"

    def test_strips_www(self):
        assert get_domain("https://www.example.com/path") == "example.com"

    def test_empty_url(self):
        assert get_domain("") == ""


class TestGetDomainHistogram:
    def test_counts_domains(self):
        results = [
            {"url": "https://a.com/1"},
            {"url": "https://a.com/2"},
            {"url": "https://b.com/1"},
        ]
        hist = get_domain_histogram(results)
        assert hist["a.com"] == 2
        assert hist["b.com"] == 1

    def test_empty_results(self):
        assert get_domain_histogram([]) == {}


class TestEnforceSourceDiversity:
    def test_diverse_sources_pass(self):
        results = [{"url": f"https://{d}.com/1"} for d in ["a", "b", "c", "d"]]
        report = enforce_source_diversity(results, min_unique_domains=3, max_same_domain_ratio=0.4)
        assert report["passed"] is True

    def test_concentrated_sources_fail(self):
        results = [
            {"url": "https://a.com/1"},
            {"url": "https://a.com/2"},
            {"url": "https://a.com/3"},
            {"url": "https://b.com/1"},
        ]
        report = enforce_source_diversity(results, min_unique_domains=3, max_same_domain_ratio=0.4)
        assert report["passed"] is False
        assert report["dominant_ratio"] == 0.75

    def test_empty_results_pass(self):
        report = enforce_source_diversity([])
        assert report["passed"] is True

    def test_needs_supplement(self):
        results = [{"url": "https://a.com/1"}, {"url": "https://a.com/2"}]
        report = enforce_source_diversity(results, min_unique_domains=3)
        assert report["needs_supplement"] is True


class TestTemporalRelevanceBoost:
    def test_recent_sources_score_higher(self):
        results = [
            {"url": "https://old.com", "date": "2025-01-01"},
            {"url": "https://new.com", "date": "2026-06-15"},
        ]
        boosted = temporal_relevance_boost(results)
        new_score = next(r["recency_score"] for r in boosted if "new.com" in r["url"])
        old_score = next(r["recency_score"] for r in boosted if "old.com" in r["url"])
        assert new_score > old_score

    def test_unknown_date_gets_default(self):
        results = [{"url": "https://example.com"}]
        boosted = temporal_relevance_boost(results)
        assert boosted[0]["recency_score"] == 0.5

    def test_crossref_date_parts_handled(self):
        results = [{"url": "https://example.com", "publication_date": [2024, 6, 15]}]
        boosted = temporal_relevance_boost(results)
        assert boosted[0]["recency_score"] != 0.5

    def test_sorted_by_recency(self):
        results = [
            {"url": "https://old.com", "date": "2020-01-01"},
            {"url": "https://new.com", "date": "2026-06-15"},
        ]
        boosted = temporal_relevance_boost(results)
        assert boosted[0]["url"] == "https://new.com"
