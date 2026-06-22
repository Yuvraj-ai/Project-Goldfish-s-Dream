"""Tests for research caching layer."""
import json
import time
from unittest.mock import patch

import pytest

from open_deep_research.research_cache import MODE_TTLS, ResearchCache


@pytest.fixture
def cache(tmp_path):
    return ResearchCache(cache_dir=str(tmp_path / "test_cache"))


class TestCacheKey:
    def test_key_generation(self):
        key = ResearchCache.make_key("search", "python tutorial", "tavily")
        assert key.startswith("search:")
        assert "tavily" in key

    def test_key_with_mode(self):
        key = ResearchCache.make_key("search", "query", mode="academic_literature_review")
        assert "academic" in key

    def test_same_query_same_key(self):
        key1 = ResearchCache.make_key("search", "test query")
        key2 = ResearchCache.make_key("search", "test query")
        assert key1 == key2

    def test_key_without_provider(self):
        key = ResearchCache.make_key("evidence", "topic", mode="default")
        assert key.startswith("evidence:")
        assert "default" not in key

    def test_key_with_mode_and_provider(self):
        key = ResearchCache.make_key("search", "q", provider="arxiv", mode="academic_literature_review")
        assert "arxiv" in key
        assert "academic" in key


class TestModeTTLs:
    def test_news_bypasses_cache(self):
        assert MODE_TTLS["news_or_current_events"] == 0

    def test_academic_long_ttl(self):
        assert MODE_TTLS["academic_literature_review"] == 168

    def test_technical_ttl(self):
        assert MODE_TTLS["technical_implementation"] == 72

    def test_default_ttl(self):
        assert MODE_TTLS["default"] == 24


class TestCacheOperations:
    def test_set_and_get(self, cache):
        cache.set("key1", {"data": "test"})
        assert cache.get("key1") == {"data": "test"}

    def test_get_nonexistent(self, cache):
        assert cache.get("nonexistent") is None

    def test_overwrite(self, cache):
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_invalidate(self, cache):
        cache.set("key1", "value1")
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None

    def test_invalidate_nonexistent(self, cache):
        assert cache.invalidate("nonexistent") is False

    def test_clear(self, cache):
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheExpiration:
    def test_expired_entry_returns_none(self, cache):
        cache.set("key1", "value1", ttl_hours=0)
        assert cache.get("key1") is None

    def test_active_entry_returns_value(self, cache):
        cache.set("key1", "value1", ttl_hours=24)
        assert cache.get("key1") == "value1"

    def test_time_based_expiry_removes_entry(self, cache):
        cache.set("key1", "value1", ttl_hours=1)
        with patch("open_deep_research.research_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 7200
            assert cache.get("key1") is None
        assert cache.get("key1") is None

    def test_age_hours_property(self, cache):
        cache.set("key1", "value1", ttl_hours=24)
        assert cache._cache["key1"].age_hours >= 0.0

    def test_is_expired_zero_ttl(self, cache):
        from open_deep_research.research_cache import CacheEntry
        entry = CacheEntry(key="k", value="v", created_at=9999999999.0, ttl_hours=0)
        assert entry.is_expired is True

    def test_is_expired_positive_ttl_not_expired(self, cache):
        from open_deep_research.research_cache import CacheEntry
        entry = CacheEntry(key="k", value="v", created_at=time.time(), ttl_hours=24)
        assert entry.is_expired is False


class TestCachePersistence:
    def test_load_from_disk(self, tmp_path):
        cache_dir = tmp_path / "persist"
        cache = ResearchCache(cache_dir=str(cache_dir))
        cache.set("key1", "value1", mode="default")
        cache2 = ResearchCache(cache_dir=str(cache_dir))
        assert cache2.get("key1") == "value1"
        assert cache2._cache["key1"].mode == "default"

    def test_load_from_disk_corrupt_json(self, tmp_path):
        cache_dir = tmp_path / "corrupt"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "cache.json").write_text("not valid json{{{")
        cache = ResearchCache(cache_dir=str(cache_dir))
        assert cache.get("any") is None

    def test_load_from_disk_missing_file(self, tmp_path):
        cache_dir = tmp_path / "empty"
        cache = ResearchCache(cache_dir=str(cache_dir))
        assert cache.stats["total_entries"] == 0

    def test_persistence_round_trip(self, tmp_path):
        cache_dir = tmp_path / "rt"
        cache = ResearchCache(cache_dir=str(cache_dir))
        cache.set_search_results("test query", [{"result": "data"}], provider="tavily")
        cache2 = ResearchCache(cache_dir=str(cache_dir))
        assert cache2.get_search_results("test query", provider="tavily") == [{"result": "data"}]


class TestSearchCache:
    def test_set_and_get_search_results(self, cache):
        results = [{"url": "https://example.com", "title": "Test"}]
        cache.set_search_results("python tutorial", results, provider="tavily")
        cached = cache.get_search_results("python tutorial", provider="tavily")
        assert cached == results

    def test_news_mode_bypasses(self, cache):
        results = [{"url": "https://news.com"}]
        cache.set_search_results("breaking news", results, mode="news_or_current_events")
        cached = cache.get_search_results("breaking news")
        assert cached is None

    def test_miss_on_different_provider(self, cache):
        results = [{"url": "https://example.com"}]
        cache.set_search_results("test", results, provider="tavily")
        cached = cache.get_search_results("test", provider="duckduckgo")
        assert cached is None

    def test_miss_on_different_mode(self, cache):
        results = [{"url": "https://example.com"}]
        cache.set_search_results("test", results, mode="technical_implementation")
        cached = cache.get_search_results("test", mode="default")
        assert cached is None


class TestEvidenceCache:
    def test_set_and_get_evidence(self, cache):
        evidence = [{"claim": "test claim", "confidence": 0.9}]
        cache.set_evidence("python performance", evidence)
        cached = cache.get_evidence("python performance")
        assert cached == evidence

    def test_set_and_get_evidence_with_mode(self, cache):
        evidence = [{"claim": "test", "confidence": 0.8}]
        cache.set_evidence("topic", evidence, mode="academic_literature_review")
        cached = cache.get_evidence("topic", mode="academic_literature_review")
        assert cached == evidence


class TestCacheStats:
    def test_stats_structure(self, cache):
        cache.set("key1", "value1")
        stats = cache.stats
        assert "total_entries" in stats
        assert stats["total_entries"] == 1

    def test_stats_with_expired_entries(self, cache):
        cache.set("key1", "value1", ttl_hours=1)
        cache.set("key2", "value2", ttl_hours=24)
        with patch("open_deep_research.research_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 7200
            stats = cache.stats
            assert stats["expired_entries"] == 1
            assert stats["active_entries"] == 1

    def test_hit_rate_on_repeated_queries(self, cache):
        queries = [f"query_{i}" for i in range(5)]
        for q in queries:
            assert cache.get_search_results(q) is None
            cache.set_search_results(q, [{"result": q}])
        hits = sum(1 for q in queries if cache.get_search_results(q) is not None)
        assert hits / len(queries) == 1.0
