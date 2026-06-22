"""Tests for multi-provider search aggregator."""

import pytest

from open_deep_research.search_aggregator import (
    SearchAggregator,
    SearchProviderConfig,
    SearchResult,
)


class TestSearchResult:
    def test_result_creation(self):
        result = SearchResult(title="Test", url="https://example.com", snippet="S", provider="test")
        assert result.title == "Test"

    def test_result_defaults(self):
        result = SearchResult(title="T", url="https://example.com", snippet="S", provider="test")
        assert result.source_type == "web"
        assert result.relevance_score == 0.5


class TestSearchAggregator:
    def test_deduplicate_by_url(self):
        providers = [SearchProviderConfig(name="test", priority=1)]
        aggregator = SearchAggregator(providers, {})
        results = [
            SearchResult(title="V1", url="https://example.com", snippet="A", provider="p1", relevance_score=0.3),
            SearchResult(title="V2", url="https://example.com", snippet="B", provider="p2", relevance_score=0.8),
        ]
        deduplicated = aggregator._deduplicate(results)
        assert len(deduplicated) == 1
        assert deduplicated[0].relevance_score == 0.8

    def test_deduplicate_preserves_different_urls(self):
        providers = [SearchProviderConfig(name="test", priority=1)]
        aggregator = SearchAggregator(providers, {})
        results = [
            SearchResult(title="A", url="https://a.com", snippet="S", provider="p"),
            SearchResult(title="B", url="https://b.com", snippet="S", provider="p"),
        ]
        deduplicated = aggregator._deduplicate(results)
        assert len(deduplicated) == 2

    def test_deduplicate_handles_trailing_slash(self):
        providers = [SearchProviderConfig(name="test", priority=1)]
        aggregator = SearchAggregator(providers, {})
        results = [
            SearchResult(title="A", url="https://example.com/", snippet="S", provider="p", relevance_score=0.3),
            SearchResult(title="B", url="https://example.com", snippet="S", provider="p", relevance_score=0.8),
        ]
        deduplicated = aggregator._deduplicate(results)
        assert len(deduplicated) == 1

    @pytest.mark.asyncio
    async def test_search_calls_providers(self):
        call_log = []
        async def mock_search(query):
            call_log.append("called")
            return [SearchResult(title="R", url="https://example.com", snippet="S", provider="mock")]

        providers = [SearchProviderConfig(name="mock", priority=1)]
        aggregator = SearchAggregator(providers, {"mock": mock_search})
        results = await aggregator.search("test")
        assert len(results) == 1
        assert "called" in call_log

    @pytest.mark.asyncio
    async def test_search_handles_failure(self):
        async def fail(q):
            raise Exception("boom")
        async def ok(q):
            return [SearchResult(title="R", url="https://a.com", snippet="S", provider="ok")]

        providers = [SearchProviderConfig(name="fail", priority=1), SearchProviderConfig(name="ok", priority=2)]
        aggregator = SearchAggregator(providers, {"fail": fail, "ok": ok})
        results = await aggregator.search("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_no_providers(self):
        aggregator = SearchAggregator([], {})
        results = await aggregator.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        async def many(q):
            return [SearchResult(title=f"R{i}", url=f"https://example{i}.com", snippet="S", provider="p") for i in range(10)]

        providers = [SearchProviderConfig(name="many", priority=1)]
        aggregator = SearchAggregator(providers, {"many": many})
        results = await aggregator.search("test", max_results=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_filters_by_query_pattern(self):
        call_log = []
        async def search_fn(q):
            call_log.append("called")
            return [SearchResult(title="R", url="https://example.com", snippet="S", provider="p")]

        providers = [
            SearchProviderConfig(name="filtered", priority=1, query_pattern=r"(?:academic|paper)"),
            SearchProviderConfig(name="always", priority=2),
        ]
        aggregator = SearchAggregator(providers, {"filtered": search_fn, "always": search_fn})
        await aggregator.search("general query")
        assert call_log.count("called") == 1  # only "always" matches


@pytest.mark.asyncio
async def test_search_aggregator_with_plugins():
    from open_deep_research.api.plugins.base import NormalizedResult, SourcePlugin

    class MockPlugin(SourcePlugin):
        name = "mock_plugin"

        async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
            return [NormalizedResult(
                url="https://plugin.example.com/result",
                title="Plugin Result",
                snippet="Plugin snippet",
                source_type="api",
            )]

        async def fetch(self, url: str):
            from open_deep_research.api.plugins.base import ContentResult
            return ContentResult(url=url, content="content")

    from open_deep_research.api.plugins.loader import PluginLoader
    loader = PluginLoader()
    loader.register("mock_plugin", MockPlugin())

    config = SearchProviderConfig(name="test", priority=100, enabled=True)

    async def dummy_search(q):
        return [SearchResult(title="Web", url="https://web.example.com", snippet="web", source_type="web", provider="test")]

    aggregator = SearchAggregator(providers=[config], search_functions={"test": dummy_search}, plugin_loader=loader)
    results = await aggregator.search("test query")

    urls = [r.url for r in results]
    assert "https://plugin.example.com/result" in urls
    assert "https://web.example.com" in urls


@pytest.mark.asyncio
async def test_search_aggregator_plugin_failure():
    from open_deep_research.api.plugins.base import (
        ContentResult,
        NormalizedResult,
        SourcePlugin,
    )

    class FailingPlugin(SourcePlugin):
        name = "fail_plugin"

        async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
            raise Exception("plugin search failed")

        async def fetch(self, url: str):
            return ContentResult(url=url, content="")

    from open_deep_research.api.plugins.loader import PluginLoader
    loader = PluginLoader()
    loader.register("fail_plugin", FailingPlugin())

    config = SearchProviderConfig(name="ok", priority=1, enabled=True)

    async def ok_search(q):
        return [SearchResult(title="OK", url="https://ok.example.com", snippet="s", provider="ok")]

    aggregator = SearchAggregator(providers=[config], search_functions={"ok": ok_search}, plugin_loader=loader)
    results = await aggregator.search("test query")
    assert len(results) == 1
    assert results[0].url == "https://ok.example.com"


class TestSearchAggregatorPattern:
    @pytest.mark.asyncio
    async def test_query_pattern_matches_provider(self):
        call_log = []
        async def search_fn(q):
            call_log.append(q)
            return [SearchResult(title="R", url="https://example.com", snippet="S", provider="academic")]

        providers = [
            SearchProviderConfig(name="academic", priority=1, query_pattern=r"(?:research|paper|study)"),
        ]
        aggregator = SearchAggregator(providers, {"academic": search_fn})
        await aggregator.search("research about AI")
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_no_providers_matched_falls_back_to_all(self):
        call_log = []
        async def search_fn(q):
            call_log.append(q)
            return [SearchResult(title="R", url="https://example.com", snippet="S", provider="p")]

        providers = [
            SearchProviderConfig(name="p1", priority=1, query_pattern=r"(?:specific)"),
            SearchProviderConfig(name="p2", priority=2, query_pattern=r"(?:exact)"),
        ]
        aggregator = SearchAggregator(providers, {"p1": search_fn, "p2": search_fn})
        await aggregator.search("no match for either pattern")
        assert len(call_log) == 2
