"""Multi-provider search aggregation with deduplication and fallback."""
import asyncio
import logging
import re
from typing import Callable, Dict, List, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Normalized search result from any provider."""
    title: str
    url: str
    snippet: str
    source_type: Literal["web", "academic", "document", "api"] = "web"
    date: str | None = None
    provider: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    abstract: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    pdf_url: str | None = None
    authors: List[str] = Field(default_factory=list)
    publication_date: str | None = None


class SearchProviderConfig(BaseModel):
    """Configuration for a search provider."""
    name: str
    priority: int = 0
    query_pattern: str | None = None
    fallback_order: int = 0
    enabled: bool = True
    rate_limit_requests_per_minute: int = 60


class SearchAggregator:
    """Aggregates search results from multiple providers with deduplication."""

    def __init__(self, providers: List[SearchProviderConfig], search_functions: Dict[str, Callable], plugin_loader=None):
        self.providers = sorted(providers, key=lambda p: p.priority)
        self.search_functions = search_functions
        self.plugin_loader = plugin_loader

    async def search(self, query: str, source_types: List[str] | None = None, max_results: int = 20) -> List[SearchResult]:
        """Fire parallel searches across enabled providers, deduplicate by URL."""
        enabled_providers = [p for p in self.providers if p.enabled]

        matching_providers = []
        for provider in enabled_providers:
            if provider.query_pattern:
                if re.search(provider.query_pattern, query, re.IGNORECASE):
                    matching_providers.append(provider)
            else:
                matching_providers.append(provider)

        if not matching_providers:
            matching_providers = enabled_providers

        tasks = []
        for provider in matching_providers:
            if provider.name in self.search_functions:
                tasks.append(self._search_with_fallback(provider, query))

        if not tasks:
            return []

        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for i, results in enumerate(results_nested):
            if isinstance(results, Exception):
                logger.warning(f"Search failed for {matching_providers[i].name}: {results}")
                continue
            all_results.extend(results)

        if self.plugin_loader:
            plugins = self.plugin_loader.list()
            for plugin in plugins:
                try:
                    plugin_results = await plugin.search(query, max_results)
                    for pr in plugin_results:
                        all_results.append(SearchResult(
                            title=pr.title,
                            url=pr.url,
                            snippet=pr.snippet,
                            source_type=pr.source_type,
                            provider=f"plugin:{plugin.name}",
                            date=pr.published_date,
                            relevance_score=pr.credibility_score,
                        ))
                except Exception as e:
                    logger.warning("Plugin %s search failed: %s", plugin.name, e)

        deduplicated = self._deduplicate(all_results)
        deduplicated.sort(key=lambda r: r.relevance_score, reverse=True)
        return deduplicated[:max_results]

    async def _search_with_fallback(self, provider: SearchProviderConfig, query: str) -> List[SearchResult]:
        try:
            search_fn = self.search_functions[provider.name]
            return await search_fn(query)
        except Exception as e:
            logger.warning(f"Search failed for {provider.name}: {e}")
            return []

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        seen_urls: Dict[str, int] = {}
        deduplicated = []
        for result in results:
            url = result.url.rstrip("/")
            if url in seen_urls:
                existing_idx = seen_urls[url]
                if result.relevance_score > deduplicated[existing_idx].relevance_score:
                    deduplicated[existing_idx] = result
            else:
                seen_urls[url] = len(deduplicated)
                deduplicated.append(result)
        return deduplicated
