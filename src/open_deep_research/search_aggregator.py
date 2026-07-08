"""Multi-provider search aggregation with deduplication and fallback."""
import asyncio
import logging
import re
from typing import Callable, Dict, List, Literal

from pydantic import BaseModel, Field

from open_deep_research.api.plugins.loader import PluginLoader

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

    def __init__(self, providers: List[SearchProviderConfig], search_functions: Dict[str, Callable], plugin_loader: PluginLoader | None = None):
        self.providers = sorted(providers, key=lambda p: p.priority)
        self.search_functions = search_functions
        self.plugin_loader = plugin_loader

    async def search(self, query: str, source_types: List[str] | None = None, max_results: int = 20) -> List[SearchResult]:
        """Fire parallel searches across enabled providers, deduplicate by URL."""
        logger.info(
            "search: query=%.80r source_types=%s max_results=%d",
            query, source_types, max_results,
        )
        enabled_providers = [p for p in self.providers if p.enabled]
        logger.debug(
            "search: %d/%d providers enabled", len(enabled_providers), len(self.providers),
        )

        matching_providers = []
        for provider in enabled_providers:
            if provider.query_pattern:
                if re.search(provider.query_pattern, query, re.IGNORECASE):
                    matching_providers.append(provider)
                    logger.debug("provider %s matched query_pattern", provider.name)
            else:
                matching_providers.append(provider)

        if not matching_providers:
            logger.debug(
                "search: no providers matched query_pattern, falling back to all %d enabled",
                len(enabled_providers),
            )
            matching_providers = enabled_providers

        tasks = []
        for provider in matching_providers:
            if provider.name in self.search_functions:
                tasks.append(self._search_with_fallback(provider, query))
            else:
                logger.warning(
                    "provider %s has no registered search function, skipping", provider.name,
                )

        if not tasks:
            logger.warning("search: no runnable provider tasks for query, returning no results")
            return []

        logger.info("search: dispatching %d provider tasks", len(tasks))
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for i, results in enumerate(results_nested):
            if isinstance(results, Exception):
                logger.warning(
                    "Search failed for %s: %s", matching_providers[i].name, results,
                )
                continue
            logger.debug("provider %s returned %d results", matching_providers[i].name, len(results))
            all_results.extend(results)
        logger.info(
            "search: aggregated %d results from %d providers", len(all_results), len(tasks),
        )

        if self.plugin_loader:
            plugins = self.plugin_loader.list()
            logger.info("search: querying %d search plugins", len(plugins))
            plugin_tasks = [plugin.search(query, max_results) for plugin in plugins]
            plugin_results_list = await asyncio.gather(*plugin_tasks, return_exceptions=True)
            for plugin, results in zip(plugins, plugin_results_list):
                if isinstance(results, Exception):
                    logger.warning("Plugin %s search failed: %s", plugin.name, results)
                    continue
                logger.debug("plugin %s returned %d results", plugin.name, len(results))
                for pr in results:
                    all_results.append(SearchResult(
                        title=pr.title,
                        url=pr.url,
                        snippet=pr.snippet,
                        source_type=pr.source_type,
                        provider=f"plugin:{plugin.name}",
                        date=pr.published_date,
                        relevance_score=pr.credibility_score,
                    ))

        deduplicated = self._deduplicate(all_results)
        deduplicated.sort(key=lambda r: r.relevance_score, reverse=True)
        logger.info(
            "search: returning %d results (%d after dedup, capped at %d)",
            min(len(deduplicated), max_results), len(deduplicated), max_results,
        )
        return deduplicated[:max_results]

    async def _search_with_fallback(self, provider: SearchProviderConfig, query: str) -> List[SearchResult]:
        logger.debug("provider %s: invoking search function", provider.name)
        try:
            search_fn = self.search_functions[provider.name]
            results = await search_fn(query)
            if not results:
                logger.warning("provider %s returned no results", provider.name)
            else:
                logger.debug("provider %s: search function returned %d results", provider.name, len(results))
            return results
        except Exception:
            logger.exception("Search failed for provider %s", provider.name)
            return []

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        logger.debug("_deduplicate: %d input results", len(results))
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
        logger.debug(
            "_deduplicate: %d results after dedup (%d duplicates removed)",
            len(deduplicated), len(results) - len(deduplicated),
        )
        return deduplicated
