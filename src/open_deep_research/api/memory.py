from __future__ import annotations

import re

from open_deep_research.api.repository import ResearchRepository

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "about",
}


class ResearchMemory:
    def __init__(self, repo: ResearchRepository) -> None:
        self._repo = repo

    async def save_research(self, topic_hash: str, data: dict) -> None:
        await self._repo.save_memory("research", topic_hash, data)

    async def load_research(self, topic_hash: str) -> dict | None:
        return await self._repo.load_memory("research", topic_hash)

    async def save_preferences(self, user_id: str, prefs: dict) -> None:
        await self._repo.save_memory("preferences", user_id, prefs)

    async def load_preferences(self, user_id: str) -> dict | None:
        return await self._repo.load_memory("preferences", user_id)

    async def save_source_summary(self, url: str, summary: dict) -> None:
        await self._repo.save_memory("sources", url, summary)

    async def get_source_summary(self, url: str) -> dict | None:
        return await self._repo.load_memory("sources", url)

    async def list_research_topics(self) -> list[str]:
        return await self._repo.list_memory_keys("research")

    async def find_similar_topics(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        query_words = self._tokenize(query)
        if not query_words:
            return []

        keys = await self._repo.list_memory_keys("research")
        scored: list[tuple[str, float]] = []
        for key in keys:
            data = await self._repo.load_memory("research", key)
            if not data:
                continue
            title = data.get("title", "")
            summary = data.get("summary", "")
            text_to_score = f"{title} {summary}"
            topic_words = self._tokenize(text_to_score)
            if not topic_words:
                continue
            intersection = len(query_words & topic_words)
            union = len(query_words | topic_words)
            score = intersection / union if union > 0 else 0.0
            if score > 0.0:
                scored.append((key, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _tokenize(self, text: str) -> set[str]:
        words = set(re.findall(r"\w+", text.lower()))
        return words - _STOPWORDS
