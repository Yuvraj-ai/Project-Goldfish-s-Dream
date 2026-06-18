from __future__ import annotations

from open_deep_research.api.repository import ResearchRepository


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

    async def find_similar_topics(self, query: str) -> list[str]:
        return []
