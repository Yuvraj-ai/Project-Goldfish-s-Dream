from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_save_and_load_research():
    from open_deep_research.api.memory import ResearchMemory
    repo = AsyncMock()
    memory = ResearchMemory(repo)

    topic_hash = "abc123"
    data = {"evidence": [], "plan": {}, "report": {"title": "Test"}}
    await memory.save_research(topic_hash, data)

    repo.save_memory.assert_awaited_once_with(
        "research", topic_hash, data,
    )


@pytest.mark.asyncio
async def test_load_research():
    from open_deep_research.api.memory import ResearchMemory
    repo = AsyncMock()
    repo.load_memory = AsyncMock(return_value={"report": {"title": "Test"}})
    memory = ResearchMemory(repo)

    result = await memory.load_research("abc123")
    assert result == {"report": {"title": "Test"}}


@pytest.mark.asyncio
async def test_save_and_load_preferences():
    from open_deep_research.api.memory import ResearchMemory
    repo = AsyncMock()
    memory = ResearchMemory(repo)

    prefs = {"citation_style": "apa", "report_profile": "default"}
    await memory.save_preferences("user_1", prefs)
    repo.save_memory.assert_awaited_with("preferences", "user_1", prefs)


@pytest.mark.asyncio
async def test_save_and_load_source_summary():
    from open_deep_research.api.memory import ResearchMemory
    repo = AsyncMock()
    memory = ResearchMemory(repo)

    summary = {"title": "AI Paper", "summary": "..."}
    await memory.save_source_summary("https://example.com", summary)
    repo.save_memory.assert_awaited_with("sources", "https://example.com", summary)


@pytest.mark.asyncio
async def test_find_similar_topics_with_mock():
    from open_deep_research.api.memory import ResearchMemory
    repo = AsyncMock()
    repo.list_memory_keys.return_value = []
    memory = ResearchMemory(repo)
    topics = await memory.find_similar_topics("artificial intelligence")
    assert isinstance(topics, list)
    assert len(topics) == 0


@pytest.mark.asyncio
async def test_find_similar_topics_integration():
    from open_deep_research.api.memory import ResearchMemory
    from open_deep_research.api.repository import ResearchRepository

    class MockRepo(ResearchRepository):
        def __init__(self):
            self._data = {}
        async def create_run(self, query, config, idempotency_key=None): return "id"
        async def get_run(self, run_id): return None
        async def update_run_status(self, run_id, status, error=None): pass
        async def append_progress(self, run_id, event): pass
        async def progress_after(self, run_id, seq): return []
        async def next_seq(self, run_id): return 0
        async def save_checkpoint(self, run_id, state): pass
        async def load_checkpoint(self, run_id): return None
        async def save_report(self, run_id, report): pass
        async def get_report(self, run_id): return None
        async def save_memory(self, namespace, key, data):
            self._data[(namespace, key)] = data
        async def load_memory(self, namespace, key):
            return self._data.get((namespace, key))
        async def list_memory_keys(self, namespace: str) -> list[str]:
            return [k for (ns, k) in self._data if ns == namespace]
        async def list_webhooks(self): return []
        async def save_webhook(self, config): pass
        async def delete_webhook(self, webhook_id): pass

    repo = MockRepo()
    memory = ResearchMemory(repo)

    await memory.save_research("topic_ai", {"title": "Artificial Intelligence", "summary": "AI advances and deep learning"})
    await memory.save_research("topic_ml", {"title": "Machine Learning", "summary": "ML techniques and algorithms"})
    await memory.save_research("topic_cooking", {"title": "Cooking Recipes", "summary": "Pasta and Italian cuisine"})

    results = await memory.find_similar_topics("Artificial Intelligence and deep learning")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0][0] == "topic_ai"

    results_empty = await memory.find_similar_topics("quantum physics")
    assert isinstance(results_empty, list)
    assert len(results_empty) == 0

    empty_results = await memory.find_similar_topics("anything")
    assert isinstance(empty_results, list)
