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
async def test_find_similar_topics():
    from open_deep_research.api.memory import ResearchMemory
    repo = AsyncMock()
    memory = ResearchMemory(repo)

    topics = await memory.find_similar_topics("artificial intelligence")
    assert isinstance(topics, list)
