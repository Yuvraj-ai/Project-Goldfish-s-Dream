from __future__ import annotations

import pytest


def test_source_plugin_abc():
    from open_deep_research.api.plugins.base import SourcePlugin
    with pytest.raises(TypeError):
        SourcePlugin()  # type: ignore


def test_normalized_result_model():
    from open_deep_research.api.plugins.base import NormalizedResult
    result = NormalizedResult(
        url="https://example.com",
        title="Test",
        snippet="snippet",
        source_type="web",
    )
    assert result.url == "https://example.com"
    assert result.credibility_score == 0.5


@pytest.mark.asyncio
async def test_web_search_plugin():
    from open_deep_research.api.plugins.web_search import WebSearchPlugin
    plugin = WebSearchPlugin()
    results = await plugin.search("test query", max_results=2)
    assert isinstance(results, list)
    if results:
        assert results[0].url is not None


@pytest.mark.asyncio
async def test_academic_plugin():
    from open_deep_research.api.plugins.academic import AcademicPlugin
    plugin = AcademicPlugin()
    results = await plugin.search("test query", max_results=2)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_document_plugin():
    from open_deep_research.api.plugins.document import DocumentPlugin
    plugin = DocumentPlugin()
    results = await plugin.search("test query", max_results=2)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_plugin_fetch():
    from open_deep_research.api.plugins.document import DocumentPlugin
    plugin = DocumentPlugin()
    result = await plugin.fetch("https://example.com/doc")
    assert result.url == "https://example.com/doc"
    assert result.content == ""


def test_plugin_loader_from_directory(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader
    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))
    assert isinstance(plugins, list)
