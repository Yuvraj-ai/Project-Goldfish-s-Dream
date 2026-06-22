# Plugin Development Guide

Extend Open Deep Research with custom source plugins for any data source — web APIs, databases, internal knowledge bases, or proprietary search engines.

---

## Plugin Architecture

Every plugin subclasses `SourcePlugin` and implements two async methods:

| Method | Signature | Returns |
|---|---|---|
| `search` | `(query: str, max_results: int = 10) -> list[NormalizedResult]` | Search results |
| `fetch` | `(url: str) -> ContentResult` | Full content for a single URL |

### Data Models

**`NormalizedResult`** — A single search result:

```python
@dataclass
class NormalizedResult:
    url: str
    title: str
    snippet: str
    source_type: str       # e.g. "web", "academic", "document", "news"
    published_date: str | None = None
    credibility_score: float = 0.5   # 0.0 - 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

**`ContentResult`** — Full page/content fetched from a URL:

```python
@dataclass
class ContentResult:
    url: str
    content: str
    content_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Base Class

```python
class SourcePlugin(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]: ...

    @abstractmethod
    async def fetch(self, url: str) -> ContentResult: ...
```

Set `name` to a unique slug — this is used to look up the plugin at runtime via `PluginLoader.get(name)`.

---

## Quick Start

Create a plugin file in 5 steps:

### 1. Create a plugin file

Place a `.py` file in your plugin directory (default: any directory listed in `configurable.plugin_directories`). Files prefixed with `_` are skipped.

```bash
mkdir -p my_plugins
touch my_plugins/hackernews.py
```

### 2. Subclass `SourcePlugin`

```python
from __future__ import annotations

import httpx

from open_deep_research.api.plugins.base import (
    ContentResult,
    NormalizedResult,
    SourcePlugin,
)


class HackerNewsPlugin(SourcePlugin):
    name = "hackernews"

    async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": query, "hitsPerPage": max_results},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for hit in data.get("hits", []):
            results.append(
                NormalizedResult(
                    url=hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    title=hit["title"],
                    snippet=hit.get("story_text", "")[:200],
                    source_type="news",
                    published_date=hit.get("created_at"),
                    credibility_score=0.6,
                    metadata={"points": hit.get("points", 0), "author": hit.get("author")},
                )
            )
        return results

    async def fetch(self, url: str) -> ContentResult:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return ContentResult(url=url, content=resp.text, content_type="html")
```

### 3. Register your plugin directory

Set `plugin_directories` in your `Configuration`:

```python
config = Configuration(
    plugin_directories=["/path/to/my_plugins"],
    # ... other config
)
```

Or via environment variable:

```bash
export PLUGIN_DIRECTORIES='["/path/to/my_plugins"]'
```

### 4. Load and use it

```python
from open_deep_research.api.plugins.loader import PluginLoader

loader = PluginLoader()
loader.load_from_directory("/path/to/my_plugins")
plugin = loader.get("hackernews")

results = await plugin.search("langchain", max_results=5)
for r in results:
    print(f"{r.title} ({r.url})")
```

### 5. Verify it works

```bash
.venv/bin/python -c "
import asyncio
from open_deep_research.api.plugins.loader import PluginLoader

loader = PluginLoader()
loader.load_from_directory('/path/to/my_plugins')
plugin = loader.get('hackernews')

async def main():
    results = await plugin.search('deep learning', max_results=3)
    for r in results:
        print(f'  [{r.source_type}] {r.title}')

asyncio.run(main())
"
```

---

## Reference Implementations

Study these files for real-world patterns:

| Plugin | File | Source |
|---|---|---|
| Web search (Tavily/DuckDuckGo) | `src/open_deep_research/api/plugins/web_search.py` | 17 lines |
| Academic (arXiv/Semantic Scholar) | `src/open_deep_research/api/plugins/academic.py` | 17 lines |
| Document loader | `src/open_deep_research/api/plugins/document.py` | 17 lines |
| Loader logic | `src/open_deep_research/api/plugins/loader.py` | 49 lines |

All reference implementations are stubs — replace the method bodies with real API calls.

---

## Registration

Plugins are auto-discovered via `PluginLoader.load_from_directory(directory)`:

1. Scans `directory` for `.py` files (skips `_`-prefixed files)
2. Imports each module
3. Finds all `SourcePlugin` subclasses (excluding the ABC itself)
4. Instantiates each and registers by `instance.name`

Set one or more plugin directories in `Configuration.plugin_directories`:

```python
configurable.plugin_directories: list[str] = []
```

The system passes these directories to `PluginLoader` during initialization.

---

## Testing Your Plugin

Follow the patterns in `tests/test_plugins.py`. Use `pytest` with `pytest-asyncio`:

```python
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_my_plugin_search():
    from my_plugins.hackernews import HackerNewsPlugin

    plugin = HackerNewsPlugin()
    results = await plugin.search("test query", max_results=3)

    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert r.url.startswith("http")
        assert r.title
        assert r.source_type == "news"


@pytest.mark.asyncio
async def test_my_plugin_fetch():
    from my_plugins.hackernews import HackerNewsPlugin

    plugin = HackerNewsPlugin()
    result = await plugin.fetch("https://example.com")
    assert result.url == "https://example.com"
    assert isinstance(result.content, str)


def test_plugin_loader_discovers_my_plugin(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    # Copy or symlink your plugin into tmp_path
    plugin_file = tmp_path / "hackernews.py"
    plugin_file.write_text((Path("my_plugins/hackernews.py").read_text()))

    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))

    names = [p.name for p in plugins]
    assert "hackernews" in names
```

Run with:

```bash
.venv/bin/python -m pytest tests/test_my_plugin.py -x -q
```

---

## Best Practices

### Error Handling

Wrap external API calls in try/except and return partial results instead of failing entirely:

```python
async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
    try:
        resp = await client.get(self.api_url, params=...)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Search API error: %s", e)
        return []   # graceful degradation
    # ...
```

### Rate Limiting

Use `asyncio.Semaphore` or `time.sleep` for polite API usage:

```python
_semaphore = asyncio.Semaphore(5)

async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
    async with _semaphore:
        # ... API call
```

For production rate limiting, also register your plugin as a consumer in the governor (`src/open_deep_research/governor.py`).

### Timeouts

Always set timeouts on HTTP clients:

```python
async with httpx.AsyncClient(timeout=30.0) as client:
    ...
```

### Async Only

Both `search` and `fetch` are `async def` — use `httpx.AsyncClient`, `aiohttp`, or `asyncio`-compatible libraries. Avoid blocking calls.

### Source Type

Set `source_type` to a recognizable string — it's used for filtering and display in the research graph. Use values like `"web"`, `"news"`, `"academic"`, `"document"`, `"social"`, `"video"`, `"code"`, etc.

### Credibility Scoring

Set `credibility_score` based on the source's trustworthiness:

- **0.9+**: Official / verified sources (government, official documentation)
- **0.7-0.9**: Established organizations, peer-reviewed journals
- **0.5-0.7**: General web, news outlets
- **0.3-0.5**: User-generated content, forums
- **<0.3**: Unverified or anonymous sources

The score should be deterministic, not LLM-estimated.

### Metadata

Use `metadata` to pass structured data the graph can use for filtering or display:

```python
NormalizedResult(
    ...
    metadata={
        "author": "Jane Doe",
        "citation_count": 42,
        "language": "en",
        "content_type": "paper",
    },
)
```

---

## Example: Custom API Plugin

Complete plugin for a hypothetical internal documentation API:

```python
from __future__ import annotations

import logging
from datetime import datetime

import httpx

from open_deep_research.api.plugins.base import (
    ContentResult,
    NormalizedResult,
    SourcePlugin,
)

logger = logging.getLogger(__name__)

API_TOKEN = "your-internal-api-token"   # use env var in production


class InternalDocsPlugin(SourcePlugin):
    name = "internal_docs"

    def __init__(self, base_url: str = "https://docs.internal.example.com/api"):
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15.0,
        )

    async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
        try:
            resp = await self._client.get(
                "/search",
                params={"q": query, "limit": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("Internal docs API error: %s", e)
            return []

        results = []
        for doc in data.get("results", []):
            results.append(
                NormalizedResult(
                    url=doc["url"],
                    title=doc["title"],
                    snippet=doc.get("excerpt", "")[:300],
                    source_type="document",
                    published_date=doc.get("updated_at"),
                    credibility_score=0.85,
                    metadata={
                        "section": doc.get("section"),
                        "product_area": doc.get("product"),
                    },
                )
            )
        return results

    async def fetch(self, url: str) -> ContentResult:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return ContentResult(
                url=url,
                content=resp.text,
                content_type="html",
            )
        except httpx.HTTPError as e:
            logger.error("Failed to fetch %s: %s", url, e)
            return ContentResult(url=url, content="", content_type="text")
```

---

## Imports

```python
from open_deep_research.api.plugins.base import SourcePlugin, NormalizedResult, ContentResult
from open_deep_research.api.plugins.loader import PluginLoader
```
