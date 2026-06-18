from __future__ import annotations

from open_deep_research.api.plugins.base import (
    ContentResult,
    NormalizedResult,
    SourcePlugin,
)


class AcademicPlugin(SourcePlugin):
    name = "academic"

    async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]:
        return []

    async def fetch(self, url: str) -> ContentResult:
        return ContentResult(url=url, content="")
