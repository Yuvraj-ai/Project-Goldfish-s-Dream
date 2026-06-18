from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedResult:
    url: str
    title: str
    snippet: str
    source_type: str
    published_date: str | None = None
    credibility_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentResult:
    url: str
    content: str
    content_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


class SourcePlugin(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[NormalizedResult]: ...

    @abstractmethod
    async def fetch(self, url: str) -> ContentResult: ...
