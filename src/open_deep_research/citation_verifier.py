"""Citation verification system — HTTP checks with concurrency control."""

import asyncio
import logging
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

# Global semaphore to prevent hammering servers
_verify_semaphore = asyncio.Semaphore(5)


class CitationVerifier:
    """Verify citations via HTTP requests with concurrency control."""

    def __init__(self, timeout: float = 10.0):
        """Initialize citation verifier with HTTP timeout."""
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (compatible; DeepResearchBot/1.0; "
            "+https://github.com/langchain-ai/open_deep_research)"
        )
        self._shared_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._shared_client is None or self._shared_client.is_closed:
            self._shared_client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._shared_client

    async def close(self) -> None:
        if self._shared_client:
            await self._shared_client.aclose()
            self._shared_client = None

    async def _check_url(self, client: httpx.AsyncClient, url: str) -> dict:
        """Check a single URL. Try GET with stream=True (don't download body), fallback to HEAD."""
        async with _verify_semaphore:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return {"status": "alive", "status_code": 200}
                elif response.status_code in (403, 406, 429):
                    return {"status": "unverified", "status_code": response.status_code}
                elif response.status_code in (404, 410):
                    return {"status": "dead", "status_code": response.status_code}
                else:
                    return {"status": "alive", "status_code": response.status_code}
            except httpx.TimeoutException:
                try:
                    response = await client.head(url)
                    return {"status": "alive", "status_code": response.status_code}
                except Exception:
                    return {"status": "unverified", "status_code": 0}
            except Exception as e:
                return {"status": "unverified", "status_code": 0, "error": str(e)}

    async def verify_url(self, url: str) -> dict:
        """Check if URL is alive. Try GET first (sites block HEAD), fallback to HEAD."""
        client = await self._get_client()
        return await self._check_url(client, url)

    async def verify_batch(self, urls: List[str]) -> Dict[str, dict]:
        """Verify multiple URLs in parallel (bounded by semaphore)."""
        client = await self._get_client()
        tasks = [self._check_url(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            url: (
                result
                if isinstance(result, dict)
                else {"status": "unverified", "error": str(result)}
            )
            for url, result in zip(urls, results)
        }
