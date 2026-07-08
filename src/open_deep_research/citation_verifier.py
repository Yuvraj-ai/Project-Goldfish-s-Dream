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
            logger.debug("creating new shared httpx client (timeout=%.1f)", self.timeout)
            self._shared_client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._shared_client

    async def close(self) -> None:
        if self._shared_client:
            logger.debug("closing shared httpx client")
            await self._shared_client.aclose()
            self._shared_client = None

    async def _check_url(self, client: httpx.AsyncClient, url: str) -> dict:
        """Check a single URL. Try GET with stream=True (don't download body), fallback to HEAD."""
        async with _verify_semaphore:
            logger.info("checking url=%s", url)
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    logger.debug("url=%s alive status=200", url)
                    return {"status": "alive", "status_code": 200}
                elif response.status_code in (403, 406, 429):
                    logger.warning(
                        "url=%s unverified status=%d", url, response.status_code
                    )
                    return {"status": "unverified", "status_code": response.status_code}
                elif response.status_code in (404, 410):
                    logger.warning(
                        "url=%s dead status=%d", url, response.status_code
                    )
                    return {"status": "dead", "status_code": response.status_code}
                else:
                    logger.debug(
                        "url=%s alive status=%d", url, response.status_code
                    )
                    return {"status": "alive", "status_code": response.status_code}
            except httpx.TimeoutException:
                logger.warning("url=%s GET timed out, falling back to HEAD", url)
                try:
                    response = await client.head(url)
                    logger.debug(
                        "url=%s HEAD fallback status=%d", url, response.status_code
                    )
                    return {"status": "alive", "status_code": response.status_code}
                except Exception:
                    logger.exception("url=%s HEAD fallback failed", url)
                    return {"status": "unverified", "status_code": 0}
            except Exception as e:
                logger.exception("url=%s check failed", url)
                return {"status": "unverified", "status_code": 0, "error": str(e)}

    async def verify_url(self, url: str) -> dict:
        """Check if URL is alive. Try GET first (sites block HEAD), fallback to HEAD."""
        client = await self._get_client()
        return await self._check_url(client, url)

    async def verify_batch(self, urls: List[str]) -> Dict[str, dict]:
        """Verify multiple URLs in parallel (bounded by semaphore)."""
        logger.info("verify_batch entry urls=%d", len(urls))
        if not urls:
            logger.warning("verify_batch received empty urls")
        client = await self._get_client()
        tasks = [self._check_url(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        verified = {
            url: (
                result
                if isinstance(result, dict)
                else {"status": "unverified", "error": str(result)}
            )
            for url, result in zip(urls, results)
        }
        counts: Dict[str, int] = {}
        for outcome in verified.values():
            status = outcome.get("status", "unverified")
            counts[status] = counts.get(status, 0) + 1
        logger.info(
            "verify_batch complete: %d urls alive=%d dead=%d unverified=%d",
            len(urls), counts.get("alive", 0), counts.get("dead", 0),
            counts.get("unverified", 0)
        )
        return verified
