"""Concurrency and rate-limit governor for API providers."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100_000
    max_concurrent: int = 10
    min_delay_seconds: float = 0.0
    backoff_base: float = 2.0
    backoff_max_seconds: float = 60.0


@dataclass
class ProviderState:
    config: RateLimitConfig
    semaphore: asyncio.Semaphore = field(init=False)
    request_times: deque = field(default_factory=deque)
    token_counts: deque = field(default_factory=deque)
    failure_count: int = 0
    circuit_open_until: float = 0.0

    def __post_init__(self):
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)


PROVIDER_CONFIGS = {
    "openai": RateLimitConfig(max_requests_per_minute=60, max_concurrent=10),
    "anthropic": RateLimitConfig(max_requests_per_minute=60, max_concurrent=10),
    "tavily": RateLimitConfig(max_requests_per_minute=30, max_concurrent=5),
    "pubmed": RateLimitConfig(max_requests_per_minute=3, max_concurrent=3, min_delay_seconds=0.4),
    "crossref": RateLimitConfig(max_requests_per_minute=20, max_concurrent=5),
    "semantic_scholar": RateLimitConfig(max_requests_per_minute=20, max_concurrent=5),
    "default": RateLimitConfig(max_requests_per_minute=30, max_concurrent=5),
}


class ConcurrencyGovernor:
    _instance: ConcurrencyGovernor | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._providers: Dict[str, ProviderState] = {}
        self._global_semaphore = asyncio.Semaphore(10)
        self._setup_providers()
        logger.info(
            "ConcurrencyGovernor initialized: global_concurrency=%d, providers=%d",
            10, len(self._providers),
        )

    def _setup_providers(self):
        for provider, config in PROVIDER_CONFIGS.items():
            self._providers[provider] = ProviderState(config=config)

    def get_provider(self, provider: str) -> ProviderState:
        if provider not in self._providers:
            logger.info(
                "Registering new provider '%s' with default rate-limit config", provider
            )
            self._providers[provider] = ProviderState(
                config=PROVIDER_CONFIGS.get("default", RateLimitConfig())
            )
        return self._providers[provider]

    def _is_circuit_open(self, state: ProviderState) -> bool:
        if state.circuit_open_until > 0:
            if time.time() < state.circuit_open_until:
                logger.debug(
                    "Circuit still open until %.1f (now=%.1f)",
                    state.circuit_open_until, time.time(),
                )
                return True
            else:
                logger.info("Circuit breaker reset; clearing failure count")
                state.circuit_open_until = 0.0
                state.failure_count = 0
        return False

    def _record_failure(self, provider: str):
        state = self.get_provider(provider)
        state.failure_count += 1
        logger.warning(
            "Recorded failure for provider '%s' (failure_count=%d)",
            provider, state.failure_count,
        )
        if state.failure_count >= 5:
            state.circuit_open_until = time.time() + 300
            logger.critical(
                "Circuit breaker OPEN for provider '%s' after %d failures; "
                "blocking requests for 300s",
                provider, state.failure_count,
            )

    def _record_success(self, provider: str):
        state = self.get_provider(provider)
        if state.failure_count:
            logger.debug(
                "Success for provider '%s'; resetting failure_count from %d",
                provider, state.failure_count,
            )
        state.failure_count = 0

    def _cleanup_old_requests(self, state: ProviderState):
        cutoff = time.time() - 60
        while state.request_times and state.request_times[0] < cutoff:
            state.request_times.popleft()
        while state.token_counts and state.token_counts[0][0] < cutoff:
            state.token_counts.popleft()

    def _check_rate_limit(self, state: ProviderState) -> bool:
        self._cleanup_old_requests(state)
        if len(state.request_times) >= state.config.max_requests_per_minute:
            logger.warning(
                "Request rate limit reached: %d/%d requests in window",
                len(state.request_times), state.config.max_requests_per_minute,
            )
            return False
        total_tokens = sum(count for _, count in state.token_counts)
        if total_tokens >= state.config.max_tokens_per_minute:
            logger.warning(
                "Token rate limit reached: %d/%d tokens in window",
                total_tokens, state.config.max_tokens_per_minute,
            )
            return False
        return True

    def _wait_time(self, state: ProviderState) -> float:
        self._cleanup_old_requests(state)
        if not state.request_times:
            return state.config.min_delay_seconds
        oldest = state.request_times[0]
        time_since_oldest = time.time() - oldest
        if len(state.request_times) >= state.config.max_requests_per_minute:
            return max(0, 60 - time_since_oldest)
        return state.config.min_delay_seconds

    async def acquire(self, provider: str, cost_estimate: int = 0):
        state = self.get_provider(provider)
        logger.debug(
            "acquire() requested for provider '%s' (cost_estimate=%d, slots_available=%d)",
            provider, cost_estimate, state.semaphore._value,
        )
        if self._is_circuit_open(state):
            logger.critical(
                "acquire() blocked: circuit breaker open for provider '%s'", provider
            )
            raise Exception(f"Circuit breaker open for {provider}")
        wait_time = self._wait_time(state)
        if wait_time > 0:
            logger.warning(
                "Rate-limit wait for provider '%s': sleeping %.2fs before acquire",
                provider, wait_time,
            )
            await asyncio.sleep(wait_time)
        await state.semaphore.acquire()
        await self._global_semaphore.acquire()
        state.request_times.append(time.time())
        if cost_estimate > 0:
            state.token_counts.append((time.time(), cost_estimate))
        logger.debug(
            "acquire() granted for provider '%s' (provider_slots_left=%d, "
            "global_slots_left=%d, requests_in_window=%d)",
            provider, state.semaphore._value, self._global_semaphore._value,
            len(state.request_times),
        )

    def release(self, provider: str, success: bool = True):
        state = self.get_provider(provider)
        state.semaphore.release()
        self._global_semaphore.release()
        logger.debug(
            "release() for provider '%s' (success=%s, provider_slots_left=%d, "
            "global_slots_left=%d)",
            provider, success, state.semaphore._value, self._global_semaphore._value,
        )
        if success:
            self._record_success(provider)
        else:
            self._record_failure(provider)

    def get_stats(self) -> Dict:
        logger.debug("Collecting governor stats for %d providers", len(self._providers))
        stats = {}
        for provider, state in self._providers.items():
            self._cleanup_old_requests(state)
            stats[provider] = {
                "requests_in_window": len(state.request_times),
                "tokens_in_window": sum(count for _, count in state.token_counts),
                "failure_count": state.failure_count,
                "circuit_open": self._is_circuit_open(state),
                "concurrent_available": state.semaphore._value,
            }
        return stats

    @classmethod
    def reset(cls):
        cls._instance = None
