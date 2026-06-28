"""Tests for concurrency and rate-limit governor."""
import time

import pytest

from open_deep_research.governor import (
    PROVIDER_CONFIGS,
    ConcurrencyGovernor,
    ProviderState,
    RateLimitConfig,
)


@pytest.fixture(autouse=True)
def reset_governor():
    ConcurrencyGovernor.reset()
    yield
    ConcurrencyGovernor.reset()


class TestRateLimitConfig:
    def test_default_config(self):
        config = RateLimitConfig()
        assert config.max_requests_per_minute == 60
        assert config.max_concurrent == 10

    def test_pubmed_config(self):
        config = PROVIDER_CONFIGS["pubmed"]
        assert config.max_requests_per_minute == 3
        assert config.min_delay_seconds == 0.4


class TestProviderState:
    def test_initial_state(self):
        config = RateLimitConfig(max_concurrent=5)
        state = ProviderState(config=config)
        assert state.failure_count == 0
        assert state.circuit_open_until == 0.0
        assert state.semaphore._value == 5


class TestCircuitBreaker:
    def test_circuit_opens_after_failures(self):
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        for _ in range(5):
            governor._record_failure("test")
        assert governor._is_circuit_open(state) is True

    def test_circuit_resets_after_timeout(self):
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.circuit_open_until = 0.0
        state.failure_count = 0
        assert governor._is_circuit_open(state) is False

    def test_success_resets_failure_count(self):
        governor = ConcurrencyGovernor()
        governor._record_failure("test")
        governor._record_failure("test")
        governor._record_success("test")
        state = governor.get_provider("test")
        assert state.failure_count == 0


class TestRateLimiting:
    def test_within_limits(self):
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        assert governor._check_rate_limit(state) is True

    def test_exceeds_request_limit(self):
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.max_requests_per_minute = 2
        state.request_times.append(time.time())
        state.request_times.append(time.time())
        assert governor._check_rate_limit(state) is False


class TestGovernorStats:
    def test_stats_structure(self):
        governor = ConcurrencyGovernor()
        stats = governor.get_stats()
        assert "openai" in stats
        assert "requests_in_window" in stats["openai"]


class TestCircuitBreakerAdvanced:
    """Edge cases: timeout expiry, open_until, circuit re-opening."""

    def test_circuit_breaker_timeout_expiry_resets_state(self):
        """After recovery timeout expires, _is_circuit_open resets state."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.failure_count = 7
        state.circuit_open_until = time.time() - 0.001
        assert governor._is_circuit_open(state) is False
        assert state.circuit_open_until == 0.0
        assert state.failure_count == 0

    def test_circuit_breaker_sets_open_until_after_threshold(self):
        """After 5 failures, circuit_open_until is set to a future time."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        for _ in range(5):
            governor._record_failure("test")
        assert state.failure_count >= 5
        assert state.circuit_open_until > time.time()

    def test_record_failure_after_timeout_reopens_circuit(self):
        """After timeout resets state, new failures re-open circuit."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.failure_count = 5
        state.circuit_open_until = time.time() - 0.001
        governor._is_circuit_open(state)
        assert state.circuit_open_until == 0.0
        assert state.failure_count == 0
        for _ in range(5):
            governor._record_failure("test")
        assert governor._is_circuit_open(state) is True

    def test_record_success_in_closed_resets_count(self):
        """record_success in closed state resets failure count to 0."""
        governor = ConcurrencyGovernor()
        governor._record_failure("test")
        governor._record_failure("test")
        assert governor.get_provider("test").failure_count == 2
        governor._record_success("test")
        assert governor.get_provider("test").failure_count == 0


class TestCircuitBreakerAsync:
    """Async circuit breaker tests."""

    @pytest.mark.asyncio
    async def test_acquire_raises_when_circuit_open(self):
        """acquire() raises Exception when circuit is open."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.failure_count = 5
        state.circuit_open_until = time.time() + 300
        with pytest.raises(Exception, match="Circuit breaker open for test"):
            await governor.acquire("test")


class TestWaitTime:
    """RateLimiter wait_time calculations."""

    def test_wait_time_returns_min_delay_when_no_requests(self):
        """Without any requests, wait_time returns min_delay."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.min_delay_seconds = 0.5
        assert governor._wait_time(state) == 0.5

    def test_wait_time_returns_min_delay_when_under_limit(self):
        """With requests but under rate limit, wait_time returns min_delay."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.min_delay_seconds = 0.3
        state.config.max_requests_per_minute = 5
        state.request_times.append(time.time())
        assert governor._wait_time(state) == 0.3

    def test_wait_time_returns_positive_when_window_full(self):
        """When request window is full, wait_time returns > 0."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.max_requests_per_minute = 2
        now = time.time()
        state.request_times.append(now - 1)
        state.request_times.append(now)
        wait = governor._wait_time(state)
        assert wait > 0


class TestRequestCleanup:
    """Request/time-window cleanup."""

    def test_cleanup_removes_expired_requests(self):
        """Requests older than 60s are removed from the sliding window."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.request_times.append(time.time() - 120)
        state.request_times.append(time.time())
        governor._cleanup_old_requests(state)
        assert len(state.request_times) == 1

    def test_cleanup_removes_expired_token_counts(self):
        """Token counts older than 60s are removed from the sliding window."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.token_counts.append((time.time() - 120, 100))
        state.token_counts.append((time.time(), 200))
        governor._cleanup_old_requests(state)
        assert len(state.token_counts) == 1

    def test_cleanup_empty_state_does_not_error(self):
        """Cleanup on an empty state does not raise."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        governor._cleanup_old_requests(state)


class TestTokenRateLimit:
    """Token-based rate limiting."""

    def test_token_limit_exceeded(self):
        """When token window is full, _check_rate_limit returns False."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.max_tokens_per_minute = 100
        state.token_counts.append((time.time(), 100))
        assert governor._check_rate_limit(state) is False

    def test_token_limit_within_boundary(self):
        """When token window has capacity, _check_rate_limit returns True."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.max_tokens_per_minute = 100
        state.token_counts.append((time.time(), 50))
        assert governor._check_rate_limit(state) is True

    def test_token_limit_zero_estimate(self):
        """Zero-cost token estimates do not affect the rate limit."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        state.config.max_tokens_per_minute = 100
        state.token_counts.append((time.time(), 0))
        assert governor._check_rate_limit(state) is True


class TestAcquireRelease:
    """Async acquire/release pattern."""

    @pytest.mark.asyncio
    async def test_acquire_appends_request_time(self):
        """acquire() records the request timestamp."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        await governor.acquire("test")
        assert len(state.request_times) == 1
        governor.release("test")

    @pytest.mark.asyncio
    async def test_acquire_with_cost_estimate(self):
        """acquire() with a cost estimate records token count."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        await governor.acquire("test", cost_estimate=500)
        assert len(state.token_counts) == 1
        assert state.token_counts[0][1] == 500
        governor.release("test")

    @pytest.mark.asyncio
    async def test_acquire_without_cost_estimate(self):
        """acquire() without a cost estimate does not record tokens."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("test")
        await governor.acquire("test", cost_estimate=0)
        assert len(state.token_counts) == 0
        governor.release("test")

    @pytest.mark.asyncio
    async def test_release_success_resets_failures(self):
        """release() with success=True resets the failure count."""
        governor = ConcurrencyGovernor()
        governor._record_failure("test")
        governor._record_failure("test")
        await governor.acquire("test")
        governor.release("test", success=True)
        state = governor.get_provider("test")
        assert state.failure_count == 0

    @pytest.mark.asyncio
    async def test_release_failure_increments_count(self):
        """release() with success=False increments the failure count."""
        governor = ConcurrencyGovernor()
        await governor.acquire("test")
        governor.release("test", success=False)
        state = governor.get_provider("test")
        assert state.failure_count == 1


class TestProviderManagement:
    """Provider creation and caching."""

    def test_get_provider_creates_new_for_unknown(self):
        """Unknown provider gets the default config."""
        governor = ConcurrencyGovernor()
        state = governor.get_provider("non_existent")
        assert state.config is PROVIDER_CONFIGS["default"]
        assert state.config.max_concurrent == 5

    def test_get_provider_returns_cached_instance(self):
        """Same provider name returns the cached ProviderState."""
        governor = ConcurrencyGovernor()
        state1 = governor.get_provider("openai")
        state2 = governor.get_provider("openai")
        assert state1 is state2

    def test_get_provider_unknown_different_returns_different(self):
        """Different unknown provider names return different states."""
        governor = ConcurrencyGovernor()
        state1 = governor.get_provider("unknown_a")
        state2 = governor.get_provider("unknown_b")
        assert state1 is not state2

    def test_singleton_already_initialized(self):
        """Second ConcurrencyGovernor() call returns cached instance (hits early-return)."""
        governor = ConcurrencyGovernor()
        governor2 = ConcurrencyGovernor()
        assert governor is governor2
