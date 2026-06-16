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
