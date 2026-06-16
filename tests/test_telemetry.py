"""Tests for telemetry and budget enforcement."""
import pytest

from open_deep_research.exceptions import BudgetExceededError
from open_deep_research.telemetry import (
    PRICING_TABLE,
    BudgetConfig,
    TelemetryCollector,
    TelemetryState,
    check_budget,
    get_telemetry_summary,
)


class TestPricingTable:
    """Test pricing data."""

    def test_gpt4_pricing_exists(self):
        assert "gpt-4.1" in PRICING_TABLE
        assert "input" in PRICING_TABLE["gpt-4.1"]
        assert "output" in PRICING_TABLE["gpt-4.1"]

    def test_default_pricing_exists(self):
        assert "default" in PRICING_TABLE


class TestTelemetryCollector:
    """Test telemetry collection."""

    def test_records_latency(self):
        state = TelemetryState()
        with TelemetryCollector("test_node", "gpt-4.1", state) as tc:
            pass
        assert state.nodes["test_node"].latency_ms >= 0

    def test_records_tokens(self):
        state = TelemetryState()
        with TelemetryCollector("test_node", "gpt-4.1", state) as tc:
            tc.record_tokens(input_tokens=100, output_tokens=50)
        assert state.total_input_tokens == 100
        assert state.total_output_tokens == 50

    def test_calculates_cost(self):
        state = TelemetryState()
        with TelemetryCollector("test_node", "gpt-4.1", state) as tc:
            tc.record_tokens(input_tokens=1000, output_tokens=500)
        assert state.total_cost_usd > 0

    def test_records_success(self):
        state = TelemetryState()
        with TelemetryCollector("test_node", "gpt-4.1", state) as tc:
            pass
        assert state.nodes["test_node"].success is True

    def test_records_error(self):
        state = TelemetryState()
        try:
            with TelemetryCollector("test_node", "gpt-4.1", state) as tc:
                raise ValueError("test error")
        except ValueError:
            pass
        assert state.nodes["test_node"].success is False
        assert "test error" in state.nodes["test_node"].error


class TestBudgetEnforcement:
    """Test budget checking."""

    def test_within_budget(self):
        state = TelemetryState(budget=BudgetConfig(max_total_tokens=10000))
        state.total_input_tokens = 5000
        check_budget(state)  # Should not raise

    def test_exceeds_token_budget(self):
        state = TelemetryState(budget=BudgetConfig(max_total_tokens=1000))
        state.total_input_tokens = 2000
        with pytest.raises(BudgetExceededError) as exc_info:
            check_budget(state)
        assert "Total token limit" in str(exc_info.value)

    def test_exceeds_cost_budget(self):
        state = TelemetryState(budget=BudgetConfig(max_cost_usd=0.01))
        state.total_cost_usd = 0.05
        with pytest.raises(BudgetExceededError) as exc_info:
            check_budget(state)
        assert "Cost limit" in str(exc_info.value)

    def test_disabled_budget(self):
        state = TelemetryState(budget=BudgetConfig(enabled=False, max_total_tokens=1))
        state.total_input_tokens = 10000
        check_budget(state)  # Should not raise


class TestTelemetrySummary:
    """Test telemetry summary generation."""

    def test_summary_structure(self):
        state = TelemetryState()
        summary = get_telemetry_summary(state)
        assert "total_input_tokens" in summary
        assert "total_cost_usd" in summary
        assert "nodes" in summary
