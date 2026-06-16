"""Token/cost tracking, budget enforcement, and LangSmith integration."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict

from open_deep_research.exceptions import BudgetExceededError

# Pricing per 1M tokens (USD) — as of 2026
PRICING_TABLE = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
    "default": {"input": 1.00, "output": 3.00},
}


@dataclass
class NodeTelemetry:
    """Telemetry for a single node execution."""
    node_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    success: bool = True
    error: str = ""


@dataclass
class BudgetConfig:
    """Budget limits for research sessions."""
    max_total_tokens: int = 1_000_000
    max_research_tokens: int = 500_000
    max_report_tokens: int = 200_000
    max_cost_usd: float = 10.0
    enabled: bool = True


@dataclass
class TelemetryState:
    """Cumulative telemetry for a session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    nodes: Dict[str, NodeTelemetry] = field(default_factory=dict)
    budget: BudgetConfig = field(default_factory=BudgetConfig)


class TelemetryCollector:
    """Context manager for collecting telemetry per LLM call."""

    def __init__(self, node_name: str, model: str, state: TelemetryState):
        self.node_name = node_name
        self.model = model
        self.state = state
        self.start_time = 0.0
        self.telemetry = NodeTelemetry(node_name=node_name, model=model)

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        self.telemetry.latency_ms = latency_ms

        if exc_type is not None:
            self.telemetry.success = False
            self.telemetry.error = str(exc_val)

        # Calculate cost
        pricing = PRICING_TABLE.get(self.model, PRICING_TABLE["default"])
        input_cost = (self.telemetry.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.telemetry.output_tokens / 1_000_000) * pricing["output"]
        self.telemetry.cost_usd = round(input_cost + output_cost, 6)

        # Update cumulative state
        self.state.total_input_tokens += self.telemetry.input_tokens
        self.state.total_output_tokens += self.telemetry.output_tokens
        self.state.total_cost_usd += self.telemetry.cost_usd
        self.state.total_latency_ms += self.telemetry.latency_ms
        self.state.nodes[self.node_name] = self.telemetry

    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token usage for this call."""
        self.telemetry.input_tokens += input_tokens
        self.telemetry.output_tokens += output_tokens


def check_budget(state: TelemetryState, budget_type: str = "total") -> None:
    """Check if budget is exceeded, raise BudgetExceededError if so."""
    if not state.budget.enabled:
        return

    budget = state.budget
    total_tokens = state.total_input_tokens + state.total_output_tokens

    if total_tokens > budget.max_total_tokens:
        raise BudgetExceededError(
            f"Total token limit exceeded: {total_tokens}/{budget.max_total_tokens}",
            budget_type="total"
        )

    if state.total_cost_usd > budget.max_cost_usd:
        raise BudgetExceededError(
            f"Cost limit exceeded: ${state.total_cost_usd:.2f}/${budget.max_cost_usd:.2f}",
            budget_type="cost"
        )


def get_telemetry_summary(state: TelemetryState) -> Dict[str, Any]:
    """Get a summary of session telemetry."""
    return {
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
        "total_tokens": state.total_input_tokens + state.total_output_tokens,
        "total_cost_usd": round(state.total_cost_usd, 4),
        "total_latency_ms": round(state.total_latency_ms, 2),
        "node_count": len(state.nodes),
        "nodes": {
            name: {
                "tokens": t.input_tokens + t.output_tokens,
                "cost_usd": t.cost_usd,
                "latency_ms": t.latency_ms,
                "success": t.success
            }
            for name, t in state.nodes.items()
        }
    }
