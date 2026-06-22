# ADR 001: Custom Exception Hierarchy for Open Deep Research

**Status:** Accepted

**Date:** 2025-07-15

## Context

Open Deep Research is a LangGraph-based deep research platform with a 19-node graph architecture. The original codebase used bare Python exceptions with no structured error handling, making it difficult to:

- Attribute errors to specific graph nodes for routing and recovery
- Distinguish between retryable and permanent failures
- Handle LLM-specific failure modes (token limits, model errors) separately from tool failures
- Integrate with the budget enforcement system
- Support LangGraph's checkpointing mechanism (which requires pickle-able objects)

These limitations became critical as the graph grew to 19 nodes with heterogeneous error profiles — LLM calls, tool invocations, web searches, and budget management each have distinct failure modes requiring different recovery strategies.

## Decision

We implemented a 6-type exception hierarchy rooted at `ODRError` (inherits `Exception`):

```
ODRError (base)
├── message: str
├── node: str — graph node where error occurred
├── config_snapshot: dict — state at error time
│
├── TokenLimitError — model exceeded token budget
├── ToolTransientError — retryable tool failure (has retry_after: float)
├── ToolPermanentError — permanent tool failure
├── ModelError — LLM call failure
└── BudgetExceededError — budget limit hit (has budget_type: str)
```

### Key Architectural Decisions

1. **`ODRError.__init__`** accepts `node` and `config_snapshot` for LangGraph node attribution. Every exception in the graph carries provenance — which node raised it and what configuration was active — enabling precise error routing in the graph's edge logic.

2. **Subclass constructors** add domain-specific fields via `super().__init__(message, **kwargs)`, maintaining a consistent initialization pattern while extending with specialized attributes.

3. **`ToolTransientError` includes `retry_after`** (float, default 1.0) enabling downstream retry logic to implement backoff strategies without inspecting the error message or external state.

4. **`BudgetExceededError` includes `budget_type`** (str, default "total") for per-bucket tracking (e.g., "total", "research", "api_calls"), integrating with the telemetry budget tracker.

5. **Pickle compatibility** is preserved for LangGraph checkpointing — all fields are simple types (str, dict, float) with no lambdas, closures, or unhashable dynamic types.

6. **Inheritance-only hierarchy** — `TokenLimitError`, `ToolPermanentError`, and `ModelError` use `pass` bodies, relying entirely on the base class for structure. Only exceptions that need additional fields (`ToolTransientError`, `BudgetExceededError`) override `__init__`.

## Consequences

### Positive

- **Node-attributed errors** enable precise LangGraph error routing, allowing the graph to implement node-level retry, fallback, or abort logic based on which node raised the exception.
- **Type discrimination** enables differentiated recovery strategies: transient tool errors trigger retry with backoff, permanent errors trigger abort and reporting, budget errors trigger scale-down or operator notification.
- **`BudgetExceededError`** integrates cleanly with the telemetry budget tracker, giving the graph a mechanism to halt research mid-execution when budgets are exhausted.
- **Pickle compatibility** ensures seamless LangGraph checkpointing without serialization workarounds.
- **Consistent constructor pattern** makes exception creation predictable across the codebase.

### Trade-offs

- **6 classes vs. single exception** — more code to maintain, but dramatically better error discrimination. Given the 19-node graph's complexity, the upfront class count is justified.
- **Pickle compatibility constraints** — future field types must remain simple (str, int, float, dict, list). Adding callbacks, file handles, or other non-pickle-able types would require serialization workarounds.
- **`pass` subclasses** (`TokenLimitError`, `ToolPermanentError`, `ModelError`) provide no additional functionality over `ODRError` — their value is purely in type discrimination at catch sites, which could alternatively be achieved with enum flags on a single exception class. The subclass approach was chosen for idiomatic Python and `isinstance`-based routing.

### File

Implementation at `src/open_deep_research/exceptions.py` — 62 lines, 6 exception classes.
