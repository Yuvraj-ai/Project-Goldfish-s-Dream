# ADR 003: 5-Dimension Golden-Set Evaluation Rubric

**Status:** Accepted

**Date:** 2025-07-15

## Context

The platform needed a structured evaluation pipeline for regression testing across research sessions. Key drivers:

- 10 golden-set queries stored in `tests/golden_set/seed_queries.json`
- Need for objective, multi-dimensional quality measurement
- Regression detection between model versions/config changes
- Balance between blocking (CI gate) and advisory modes

## Decision

Implement a 5-dimension rubric with weighted scoring:

| Dimension | Weight | Scale | What it measures |
|---|---|---|---|
| Completeness | 0.30 | 0-10 | Coverage of all subquestions, depth of analysis |
| Citation Accuracy | 0.25 | 0-10 | Correctness and relevance of citations |
| Source Diversity | 0.15 | 0-10 | Variety of sources across domains/types |
| Structure | 0.15 | 0-10 | Logical flow, section organization, readability |
| Temporal Relevance | 0.15 | 0-10 | Recency of sources relative to query context |

**Weighted average** = sum(dimension_score × weight) for each dimension.
Scale is 0-10 (not 0-1) for granular human scoring.

### Key Architecture Decisions

1. `EvalHarness` class manages golden-set loading, scoring, batch evaluation, and regression checking
2. `EvalRubricScore` Pydantic model enforces 0-10 range per dimension
3. Regression check supports two modes: `advisory` (default, logs warning) and `blocking` (return `passed=False` for CI)
4. Baseline averages stored externally (not in code) — compared at evaluation time
5. Golden set path configurable via constructor parameter

### Source Code

Implementation at `src/open_deep_research/eval_harness.py` — 131 lines, 3 classes.

## Consequences

### Positive

- **Multi-dimensional scoring** prevents gaming any single metric — improving one dimension at the expense of others is visible in the breakdown.
- **Configurable regression threshold** (default 0.3) adapts to different quality requirements across environments.
- **Golden set** provides reproducible baselines for version-to-version comparison, enabling data-driven decisions about model/config changes.
- **Advisory/blocking modes** allow teams to tune CI strictness without changing the evaluation logic.

### Trade-offs

- **Human scoring required** (no automated scoring yet) — scales with team size; manual effort per evaluation round.
- **10-query golden set** may not represent full query distribution — limited coverage risks overfitting to evaluation queries.
- **Weight selection** is subjective and may need periodic recalibration as quality priorities evolve.
