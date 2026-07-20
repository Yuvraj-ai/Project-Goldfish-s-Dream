# Task 3-1: Classifier Accuracy Verification

**Date:** 2026-06-23
**Model:** `openai/gpt-oss-120b:free` via OpenRouter
**Test Set:** `tests/golden_set/classifier_test_set.json` (51 queries across 9 modes)

## Result: PASS (86.3%)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | **86.3%** (44/51) | >85% | ✅ PASS |

## Breakdown by Mode

| Mode | Accuracy | Notes |
|------|----------|-------|
| `academic_literature_review` | 5/5 (100%) | |
| `company_due_diligence` | 5/5 (100%) | |
| `comparison` | 8/8 (100%) | |
| `market_landscape` | 5/5 (100%) | |
| `news_or_current_events` | 5/5 (100%) | |
| `policy_legal_regulatory` | 5/5 (100%) | |
| `technical_implementation` | 5/5 (100%) | |
| `validation_or_fact_check` | 5/6 (83%) | 1 structured output parse failure |
| `custom` | 1/7 (14%) | See analysis below |

## Misclassifications (7)

| Query | Expected | Got |
|-------|----------|-----|
| Is the Earth's core really as hot as the surface of the sun? | `validation_or_fact_check` | `ValidationError` |
| Explain the concept of entanglement entropy in quantum field theory | `custom` | `ValidationError` |
| How does the scoring system work in Olympic figure skating? | `custom` | `technical_implementation` |
| Explain the philosophy of Stoicism and its modern applications | `custom` | `academic_literature_review` |
| What is the best strategy for winning at chess as Black? | `custom` | `technical_implementation` |
| How do tsunamis form and what determines their destructive power? | `custom` | `academic_literature_review` |
| Design a theoretical mission to colonize Mars with current technology | `custom` | `technical_implementation` |

## Analysis

### Strengths
- All 7 structured modes (comparison, market_landscape, academic, company_due_diligence, technical_implementation, news, policy) achieved **100% accuracy** — the classifier prompt clearly defines these modes well.
- The model's structured output (`with_structured_output`) works reliably for well-defined categories.

### Weaknesses
1. **`custom` mode at 14% (1/7):** The classifier prompt defines `custom` as a fallback for queries that don't fit other modes. But the model over-classifies into other modes (especially `technical_implementation` and `academic_literature_review`) instead of recognizing unfamiliar topics and falling back. The confidence threshold mechanism exists but isn't triggering effectively.
2. **2 ValidationErrors:** The structured output schema sometimes fails — the model returns malformed JSON that Pydantic rejects. This is likely a model-specific issue with `gpt-oss-120b:free` rather than the code.
3. **`validation_or_fact_check`:** 83% is below the mode average — 1 misclassification eroded the score.

### Potential Fixes
1. **Classifier prompt tuning:** Add more explicit instructions for when to use `custom` fallback. Example: "If the query is a general knowledge question, tutorial request, or philosophical inquiry, classify as `custom`."
2. **Confidence threshold:** Lower the threshold or add a secondary check — if the classifier shows low confidence (<0.7), force `custom` fallback.
3. **Retry on ValidationError:** Add `max_structured_output_retries` — the current run had retries=0.

## Commands to Reproduce

```bash
cd open_deep_research
OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  .venv/bin/python scripts/run_classifier.py
```
