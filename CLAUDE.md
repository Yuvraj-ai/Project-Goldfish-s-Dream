# Open Deep Research Repository Overview

## Project Description
Enhanced version of LangChain's `open_deep_research` — a production-grade, evidence-first deep research platform. 19-node LangGraph graph with structured EvidenceCards, citation verification, adaptive research strategies, QA reviewer loops, REST API, and multi-format export.

## Key Stats
- **350 tests** across 24 test files
- **64% test coverage** (gate: 60%)
- **19 graph nodes** across 5 phases
- **3 commits** on `dev/ph1-3` for Phase 5

## Repository Structure

### Core Implementation (`src/open_deep_research/`)
- `deep_researcher.py` - Main LangGraph graph (19 nodes, entry point: `deep_researcher`)
- `configuration.py` - Configuration management (feature flags, model/router, plugin dirs)
- `state.py` - Graph state definitions, Pydantic models (EvidenceCard, Source, etc.)
- `prompts.py` - System prompts and prompt templates
- `utils.py` - Utility functions, academic DB wrappers, helpers
- **Phase 1:** `evidence.py`, `exceptions.py`, `telemetry.py`, `research_cache.py`, `sanitization.py`, `governor.py`, `eval_harness.py`, `state_legacy.py`
- **Phase 2:** `search_aggregator.py`, `document_reader.py`, `perspectives.py`
- **Phase 3:** `citation_verifier.py`, `report_profiles.py`, `citation.py`, `exporters.py`, `reviewers.py`
- **Phase 4:** `api/` — FastAPI server, repository, runner, streaming, memory, webhooks, model router, plugins, routes
- **Phase 5:** `api/metrics.py` — Prometheus metrics collector

### Tests (`tests/`)
- 24 test files covering all 5 phases, 350 tests total
- `golden_set/` — 10 evaluation queries with baseline scores

### Other
- `scripts/verify.sh` — Combined lint + audit + test runner
- `docs/ops/runbook.md` — Operations runbook
- `src/legacy/` — Original ODR implementations (graph.py, multi_agent.py)
- `src/security/auth.py` — LangGraph deployment auth

## Development Commands
- `uvx langgraph dev` — Start LangGraph Studio
- `.venv/bin/python -m pytest tests/ -x -q` — Run tests
- `.venv/bin/python -m pytest tests/ --cov=src/open_deep_research --cov-fail-under=60` — Tests + coverage gate
- `.venv/bin/ruff check src/open_deep_research/ --ignore=D1` — Lint
- `bash scripts/verify.sh` — Full verification
- `ENABLE_REST_API=true API_KEY=test-key .venv/bin/python -m open_deep_research.api.main` — Start API server