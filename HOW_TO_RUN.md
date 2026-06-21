# HOW TO RUN — Advanced Deep Research

A complete guide to setting up, running, and testing the enhanced Open Deep Research framework.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Environment Setup](#3-environment-setup)
4. [Running the App](#4-running-the-app)
5. [Running Tests](#5-running-tests)
6. [Project Architecture](#6-project-architecture)
7. [Configuration Reference](#7-configuration-reference)
8. [Component Guide](#8-component-guide)
9. [Examples](#9-examples)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

- **Python 3.11+** (required)
- **uv** package manager (recommended) or pip
- **Git**
- API keys (at minimum: one LLM provider + Tavily for search)

Check your Python version:
```bash
python3 --version
```

Install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url> advanced_deep_research
cd advanced_deep_research/open_deep_research

# 2. Create virtual environment
uv venv --python 3.11 .venv

# 3. Install dependencies
uv pip install --python .venv/bin/python -e ".[dev]"

# 4. Configure API keys
cp .env.example .env
# Edit .env with your keys (see Section 3)

# 5. Verify installation
.venv/bin/python -c "from open_deep_research import deep_researcher; print('OK')"

# 6. Run tests
.venv/bin/python -m pytest tests/ -x -q

# 7. Start the app
uvx langgraph dev
```

---

## 3. Environment Setup

### Required API Keys

At minimum, you need **one LLM provider** and **Tavily** for web search.

| Key | Provider | Where to get it |
|-----|----------|-----------------|
| `TAVILY_API_KEY` | Tavily Search | [tavily.com](https://tavily.com) — free tier available |
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com) |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com](https://console.anthropic.com) |
| `GOOGLE_API_KEY` | Google Gemini | [aistudio.google.com](https://aistudio.google.com) |
| `LANGSMITH_API_KEY` | LangSmith (optional) | [smith.langchain.com](https://smith.langchain.com) |

### Configure .env

```bash
cp .env.example .env
```

Edit `.env` with your keys. Minimal config (using Google Gemini + Tavily):

```env
GOOGLE_API_KEY=your-google-key-here
TAVILY_API_KEY=your-tavily-key-here
LANGSMITH_API_KEY=your-langsmith-key-here
LANGSMITH_PROJECT=advanced-deep-research
LANGSMITH_TRACING=true
```

### Verify Environment

```bash
.venv/bin/python -c "
from dotenv import load_dotenv
import os
load_dotenv()
keys = ['TAVILY_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
for k in keys:
    v = os.getenv(k)
    print(f'{k}: {\"SET\" if v else \"NOT SET\"} ({len(v) if v else 0} chars)')
"
```

---

## 4. Running the App

### Option A: LangGraph Studio (Recommended)

```bash
cd open_deep_research
uvx langgraph dev
```

This opens a visual graph editor at `http://localhost:2024` where you can:
- See the full research graph visually (19 nodes)
- Step through nodes manually
- Inspect state at each step
- Debug research flow

### Option B: REST API Server

```bash
cd open_deep_research
ENABLE_REST_API=true API_KEY=test-key .venv/bin/python -m open_deep_research.api.main
```

```bash
# In another terminal:
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{"query": "Compare React vs Vue for enterprise apps"}'

# Stream SSE progress
curl -N http://localhost:8000/research/{run_id}/stream

# Get final report
curl http://localhost:8000/research/{run_id}/report

# Export in different formats
curl http://localhost:8000/research/{run_id}/export/markdown

# Metrics
curl localhost:8000/metrics

# Health check
curl localhost:8000/health

# OpenAPI docs
curl localhost:8000/docs
```

### Option C: Python API (Direct Invocation)

```python
import asyncio
from open_deep_research.deep_researcher import deep_researcher

async def main():
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": "What is LangGraph?"}]},
        config={
            "configurable": {
                "research_model": "google_genai:gemini-2.5-flash",
                "search_api": "tavily",
            }
        }
    )
    print(result.get("final_report", "No report generated"))

asyncio.run(main())
```

---

## 5. Running Tests

### Run All Tests

```bash
cd open_deep_research
.venv/bin/python -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# Phase 1 tests
.venv/bin/python -m pytest tests/test_exceptions.py -v      # Exception hierarchy (11 tests)
.venv/bin/python -m pytest tests/test_state.py -v           # State models & reducers (20 tests)
.venv/bin/python -m pytest tests/test_evidence.py -v        # Evidence extraction (26 tests)
.venv/bin/python -m pytest tests/test_telemetry.py -v       # Budget & telemetry (12 tests)
.venv/bin/python -m pytest tests/test_cache.py -v           # Research caching (18 tests)
.venv/bin/python -m pytest tests/test_sanitization.py -v    # Input sanitization (24 tests)
.venv/bin/python -m pytest tests/test_governor.py -v        # Concurrency governor (9 tests)
.venv/bin/python -m pytest tests/test_eval_harness.py -v    # Eval harness (8 tests)

# Phase 2 tests
.venv/bin/python -m pytest tests/test_classifier.py -v      # Strategy classifier (21 tests)
.venv/bin/python -m pytest tests/test_search_aggregator.py -v  # Search aggregator (10 tests)
.venv/bin/python -m pytest tests/test_academic_search.py -v # Academic DBs (10 tests)
.venv/bin/python -m pytest tests/test_diversity.py -v       # Source diversity (15 tests)
.venv/bin/python -m pytest tests/test_document_reader.py -v # Document reader (10 tests)
.venv/bin/python -m pytest tests/test_perspectives.py -v    # STORM perspectives (14 tests)

# Phase 3 tests
.venv/bin/python -m pytest tests/test_citation_verifier.py -v   # Citation verification (5 tests)
.venv/bin/python -m pytest tests/test_report_profiles.py -v    # Report profiles (8 tests)
.venv/bin/python -m pytest tests/test_citation.py -v           # Citation styles (12 tests)
.venv/bin/python -m pytest tests/test_exporters.py -v          # Multi-format export (8 tests)
.venv/bin/python -m pytest tests/test_reviewers.py -v          # Reviewer loop (9 tests)

# Phase 4 tests
.venv/bin/python -m pytest tests/test_repository.py -v    # Repository ABC + SQLite (15 tests)
.venv/bin/python -m pytest tests/test_runner.py -v        # Research runner (8 tests)
.venv/bin/python -m pytest tests/test_streaming.py -v     # SSE streaming (7 tests)
.venv/bin/python -m pytest tests/test_memory.py -v        # Cross-session memory (12 tests)
.venv/bin/python -m pytest tests/test_webhooks.py -v      # Webhook notification (10 tests)
.venv/bin/python -m pytest tests/test_plugins.py -v       # Plugin system (12 tests)
.venv/bin/python -m pytest tests/test_model_router.py -v  # Model router (10 tests)
.venv/bin/python -m pytest tests/test_routes.py -v        # API routes (15 tests)
```

### Run Tests by Phase

```bash
# Phase 1 only (145 tests)
.venv/bin/python -m pytest tests/test_exceptions.py tests/test_state.py tests/test_evidence.py tests/test_telemetry.py tests/test_cache.py tests/test_sanitization.py tests/test_governor.py tests/test_eval_harness.py -v

# Phase 2 only (90 tests)
.venv/bin/python -m pytest tests/test_classifier.py tests/test_search_aggregator.py tests/test_academic_search.py tests/test_diversity.py tests/test_document_reader.py tests/test_perspectives.py -v

# Phase 3 only (42 tests)
.venv/bin/python -m pytest tests/test_citation_verifier.py tests/test_report_profiles.py tests/test_citation.py tests/test_exporters.py tests/test_reviewers.py -v

# Phase 4 only (79 tests)
.venv/bin/python -m pytest tests/test_repository.py tests/test_runner.py tests/test_streaming.py tests/test_memory.py tests/test_webhooks.py tests/test_plugins.py tests/test_model_router.py tests/test_routes.py -v
```

### Run with Coverage

```bash
.venv/bin/python -m pytest tests/ --cov=src/open_deep_research --cov-report=term
```

### Lint

```bash
.venv/bin/ruff check src/open_deep_research/ --ignore=D1
```

### Full Verification Script

```bash
bash scripts/verify.sh
```

### Quick Smoke Test

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected output:
```
350 passed, X warnings in Y.Zs
```

---

## 6. Project Architecture

### Directory Structure

```
open_deep_research/
├── .env                          # API keys (gitignored)
├── .env.example                  # Template for .env
├── pyproject.toml                # Project config, deps, ruff, pytest, coverage
├── langgraph.json                # LangGraph Studio config
├── LICENSE                       # MIT License
├── scripts/
│   └── verify.sh                 # Combined lint + audit + test runner
├── docs/
│   └── ops/
│       └── runbook.md            # Operations runbook
├── src/open_deep_research/       # Main source code
│   ├── deep_researcher.py        # Main LangGraph graph (19 nodes)
│   ├── configuration.py          # All config settings
│   ├── state.py                  # State definitions & Pydantic models
│   ├── prompts.py                # System prompts
│   ├── utils.py                  # Tools, search, helpers
│   ├── evidence.py               # Evidence extraction & dedup (Phase 1)
│   ├── exceptions.py             # 6-type exception hierarchy (Phase 1)
│   ├── telemetry.py              # Cost tracking & budget (Phase 1)
│   ├── research_cache.py         # Mode-aware TTL caching (Phase 1)
│   ├── sanitization.py           # 56-pattern injection defense (Phase 1)
│   ├── governor.py               # Rate limiting & circuit breaker (Phase 1)
│   ├── eval_harness.py           # Evaluation framework (Phase 1)
│   ├── search_aggregator.py      # Multi-provider search (Phase 2)
│   ├── document_reader.py        # PDF parsing & agentic reading (Phase 2)
│   ├── perspectives.py           # STORM multi-perspective (Phase 2)
│   ├── citation_verifier.py      # HTTP citation verification (Phase 3)
│   ├── report_profiles.py        # 8 built-in report profiles (Phase 3)
│   ├── citation.py               # 6 citation styles + BibTeX (Phase 3)
│   ├── exporters.py              # Multi-format export (Phase 3)
│   ├── reviewers.py              # 4 QA reviewer agents (Phase 3)
│   ├── state_legacy.py           # Legacy state backup
│   └── api/                      # REST API server (Phase 4)
│       ├── main.py               # FastAPI app with middleware
│       ├── config.py             # API configuration
│       ├── deps.py               # Dependencies, auth, rate limiting
│       ├── exceptions.py         # API error hierarchy
│       ├── models.py             # Pydantic models
│       ├── repository.py         # Repository ABC
│       ├── repository_sqlite.py  # SQLite implementation
│       ├── runner.py             # Background graph runner
│       ├── streaming.py          # SSE progress streaming
│       ├── memory.py             # Cross-session memory service
│       ├── webhooks.py           # Webhook notification service
│       ├── model_router.py       # Multi-model routing
│       ├── metrics.py            # Prometheus metrics collector (Phase 5)
│       ├── plugins/              # Plugin system
│       │   ├── base.py           # SourcePlugin ABC
│       │   ├── loader.py         # Plugin directory scanner
│       │   ├── web_search.py     # Web search stub
│       │   ├── academic.py       # Academic search stub
│       │   └── document.py       # Document source stub
│       └── routes/               # Route handlers
│           ├── research.py        # Research CRUD + streaming
│           ├── memory.py          # Memory endpoints
│           └── admin.py           # Admin endpoints (webhooks, plugins)
├── tests/                        # 350 tests across 24 test files
│   ├── test_exceptions.py        # 11 tests
│   ├── test_state.py             # 20 tests
│   ├── test_evidence.py          # 26 tests
│   ├── test_telemetry.py         # 12 tests
│   ├── test_cache.py             # 18 tests
│   ├── test_sanitization.py      # 24 tests
│   ├── test_governor.py          # 9 tests
│   ├── test_eval_harness.py      # 8 tests
│   ├── test_classifier.py        # 21 tests (Phase 2)
│   ├── test_search_aggregator.py # 10 tests (Phase 2)
│   ├── test_academic_search.py   # 10 tests (Phase 2)
│   ├── test_diversity.py         # 15 tests (Phase 2)
│   ├── test_document_reader.py   # 10 tests (Phase 2)
│   ├── test_perspectives.py      # 14 tests (Phase 2)
│   ├── test_citation_verifier.py # 5 tests (Phase 3)
│   ├── test_report_profiles.py   # 8 tests (Phase 3)
│   ├── test_citation.py          # 12 tests (Phase 3)
│   ├── test_exporters.py         # 8 tests (Phase 3)
│   ├── test_reviewers.py         # 9 tests (Phase 3)
│   ├── test_repository.py        # 15 tests (Phase 4)
│   ├── test_runner.py            # 8 tests (Phase 4)
│   ├── test_streaming.py         # 7 tests (Phase 4)
│   ├── test_memory.py            # 12 tests (Phase 4)
│   ├── test_webhooks.py          # 10 tests (Phase 4)
│   ├── test_plugins.py           # 12 tests (Phase 4)
│   ├── test_model_router.py      # 10 tests (Phase 4)
│   ├── test_routes.py            # 15 tests (Phase 4)
│   └── golden_set/               # Evaluation seed queries
│       ├── seed_queries.json
│       └── baseline_scores.json
└── examples/                     # Example research outputs
    ├── arxiv.md
    ├── pubmed.md
    └── inference-market.md
```

### Graph Flow

```
START
  │
  ▼
clarify_with_user           ── Ask clarifying questions
  │
  ▼
parse_document              ── Parse uploaded PDFs
  │
  ▼
write_research_brief        ── Transform into research brief
  │
  ▼
classify_research_request   ── Classify research mode
  │
  ▼
generate_research_plan      ── Create plan with subquestions
  │
  ▼
optional_plan_review        ── HITL review (if enabled)
  │
  ▼
research_supervisor         ── Orchestrates parallel researchers
  │  ├── researcher_1        ── search → think → compress
  │  ├── researcher_2
  │  └── ...
  │
  ▼
extract_structured_evidence ── Build EvidenceCards (Phase 1)
  │
  ▼
compress_research           ── Dedup & compress evidence
  │
  ▼
verify_citations            ── HTTP citation checks (Phase 3)
  │
  ▼
generate_report_outline     ── Map evidence to sections
  │
  ▼
write_sections_parallel     ── Parallel section writers (Phase 3)
  │
  ▼
compile_report              ── Stitch sections + TOC + bib
  │
  ▼
final_review                ── 4 parallel reviewers (Phase 3)
  │
  ├── [score < threshold] ── rewrite_sections → compile_report (max 2x)
  │
  ▼
export_report               ── Multi-format export (Phase 3)
  │
  ▼
END
```

**Total: 19 nodes (5 Phase 1, 7 Phase 2, 7 Phase 3)**

---

## 7. Configuration Reference

All settings can be configured via environment variables, `.env` file, or the LangGraph Studio UI.

### Model Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `research_model` | `openai:gpt-4.1` | Model for conducting research |
| `compression_model` | `openai:gpt-4.1` | Model for compressing findings |
| `final_report_model` | `openai:gpt-4.1` | Model for writing final report |
| `summarization_model` | `openai:gpt-4.1-mini` | Model for summarizing web pages |

### Research Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `search_api` | `tavily` | Search provider (tavily/openai/anthropic/none) |
| `max_researcher_iterations` | `6` | Max supervisor delegation rounds |
| `max_react_tool_calls` | `10` | Max tool calls per researcher |
| `max_concurrent_research_units` | `5` | Max parallel researchers |
| `allow_clarification` | `true` | Allow clarifying questions |

### Phase 2 Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_mode_classification` | `true` | Classify queries into research modes |
| `classifier_confidence_threshold` | `0.6` | Min confidence for mode classification |
| `plan_review_mode` | `none` | Plan review: none/interrupt/auto_review |
| `enable_search_aggregation` | `false` | Multi-provider search aggregation |
| `enable_academic_search` | `true` | Enable arXiv, Semantic Scholar, PubMed, Crossref |
| `enable_document_reading` | `true` | Enable PDF upload and parsing |
| `enable_storm_research` | `false` | Enable STORM-style multi-perspective |

### Using Google Gemini

```python
config = {
    "configurable": {
        "research_model": "google_genai:gemini-2.5-flash",
        "compression_model": "google_genai:gemini-2.5-flash",
        "final_report_model": "google_genai:gemini-2.5-flash",
        "summarization_model": "google_genai:gemini-2.5-flash",
        "search_api": "tavily",
    }
}
```

---

## 8. Component Guide

### Phase 1 Components

| Component | Module | Purpose |
|-----------|--------|---------|
| **Exceptions** | `exceptions.py` | 6-type hierarchy: ToolTransientError, ToolPermanentError, ModelError, BudgetExceededError, etc. |
| **State Models** | `state.py` | Source, EvidenceCard, ConflictFlag, ResearchPlan, CitationCheck, ReviewResult |
| **Evidence Engine** | `evidence.py` | Extract, deduplicate, detect conflicts, compress evidence |
| **Budget/Telemetry** | `telemetry.py` | Track costs, enforce budgets, per-provider pricing |
| **Caching** | `research_cache.py` | Mode-aware TTLs, JSON persistence |
| **Sanitization** | `sanitization.py` | 56 injection patterns, XML escaping |
| **Governor** | `governor.py` | Rate limiting, circuit breaker per provider |
| **Eval Harness** | `eval_harness.py` | 10 golden-set queries, weighted scoring |

### Phase 2 Components

| Component | Module | Purpose |
|-----------|--------|---------|
| **Strategy Classifier** | `deep_researcher.py` | Classifies queries into 9 modes (comparison, market, academic, etc.) |
| **Research Planner** | `deep_researcher.py` | Generates detailed plans with subquestions and search strategies |
| **Search Aggregator** | `search_aggregator.py` | Multi-provider search with dedup and fallback |
| **Academic DBs** | `utils.py` | arXiv, Semantic Scholar, PubMed, Crossref wrappers |
| **Diversity** | `utils.py` | Domain histogram, temporal relevance, source diversity enforcement |
| **Document Reader** | `document_reader.py` | PDF parsing, section extraction, evidence card generation |
| **STORM Perspectives** | `perspectives.py` | Multi-perspective research with coverage matrix |

### Phase 3 Components

| Component | Module | Purpose |
|-----------|--------|---------|
| **Citation Verifier** | `citation_verifier.py` | HTTP citation checks with concurrency semaphore |
| **Report Profiles** | `report_profiles.py` | 8 built-in profiles (executive brief, deep research, academic, etc.) |
| **Section Writers** | `deep_researcher.py` | Parallel section generation with structured output |
| **Citation Styles** | `citation.py` | 6 styles (APA, MLA, Chicago, Harvard, IEEE, vanilla) + BibTeX |
| **Export Pipeline** | `exporters.py` | Markdown, HTML, PDF, DOCX, JSON, BibTeX export |
| **Reviewers** | `reviewers.py` | 4 QA reviewers (coverage, evidence, contradiction, style) |

### Phase 4 Components

| Component | Module | Purpose |
|-----------|--------|---------|
| **REST API** | `api/main.py` | FastAPI server with 15 endpoints, middleware |
| **Repository** | `api/repository.py` | ABC for persistence (SQLite default) |
| **Runner** | `api/runner.py` | Background asyncio graph execution |
| **Streaming** | `api/streaming.py` | SSE progress with Last-Event-ID replay |
| **Memory** | `api/memory.py` | Cross-session topic/preference storage |
| **Webhooks** | `api/webhooks.py` | HMAC-signed event notifications |
| **Model Router** | `api/model_router.py` | 3-tier multi-model routing |
| **Plugins** | `api/plugins/` | SourcePlugin ABC + directory loader |
| **Auth** | `api/deps.py` | API key validation + scope-based auth |

### Phase 5 Components

| Component | Module | Purpose |
|-----------|--------|---------|
| **Metrics** | `api/metrics.py` | Prometheus-compatible in-process metrics |
| **Runbook** | `docs/ops/runbook.md` | Operations runbook for failure modes |
| **Verify Script** | `scripts/verify.sh` | Combined lint + audit + test runner |

---

## 9. Examples

### Example Research Queries by Mode

**Comparison:**
```
"Compare React vs Vue vs Angular for enterprise web applications"
"AWS vs Azure vs GCP for machine learning workloads"
```

**Market Landscape:**
```
"AI agent market in 2026 — key players, trends, and opportunities"
"Global fintech landscape overview"
```

**Academic Literature Review:**
```
"Research on transformer architectures for time series forecasting"
"Recent advances in reinforcement learning from human feedback"
```

**Technical Implementation:**
```
"How to build a RAG system with LangGraph"
"Best practices for deploying LLMs in production"
```

**Fact Check:**
```
"Is quantum computing practical for cryptography in 2026?"
"Does intermittent fasting have scientific backing?"
```

---

## 10. Troubleshooting

### Common Issues

**"No tools found to conduct research"**
```bash
# Make sure TAVILY_API_KEY is set
.venv/bin/python -c "import os; print(os.getenv('TAVILY_API_KEY', 'NOT SET'))"
```

**"Unable to infer model provider"**
```bash
# Use full model prefix: google_genai:gemini-2.5-flash (not google:gemini-2.5-flash)
# Or: openai:gpt-4.1, anthropic:claude-sonnet-4-20250514
```

**Import errors after changes**
```bash
# Reinstall in editable mode
uv pip install --python .venv/bin/python -e ".[dev]"
```

**Tests failing with "no tests ran"**
```bash
# Make sure you're in the open_deep_research directory
cd open_deep_research
.venv/bin/python -m pytest tests/ -v
```

**Graph compilation errors**
```bash
# Verify the graph compiles
.venv/bin/python -c "from open_deep_research.deep_researcher import deep_researcher; print('OK')"
```

### Useful Debug Commands

```bash
# Check all imports work
.venv/bin/python -c "
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration, ResearchMode
from open_deep_research.search_aggregator import SearchAggregator
from open_deep_research.document_reader import DocumentReader
from open_deep_research.perspectives import PerspectiveGenerator
print('All imports OK')
"

# Check test count
.venv/bin/python -m pytest tests/ --co -q 2>&1 | tail -1

# Run ruff lint
.venv/bin/python -m ruff check src/open_deep_research/ --select E,F

# Check graph nodes
.venv/bin/python -c "
from open_deep_research.deep_researcher import deep_researcher
print('Nodes:', list(deep_researcher.nodes.keys()))
"
```

---

## Quick Reference

| Command | What it does |
|---------|--------------|
| `uvx langgraph dev` | Start LangGraph Studio |
| `.venv/bin/python -m pytest tests/ -x -q` | Run all tests (fast) |
| `.venv/bin/python -m pytest tests/ --cov=src/open_deep_research --cov-fail-under=60` | Run tests with coverage gate |
| `.venv/bin/ruff check src/open_deep_research/ --ignore=D1` | Lint (skip docstring warnings) |
| `bash scripts/verify.sh` | Full verification: lint + audit + tests |
| `ENABLE_REST_API=true API_KEY=test-key .venv/bin/python -m open_deep_research.api.main` | Start API server |
| `.venv/bin/python -c "from open_deep_research.deep_researcher import deep_researcher; print('OK')"` | Verify graph compiles |
