# Migration Guide: Legacy to Enhanced Open Deep Research

Migrate from the original LangChain `open_deep_research` to the enhanced evidence-first architecture. Legacy code is preserved in `src/legacy/` for reference.

---

## 1. Overview of Changes

| Dimension | Legacy | Enhanced |
|-----------|--------|----------|
| **Graph** | ~8 nodes (monolithic `graph.py` + `multi_agent.py`) | 19-node LangGraph with supervisor/researcher subgraphs |
| **State** | Flat TypedDict in single `state.py` | TypedDict top-level + Pydantic models (EvidenceCard, Source, CitationCheck, etc.) |
| **Exceptions** | Generic `Exception` | 6-type `ODRError` hierarchy (`ODRError`, `TokenLimitError`, `ToolTransientError`, `ToolPermanentError`, `ModelError`, `BudgetExceededError`) |
| **Evidence** | Flat text compression | Optional evidence-first pipeline: extract → deduplicate → detect conflicts → compress (behind `enable_evidence_first` flag) |
| **Configuration** | `@dataclass` with 10-15 fields | Pydantic `BaseModel` with 40+ fields, env var overrides, LangSmith tracing |
| **API** | None | FastAPI server with research/memory/admin routes, webhooks, streaming, plugins (behind `ENABLE_REST_API`) |
| **Testing** | Manual evaluation | 350+ tests, 60% coverage gate, golden evaluation set |
| **Prompts** | Single `prompts.py` | Modular prompts + section writers + reviewers + profiles |
| **Search** | Single provider | Aggregated multi-provider search, academic DBs, MCP tools |

### Key Architectural Differences

- **Legacy:** Two independent entry points (`graph.py` plan-execute, `multi_agent.py` supervisor-researcher) with duplicated logic
- **Enhanced:** Single unified graph with 5 phases — clarification → planning → research → writing → review. Research mode classifier selects adaptive strategy. Evidence cards flow through the entire pipeline.

---

## 2. Backward Compatibility

All legacy code is preserved at `src/legacy/` and continues to work:

```
src/legacy/
├── __init__.py
├── graph.py              # Original plan-execute workflow
├── multi_agent.py         # Original supervisor-researcher
├── state.py               # Original flat TypedDict state
├── prompts.py             # Original prompt templates
├── configuration.py       # Original @dataclass config
├── utils.py               # Original utilities
└── tests/
    ├── conftest.py
    ├── run_test.py
    └── test_report_quality.py
```

**All new feature flags default to `False`**, so a bare install behaves identically to the legacy path:

| Flag | Default | Effect When Enabled |
|------|---------|-------------------|
| `enable_evidence_first` | `False` | Enables structured EvidenceCard pipeline instead of flat text compression |
| `ENABLE_REST_API` | `False` | Starts FastAPI server alongside graph |
| `enable_search_aggregation` | `False` | Multi-provider search (Tavily + OpenAI + Anthropic) |
| `enable_storm_research` | `False` | STORM-style multi-perspective research |
| `enable_model_routing` | `False` | Cost-optimized model tier selection |

---

## 3. Migration Path

### Step 1: Install the New Version

```bash
# Create fresh venv (Python 3.11+)
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install with dev extras
uv pip install -e ".[dev]"

# Verify installation
python -c "from open_deep_research import deep_researcher; print('OK')"
```

### Step 2: Run Legacy Tests to Verify Baseline

```bash
# Original legacy tests still pass
cd src/legacy/tests
python run_test.py --agent graph
python run_test.py --agent multi_agent

# New test suite (backward-compatible path)
cd ../..
python -m pytest tests/ -x -q --cov=src/open_deep_research
```

### Step 3: Enable Feature Flags One at a Time

Start with non-breaking flags, verify at each step:

```bash
# 1. Enable section writers (uses legacy flat text approach)
export ENABLE_SECTION_WRITERS=true

# 2. Enable reviewer loop
export ENABLE_REVIEWER_LOOP=true
export MAX_REVIEW_ITERATIONS=2

# 3. Enable citation verification
export ENABLE_CITATION_VERIFICATION=true

# 4. Enable academic search
export ENABLE_ACADEMIC_SEARCH=true

# 5. Run tests after each flag
python -m pytest tests/ -x -q
```

### Step 4: Run Regression Tests Using Golden Set

The golden evaluation set at `tests/golden_set/` provides 10 benchmark queries:

```bash
# Run evaluation against golden set
python -m pytest tests/ -x -q --eval

# Compare against baseline scores (stored in golden_set/baseline.json)
python -m open_deep_research.eval_harness --compare baseline
```

### Step 5: Enable Evidence-First Mode

```bash
export ENABLE_EVIDENCE_FIRST=true

# Run full test suite
python -m pytest tests/ -x -q

# Run golden evaluation to verify quality
python -m pytest tests/ -x -q --eval
```

Evidence-first mode changes the internal pipeline:
1. `extract_structured_evidence` node extracts claims into `EvidenceCard` objects
2. `deduplicate_claims` merges duplicate claims across researchers
3. `detect_conflicts` flags contradictory claims (never averaged)
4. `compress_evidence` produces quality-weighted summaries

Confidence is computed deterministically:
```
confidence = (source_credibility × 0.4) + (corroboration_strength × 0.35) + (recency_score × 0.25)
```

### Step 6: Enable the API Server

```bash
# Start the API
ENABLE_REST_API=true API_KEY=your-secret-key python -m open_deep_research.api.main

# In another terminal, submit a research request
curl -H "Authorization: Bearer your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"topic": "Your research topic", "enable_evidence_first": true}' \
     http://localhost:8000/research
```

API routes:
- `POST /research` — Submit research request
- `GET /research/{run_id}` — Poll status
- `GET /research/{run_id}/stream` — SSE streaming
- `GET /admin/health` — Health check
- `POST /memory/set` — Persistent memory
- `GET /memory/get` — Retrieve memory

---

## 4. Configuration Changes

### Legacy → New Config Field Mapping

| Legacy Field | New Field | Notes |
|-------------|-----------|-------|
| `search_api` | `search_api` | Same name, added OpenAI/Anthropic native search |
| `number_of_queries` | `max_subquestions` | Renamed, semantics changed |
| `max_search_depth` | `max_researcher_iterations` | Renamed |
| `planner_model` | *removed* | Planning handled by `research_model` |
| `writer_model` | `final_report_model` | Renamed |
| `summarization_model` | `summarization_model` | Unchanged |
| `mcp_server_config` | `mcp_config` | Now a Pydantic model, not a dict |

### New Configuration Fields

```python
# Evidence-first
enable_evidence_first: bool = False

# Research modes
enable_mode_classification: bool = True
default_research_mode: ResearchMode = ResearchMode.CUSTOM
classifier_confidence_threshold: float = 0.6

# Search aggregation
enable_search_aggregation: bool = False
search_providers: list[str] = ["tavily"]
max_search_results: int = 20

# STORM multi-perspective
enable_storm_research: bool = False
max_perspectives: int = 7
max_storm_parallel_researchers: int = 15

# Citation verification
enable_citation_verification: bool = True

# Reviewer loop
enable_reviewer_loop: bool = True
max_review_iterations: int = 2
review_score_threshold: float = 0.7

# Budget
max_total_tokens: int = 500000
max_cost_usd: float = 5.0

# Plugin system
plugin_directories: list[str] = []

# Model routing
enable_model_routing: bool = False
```

### Environment Variable Naming

All config fields can be set via uppercase environment variables:
- `ENABLE_EVIDENCE_FIRST=true`
- `MAX_TOTAL_TOKENS=1000000`
- `SEARCH_API=tavily`
- `ENABLE_REST_API=true`
- `API_KEY=sk-...`
- `PLUGIN_DIRECTORIES=/path/to/plugins`

---

## 5. Breaking Changes

### Exception Handling

Legacy code raised generic exceptions. New code raises `ODRError` subtypes:

```python
# Legacy — catch-all
try:
    result = await graph.ainvoke(...)
except Exception as e:
    print(f"Failed: {e}")

# New — structured hierarchy
from open_deep_research.exceptions import (
    ODRError, TokenLimitError, ToolTransientError,
    ToolPermanentError, ModelError, BudgetExceededError,
)

try:
    result = await deep_researcher.ainvoke(...)
except BudgetExceededError as e:
    print(f"Budget ({e.budget_type}) exceeded")
except ToolTransientError as e:
    print(f"Retry after {e.retry_after}s")
except ODRError as e:
    print(f"Error in node {e.node}: {e}")
```

### State Access

Legacy flat dict state is replaced by typed state:

```python
# Legacy
state["sections"]  # bare dict

# New — Pydantic models stored as dicts in state
from open_deep_research.state import EvidenceCard, Source

# Access via model_validate when needed
card = EvidenceCard.model_validate(raw_card_dict)

# Sources are deduplicated automatically by the merge_sources reducer
# Evidence cards are appended by the append_evidence reducer
```

### Model Naming Convention

Legacy models used short names (`gpt-4`, `claude-sonnet`). New config uses `provider:model` format:

| Legacy | New |
|--------|-----|
| `gpt-4` | `openai:gpt-4` |
| `claude-sonnet-4-20250514` | `anthropic:claude-sonnet-4-20250514` |
| `gemini-1.5-pro` | `google:gemini-1.5-pro` |

### API Dependencies

The REST API requires FastAPI + uvicorn:

```bash
uv pip install "open_deep_research[api]"
# or: uv pip install fastapi uvicorn[standard]
```

### Removed Features

- `SearchAPI.PERPLEXITY`, `SearchAPI.EXA`, `SearchAPI.LINKUP` — removed; use Tavily, OpenAI, or Anthropic
- `process_search_results` config field — removed; replaced by evidence-first pipeline
- `include_source_str` — removed; sources are tracked via `Source` objects in state

---

## 6. Rollback Plan

### Option A: Use Legacy Entry Points

Legacy code remains importable and runnable:

```python
# Use legacy graph (unchanged behavior)
from legacy.graph import graph  # or multi_agent

result = await graph.ainvoke({
    "topic": "Your topic",
    "configurable": {"search_api": "tavily"}
})
```

### Option B: Disable All Feature Flags

Set all flags to `False` to approximate legacy behavior:

```bash
export ENABLE_EVIDENCE_FIRST=false
export ENABLE_SEARCH_AGGREGATION=false
export ENABLE_STORM_RESEARCH=false
export ENABLE_CITATION_VERIFICATION=false
export ENABLE_SECTION_WRITERS=false
export ENABLE_REVIEWER_LOOP=false
export ENABLE_ACADEMIC_SEARCH=false
export ENABLE_MODEL_ROUTING=false
export ENABLE_REST_API=false
```

### Option C: Full Reversion

1. Pin to the original `open_deep_research` package:
   ```bash
   uv pip install "open_deep_research<2.0.0"
   ```
2. Remove `src/legacy/` is already isolated — no cleanup needed
3. The new `src/open_deep_research/` can coexist with the legacy package (different namespace)

### Feature Flag Dependency Graph

```
enable_section_writers ← depends on nothing (safe to enable first)
enable_reviewer_loop   ← depends on section_writers
enable_citation_verification ← depends on nothing
enable_evidence_first  ← depends on nothing (but changes data pipeline)
enable_storm_research  ← independent flag
enable_search_aggregation ← independent flag
ENABLE_REST_API        ← independent (separate process)
```

Each flag can be rolled back independently — no cascading dependency issues.
