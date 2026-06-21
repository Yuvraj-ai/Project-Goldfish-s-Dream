# Advanced Deep Research

Enhanced version of LangChain's `open_deep_research` framework — transforming it from a "text-in, text-out" prototype into a production-grade, evidence-first deep research platform.

## What's New (Phases 1-3)

### Phase 1: Foundation
- **Evidence-first architecture** — Structured EvidenceCards with deterministic confidence scores
- **Source verification** — Credibility scoring, corroboration, and recency analysis
- **Budget controls** — Per-session token/cost limits with telemetry
- **Research caching** — Mode-aware TTL caching for search results and evidence
- **Input sanitization** — 56-pattern prompt injection defense
- **Concurrency governor** — Per-provider rate limiting with circuit breaker

### Phase 2: Intelligence
- **Adaptive research** — 9 research modes (comparison, market landscape, academic review, etc.)
- **Research planning** — Structured subquestions with search strategies
- **Multi-provider search** — Tavily, DuckDuckGo, academic DBs (arXiv, Semantic Scholar, PubMed, Crossref)
- **Source diversity** — Domain histogram analysis and temporal relevance
- **Document reading** — PDF parsing with section-by-section evidence extraction
- **STORM perspectives** — Multi-perspective research with coverage matrix

### Phase 3: Generation
- **Citation verification** — HTTP checks with concurrency control (Semaphore)
- **Report profiles** — 8 built-in profiles (executive brief, deep research, academic review, investment memo, etc.)
- **Section writers** — Parallel section generation with structured output
- **Citation styles** — 6 styles (vanilla, APA, MLA, Chicago, Harvard, IEEE) + BibTeX
- **Multi-format export** — Markdown, HTML, PDF, DOCX, JSON, BibTeX
- **Reviewer agents** — Coverage, evidence, contradiction, and style reviewers with auto-rewrite loop

## Architecture

```
START → clarify_with_user → parse_document → write_research_brief
→ classify_research_request → generate_research_plan → optional_plan_review
→ research_supervisor → verify_citations → generate_report_outline
→ write_sections_parallel → compile_report → final_review
→ [if score < threshold: rewrite_sections → compile_report (max 2x)]
→ export_report → END
```

## Quickstart

1. Clone the repository:
```bash
git clone https://github.com/Yuvraj-ai/Project-Goldfish-s-Dream.git
cd Project-Goldfish-s-Dream/open_deep_research
```

2. Set up Python 3.11 virtual environment:
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run tests:
```bash
python -m pytest tests/ -x -q
```

## Configuration

All settings configurable via `Configuration` class in `src/open_deep_research/configuration.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_citation_verification` | True | HTTP-check citations before report |
| `enable_section_writers` | True | Parallel section generation |
| `enable_reviewer_loop` | True | Auto-rewrite with QA reviewers |
| `citation_style` | vanilla | Citation format (APA, MLA, etc.) |
| `export_formats` | [markdown] | Output formats |
| `report_profile_override` | None | Force specific report profile |

## Report Profiles

| Profile | Description |
|---------|-------------|
| `executive_brief` | Concise summary with key findings |
| `deep_research_report` | Comprehensive multi-section report |
| `academic_literature_review` | Systematic literature review |
| `investment_memo` | Investment analysis with risk assessment |
| `competitive_landscape` | Competitive analysis with positioning |
| `technical_design_research` | Technical architecture research |
| `policy_memo` | Policy analysis with regulatory considerations |
| `news_brief` | Timely news summary |

## Testing

```bash
# Run all tests
python -m pytest tests/ -x -q

# Run specific test file
python -m pytest tests/test_citation_verifier.py -v
```

**Test coverage:** 349+ tests across 24 test files

## Project Structure

```
open_deep_research/
├── src/open_deep_research/
│   ├── deep_researcher.py      # Main graph (17 nodes)
│   ├── configuration.py        # All settings
│   ├── state.py                # AgentState, EvidenceCard, Source, etc.
│   ├── citation_verifier.py    # HTTP citation verification
│   ├── report_profiles.py      # 8 built-in report profiles
│   ├── citation.py             # 6 citation styles + BibTeX
│   ├── exporters.py            # Multi-format export pipeline
│   ├── reviewers.py            # 4 QA reviewer agents
│   ├── evidence.py             # Evidence extraction engine
│   ├── search_aggregator.py    # Multi-provider search
│   ├── document_reader.py      # PDF parsing
│   ├── perspectives.py         # STORM multi-perspective
│   ├── utils.py                # Academic DB wrappers
│   ├── sanitization.py         # Prompt injection defense
│   ├── governor.py             # Rate limiting + circuit breaker
│   ├── telemetry.py            # Budget tracking
│   ├── research_cache.py       # Mode-aware caching
│   ├── exceptions.py           # Exception taxonomy
│   ├── api/
│   │   ├── main.py             # FastAPI app with middleware
│   │   ├── config.py           # API configuration
│   │   ├── deps.py             # Dependencies, auth, rate limiting
│   │   ├── exceptions.py       # API error hierarchy
│   │   ├── models.py           # Pydantic models
│   │   ├── repository.py       # Repository ABC
│   │   ├── repository_sqlite.py # SQLite implementation
│   │   ├── runner.py           # Background graph runner
│   │   ├── streaming.py        # SSE progress streaming
│   │   ├── memory.py           # Cross-session memory service
│   │   ├── webhooks.py         # Webhook notification service
│   │   ├── model_router.py     # Multi-model routing
│   │   ├── metrics.py          # Prometheus metrics collector
│   │   ├── plugins/            # Plugin system (base, loader, stubs)
│   │   └── routes/             # Route handlers (research, memory, admin)
│   └── scripts/                # Utility scripts
└── tests/                      # 349+ tests
```

## API Server

Open Deep Research includes a FastAPI server for programmatic access.

### Quickstart

```bash
# Install with API dependencies
pip install open-deep-research[api]

# Start the server
python -m open_deep_research.api.main

# Or with configuration
ENABLE_REST_API=true API_KEY=sk-your-key python -m open_deep_research.api.main
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/research` | Start new research session |
| GET | `/research/{id}` | Get run status |
| GET | `/research/{id}/stream` | SSE progress stream |
| GET | `/research/{id}/report` | Get final report |
| GET | `/research/{id}/export/{fmt}` | Export report (markdown, html, pdf, docx) |
| POST | `/research/{id}/cancel` | Cancel running research |
| GET | `/memory` | List saved research topics |
| GET | `/memory/{topic_hash}` | Load saved research |
| POST | `/memory/feedback` | Save user preference |
| POST | `/webhooks` | Register webhook |
| GET | `/webhooks` | List webhooks |
| DELETE | `/webhooks/{id}` | Remove webhook |
| GET | `/plugins` | List registered plugins |
| GET | `/health` | Health check |

### Authentication

Set `API_KEY` env var or `api_keys` in config. Pass via `X-API-Key` header.

### Example

```bash
# Start a research session
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "Latest advances in transformer architectures"}'

# Stream progress (SSE)
curl -N http://localhost:8000/research/{run_id}/stream

# Get the final report
curl http://localhost:8000/research/{run_id}/report
```

## Key Design Decisions

- **Evidence-first:** Structured EvidenceCards with deterministic confidence scores (not LLM self-reported)
- **No vector-RAG:** Agentic document reading (section-by-section LLM interrogation)
- **Conflict presentation:** Never average contradictory claims — present disagreements explicitly
- **Feature flags:** All new components behind flags in Configuration (default True)
- **Backward compatible:** Legacy single-pass path preserved when flags disabled

## License

MIT
