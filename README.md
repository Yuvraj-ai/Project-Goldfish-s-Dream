# Advanced Deep Research

A production-grade deep research agent that autonomously investigates complex questions, verifies sources, and generates structured reports with citations, evidence cards, and multi-format export.

Built on LangGraph's `open_deep_research` — extended with evidence-first architecture, citation verification, adaptive research strategies, quality assurance review loops, and a REST API.

---

## Features

- **Multi-provider search** — Tavily, DuckDuckGo, arXiv, Semantic Scholar, PubMed, Crossref
- **Evidence extraction** — Structured evidence cards with deterministic confidence scoring (source credibility × corroboration × recency)
- **Citation verification** — HTTP checks verify sources are alive; dead links trigger fallback search
- **Adaptive research strategies** — 9 research modes (comparison, market landscape, academic review, etc.) with auto-classification
- **Agentic document reading** — PDF parsing with section-by-section LLM interrogation (no vector DB)
- **Multi-perspective research** — STORM-style parallel researchers across customer/investor/regulator/etc. lenses
- **8 report profiles** — Executive brief, deep research report, academic literature review, investment memo, competitive landscape, technical design, policy memo, news brief
- **6 citation styles** — APA, MLA, Chicago, Harvard, IEEE, vanilla + BibTeX export
- **Parallel section writers** — Structured section generation with conflict-aware writing
- **QA reviewer loop** — 4 parallel reviewers (coverage, evidence, contradiction, style) with auto-rewrite
- **Multi-format export** — Markdown, HTML, PDF, DOCX, JSON
- **REST API** — FastAPI server with SSE progress streaming, webhooks, cross-session memory, API key auth
- **Observability** — Prometheus metrics endpoint, LangSmith tracing, and centralized leveled logging (colored console + daily-rotating file, with JSON/`request_id` mode for aggregators)
- **Security** — Prompt injection defense (56 patterns), rate limiting, circuit breakers, input validation
- **Model routing** — 3-tier model selection (fast/balanced/quality) per task type for cost optimization

---

## Quickstart

```bash
# Prerequisites: Python 3.11+, uv

# 1. Clone and enter the directory
git clone <repo-url>
cd open_deep_research

# 2. Create virtual environment and install
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"

# 3. Configure API keys (at minimum: one LLM + one search provider)
cp .env.example .env

# 4. Verify installation
.venv/bin/python -c "from open_deep_research.deep_researcher import deep_researcher; print('OK')"

# 5. Run tests
.venv/bin/python -m pytest tests/ -x -q
```

---

## Usage

### LangGraph Studio (Web UI)

```bash
uvx langgraph dev
```
Opens a visual graph editor at `http://localhost:2024` — type a query, step through nodes, inspect state.

### REST API

```bash
ENABLE_REST_API=true API_KEY=your-key .venv/bin/python -m open_deep_research.api.main

# Start research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "Latest advances in transformer architectures"}'

# Stream progress (SSE)
curl -N http://localhost:8000/research/{run_id}/stream

# Get report
curl http://localhost:8000/research/{run_id}/report

# Export
curl http://localhost:8000/research/{run_id}/export/markdown

# Metrics
curl localhost:8000/metrics
```

### Python SDK

```python
import asyncio
from open_deep_research.deep_researcher import deep_researcher

async def main():
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": "Compare React vs Vue for enterprise apps"}]},
        config={
            "configurable": {
                "research_model": "google_genai:gemini-2.5-flash",
                "search_api": "tavily",
            }
        }
    )
    print(result.get("final_report", ""))

asyncio.run(main())
```

### Full Verification

```bash
# Lint + dependency audit + tests with coverage
bash scripts/verify.sh
```

---

## Architecture

### System Overview

The platform is layered: a FastAPI platform layer drives a LangGraph orchestration core, which in
turn calls the intelligence (evidence) and report-generation layers. Cross-cutting services —
telemetry, the rate-limit governor, caching, and input sanitization — wrap the orchestration core
rather than sitting in the linear flow.

```mermaid
flowchart TB
    subgraph CLIENT["Clients"]
        UI["CLI / LangGraph Studio"]
        HTTP["HTTP clients / SSE consumers"]
    end

    subgraph PLATFORM["Platform / API Layer (FastAPI, ASGI-mountable)"]
        API["FastAPI app + middleware<br/>(body-size · request-id · metrics · auth)"]
        RUNNER["ResearchRunner<br/>(async tasks + semaphore)"]
        REPO["ResearchRepository → SQLite"]
        EXTRAS["streaming · memory · webhooks<br/>model router · plugins · metrics"]
    end

    subgraph ORCH["Orchestration — LangGraph (deep_researcher.py)"]
        GRAPH["Main graph"]
        SUPER["Supervisor subgraph"]
        RSCH["Researcher subgraph"]
    end

    subgraph INTEL["Intelligence / Evidence Layer"]
        EVID["evidence.py"]
        CLASS["mode classifier + planner"]
        DOC["document_reader.py"]
        STORM["perspectives.py (STORM)"]
        AGG["search_aggregator.py"]
    end

    subgraph GEN["Report Generation Layer"]
        VERIFY["citation_verifier.py"]
        PROFILES["report_profiles.py"]
        CITE["citation.py"]
        REVIEW["reviewers.py"]
        EXPORT["exporters.py"]
    end

    subgraph CROSS["Cross-cutting Services"]
        TEL["telemetry (budget + cost)"]
        GOV["governor (rate limit + breaker)"]
        CACHE["research_cache (mode-aware TTL)"]
        SANI["sanitization (injection defense)"]
    end

    subgraph EXT["External Services"]
        SEARCH["Tavily / web search"]
        ACAD["arXiv · Semantic Scholar · PubMed · Crossref"]
        LLM["LLM providers"]
        MCP["MCP tool servers"]
    end

    UI --> GRAPH
    HTTP --> API --> RUNNER --> GRAPH
    RUNNER --> REPO
    API --> EXTRAS

    GRAPH --> SUPER --> RSCH
    GRAPH --> CLASS
    GRAPH --> DOC
    RSCH --> EVID
    RSCH --> AGG
    SUPER --> STORM
    GRAPH --> VERIFY
    GRAPH --> PROFILES
    GRAPH --> REVIEW
    GRAPH --> EXPORT --> CITE

    EVID -.uses.-> SANI
    AGG -.through.-> GOV
    AGG -.checks.-> CACHE
    GRAPH -.wrapped by.-> TEL

    AGG --> SEARCH
    AGG --> ACAD
    RSCH --> MCP
    GRAPH --> LLM
```

### Research Pipeline

The research pipeline is a LangGraph with 19 nodes. Diamonds are conditional edges; the
`research_supervisor` node is a subgraph (see below).

```mermaid
flowchart TD
    START([START]) --> CLAR["clarify_with_user"]
    CLAR -->|need_clarification| ENDQ([END · ask user])
    CLAR -->|ok| PARSE["parse_document<br/>(PDF → evidence_cards)"]
    PARSE --> BRIEF["write_research_brief"]
    BRIEF --> CLASSIFY["classify_research_request<br/>(→ research_mode + profile)"]
    CLASSIFY --> PLAN["generate_research_plan<br/>(objective, subquestions, strategy)"]
    PLAN --> REVIEWP{"optional_plan_review<br/>mode?"}
    REVIEWP -->|reject & revs < max| PLAN
    REVIEWP -->|none / auto / approved| SUPER[["research_supervisor<br/>(subgraph)"]]

    SUPER --> VERIFY["verify_citations<br/>(GET→HEAD URL checks)"]
    VERIFY --> OUTLINE["generate_report_outline<br/>(profile + evidence allocation)"]
    OUTLINE --> SECT["write_sections_parallel"]
    SECT --> COMPILE["compile_report<br/>(+ TOC + bibliography)"]
    COMPILE --> FREVIEW["final_review<br/>(4 reviewers in parallel)"]
    FREVIEW --> DECIDE{"avg_score < threshold<br/>AND iter < max(2)?"}
    DECIDE -->|yes| REWRITE["rewrite_sections"]
    REWRITE --> COMPILE
    DECIDE -->|no| EXPORT["export_report<br/>(md/html/pdf/docx/json/bib)"]
    EXPORT --> DONE([END])
```

### Research Graph Nodes

| Phase | Nodes | Description |
|-------|-------|-------------|
| Planning | clarify, parse, brief, classify, plan, review | Understand query, classify mode, generate subquestions |
| Research | supervisor, researchers | Parallel multi-provider search across NC researchers |
| Evidence | extract_evidence, compress | Build evidence cards, dedup, detect conflicts |
| Verification | verify_citations | HTTP check every source URL |
| Generation | outline, section_writers, compile | Profile-aware section writing with citation formatting |
| Review | final_review, rewrite_sections | 4 parallel reviewers (coverage/evidence/contradiction/style) |
| Export | export_report | Markdown, HTML, PDF, DOCX, JSON |

### Supervisor & Researcher Subgraphs

The `research_supervisor` node delegates `ConductResearch` calls to parallel researcher subgraphs
(up to `max_concurrent_research_units`, default 5). Each researcher runs a ReAct tool loop, then
compresses and extracts structured evidence.

```mermaid
flowchart TD
    subgraph SUP["Supervisor subgraph (max_researcher_iterations = 6)"]
        S0([START]) --> S1["supervisor (LLM + tools)"]
        S1 --> S2["supervisor_tools"]
        S2 -->|"ConductResearch (×N parallel)"| RGRAPH[["researcher subgraph"]]
        S2 -->|think_tool| S1
        S2 -->|"ResearchComplete / max iters"| SEND([END])
        S2 -->|else| S1
    end

    subgraph RES["Researcher subgraph (max_react_tool_calls = 10)"]
        R0([START]) --> R1["researcher (LLM + tools)"]
        R1 --> R2["researcher_tools<br/>(search / MCP / think)"]
        R2 -->|more tool calls| R1
        R2 -->|"done / max calls"| R3["compress_research<br/>(token-limit retry)"]
        R3 --> R4["extract_structured_evidence<br/>(if enable_evidence_first)"]
        R4 --> REND([END])
    end

    RGRAPH -.spawns.-> R0
```

### Evidence & Confidence Scoring

Raw search results become scored, deduplicated, conflict-flagged evidence cards. The confidence
score is computed **programmatically** — never self-reported by the LLM.

```mermaid
flowchart LR
    RAW["raw search results"] --> SAN["ContentSanitizer.sanitize<br/>(injection patterns + isolation)"]
    SAN --> EXT["extract_evidence()"]

    subgraph SCORE["confidence = 0.40·cred + 0.35·corrob + 0.25·recency"]
        CRED["source_credibility<br/>0.5 base · +0.2 academic<br/>+0.15 authoritative · −0.2 untrusted"]
        CORR["corroboration_strength<br/>min(1, supporting/3) via Jaccard ≥ 0.7"]
        REC["recency_score<br/>mode-aware window<br/>news 7d · tech 90d · academic 3y · def 180d"]
    end

    EXT --> CRED & CORR & REC
    CRED & CORR & REC --> CARD["EvidenceCard"]
    CARD --> DEDUP["deduplicate_claims()<br/>(keep highest confidence)"]
    DEDUP --> CONF["detect_conflicts()<br/>(negation regex + token overlap)"]
    CONF --> COMP["compress_evidence()<br/>(merge groups, top-5 excerpts)"]
    COMP --> OUT["evidence_cards · sources · conflicts"]
```

### State Model

The top-level LangGraph state is a `TypedDict`; Pydantic models are stored as dicts and
reconstructed when needed. Custom reducers (`merge_sources`, `append_evidence`, `override_reducer`)
govern how concurrent updates merge.

```mermaid
classDiagram
    class AgentState {
        <<TypedDict / MessagesState>>
        messages : list [operator.add]
        supervisor_messages : list [override_reducer]
        research_brief : str
        final_report : str
        sources : list [merge_sources]
        evidence_cards : list [append_evidence]
        conflicts : list
        citation_checks : list [operator.add]
        review_results : list
        research_mode : str
        research_plan : dict
        report_outline : dict
        written_sections : list
        exported_files : dict
        telemetry : dict
    }
    class EvidenceCard {
        id · claim · confidence
        supporting_source_ids
        conflicting_source_ids
        exact_excerpts
        subquestion_id · researcher_id
        deduplicated_from
    }
    class Source {
        url · title · publisher · date
        credibility_score
        source_type · provider · raw_excerpts
    }
    class ConflictFlag {
        card_a_id · card_b_id
        conflict_description · severity
    }
    class CitationCheck {
        claim · url · supports_claim
        problem · fix · status
    }
    class ReviewResult {
        coverage_score · evidence_score · style_score
        contradiction_flags · iteration_count · feedback
    }
    class ResearchPlanExtended {
        objective · subquestions · search_strategy
        expected_source_types · proposed_sections
        stop_conditions · risks
    }

    AgentState "1" o-- "*" EvidenceCard : evidence_cards
    AgentState "1" o-- "*" Source : sources
    AgentState "1" o-- "*" ConflictFlag : conflicts
    AgentState "1" o-- "*" CitationCheck : citation_checks
    AgentState "1" o-- "1" ResearchPlanExtended : research_plan
    AgentState "1" o-- "*" ReviewResult : review_results
    EvidenceCard ..> Source : supporting_source_ids
```

> A standalone reference with all of these diagrams collected in one place lives in
> [`docs/architecture-diagrams.md`](docs/architecture-diagrams.md).

---

## Configuration

Key settings in `src/open_deep_research/configuration.py`. Configurable via `.env`, environment variables, or LangGraph Studio.

### Model Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `research_model` | `openai:gpt-4.1` | Model for research |
| `compression_model` | `openai:gpt-4.1` | Model for compression |
| `final_report_model` | `openai:gpt-4.1` | Model for report writing |
| `search_api` | `tavily` | Search provider |

### Feature Flags

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_citation_verification` | `true` | HTTP-check citations |
| `enable_section_writers` | `true` | Parallel section generation |
| `enable_reviewer_loop` | `true` | QA review + auto-rewrite |
| `enable_evidence_first` | `true` | Evidence-card architecture |
| `enable_model_routing` | `false` | Multi-tier model routing |

Each flag gates a specific node or component, with a legacy fallback preserved when it is off.
Numeric guardrails bound cost and latency regardless of flags.

```mermaid
flowchart LR
    subgraph FLAGS["Feature flags → what they gate"]
        F1["enable_evidence_first"] --> N1["extract_structured_evidence node"]
        F2["enable_mode_classification"] --> N2["classify_research_request node"]
        F3["enable_document_reading"] --> N3["parse_document node"]
        F4["enable_citation_verification"] --> N4["verify_citations node"]
        F5["enable_section_writers"] --> N5["outline / sections / compile nodes"]
        F6["enable_reviewer_loop"] --> N6["final_review + rewrite loop"]
        F7["enable_search_aggregation"] --> N7["multi-provider SearchAggregator"]
        F8["enable_academic_search"] --> N8["arXiv / Semantic Scholar / PubMed / Crossref"]
        F9["enable_storm_research"] --> N9["PerspectiveGenerator (STORM)"]
        F10["enable_model_routing"] --> N10["ModelRouter tier selection"]
    end

    subgraph GUARDS["Numeric guardrails"]
        G1["max_researcher_iterations = 6"]
        G2["max_react_tool_calls = 10"]
        G3["max_concurrent_research_units = 5"]
        G4["max_review_iterations = 2 (hard cap)"]
        G5["review_score_threshold = 0.7"]
        G6["BudgetConfig: 1M tokens / $10 cap"]
    end
```

### Report Profiles

| Profile | Best for |
|---------|----------|
| `executive_brief` | Concise summary with key findings |
| `deep_research_report` | Comprehensive multi-section report |
| `academic_literature_review` | Systematic literature review |
| `investment_memo` | Investment analysis with risk assessment |
| `competitive_landscape` | Competitive analysis |
| `technical_design_research` | Technical architecture research |
| `policy_memo` | Policy analysis |
| `news_brief` | Timely news summary |

### Logging

Logging is configured centrally by `logging_config.setup_logging()` (built on the
vendored `pretty_logger`). It is wired at both entry points — the API server and
`deep_researcher` import — so library, LangGraph Studio, and REST usage all get the
same output. Every module logs through the standard `logging.getLogger(__name__)`
pattern; handlers are attached once to the `open_deep_research` package logger.

Two sinks are always active:
- **Console** — colored by level when stdout is a TTY (auto-plain otherwise).
- **File** — plain text at `LOG_DIR/applog.log`, rotated daily, kept for 30 days.
  Best-effort: a read-only filesystem disables the file sink with a warning instead
  of crashing.

Controlled entirely by environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `LOG_DIR` | `logs` | Directory for the rotating file log |
| `LOG_FORMAT` | auto | `color`, `plain`, or `json` (structured, for Loki/Datadog/etc.) |
| `NO_COLOR` | unset | If set, forces plain console ([no-color.org](https://no-color.org)) |

In `json` mode each line includes a `request_id`, populated per-request by the API
middleware, so a single request can be traced across every module.

```bash
# Verbose colored dev logs
LOG_LEVEL=DEBUG LOG_FORMAT=color uvx langgraph dev

# Structured logs for a container/aggregator
LOG_LEVEL=INFO LOG_FORMAT=json LOG_DIR=/var/log/odr ENABLE_REST_API=true \
  API_KEY=... .venv/bin/python -m open_deep_research.api.main
```

---

## API Endpoints

| Method | Path | Auth Scope | Description |
|--------|------|------------|-------------|
| `POST` | `/research` | `research:write` | Start research |
| `GET` | `/research/{id}` | `research:read` | Get run status |
| `GET` | `/research/{id}/stream` | `research:read` | SSE progress stream |
| `GET` | `/research/{id}/report` | `research:read` | Get final report |
| `GET` | `/research/{id}/export/{fmt}` | `research:read` | Export report |
| `POST` | `/research/{id}/cancel` | `research:write` | Cancel run |
| `GET` | `/memory` | `research:read` | List saved topics |
| `GET` | `/memory/{hash}` | `research:read` | Load saved research |
| `POST` | `/memory/feedback` | `research:write` | Save preference |
| `POST` | `/webhooks` | `admin` | Register webhook |
| `GET` | `/webhooks` | `admin` | List webhooks |
| `DELETE` | `/webhooks/{id}` | `admin` | Remove webhook |
| `GET` | `/plugins` | `admin` | List plugins |
| `GET` | `/health` | none | Health check |
| `GET` | `/metrics` | none | Prometheus metrics |

### Request Lifecycle

A research run executes in the background. The client receives a `run_id` immediately, then follows
progress over SSE while the graph runs and a webhook fires on completion.

```mermaid
sequenceDiagram
    actor Client
    participant API as FastAPI (/research)
    participant Runner as ResearchRunner
    participant Graph as deep_researcher
    participant Repo as SQLite Repository
    participant Hook as WebhookNotifier

    Client->>API: POST /research {query, config}
    API->>API: verify_api_key + body-size + length checks
    API->>Runner: start(repo, query, config)
    Runner->>Repo: create_run() → run_id
    API-->>Client: 200 {run_id, status:"pending"}

    Note over Runner: acquire semaphore (max_concurrent_runs)
    Runner->>Repo: update_run_status("running")
    Runner->>Graph: astream(messages, config{repo, run_id})

    par SSE streaming
        Client->>API: GET /research/{id}/stream
        API->>Repo: watch_run() polls progress_after(seq) every 0.5s
        Repo-->>Client: data: ProgressEvent (SSE)
    and Graph execution
        loop each node
            Graph->>Repo: append_progress(phase_complete)
        end
    end

    Graph-->>Runner: final_report
    Runner->>Repo: save_report() + update_run_status("completed")
    Runner->>Hook: notify("run_completed") → HMAC-sign + POST (retry w/ backoff)
    Client->>API: GET /research/{id}/report
    Repo-->>Client: {markdown, ...}
```

---

## Project Structure

```
src/open_deep_research/
├── deep_researcher.py       # 19-node LangGraph
├── configuration.py         # All settings + feature flags
├── state.py                 # State + Pydantic models
├── evidence.py              # Evidence extraction engine
├── citation_verifier.py     # HTTP citation verification
├── report_profiles.py       # 8 report profiles
├── citation.py              # 6 citation styles + BibTeX
├── exporters.py             # Multi-format export
├── reviewers.py             # 4 QA reviewers
├── search_aggregator.py     # Multi-provider search
├── document_reader.py       # PDF parsing
├── perspectives.py          # STORM multi-perspective
├── sanitization.py          # Injection defense
├── governor.py              # Rate limiting
├── telemetry.py             # Cost tracking
├── research_cache.py        # Mode-aware caching
├── exceptions.py            # Exception hierarchy
├── logging_config.py        # Centralized logging setup (levels, formats, rotation)
├── _vendor/                 # Vendored third-party (pretty_logger, upstream snapshot)
└── api/                     # REST API server
    ├── main.py              # FastAPI app
    ├── repository.py        # Persistence ABC
    ├── repository_sqlite.py # SQLite backend
    ├── runner.py            # Background execution
    ├── streaming.py         # SSE streaming
    ├── memory.py            # Cross-session memory
    ├── webhooks.py          # Webhook notifications
    ├── model_router.py      # Multi-model routing
    ├── metrics.py           # Prometheus metrics
    ├── plugins/             # Source plugin system
    └── routes/              # Route handlers
```

### Module Dependencies

How the modules above depend on one another. `deep_researcher.py` is the hub; the `api/` package
wraps it for background/HTTP execution.

```mermaid
flowchart LR
    subgraph core["Core"]
        DR["deep_researcher.py"]
        ST["state.py"]
        CFG["configuration.py"]
        EXC["exceptions.py"]
    end
    subgraph intel["Intelligence"]
        EV["evidence.py"]
        DOCR["document_reader.py"]
        PER["perspectives.py"]
        SA["search_aggregator.py"]
    end
    subgraph gen["Generation"]
        CV["citation_verifier.py"]
        RP["report_profiles.py"]
        CT["citation.py"]
        RV["reviewers.py"]
        EXP["exporters.py"]
    end
    subgraph crosscut["Cross-cutting"]
        TE["telemetry.py"]
        GV["governor.py"]
        CA["research_cache.py"]
        SN["sanitization.py"]
    end
    subgraph api["api/"]
        RUN["runner.py"]
        RPO["repository*.py"]
        STRm["streaming.py"]
        MR["model_router.py"]
        PL["plugins/*"]
    end

    DR --> ST & CFG & EXC
    DR --> EV & DOCR & PER
    DR --> CV & RP & RV & EXP
    EV --> SN
    EV --> ST
    SA --> GV & CA & PL
    EXP --> CT
    DR -.telemetry.-> TE
    RUN --> DR
    RUN --> RPO
    STRm --> RPO
    DR --> MR
```

---

## Design Principles

- **Evidence-first:** Every claim backed by structured evidence with deterministic confidence scores — not LLM self-reports
- **No vector-RAG:** Agentic document reading via section-by-section LLM interrogation; no chunking or embeddings
- **Conflict-aware:** Contradictory claims are presented explicitly, never averaged into false consensus
- **Feature-flagged:** All new components behind configuration flags with preserved legacy fallback paths
- **Observable by default:** LangSmith tracing, Prometheus metrics, structured logging on every API request

---

## License

MIT — see `LICENSE`. Original `open_deep_research` Copyright (c) 2025 LangChain.
