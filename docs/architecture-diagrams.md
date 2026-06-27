# Advanced Deep Research — Architecture, Implementation & Workflow Diagrams

> All diagrams use [Mermaid](https://mermaid.js.org/) and render natively on GitLab/GitHub, in
> VS Code (Markdown Preview Mermaid), JetBrains, and at <https://mermaid.live>.
>
> Scope: the enhanced LangChain `open_deep_research` (ODR) platform —
> evidence-first research, adaptive strategies, QA loops, report profiles, and the platform API.
> Source: `open_deep_research/src/open_deep_research/`. Last mapped: 2026-06-24.

---

## Table of Contents

1. [System Architecture (layered)](#1-system-architecture)
2. [Component / Implementation Diagram](#2-component--implementation-diagram)
3. [Workflow Diagram — the LangGraph pipeline](#3-workflow-diagram--the-langgraph-pipeline)
4. [Sub-workflows: Supervisor & Researcher subgraphs](#4-sub-workflows-supervisor--researcher-subgraphs)
5. [Evidence confidence pipeline (data flow)](#5-evidence-confidence-pipeline)
6. [State & data model (class diagram)](#6-state--data-model)
7. [Platform API request lifecycle (sequence)](#7-platform-api-request-lifecycle)
8. [Feature flags & configuration map](#8-feature-flags--configuration-map)
9. [Key design decisions](#9-key-design-decisions)

---

## 1. System Architecture

Layered view of the whole platform. Cross-cutting services (telemetry, governor, cache,
sanitization) wrap the orchestration layer rather than sitting in the linear flow.

```mermaid
flowchart TB
    subgraph CLIENT["Clients"]
        UI["CLI / LangGraph Studio"]
        HTTP["HTTP clients / SSE consumers"]
    end

    subgraph PLATFORM["Platform / API Layer — Phase 4 & 5 (FastAPI, ASGI-mountable)"]
        API["FastAPI app<br/>main.py + middleware<br/>(body-size, request-id, metrics, auth)"]
        ROUTES["Routers<br/>/research · /memory · /admin · /metrics"]
        RUNNER["ResearchRunner<br/>(async tasks + semaphore)"]
        STREAM["watch_run()<br/>SSE progress stream"]
        REPO["ResearchRepository (ABC)<br/>→ SqliteResearchRepository"]
        MEM["ResearchMemory<br/>(cross-session)"]
        HOOK["WebhookNotifier<br/>(HMAC + retry)"]
        ROUTER["ModelRouter<br/>(FAST/BALANCED/QUALITY)"]
        PLUG["PluginLoader<br/>+ SourcePlugin ABC"]
        METRICS["MetricsCollector<br/>(Prometheus)"]
    end

    subgraph ORCH["Orchestration Layer — LangGraph (deep_researcher.py)"]
        GRAPH["Main graph<br/>clarify → brief → classify → plan → supervise → report → export"]
        SUPER["Supervisor subgraph"]
        RSCH["Researcher subgraph"]
    end

    subgraph INTEL["Intelligence / Evidence Layer — Phase 1 & 2"]
        EVID["evidence.py<br/>extract · dedup · conflicts · compress"]
        CLASS["Mode classifier + planner<br/>(9 ResearchModes)"]
        DOC["document_reader.py<br/>(agentic PDF, no vector-RAG)"]
        STORM["perspectives.py<br/>(STORM coverage matrix)"]
        AGG["search_aggregator.py<br/>(multi-provider + dedup)"]
    end

    subgraph GEN["Report Generation Layer — Phase 3"]
        VERIFY["citation_verifier.py"]
        PROFILES["report_profiles.py<br/>(8 profiles)"]
        CITE["citation.py<br/>(6 styles + BibTeX)"]
        REVIEW["reviewers.py<br/>(4 algorithmic reviewers)"]
        EXPORT["exporters.py<br/>(md/html/pdf/docx/json/bib)"]
    end

    subgraph CROSS["Cross-cutting Services"]
        TEL["telemetry.py<br/>(budget + cost)"]
        GOV["governor.py<br/>(rate limit + circuit breaker)"]
        CACHE["research_cache.py<br/>(mode-aware TTL)"]
        SANI["sanitization.py<br/>(injection defense)"]
    end

    subgraph EXT["External Services"]
        SEARCH["Tavily / OpenAI / Anthropic web search"]
        ACAD["arXiv · Semantic Scholar · PubMed · Crossref"]
        LLM["LLM providers (OpenAI / Anthropic / Google)"]
        MCP["MCP tool servers"]
    end

    UI --> GRAPH
    HTTP --> API
    API --> ROUTES --> RUNNER --> GRAPH
    ROUTES --> STREAM
    RUNNER --> REPO
    STREAM --> REPO
    ROUTES --> MEM --> REPO
    RUNNER --> HOOK
    ROUTES --> METRICS
    GRAPH --> ROUTER
    GRAPH --> PLUG

    GRAPH --> SUPER --> RSCH
    GRAPH --> CLASS
    GRAPH --> DOC
    RSCH --> EVID
    RSCH --> AGG
    SUPER --> STORM
    GRAPH --> VERIFY
    GRAPH --> PROFILES
    GRAPH --> REVIEW
    GRAPH --> EXPORT
    EXPORT --> CITE

    EVID -.uses.-> SANI
    AGG -.through.-> GOV
    AGG -.checks.-> CACHE
    GRAPH -.wrapped by.-> TEL
    ROUTER --> LLM

    AGG --> SEARCH
    AGG --> ACAD
    RSCH --> MCP
    GRAPH --> LLM
```

---

## 2. Component / Implementation Diagram

Module-level map showing concrete classes/functions per file and how they depend on each other.

```mermaid
flowchart LR
    subgraph core["Core (orchestration + state)"]
        DR["deep_researcher.py<br/>~17 nodes + 2 subgraphs"]
        ST["state.py<br/>AgentState · SupervisorState · ResearcherState<br/>EvidenceCard · Source · ConflictFlag<br/>CitationCheck · ReviewResult · ResearchPlanExtended<br/>reducers: merge_sources · append_evidence · override_reducer"]
        CFG["configuration.py<br/>Configuration · ResearchMode(9) · SearchAPI · MCPConfig"]
        EXC["exceptions.py<br/>ODRError → TokenLimit/Transient/Permanent/Model/Budget"]
        PR["prompts.py"]
    end

    subgraph intel["Intelligence layer"]
        EV["evidence.py<br/>extract_evidence · deduplicate_claims<br/>detect_conflicts · compress_evidence<br/>compute_{credibility,corroboration,recency}"]
        DOCR["document_reader.py<br/>DocumentReader.parse_pdf / extract_evidence_cards"]
        PER["perspectives.py<br/>PerspectiveGenerator · CoverageMatrix"]
        SA["search_aggregator.py<br/>SearchAggregator · SearchResult · ProviderConfig"]
    end

    subgraph gen["Generation layer"]
        CV["citation_verifier.py<br/>CitationVerifier (GET→HEAD, semaphore=5)"]
        RP["report_profiles.py<br/>ReportProfile · 8 profiles · MODE_TO_PROFILE"]
        CT["citation.py<br/>CitationFormatter (6 styles + BibTeX)"]
        RV["reviewers.py<br/>Coverage/Evidence/Contradiction/Style"]
        EXP["exporters.py<br/>get_exporter() · 6 exporters"]
    end

    subgraph crosscut["Cross-cutting"]
        TE["telemetry.py<br/>TelemetryCollector · BudgetConfig · check_budget"]
        GV["governor.py<br/>ConcurrencyGovernor (per-provider, circuit breaker)"]
        CA["research_cache.py<br/>ResearchCache (mode-aware TTL)"]
        SN["sanitization.py<br/>ContentSanitizer (63 patterns)"]
    end

    subgraph api["Platform API"]
        MAIN["api/main.py + config.py + deps.py"]
        RTS["api/routes/{research,memory,admin}.py"]
        RUN["api/runner.py ResearchRunner"]
        STR["api/streaming.py watch_run"]
        RPO["api/repository*.py (ABC + SQLite)"]
        MDL["api/models.py<br/>ProgressEvent · RunRecord · WebhookConfig"]
        MM["api/memory.py ResearchMemory"]
        WH["api/webhooks.py WebhookNotifier"]
        MR["api/model_router.py ModelRouter"]
        MET["api/metrics.py MetricsCollector"]
        PL["api/plugins/* SourcePlugin · PluginLoader"]
    end

    DR --> ST
    DR --> CFG
    DR --> EXC
    DR --> PR
    DR --> EV
    DR --> DOCR
    DR --> PER
    DR --> CV
    DR --> RP
    DR --> RV
    DR --> EXP
    EV --> SN
    EV --> ST
    SA --> GV
    SA --> CA
    SA --> PL
    EXP --> CT
    DR -.telemetry.-> TE

    MAIN --> RTS
    RTS --> RUN
    RTS --> STR
    RTS --> MM
    RUN --> DR
    RUN --> RPO
    STR --> RPO
    MM --> RPO
    RUN --> WH
    RTS --> MET
    DR --> MR
    RPO --> MDL
```

---

## 3. Workflow Diagram — the LangGraph pipeline

End-to-end runtime flow through the main graph, including the clarification early-exit, the
research subgraph, and the reviewer rewrite loop. Diamonds are conditional edges.

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

    LEGACY["final_report_generation<br/>(legacy flat-text fallback)"]:::legacy
    classDef legacy fill:#eee,stroke:#999,stroke-dasharray: 5 5;
```

> The Phase-3 report path (`verify_citations` → … → `export_report`) is active when
> `enable_section_writers` is on. With evidence-first/section-writers disabled, the graph falls
> back to the legacy `final_report_generation` node (dashed).

---

## 4. Sub-workflows: Supervisor & Researcher subgraphs

The supervisor delegates `ConductResearch` calls to parallel researcher subgraphs
(`max_concurrent_research_units`, default 5). Each researcher runs a ReAct loop, compresses, and
extracts structured evidence.

```mermaid
flowchart TD
    subgraph SUP["Supervisor subgraph (max_researcher_iterations=6)"]
        S0([START]) --> S1["supervisor<br/>(LLM + tools)"]
        S1 --> S2["supervisor_tools"]
        S2 -->|"ConductResearch (×N parallel)"| RGRAPH[["researcher subgraph"]]
        S2 -->|think_tool| S1
        S2 -->|"ResearchComplete / max iters"| SEND([END])
        S2 -->|else| S1
    end

    subgraph RES["Researcher subgraph (max_react_tool_calls=10)"]
        R0([START]) --> R1["researcher<br/>(LLM + tools)"]
        R1 --> R2["researcher_tools<br/>(search / MCP / think)"]
        R2 -->|more tool calls| R1
        R2 -->|"done / max calls"| R3["compress_research<br/>(token-limit retry)"]
        R3 --> R4["extract_structured_evidence<br/>(if enable_evidence_first)"]
        R4 --> REND([END])
    end

    RGRAPH -.spawns.-> R0
    R2 -.through governor.-> EXTSEARCH["search_aggregator → Tavily/academic/MCP"]
    R4 -.uses.-> EVMOD["evidence.py + sanitization.py"]
```

---

## 5. Evidence confidence pipeline

How raw search results become scored, deduplicated, conflict-flagged EvidenceCards. The
confidence score is computed **programmatically**, never self-reported by the LLM.

```mermaid
flowchart LR
    RAW["raw search results"] --> SAN["ContentSanitizer.sanitize<br/>(63 injection patterns,<br/>isolation wrapping)"]
    SAN --> EXT["extract_evidence()"]

    subgraph SCORE["confidence = 0.40·cred + 0.35·corrob + 0.25·recency"]
        CRED["source_credibility<br/>0.5 base +0.2 academic<br/>+0.15 authoritative −0.2 untrusted"]
        CORR["corroboration_strength<br/>min(1, supporting/3)<br/>via Jaccard ≥ 0.7"]
        REC["recency_score<br/>mode-aware window<br/>news 7d · tech 90d · academic 3y · def 180d"]
    end

    EXT --> CRED
    EXT --> CORR
    EXT --> REC
    CRED --> CARD["EvidenceCard"]
    CORR --> CARD
    REC --> CARD

    CARD --> DEDUP["deduplicate_claims()<br/>(Jaccard, keep highest confidence)"]
    DEDUP --> CONF["detect_conflicts()<br/>(negation regex + token overlap)"]
    CONF --> COMP["compress_evidence()<br/>(merge groups, top-5 excerpts)"]
    COMP --> OUT["evidence_cards[] + sources[] + conflicts[]"]
```

---

## 6. State & data model

The top-level LangGraph state is a `TypedDict`; Pydantic models are stored as dicts and
reconstructed when needed. Custom reducers govern how concurrent updates merge.

```mermaid
classDiagram
    class AgentState {
        <<TypedDict / MessagesState>>
        messages : list [operator.add]
        supervisor_messages : list [override_reducer]
        research_brief : str
        raw_notes : list [override_reducer]
        notes : list [override_reducer]
        final_report : str
        sources : list [merge_sources]
        evidence_cards : list [append_evidence]
        conflicts : list
        citation_checks : list [operator.add]
        review_results : list
        research_mode : str
        report_profile : dict
        research_plan : dict
        document_artifacts : list
        report_outline : dict
        written_sections : list
        exported_files : dict
        rewrite_instructions : list
        telemetry : dict
        total_tokens : int
    }

    class EvidenceCard {
        id : str
        claim : str
        confidence : float
        supporting_source_ids : list
        conflicting_source_ids : list
        exact_excerpts : list
        subquestion_id : str
        researcher_id : str
        deduplicated_from : list
    }
    class Source {
        url · title · publisher · date
        credibility_score : float
        source_type · provider
        raw_excerpts : list
        accessed_at
    }
    class ConflictFlag {
        card_a_id · card_b_id
        conflict_description
        severity
    }
    class CitationCheck {
        claim · url
        supports_claim : bool
        problem · fix · status
    }
    class ReviewResult {
        coverage_score · evidence_score
        style_score
        contradiction_flags : list
        iteration_count : int
        feedback
    }
    class ResearchPlanExtended {
        objective
        subquestions : list
        search_strategy : dict
        expected_source_types : list
        proposed_sections : list
        stop_conditions : list
        risks : list
    }

    AgentState "1" o-- "*" EvidenceCard : evidence_cards
    AgentState "1" o-- "*" Source : sources
    AgentState "1" o-- "*" ConflictFlag : conflicts
    AgentState "1" o-- "*" CitationCheck : citation_checks
    AgentState "1" o-- "1" ResearchPlanExtended : research_plan
    AgentState "1" o-- "*" ReviewResult : review_results
    EvidenceCard ..> Source : supporting_source_ids
    ConflictFlag ..> EvidenceCard : references
```

---

## 7. Platform API request lifecycle

Sequence for a research run submitted over HTTP, including background execution, SSE streaming,
webhook delivery, and persistence.

```mermaid
sequenceDiagram
    actor Client
    participant API as FastAPI (routes/research)
    participant Runner as ResearchRunner
    participant Graph as deep_researcher
    participant Repo as SqliteRepository
    participant Hook as WebhookNotifier

    Client->>API: POST /research {query, config}
    API->>API: verify_api_key + body-size + length checks
    API->>Runner: start(repo, query, config)
    Runner->>Repo: create_run() → run_id
    Runner-->>API: run_id (status=pending)
    API-->>Client: 200 {run_id, status:"pending"}

    Note over Runner: acquire semaphore (max_concurrent_runs)
    Runner->>Repo: update_run_status("running")
    Runner->>Repo: append_progress(run_started)
    Runner->>Graph: astream(messages, config{repo,run_id})

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
    Runner->>Hook: notify("run_completed", payload)
    Hook->>Hook: HMAC-sign + POST (retry w/ backoff)
    Client->>API: GET /research/{id}/report
    API->>Repo: get_report()
    Repo-->>Client: {markdown, ...}
```

---

## 8. Feature flags & configuration map

Every enhancement is gated by a flag in `Configuration` with a legacy fallback. Default values
shown; the kill switch resets Phase 2+ flags in <30s.

```mermaid
flowchart TB
    subgraph FLAGS["Configuration feature flags"]
        F1["enable_evidence_first = false"]
        F2["enable_mode_classification = true"]
        F3["enable_document_reading = true"]
        F4["enable_citation_verification = true"]
        F5["enable_section_writers = true"]
        F6["enable_reviewer_loop = true"]
        F7["enable_search_aggregation = false"]
        F8["enable_academic_search = true"]
        F9["enable_storm_research = false"]
        F10["enable_model_routing = false"]
    end

    F1 --> N1["extract_structured_evidence node"]
    F2 --> N2["classify_research_request node"]
    F3 --> N3["parse_document node"]
    F4 --> N4["verify_citations node"]
    F5 --> N5["outline/sections/compile nodes"]
    F6 --> N6["final_review + rewrite loop"]
    F7 --> N7["SearchAggregator multi-provider"]
    F8 --> N8["arXiv/SemanticScholar/PubMed/Crossref"]
    F9 --> N9["PerspectiveGenerator (STORM)"]
    F10 --> N10["ModelRouter tier selection"]

    subgraph GUARDS["Guardrails (numeric)"]
        G1["max_researcher_iterations = 6"]
        G2["max_react_tool_calls = 10"]
        G3["max_concurrent_research_units = 5"]
        G4["max_review_iterations = 2 (hard cap)"]
        G5["review_score_threshold = 0.7"]
        G6["max_plan_revisions = 2"]
        G7["BudgetConfig: 1M tokens / $10 cap"]
    end
```

---

## 9. Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Evidence-first** | Structured `EvidenceCard`s with deterministic confidence replace flat-text compression — claim-level citations and corroboration tracking. |
| **No vector-RAG** | Agentic, section-by-section document interrogation (`document_reader.py`) instead of chunking/embeddings. Non-negotiable per plan. |
| **Deterministic confidence** | `0.40·credibility + 0.35·corroboration + 0.25·recency`, computed in code — never LLM self-reported. |
| **Conflict presentation** | Contradictions surfaced explicitly ("Source A claims X, Source B argues Y"); never averaged into false consensus. |
| **Feature flags + legacy fallback** | Every new component behind a flag; `final_report_generation` legacy path always preserved; <30s kill switch. |
| **TypedDict state, Pydantic values** | LangGraph reducers need dict state; Pydantic models stored as dicts and reconstructed on use. |
| **Reviewer loop hard cap** | Max 2 rewrite iterations — fixed latency/cost guardrail. |
| **Citation rule** | GET-first (not HEAD); bot-blocked URLs tagged `unverified` (not re-researched); only `dead` URLs trigger fallback. |
| **Cross-cutting governor/telemetry/cache** | Rate-limit circuit breaker, budget enforcement, and mode-aware caching wrap external calls rather than living in the linear flow. |
| **Platform API is ASGI-mountable** | FastAPI app with SQLite repository, SSE streaming, HMAC webhooks, and Prometheus metrics — mountable into a larger ASGI app. |

---

*Generated from a structured read of the ODR source tree. To regenerate after code changes, re-map
`deep_researcher.py` (nodes/edges), `state.py` (models/reducers), `configuration.py` (flags), and the
`api/` package, then update the affected diagram blocks.*
