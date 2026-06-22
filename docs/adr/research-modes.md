# ADR 004: Research Mode Classification System

**Status:** Accepted

**Date:** 2025-07-15

## Context

Open Deep Research supports diverse query types — comparison shopping, fact-checking, market analysis, academic surveys, and more. Each type demands a fundamentally different research strategy: different search providers, recency requirements, perspective sets, and report structures. The original flat pipeline treated all queries identically, producing generic reports regardless of the query's nature.

Key drivers for a structured classification system:

1. **Search strategy varies by domain** — Academic queries need arXiv/Semantic Scholar; news queries need recency-biased web search; comparison queries need multi-entity frameworks
2. **Recency sensitivity differs** — A news query is stale after 7 days; an academic review tolerates 3-year-old sources; a technical implementation falls in between
3. **Report structure should mirror query intent** — A comparison needs side-by-side pro/con tables; a fact-check needs claim-evidence-verdict; a market landscape needs competitor maps
4. **Classification must be deterministic enough to cache and route** — Downstream nodes (planner, search aggregator, perspective generator, profile selector, recency scorer) all branch on mode

## Decision

We implemented a 9-mode `ResearchMode` enum backed by an LLM-based classifier with confidence-based fallback.

### The Enum

Defined in `configuration.py:11`:

| Mode | Purpose | Example Queries |
|------|---------|-----------------|
| `comparison` | Side-by-side comparison | "React vs Vue", "AWS vs Azure vs GCP" |
| `validation_or_fact_check` | Verifying claims | "Is dark matter real?", "Does meditation reduce anxiety?" |
| `market_landscape` | Industry/competitive analysis | "AI market 2026", "fintech landscape" |
| `academic_literature_review` | Surveying academic research | "Research on transformer architectures" |
| `company_due_diligence` | Company-specific research | "OpenAI financials", "Tesla competitive position" |
| `technical_implementation` | How-to guides, architecture decisions | "How to build a RAG system", "microservices vs monolith" |
| `news_or_current_events` | Time-sensitive developments | "Latest AI news", "Recent developments in Y" |
| `policy_legal_regulatory` | Policy/legal/regulatory landscape | "EU AI Act impact", "Data privacy regulations" |
| `custom` | Fallback for anything else | Adaptive structure |

### Classification Pipeline

1. User query enters the graph via `classify_research_request` node
2. `classify_query_mode()` calls the classifier LLM with `CLASSIFIER_SYSTEM_PROMPT` (prompts.py:374) + `CLASSIFIER_HUMAN_PROMPT` (prompts.py:423)
3. Classifier returns `{mode, confidence, reasoning, suggested_report_profile}`
4. If confidence < `classifier_confidence_threshold` (default 0.6), falls back to `custom` mode
5. Mode stored in `AgentState.research_mode` and propagated to downstream nodes

### Configuration (configuration.py:249-294)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_research_mode` | `ResearchMode` | `custom` | Mode when classification disabled or uncertain |
| `enable_mode_classification` | `bool` | `True` | Toggle for the classifier node |
| `classifier_model` | `str \| None` | `None` | Override model for classifier (defaults to `research_model`) |
| `classifier_confidence_threshold` | `float` | `0.6` | Minimum confidence to accept classification |

### Downstream Impacts

**Planner** (`prompts.py:433`): `PLANNER_SYSTEM_PROMPT` includes the mode in the prompt, steering search strategy generation toward mode-appropriate approaches.

**Search aggregation**: Mode influences provider selection — `academic_literature_review` queries route to arXiv/Semantic Scholar/PubMed; news queries favor recency-biased web search.

**Recency scoring** (`evidence.py:66-72`): `compute_recency_score()` uses mode-specific windows:

| Mode | Recency Window |
|------|---------------|
| `news_or_current_events` | 7 days |
| `technical_implementation` | 90 days |
| `academic_literature_review` | 3 years |
| `custom` / default | 6 months |

**Perspectives** (`perspectives.py`): Mode-specific perspective generation strategies — `market_landscape` uses competitor/market/customer perspectives; `policy_legal_regulatory` uses regulator/advocate/critic/affected perspectives; `comparison` uses entity-A/entity-B/neutral-comparison perspectives.

**Report profile** (`report_profiles.py:188`): `MODE_TO_PROFILE` maps modes to report profiles:

| Mode | Profile |
|------|---------|
| `comparison` | `competitive_landscape` |
| `validation_or_fact_check` | `deep_research_report` |
| `market_landscape` | `competitive_landscape` |
| `academic_literature_review` | `academic_literature_review` |
| `company_due_diligence` | `investment_memo` |
| `technical_implementation` | `technical_design_research` |
| `news_or_current_events` | `news_brief` |
| `policy_legal_regulatory` | `policy_memo` |
| `custom` | `deep_research_report` |

## Consequences

### Positive

- **Adaptive search per query type** — Academic queries use specialized databases; news queries prioritize recency; comparison queries frame results as entity comparisons
- **`custom` fallback with confidence threshold** — Prevents misclassification from corrupting research strategy; sub-threshold queries get the safest default
- **Mode-specific recency windows** — Appropriate time sensitivity per domain (7 days for news vs 3 years for academic)
- **Deterministic after classification** — Once classified, all downstream branching is purely rule-based (mode → windows, mode → profile, mode → perspectives), keeping the system predictable
- **Feature-flagged** — `enable_mode_classification` defaults to True but can be disabled, making the classifier a safe add-on rather than a hard dependency

### Trade-offs

- **LLM classification cost/latency** — Each research session incurs 1 extra LLM call for classification (+1 round-trip, ~$0.001–$0.01 depending on model). The response is small (JSON), so the token cost is minimal but non-zero.
- **9-mode surface area** — Every new mode impacts 5+ subsystems (planner, search aggregator, recency scorer, perspective generator, profile registry). Adding a mode requires coordinated changes across all these files.
- **Fixed enum** — Modes are compile-time constants. Dynamic or user-defined modes are not supported. Adding a mode requires code changes, config updates, and prompt modifications.
- **Classifier quality tail-risk** — Low-quality classification (e.g., calling an academic query "news") cascades errors through the entire pipeline. The confidence threshold mitigates this but only for low-confidence cases — confident misclassifications are undetected.

### File Locations

- `src/open_deep_research/configuration.py:11` — `ResearchMode` enum
- `src/open_deep_research/configuration.py:249-294` — Classifier configuration fields
- `src/open_deep_research/prompts.py:374-427` — `CLASSIFIER_SYSTEM_PROMPT` and `CLASSIFIER_HUMAN_PROMPT`
- `src/open_deep_research/deep_researcher.py:314-315` — Classifier node invocation
- `src/open_deep_research/evidence.py:66-72` — Mode-specific recency windows
- `src/open_deep_research/perspectives.py:149-151` — Mode-specific perspective strategies
- `src/open_deep_research/report_profiles.py:188-198` — `MODE_TO_PROFILE` mapping
