# ADR 002: Two-Tier State Architecture (TypedDict + Pydantic)

**Status:** Accepted

**Date:** 2025-07-15

## Context

LangGraph, the framework underpinning Open Deep Research, requires graph state to be defined as a `TypedDict` with `Annotated` type annotations for reducer functions on message-passing nodes. Without `TypedDict`, LangGraph cannot manage state transitions, apply reducers, or checkpoint execution.

However, the platform's domain logic needs structured, validated data models — `Source`, `EvidenceCard`, `ConflictFlag`, `CitationCheck`, `ReviewResult`, `ResearchPlan`, and `ResearchPlanExtended`. These objects require field validation, default values, serialization, and type coercion — all features of Pydantic `BaseModel`.

The tension: LangGraph demands `TypedDict` (for reducers), but domain logic needs Pydantic (for validation).

A flat `TypedDict` with loose dict types sacrifices all the benefits of Pydantic validation. A pure-Pydantic state breaks LangGraph reducer compatibility. A compromise was needed.

## Decision

We implemented a **two-tier architecture**:

- **Top-level state**: `TypedDict` classes (`AgentState`, `SupervisorState`, `ResearcherState`) with LangGraph reducer annotations (`operator.add`, `merge_sources`, `append_evidence`, `override_reducer`)
- **Values inside state**: Pydantic models stored as dicts via `.model_dump()` and reconstructed when needed via `.model_validate()`

### Key State Fields and Reducers

```
AgentState (extends MessagesState):
  sources:             Annotated[List[dict], merge_sources]       # dedup by URL, keep highest credibility
  evidence_cards:      Annotated[List[dict], append_evidence]    # append new cards
  conflicts:           List[dict]                                 # single-writer, no reducer
  citation_checks:     Annotated[List[dict], operator.add]        # append new checks
  review_results:      List[dict]                                 # single-writer, no reducer
  supervisor_messages: Annotated[list, override_reducer]          # additive or override
  raw_notes:           Annotated[list[str], override_reducer]
  notes:               Annotated[list[str], override_reducer]
  telemetry:           dict
  total_tokens:        int
```

### Three State Types for Multi-Agent Graph

1. **`AgentInputState`** (extends `MessagesState`) — minimal input-only state with `messages` and `document_paths`
2. **`AgentState`** (extends `MessagesState`) — full 18-field main state with all evidence-first fields, adaptive research fields, and report generation fields
3. **`SupervisorState`** (`TypedDict`) — supervisor subgraph state with oversight fields
4. **`ResearcherState`** (`TypedDict`) — individual researcher subgraph state
5. **`ResearcherOutputState`** (Pydantic `BaseModel`) — output from researcher subgraph for structured parsing

### Custom Reducers

Three custom reducers handle specific merging strategies:

- **`merge_sources`**: Deduplicates sources by URL, keeping the entry with the highest `credibility_score`. Iterates existing list to build a URL index, then for each new source either updates (if higher credibility) or appends.

- **`append_evidence`**: Simple concatenation of new evidence cards to existing list. Always additive.

- **`override_reducer`**: Mixed-mode reducer — when a dict with `{"type": "override"}` is received, replaces the current value; otherwise delegates to `operator.add` for standard concatenation.

## Consequences

### Positive

- **Full Pydantic validation** on all structured data — sources, evidence cards, conflicts, citations, review results all benefit from field validation, type coercion, and default values.
- **LangGraph reducers work natively** — the `TypedDict` with `Annotated` reducers satisfies LangGraph's state management requirements, enabling checkpointing and message routing.
- **Additive reducers for evidence/sources** enable multi-node parallel collection — multiple researcher nodes can contribute sources and evidence concurrently without conflicts, all deduplicated at the state level.
- **`merge_sources` reducer deduplicates by URL** at state level — not just application level — ensuring source uniqueness is maintained across graph transitions regardless of which node adds them.
- **Backward compatibility** — new fields with defaults (`[]`, `{}`, `0`) don't break existing serialized state snapshots.

### Trade-offs

- **Dict-in-TypedDict overhead** — `.model_dump()` / `.model_validate()` conversion at each node boundary adds serialization cost. For large evidence lists, this can accumulate across 19 graph nodes.
- **`conflicts` and `review_results` use no reducer** — they are single-writer fields (only the conflict detector and reviewer nodes write to them). If future parallelism requires multiple writers, reducers must be added. Current design relies on node scheduling to avoid races.
- **Pydantic models duplicated in state (dict form) and memory (model form)** — when a node reads a dict from state and converts to a Pydantic model, the original dict remains in state. If the node modifies the model and re-dumps, there's a brief window where two representations could diverge. This is mitigated by the convention of converting at boundary and discarding after use.
- **No type safety at state access** — state fields are `List[dict]` not `List[Source]`, so consumers must know to call `.model_validate()`. This is enforced by convention and utility functions rather than the type system.

### File

Implementation at `src/open_deep_research/state.py` — 278 lines, 7 Pydantic models, 3 TypedDict states, 3 custom reducers.
