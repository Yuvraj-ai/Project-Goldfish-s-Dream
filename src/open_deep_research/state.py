"""Graph state definitions and data structures for the Deep Research agent.

State typing boundary:
- Top-level state: TypedDict (required for LangGraph reducers)
- Values inside state: Pydantic models (for validation/serialization)
- Pydantic models stored as dicts in state; reconstructed when needed
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ──────────────────────────────────────────────
# Structured Outputs (existing — unchanged)
# ──────────────────────────────────────────────

class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""

    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


# ──────────────────────────────────────────────
# Evidence-First Pydantic Models (new)
# ──────────────────────────────────────────────

class Source(BaseModel):
    """A deduplicated source with metadata."""
    url: str
    title: str = ""
    publisher: str = ""
    date: Optional[str] = None
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_type: Literal["web", "academic", "document", "api"] = "web"
    raw_excerpts: List[str] = Field(default_factory=list)
    provider: str = ""
    accessed_at: Optional[str] = None


class EvidenceCard(BaseModel):
    """A structured claim with supporting evidence."""
    id: str = ""
    claim: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_source_ids: List[str] = Field(default_factory=list)
    conflicting_source_ids: List[str] = Field(default_factory=list)
    exact_excerpts: List[str] = Field(default_factory=list)
    subquestion_id: str = ""
    researcher_id: str = ""
    deduplicated_from: List[str] = Field(default_factory=list)


class ConflictFlag(BaseModel):
    """A detected conflict between evidence cards."""
    card_a_id: str
    card_b_id: str
    conflict_description: str
    severity: Literal["low", "medium", "high"] = "medium"


class ResearchPlan(BaseModel):
    """A structured research plan."""
    objective: str
    subquestions: List[str] = Field(default_factory=list)
    strategy: str = ""
    expected_source_types: List[str] = Field(default_factory=list)
    stop_conditions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class CitationCheck(BaseModel):
    """Result of verifying a citation."""
    claim: str
    url: str
    supports_claim: Optional[bool] = None
    problem: str = ""
    fix: str = ""
    status: Literal["verified", "unverified", "dead", "stale"] = "unverified"


class ReviewResult(BaseModel):
    """Result of a quality review."""
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    style_score: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_flags: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    feedback: str = ""


# ──────────────────────────────────────────────
# Reducers
# ──────────────────────────────────────────────

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


def merge_sources(
    existing: Annotated[List[dict], operator.add],
    new: List[dict],
) -> List[dict]:
    """Reducer: merge sources, dedup by URL, keep highest credibility."""
    seen_urls: dict[str, int] = {}
    result = list(existing)

    for i, src in enumerate(result):
        url = src.get("url", "")
        if url:
            seen_urls[url] = i

    for src in new:
        url = src.get("url", "")
        if url and url in seen_urls:
            idx = seen_urls[url]
            existing_score = result[idx].get("credibility_score", 0.5)
            new_score = src.get("credibility_score", 0.5)
            if new_score > existing_score:
                result[idx] = src
        elif url:
            seen_urls[url] = len(result)
            result.append(src)
        else:
            result.append(src)

    return result


def append_evidence(
    existing: Annotated[List[dict], operator.add],
    new: List[dict],
) -> List[dict]:
    """Reducer: append new evidence cards."""
    return existing + new


# ──────────────────────────────────────────────
# State Definitions
# ──────────────────────────────────────────────

class AgentInputState(MessagesState):
    """InputState is only 'messages'."""


class AgentState(MessagesState):
    """Main agent state containing messages and research data."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    error_artifact: Optional[dict] = None

    # Evidence-first fields (additive, backward-compatible)
    sources: Annotated[List[dict], merge_sources] = []
    evidence_cards: Annotated[List[dict], append_evidence] = []
    conflicts: List[dict] = []  # NOTE: no reducer — overwritten on update (add reducer when multi-node writes needed)
    citation_checks: Annotated[List[dict], operator.add] = []
    review_results: List[dict] = []  # NOTE: no reducer — overwritten on update (add reducer when multi-node writes needed)
    telemetry: dict = {}
    total_tokens: int = 0


class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    error_artifact: Optional[dict] = None

    # Evidence-first fields (additive, backward-compatible)
    sources: Annotated[List[dict], merge_sources] = []
    evidence_cards: Annotated[List[dict], append_evidence] = []
    conflicts: List[dict] = []  # NOTE: no reducer — overwritten on update (add reducer when multi-node writes needed)
    total_tokens: int = 0


class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

    # Evidence-first fields (additive, backward-compatible)
    sources: Annotated[List[dict], merge_sources] = []
    evidence_cards: Annotated[List[dict], append_evidence] = []
    total_tokens: int = 0


class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

    # Evidence-first fields (additive, backward-compatible)
    sources: List[dict] = []
    evidence_cards: List[dict] = []
    total_tokens: int = 0
