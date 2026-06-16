"""Report profile registry — defines report types with sections, tone, and citation style."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class SectionDefinition:
    """Defines a section in a report profile."""

    name: str
    description: str
    required: bool = True
    evidence_allocation: str = "auto"
    prompt_template: str = ""


@dataclass
class ReportProfile:
    """Defines a report type with sections, tone, and citation style."""

    name: str
    description: str
    required_sections: List[SectionDefinition] = field(default_factory=list)
    optional_sections: List[SectionDefinition] = field(default_factory=list)
    citation_style: Literal["vanilla", "apa", "mla", "chicago", "harvard", "ieee"] = "vanilla"
    tone: Literal["formal", "technical", "conversational"] = "formal"
    max_length: int = 10000
    include_appendix: bool = False
    include_source_quality_notes: bool = False
    include_table_of_contents: bool = True


BUILTIN_PROFILES: Dict[str, ReportProfile] = {}


def _register_profiles():
    """Register all built-in report profiles."""
    BUILTIN_PROFILES["executive_brief"] = ReportProfile(
        name="executive_brief",
        description="Concise executive summary with key findings and recommendations",
        required_sections=[
            SectionDefinition("Executive Summary", "High-level overview of findings"),
            SectionDefinition("Key Findings", "3-5 critical findings with evidence"),
            SectionDefinition("Recommendations", "Actionable next steps"),
        ],
        optional_sections=[
            SectionDefinition("Methodology", "Research approach", required=False),
        ],
        citation_style="vanilla",
        tone="formal",
        max_length=3000,
        include_table_of_contents=False,
    )

    BUILTIN_PROFILES["deep_research_report"] = ReportProfile(
        name="deep_research_report",
        description="Comprehensive multi-section research report with full citations",
        required_sections=[
            SectionDefinition("Executive Summary", "Overview of research and findings"),
            SectionDefinition("Background", "Context and research objectives"),
            SectionDefinition("Methodology", "Research approach and sources"),
            SectionDefinition("Findings", "Detailed analysis organized by theme"),
            SectionDefinition("Analysis", "Synthesis of evidence across sources"),
            SectionDefinition("Conclusion", "Summary and implications"),
            SectionDefinition("References", "Full bibliography"),
        ],
        optional_sections=[
            SectionDefinition("Limitations", "Research constraints and caveats", required=False),
            SectionDefinition("Appendix", "Supporting data and tables", required=False),
        ],
        citation_style="apa",
        tone="formal",
        max_length=15000,
        include_appendix=True,
        include_source_quality_notes=True,
    )

    BUILTIN_PROFILES["academic_literature_review"] = ReportProfile(
        name="academic_literature_review",
        description="Systematic literature review with academic citation style",
        required_sections=[
            SectionDefinition("Abstract", "Brief summary of review scope and findings"),
            SectionDefinition("Introduction", "Research question and review objectives"),
            SectionDefinition("Methods", "Search strategy and inclusion criteria"),
            SectionDefinition("Results", "Thematic synthesis of literature"),
            SectionDefinition("Discussion", "Critical analysis and gaps"),
            SectionDefinition("Conclusion", "Summary and future directions"),
            SectionDefinition("References", "Academic bibliography"),
        ],
        citation_style="apa",
        tone="technical",
        max_length=12000,
        include_source_quality_notes=True,
    )

    BUILTIN_PROFILES["investment_memo"] = ReportProfile(
        name="investment_memo",
        description="Investment analysis memo with risk assessment",
        required_sections=[
            SectionDefinition("Thesis", "Investment hypothesis"),
            SectionDefinition("Market Analysis", "Market size, growth, dynamics"),
            SectionDefinition("Competitive Landscape", "Key players and positioning"),
            SectionDefinition("Risk Factors", "Material risks and mitigations"),
            SectionDefinition("Financial Analysis", "Key metrics and projections"),
            SectionDefinition("Recommendation", "Buy/hold/sell with rationale"),
        ],
        citation_style="vanilla",
        tone="formal",
        max_length=8000,
    )

    BUILTIN_PROFILES["competitive_landscape"] = ReportProfile(
        name="competitive_landscape",
        description="Competitive analysis with positioning maps",
        required_sections=[
            SectionDefinition("Market Overview", "Total addressable market"),
            SectionDefinition("Key Players", "Detailed competitor profiles"),
            SectionDefinition("Positioning Analysis", "Feature and market positioning"),
            SectionDefinition("Strengths & Weaknesses", "Comparative analysis"),
            SectionDefinition("Opportunities", "Gaps and white spaces"),
        ],
        citation_style="vanilla",
        tone="technical",
        max_length=10000,
    )

    BUILTIN_PROFILES["technical_design_research"] = ReportProfile(
        name="technical_design_research",
        description="Technical architecture and implementation research",
        required_sections=[
            SectionDefinition("Technical Overview", "System architecture summary"),
            SectionDefinition("Implementation Details", "Code patterns and approaches"),
            SectionDefinition("Performance Considerations", "Benchmarks and tradeoffs"),
            SectionDefinition("Security Analysis", "Threat model and mitigations"),
            SectionDefinition("Recommendations", "Technical decisions and rationale"),
        ],
        citation_style="ieee",
        tone="technical",
        max_length=10000,
    )

    BUILTIN_PROFILES["policy_memo"] = ReportProfile(
        name="policy_memo",
        description="Policy analysis memo with regulatory considerations",
        required_sections=[
            SectionDefinition("Policy Background", "Regulatory context"),
            SectionDefinition("Current Landscape", "Existing policies and enforcement"),
            SectionDefinition("Analysis", "Impact and effectiveness assessment"),
            SectionDefinition("Stakeholder Perspectives", "Industry, government, public views"),
            SectionDefinition("Recommendations", "Policy proposals with rationale"),
        ],
        citation_style="chicago",
        tone="formal",
        max_length=8000,
    )

    BUILTIN_PROFILES["news_brief"] = ReportProfile(
        name="news_brief",
        description="Timely news summary with source diversity",
        required_sections=[
            SectionDefinition("Headline Summary", "Key developments in 2-3 sentences"),
            SectionDefinition("Key Events", "Chronological event listing"),
            SectionDefinition("Context", "Background and significance"),
            SectionDefinition("Outlook", "Expected developments"),
        ],
        citation_style="vanilla",
        tone="conversational",
        max_length=3000,
        include_table_of_contents=False,
    )


_register_profiles()


def get_profile(name: str) -> ReportProfile | None:
    """Get a report profile by name."""
    return BUILTIN_PROFILES.get(name)


def list_profiles() -> List[str]:
    """List all available profile names."""
    return list(BUILTIN_PROFILES.keys())


# Mapping from ResearchMode to default profile
MODE_TO_PROFILE = {
    "comparison": "competitive_landscape",
    "market_landscape": "competitive_landscape",
    "academic_literature_review": "academic_literature_review",
    "company_due_diligence": "investment_memo",
    "technical_implementation": "technical_design_research",
    "news_or_current_events": "news_brief",
    "policy_legal_regulatory": "policy_memo",
    "validation_or_fact_check": "deep_research_report",
    "custom": "deep_research_report",
}
