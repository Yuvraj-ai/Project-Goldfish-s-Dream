# Custom Report Profiles

Control report structure, tone, citation style, and length by creating or selecting report profiles.

---

## What Are Report Profiles?

A report profile is a `ReportProfile` dataclass that defines:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier |
| `description` | `str` | Human-readable summary |
| `required_sections` | `list[SectionDefinition]` | Sections always included |
| `optional_sections` | `list[SectionDefinition]` | Sections included when evidence warrants |
| `citation_style` | `Literal` | `"vanilla"`, `"apa"`, `"mla"`, `"chicago"`, `"harvard"`, `"ieee"` |
| `tone` | `Literal` | `"formal"`, `"technical"`, `"conversational"` |
| `max_length` | `int` | Target character count |
| `include_appendix` | `bool` | Whether to add an appendix |
| `include_source_quality_notes` | `bool` | Whether to note source credibility |
| `include_table_of_contents` | `bool` | Whether to generate a ToC |

Each section is a `SectionDefinition`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Section heading |
| `description` | `str` | — | What the section should cover |
| `required` | `bool` | `True` | Always included vs. conditionally added |
| `evidence_allocation` | `str` | `"auto"` | Evidence budget strategy |
| `prompt_template` | `str` | `""` | Custom prompt for section writing |

---

## Built-In Profiles

| Profile | Sections (req + opt) | Tone | Citation | Max Length |
|---------|---------------------|------|----------|------------|
| `executive_brief` | 3 + 1 | formal | vanilla | 3,000 |
| `deep_research_report` | 7 + 2 | formal | apa | 15,000 |
| `academic_literature_review` | 7 + 0 | technical | apa | 12,000 |
| `investment_memo` | 6 + 0 | formal | vanilla | 8,000 |
| `competitive_landscape` | 5 + 0 | technical | vanilla | 10,000 |
| `technical_design_research` | 5 + 0 | technical | ieee | 10,000 |
| `policy_memo` | 5 + 0 | formal | chicago | 8,000 |
| `news_brief` | 4 + 0 | conversational | vanilla | 3,000 |

### Profile Details

**executive_brief** — Concise executive summary with key findings and recommendations.
- Required: Executive Summary, Key Findings, Recommendations
- Optional: Methodology
- ToC disabled

**deep_research_report** — Comprehensive multi-section research report with full APA citations.
- Required: Executive Summary, Background, Methodology, Findings, Analysis, Conclusion, References
- Optional: Limitations, Appendix
- Appendix and source quality notes enabled

**academic_literature_review** — Systematic literature review with academic APA citations.
- Required: Abstract, Introduction, Methods, Results, Discussion, Conclusion, References
- Source quality notes enabled

**investment_memo** — Investment analysis memo with risk assessment.
- Required: Thesis, Market Analysis, Competitive Landscape, Risk Factors, Financial Analysis, Recommendation

**competitive_landscape** — Competitive analysis with positioning maps.
- Required: Market Overview, Key Players, Positioning Analysis, Strengths & Weaknesses, Opportunities

**technical_design_research** — Technical architecture and implementation research with IEEE citations.
- Required: Technical Overview, Implementation Details, Performance Considerations, Security Analysis, Recommendations

**policy_memo** — Policy analysis memo with Chicago-style citations.
- Required: Policy Background, Current Landscape, Analysis, Stakeholder Perspectives, Recommendations

**news_brief** — Timely news summary with source diversity, conversational tone.
- Required: Headline Summary, Key Events, Context, Outlook
- ToC disabled

---

## Creating Custom Profiles

Define a new `ReportProfile` and register it in `BUILTIN_PROFILES`:

```python
from open_deep_research.report_profiles import (
    BUILTIN_PROFILES,
    ReportProfile,
    SectionDefinition,
)

BUILTIN_PROFILES["security_audit"] = ReportProfile(
    name="security_audit",
    description="Security audit report with vulnerability assessment",
    required_sections=[
        SectionDefinition(
            "Scope",
            "Systems, networks, and applications in scope"
        ),
        SectionDefinition(
            "Findings Summary",
            "Critical, high, and medium findings"
        ),
        SectionDefinition(
            "Vulnerability Details",
            "Per-vulnerability analysis with CVSS scores",
            evidence_allocation="high",
        ),
        SectionDefinition(
            "Remediation Plan",
            "Prioritized actions with timelines"
        ),
    ],
    optional_sections=[
        SectionDefinition(
            "Compliance Mapping",
            "Findings mapped to regulatory frameworks",
            required=False,
        ),
    ],
    citation_style="vanilla",
    tone="technical",
    max_length=12000,
)
```

Place this in your application's startup code or a plugin entry point **after** the `open_deep_research` package is imported but **before** report generation begins.

### Using Your Profile

```python
# Via config override
config = Configuration(report_profile_override="security_audit")

# Or resolve it programmatically
from open_deep_research.report_profiles import BUILTIN_PROFILES
profile = BUILTIN_PROFILES["security_audit"]
```

---

## Mode-to-Profile Mapping

The `MODE_TO_PROFILE` dict in `report_profiles.py` maps each `ResearchMode` to a default profile:

| Research Mode | Default Profile |
|---------------|----------------|
| `comparison` | `competitive_landscape` |
| `market_landscape` | `competitive_landscape` |
| `academic_literature_review` | `academic_literature_review` |
| `company_due_diligence` | `investment_memo` |
| `technical_implementation` | `technical_design_research` |
| `news_or_current_events` | `news_brief` |
| `policy_legal_regulatory` | `policy_memo` |
| `validation_or_fact_check` | `deep_research_report` |
| `custom` | `deep_research_report` |

The resolution order in `_resolve_profile_name()` is:

1. `config.report_profile_override` — if set, use it unconditionally
2. `state.research_mode` looked up in `MODE_TO_PROFILE` — if found, use it
3. Fall back to `"deep_research_report"`

---

## Runtime Override

Set `report_profile_override` in your `Configuration` to bypass mode-based profile selection:

```python
config = Configuration(
    report_profile_override="executive_brief",
)
```

This takes **highest priority** — it overrides whatever `ResearchMode` the graph resolves.

---

## Extending Via Startup Code

Since `BUILTIN_PROFILES` is a module-level `dict`, you can add profiles at application startup:

```python
# my_app/startup.py
from open_deep_research.report_profiles import BUILTIN_PROFILES, ReportProfile, SectionDefinition

def register_my_profiles():
    BUILTIN_PROFILES["my_custom_profile"] = ReportProfile(
        name="my_custom_profile",
        description="My custom report format",
        required_sections=[
            SectionDefinition("Introduction", "Background and context"),
            SectionDefinition("Deep Dive", "Detailed analysis"),
            SectionDefinition("Conclusions", "Key takeaways"),
        ],
        citation_style="harvard",
        tone="formal",
        max_length=5000,
    )
```

Call `register_my_profiles()` during your application's initialization, before invoking the research graph. This pattern works with the existing plugin loader — place the registration call in your plugin module's top-level scope:

```python
# plugins/my_profile_plugin.py
from open_deep_research.report_profiles import BUILTIN_PROFILES, ReportProfile, SectionDefinition

BUILTIN_PROFILES["my_profile"] = ReportProfile(
    name="my_profile",
    description="Loaded via plugin system",
    required_sections=[
        SectionDefinition("Overview", "Summary of findings"),
    ],
    citation_style="vanilla",
    tone="formal",
    max_length=5000,
)
```

The registration happens at import time — when `PluginLoader` imports the module, the profile is automatically added to `BUILTIN_PROFILES`.

---

## Source Reference

- Dataclass definitions: `src/open_deep_research/report_profiles.py`
- Profile resolution: `deep_researcher.py:_resolve_profile_name()` (line ~1068)
- Config field: `configuration.py:Configuration.report_profile_override` (line 473)
