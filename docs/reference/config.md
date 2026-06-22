# Configuration Reference

All configuration fields for the Open Deep Research system, organized by category.

## How Configuration Works

Configuration is loaded from three sources (in priority order):

1. **Environment variables** — uppercase field name (e.g. `MAX_TOTAL_TOKENS=100000`)
2. **`RunnableConfig.configurable` dict** — passed at graph invocation
3. **Default values** — defined in the `Configuration` Pydantic model

All fields in the `Configuration` model can be overridden via environment variables using their uppercased name.

---

## General

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_structured_output_retries` | `int` | `3` | Max retries for structured output calls from models (range: 1–10) |
| `allow_clarification` | `bool` | `True` | Allow the researcher to ask clarifying questions before starting |
| `max_concurrent_research_units` | `int` | `5` | Max concurrent research sub-agents (range: 1–20, higher may hit rate limits) |

---

## Research

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `search_api` | `SearchAPI` enum | `tavily` | Search API provider. Options: `tavily`, `openai`, `anthropic`, `none` |
| `max_researcher_iterations` | `int` | `6` | Max research iterations for the Research Supervisor (range: 1–10) |
| `max_react_tool_calls` | `int` | `10` | Max tool-calling iterations per researcher step (range: 1–30) |

### `SearchAPI` enum values

| Value | Label |
|-------|-------|
| `tavily` | Tavily |
| `openai` | OpenAI Native Web Search |
| `anthropic` | Anthropic Native Web Search |
| `none` | None |

---

## Model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `summarization_model` | `str` | `openai:gpt-4.1-mini` | Model for summarizing Tavily search results |
| `summarization_model_max_tokens` | `int` | `8192` | Max output tokens for summarization model |
| `max_content_length` | `int` | `50000` | Max character length for webpage content before summarization (range: 1000–200000) |
| `research_model` | `str` | `openai:gpt-4.1` | Model for conducting research |
| `research_model_max_tokens` | `int` | `10000` | Max output tokens for research model |
| `compression_model` | `str` | `openai:gpt-4.1` | Model for compressing research findings from sub-agents |
| `compression_model_max_tokens` | `int` | `8192` | Max output tokens for compression model |
| `final_report_model` | `str` | `openai:gpt-4.1` | Model for writing the final report |
| `final_report_model_max_tokens` | `int` | `10000` | Max output tokens for final report model |

Model strings use the format `provider:model_name` (e.g. `openai:gpt-4.1`, `anthropic:claude-sonnet-4-20250514`).

---

## MCP (Model Context Protocol)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcp_config` | `MCPConfig \| None` | `None` | MCP server configuration (URL, tools, auth) |
| `mcp_prompt` | `str \| None` | `None` | Additional instructions regarding available MCP tools |

### `MCPConfig` sub-fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str \| None` | `None` | URL of the MCP server |
| `tools` | `list[str] \| None` | `None` | Tools to make available to the LLM |
| `auth_required` | `bool \| None` | `False` | Whether the MCP server requires authentication |

---

## Research Mode Classifier

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_research_mode` | `ResearchMode` enum | `custom` | Default mode when classification is disabled or uncertain |
| `enable_mode_classification` | `bool` | `True` | Whether to classify research queries into adaptive modes |
| `classifier_model` | `str \| None` | `None` | Model for classifying research queries. Falls back to `research_model` if unset |
| `classifier_confidence_threshold` | `float` | `0.6` | Minimum confidence to accept classifier mode. Below this, falls back to `custom` (range: 0.0–1.0) |

### `ResearchMode` enum values

| Value | Description |
|-------|-------------|
| `comparison` | Compare/contrast multiple items |
| `validation_or_fact_check` | Verify claims or assertions |
| `market_landscape` | Market overview and competitive analysis |
| `academic_literature_review` | Scholarly literature review |
| `company_due_diligence` | Company background and financials |
| `technical_implementation` | Technical how-to and implementation details |
| `news_or_current_events` | Recent news and timely information |
| `policy_legal_regulatory` | Policy, legal, and regulatory analysis |
| `custom` | Default/general purpose mode |

---

## Research Planning

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `plan_review_mode` | `Literal["none", "interrupt", "auto_review"]` | `none` | How to handle research plan review |
| `max_subquestions` | `int` | `7` | Max subquestions in a research plan (range: 1–15) |
| `min_subquestions` | `int` | `3` | Min subquestions in a research plan (range: 1–10) |
| `max_plan_revisions` | `int` | `2` | Max plan revisions before proceeding (range: 0–5) |

### `plan_review_mode` options

| Value | Behavior |
|-------|----------|
| `none` | No review — proceed directly |
| `interrupt` | Human-in-the-loop — pause for approval |
| `auto_review` | LLM auto-review of the plan |

---

## Search Aggregation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_search_aggregation` | `bool` | `False` | Enable multi-provider search aggregation |
| `search_providers` | `list[str]` | `["tavily"]` | Comma-separated list of search providers |
| `max_search_results` | `int` | `20` | Max results from search aggregator |

---

## Academic Search

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_academic_search` | `bool` | `True` | Enable academic database search (arXiv, Semantic Scholar, PubMed, Crossref) |
| `arxiv_enabled` | `bool` | `True` | Enable arXiv as an academic source |
| `semantic_scholar_enabled` | `bool` | `True` | Enable Semantic Scholar as an academic source |
| `pubmed_enabled` | `bool` | `True` | Enable PubMed as an academic source |
| `crossref_enabled` | `bool` | `True` | Enable Crossref as an academic source |

---

## Source Diversity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_unique_domains` | `int` | `3` | Minimum unique domains required for source diversity |
| `max_same_domain_ratio` | `float` | `0.4` | Maximum ratio of results from a single domain (0.0–1.0) |
| `max_age_days_latest_mode` | `int` | `30` | Max age in days for sources in `news_or_current_events` mode |

---

## Document Reading

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_document_reading` | `bool` | `True` | Enable document upload and parsing |
| `max_document_pages` | `int` | `500` | Max pages to read from uploaded documents |

---

## STORM Multi-Perspective Research

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_storm_research` | `bool` | `False` | Enable STORM-style multi-perspective research |
| `max_perspectives` | `int` | `7` | Max perspectives to generate |
| `max_storm_parallel_researchers` | `int` | `15` | Max parallel researchers in STORM mode |

---

## Citation Verification

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_citation_verification` | `bool` | `True` | Enable citation verification before report generation |

---

## Section Writers

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_section_writers` | `bool` | `True` | Enable section-level report generation |

---

## Reviewer Loop

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_reviewer_loop` | `bool` | `True` | Enable reviewer agent loop for quality assurance |
| `max_review_iterations` | `int` | `2` | Max review iterations before finalizing |
| `review_score_threshold` | `float` | `0.7` | Quality score threshold to pass review (0.0–1.0) |

---

## Citation Style

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `citation_style` | `str` | `"vanilla"` | Citation style for the report |

---

## Export

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `export_formats` | `list[str]` | `["markdown"]` | Export formats to generate |

---

## Report Profile

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `report_profile_override` | `str \| None` | `None` | Override the default report profile |

---

## Feature Flags

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_evidence_first` | `bool` | `False` | Enable evidence-first architecture with structured EvidenceCards. When disabled, falls back to flat text compression |
| `plugin_directories` | `list[str]` | `[]` | Directories to scan for custom source plugins |
| `enable_model_routing` | `bool` | `False` | Enable model routing for cost optimization. ModelRouter selects the appropriate model tier per task |

---

## Budget

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_total_tokens` | `int` | `500000` | Max total tokens allowed per research session (`0` = unlimited) |
| `max_cost_usd` | `float` | `5.0` | Max cost in USD allowed per research session (`0` = unlimited) |

---

## API Configuration (`ApiConfig`)

Defined in `src/open_deep_research/api/config.py`. Controls the REST API server.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_rest_api` | `bool` | `False` | Enable the REST API server (alternative to env var `ENABLE_REST_API=true`) |
| `api_host` | `str` | `"0.0.0.0"` | Host to bind the API server to |
| `api_port` | `int` | `8000` | Port to bind the API server to |
| `api_db_path` | `str` | `"research.db"` | SQLite database path for research run persistence |
| `api_key` | `str \| None` | `None` | API key for authenticating requests |
| `max_concurrent_runs` | `int` | `3` | Max concurrent research runs via the API |
| `api_keys` | `dict[str, dict[str, str \| list[str]]]` | `{}` | API key configuration map for multi-key auth |

### API environment variables

| Variable | Maps to | Description |
|----------|---------|-------------|
| `ENABLE_REST_API` | `enable_rest_api` | Set to `true` to enable the API server |
| `API_KEY` | `api_key` | Authentication key for API requests |

---

## Enum Reference

### `SearchAPI`

| Python value | String value |
|-------------|--------------|
| `SearchAPI.TAVILY` | `"tavily"` |
| `SearchAPI.OPENAI` | `"openai"` |
| `SearchAPI.ANTHROPIC` | `"anthropic"` |
| `SearchAPI.NONE` | `"none"` |

### `ResearchMode`

| Python value | String value |
|-------------|--------------|
| `ResearchMode.COMPARISON` | `"comparison"` |
| `ResearchMode.VALIDATION_OR_FACT_CHECK` | `"validation_or_fact_check"` |
| `ResearchMode.MARKET_LANDSCAPE` | `"market_landscape"` |
| `ResearchMode.ACADEMIC_LITERATURE_REVIEW` | `"academic_literature_review"` |
| `ResearchMode.COMPANY_DUE_DILIGENCE` | `"company_due_diligence"` |
| `ResearchMode.TECHNICAL_IMPLEMENTATION` | `"technical_implementation"` |
| `ResearchMode.NEWS_OR_CURRENT_EVENTS` | `"news_or_current_events"` |
| `ResearchMode.POLICY_LEGAL_REGULATORY` | `"policy_legal_regulatory"` |
| `ResearchMode.CUSTOM` | `"custom"` |
