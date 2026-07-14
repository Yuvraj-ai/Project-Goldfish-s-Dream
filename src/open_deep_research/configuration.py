"""Configuration management for the Open Deep Research system."""

import copy
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, SecretStr


class ResearchMode(str, Enum):
    """Research modes for adaptive strategy selection."""
    COMPARISON = "comparison"
    VALIDATION_OR_FACT_CHECK = "validation_or_fact_check"
    MARKET_LANDSCAPE = "market_landscape"
    ACADEMIC_LITERATURE_REVIEW = "academic_literature_review"
    COMPANY_DUE_DILIGENCE = "company_due_diligence"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    NEWS_OR_CURRENT_EVENTS = "news_or_current_events"
    POLICY_LEGAL_REGULATORY = "policy_legal_regulatory"
    CUSTOM = "custom"


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    base_url: str = ""
    default_model: str = ""
    allowed_models: list[str] = []
    model_token_limits: dict[str, int] = {}
    aliases: list[str] = []
    api_key_env: str = ""           # env var name for this provider's key
    auth_strategy: Literal["api_key", "aws", "none"] = "api_key"
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None


class ModelsConfig(BaseModel):
    """Model slot assignments."""
    research_model: str = "openai:gpt-4.1"
    summarization_model: str = "openai:gpt-4.1-mini"
    compression_model: str = "openai:gpt-4.1"
    final_report_model: str = "openai:gpt-4.1"
    classifier_model: str | None = None


class ResolvedModel:
    """Result of model resolution — carries everything needed to create a model client."""
    __slots__ = (
        "provider", "model_string", "canonical_model_string",
        "model_name", "base_url", "api_key", "token_limit", "provider_kwargs",
    )

    def __init__(
        self,
        provider: str,
        model_string: str,
        canonical_model_string: str,
        model_name: str,
        base_url: str,
        api_key: str,
        token_limit: int | None,
        provider_kwargs: dict,
    ):
        """Initialize ResolvedModel with provider connection details."""
        self.provider = provider
        self.model_string = model_string
        self.canonical_model_string = canonical_model_string
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.token_limit = token_limit
        self.provider_kwargs = provider_kwargs

    def __repr__(self) -> str:
        """Return concise string representation with key fields."""
        return (
            f"ResolvedModel(provider={self.provider!r}, "
            f"model_name={self.model_name!r}, base_url={self.base_url!r})"
        )


BUILTIN_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        allowed_models=[
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini",
            "o4-mini", "o3-mini", "o3", "o3-pro", "o1", "o1-pro",
            "qwen2.5", "qwen3.6",
        ],
        model_token_limits={
            "gpt-4.1": 1047576, "gpt-4.1-mini": 1047576, "gpt-4.1-nano": 1047576,
            "gpt-4o": 128000, "gpt-4o-mini": 128000,
            "o4-mini": 200000, "o3-mini": 200000, "o3": 200000,
            "o3-pro": 200000, "o1": 200000, "o1-pro": 200000,
            "qwen2.5": 32768, "qwen3.6": 32768,
        },
    ),
    "anthropic": ProviderConfig(
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        allowed_models=[
            "claude-opus-4", "claude-sonnet-4-20250514",
            "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku",
        ],
        model_token_limits={
            "claude-opus-4": 200000, "claude-sonnet-4-20250514": 200000,
            "claude-3-7-sonnet": 200000, "claude-3-5-sonnet": 200000,
            "claude-3-5-haiku": 200000,
        },
    ),
    "google_genai": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.5-flash",
        aliases=["google"],
        api_key_env="GOOGLE_API_KEY",
        allowed_models=[
            "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-1.5-pro", "gemini-1.5-flash",
        ],
        model_token_limits={
            "gemini-2.5-pro": 1048576, "gemini-2.5-flash": 1048576,
            "gemini-1.5-pro": 2097152, "gemini-1.5-flash": 1048576,
        },
    ),
    "bedrock": ProviderConfig(
        base_url="",
        default_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        api_key_env="AWS_ACCESS_KEY_ID",
        auth_strategy="aws",
        allowed_models=[
            "us.amazon.nova-premier-v1:0", "us.amazon.nova-pro-v1:0",
            "us.amazon.nova-lite-v1:0", "us.amazon.nova-micro-v1:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0",
        ],
        model_token_limits={
            "us.amazon.nova-premier-v1:0": 1000000,
            "us.amazon.nova-pro-v1:0": 300000,
            "us.amazon.nova-lite-v1:0": 300000,
            "us.amazon.nova-micro-v1:0": 128000,
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
            "us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
            "us.anthropic.claude-opus-4-20250514-v1:0": 200000,
        },
    ),
    "cohere": ProviderConfig(
        base_url="https://api.cohere.com/v1",
        default_model="command-r-plus",
        api_key_env="COHERE_API_KEY",
        allowed_models=["command-r-plus", "command-r", "command-light", "command"],
        model_token_limits={
            "command-r-plus": 128000, "command-r": 128000,
            "command-light": 4096, "command": 4096,
        },
    ),
    "mistral": ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large",
        api_key_env="MISTRAL_API_KEY",
        allowed_models=["mistral-large", "mistral-medium", "mistral-small", "mistral-7b-instruct"],
        model_token_limits={
            "mistral-large": 32768, "mistral-medium": 32768,
            "mistral-small": 32768, "mistral-7b-instruct": 32768,
        },
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        default_model="llama2",
        api_key_env="",
        auth_strategy="none",
        allowed_models=["codellama", "llama2:70b", "llama2:13b", "llama2", "mistral"],
        model_token_limits={
            "codellama": 16384, "llama2:70b": 4096,
            "llama2:13b": 4096, "llama2": 4096, "mistral": 32768,
        },
    ),
}


def build_provider_registry(
    config_json: dict[str, Any],
    builtin: dict[str, ProviderConfig] | None = None,
) -> dict[str, ProviderConfig]:
    """Merge config.json providers into built-in registry.

    Rules:
    - Unknown providers are added to the registry.
    - Known providers are deep-merged: scalar fields override, lists are unioned
      (allowed_models) or replaced (model_token_limits).
    - Built-in entries not present in config.json are preserved unchanged.
    """
    registry = copy.deepcopy(builtin or BUILTIN_PROVIDERS)

    for provider_id, overrides in config_json.get("providers", {}).items():
        if provider_id not in registry:
            registry[provider_id] = ProviderConfig(**overrides)
        else:
            existing = registry[provider_id]
            merged = existing.model_dump()

            for key, value in overrides.items():
                if key == "allowed_models" and isinstance(value, list):
                    merged["allowed_models"] = list(
                        dict.fromkeys(merged.get("allowed_models", []) + value)
                    )
                elif key == "model_token_limits" and isinstance(value, dict):
                    merged["model_token_limits"] = {
                        **merged.get("model_token_limits", {}),
                        **value,
                    }
                elif key == "aliases" and isinstance(value, list):
                    merged["aliases"] = list(
                        dict.fromkeys(merged.get("aliases", []) + value)
                    )
                else:
                    merged[key] = value

            registry[provider_id] = ProviderConfig(**merged)

    return registry


def _load_config_json() -> dict:
    """Load config.json with deterministic path resolution.

    Resolution order:
    1. ODR_CONFIG_FILE env var (explicit path)
    2. Package-relative: <package_root>/config.json (source checkout)
    3. Empty dict (no config.json = use built-in defaults)

    If ODR_CONFIG_FILE is set but malformed, raise an error (not silent fallback).
    """
    env_path = os.getenv("ODR_CONFIG_FILE")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ODR_CONFIG_FILE={env_path} specified but file not found"
            )
        return json.loads(path.read_text())

    package_root = Path(__file__).resolve().parent.parent.parent
    path = package_root / "config.json"
    if path.exists():
        return json.loads(path.read_text())

    return {}


def _load_dotenv() -> dict:
    """Load .env file values as a flat dict."""
    try:
        from dotenv import dotenv_values
        return dotenv_values() or {}
    except ImportError:
        return {}


_SECRET_KEY_EXACT = {
    "openai_api_key", "anthropic_api_key", "google_api_key",
    "tavily_api_key", "cohere_api_key", "mistral_api_key",
    "ollama_api_key", "bedrock_api_key", "bedrock_secret_key",
    "bedrock_session_token",
}
_SECRET_KEY_SUFFIXES = ("_api_key", "_secret", "_secret_key", "_session_token")
_SECRET_KEY_EXACT_LOWER = {k.lower() for k in _SECRET_KEY_EXACT}


def _is_secret_key(key: str) -> bool:
    """Check if a key name refers to a secret field."""
    low = key.lower()
    if low in _SECRET_KEY_EXACT_LOWER:
        return True
    if low in _SECRET_KEY_EXACT:
        return True
    return any(low.endswith(s) for s in _SECRET_KEY_SUFFIXES)


def redact_secrets(config: dict) -> dict:
    """Redact secret fields from config before storage.

    Recurses into nested dicts (including apiKeys) and lists of dicts.
    Only matches known secret key patterns — does not touch fields like
    max_total_tokens or research_model_max_tokens.
    """
    redacted = {}
    for key, value in config.items():
        if _is_secret_key(key):
            redacted[key] = "***REDACTED***" if value else None
        elif isinstance(value, dict):
            redacted[key] = redact_secrets(value)
        elif isinstance(value, list):
            redacted[key] = [redact_secrets(v) if isinstance(v, dict) else v for v in value]
        else:
            redacted[key] = value
    return redacted


_FLAT_TO_NESTED = {
    "research_model": ("models", "research_model"),
    "summarization_model": ("models", "summarization_model"),
    "compression_model": ("models", "compression_model"),
    "final_report_model": ("models", "final_report_model"),
    "classifier_model": ("models", "classifier_model"),
}


def _migrate_flat_keys(configurable: dict) -> dict:
    """Migrate flat model keys to nested models section."""
    migrated = dict(configurable)
    for flat_key, (section, field) in _FLAT_TO_NESTED.items():
        if flat_key in migrated:
            migrated.setdefault(section, {})[field] = migrated.pop(flat_key)
    return migrated


class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: str | None = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: List[str] | None = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: bool | None = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # Provider registry (merged from BUILTIN_PROVIDERS + config.json + runtime)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    # Model slot assignments (nested config for model routing)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: MCPConfig | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    # Research mode classifier
    default_research_mode: ResearchMode = Field(
        default=ResearchMode.CUSTOM,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "custom",
                "description": "Default research mode when classification is disabled or uncertain.",
                "options": [
                    {"label": m.value, "value": m.value}
                    for m in ResearchMode
                ]
            }
        }
    )
    enable_mode_classification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to classify research queries into adaptive modes."
            }
        }
    )
    classifier_model: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Model for classifying research queries. Defaults to research_model if not set."
            }
        }
    )
    classifier_confidence_threshold: float = Field(
        default=0.6,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.6,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence to accept classifier mode. Below this, falls back to CUSTOM."
            }
        }
    )

    # Research planning
    plan_review_mode: Literal["none", "interrupt", "auto_review"] = Field(
        default="none",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "none",
                "description": "How to handle research plan review.",
                "options": [
                    {"label": "None", "value": "none"},
                    {"label": "Interrupt (HITL)", "value": "interrupt"},
                    {"label": "Auto Review", "value": "auto_review"},
                ]
            }
        }
    )
    max_subquestions: int = Field(
        default=7,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 7,
                "min": 1,
                "max": 15,
                "description": "Maximum number of subquestions in a research plan."
            }
        }
    )
    min_subquestions: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Minimum number of subquestions in a research plan."
            }
        }
    )
    max_plan_revisions: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2,
                "min": 0,
                "max": 5,
                "description": "Maximum plan revisions before proceeding."
            }
        }
    )

    # Search aggregation
    enable_search_aggregation: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable multi-provider search aggregation."
            }
        }
    )
    search_providers: List[str] = Field(
        default=["tavily"],
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Comma-separated list of search providers to use."
            }
        }
    )
    max_search_results: int = Field(
        default=20,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 20,
                "description": "Maximum results from search aggregator."
            }
        }
    )

    # Academic search
    enable_academic_search: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable academic database search (arXiv, Semantic Scholar, PubMed, Crossref)."
            }
        }
    )
    arxiv_enabled: bool = Field(default=True)
    semantic_scholar_enabled: bool = Field(default=True)
    pubmed_enabled: bool = Field(default=True)
    crossref_enabled: bool = Field(default=True)

    # Source diversity
    min_unique_domains: int = Field(default=3)
    max_same_domain_ratio: float = Field(default=0.4)
    max_age_days_latest_mode: int = Field(default=30)

    # Document reading
    enable_document_reading: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable document upload and parsing."
            }
        }
    )
    max_document_pages: int = Field(default=500)

    # STORM multi-perspective
    enable_storm_research: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable STORM-style multi-perspective research."
            }
        }
    )
    max_perspectives: int = Field(default=7)
    max_storm_parallel_researchers: int = Field(default=15)

    # Phase 3: Citation verification
    enable_citation_verification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable citation verification before report generation."
            }
        }
    )

    # Phase 3: Section writers
    enable_section_writers: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable section-level report generation."
            }
        }
    )

    # Phase 3: Reviewer loop
    enable_reviewer_loop: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable reviewer agent loop for quality assurance."
            }
        }
    )
    max_review_iterations: int = Field(default=2)
    review_score_threshold: float = Field(default=0.7)

    # Phase 3: Citation style
    citation_style: str = Field(default="vanilla")

    # Phase 3: Export
    export_formats: List[str] = Field(default=["markdown"])

    # Phase 3: Report profile
    report_profile_override: str | None = Field(default=None)

    # Feature Flags
    enable_evidence_first: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable evidence-first architecture with structured EvidenceCards. When disabled, falls back to flat text compression."
            }
        }
    )
    # Plugin directories
    plugin_directories: list[str] = Field(
        default=[],
        description="Directories to scan for custom source plugins",
    )
    enable_model_routing: bool = Field(
        default=False,
        description="Enable model routing for cost optimization. When enabled, "
                    "the ModelRouter selects the appropriate model tier per task.",
    )

    # Budget Configuration
    max_total_tokens: int = Field(
        default=500000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 500000,
                "description": "Maximum total tokens allowed per research session (0 = unlimited)"
            }
        }
    )
    max_cost_usd: float = Field(
        default=5.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 5.0,
                "description": "Maximum cost in USD allowed per research session (0 = unlimited)"
            }
        }
    )

    @classmethod
    def _build_values(cls, config: RunnableConfig | None = None) -> dict[str, Any]:
        """Build final config values with canonical precedence.

        Priority (highest to lowest):
        4. configurable dict (per-request / eval script)
        3. OS environment variables
        2. .env file
        1. config.json
        0. Built-in defaults (BUILTIN_PROVIDERS, field defaults)
        """
        configurable = config.get("configurable", {}) if config else {}

        values: dict[str, Any] = {}

        # 1. Layer config.json
        json_config = _load_config_json()
        for key in cls.model_fields:
            if key in json_config:
                values[key] = json_config[key]

        # 2. Layer .env file
        dotenv_values_dict = _load_dotenv()
        for key in cls.model_fields:
            if key in dotenv_values_dict:
                values[key] = dotenv_values_dict[key]

        # 3. Layer OS environment (wins over .env)
        for key in cls.model_fields:
            env_val = os.environ.get(key.upper())
            if env_val is not None:
                values[key] = env_val

        # 4. Layer configurable dict (wins over everything)
        # Migrate flat model keys to nested models section
        migrated = _migrate_flat_keys(configurable)
        for key, val in migrated.items():
            if key in cls.model_fields:
                values[key] = val
            elif isinstance(val, dict):
                # Nested dict (e.g., models) — merge into existing
                if key in values and isinstance(values[key], dict):
                    values[key].update(val)
                else:
                    values[key] = val

        return values

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        values = cls._build_values(config)

        # Merge providers: BUILTIN < config.json < configurable (runtime)
        json_providers = values.pop("providers", {})
        configurable = config.get("configurable", {}) if config else {}
        runtime_providers = configurable.get("providers", {})

        registry = build_provider_registry({"providers": json_providers})
        if runtime_providers:
            registry = build_provider_registry({"providers": runtime_providers}, registry)

        values["providers"] = registry

        return cls(**{k: v for k, v in values.items() if v is not None})

    def _resolve_provider_alias(self, provider: str) -> str:
        """Resolve provider alias (e.g., 'google' -> 'google_genai')."""
        for pid, pc in self.providers.items():
            if provider == pid:
                return pid
            if provider in pc.aliases:
                return pid
        return provider

    def resolve_model(
        self,
        model_string: str,
        runnable_config: RunnableConfig | None = None,
    ) -> ResolvedModel:
        """Resolve 'openai:gpt-4.1' -> ResolvedModel."""
        provider, model_name = model_string.split(":", 1)
        canonical_provider = self._resolve_provider_alias(provider)

        provider_config = self.providers.get(canonical_provider)
        if not provider_config:
            available = list(self.providers.keys())
            raise ValueError(
                f"Unknown provider '{provider}' (canonical: '{canonical_provider}'). "
                f"Available: {available}."
            )

        if provider_config.allowed_models and model_name not in provider_config.allowed_models:
            raise ValueError(
                f"Model '{model_name}' not allowed for {canonical_provider}. "
                f"Allowed: {provider_config.allowed_models}"
            )

        api_key = self._resolve_api_key(canonical_provider, provider_config, runnable_config)
        token_limit = provider_config.model_token_limits.get(model_name)

        return ResolvedModel(
            provider=canonical_provider,
            model_string=model_string,
            canonical_model_string=f"{canonical_provider}:{model_name}",
            model_name=model_name,
            base_url=provider_config.base_url,
            api_key=api_key,
            token_limit=token_limit,
            provider_kwargs={},
        )

    def _resolve_api_key(
        self,
        provider: str,
        provider_config: ProviderConfig,
        runnable_config: RunnableConfig | None = None,
    ) -> str:
        """Resolve API key using provider's api_key_env, not derived from provider ID."""
        if provider_config.auth_strategy == "none":
            return ""

        if runnable_config and os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() == "true":
            api_keys = runnable_config.get("configurable", {}).get("apiKeys", {})
            if provider_config.api_key_env and api_keys.get(provider_config.api_key_env):
                return api_keys[provider_config.api_key_env]
            if api_keys.get(f"{provider}_api_key"):
                return api_keys[f"{provider}_api_key"]

        attr_name = f"{provider}_api_key"
        api_key_secret: SecretStr | None = getattr(self, attr_name, None)
        if api_key_secret:
            return api_key_secret.get_secret_value()

        if provider_config.api_key_env:
            env_val = os.getenv(provider_config.api_key_env)
            if env_val:
                return env_val

        if provider_config.auth_strategy == "aws":
            return ""

        raise ValueError(
            f"No API key for provider '{provider}'. "
            f"Set {provider_config.api_key_env} in .env or provide via config.apiKeys."
        )

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True


def build_model_config(
    configurable: Configuration,
    model_string: str,
    max_tokens: int,
    runnable_config: RunnableConfig | None = None,
) -> dict:
    """Build a model config dict for init_chat_model() from a ResolvedModel."""
    resolved = configurable.resolve_model(model_string, runnable_config)
    config = {
        "model": resolved.canonical_model_string,
        "max_tokens": max_tokens,
        "api_key": resolved.api_key,
        "tags": ["langsmith:nostream"],
    }
    if resolved.base_url:
        config["base_url"] = resolved.base_url
    config.update(resolved.provider_kwargs)
    return config