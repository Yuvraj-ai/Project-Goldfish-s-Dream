# Config System Redesign: Provider-Aware Model & Key Management

**Date:** 2026-07-08
**Status:** Approved
**Author:** OpenCode (generated from brainstorming session)

---

## Problem Statement

The current model/key configuration system has four issues:

1. **No provider-level control**: API keys are resolved by prefix matching (`openai:*` → `OPENAI_API_KEY`), but there's no way to configure base URLs, allowed models, or token limits per provider.
2. **Hardcoded token limits**: `MODEL_TOKEN_LIMITS` in `utils.py:829-872` is a manually maintained dict that must be updated for every new model.
3. **No model validation**: Any `provider:model` string is accepted — there's no check against an allowed list.
4. **Flat .env-only config**: All configuration lives in flat key-value pairs, making structured settings (provider → models → limits) impossible without ugly prefixes.

## Approach: Hybrid .env + config.json + pydantic-settings

- **`.env`** (gitignored): API keys only
- **`config.json`** (committed): Provider definitions, model assignments, defaults
- **`pydantic-settings`**: Unified loading with type validation and env var overrides

### Why this approach?

| Alternative | Why rejected |
|-------------|-------------|
| Pure `.env` | Can't express nested structure without prefixes like `PROVIDER_OPENAI__DEFAULT_MODEL` |
| Pure `config.json` | API keys would be committed to git (security violation) |
| Pure `config.yaml` | Adds PyYAML dependency for minimal benefit over JSON |
| Dynaconf | Adds a dependency; pydantic-settings is already in the LangChain ecosystem |

---

## Design

### 1. Config File Structure

**`config.json`** (committed to git):

```json
{
  "providers": {
    "openai": {
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-4.1",
      "allowed_models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o4-mini"],
      "model_token_limits": {
        "gpt-4.1": 1047576,
        "gpt-4.1-mini": 1047576,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "o4-mini": 200000
      }
    },
    "anthropic": {
      "base_url": "https://api.anthropic.com",
      "default_model": "claude-sonnet-4-20250514",
      "allowed_models": ["claude-opus-4", "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
      "model_token_limits": {
        "claude-opus-4": 200000,
        "claude-sonnet-4-20250514": 200000,
        "claude-3-5-haiku-20241022": 200000
      }
    },
    "google": {
      "base_url": "https://generativelanguage.googleapis.com/v1beta",
      "default_model": "gemini-2.5-flash",
      "allowed_models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
      "model_token_limits": {
        "gemini-2.5-pro": 1048576,
        "gemini-2.5-flash": 1048576,
        "gemini-1.5-pro": 2097152
      }
    }
  },
  "models": {
    "research_model": "openai:gpt-4.1",
    "summarization_model": "openai:gpt-4.1-mini",
    "compression_model": "openai:gpt-4.1",
    "final_report_model": "openai:gpt-4.1",
    "classifier_model": null
  },
  "defaults": {
    "max_structured_output_retries": 3,
    "max_researcher_iterations": 6,
    "max_concurrent_research_units": 5,
    "search_api": "tavily"
  }
}
```

**`.env`** (gitignored, secrets only):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
TAVILY_API_KEY=tvly-dev-...
```

### 2. Pydantic Model Design

```python
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    base_url: str
    default_model: str
    allowed_models: list[str] = []
    model_token_limits: dict[str, int] = {}


class ModelsConfig(BaseModel):
    """Which model to use for each task slot."""
    research_model: str = "openai:gpt-4.1"
    summarization_model: str = "openai:gpt-4.1-mini"
    compression_model: str = "openai:gpt-4.1"
    final_report_model: str = "openai:gpt-4.1"
    classifier_model: str | None = None


class Configuration(BaseSettings):
    """Main configuration — loads from .env + config.json with env var overrides."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # config.json is loaded via a custom settings source (init_json_source)
    # that reads config.json at startup and merges it into the settings.
    # Env vars override config.json values (correct precedence).

    # Secrets (from .env)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    # Structured config (from config.json, overridable by env vars)
    providers: dict[str, ProviderConfig] = {}
    models: ModelsConfig = ModelsConfig()

    # Existing fields (kept for backward compat)
    max_structured_output_retries: int = 3
    allow_clarification: bool = True
    max_concurrent_research_units: int = 5
    max_researcher_iterations: int = 6
    max_react_tool_calls: int = 10
    search_api: str = "tavily"
    # ... all existing fields preserved
```

### 3. Key Resolution Flow

**Current** (15 call sites):
```python
api_key = get_api_key_for_model(configurable.research_model, config)
```

**New** (single method):
```python
def resolve_model(self, model_string: str) -> tuple[str, str, int | None]:
    """Resolve 'openai:gpt-4.1' → (base_url, api_key, token_limit)."""
    provider, model_name = model_string.split(":", 1)

    provider_config = self.providers.get(provider)
    if not provider_config:
        raise ValueError(f"Unknown provider '{provider}' — add it to config.json")

    if model_name not in provider_config.allowed_models:
        raise ValueError(
            f"Model '{model_name}' not in allowed_models for {provider}. "
            f"Allowed: {provider_config.allowed_models}"
        )

    api_key_attr = f"{provider}_api_key"
    api_key_secret: SecretStr | None = getattr(self, api_key_attr, None)
    if not api_key_secret:
        raise ValueError(f"No API key for {provider} — set {api_key_attr.upper()} in .env")

    token_limit = provider_config.model_token_limits.get(model_name)
    return provider_config.base_url, api_key_secret.get_secret_value(), token_limit
```

**Usage in nodes:**
```python
# Before:
model_config = {
    "model": configurable.research_model,
    "max_tokens": configurable.research_model_max_tokens,
    "api_key": get_api_key_for_model(configurable.research_model, config),
}

# After:
base_url, api_key, token_limit = configurable.resolve_model(configurable.models.research_model)
model_config = {
    "model": configurable.models.research_model,
    "max_tokens": configurable.research_model_max_tokens,
    "api_key": api_key,
    "base_url": base_url,
}
```

### 4. What Goes Where

| Category | Location | Examples |
|----------|----------|---------|
| API keys | `.env` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Provider definitions | `config.json` | base_url, allowed_models, token_limits |
| Model slot assignments | `config.json` | which model for research, summarization, etc. |
| Feature flags | `config.json` / `configurable` | enable_section_writers, enable_reviewer_loop |
| Max iterations/tokens | `config.json` / `configurable` | max_researcher_iterations, max_tokens |
| Runtime overrides | `configurable` dict | Per-request overrides from API server / eval scripts |

### 5. config.json Loading

`config.json` is loaded via a custom `init_json_source()` method that reads the JSON file and injects it into pydantic-settings. The loading order is:

1. `config.json` (lowest priority — defaults)
2. `.env` file (overrides config.json)
3. Actual environment variables (highest priority — overrides both)

```python
import json
from pathlib import Path

def init_json_source(config_path: str = "config.json") -> dict:
    """Load config.json as a dict for pydantic-settings."""
    path = Path(config_path)
    if path.exists():
        return json.loads(path.read_text())
    return {}
```

This is wired into `Configuration.settings_customise_sources()` so that pydantic-settings treats `config.json` as a fallback source.

### 6. Backward Compatibility

- **`config.json` is optional**: If missing, falls back to current hardcoded defaults
- **`.env` format unchanged**: Same keys, same loading via `python-dotenv`
- **`GET_API_KEYS_FROM_CONFIG` mode preserved**: `apiKeys` dict in `configurable` still works
- **`configurable` dict override still works**: Eval scripts and API server pass overrides via the same dict
- **`Configuration.from_runnable_config()` preserved**: Same method, same signature

### 6. Migration Path

1. Add `pydantic-settings` dependency
2. Create `config.json` with provider definitions
3. Refactor `Configuration` to extend `BaseSettings`
4. Replace `get_api_key_for_model()` with `configurable.resolve_model()`
5. Replace `MODEL_TOKEN_LIMITS` with `providers[x].model_token_limits`
6. Update `.env.example` with new keys
7. Update all 15 `get_api_key_for_model()` call sites
8. Update eval scripts to use `models` section

### 8. Rate Limit Configuration

Add to `ProviderConfig`:
```python
class ProviderConfig(BaseModel):
    base_url: str
    default_model: str
    allowed_models: list[str] = []
    model_token_limits: dict[str, int] = {}
    rate_limit_rpm: int | None = None  # requests per minute
    rate_limit_tpm: int | None = None  # tokens per minute
```

This enables the summarization concurrency limiter and retry-with-backoff to use provider-specific limits instead of hardcoded values.

---

## Files Modified

| File | Change |
|------|--------|
| `configuration.py` | Extend `BaseSettings`, add `ProviderConfig`, `ModelsConfig`, `resolve_model()`, `init_json_source()` |
| `utils.py` | Remove `MODEL_TOKEN_LIMITS` dict, remove `get_api_key_for_model()` |
| `deep_researcher.py` | Update 15 call sites to use `resolve_model()` |
| `config.json` | New file — provider definitions |
| `.env.example` | Update with new key names |
| `tests/run_eval_qwen.py` | Update to use `models` section |
| `pyproject.toml` | Add `pydantic-settings` dependency |

## Testing

- Unit tests for `resolve_model()` — valid provider, invalid provider, missing key, missing model
- Unit tests for `ProviderConfig` validation — allowed_models check
- Unit tests for backward compat — `configurable` dict override still works
- Integration test — full graph run with `config.json` loaded
