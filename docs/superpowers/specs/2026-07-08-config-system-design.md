# Config System Redesign: Provider-Aware Model & Key Management

**Date:** 2026-07-08
**Status:** Revised (gap fixes applied)
**Author:** OpenCode (generated from brainstorming session)

---

## Problem Statement

The current model/key configuration system has four issues:

1. **No provider-level control**: API keys are resolved by prefix matching (`openai:*` -> `OPENAI_API_KEY`), but there's no way to configure base URLs, allowed models, or token limits per provider.
2. **Hardcoded token limits**: `MODEL_TOKEN_LIMITS` in `utils.py:829-872` is a manually maintained dict that must be updated for every new model.
3. **No model validation**: Any `provider:model` string is accepted with no check against an allowed list.
4. **Flat .env-only config**: All configuration lives in flat key-value pairs, making structured settings (provider -> models -> limits) impossible without ugly prefixes.

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

### 1. Built-in Provider Registry (No-config.json Fallback)

A built-in registry ensures the system works even when `config.json` is missing. `config.json` extends or overrides this registry.

```python
BUILTIN_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1",
        allowed_models=["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
                        "o4-mini", "qwen2.5", "qwen3.6"],
        model_token_limits={
            "gpt-4.1": 1047576, "gpt-4.1-mini": 1047576, "gpt-4.1-nano": 1047576,
            "gpt-4o": 128000, "gpt-4o-mini": 128000,
            "o4-mini": 200000, "o3-mini": 200000, "o3": 200000,
            "qwen2.5": 32768, "qwen3.6": 32768,
        },
    ),
    "anthropic": ProviderConfig(
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
        allowed_models=["claude-opus-4", "claude-sonnet-4-20250514",
                        "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku"],
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
        allowed_models=["gemini-2.5-pro", "gemini-2.5-flash",
                        "gemini-1.5-pro", "gemini-1.5-flash"],
        model_token_limits={
            "gemini-2.5-pro": 1048576, "gemini-2.5-flash": 1048576,
            "gemini-1.5-pro": 2097152, "gemini-1.5-flash": 1048576,
        },
    ),
    "bedrock": ProviderConfig(
        base_url="",
        default_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        allowed_models=["us.amazon.nova-premier-v1:0", "us.amazon.nova-pro-v1:0",
                        "us.amazon.nova-lite-v1:0", "us.amazon.nova-micro-v1:0",
                        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "us.anthropic.claude-sonnet-4-20250514-v1:0",
                        "us.anthropic.claude-opus-4-20250514-v1:0"],
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
        allowed_models=["command-r-plus", "command-r", "command-light", "command"],
        model_token_limits={
            "command-r-plus": 128000, "command-r": 128000,
            "command-light": 4096, "command": 4096,
        },
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        default_model="llama2",
        allowed_models=["codellama", "llama2:70b", "llama2:13b", "llama2", "mistral"],
        model_token_limits={
            "codellama": 16384, "llama2:70b": 4096,
            "llama2:13b": 4096, "llama2": 4096, "mistral": 32768,
        },
    ),
}
```

**Provider aliases** resolve naming mismatches:
- `"google"` -> `"google_genai"` (router uses `google_genai:*`, user may write `google:*`)
- Aliases checked first in `resolve_model()` before provider lookup

### 2. Config File Structure

**`config.json`** (committed to git, extends built-in registry):

```json
{
  "providers": {
    "openai": {
      "base_url": "https://api.openai.com/v1",
      "allowed_models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "mimo-v2.5"],
      "model_token_limits": { "mimo-v2.5": 32768 }
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

Merging rules:
- `config.json` providers **merge into** built-in providers (not replace)
- If `config.json` specifies `"openai"`, its fields override the built-in `openai` entry
- If `config.json` specifies a new provider (e.g., `"openrouter"`), it is added to the registry
- If `config.json` is missing entirely, built-in registry is used as-is

**`.env`** (gitignored, secrets only):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
TAVILY_API_KEY=tvly-dev-...
```

### 3. Pydantic Model Design

```python
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    base_url: str = ""
    default_model: str = ""
    allowed_models: list[str] = []
    model_token_limits: dict[str, int] = {}
    aliases: list[str] = []
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None


class ModelsConfig(BaseModel):
    research_model: str = "openai:gpt-4.1"
    summarization_model: str = "openai:gpt-4.1-mini"
    compression_model: str = "openai:gpt-4.1"
    final_report_model: str = "openai:gpt-4.1"
    classifier_model: str | None = None


class ResolvedModel:
    """Result of model resolution -- carries everything needed to create a model client."""
    provider: str
    model_string: str
    model_name: str
    base_url: str
    api_key: str
    token_limit: int | None
    provider_kwargs: dict


class Configuration(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Secrets (from .env)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    ollama_api_key: SecretStr | None = None
    bedrock_api_key: SecretStr | None = None
    bedrock_secret_key: SecretStr | None = None
    bedrock_session_token: SecretStr | None = None

    # Structured config
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

### 4. Key Resolution Flow

**`resolve_model()` accepts both Configuration-level and RunnableConfig-level keys:**

```python
def resolve_model(
    self,
    model_string: str,
    runnable_config: RunnableConfig | None = None,
) -> ResolvedModel:
    """Resolve 'openai:gpt-4.1' -> ResolvedModel with all needed fields.

    Resolution order for API keys:
    1. runnable_config["configurable"]["apiKeys"] (per-request, highest priority)
    2. self.<provider>_api_key (from .env / config.json)
    3. os.environ["<PROVIDER>_API_KEY"] (environment variable)
    """
    provider, model_name = model_string.split(":", 1)
    canonical_provider = self._resolve_provider_alias(provider)

    provider_config = self.providers.get(canonical_provider)
    if not provider_config:
        available = list(self.providers.keys())
        raise ValueError(
            f"Unknown provider '{provider}' (canonical: '{canonical_provider}'). "
            f"Available: {available}. Add it to config.json or use an alias."
        )

    if provider_config.allowed_models and model_name not in provider_config.allowed_models:
        raise ValueError(
            f"Model '{model_name}' not in allowed_models for {canonical_provider}. "
            f"Allowed: {provider_config.allowed_models}"
        )

    api_key = self._resolve_api_key(canonical_provider, runnable_config)
    token_limit = provider_config.model_token_limits.get(model_name)

    return ResolvedModel(
        provider=canonical_provider,
        model_string=model_string,
        model_name=model_name,
        base_url=provider_config.base_url,
        api_key=api_key,
        token_limit=token_limit,
        provider_kwargs={},
    )
```

**`_resolve_api_key()` handles per-request keys (GET_API_KEYS_FROM_CONFIG compat):**

```python
def _resolve_api_key(
    self,
    provider: str,
    runnable_config: RunnableConfig | None = None,
) -> str:
    """Resolve API key with per-request override support."""
    if runnable_config and os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() == "true":
        api_keys = runnable_config.get("configurable", {}).get("apiKeys", {})
        if api_keys.get(f"{provider.upper()}_API_KEY"):
            return api_keys[f"{provider.upper()}_API_KEY"]

    api_key_secret: SecretStr | None = getattr(self, f"{provider}_api_key", None)
    if api_key_secret:
        return api_key_secret.get_secret_value()

    env_key = os.getenv(f"{provider.upper()}_API_KEY")
    if env_key:
        return env_key

    raise ValueError(
        f"No API key for provider '{provider}'. "
        f"Set {provider.upper()}_API_KEY in .env or provide via config.apiKeys."
    )
```

**Shared model-config builder (replaces per-node boilerplate):**

```python
def build_model_config(
    configurable: Configuration,
    model_string: str,
    max_tokens: int,
    runnable_config: RunnableConfig | None = None,
) -> dict:
    """Build a model config dict for init_chat_model()."""
    resolved = configurable.resolve_model(model_string, runnable_config)
    config = {
        "model": resolved.model_string,
        "max_tokens": max_tokens,
        "api_key": resolved.api_key,
        "tags": ["langsmith:nostream"],
    }
    if resolved.base_url:
        config["base_url"] = resolved.base_url
    return config
```

### 5. config.json Loading & Precedence

**Full precedence matrix (highest to lowest):**

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | OS environment variables | `MAX_RESEARCHER_ITERATIONS=9` |
| 2 | `configurable` dict (API request / eval script) | `{"research_model": "openai:gpt-4o"}` |
| 3 | `.env` file | `OPENAI_API_KEY=sk-...` |
| 4 (lowest) | `config.json` | `{"models": {"research_model": "openai:gpt-4.1"}}` |

**Flat-to-nested migration:** Existing flat keys in `configurable` dict map to nested paths:

```python
_FLAT_TO_NESTED = {
    "research_model": "models.research_model",
    "summarization_model": "models.summarization_model",
    "compression_model": "models.compression_model",
    "final_report_model": "models.final_report_model",
    "classifier_model": "models.classifier_model",
}

def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
    configurable = config.get("configurable", {}) if config else {}
    migrated = dict(configurable)
    for flat_key, nested_path in _FLAT_TO_NESTED.items():
        if flat_key in migrated and flat_key not in migrated.get("models", {}):
            migrated.setdefault("models", {})[nested_path.split(".")[-1]] = migrated.pop(flat_key)

    json_config = _load_config_json()
    values = {}
    for field_name in cls.model_fields:
        if field_name in json_config:
            values[field_name] = json_config[field_name]
        env_val = os.environ.get(field_name.upper())
        if env_val is not None:
            values[field_name] = env_val
        if field_name in migrated:
            values[field_name] = migrated[field_name]

    return cls(**{k: v for k, v in values.items() if v is not None})
```

**Config file resolution (not just `Path("config.json")`):**

```python
import os
from pathlib import Path

def _load_config_json() -> dict:
    config_path = os.getenv("ODR_CONFIG_FILE")
    if config_path:
        path = Path(config_path)
    else:
        package_root = Path(__file__).resolve().parent.parent.parent
        path = package_root / "config.json"

    if path.exists():
        return json.loads(path.read_text())
    return {}
```

### 6. Secret Redaction for API Persistence

The REST API stores config in SQLite. Secrets must be redacted before storage.

```python
_SECRET_FIELDS = {
    "openai_api_key", "anthropic_api_key", "google_api_key",
    "tavily_api_key", "cohere_api_key", "ollama_api_key",
    "bedrock_api_key", "bedrock_secret_key", "bedrock_session_token",
}

def redact_secrets(config: dict) -> dict:
    redacted = {}
    for key, value in config.items():
        if key in _SECRET_FIELDS:
            redacted[key] = "***REDACTED***" if value else None
        elif isinstance(value, dict):
            redacted[key] = redact_secrets(value)
        else:
            redacted[key] = value
    return redacted
```

Applied in `ResearchRunner.start()` before persisting to SQLite.

### 7. Model Router Integration

The model router currently calls `get_api_key_for_model()`. After migration, it calls `resolve_model()` instead.

```python
# In deep_researcher.py _resolve_model_via_router():
def _resolve_model_via_router(configurable, config, task_type, current_model, ...):
    if not configurable.enable_model_routing:
        return current_model, None

    router = get_model_router()
    result = router.select_with_tier_if_enabled(task_type=task_type, ...)
    if result is None:
        return current_model, None

    routed_model, _ = result

    # NEW: use resolve_model() for validation and key resolution
    try:
        resolved = configurable.resolve_model(routed_model, config)
        return resolved.model_string, resolved.api_key
    except ValueError:
        # Router selected a model that is not in the registry -- fall back
        return current_model, None
```

### 8. What Goes Where

| Category | Location | Examples |
|----------|----------|---------|
| API keys | `.env` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Provider definitions | `config.json` | base_url, allowed_models, token_limits |
| Model slot assignments | `config.json` | which model for research, summarization, etc. |
| Feature flags | `configurable` dict | enable_section_writers, enable_reviewer_loop |
| Max iterations/tokens | `configurable` dict | max_researcher_iterations, max_tokens |
| Runtime overrides | `configurable` dict | Per-request overrides from API server / eval scripts |

### 9. Rate Limit Configuration (Deferred)

Rate limit fields (`rate_limit_rpm`, `rate_limit_tpm`) are defined on `ProviderConfig` but **not wired into the governor in this phase**. The governor (`governor.py`) is not currently wired into model execution. Wiring it is a separate task.

Fields are included in the config schema so they are available when the governor integration happens.

### 10. Migration Path

1. Add `pydantic-settings` dependency
2. Create `BUILTIN_PROVIDERS` dict in `configuration.py`
3. Create `config.json` with provider overrides
4. Refactor `Configuration` to extend `BaseSettings`
5. Add `ResolvedModel` class, `resolve_model()`, `_resolve_api_key()`, `build_model_config()`
6. Add `redact_secrets()` to `utils.py` and wire into `runner.py`
7. Update `_resolve_model_via_router()` to use `resolve_model()`
8. Update all `get_api_key_for_model()` call sites (11 production + 4 test)
9. Remove `MODEL_TOKEN_LIMITS` dict and `get_api_key_for_model()` from `utils.py`
10. Update `.env.example` with new keys
11. Update eval scripts to use `models` section

### 11. Backward Compatibility

- **Built-in registry**: No `config.json` needed for defaults to work
- **`.env` format unchanged**: Same keys, same loading via `python-dotenv`
- **`GET_API_KEYS_FROM_CONFIG` mode preserved**: `apiKeys` dict in `configurable` still works via `_resolve_api_key()`
- **Flat key migration**: `{"research_model": "openai:gpt-4o"}` in `configurable` maps to `models.research_model`
- **`Configuration.from_runnable_config()` preserved**: Same method, same signature

---

## Files Modified

| File | Change |
|------|--------|
| `configuration.py` | Extend `BaseSettings`, add `ProviderConfig`, `ModelsConfig`, `ResolvedModel`, `resolve_model()`, `build_model_config()`, `BUILTIN_PROVIDERS` |
| `utils.py` | Remove `MODEL_TOKEN_LIMITS` dict, remove `get_api_key_for_model()`, add `redact_secrets()` |
| `deep_researcher.py` | Migrate 11 call sites to use `build_model_config()`, update `_resolve_model_via_router()` |
| `api/runner.py` | Wire `redact_secrets()` before SQLite persistence |
| `api/model_router.py` | No changes needed (router returns model string, resolution happens in `_resolve_model_via_router`) |
| `config.json` | New file -- provider overrides |
| `.env.example` | Update with new key names |
| `tests/run_eval_qwen.py` | Update to use `models` section |
| `pyproject.toml` | Add `pydantic-settings` dependency |

## Testing

- **Precedence matrix**: config.json < .env < env vars < configurable dict
- **Flat-key migration**: `{"research_model": "openai:gpt-4o"}` maps correctly
- **Built-in fallback**: No config.json present, system works with BUILTIN_PROVIDERS
- **Missing/malformed JSON**: Graceful fallback to built-in registry
- **Provider aliases**: `google:*` resolves to `google_genai:*`
- **Model validation**: Rejected model raises clear error with allowed list
- **Missing API key**: Clear error message with remediation instructions
- **Per-request keys**: `apiKeys` in `configurable` overrides `.env`
- **All model slots**: research, summarization, compression, final_report, classifier
- **Routed models**: Router-selected model resolves through `resolve_model()`
- **base_url propagation**: `init_chat_model()` receives base_url when present
- **Secret redaction**: `redact_secrets()` strips all key fields before storage
- **API persistence**: Stored config contains no plaintext secrets
- **Package/container**: `ODR_CONFIG_FILE` env var and package-relative resolution
