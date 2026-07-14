# Config System Redesign: Provider-Aware Model & Key Management

**Date:** 2026-07-08
**Status:** Revised v2 (all contradictions resolved)
**Author:** OpenCode (generated from brainstorming session)

---

## Problem Statement

The current model/key configuration system has four issues:

1. **No provider-level control**: API keys are resolved by prefix matching (`openai:*` -> `OPENAI_API_KEY`), but there is no way to configure base URLs, allowed models, or token limits per provider.
2. **Hardcoded token limits**: `MODEL_TOKEN_LIMITS` in `utils.py:829-872` is a manually maintained dict that must be updated for every new model.
3. **No model validation**: Any `provider:model` string is accepted with no check against an allowed list.
4. **Flat .env-only config**: All configuration lives in flat key-value pairs, making structured settings (provider -> models -> limits) impossible without ugly prefixes.

## Approach: Hybrid .env + config.json + pydantic-settings

- **`.env`** (gitignored): API keys only
- **`config.json`** (committed): Provider definitions, model assignments, defaults
- **`pydantic-settings`**: Unified loading with type validation and env var overrides

---

## Design

### 1. Precedence Order (Canonical)

**Runtime values always win.** This is the single source of truth. All code, tests, and documentation must match this order:

| Priority | Source | Wins over |
|----------|--------|-----------|
| 4 (highest) | `configurable` dict (API request / eval script) | everything below |
| 3 | OS environment variables | .env and config.json |
| 2 | `.env` file | config.json |
| 1 (lowest) | `config.json` | built-in registry |
| 0 (fallback) | Built-in `BUILTIN_PROVIDERS` | nothing (last resort) |

**Rationale:** The `configurable` dict is the per-request override mechanism. An API client sending `{"research_model": "openai:gpt-4o"}` in a request body must win over the server's `.env` file. This matches the current behavior where `Configuration.from_runnable_config()` checks `configurable` after env vars, and the test at `test_utils.py:419` confirms env vars override `configurable` for flat keys -- we are **changing** that for the new nested system so runtime overrides take precedence.

**Implementation:** A single `_build_values()` method constructs the final config dict. No pydantic-settings implicit sources are used -- all loading is explicit.

### 2. Built-in Provider Registry

`BUILTIN_PROVIDERS` is a module-level constant in `configuration.py`. It preserves the full current `MODEL_TOKEN_LIMITS` set and is never mutated at runtime.

```python
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
```

**Key design choices:**
- `api_key_env` is an explicit field, not derived from provider ID. This solves the `google` -> `google_genai` canonicalization problem: `api_key_env="GOOGLE_API_KEY"` regardless of provider ID.
- `auth_strategy` distinguishes `"api_key"` (default), `"aws"` (Bedrock), and `"none"` (Ollama).
- `aliases` maps user-facing names to canonical IDs: `"google"` -> `"google_genai"`.

### 3. Provider Merge Algorithm

`config.json` providers are deep-merged into `BUILTIN_PROVIDERS`. The algorithm:

```python
import copy
from typing import Any

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
            # New provider: add as-is
            registry[provider_id] = ProviderConfig(**overrides)
        else:
            # Existing provider: deep-merge
            existing = registry[provider_id]
            merged = existing.model_dump()

            for key, value in overrides.items():
                if key == "allowed_models" and isinstance(value, list):
                    # Union: add new models, preserve existing, deduplicate
                    merged["allowed_models"] = list(
                        dict.fromkeys(merged.get("allowed_models", []) + value)
                    )
                elif key == "model_token_limits" and isinstance(value, dict):
                    # Replace: config.json overrides specific model limits
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
```

**Merge behavior for the sample override:**

```json
{
  "providers": {
    "openai": {
      "allowed_models": ["mimo-v2.5"],
      "model_token_limits": { "mimo-v2.5": 32768 }
    }
  }
}
```

Result: `openai.allowed_models` = `["gpt-4.1", "gpt-4.1-mini", ..., "mimo-v2.5"]` (unioned).
`openai.model_token_limits["mimo-v2.5"]` = `32768` (added). All built-in models preserved.

### 4. Pydantic Model Design

```python
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    base_url: str = ""
    default_model: str = ""
    allowed_models: list[str] = []
    model_token_limits: dict[str, int] = {}
    aliases: list[str] = []
    api_key_env: str = ""           # env var name for this provider's key
    auth_strategy: str = "api_key"  # "api_key" | "aws" | "none"
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
    provider: str              # canonical provider ID (e.g., "google_genai")
    model_string: str          # original "provider:model" string (e.g., "openai:gpt-4.1")
    canonical_model_string: str # canonical "provider:model" (e.g., "google_genai:gemini-2.5-flash")
    model_name: str            # just the model name (e.g., "gpt-4.1")
    base_url: str              # provider base URL
    api_key: str               # resolved API key (plain text)
    token_limit: int | None    # max tokens for this model
    provider_kwargs: dict      # provider-specific extra kwargs


class Configuration(BaseSettings):
    # Secrets (from .env)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    ollama_api_key: SecretStr | None = None
    bedrock_api_key: SecretStr | None = None     # AWS_ACCESS_KEY_ID
    bedrock_secret_key: SecretStr | None = None   # AWS_SECRET_ACCESS_KEY
    bedrock_session_token: SecretStr | None = None

    # Structured config (built at load time, not from pydantic-settings sources)
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

### 5. Single Loader: `_build_values()`

No pydantic-settings implicit sources. One method, one precedence order:

```python
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

    # 0. Start with field defaults
    values: dict[str, Any] = {}

    # 1. Layer config.json
    json_config = _load_config_json()
    for key in cls.model_fields:
        if key in json_config:
            values[key] = json_config[key]

    # 2. Layer .env file
    dotenv_values = _load_dotenv()
    for key in cls.model_fields:
        if key in dotenv_values:
            values[key] = dotenv_values[key]

    # 3. Layer OS environment (wins over .env)
    for key in cls.model_fields:
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            values[key] = env_val

    # 4. Layer configurable dict (wins over everything)
    migrated = _migrate_flat_keys(configurable)
    for key in cls.model_fields:
        if key in migrated:
            values[key] = migrated[key]

    return values


def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
    values = cls._build_values(config)

    # Merge providers: BUILTIN_PROVIDERS < config.json < configurable
    config_providers = values.pop("providers", {})
    registry = build_provider_registry({"providers": config_providers})

    # Also check configurable for provider overrides
    configurable = config.get("configurable", {}) if config else {}
    if "providers" in configurable:
        registry = build_provider_registry({"providers": configurable["providers"]}, registry)

    values["providers"] = registry

    return cls(**{k: v for k, v in values.items() if v is not None})
```

**Flat-to-nested migration:**

```python
_FLAT_TO_NESTED = {
    "research_model": ("models", "research_model"),
    "summarization_model": ("models", "summarization_model"),
    "compression_model": ("models", "compression_model"),
    "final_report_model": ("models", "final_report_model"),
    "classifier_model": ("models", "classifier_model"),
}

def _migrate_flat_keys(configurable: dict) -> dict:
    migrated = dict(configurable)
    for flat_key, (section, field) in _FLAT_TO_NESTED.items():
        if flat_key in migrated:
            migrated.setdefault(section, {})[field] = migrated.pop(flat_key)
    return migrated
```

### 6. Key Resolution Flow

**`resolve_model()` with explicit `api_key_env` lookup:**

```python
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
    # Skip for providers that don't use API keys
    if provider_config.auth_strategy == "none":
        return ""

    # 1. Per-request keys (highest priority)
    if runnable_config and os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() == "true":
        api_keys = runnable_config.get("configurable", {}).get("apiKeys", {})
        # Check by api_key_env name (e.g., "OPENAI_API_KEY")
        if provider_config.api_key_env and api_keys.get(provider_config.api_key_env):
            return api_keys[provider_config.api_key_env]
        # Also check by provider key (backward compat)
        if api_keys.get(f"{provider}_api_key"):
            return api_keys[f"{provider}_api_key"]

    # 2. Configuration field (from .env / env vars)
    attr_name = f"{provider}_api_key"
    api_key_secret: SecretStr | None = getattr(self, attr_name, None)
    if api_key_secret:
        return api_key_secret.get_secret_value()

    # 3. Environment variable using provider's api_key_env
    if provider_config.api_key_env:
        env_val = os.getenv(provider_config.api_key_env)
        if env_val:
            return env_val

    # 4. AWS credential chain for Bedrock
    if provider_config.auth_strategy == "aws":
        return ""  # Let boto3 handle credentials via its own chain

    raise ValueError(
        f"No API key for provider '{provider}'. "
        f"Set {provider_config.api_key_env} in .env or provide via config.apiKeys."
    )
```

### 7. Shared Model-Config Builder

```python
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
```

### 8. Model Router Integration

Router returns `ResolvedModel`, not a tuple:

```python
def _resolve_model_via_router(
    configurable: Configuration,
    config: RunnableConfig,
    task_type: TaskType,
    current_model: str,
    input_length: int = 0,
    complexity_hint: ModelTier | None = None,
) -> ResolvedModel | None:
    """Resolve model via router. Returns ResolvedModel or None (fallback to current)."""
    if not configurable.enable_model_routing:
        return None

    router = get_model_router()
    result = router.select_with_tier_if_enabled(task_type=task_type, ...)
    if result is None:
        return None

    routed_model_string, _ = result

    try:
        resolved = configurable.resolve_model(routed_model_string, config)
        return resolved  # Full ResolvedModel with base_url, provider_kwargs, etc.
    except ValueError:
        return None  # Router selected unknown model; caller falls back to current
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
resolved = _resolve_model_via_router(configurable, config, TaskType.REASONING, ...)
if resolved is None:
    resolved = configurable.resolve_model(configurable.models.research_model, config)
model_config = build_model_config(configurable, resolved.model_string,
                                  configurable.research_model_max_tokens, config)
```

### 9. Secret Redaction for API Persistence

```python
_SECRET_PATTERNS = re.compile(
    r"(api_key|secret|token|password|authorization)", re.IGNORECASE
)

def redact_secrets(config: dict) -> dict:
    """Redact all secret fields from config before storage."""
    redacted = {}
    for key, value in config.items():
        if _SECRET_PATTERNS.search(key):
            redacted[key] = "***REDACTED***" if value else None
        elif isinstance(value, dict):
            redacted[key] = redact_secrets(value)
        elif isinstance(value, list):
            redacted[key] = [redact_secrets(v) if isinstance(v, dict) else v for v in value]
        else:
            redacted[key] = value
    return redacted
```

**Two separate objects in the runner:**

```python
# In runner.py
runtime_config = {
    "configurable": {
        "repo": repo,
        "run_id": run_id,
        **body.config,  # Contains secrets for this run
    }
}

# Persist redacted version to SQLite
persisted_config = redact_secrets(body.config)
await repo.create_run(run_id, query, persisted_config)
```

### 10. Config File Resolution

```python
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

    # Package-relative lookup
    package_root = Path(__file__).resolve().parent.parent.parent
    path = package_root / "config.json"
    if path.exists():
        return json.loads(path.read_text())

    return {}
```

**Decision: `config.json` is NOT package data.** It is a repository-local file for source checkouts. Installed deployments must use `ODR_CONFIG_FILE`. This is documented in `.env.example` and `HOW_TO_RUN.md`.

### 11. Provider-Specific kwargs

`provider_kwargs` is populated by `resolve_model()` for providers that need extra arguments:

```python
def resolve_model(self, model_string, runnable_config=None):
    # ... existing logic ...

    # Provider-specific kwargs
    provider_kwargs = {}
    if canonical_provider == "bedrock":
        # Bedrock needs region, session credentials
        provider_kwargs["region_name"] = os.getenv("AWS_REGION", "us-east-1")
    elif canonical_provider == "ollama":
        provider_kwargs["num_ctx"] = 4096

    return ResolvedModel(
        # ... existing fields ...
        provider_kwargs=provider_kwargs,
    )
```

`build_model_config()` merges them into the config dict:

```python
config.update(resolved.provider_kwargs)
```

### 12. What Goes Where

| Category | Location | Examples |
|----------|----------|---------|
| API keys | `.env` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Provider definitions | `config.json` | base_url, allowed_models, token_limits |
| Model slot assignments | `config.json` | which model for research, summarization, etc. |
| Feature flags | `configurable` dict | enable_section_writers, enable_reviewer_loop |
| Max iterations/tokens | `configurable` dict | max_researcher_iterations, max_tokens |
| Runtime overrides | `configurable` dict | Per-request overrides from API server / eval scripts |

### 13. Migration Path

1. Add `pydantic-settings` dependency
2. Create `BUILTIN_PROVIDERS` dict in `configuration.py`
3. Create `config.json` with provider overrides
4. Refactor `Configuration` to use `_build_values()` (no BaseSettings implicit sources)
5. Add `ResolvedModel`, `resolve_model()`, `_resolve_api_key()`, `build_model_config()`
6. Add `redact_secrets()` to `utils.py` and wire into `runner.py`
7. Update `_resolve_model_via_router()` to return `ResolvedModel`
8. Update all `get_api_key_for_model()` call sites (11 production + 4 test)
9. Remove `MODEL_TOKEN_LIMITS` dict and `get_api_key_for_model()` from `utils.py`
10. Update `.env.example` with new keys and `ODR_CONFIG_FILE` docs
11. Update eval scripts to use `models` section

### 14. Backward Compatibility

- **Built-in registry**: No `config.json` needed for defaults to work
- **`.env` format unchanged**: Same keys, same loading via `python-dotenv`
- **`GET_API_KEYS_FROM_CONFIG` mode preserved**: `apiKeys` dict in `configurable` still works via `_resolve_api_key()`
- **Flat key migration**: `{"research_model": "openai:gpt-4o"}` in `configurable` maps to `models.research_model`
- **`Configuration.from_runnable_config()` preserved**: Same method, same signature

---

## Files Modified

| File | Change |
|------|--------|
| `configuration.py` | `BUILTIN_PROVIDERS`, `ProviderConfig`, `ModelsConfig`, `ResolvedModel`, `resolve_model()`, `build_model_config()`, `_build_values()`, `build_provider_registry()` |
| `utils.py` | Remove `MODEL_TOKEN_LIMITS`, remove `get_api_key_for_model()`, add `redact_secrets()` |
| `deep_researcher.py` | Migrate 11 call sites to `build_model_config()`, update `_resolve_model_via_router()` to return `ResolvedModel` |
| `api/runner.py` | Separate `runtime_config` vs `persisted_config`, wire `redact_secrets()` |
| `config.json` | New file -- provider overrides |
| `.env.example` | Update with new keys, `ODR_CONFIG_FILE` docs |
| `tests/run_eval_qwen.py` | Update to use `models` section |
| `pyproject.toml` | Add `pydantic-settings` dependency |

## Testing

- **Precedence**: configurable dict > env vars > .env > config.json > built-in defaults
- **Flat-key migration**: `{"research_model": "openai:gpt-4o"}` maps to `models.research_model`
- **Built-in fallback**: No config.json, system works with BUILTIN_PROVIDERS
- **Deep merge**: config.json `allowed_models` unions with built-in, `model_token_limits` merges
- **Missing config.json**: Silent fallback to built-in registry
- **Malformed ODR_CONFIG_FILE**: Raises error, not silent fallback
- **Provider aliases**: `google:*` resolves to `google_genai:*`, canonical string uses `google_genai:`
- **API key canonical**: `google_genai` looks up `GOOGLE_API_KEY` via `api_key_env`, not `GOOGLE_GENAI_API_KEY`
- **Model validation**: Rejected model raises error with allowed list
- **Missing API key**: Error with remediation using `provider_config.api_key_env`
- **Per-request keys**: `apiKeys.OPENAI_API_KEY` overrides `.env`
- **Secret redaction**: `redact_secrets()` catches nested `apiKeys`, dynamic provider keys
- **Runtime vs persisted**: `runtime_config` retains secrets, `persisted_config` redacted
- **All model slots**: research, summarization, compression, final_report, classifier
- **Routed models**: Router returns `ResolvedModel`, `build_model_config()` uses it
- **base_url propagation**: `init_chat_model()` receives base_url when present
- **provider_kwargs**: Bedrock gets `region_name`, Ollama gets `num_ctx`
- **Bedrock auth**: `auth_strategy="aws"` skips API key resolution
- **Ollama auth**: `auth_strategy="none"` returns empty string
- **Package/container**: `ODR_CONFIG_FILE` works for installed wheels
- **Router propagation**: `ResolvedModel` carries base_url and provider_kwargs through routing
