"""Tests for the new provider-aware config system."""


def test_provider_config_defaults():
    """ProviderConfig has sensible defaults."""
    from open_deep_research.configuration import ProviderConfig

    pc = ProviderConfig()
    assert pc.base_url == ""
    assert pc.default_model == ""
    assert pc.allowed_models == []
    assert pc.model_token_limits == {}
    assert pc.aliases == []
    assert pc.api_key_env == ""
    assert pc.auth_strategy == "api_key"


def test_models_config_defaults():
    """ModelsConfig has correct defaults matching current Configuration."""
    from open_deep_research.configuration import ModelsConfig

    mc = ModelsConfig()
    assert mc.research_model == "openai:gpt-4.1"
    assert mc.summarization_model == "openai:gpt-4.1-mini"
    assert mc.compression_model == "openai:gpt-4.1"
    assert mc.final_report_model == "openai:gpt-4.1"
    assert mc.classifier_model is None


def test_resolved_model_fields():
    """ResolvedModel carries all fields needed for init_chat_model()."""
    from open_deep_research.configuration import ResolvedModel

    rm = ResolvedModel(
        provider="openai",
        model_string="openai:gpt-4.1",
        canonical_model_string="openai:gpt-4.1",
        model_name="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        token_limit=1047576,
        provider_kwargs={},
    )
    assert rm.provider == "openai"
    assert rm.base_url == "https://api.openai.com/v1"


def test_builtin_providers_preserves_current_limits():
    """BUILTIN_PROVIDERS contains all models from current MODEL_TOKEN_LIMITS."""
    from open_deep_research.configuration import BUILTIN_PROVIDERS

    # OpenAI models
    assert "openai" in BUILTIN_PROVIDERS
    assert BUILTIN_PROVIDERS["openai"].model_token_limits["gpt-4.1"] == 1047576
    assert BUILTIN_PROVIDERS["openai"].model_token_limits["gpt-4o"] == 128000

    # Anthropic models
    assert "anthropic" in BUILTIN_PROVIDERS
    assert BUILTIN_PROVIDERS["anthropic"].model_token_limits["claude-opus-4"] == 200000

    # Google models (google_genai with alias google)
    assert "google_genai" in BUILTIN_PROVIDERS
    assert "google" in BUILTIN_PROVIDERS["google_genai"].aliases

    # Bedrock
    assert "bedrock" in BUILTIN_PROVIDERS
    assert BUILTIN_PROVIDERS["bedrock"].auth_strategy == "aws"

    # Ollama
    assert "ollama" in BUILTIN_PROVIDERS
    assert BUILTIN_PROVIDERS["ollama"].auth_strategy == "none"


def test_builtin_providers_api_key_env():
    """api_key_env is set correctly for each provider."""
    from open_deep_research.configuration import BUILTIN_PROVIDERS

    assert BUILTIN_PROVIDERS["openai"].api_key_env == "OPENAI_API_KEY"
    assert BUILTIN_PROVIDERS["anthropic"].api_key_env == "ANTHROPIC_API_KEY"
    assert BUILTIN_PROVIDERS["google_genai"].api_key_env == "GOOGLE_API_KEY"
    assert BUILTIN_PROVIDERS["bedrock"].api_key_env == "AWS_ACCESS_KEY_ID"
    assert BUILTIN_PROVIDERS["cohere"].api_key_env == "COHERE_API_KEY"
    assert BUILTIN_PROVIDERS["mistral"].api_key_env == "MISTRAL_API_KEY"
    assert BUILTIN_PROVIDERS["ollama"].api_key_env == ""


def test_build_provider_registry_unknown_provider_added():
    """Unknown provider in config.json is added to registry."""
    from open_deep_research.configuration import build_provider_registry

    registry = build_provider_registry({
        "providers": {
            "my_custom": {
                "base_url": "https://custom.api.com",
                "api_key_env": "CUSTOM_API_KEY",
            }
        }
    })
    assert "my_custom" in registry
    assert registry["my_custom"].base_url == "https://custom.api.com"


def test_build_provider_registry_known_provider_union():
    """Known provider: allowed_models unions, model_token_limits merges."""
    from open_deep_research.configuration import build_provider_registry

    registry = build_provider_registry({
        "providers": {
            "openai": {
                "allowed_models": ["mimo-v2.5"],
                "model_token_limits": {"mimo-v2.5": 32768},
            }
        }
    })
    # Original models preserved
    assert "gpt-4.1" in registry["openai"].allowed_models
    # New model added (union)
    assert "mimo-v2.5" in registry["openai"].allowed_models
    # Original limits preserved
    assert registry["openai"].model_token_limits["gpt-4.1"] == 1047576
    # New limit added
    assert registry["openai"].model_token_limits["mimo-v2.5"] == 32768


def test_build_provider_registry_preserves_builtins():
    """Providers not in config.json are preserved unchanged."""
    from open_deep_research.configuration import build_provider_registry

    registry = build_provider_registry({"providers": {}})
    assert "openai" in registry
    assert "anthropic" in registry
    assert "google_genai" in registry
