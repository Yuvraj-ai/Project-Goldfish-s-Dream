"""Tests for the new provider-aware config system."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig


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


def test_load_config_json_missing_file_returns_empty():
    """Missing config.json returns empty dict."""
    from open_deep_research.configuration import _load_config_json

    with patch.dict(os.environ, {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            result = _load_config_json()
            assert result == {}


def test_load_config_json_env_var():
    """ODR_CONFIG_FILE env var loads specified path."""
    from open_deep_research.configuration import _load_config_json

    config_data = {"providers": {"openai": {"allowed_models": ["test-model"]}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        with patch.dict(os.environ, {"ODR_CONFIG_FILE": f.name}):
            result = _load_config_json()
            assert result == config_data
    os.unlink(f.name)


def test_load_config_json_env_var_missing_raises():
    """ODR_CONFIG_FILE pointing to missing file raises error."""
    from open_deep_research.configuration import _load_config_json

    with patch.dict(os.environ, {"ODR_CONFIG_FILE": "/nonexistent/config.json"}):
        with pytest.raises(FileNotFoundError, match="ODR_CONFIG_FILE"):
            _load_config_json()


def test_is_secret_key_matches_known_names():
    """_is_secret_key matches known secret field names."""
    from open_deep_research.configuration import _is_secret_key

    assert _is_secret_key("openai_api_key")
    assert _is_secret_key("OPENAI_API_KEY")
    assert _is_secret_key("anthropic_api_key")
    assert _is_secret_key("bedrock_secret_key")
    assert _is_secret_key("bedrock_session_token")
    assert not _is_secret_key("max_total_tokens")
    assert not _is_secret_key("research_model_max_tokens")
    assert not _is_secret_key("research_model")


def test_redact_secrets_nested_api_keys():
    """redact_secrets recurses into nested apiKeys dict."""
    from open_deep_research.configuration import redact_secrets

    config = {
        "openai_api_key": "sk-secret",
        "apiKeys": {
            "OPENAI_API_KEY": "sk-secret",
            "ANTHROPIC_API_KEY": "sk-ant",
        },
        "max_total_tokens": 500000,
        "research_model": "openai:gpt-4.1",
    }
    redacted = redact_secrets(config)
    assert redacted["openai_api_key"] == "***REDACTED***"
    assert redacted["apiKeys"]["OPENAI_API_KEY"] == "***REDACTED***"
    assert redacted["apiKeys"]["ANTHROPIC_API_KEY"] == "***REDACTED***"
    assert redacted["max_total_tokens"] == 500000  # NOT redacted
    assert redacted["research_model"] == "openai:gpt-4.1"  # NOT redacted


def test_migrate_flat_keys():
    """Flat keys like research_model map to models.research_model."""
    from open_deep_research.configuration import _migrate_flat_keys

    flat = {"research_model": "openai:gpt-4o", "enable_section_writers": True}
    migrated = _migrate_flat_keys(flat)
    assert migrated["models"]["research_model"] == "openai:gpt-4o"
    assert migrated["enable_section_writers"] is True  # non-flat key preserved
    assert "research_model" not in migrated  # flat key removed


def test_precedence_configurable_wins_over_env():
    """configurable dict wins over OS environment variables."""
    from open_deep_research.configuration import Configuration

    with patch.dict(os.environ, {"RESEARCH_MODEL": "openai:gpt-4o"}):
        config = Configuration.from_runnable_config(
            RunnableConfig(configurable={"models": {"research_model": "openai:gpt-4.1"}})
        )
    assert config.models.research_model == "openai:gpt-4.1"


def test_precedence_env_wins_over_dotenv():
    """OS env wins over .env file."""
    from open_deep_research.configuration import Configuration

    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("RESEARCH_MODEL=openai:gpt-4o\n")
        f.flush()
        with patch.dict(os.environ, {"RESEARCH_MODEL": "openai:gpt-4.1"}):
            with patch("open_deep_research.configuration._load_dotenv",
                       return_value={"RESEARCH_MODEL": "openai:o3"}):
                config = Configuration.from_runnable_config(None)
    assert config.research_model == "openai:gpt-4.1"
    os.unlink(f.name)


def test_builtin_providers_fallback():
    """No config.json = system works with BUILTIN_PROVIDERS."""
    from open_deep_research.configuration import Configuration

    with patch("open_deep_research.configuration._load_config_json", return_value={}):
        config = Configuration.from_runnable_config(None)
    assert "openai" in config.providers
    assert config.providers["openai"].model_token_limits["gpt-4.1"] == 1047576


def test_from_runnable_config_models_section():
    """Models section in configurable maps to config.models."""
    from open_deep_research.configuration import Configuration

    config = Configuration.from_runnable_config(
        RunnableConfig(configurable={"models": {"research_model": "openai:gpt-4o"}})
    )
    assert config.models.research_model == "openai:gpt-4o"
