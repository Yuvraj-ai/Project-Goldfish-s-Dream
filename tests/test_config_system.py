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
