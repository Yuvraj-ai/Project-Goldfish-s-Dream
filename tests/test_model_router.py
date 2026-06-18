from __future__ import annotations


def test_model_tier_enum():
    from open_deep_research.api.model_router import ModelTier
    assert ModelTier.FAST.value == "fast"
    assert ModelTier.BALANCED.value == "balanced"
    assert ModelTier.QUALITY.value == "quality"


def test_task_type_enum():
    from open_deep_research.api.model_router import TaskType
    assert TaskType.SUMMARIZATION in TaskType


def test_select_default_tier():
    from open_deep_research.api.model_router import ModelRouter, TaskType
    router = ModelRouter()
    model = router.select(TaskType.SUMMARIZATION, input_length=1000)
    assert isinstance(model, str)
    assert len(model) > 0


def test_select_upgrades_for_long_input():
    from open_deep_research.api.model_router import ModelRouter, TaskType
    router = ModelRouter()
    short_model = router.select(TaskType.SUMMARIZATION, input_length=1000)
    long_model = router.select(TaskType.SUMMARIZATION, input_length=10000)
    assert short_model != long_model


def test_complexity_hint_overrides():
    from open_deep_research.api.model_router import (
        ModelRouter,
        ModelTier,
        TaskType,
    )
    router = ModelRouter()
    fast_model = router.select(
        TaskType.REPORT_WRITING, input_length=1000, complexity_hint=ModelTier.FAST,
    )
    default_model = router.select(TaskType.REPORT_WRITING, input_length=1000)
    assert fast_model != default_model


def test_provider_config():
    from open_deep_research.api.model_router import ProviderConfig
    config = ProviderConfig(
        name="google",
        models={},
    )
    assert config.name == "google"


def test_router_logs_selection():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    model, tier = router.select_with_tier(TaskType.SUMMARIZATION, input_length=100)
    assert isinstance(model, str)
    assert tier in (ModelTier.FAST, ModelTier.BALANCED, ModelTier.QUALITY)
