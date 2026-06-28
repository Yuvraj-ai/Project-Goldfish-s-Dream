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


def test_select_with_tier_if_enabled():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    result = router.select_with_tier_if_enabled(TaskType.SUMMARIZATION, enabled=True)
    assert result is not None
    assert isinstance(result[0], str)
    assert isinstance(result[1], ModelTier)

    result_disabled = router.select_with_tier_if_enabled(TaskType.SUMMARIZATION, enabled=False)
    assert result_disabled is None


def test_all_task_types_have_tier():
    from open_deep_research.api.model_router import ModelRouter, TaskType
    router = ModelRouter()
    for task in TaskType:
        model = router.select(task, input_length=100)
        assert isinstance(model, str) and len(model) > 0


def test_long_input_upgrade_all_tiers():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    for task in (TaskType.SUMMARIZATION, TaskType.CLASSIFICATION):
        short_result = router.select_with_tier(task, input_length=100)
        long_result = router.select_with_tier(task, input_length=10000)
        short_model, short_tier = short_result
        long_model, long_tier = long_result
        if short_tier == ModelTier.FAST and long_tier == ModelTier.BALANCED:
            assert short_model != long_model


def test_router_thread_safe():
    import concurrent.futures
    from open_deep_research.api.model_router import ModelRouter, TaskType
    router = ModelRouter()
    def select_task(task_name):
        return router.select(TaskType(task_name), input_length=500)
    task_names = [t.value for t in TaskType]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(select_task, task_names * 3))
    assert len(results) == len(task_names * 3)
    assert all(isinstance(r, str) for r in results)


def test_provider_config_defaults():
    from open_deep_research.api.model_router import ProviderConfig
    config = ProviderConfig(name="test", models={"fast": "model-f", "balanced": "model-b"})
    assert config.api_key is None
    assert config.models["fast"] == "model-f"


def test_router_logs_selection():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    model, tier = router.select_with_tier(TaskType.SUMMARIZATION, input_length=100)
    assert isinstance(model, str)
    assert tier in (ModelTier.FAST, ModelTier.BALANCED, ModelTier.QUALITY)


def test_long_input_upgrades_from_fast():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    model = router.select(TaskType.SUMMARIZATION, input_length=100)
    long_model = router.select(TaskType.SUMMARIZATION, input_length=100_000)
    assert model != long_model


def test_unsupported_task_type_fallsback():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    from open_deep_research.api.model_router import _DEFAULT_TIER_MAP, _LONG_INPUT_THRESHOLD
    # Non-existent task enum value - use _resolve_tier directly to test fallback
    tier = router._resolve_tier("nonexistent", 100, None)
    assert tier == ModelTier.BALANCED


def test_empty_tier_map_fallback():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter(tier_map={})
    # Empty map means all tasks fall back to BALANCED via _resolve_tier
    tier = router._resolve_tier("anything", 100, None)
    assert tier == ModelTier.BALANCED


def test_select_with_tier_if_enabled_disabled():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    result = router.select_with_tier_if_enabled(TaskType.SUMMARIZATION, enabled=False)
    assert result is None


def test_complexity_hint_overrides_default():
    from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
    router = ModelRouter()
    fast_model = router.select(
        TaskType.REPORT_WRITING, input_length=100, complexity_hint=ModelTier.FAST
    )
    quality_model = router.select(TaskType.REPORT_WRITING, input_length=100)
    # FAST should differ from default QUALITY for report writing
    assert fast_model != quality_model


def test_thread_safety():
    import concurrent.futures
    from open_deep_research.api.model_router import ModelRouter, TaskType
    router = ModelRouter()

    def select():
        return router.select(TaskType.SUMMARIZATION, input_length=1000)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(select) for _ in range(20)]
        results = [f.result() for f in futures]
    assert all(r is not None for r in results)
