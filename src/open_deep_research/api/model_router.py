from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)


class ModelTier(str, enum.Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class TaskType(str, enum.Enum):
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SEARCH_QUERY_GENERATION = "search_query_generation"
    REPORT_WRITING = "report_writing"
    REVIEW = "review"
    PLANNING = "planning"
    REASONING = "reasoning"


_DEFAULT_TIER_MAP: dict[TaskType, ModelTier] = {
    TaskType.SUMMARIZATION: ModelTier.FAST,
    TaskType.CLASSIFICATION: ModelTier.FAST,
    TaskType.SEARCH_QUERY_GENERATION: ModelTier.FAST,
    TaskType.EXTRACTION: ModelTier.BALANCED,
    TaskType.PLANNING: ModelTier.BALANCED,
    TaskType.REPORT_WRITING: ModelTier.QUALITY,
    TaskType.REVIEW: ModelTier.QUALITY,
    TaskType.REASONING: ModelTier.QUALITY,
}

_LONG_INPUT_THRESHOLD = 8000


class ProviderConfig:
    def __init__(
        self,
        name: str,
        models: dict[ModelTier, str],
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.models = models
        self.api_key = api_key


class ModelRouter:
    def __init__(
        self,
        tier_map: dict[TaskType, ModelTier] | None = None,
        providers: list[ProviderConfig] | None = None,
        long_input_threshold: int = _LONG_INPUT_THRESHOLD,
    ) -> None:
        self._tier_map = tier_map or dict(_DEFAULT_TIER_MAP)
        self._providers = providers or [
            ProviderConfig(
                name="google",
                models={
                    ModelTier.FAST: "google_genai:gemini-2.5-flash",
                    ModelTier.BALANCED: "google_genai:gemini-1.5-pro",
                    ModelTier.QUALITY: "google_genai:gemini-2.5-pro",
                },
            ),
        ]
        self._long_input_threshold = long_input_threshold

    def select(
        self,
        task_type: TaskType,
        input_length: int = 0,
        complexity_hint: ModelTier | None = None,
    ) -> str:
        tier = self._resolve_tier(task_type, input_length, complexity_hint)
        model = self._providers[0].models.get(tier, self._providers[0].models[ModelTier.BALANCED])
        logger.debug("Selected model %s for task %s (tier=%s)", model, task_type.value, tier.value)
        return model

    def select_with_tier(
        self,
        task_type: TaskType,
        input_length: int = 0,
        complexity_hint: ModelTier | None = None,
    ) -> tuple[str, ModelTier]:
        tier = self._resolve_tier(task_type, input_length, complexity_hint)
        model = self._providers[0].models.get(tier, self._providers[0].models[ModelTier.BALANCED])
        return model, tier

    def _resolve_tier(
        self,
        task_type: TaskType,
        input_length: int,
        complexity_hint: ModelTier | None,
    ) -> ModelTier:
        if complexity_hint:
            return complexity_hint
        base = self._tier_map.get(task_type, ModelTier.BALANCED)
        if input_length > self._long_input_threshold and base == ModelTier.FAST:
            return ModelTier.BALANCED
        return base
