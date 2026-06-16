"""Tests for research strategy classifier."""
from open_deep_research.configuration import Configuration, ResearchMode


class TestResearchMode:
    def test_all_modes_exist(self):
        assert len(list(ResearchMode)) == 9

    def test_mode_values_are_strings(self):
        for mode in ResearchMode:
            assert isinstance(mode.value, str)

    def test_comparison_mode(self):
        assert ResearchMode.COMPARISON.value == "comparison"

    def test_custom_mode(self):
        assert ResearchMode.CUSTOM.value == "custom"


class TestClassifierConfig:
    def test_default_mode_is_custom(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.default_research_mode == ResearchMode.CUSTOM

    def test_classification_enabled_by_default(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.enable_mode_classification is True

    def test_confidence_threshold_default(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.classifier_confidence_threshold == 0.6

    def test_classifier_model_default_is_none(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.classifier_model is None


class TestStateFields:
    def test_agent_state_has_research_mode(self):
        from open_deep_research.state import AgentState
        assert "research_mode" in AgentState.__annotations__

    def test_agent_state_has_report_profile(self):
        from open_deep_research.state import AgentState
        assert "report_profile" in AgentState.__annotations__

    def test_agent_state_has_research_plan(self):
        from open_deep_research.state import AgentState
        assert "research_plan" in AgentState.__annotations__

    def test_supervisor_state_has_research_mode(self):
        from open_deep_research.state import SupervisorState
        assert "research_mode" in SupervisorState.__annotations__

    def test_supervisor_state_has_perspectives(self):
        from open_deep_research.state import SupervisorState
        assert "perspectives" in SupervisorState.__annotations__

    def test_agent_input_state_has_document_paths(self):
        from open_deep_research.state import AgentInputState
        assert "document_paths" in AgentInputState.__annotations__


class TestResearchPlanExtended:
    def test_plan_creation(self):
        from open_deep_research.state import ResearchPlanExtended
        plan = ResearchPlanExtended(
            objective="Test",
            subquestions=["Q1", "Q2"],
            search_strategy={"Q1": ["web"], "Q2": ["academic"]},
        )
        assert len(plan.subquestions) == 2

    def test_plan_defaults(self):
        from open_deep_research.state import ResearchPlanExtended
        plan = ResearchPlanExtended(objective="Test")
        assert plan.subquestions == []
        assert plan.search_strategy == {}

    def test_plan_serialization(self):
        from open_deep_research.state import ResearchPlanExtended
        plan = ResearchPlanExtended(objective="Test", subquestions=["Q1"])
        d = plan.model_dump()
        assert isinstance(d, dict)
        assert d["objective"] == "Test"


class TestPlannerConfig:
    def test_default_review_mode(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.plan_review_mode == "none"

    def test_max_plan_revisions(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.max_plan_revisions == 2

    def test_max_subquestions(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.max_subquestions == 7

    def test_min_subquestions(self):
        config = Configuration(research_model="test-model", research_model_max_tokens=1000)
        assert config.min_subquestions == 3
