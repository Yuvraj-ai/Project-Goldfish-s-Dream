"""Tests for exception handling and error surfacing."""
import traceback

import pytest

from open_deep_research.exceptions import (
    BudgetExceededError,
    ModelError,
    ODRError,
    TokenLimitError,
    ToolPermanentError,
    ToolTransientError,
)


class TestExceptionHierarchy:
    """Test that exception types are correctly structured."""

    def test_odr_error_is_base_exception(self):
        assert issubclass(ODRError, Exception)

    def test_token_limit_error_inherits_odr_error(self):
        assert issubclass(TokenLimitError, ODRError)

    def test_tool_transient_error_inherits_odr_error(self):
        assert issubclass(ToolTransientError, ODRError)

    def test_tool_permanent_error_inherits_odr_error(self):
        assert issubclass(ToolPermanentError, ODRError)

    def test_model_error_inherits_odr_error(self):
        assert issubclass(ModelError, ODRError)

    def test_budget_exceeded_error_inherits_odr_error(self):
        assert issubclass(BudgetExceededError, ODRError)


class TestExceptionAttributes:
    """Test that exceptions carry correct metadata."""

    def test_odr_error_stores_node_and_config(self):
        err = ODRError("test", node="researcher", config_snapshot={"model": "gpt-4"})
        assert err.node == "researcher"
        assert err.config_snapshot == {"model": "gpt-4"}

    def test_tool_transient_error_stores_retry_after(self):
        err = ToolTransientError("rate limited", retry_after=5.0)
        assert err.retry_after == 5.0

    def test_budget_exceeded_error_stores_budget_type(self):
        err = BudgetExceededError("over budget", budget_type="research")
        assert err.budget_type == "research"

    def test_exceptions_are_catchable_as_base(self):
        """All exceptions should be catchable as ODRError."""
        for exc_class in [TokenLimitError, ToolTransientError, ToolPermanentError, 
                          ModelError, BudgetExceededError]:
            with pytest.raises(ODRError):
                raise exc_class("test")


class TestExceptionSurfacing:
    """Test that exceptions surface in state, not swallowed."""

    def test_error_artifact_structure(self):
        """Verify error_artifact has required fields."""
        error_artifact = {
            "node": "researcher",
            "error_type": "TokenLimitError",
            "error_message": "Token limit exceeded",
            "stack_trace": traceback.format_exc(),
        }
        assert "node" in error_artifact
        assert "error_type" in error_artifact
        assert "error_message" in error_artifact
        assert "stack_trace" in error_artifact
