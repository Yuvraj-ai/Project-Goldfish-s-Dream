"""Custom exception hierarchy for Open Deep Research."""


class ODRError(Exception):
    """Base exception for all ODR errors."""
    
    def __init__(self, message: str, node: str = "", config_snapshot: dict = None):
        """Initialize ODR error with context information.
        
        Args:
            message: Human-readable error description
            node: Graph node where the error occurred
            config_snapshot: Relevant configuration at time of error
        """
        super().__init__(message)
        self.node = node
        self.config_snapshot = config_snapshot or {}


class TokenLimitError(ODRError):
    """Raised when a model exceeds its token limit."""
    pass


class ToolTransientError(ODRError):
    """Raised on temporary tool failures (retryable)."""
    
    def __init__(self, message: str, retry_after: float = 1.0, **kwargs):
        """Initialize transient error with retry delay.
        
        Args:
            message: Human-readable error description
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments passed to ODRError
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ToolPermanentError(ODRError):
    """Raised on permanent tool failures (non-retryable)."""
    pass


class ModelError(ODRError):
    """Raised when an LLM call fails."""
    pass


class BudgetExceededError(ODRError):
    """Raised when a budget limit is exceeded."""
    
    def __init__(self, message: str, budget_type: str = "total", **kwargs):
        """Initialize budget exceeded error with type info.
        
        Args:
            message: Human-readable error description
            budget_type: Type of budget exceeded (e.g., 'total', 'research')
            **kwargs: Additional arguments passed to ODRError
        """
        super().__init__(message, **kwargs)
        self.budget_type = budget_type
