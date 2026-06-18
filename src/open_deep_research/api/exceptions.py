from __future__ import annotations


class APIError(Exception):
    def __init__(self, detail: str, code: str, status_code: int = 500) -> None:
        self.detail = detail
        self.code = code
        self.status_code = status_code
        super().__init__(detail)


class NotFoundError(APIError):
    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(detail, "not_found", 404)


class ConflictError(APIError):
    def __init__(self, detail: str = "Resource already exists") -> None:
        super().__init__(detail, "conflict", 409)


class AuthError(APIError):
    def __init__(self, detail: str = "Unauthorized") -> None:
        super().__init__(detail, "unauthorized", 401)
