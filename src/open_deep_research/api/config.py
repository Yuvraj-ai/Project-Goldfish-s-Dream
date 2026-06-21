from __future__ import annotations

from pydantic import BaseModel


class ApiConfig(BaseModel):
    enable_rest_api: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_db_path: str = "research.db"
    api_key: str | None = None
    max_concurrent_runs: int = 3
    api_keys: dict[str, dict[str, str | list[str]]] = {}
