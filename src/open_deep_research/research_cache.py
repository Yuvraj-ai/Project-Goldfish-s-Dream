"""Search and evidence caching with mode-aware TTLs."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MODE_TTLS = {
    "news_or_current_events": 0,
    "academic_literature_review": 168,
    "technical_implementation": 72,
    "default": 24,
}


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    ttl_hours: int
    mode: str = "default"

    @property
    def is_expired(self) -> bool:
        if self.ttl_hours == 0:
            return True
        age_hours = (time.time() - self.created_at) / 3600
        return age_hours > self.ttl_hours

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


class ResearchCache:
    def __init__(self, cache_dir: str = ".cache/research"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CacheEntry] = {}
        self._load_from_disk()
        logger.info(
            "ResearchCache initialized: cache_dir=%s, entries_loaded=%d",
            self.cache_dir, len(self._cache),
        )

    def _load_from_disk(self):
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                for key, entry in data.items():
                    self._cache[key] = CacheEntry(
                        key=key, value=entry["value"], created_at=entry["created_at"],
                        ttl_hours=entry["ttl_hours"], mode=entry.get("mode", "default")
                    )
                logger.debug(
                    "Loaded %d cache entries from disk (%s)",
                    len(self._cache), cache_file,
                )
            except (json.JSONDecodeError, KeyError):
                logger.exception(
                    "Failed to load cache from %s; starting with empty cache", cache_file
                )
                self._cache = {}

    def _save_to_disk(self):
        cache_file = self.cache_dir / "cache.json"
        data = {}
        for key, entry in self._cache.items():
            data[key] = {"value": entry.value, "created_at": entry.created_at,
                         "ttl_hours": entry.ttl_hours, "mode": entry.mode}
        with open(cache_file, "w") as f:
            json.dump(data, f)

    @staticmethod
    def make_key(prefix: str, query: str, provider: str = "", mode: str = "default") -> str:
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        parts = [prefix, query_hash]
        if provider:
            parts.append(provider)
        if mode and mode != "default":
            parts.append(mode)
        key = ":".join(parts)
        logger.debug("Computed cache key '%s' (prefix=%s, mode=%s)", key, prefix, mode)
        return key

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            logger.debug("Cache MISS for key '%s' (not present)", key)
            return None
        if entry.is_expired:
            logger.warning(
                "Cache eviction: key '%s' expired (age=%.1fh, ttl=%dh); removing",
                key, entry.age_hours, entry.ttl_hours,
            )
            del self._cache[key]
            return None
        logger.debug(
            "Cache HIT for key '%s' (age=%.1fh, ttl=%dh)",
            key, entry.age_hours, entry.ttl_hours,
        )
        return entry.value

    def set(self, key: str, value: Any, mode: str = "default", ttl_hours: int | None = None):
        if ttl_hours is None:
            ttl_hours = MODE_TTLS.get(mode, MODE_TTLS["default"])
        logger.debug("Chosen TTL=%dh for key '%s' (mode=%s)", ttl_hours, key, mode)
        if ttl_hours == 0:
            logger.debug("Skipping cache set for key '%s' (ttl=0, non-cacheable mode)", key)
            return
        self._cache[key] = CacheEntry(key=key, value=value, created_at=time.time(),
                                       ttl_hours=ttl_hours, mode=mode)
        self._save_to_disk()
        logger.debug(
            "Cache SET key '%s' (mode=%s, ttl=%dh, total_entries=%d)",
            key, mode, ttl_hours, len(self._cache),
        )

    def invalidate(self, key: str) -> bool:
        if key in self._cache:
            logger.warning("Cache eviction: invalidating key '%s'", key)
            del self._cache[key]
            self._save_to_disk()
            return True
        logger.debug("Invalidate no-op: key '%s' not present", key)
        return False

    def clear(self):
        logger.warning("Clearing entire cache (%d entries evicted)", len(self._cache))
        self._cache.clear()
        self._save_to_disk()

    def get_search_results(self, query: str, provider: str = "", mode: str = "default") -> List[Dict] | None:
        logger.debug("get_search_results(provider=%s, mode=%s)", provider, mode)
        key = self.make_key("search", query, provider, mode)
        return self.get(key)

    def set_search_results(self, query: str, results: List[Dict], provider: str = "", mode: str = "default"):
        logger.debug(
            "set_search_results(provider=%s, mode=%s, result_count=%d)",
            provider, mode, len(results),
        )
        key = self.make_key("search", query, provider, mode)
        self.set(key, results, mode=mode)

    def get_evidence(self, topic: str, mode: str = "default") -> List[Dict] | None:
        logger.debug("get_evidence(mode=%s)", mode)
        key = self.make_key("evidence", topic, mode=mode)
        return self.get(key)

    def set_evidence(self, topic: str, evidence: List[Dict], mode: str = "default"):
        logger.debug("set_evidence(mode=%s, evidence_count=%d)", mode, len(evidence))
        key = self.make_key("evidence", topic, mode=mode)
        self.set(key, evidence, mode=mode)

    @property
    def stats(self) -> Dict:
        total = len(self._cache)
        expired = sum(1 for e in self._cache.values() if e.is_expired)
        logger.debug(
            "Cache stats: total=%d, active=%d, expired=%d",
            total, total - expired, expired,
        )
        return {"total_entries": total, "active_entries": total - expired,
                "expired_entries": expired, "hit_rate": "N/A (requires tracking)"}
