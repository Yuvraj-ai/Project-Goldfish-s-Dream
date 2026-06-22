from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsCollector:
    """Lightweight in-process Prometheus-compatible metrics collector."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, ...], int] = defaultdict(int)
        self._histograms: dict[tuple[str, ...], list[float]] = defaultdict(list)

    def inc(self, name: str, labels: dict[str, str] | None = None, value: int = 1) -> None:
        key = (name, *(f"{k}={v}" for k, v in (labels or {}).items()))
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (name, *(f"{k}={v}" for k, v in (labels or {}).items()))
        with self._lock:
            self._histograms[key].append(value)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, count in self._counters.items():
                name = key[0]
                label_str = "{" + ",".join(key[1:]) + "}" if len(key) > 1 else ""
                lines.append(f"# HELP {name} Counter")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{label_str} {count}")
            for key, values in self._histograms.items():
                name = key[0]
                label_str = "{" + ",".join(key[1:]) + "}" if len(key) > 1 else ""
                count = len(values)
                total = sum(values)
                lines.append(f"# HELP {name} Request latency seconds")
                lines.append(f"# TYPE {name} histogram")
                # Cumulative histogram buckets
                buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, float("inf")]
                for bucket in buckets:
                    le = float("inf") if bucket == float("inf") else bucket
                    le_label = "+Inf" if bucket == float("inf") else f"{le:.2f}"
                    bucket_count = sum(1 for v in values if v <= le)
                    bucket_labels = label_str.replace("}", f",le={le_label}}}") if label_str else f"{{le=\"{le_label}\"}}"
                    lines.append(f"{name}_bucket{bucket_labels} {bucket_count}")
                lines.append(f"{name}_count{label_str} {count}")
                lines.append(f"{name}_sum{label_str} {total}")
        return "\n".join(lines) + "\n"


metrics = MetricsCollector()
