"""Centralized logging configuration for open_deep_research.

Built on the vendored ``pretty_logger`` (its :class:`ColoredFormatter` is the
reusable asset). Instead of each module calling ``pretty_logger.get_logger()``
(which would attach duplicate handlers per file and lose module-qualified
logger names), we attach handlers **once** to the package root logger
``open_deep_research`` and let every ``logging.getLogger(__name__)`` in the
codebase propagate up to it.

Deployer-facing behavior (things the upstream logger does not do):

* ``LOG_LEVEL``   — one of DEBUG/INFO/WARNING/ERROR/CRITICAL (default INFO).
* ``LOG_DIR``     — directory for the rotating file log (default ``logs``).
* ``LOG_FORMAT``  — ``color`` (default), ``plain``, or ``json``.
* ``NO_COLOR``    — if set (see https://no-color.org) force plain console.
* Color is auto-disabled when stdout is not a TTY (clean container/file logs),
  unless ``LOG_FORMAT=color`` is explicitly requested.
* File logging is best-effort: a read-only filesystem disables it with a
  warning instead of crashing the app.

Typical usage::

    from open_deep_research.logging_config import setup_logging
    setup_logging()                      # idempotent; call once at startup

    import logging
    logger = logging.getLogger(__name__)  # existing pattern keeps working
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler

from open_deep_research._vendor.pretty_logger import ColoredFormatter

# Root logger for the whole package. Every module's ``getLogger(__name__)``
# (e.g. "open_deep_research.deep_researcher") is a descendant and propagates
# here.
PACKAGE_LOGGER_NAME = "open_deep_research"

# Plain (no-ANSI) format mirroring pretty_logger's layout, plus the logger name
# so multi-module output is traceable in aggregated logs.
_PLAIN_FORMAT = "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)-8s %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Request-scoped correlation id, populated by the API middleware. Available to
# every formatter so a single request can be traced across modules.
request_id_var: ContextVar[str] = ContextVar("request_id")

_configured = False


class RequestIdFilter(logging.Filter):
    """Attach the current request id (if any) to every record as ``request_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.request_id = request_id_var.get()
        except LookupError:
            record.request_id = "-"
        return True


class JSONFormatter(logging.Formatter):
    """Structured single-line JSON, for log aggregators (Loki/Datadog/etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "location": f"{record.filename}:{record.lineno}",
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def _resolve_level(level: int | str | None) -> int:
    """Resolve an explicit level or the ``LOG_LEVEL`` env var to a logging int."""
    if isinstance(level, int):
        return level
    name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def _want_color(stream) -> bool:
    """Decide whether ANSI color is appropriate for ``stream``."""
    fmt = (os.getenv("LOG_FORMAT") or "").lower()
    if fmt == "color":
        return True  # explicit opt-in overrides TTY detection
    if fmt in ("plain", "json"):
        return False
    if os.getenv("NO_COLOR") is not None:
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def _console_formatter(stream) -> logging.Formatter:
    """Pick the console formatter from LOG_FORMAT / TTY detection."""
    if (os.getenv("LOG_FORMAT") or "").lower() == "json":
        return JSONFormatter()
    if _want_color(stream):
        return ColoredFormatter()
    return logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT)


def setup_logging(
    log_dir: str | None = None,
    backup_count: int = 30,
    level: int | str | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger. Idempotent unless ``force=True``.

    Args:
        log_dir: Directory for the rotating file log. Defaults to ``LOG_DIR``
            env or ``logs``.
        backup_count: Days of rotated logs to retain.
        level: Explicit level (int or name). Defaults to ``LOG_LEVEL`` env or
            ``INFO``.
        force: Re-run setup even if already configured (clears prior handlers).

    Returns:
        The configured ``open_deep_research`` package logger.
    """
    global _configured

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if _configured and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    resolved_level = _resolve_level(level)
    resolved_dir = log_dir or os.getenv("LOG_DIR") or "logs"

    logger.setLevel(resolved_level)
    request_id_filter = RequestIdFilter()

    # Console handler — colored for humans, plain/json for machines.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(resolved_level)
    console.setFormatter(_console_formatter(sys.stdout))
    console.addFilter(request_id_filter)
    logger.addHandler(console)

    # File handler — plain text, daily rotation, best-effort.
    try:
        os.makedirs(resolved_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(resolved_dir, "applog.log"),
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT))
        file_handler.addFilter(request_id_filter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("File logging disabled (could not use %s): %s", resolved_dir, exc)

    # Keep propagation ON. We attach handlers only to this package logger (never
    # the root logger), so records reach our handlers and still propagate to the
    # root — which normally has no handlers, so there is no duplicate output.
    # Propagation is what lets pytest's `caplog`, and any application-level root
    # aggregation, observe our logs. (pretty_logger disables it to guard against
    # a doubly-configured root; we prefer compatibility and simply never touch
    # root ourselves.)
    logger.propagate = True
    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, ensuring the package logging is configured first.

    Convenience for modules/scripts that want a one-liner. Existing code using
    ``logging.getLogger(__name__)`` continues to work unchanged as long as
    :func:`setup_logging` has been called at an entry point.
    """
    setup_logging()
    return logging.getLogger(name or PACKAGE_LOGGER_NAME)
