"""Tests for centralized logging configuration (open_deep_research.logging_config)."""

import json
import logging

import pytest

from open_deep_research import logging_config as lc


@pytest.fixture
def restore_package_logger():
    """Snapshot and restore the package logger + module _configured flag.

    setup_logging() mutates the shared "open_deep_research" logger and a module
    global; without this, tests would leak handlers into each other and into
    the rest of the suite.
    """
    logger = logging.getLogger(lc.PACKAGE_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    saved_configured = lc._configured
    try:
        yield logger
    finally:
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate
        lc._configured = saved_configured


# ── _resolve_level ────────────────────────────────────────────────────────

def test_resolve_level_explicit_int():
    assert lc._resolve_level(logging.WARNING) == logging.WARNING


def test_resolve_level_explicit_name():
    assert lc._resolve_level("debug") == logging.DEBUG


def test_resolve_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert lc._resolve_level(None) == logging.ERROR


def test_resolve_level_default_is_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert lc._resolve_level(None) == logging.INFO


def test_resolve_level_invalid_name_falls_back_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert lc._resolve_level("NOT_A_LEVEL") == logging.INFO


# ── _want_color ───────────────────────────────────────────────────────────

class _FakeStream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_want_color_explicit_color_overrides_non_tty(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert lc._want_color(_FakeStream(tty=False)) is True


def test_want_color_plain_and_json_disable(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "plain")
    assert lc._want_color(_FakeStream(tty=True)) is False
    monkeypatch.setenv("LOG_FORMAT", "json")
    assert lc._want_color(_FakeStream(tty=True)) is False


def test_want_color_no_color_env_disables(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert lc._want_color(_FakeStream(tty=True)) is False


def test_want_color_follows_tty_when_unset(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert lc._want_color(_FakeStream(tty=True)) is True
    assert lc._want_color(_FakeStream(tty=False)) is False


# ── _console_formatter ────────────────────────────────────────────────────

def test_console_formatter_json(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    assert isinstance(lc._console_formatter(_FakeStream(tty=True)), lc.JSONFormatter)


def test_console_formatter_color(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "color")
    from open_deep_research._vendor.pretty_logger import ColoredFormatter
    assert isinstance(lc._console_formatter(_FakeStream(tty=False)), ColoredFormatter)


def test_console_formatter_plain(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "plain")
    fmt = lc._console_formatter(_FakeStream(tty=True))
    assert type(fmt) is logging.Formatter  # noqa: E721 - exact type, not a subclass


# ── JSONFormatter + RequestIdFilter ───────────────────────────────────────

def test_json_formatter_emits_valid_json_with_fields():
    record = logging.LogRecord(
        name="open_deep_research.x", level=logging.INFO, pathname="x.py",
        lineno=42, msg="hello %s", args=("world",), exc_info=None,
    )
    record.filename = "x.py"
    out = lc.JSONFormatter().format(record)
    parsed = json.loads(out)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello world"
    assert parsed["logger"] == "open_deep_research.x"
    assert parsed["request_id"] == "-"  # not set → default


def test_request_id_filter_sets_default_when_unset():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="x.py", lineno=1,
        msg="m", args=(), exc_info=None,
    )
    assert lc.RequestIdFilter().filter(record) is True
    assert record.request_id == "-"


def test_request_id_filter_uses_context_value():
    token = lc.request_id_var.set("abc-123")
    try:
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="x.py", lineno=1,
            msg="m", args=(), exc_info=None,
        )
        lc.RequestIdFilter().filter(record)
        assert record.request_id == "abc-123"
    finally:
        lc.request_id_var.reset(token)


# ── setup_logging ─────────────────────────────────────────────────────────

def test_setup_logging_attaches_console_and_file_handlers(tmp_path, restore_package_logger):
    logger = lc.setup_logging(log_dir=str(tmp_path), level=logging.DEBUG, force=True)
    assert logger.name == lc.PACKAGE_LOGGER_NAME
    assert logger.level == logging.DEBUG
    # Propagation stays ON so caplog / root aggregation can observe our logs.
    assert logger.propagate is True
    handler_types = {type(h).__name__ for h in logger.handlers}
    assert "StreamHandler" in handler_types
    assert "TimedRotatingFileHandler" in handler_types
    assert (tmp_path / "applog.log").exists()


def test_setup_logging_is_idempotent(tmp_path, restore_package_logger):
    lc._configured = False
    first = lc.setup_logging(log_dir=str(tmp_path), force=True)
    count_after_first = len(first.handlers)
    # Second call without force must NOT add more handlers.
    second = lc.setup_logging(log_dir=str(tmp_path))
    assert len(second.handlers) == count_after_first


def test_setup_logging_force_replaces_handlers(tmp_path, restore_package_logger):
    lc.setup_logging(log_dir=str(tmp_path), force=True)
    lc.setup_logging(log_dir=str(tmp_path), force=True)
    logger = logging.getLogger(lc.PACKAGE_LOGGER_NAME)
    # Force clears then re-adds: console + file = 2 (not accumulating).
    assert len(logger.handlers) == 2


def test_setup_logging_level_from_env(tmp_path, monkeypatch, restore_package_logger):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    logger = lc.setup_logging(log_dir=str(tmp_path), force=True)
    assert logger.level == logging.WARNING


def test_setup_logging_file_disabled_on_bad_dir(tmp_path, restore_package_logger):
    # Point log_dir at a path under a regular file → makedirs raises OSError,
    # and setup must degrade gracefully (console only), not crash.
    a_file = tmp_path / "iamafile"
    a_file.write_text("x")
    logger = lc.setup_logging(log_dir=str(a_file / "sub"), force=True)
    handler_types = {type(h).__name__ for h in logger.handlers}
    assert "StreamHandler" in handler_types
    assert "TimedRotatingFileHandler" not in handler_types


def test_child_logger_propagates_to_package_handlers(tmp_path, restore_package_logger, capsys):
    lc.setup_logging(log_dir=str(tmp_path), level=logging.INFO, force=True)
    child = logging.getLogger("open_deep_research.some_module")
    child.info("propagated message")
    out = capsys.readouterr().out
    assert "propagated message" in out


def test_get_logger_triggers_setup(tmp_path, monkeypatch, restore_package_logger):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    lc._configured = False
    for h in list(logging.getLogger(lc.PACKAGE_LOGGER_NAME).handlers):
        logging.getLogger(lc.PACKAGE_LOGGER_NAME).removeHandler(h)
    returned = lc.get_logger("open_deep_research.child")
    assert returned.name == "open_deep_research.child"
    assert lc._configured is True
    assert logging.getLogger(lc.PACKAGE_LOGGER_NAME).handlers
