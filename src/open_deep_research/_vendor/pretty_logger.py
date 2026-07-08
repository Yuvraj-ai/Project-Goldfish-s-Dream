"""Pretty Logger — colorful, feature-rich logging utility.

Vendored from https://github.com/Yuvraj-ai/pretty_logger (GPL-3.0). Upstream
ships no pip package ("copy pretty_logger.py to your project"), so the source
is included here. Only this provenance docstring has been prepended; the code
below is the upstream snapshot. The reusable asset for this project is
``ColoredFormatter``; ``get_logger`` is retained for API-compatibility, but the
package configures logging centrally via ``open_deep_research.logging_config``.

Do not edit the code below — treat it as an upstream snapshot. This module is
excluded from ruff via ``extend-exclude`` in pyproject.toml.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import os
import inspect

# Default log directory
LOG_DIR = "logs"

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to the logs based on the log level.
    """
    # ANSI escape codes
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Colors
    CYAN = "\033[96m"       # For DEBUG
    GREEN = "\033[92m"      # For INFO
    YELLOW = "\033[93m"     # For Dates and WARNING
    RED = "\033[91m"        # For ERROR
    MAGENTA = "\033[95m"    # For CRITICAL
    GRAY = "\033[90m"       # For file name

    # Level to color mapping
    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: f"{BOLD}{MAGENTA}",
    }

    def format(self, record):
        # Get color for the log level
        msg_color = self.LEVEL_COLORS.get(record.levelno, self.GREEN)

        # Format the timestamp
        date_format = "%Y-%m-%d %H:%M:%S"
        record.asctime = self.formatTime(record, date_format)

        # Build the colored string:
        # [Date (Yellow)] [FileName:LineNo (Gray)] [Level (Color)] Message (Color)
        colored_date = f"{self.YELLOW}{record.asctime}{self.RESET}"
        colored_location = f"{self.GRAY}[{record.filename}:{record.lineno}]{self.RESET}"
        colored_level = f"{msg_color}{record.levelname:8}{self.RESET}"
        colored_msg = f"{msg_color}{record.getMessage()}{self.RESET}"

        # Include exception info if present
        formatted = f"{colored_date} {colored_location} {colored_level} - {colored_msg}"

        if record.exc_info:
            # Format exception with red color
            exc_text = self.formatException(record.exc_info)
            formatted += f"\n{self.RED}{exc_text}{self.RESET}"

        return formatted


def get_logger(log_dir=LOG_DIR, backup_count=30):
    """
    Creates and returns a configured logger instance.
    Automatically detects the calling file's name.

    Args:
        log_dir: Directory to store log files (default: 'logs/')
        backup_count: Number of days to keep log files (default: 30)

    Returns:
        Configured logger instance with:
        - Console output: Colored by log level
        - File output: Plain text, daily rotation, auto-cleanup
    """
    # Automatically get the caller's filename
    caller_frame = inspect.stack()[1]
    caller_filename = os.path.basename(caller_frame.filename)
    caller_name = os.path.splitext(caller_filename)[0]  # Remove .py extension

    logger = logging.getLogger(caller_name)
    logger.setLevel(logging.DEBUG)  # Capture all logs

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 1. Console Handler (Colored)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        # 2. File Handler with daily rotation
        log_filepath = os.path.join(log_dir, "applog.log")

        # Use TimedRotatingFileHandler for automatic daily rotation and cleanup
        file_handler = TimedRotatingFileHandler(
            log_filepath,
            when="midnight",      # Rotate at midnight
            interval=1,           # Every 1 day
            backupCount=backup_count,  # Keep logs for X days
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # File formatter includes filename and line number for traceability
        file_formatter = logging.Formatter(
            '%(asctime)s [%(filename)s:%(lineno)d] %(levelname)-8s - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    # Prevent logs from propagating to the root logger (avoids duplicates)
    logger.propagate = False

    return logger
