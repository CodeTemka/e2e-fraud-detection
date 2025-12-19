"""Centralized logging configuration."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

_LOGGER_NAME = "fraud_detection"


class _UTCFormatter(logging.Formatter):
    """Formatter that always emits UTC timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        if datefmt:
            return timestamp.strftime(datefmt)
        # Use ISO 8601 with Zulu suffix and second precision by default
        return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the fraud detection module.

    Args:
        name: The name of the logger. If not provided, the root fraud_detection logger will be used.
    Returns:
        A configured logger instance.
    """
    logger_name = name or _LOGGER_NAME
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = _UTCFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%SZ",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

__all__ = ["get_logger"]
