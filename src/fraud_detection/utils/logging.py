"""Centralized logging configuration."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_LOGGER_NAME = "fraud_detection"
_STANDARD_LOG_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    """Format logs as structured JSON for downstream ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_")
        }
        for key, value in extras.items():
            try:
                json.dumps(value)
            except TypeError:
                extras[key] = str(value)
        payload.update(extras)
        return json.dumps(payload)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the fraud_detection project (configured once)."""
    # Configure the project root logger exactly once
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False  # avoid double logging via global root logger

    # Return either the root logger or a child logger
    logger_name = name or _LOGGER_NAME
    logger = logging.getLogger(logger_name)

    if logger_name != _LOGGER_NAME:
        logger.setLevel(logging.NOTSET)  # inherit from root
        logger.propagate = True  # send records to root handler

    return logger


__all__ = ["get_logger"]
