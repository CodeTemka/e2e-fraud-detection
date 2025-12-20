"""Centralized logging configuration."""
from __future__ import annotations

import logging

_LOGGER_NAME = "fraud_detection"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the fraud_detection project (configured once)."""
    # Configure the project root logger exactly once
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%b %d %H:%M",  # e.g., Dec 19 16:30
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False  # avoid double logging via global root logger

    # Return either the root logger or a child logger
    logger_name = name or _LOGGER_NAME
    logger = logging.getLogger(logger_name)

    if logger_name != _LOGGER_NAME:
        logger.setLevel(logging.NOTSET)  # inherit from root
        logger.propagate = True          # send records to root handler

    return logger


__all__ = ["get_logger"]
