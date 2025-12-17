"""Centralized logging configuration."""
from __future__ import annotations

import logging

_LOGGER_NAME = "fraud_detection"


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
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

__all__ = ["get_logger"]