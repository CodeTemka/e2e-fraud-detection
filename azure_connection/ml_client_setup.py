"""Backward compatible wrapper for creating an Azure ML client.

Prefer using :mod:`fraud_detection.azure.client` directly. This module exists
for legacy scripts that still import ``azure_connection.ml_client_setup``.
"""
from fraud_detection.azure.client import get_ml_client  # noqa: F401
from fraud_detection.azure.client import get_ml_client as ml_client  # type: ignore

__all__ = ["get_ml_client", "ml_client"]
