"""Utilities for selecting registered Azure ML models for deployment."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegisteredModelChoice:
    """Minimal representation of a registered model option."""

    index: int
    name: str
    version: str | int
    description: str | None


def collect_registered_models(
    ml_client: MLClient, *, name_prefix: str | None = None
) -> list[Model]:
    """Return registered models (optionally filtered by name prefix)."""

    models = list(ml_client.models.list())
    if name_prefix:
        models = [m for m in models if str(m.name).startswith(name_prefix)]
    models = [m for m in models if "_output_mlflow_log_model_" not in str(m.name)]
    models.sort(key=lambda m: (str(m.name), str(m.version)))
    logger.info("Discovered registered models", extra={"count": len(models)})
    return models


def build_registered_models_table(models: Iterable[Model]) -> list[str]:
    """Return formatted table lines for CLI display."""

    choices: list[RegisteredModelChoice] = [
        RegisteredModelChoice(
            index=i + 1,
            name=str(m.name),
            version=getattr(m, "version", ""),
            description=(getattr(m, "description", None) or "").strip() or "-",
        )
        for i, m in enumerate(models)
    ]

    lines = ["# | Name | Version | Description", "-|-|-|-"]
    for choice in choices:
        lines.append(
            f"{choice.index} | {choice.name} | {choice.version} | {choice.description}"
        )
    return lines


def choose_registered_model(models: Sequence[Model], selection: int) -> Model:
    """Return the selected model by 1-based index."""

    if selection < 1 or selection > len(models):
        raise ValueError(f"Selection must be between 1 and {len(models)}")
    return models[selection - 1]


__all__ = [
    "RegisteredModelChoice",
    "collect_registered_models",
    "build_registered_models_table",
    "choose_registered_model",
]
