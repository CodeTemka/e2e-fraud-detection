"""Configuration utilities for the fraud detection project."""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _parse_env_file(env_file: Path) -> Iterable[tuple[str, str]]:
    """Parse simple KEY=VALUE lines from a .env file."""

    if not env_file.exists():
        return []

    pairs: list[tuple[str, str]] = []
    with env_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", maxsplit=1)
            pairs.append((key.strip(), value.strip()))
    return pairs


def load_environment(env_file: str | Path | None = None) -> None:
    """Load environment variables from an optional .env file.

    The call is idempotent and safe to invoke multiple times.
    """

    env_path = Path(env_file) if env_file else Path(".env")
    for key, value in _parse_env_file(env_path):
        os.environ.setdefault(key, value)


@dataclass
class AzureMLConfig:
    """Minimal configuration required to connect to Azure ML."""

    subscription_id: str
    resource_group: str
    workspace_name: str

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> AzureMLConfig:
        """Create a configuration object from environment variables."""

        load_environment(env_file)

        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace_name = os.getenv("WORKSPACE_NAME")

        missing = [
            key
            for key, value in {
                "SUBSCRIPTION_ID": subscription_id,
                "RESOURCE_GROUP": resource_group,
                "WORKSPACE_NAME": workspace_name,
            }.items()
            if not value
        ]

        if missing:
            missing_list = ", ".join(missing)
            raise OSError(
                f"Missing required environment variables for Azure ML: {missing_list}."
                " Define them in the shell or a .env file."
            )

        return cls(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )


def ensure_output_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the Path."""

    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
