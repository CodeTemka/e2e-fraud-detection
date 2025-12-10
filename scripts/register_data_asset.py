from __future__ import annotations

import sys
from pathlib import Path

import typer
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.azure.client import get_ml_client  # noqa: E402

app = typer.Typer(help="Register a local fraud dataset as an Azure ML data asset.")


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
DEFAULT_NAME = "creditcard-data"
DEFAULT_VERSION = "1"
DEFAULT_DESCRIPTION = "Credit Card Fraud Detection Dataset"

PATH_OPTION = typer.Option(
    DEFAULT_DATA_PATH,
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to the local data file to register.",
)


@app.command()
def register(
    path: Path = PATH_OPTION,
    name: str = typer.Option(DEFAULT_NAME, help="Name of the data asset."),
    version: str = typer.Option(DEFAULT_VERSION, help="Version of the data asset."),
    description: str = typer.Option(DEFAULT_DESCRIPTION, help="Description for the data asset."),
) -> None:
    """Register or skip a data asset based on existence."""

    ml_client = get_ml_client()
    data_asset = Data(
        name=name,
        path=str(path.resolve()),
        description=description,
        type=AssetTypes.URI_FILE,
        version=version,
    )

    try:
        ml_client.data.get(name=name, version=version)
        typer.echo(f"Data asset '{name}' version '{version}' already exists. Skipping registration.")
    except ResourceNotFoundError:
        ml_client.data.create_or_update(data_asset)
        typer.echo(f"Registered data asset '{name}' version '{version}'.")


if __name__ == "__main__":
    app()
