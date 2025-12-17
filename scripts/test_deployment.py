from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from azure.core.exceptions import ResourceNotFoundError

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.azure.client import get_ml_client  # noqa: E402
from fraud_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_REQUEST_FILE = ROOT_DIR / "sample_request.json"

app = typer.Typer(help="Smoke test a managed online endpoint deployment.")


@app.command()
def invoke(
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", help="Endpoint to call")],
    deployment_name: Annotated[
        str, typer.Option("--deployment-name", help="Deployment slot name")
    ] = "blue",
    request_file: Annotated[
        Path,
        typer.Option(
            "--request-file",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to JSON request payload",
        ),
    ] = DEFAULT_REQUEST_FILE,
) -> None:
    """Invoke the endpoint and log the response."""

    ml_client = get_ml_client()
    try:
        ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError as exc:  # pragma: no cover - runtime safeguard
        raise typer.BadParameter(f"Endpoint '{endpoint_name}' does not exist") from exc

    logger.info(
        "Invoking endpoint", extra={"endpoint_name": endpoint_name, "deployment_name": deployment_name}
    )
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=request_file,
    )

    logger.info("Invocation response: %s", response)


if __name__ == "__main__":
    app()
