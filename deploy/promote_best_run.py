"""Register the best completed AutoML job under a stable model name.

This helper keeps CI/CD-friendly behaviour by reusing the latest completed
AutoML parent job, registering its best child run as an MLflow model with a
predictable name so it can be deployed reliably.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import typer
from azure.ai.ml import MLClient

from fraud_detection.azure.client import get_ml_client
from fraud_detection.cli import DEFAULT_EXPERIMENTS
from fraud_detection.training.registration import list_completed_jobs, register_best_child_run
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Promote the best AutoML run to a stable model name for deployment.")


def _job_creation_time(ml_client: MLClient, job_name: str) -> datetime:
    try:
        job = ml_client.jobs.get(job_name)
        created = getattr(job.creation_context, "creation_time", None) or getattr(
            job.creation_context, "created_at", None
        )
        if created:
            return datetime.fromisoformat(str(created))
    except Exception:  # pragma: no cover - defensive logging only
        logger.exception("Could not resolve creation time for job %s", job_name)
    return datetime.min


@app.command()
def promote(
    model_name: str = typer.Option(
        "fraud-detection-prod",
        help="Stable model name to publish the best AutoML child run under.",
    ),
    experiments: Iterable[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Experiment names to scan; defaults to built-in AutoML experiments.",
    ),
) -> None:
    """Register the best child run from the newest completed AutoML job."""

    ml_client = get_ml_client()
    experiment_names = list(experiments) if experiments is not None else list(DEFAULT_EXPERIMENTS)
    completed_jobs = list_completed_jobs(ml_client, experiment_names)
    if not completed_jobs:
        typer.echo("No completed AutoML jobs found; skipping promotion.")
        raise typer.Exit(code=0)

    latest_job = max(
        completed_jobs.items(),
        key=lambda item: _job_creation_time(ml_client, job_name=item[0]),
    )
    job_name, experiment_name = latest_job

    model = register_best_child_run(
        ml_client,
        job_name=job_name,
        experiment_name=experiment_name,
        model_name=model_name,
    )

    if model:
        typer.echo(f"Registered model {model.name} version {model.version} from job {job_name}.")
    else:
        typer.echo("No model was registered from the latest job.")


if __name__ == "__main__":
    app()
