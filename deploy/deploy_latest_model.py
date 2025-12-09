"""Deploy the latest version of a registered model to an Azure ML endpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deploy.manage_online_endpoint import create_endpoint, deploy_model, update_traffic
from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_INSTANCE_TYPE = "Standard_E4s_v3"
DEFAULT_INSTANCE_COUNT = 1


def get_latest_model_version(model_name: str) -> str:
    """Return the newest version for the given *model_name*."""

    ml_client = get_ml_client()
    models = list(ml_client.models.list(name=model_name))
    if not models:
        raise RuntimeError(f"Model '{model_name}' does not exist in the workspace.")

    def _version_key(model) -> tuple[int, str]:
        version = str(model.version)
        return (int(version), version) if version.isdigit() else (0, version)

    latest = sorted(models, key=_version_key, reverse=True)[0]
    logger.info(
        "Selected latest model version", extra={"model_name": latest.name, "version": latest.version}
    )
    return str(latest.version)


def ensure_endpoint(endpoint_name: str) -> str:
    """Ensure an endpoint exists and return its name."""

    ml_client = get_ml_client()
    try:
        ml_client.online_endpoints.get(endpoint_name)
        return endpoint_name
    except ResourceNotFoundError:
        logger.info("Endpoint %s not found; creating it now", endpoint_name)
        return create_endpoint(name=endpoint_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the latest model to an online endpoint.")
    parser.add_argument("--model-name", required=True, help="Registered model name to deploy")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint to deploy to")
    parser.add_argument("--deployment-name", default="blue", help="Deployment slot name")
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"Instance SKU for deployment (default: {DEFAULT_INSTANCE_TYPE})",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=DEFAULT_INSTANCE_COUNT,
        help=f"Number of instances (default: {DEFAULT_INSTANCE_COUNT})",
    )

    args = parser.parse_args()

    endpoint_name = ensure_endpoint(args.endpoint_name)
    version = get_latest_model_version(args.model_name)

    deployment = deploy_model(
        endpoint_name=endpoint_name,
        model_name=args.model_name,
        model_version=version,
        deployment_name=args.deployment_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    update_traffic(endpoint_name=endpoint_name, traffic={deployment.name: 100})
    logger.info(
        "Deployment successful",
        extra={
            "endpoint_name": endpoint_name,
            "deployment": deployment.name,
            "model": args.model_name,
            "version": version,
        },
    )


if __name__ == "__main__":
    main()
