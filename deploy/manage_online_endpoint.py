"""Utilities for managing Azure ML online endpoints for fraud detection.

This module consolidates endpoint creation, deployment, traffic splitting,
re-deployment, and teardown utilities that previously lived in several small
scripts. It exposes a simple CLI so common operations can be performed with a
single entry point while keeping reusable functions for programmatic use.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def create_endpoint(name: str | None = None, description: str | None = None) -> str:
    """Create a managed online endpoint and return its name."""
    ml_client = get_ml_client()
    endpoint_name = name or f"fraud-detection-{uuid.uuid4().hex[:4]}"
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description=description or "Online endpoint for fraud detection model",
        auth_mode="key",
        tags={"project": "fraud-detection"},
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    return endpoint_name


def deploy_model(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    deployment_name: str = "blue",
    instance_type: str = "Standard_E8s_v3",
    instance_count: int = 1,
) -> ManagedOnlineDeployment:
    """Deploy a registered model to an existing endpoint."""
    ml_client = get_ml_client()
    model = ml_client.models.get(name=model_name, version=model_version)

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type=instance_type,
        instance_count=instance_count,
    )

    return ml_client.begin_create_or_update(deployment).result()


def update_traffic(endpoint_name: str, traffic: Dict[str, int]) -> None:
    """Update traffic allocation for deployments on an endpoint."""
    ml_client = get_ml_client()
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = traffic
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def delete_endpoint(endpoint_name: str) -> None:
    """Delete an online endpoint and its deployments."""
    ml_client = get_ml_client()
    ml_client.online_endpoints.begin_delete(name=endpoint_name).result()


# CLI utilities

def _parse_traffic_mapping(mapping: str) -> Dict[str, int]:
    if "=" not in mapping:
        raise argparse.ArgumentTypeError(
            f"Traffic mapping '{mapping}' must be in the form <deployment>=<percentage>."
        )
    deployment, percent = mapping.split("=", maxsplit=1)
    try:
        return {deployment: int(percent)}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Traffic percentage for '{deployment}' must be an integer."
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Azure ML online endpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a managed online endpoint")
    create_parser.add_argument("--name", help="Name of the endpoint (random if omitted)")
    create_parser.add_argument("--description", help="Description for the endpoint")

    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model to an endpoint")
    deploy_parser.add_argument("endpoint_name", help="Existing endpoint name")
    deploy_parser.add_argument("model_name", help="Registered model name")
    deploy_parser.add_argument("model_version", help="Registered model version")
    deploy_parser.add_argument("--deployment-name", default="blue", help="Deployment name")
    deploy_parser.add_argument(
        "--instance-type",
        default="Standard_E8s_v3",
        help="Azure ML instance type",
    )
    deploy_parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances to provision",
    )

    traffic_parser = subparsers.add_parser(
        "traffic", help="Update traffic allocation for deployments on an endpoint"
    )
    traffic_parser.add_argument("endpoint_name", help="Endpoint to update")
    traffic_parser.add_argument(
        "traffic",
        nargs="+",
        type=_parse_traffic_mapping,
        help="Traffic mapping(s) in the form <deployment>=<percentage>",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete an endpoint")
    delete_parser.add_argument("endpoint_name", help="Endpoint to delete")

    return parser


def _handle_create(args: argparse.Namespace) -> None:
    endpoint_name = create_endpoint(name=args.name, description=args.description)
    logger.info("Created endpoint", extra={"endpoint_name": endpoint_name})


def _handle_deploy(args: argparse.Namespace) -> None:
    deployment = deploy_model(
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=args.model_version,
        deployment_name=args.deployment_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )
    logger.info(
        "Deployment is ready",
        extra={"deployment_name": deployment.name, "endpoint_name": deployment.endpoint_name},
    )


def _handle_traffic(args: argparse.Namespace) -> None:
    combined_traffic: Dict[str, int] = {}
    for mapping in args.traffic:
        combined_traffic.update(mapping)
    update_traffic(endpoint_name=args.endpoint_name, traffic=combined_traffic)
    logger.info(
        "Updated endpoint traffic",
        extra={"endpoint_name": args.endpoint_name, "traffic": combined_traffic},
    )


def _handle_delete(args: argparse.Namespace) -> None:
    delete_endpoint(endpoint_name=args.endpoint_name)
    logger.info("Deleted endpoint", extra={"endpoint_name": args.endpoint_name})


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "create": _handle_create,
        "deploy": _handle_deploy,
        "traffic": _handle_traffic,
        "delete": _handle_delete,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
