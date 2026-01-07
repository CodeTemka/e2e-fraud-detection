from __future__ import annotations

import argparse
import json
from pathlib import Path

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import ROOT_DIR, get_settings
from fraud_detection.serving.deploy import deploy_model, resolve_scoring_environment
from fraud_detection.serving.online_endpoint import ensure_endpoint, set_initial_traffic
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def register_model_entrypoint(args: argparse.Namespace) -> None:
    settings = get_settings()
    ml_client = get_ml_client(settings=settings)

    model_dir = Path(args.model_dir)
    artifacts_dir = model_dir / "model"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Model artifacts not found in {artifacts_dir}")

    metadata_path = artifacts_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    model_name = (args.model_name or "").strip() or settings.prod_model_name
    tags = {
        "project": "fraud-detection",
        "model_source": "custom",
        "model_type": str(metadata.get("model_type") or "xgboost"),
    }

    model = Model(
        path=str(artifacts_dir),
        name=model_name,
        type=AssetTypes.URI_FOLDER,
        description="Custom trained fraud detection model",
        tags=tags,
    )
    registered = ml_client.models.create_or_update(model)

    output_name_path = Path(args.output_model_name)
    output_name_path.parent.mkdir(parents=True, exist_ok=True)
    output_name_path.write_text(str(registered.name))

    output_version_path = Path(args.output_model_version)
    output_version_path.parent.mkdir(parents=True, exist_ok=True)
    output_version_path.write_text(str(registered.version))

    logger.info(
        "Registered custom model",
        extra={
            "model_name": registered.name,
            "model_version": registered.version,
            "model_type": tags.get("model_type"),
        },
    )


def deploy_model_entrypoint(args: argparse.Namespace) -> None:
    settings = get_settings()
    ml_client = get_ml_client(settings=settings)

    model_name = Path(args.model_name_file).read_text().strip()
    model_version = Path(args.model_version_file).read_text().strip()

    endpoint_name = args.endpoint_name or settings.endpoint_name
    deployment_name = args.deployment_name or settings.deployment_name
    instance_type = args.instance_type or settings.get_deployment_instance_type()
    instance_count = args.instance_count or settings.instance_count

    ensure_endpoint(ml_client, endpoint_name=endpoint_name, tags={"project": "fraud-detection"})

    env_id = resolve_scoring_environment(ml_client, settings=settings)
    deployment = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=model_version,
        instance_type=instance_type,
        instance_count=instance_count,
        code_path=str(ROOT_DIR / "src"),
        scoring_script="fraud_detection/serving/custom_scoring.py",
        environment=env_id,
        environment_variables={"ALERT_CAP": str(settings.default_alert_cap)},
        tags={
            "project": "fraud-detection",
            "model_source": "custom",
        },
    )

    logger.info(
        "Deployment completed",
        extra={
            "endpoint_name": deployment.endpoint_name,
            "deployment_name": deployment.name,
            "model_name": model_name,
            "model_version": model_version,
        },
    )

    traffic = set_initial_traffic(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
    )
    logger.info(
        "Set initial traffic for deployment",
        extra={"endpoint_name": endpoint_name, "traffic": traffic},
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fraud detection pipeline steps")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register-model", help="Register a trained model")
    register_parser.add_argument("--model-dir", required=True)
    register_parser.add_argument("--model-name", default="")
    register_parser.add_argument("--output-model-name", required=True)
    register_parser.add_argument("--output-model-version", required=True)
    register_parser.set_defaults(handler=register_model_entrypoint)

    deploy_parser = subparsers.add_parser("deploy-model", help="Deploy a registered model")
    deploy_parser.add_argument("--model-name-file", required=True)
    deploy_parser.add_argument("--model-version-file", required=True)
    deploy_parser.add_argument("--endpoint-name", default="")
    deploy_parser.add_argument("--deployment-name", default="")
    deploy_parser.add_argument("--instance-type", default="")
    deploy_parser.add_argument("--instance-count", type=int, default=0)
    deploy_parser.set_defaults(handler=deploy_model_entrypoint)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("No command handler registered")
    handler(args)


if __name__ == "__main__":
    main()
