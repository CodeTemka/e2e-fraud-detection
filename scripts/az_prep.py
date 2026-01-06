"""Utility to prepare Azure resources for the fraud detection demo."""

import argparse
import json
import shutil
import subprocess

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import ClientSecretCredential

from fraud_detection.config import get_settings  # noqa: E402
from fraud_detection.training.compute import delete_compute, ensure_training_compute  # noqa: E402
from fraud_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleanup-endpoints",
        action="store_true",
        help="Delete online endpoints tagged for this project before provisioning.",
    )
    parser.add_argument(
        "--cleanup-compute",
        action="store_true",
        help="Delete compute clusters tagged for this project before provisioning.",
    )
    parser.add_argument(
        "--cleanup-all",
        action="store_true",
        help="Delete tagged online endpoints and compute clusters before provisioning.",
    )
    args = parser.parse_args()

    settings = get_settings()

    az_path = shutil.which("az")
    if az_path is None:
        raise RuntimeError("Azure CLI (az) not found on this system.")

    logger.info("Using AZ: %s", az_path)


    # 2 Create a resource group
    def create_resource_group():
        resource_group_name = settings.resource_group
        location = settings.location

        subprocess.run([az_path, "login"], check=True)

        # 1. Check if the resource group exists
        show_cmd = [az_path, "group", "show", "--name", resource_group_name]

        result = subprocess.run(
            show_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 2. If show failed (resource group doesn't exist), create it
        if result.returncode != 0:
            create_cmd = [
                az_path, "group", "create",
                "--name", resource_group_name,
                "--location", location,
                "--output", "json",
            ]
            logger.info("Creating resource group: %s", resource_group_name)
            result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(result.stdout)
        else:
            logger.info("Resource group '%s' already exists.", resource_group_name)
        return True

    # 3 Create a service principal
    def service_principal_create():
        if not create_resource_group():
            raise RuntimeError("Failed to create resource group.")

        subscription_id = settings.subscription_id
        resource_group = settings.resource_group
        sp_name = "e2e-fraud-detection-sp"
        role = "Contributor"
        output_file = "sp_credentials.json"

        def sp_exists(sp_name):
            cmd = [
                az_path, "ad", "sp", "list",
                "--display-name", sp_name,
                "--query", "[].appId",
                "-o", "tsv",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            app_id = result.stdout.strip()
            return bool(app_id)

        if sp_exists(sp_name):
            logger.info("Service principal '%s' already exists. Skipping creation.", sp_name)
            return True

        logger.info("Creating service principal '%s'...", sp_name)
        cmd = [
            az_path, "ad", "sp", "create-for-rbac",
            "--name", sp_name,
            "--role", role,
            "--scopes", f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}",
            "--sdk-auth",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Save SP credentials to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        logger.info("Service principal credentials saved to '%s'", output_file)
        return True

    # 4 Create workspace using service principal
    def create_workspace() -> MLClient:
        if not service_principal_create():
            raise RuntimeError("Failed to create service principal.")

        sp_credentials_path = "sp_credentials.json"
        with open(sp_credentials_path, encoding="utf-8") as f:
            sp_data = json.load(f)

        credential = ClientSecretCredential(
            tenant_id=sp_data["tenantId"],
            client_id=sp_data["clientId"],
            client_secret=sp_data["clientSecret"],
        )
        subscription_id = settings.subscription_id
        resource_group = settings.resource_group
        workspace_name = settings.workspace_name
        location = settings.location

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
        )

        resource_tags = settings.get_resource_tags()
        ws = Workspace(
            name=workspace_name,
            location=location,
            display_name="e2e-fraud-detection-ws",
            description="Workspace for e2e fraud detection demo",
            tags={**resource_tags, "purpose": "demo"},
        )

        try:
            ml_client.workspaces.get(name=workspace_name)
            logger.info("Workspace %s already exists.", workspace_name)
        except Exception:
            logger.info("Creating workspace %s...", workspace_name)
            ml_client.workspaces.begin_create(ws).result()

        return ml_client

    def _tags_match(actual: dict[str, str] | None, desired: dict[str, str]) -> bool:
        if not desired:
            return False
        if not actual:
            return False
        return all(actual.get(key) == value for key, value in desired.items())

    def cleanup_endpoints(ml_client: MLClient) -> None:
        desired = settings.get_resource_tags()
        for endpoint in ml_client.online_endpoints.list():
            if _tags_match(endpoint.tags, desired):
                logger.info("Deleting tagged endpoint: %s", endpoint.name)
                ml_client.online_endpoints.begin_delete(name=endpoint.name).result()

    def cleanup_compute(ml_client: MLClient) -> None:
        desired = settings.get_resource_tags()
        for compute in ml_client.compute.list():
            if _tags_match(compute.tags, desired):
                delete_compute(ml_client, name=compute.name, ignore_missing=True)

    ml_client = create_workspace()
    if args.cleanup_all or args.cleanup_endpoints:
        cleanup_endpoints(ml_client)
    if args.cleanup_all or args.cleanup_compute:
        cleanup_compute(ml_client)

    ensure_training_compute(
        ml_client,
        name=settings.get_training_compute(),
        size=settings.instance_type,
        min_instances=settings.compute_min_nodes,
        max_instances=settings.compute_max_nodes,
        idle_time_before_scale_down=settings.compute_idle_time_before_scale_down,
        tags=settings.get_resource_tags(),
    )


if __name__ == "__main__":
    main()
