"""Utility to prepare Azure resources for the fraud detection demo."""

import json
import subprocess
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import ClientSecretCredential

from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    settings = get_settings()

    result = subprocess.run(
        ["where", "az"],
        capture_output=True,
        text=True,
        check=True
    )
    az_path = next(
        (p.strip() for p in result.stdout.splitlines() if p.strip().lower().endswith(".cmd")),
        None
    )
    if az_path is None:
        raise RuntimeError("Azure CLI (az.cmd) not found on this system.")

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
        output_dir = Path("json")
        output_file = output_dir / "sp_credentials.json"

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
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save SP credentials to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        logger.info("Service principal credentials saved to '%s'", output_file)
        return True

    # 4 Create workspace using service principal
    def create_workspace():
        if not service_principal_create():
            raise RuntimeError("Failed to create service principal.")

        with open("sp_credentials.json", encoding="utf-8") as f:
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

        ws = Workspace(
            name=workspace_name,
            location=location,
            display_name="e2e-fraud-detection-ws",
            description="Workspace for e2e fraud detection demo",
            tags=dict(purpose="demo"),
        )

        try:
            ml_client.workspaces.get(name=workspace_name)
            logger.info("Workspace %s already exists.", workspace_name)
        except Exception:
            logger.info("Creating workspace %s...", workspace_name)
            poller = ml_client.workspaces.begin_create(ws)
            poller.result()

    create_workspace()


if __name__ == "__main__":
    main()
