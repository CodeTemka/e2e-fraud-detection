#1 Sign up to Azure portal (https://portal.azure.com/) if you don't have an account.
import os
import subprocess
import json
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Workspace
from dotenv import load_dotenv

def main():
    load_dotenv()
    #2 Create a resource group
    def create_resource_group():
        resource_group_name = os.getenv("RESOURCE_GROUP")
        location = os.getenv("LOCATION")

        subprocess.run(['az', 'login'], check=True)

        # 1. Check if the resource group exists
        show_cmd = ["az", "group", "show", "--name", resource_group_name]

        result = subprocess.run(
            show_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 2. If show failed (resource group doesn't exist), create it
        if result.returncode != 0:
            create_cmd = [
                "az", "group", "create",
                "--name", resource_group_name,
                "--location", location,
                '--output', 'json'
            ]
            print(f"Creating resource group: {resource_group_name}")
            result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        else:
            print(f"Resource group '{resource_group_name}' already exists.")
            return True
    
    #3 Create a service principal
    def service_principal_create():
        if create_resource_group():
            pass
        else:
            return Exception("Failed to create resource group.")
        
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        sp_name = 'e2e-fraud-detection-sp'
        role = 'Contributor'
        output_file = 'sp_credentials.json'

        def sp_exists(sp_name):
            cmd = [
                'az', "ad", "sp", "list",
                "--display-name", sp_name,
                "--query", "[].appId",
                "-o", "tsv"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            app_id = result.stdout.strip()
            return bool(app_id)
        
        if sp_exists(sp_name):
            print(f"Service principal '{sp_name}' already exists. Skipping creation.")
            return True
        else:
            print(f"Creating service principal '{sp_name}'...")
            cmd = [
                'az', "ad", "sp", "create-for-rbac",
                "--name", sp_name,
                "--role", role,
                "--scopes", f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}",
                "--sdk-auth"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Save SP credentials to JSON
            with open(output_file, "w") as f:
                f.write(result.stdout)

            print(f"Service principal credentials saved to '{output_file}'")
            return True

    #4 Create workspace using service principal
    def create_workspace():
        if service_principal_create():
            pass
        else:
            return Exception("Failed to create service principal.")
        
        with open("sp_credentials.json") as f:
            sp_data = json.load(f)
        
        credential = ClientSecretCredential(
            tenant_id=sp_data["tenantId"],
            client_id=sp_data["clientId"],
            client_secret=sp_data["clientSecret"]
        )
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace_name = os.getenv("WORKSPACE_NAME")
        location = os.getenv("LOCATION")

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group
        )

        ws = Workspace(
            name=workspace_name,
            location=location,
            display_name="e2e-fraud-detection-ws",
            description="Workspace for e2e fraud detection demo",
            tags=dict(purpose="demo"),
        )
        # Use MLClient to connect to the subscription and resource group and create the workspace.

        try:
            ws = ml_client.workspaces.get(name=workspace_name)
            print(f"Workspace {workspace_name} already exists.")
        except Exception as e:
            print(f"Creating workspace {workspace_name}...")
            ml_client.workspaces.begin_create(ws)


if __name__ == "__main__":
    main()