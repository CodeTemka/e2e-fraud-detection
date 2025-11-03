from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Workspace
import json
from dotenv import load_dotenv
import os
load_dotenv()

# Load SP credentials
with open("sp-credentials.json") as f:
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