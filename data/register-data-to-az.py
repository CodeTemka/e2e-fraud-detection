from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from dotenv import load_dotenv
import os

load_dotenv()

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

data_name = "creditcard-data"
version = "1"

data_asset = Data(
    name=data_name,
    path="./creditcard.csv", 
    description="Credit Card Fraud Detection Dataset",
    type=AssetTypes.URI_FILE,
    version=version,
)

try:
    registered_data = ml_client.data.get(name=data_name, version=version)
    print(f"Data asset '{data_name}' version '{version}' already exists. Skipping registration.")
except Exception as e:
    registered_data = ml_client.data.create_or_update(data_asset)
    print(f"Registered data asset '{data_name}' version '{version}'.")
